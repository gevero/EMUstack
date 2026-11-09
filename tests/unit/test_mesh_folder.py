"""Where NanoStruct writes its gmsh meshes (meshing.mesh_folder)."""
import os
import sys

import pytest

from EMUstack.backend import materials, meshing, objects

needs_gmsh = pytest.mark.skipif(not meshing.gmsh_available(), reason="gmsh not installed or cannot start")


@pytest.fixture
def rules(tmp_path, monkeypatch):
    """No mesh_dir override, working directory tmp_path/run, user cache inside tmp_path."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(run)
    monkeypatch.setattr(meshing, "mesh_dir", None)
    monkeypatch.setattr(meshing, "user_cache_dir", lambda: str(tmp_path / "cache"))
    return run


def _ellipse(geo_file, **kw):
    return objects.NanoStruct('2D_array', 600.0, height_nm=100.0, geo_file=geo_file,
                              background=materials.Air, inclusion_a=materials.Material(2.0),
                              geo_params=dict(diameter1=200, lc=0.1), **kw)


def _circle(**kw):
    return objects.NanoStruct('2D_array', 600.0, 200.0, height_nm=100.0, background=materials.Air,
                              inclusion_a=materials.Material(2.0), lc_bkg=0.1, **kw)


def _local_copy(folder, name="my_ellipse.geo"):
    folder.mkdir(parents=True, exist_ok=True)
    text = open(meshing.resolve_mesh_file_path("ellipse_msh_template.geo"), encoding="utf-8").read()
    (folder / name).write_text(text, encoding="utf-8")
    return folder / name


def _files(folder):
    return sorted(os.listdir(folder))


@needs_gmsh
def test_builtin_shape_meshes_in_the_working_directory(rules):
    ns = _circle(mesher='gmsh')
    folder = rules / "emustack_meshes"
    assert ns.mesh_dir == str(folder)
    stem = ns.mesh_file[:-5]
    assert _files(folder) == [stem + ".geo", stem + ".mail", stem + ".msh"]


@needs_gmsh
def test_shipped_template_by_name_meshes_in_the_working_directory(rules):
    ns = _ellipse("ellipse.geo")
    assert ns.mesh_dir == str(rules / "emustack_meshes")
    assert ns.mesh_file.startswith("ellipse600_0_600_0_")  # no stray "_msh"
    assert (rules / "emustack_meshes" / ns.mesh_file).exists()


@needs_gmsh
def test_local_geo_meshes_next_to_it(rules):
    _local_copy(rules)
    ns = _ellipse("my_ellipse.geo")
    assert ns.mesh_dir == str(rules / "emustack_meshes")
    assert (rules / "emustack_meshes" / ns.mesh_file).exists()


@needs_gmsh
@pytest.mark.parametrize("absolute", [False, True])
def test_geo_in_another_folder_meshes_next_to_it(rules, absolute):
    geo = _local_copy(rules / "sub" / "deeper")
    ns = _ellipse(str(geo) if absolute else os.path.join("sub", "deeper", "my_ellipse.geo"))
    folder = rules / "sub" / "deeper" / "emustack_meshes"
    assert ns.mesh_dir == str(folder)
    assert ns.mesh_file.startswith("my_ellipse600_0_")
    assert (folder / ns.mesh_file).exists()
    assert not (rules / "emustack_meshes").exists()


@needs_gmsh
def test_mesh_dir_overrides_every_rule(rules, tmp_path, monkeypatch):
    target = tmp_path / "all_meshes"
    monkeypatch.setattr(meshing, "mesh_dir", str(target))
    _local_copy(rules / "sub")
    a = _ellipse(os.path.join("sub", "my_ellipse.geo"))
    b = _circle(mesher='gmsh')
    assert a.mesh_dir == b.mesh_dir == str(target)
    assert {a.mesh_file, b.mesh_file} <= set(_files(target))
    assert not (rules / "emustack_meshes").exists() and not (rules / "sub" / "emustack_meshes").exists()


@needs_gmsh
@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_unwritable_folder_falls_back_to_the_user_cache_with_a_warning(rules, tmp_path):
    geo = _local_copy(rules / "readonly")
    os.chmod(rules / "readonly", 0o500)
    try:
        with pytest.warns(UserWarning, match="cannot write meshes"):
            ns = _ellipse(str(geo))
    finally:
        os.chmod(rules / "readonly", 0o700)
    assert ns.mesh_dir == str(tmp_path / "cache")
    assert (tmp_path / "cache" / ns.mesh_file).exists()


@needs_gmsh
def test_edited_geo_is_meshed_again(rules):
    geo = _local_copy(rules)
    first = _ellipse("my_ellipse.geo", force_mesh=False)
    mail = rules / "emustack_meshes" / first.mesh_file
    stamp = mail.stat().st_mtime_ns
    # unchanged template: with force_mesh=False the cached mesh is reused
    assert _ellipse("my_ellipse.geo", force_mesh=False).n_msh_el == first.n_msh_el
    assert mail.stat().st_mtime_ns == stamp
    # edited template, same name and parameters: meshed again
    geo.write_text(geo.read_text(encoding="utf-8") + "\nMesh.CharacteristicLengthFactor = 0.5;\n",
                   encoding="utf-8")
    assert _ellipse("my_ellipse.geo", force_mesh=False).n_msh_el > first.n_msh_el


def test_netgen_writes_no_mesh_folder(rules):
    ns = _circle(mesher='netgen')
    assert ns.mesh_dir is None
    assert not (rules / "emustack_meshes").exists()


def test_shipped_templates_are_recognized():
    assert meshing.is_shipped_template(meshing.resolve_mesh_file_path("ellipse.geo"))
    assert not meshing.is_shipped_template(os.path.abspath("my_ellipse.geo"))
