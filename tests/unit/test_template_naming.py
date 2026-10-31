"""One naming vocabulary for .geo templates, .py templates and meshing.unit_cell.

Every old file name, parameter name and inclusion key must keep working and
give exactly the same mesh as the new names.
"""
import os
import warnings

import numpy as np
import ngsolve as ngs
import pytest

from EMUstack.backend import meshing, objects, paths


def _have_gmsh():
    try:
        meshing._gmsh_binary()
        return True
    except EnvironmentError:
        return False


needs_gmsh = pytest.mark.skipif(not _have_gmsh(), reason="gmsh not installed")


def _same_mesh(m1, m2):
    assert (m1.nv, m1.ne) == (m2.nv, m2.ne)
    p1 = np.array([v.point for v in m1.vertices])
    p2 = np.array([v.point for v in m2.vertices])
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-14)
    assert [e.mat for e in m1.Elements(ngs.VOL)] == [e.mat for e in m2.Elements(ngs.VOL)]


def _template_text(name):
    with open(meshing.resolve_mesh_file_path(name)) as f:
        return f.read()


# --- file names -------------------------------------------------------------

@pytest.mark.parametrize("old,new", sorted(meshing.LEGACY_TEMPLATE_NAMES.items()))
def test_legacy_template_names_resolve(old, new):
    assert os.path.basename(meshing.resolve_mesh_file_path(old)) == new


@pytest.mark.parametrize("short,full", [
    ("ring.geo", "ring_msh_template.geo"),
    ("ellipse.py", "ellipse_msh_template.py"),
    ("ellipse_msh.geo", "ellipse_msh_template.geo"),
    ("ring1.geo", "ring_msh_template.geo"),
])
def test_short_template_names_resolve(short, full):
    assert os.path.basename(meshing.resolve_mesh_file_path(short)) == full


def test_every_template_follows_the_naming_rule():
    for name in os.listdir(paths.template_path):
        if name.endswith((".geo", ".py")) and not name.startswith(("4testing", "gmsh_import_cell")):
            assert name.endswith(("_msh_template.geo", "_msh_template.py")), name


def test_user_file_in_cwd_wins(tmp_path, monkeypatch):
    (tmp_path / "ring1_msh_template.geo").write_text("// mine\n")
    monkeypatch.chdir(tmp_path)
    assert meshing.resolve_mesh_file_path("ring1_msh_template.geo") == str(tmp_path / "ring1_msh_template.geo")


# --- .geo parameters ----------------------------------------------------------

@pytest.mark.parametrize("template,legacy,canonical", [
    ("ring1_msh_template.geo",
     {"d_in_nm": 600, "dy_in_nm": 500, "a1": 300, "a2": 150, "xshift_nm": 20},
     {"period": 600, "period_y": 500, "diameter1": 300, "diameter2": 150, "xshift": 20}),
    ("rect1_msh_template.geo", {"a1": 250, "a2": 120}, {"width": 250, "height": 120}),
    ("rect_shell1_msh_template.geo", {"a1": 250, "b1": 120, "t": 20}, {"width": 250, "height": 120, "thickness": 20}),
    ("square_shell_dimer1_msh_template.geo",
     {"a1": 150, "a2": 120, "b1": 140, "b2": 110},
     {"width1": 150, "width2": 120, "height1": 140, "height2": 110}),
    ("SRR_msh_template.geo",
     {"d_in_nm": 600, "lvert_nm": 300, "lhori_nm": 250, "width_nm": 40},
     {"period": 600, "len_vertical": 300, "len_horizontal": 250, "width": 40}),
    ("L_msh_template.geo", {"L_nm": 300, "W_nm": 100, "r": 0.3}, {"length": 300, "width": 100, "rounding": 0.3}),
    ("16_msh_template.geo", {f"d{k}": 60 + k for k in range(1, 17)}, {f"diameter{k}": 60 + k for k in range(1, 17)}),
    ("16_msh_template.geo", {f"a{k}": 60 + k for k in range(1, 17)}, {f"diameter{k}": 60 + k for k in range(1, 17)}),
])
def test_legacy_and_canonical_keys_fill_the_same(template, legacy, canonical):
    text = _template_text(template)
    old = meshing.fill_geo_template(text, legacy, template=template)
    new = meshing.fill_geo_template(text, canonical, template=template)
    assert old == new
    assert old != text


LEGACY_STYLE_GEO = """d_in_nm = 0;
dy_in_nm = 0;
a1 = 0;
thick = 0;
lc = 0;
lc2 = lc/1; // on the inclusion
"""


def test_old_custom_template_takes_new_keys():
    old = meshing.fill_geo_template(LEGACY_STYLE_GEO, {"d_in_nm": 600, "dy_in_nm": 500, "a1": 200, "lc": 0.1, "lc2": 2})
    new = meshing.fill_geo_template(LEGACY_STYLE_GEO, {"period": 600, "period_y": 500, "diameter1": 200, "lc": 0.1, "lc2": 2})
    assert old == new
    assert "d_in_nm = 600.000000;" in new
    assert "lc2 = lc/2.000000; // on the inclusion" in new


def test_exact_name_wins_over_alias():
    text = "period = 0; // something else\nd_in_nm = 0;\n"
    out = meshing.fill_geo_template(text, {"d_in_nm": 600})
    assert out == "period = 0; // something else\nd_in_nm = 600.000000;\n"


def test_fill_tolerates_spacing_and_keeps_it():
    out = meshing.fill_geo_template("d_in_nm  = 0;\nx=0;\n", {"period": 600, "x": 1})
    assert out == "d_in_nm  = 600.000000;\nx=1.000000;\n"


def test_fill_unknown_key():
    with pytest.raises(KeyError, match="Parameter 'raduis' in geo_params was not found"):
        meshing.fill_geo_template(LEGACY_STYLE_GEO, {"raduis": 1.0})
    assert meshing.fill_geo_template(LEGACY_STYLE_GEO, {"raduis": 1.0}, strict=False) == LEGACY_STYLE_GEO
    assert meshing.fill_geo_template(LEGACY_STYLE_GEO, {"raduis": 1.0}, optional=["raduis"]) == LEGACY_STYLE_GEO


def test_geo_placeholders_report_legacy_names():
    placeholders, legacy = meshing.geo_placeholders(_template_text("ring_msh_template.geo"))
    assert placeholders["diameter1"] == "value"
    assert placeholders["lc2"] == "lc"
    assert legacy["a1"] == "diameter1"
    assert legacy["xshift_nm"] == "xshift"


@needs_gmsh
def test_mesh_from_geo_legacy_and_canonical_names():
    old = meshing.mesh_from_geo("ellipse_msh_template.geo",
                                geo_params={"d_in_nm": 450, "a1": 280, "ellipticity": 0.3}, lc=0.12)
    new = meshing.mesh_from_geo("ellipse.geo",
                                geo_params={"period": 450, "diameter1": 280, "ellipticity": 0.3}, lc=0.12)
    _same_mesh(old, new)
    assert {"top", "bottom", "left", "right"} <= set(new.GetBoundaries())


@needs_gmsh
def test_nanostruct_old_template_name_and_keys():
    old = objects.NanoStruct("2D_array", 600, geo_file="rect1_msh_template.geo", mesher="netgen",
                             geo_params={"a1": 250, "a2": 120, "smooth": 0.2})
    new = objects.NanoStruct("2D_array", 600, geo_file="rectangle_msh_template.geo", mesher="netgen",
                             geo_params={"width": 250, "height": 120, "smooth": 0.2})
    _same_mesh(old.mesh, new.mesh)


@needs_gmsh
def test_square_shell_dimer_thickness_is_filled():
    # The template's placeholder is "thickness", but legacy code filled "t",
    # so the thickness stayed 0 and gmsh hung.
    s = objects.NanoStruct("2D_array", 600, inc_shape="square_shell_dimer", mesher="gmsh", force_mesh=True,
                           diameter1=150, diameter2=120, diameter3=140, diameter4=110, gap=40, smooth=0.2, t=15)
    assert s.n_msh_el > 0


@needs_gmsh
@pytest.mark.parametrize("kw", [
    dict(inc_shape="ring", diameter1=300, diameter2=150),           # boundary tags 53..56
    dict(inc_shape="SRR", diameter1=40, len_vertical=300, len_horizontal=250),
    dict(diameter1=200, is_hex=True, period_y=1039.23),             # tags 61..64
    dict(diameter1=150, diameter2=100, diameter3=80),               # tags 95..98
    dict(diameter1=150, diameter2=100),                             # tags 101..104; mesh name typo
])
def test_templates_with_other_boundary_tags_import(kw):
    s = objects.NanoStruct("2D_array", 600, mesher="gmsh", force_mesh=True, **kw)
    assert set(s.mesh.GetBoundaries()) == {"top", "bottom", "left", "right"}
    assert s.n_msh_el > 0


@needs_gmsh
@pytest.mark.parametrize("template,params", [
    ("SRR_msh_template.geo", {"len_vertical": 300, "len_horizontal": 250, "width": 40}),
    ("ring_msh_template.geo", {"diameter1": 300, "diameter2": 150}),
])
def test_mesh_from_geo_with_many_surfaces(template, params):
    # many elementary surfaces used to make netgen's ReadGmsh read out of bounds
    for _ in range(3):
        mesh = meshing.mesh_from_geo(template, geo_params=params, period=600, lc=0.1)
        assert set(mesh.GetBoundaries()) == {"top", "bottom", "left", "right"}
        assert "default" not in mesh.GetMaterials()


def test_unnamed_physical_groups_get_names():
    msh = "\n".join([
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$Nodes", "4", "1 0 0 0", "2 1 0 0", "3 1 1 0", "4 0 1 0", "$EndNodes",
        "$Elements", "6",
        "1 1 2 53 7 1 2", "2 1 2 54 8 2 3", "3 1 2 55 9 3 4", "4 1 2 56 10 4 1",
        "5 2 2 1 11 1 2 3", "6 2 2 3 12 1 3 4",
        "$EndElements", ""])
    out = meshing._prepare_msh_for_netgen(msh)
    for line in ['1 53 "bottom"', '1 54 "right"', '1 55 "top"', '1 56 "left"', '2 1 "bg"', '2 3 "inc3"',
                 "1 1 2 53 53 1 2", "6 2 2 3 3 1 3 4"]:
        assert line in out
    assert meshing._prepare_msh_for_netgen(out) is None


# --- .py templates ------------------------------------------------------------

def test_py_template_old_file_name():
    old = meshing.mesh_from_py("chiral_qbic_template.py", geo_params={"alpha": 0.1}, period=700, lc=0.15, lc2=2.0)
    new = meshing.mesh_from_py("chiral_qbic_msh_template.py", geo_params={"alpha": 0.1}, period=700, lc=0.15, lc2=2.0)
    _same_mesh(old, new)


def test_py_template_old_keywords():
    kw = dict(period=480, lc=0.15, lc2=2.0)
    old = meshing.mesh_from_py("chiral_qbic_gorkunov.py", geo_params={"theta": 10, "l": 220, "w": 70, "d": 10}, **kw)
    new = meshing.mesh_from_py("chiral_qbic_gorkunov_msh_template.py",
                               geo_params={"theta_deg": 10, "length": 220, "width": 70, "z_offset": 10}, **kw)
    _same_mesh(old, new)


def test_py_builder_called_directly():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ellipse_tpl", meshing.resolve_mesh_file_path("ellipse.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _same_mesh(mod.build_unit_cell(period=400, a1=250, lc=0.15),
               mod.build_geometry(period=400, diameter1=250, lc=0.15))
    with pytest.raises(TypeError, match="got both 'a1' and its new name 'diameter1'"):
        mod.build_unit_cell(a1=250, diameter1=250)


def test_py_builder_with_old_parameter_names(tmp_path):
    # a user's template written for the old names still gets NanoStruct's values
    (tmp_path / "old_style.py").write_text(
        "from EMUstack.backend import meshing\n"
        "def build_unit_cell(d_in_nm, a1, lc=0.1):\n"
        "    r = a1 / (2.0 * d_in_nm)\n"
        "    return meshing.unit_cell([{'kind': 'circle', 'r': r}], lc=lc)\n"
    )
    mesh = meshing.mesh_from_py(str(tmp_path / "old_style.py"), geo_params={"diameter1": 200}, period=600, lc=0.15)
    ref = meshing.unit_cell([{"kind": "circle", "radius": 200 / 1200}], lc=0.15)
    _same_mesh(mesh, ref)


# --- unit_cell ----------------------------------------------------------------

def test_unit_cell_short_and_long_keys():
    short = [{"shape": "circle", "c": (0.3, 0.5), "r": 0.12, "tag": 2},
             {"kind": "ellipse", "c": (0.7, 0.3), "rx": 0.1, "ry": 0.05, "rot": 30.0, "tag": 3},
             {"kind": "rectangle", "c": (0.7, 0.75), "w": 0.2, "h": 0.1, "tag": 4},
             {"kind": "ring", "c": (0.25, 0.15), "r_out": 0.1, "r_in": 0.05, "tag": 5},
             {"kind": "dimer", "sub_kind": "square", "c": (0.5, 0.85), "r": 0.05, "s": 0.06, "gap": 0.02}]
    long = [{"kind": "circle", "center": (0.3, 0.5), "radius": 0.12, "tag": 2},
            {"kind": "ellipse", "center": (0.7, 0.3), "rx": 0.1, "ry": 0.05, "angle_deg": 30.0, "tag": 3},
            {"kind": "rectangle", "center": (0.7, 0.75), "width": 0.2, "height": 0.1, "tag": 4},
            {"kind": "ring", "center": (0.25, 0.15), "radius_out": 0.1, "radius_in": 0.05, "tag": 5},
            {"kind": "dimer", "sub_kind": "square", "center": (0.5, 0.85), "radius": 0.05, "side": 0.06, "gap": 0.02}]
    _same_mesh(meshing.unit_cell(short, lc=0.1), meshing.unit_cell(long, lc=0.1))


def test_unit_cell_angle_in_degrees_and_radians():
    deg = meshing.unit_cell([{"kind": "ellipse", "rx": 0.2, "ry": 0.1, "angle_deg": 30.0}], lc=0.1)
    rad = meshing.unit_cell([{"kind": "ellipse", "rx": 0.2, "ry": 0.1, "angle": np.pi / 6}], lc=0.1)
    _same_mesh(deg, rad)


def test_unit_cell_unknown_key_warns():
    with pytest.warns(UserWarning, match=r"unknown inclusion key\(s\) \['radious'\]"):
        meshing.unit_cell([{"kind": "circle", "radius": 0.2, "radious": 0.3}], lc=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        meshing.unit_cell([{"kind": "circle", "r": 0.2}], lc=0.1)


def test_unit_cell_nseg_is_a_parameter():
    coarse = meshing.unit_cell([{"kind": "circle", "radius": 0.3}], lc=0.2, nseg=12)
    fine = meshing.unit_cell([{"kind": "circle", "radius": 0.3}], lc=0.2, nseg=48)
    assert fine.nv > coarse.nv
