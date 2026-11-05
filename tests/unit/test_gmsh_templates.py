"""gmsh templates that used to fail: 16-inclusion supercell, strip shapes, gmsh timeout."""
import collections
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from EMUstack.backend import materials, meshing, objects

HAVE_GMSH = meshing.gmsh_available()
needs_gmsh = pytest.mark.skipif(not HAVE_GMSH, reason="gmsh not installed or cannot start")


def _mesh(tmp_path, template, params, name="cell"):
    """Fill a shipped template, mesh it with conv_gmsh, return nodes, boundary tags, triangles."""
    text = open(meshing.resolve_mesh_file_path(template)).read()
    root = str(tmp_path / name)
    with open(root + ".geo", "w") as f:
        f.write(meshing.fill_geo_template(text, params, fmt="%f", strict=False))
    meshing.conv_gmsh(root)
    npt, nel, pts, idfn, table_nod, type_el = meshing.parse_mail_data(root + ".mail")
    return pts.T, idfn, table_nod[:3].T - 1, np.asarray(type_el)


def _check_cell(xy, idfn, tris):
    """Conforming triangulation covering the cell once, with matching opposite boundaries."""
    edges = collections.Counter()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[tuple(sorted((a, b)))] += 1
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)

    def on_boundary(v):
        x, y = xy[v]
        return min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)) < 1e-9

    assert all(c == 2 or (c == 1 and on_boundary(a) and on_boundary(b)) for (a, b), c in edges.items())
    p = xy[tris]
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    assert area == pytest.approx((x1 - x0) * (y1 - y0), rel=1e-12)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        a = np.sort(xy[np.abs(xy[:, axis] - lo) < 1e-9, 1 - axis])
        b = np.sort(xy[np.abs(xy[:, axis] - hi) < 1e-9, 1 - axis])
        np.testing.assert_allclose(a, b, atol=1e-9)
    bnd = [v for v in range(len(xy)) if on_boundary(v)]
    assert np.all(idfn[bnd] != 0), "boundary nodes need a boundary tag"


@needs_gmsh
@pytest.mark.parametrize("extra", [{}, {"ellipticity": 0.3}, {"square": 1},
                                   {"posx": 0.03, "posy": -0.02}, {"period_y": 900}])
def test_16_inclusion_supercell_is_a_conforming_4x4_grid(tmp_path, extra):
    params = dict(period=1200, period_y=1200, lc=0.1, lc2=2, lc3=2)
    params.update({f"diameter{k}": 60 + 3 * k for k in range(1, 17)})
    params.update(extra)
    xy, idfn, tris, tags = _mesh(tmp_path, "16_msh_template.geo", params)
    _check_cell(xy, idfn, tris)
    # 16 inclusions on a 4x4 grid: inclusion 1 (top left) is tag 2, the others tag 3
    cent = xy[tris].mean(axis=1)
    dy = params["period_y"] / params["period"]
    for tag, n in ((2, 1), (3, 15)):
        cells = {(int(x // 0.25), int(-y // (dy / 4))) for x, y in cent[tags == tag]}
        assert len(cells) == n
    assert {(int(x // 0.25), int(-y // (dy / 4))) for x, y in cent[tags == 2]} == {(0, 0)}


@needs_gmsh
@pytest.mark.parametrize("template,extra", [
    ("strip_msh_template.geo", {}), ("strip_msh_template.geo", {"square": 1}),
    ("double_strip_msh_template.geo", {"strip2": 100}),
    ("double_strip_msh_template.geo", {"strip2": 100, "square": 1}),
])
def test_strip_templates_mesh(tmp_path, template, extra):
    params = dict(period=600, period_y=600, diameter1=200, strip=300, lc=0.1, lc2=2, lc3=2, **extra)
    xy, idfn, tris, tags = _mesh(tmp_path, template, params)
    _check_cell(xy, idfn, tris)
    assert set(tags) == ({1, 2, 3, 4} if "strip2" in extra else {1, 2, 3})


@pytest.mark.parametrize("shape,d1,d2,d3", [
    ("strip_circle", 200.0, 60.0, 0.0),        # strip narrower than the circle
    ("strip_square", 200.0, 200.0, 0.0),       # strip as wide as the square
    ("strip_circle", 200.0, 650.0, 0.0),       # strip wider than the cell
    ("double_strip_circle", 200.0, 300.0, 400.0),  # bands wider than the cell
])
def test_strip_shapes_reject_parameters_gmsh_cannot_mesh(shape, d1, d2, d3):
    with pytest.raises(ValueError, match="diameter"):
        objects.NanoStruct('2D_array', 600.0, inc_shape=shape, diameter1=d1, diameter2=d2,
                           diameter3=d3, height_nm=100.0, background=materials.Air,
                           inclusion_a=materials.Material(2.0), mesher='gmsh')


@needs_gmsh
def test_gmsh_is_stopped_after_the_timeout(tmp_path, monkeypatch):
    # a strip narrower than its circle: gmsh never returns on this geometry
    text = open(meshing.resolve_mesh_file_path("strip_msh_template.geo")).read()
    root = str(tmp_path / "hangs")
    with open(root + ".geo", "w") as f:
        f.write(meshing.fill_geo_template(
            text, dict(period=600, period_y=600, diameter1=200, strip=60, lc=0.1, lc2=2, lc3=2),
            fmt="%f", strict=False))
    monkeypatch.setattr(meshing, "gmsh_timeout", 2.0)
    with pytest.raises(RuntimeError, match="did not finish"):
        meshing.conv_gmsh(root)
    if sys.platform != "win32":  # no pgrep on Windows
        left = subprocess.run(["pgrep", "-f", root + ".geo"], capture_output=True, text=True).stdout
        assert left.strip() == ""
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


@needs_gmsh
def test_supercell_mesh_name_depends_on_every_diameter():
    def name(d12):
        dk = {f"diameter{k}": 60.0 + k for k in range(1, 17)}
        dk["diameter12"] = d12
        return objects.NanoStruct('2D_array', 1200.0, height_nm=100.0, background=materials.Air,
                                  inclusion_a=materials.Material(2.0), mesher='gmsh', **dk).mesh_file
    assert name(72.0) != name(80.0)


def test_mismatched_periodic_boundaries_are_rejected(tmp_path):
    # unit square whose left boundary node sits at y = 0.5 and right one at y = 0.6
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0.5 0
6 1 0.6 0
$EndNodes
$Elements
10
1 1 2 2 2 1 2
2 1 2 4 4 2 6
3 1 2 4 4 6 3
4 1 2 1 1 3 4
5 1 2 3 3 4 5
6 1 2 3 3 5 1
7 2 2 1 1 1 2 6
8 2 2 1 1 1 6 5
9 2 2 1 1 5 6 3
10 2 2 1 1 5 3 4
$EndElements
"""
    path = tmp_path / "skew.msh"
    path.write_text(msh)
    with pytest.raises(AssertionError, match="do not match"):
        meshing.import_gmsh_mesh(str(path))


@needs_gmsh
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_gmsh_command_uses_a_parent_death_launcher():
    cmd = meshing._gmsh_command(["-version"])
    assert cmd[:2] == [sys.executable, "-c"]
    assert "prctl" in cmd[2]
    assert cmd[3] == str(os.getpid())
    assert cmd[4] == meshing._gmsh_binary()


@needs_gmsh
def test_gmsh_runs_through_the_launcher():
    res = meshing._run_gmsh(["-version"], "version")
    assert res.returncode == 0
    assert res.stdout.strip()


@needs_gmsh
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_gmsh_dies_when_its_parent_python_is_killed(tmp_path):
    """SIGKILL of the Python process must take a running gmsh with it."""
    text = open(meshing.resolve_mesh_file_path("strip_msh_template.geo")).read()
    root = str(tmp_path / "orphan")
    with open(root + ".geo", "w") as f:
        f.write(meshing.fill_geo_template(
            text, dict(period=600, period_y=600, diameter1=200, strip=60, lc=0.1, lc2=2, lc3=2),
            fmt="%f", strict=False))

    def running():
        out = subprocess.run(["pgrep", "-f", root + ".geo"], capture_output=True, text=True)
        return bool(out.stdout.strip())

    script = (
        "from EMUstack.backend import meshing\n"
        "meshing.gmsh_timeout = 600.0\n"
        "meshing.conv_gmsh(%r)\n" % root
    )
    proc = subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if running():
                break
            time.sleep(0.1)
        assert running(), "gmsh did not start"
        proc.kill()  # SIGKILL the parent Python, as a user or the OOM killer would
        proc.wait(timeout=10)
        for _ in range(100):
            if not running():
                break
            time.sleep(0.1)
        assert not running(), "gmsh survived its parent (PR_SET_PDEATHSIG failed)"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        subprocess.run(["pkill", "-f", root + ".geo"], capture_output=True)


def test_missing_system_library_gives_an_install_hint(monkeypatch):
    """A gmsh that cannot load its shared libraries must say which packages to install."""
    fake = [sys.executable, "-c",
            "import sys; sys.stderr.write('OSError: libGLU.so.1: cannot open shared object "
            "file: No such file or directory\\n'); sys.exit(1)"]
    monkeypatch.setattr(meshing, "_gmsh_command", lambda args: fake)
    with pytest.raises(RuntimeError, match="apt-get install libglu1-mesa"):
        meshing._run_gmsh(["-version"], "cell.geo")
