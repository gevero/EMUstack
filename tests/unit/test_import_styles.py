"""EMUstack behaves the same however it is imported.

Two styles are supported: the package (``from EMUstack.backend import objects``, used by
the docs and notebooks) and legacy scripts that put ``backend/`` on ``sys.path`` and use bare
module names (``import objects``). In both, each module exists once, so the documented
module-level switches reach the code that runs.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from EMUstack.backend import meshing, objects, materials, plotting
from EMUstack.backend.stack import Stack

PROBE = Path(__file__).with_name("_switch_probe.py")
BACKEND = Path(objects.__file__).resolve().parent
needs_gmsh = pytest.mark.skipif(not meshing.gmsh_available(), reason="gmsh not installed or cannot start")


def _probe(style, cwd):
    res = subprocess.run([sys.executable, str(PROBE), style, str(BACKEND)], cwd=cwd,
                         capture_output=True, text=True, timeout=900)
    lines = [l for l in res.stdout.splitlines() if l.startswith("PROBE ")]
    assert lines, res.stdout[-2000:] + res.stderr[-2000:]
    return json.loads(lines[-1][len("PROBE "):])


@needs_gmsh
def test_documented_switches_work_in_both_import_styles(tmp_path):
    package, legacy = _probe("package", tmp_path), _probe("legacy", tmp_path)
    assert package["T"] == pytest.approx(legacy["T"], rel=1e-12)  # same result in both styles
    for r in (package, legacy):
        assert r["eigensolver"].startswith("ValueError: eigensolver must be")
        assert r["overlap_quadrature"].startswith("ValueError: quadrature must be")
        assert r["ez_convention"].startswith("ValueError: field_eval.ez_convention must be")
        assert (r["mesh"], r["default_mesher"]) == ("y<=0", "y>=0")
        assert r["gmsh_timeout"].startswith("RuntimeError: gmsh did not finish")


def test_importing_emustack_loads_each_module_once(tmp_path):
    code = (
        "import sys\n"
        "before = list(sys.path)\n"
        "import EMUstack\n"
        "from EMUstack.backend import objects, plotting, fem_2d, field_eval, meshing\n"
        "bare = [m for m in ('objects', 'materials', 'plotting', 'stack', 'mode_calcs', 'paths',\n"
        "        'meshing', 'modes', 'fem_1d', 'fem_2d', 'field_eval', 'linalg', 'legacy_sort',\n"
        "        'mode_matching') if m in sys.modules]\n"
        "print(bare, sys.path == before)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, timeout=300).stdout.strip().splitlines()[-1]
    assert out == "[] True"


def test_user_modules_are_not_shadowed(tmp_path):
    (tmp_path / "plotting.py").write_text("WHO = 'user'\n")
    code = "import EMUstack, plotting; print(plotting.WHO)"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, timeout=300).stdout.strip().splitlines()[-1]
    assert out == "user"


@needs_gmsh
def test_field_maps_use_the_same_cell_frame_for_both_meshers():
    """gmsh meshes span y in [-1, 0], netgen meshes y in [0, 1]; the interpolators do not care."""
    light = objects.Light(700.0, max_order_PWs=2, theta=0.0, phi=0.0)
    film = objects.ThinFilm(400.0, height_nm="semi_inf", material=materials.Air)
    u = np.linspace(0.02, 0.98, 40)
    U, V = np.meshgrid(u, u)
    maps = []
    for geo_file in ("ellipse_msh_template.geo", "ellipse_msh_template.py"):
        lay = objects.NanoStruct(
            "2D_array", geo_file=geo_file, height_nm=100.0, plotting_fields=1,
            background=materials.Air, inclusion_a=materials.Material(3.0),
            geo_params={"period": 400.0, "diameter1": 250.0, "ellipticity": 0.5,
                        "angle": 0.6, "lc": 0.1, "lc2": 2.0})
        s = Stack([film.calc_modes(light), lay.calc_modes(light, num_BMs=80), film.calc_modes(light)])
        s.calc_scat(pol="TE")
        b = plotting.fields_interpolator_in_plane(s, lay_interest=1, z_value=50.0, return_H=True)
        a = np.ma.filled(np.ma.masked_invalid(b.AbsE(U, V)), np.nan)
        assert not np.isnan(a).any()  # every point of the grid is inside the cell
        maps.append(a)
    gmsh_map, netgen_map = maps
    # same rotated ellipse, different meshes: the maps agree, and not after a mirror flip
    assert np.corrcoef(gmsh_map.ravel(), netgen_map.ravel())[0, 1] > 0.9
    assert np.corrcoef(gmsh_map[::-1].ravel(), netgen_map.ravel())[0, 1] < 0.8
