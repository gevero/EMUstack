"""The code in docs/source/tutorial.rst and the README quickstart runs as written."""
import os
import re

import numpy as np
import pytest

from EMUstack.backend import meshing

TUTORIAL = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source", "tutorial.rst")

HAVE_GMSH = meshing.gmsh_available()


def tutorial_code():
    text = open(TUTORIAL).read()
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?: {4}.*)?\n)+)", text)
    return "\n".join("\n".join(line[4:] for line in b.splitlines()) for b in blocks)


@pytest.mark.skipif(not HAVE_GMSH, reason="gmsh not installed or cannot start")
def test_tutorial_code_runs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)  # keep any output files out of the repo
    code = tutorial_code()
    assert "calc_scat" in code and "fields_interpolator_in_plane" in code
    ns = {}
    exec(compile(code, TUTORIAL, "exec"), ns)
    T, R = float(ns["sim_stack"].t_list[-1]), float(ns["sim_stack"].r_list[-1])
    assert 0.0 < T < 1.0 and 0.0 < R < 1.0
    assert ns["Ex"].shape == (101, 101)
    assert len(ns["stacks"]) == 3


README = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")


def readme_quickstart_code():
    text = open(README).read()
    text = text[text.index("## Quickstart Tutorials"):text.index("## Interactive Tutorial Notebooks")]
    return "\n".join(re.findall(r"```python\n(.*?)```", text, re.S))


@pytest.mark.skipif(not HAVE_GMSH, reason="gmsh not installed or cannot start")
def test_readme_quickstart_runs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    code = readme_quickstart_code()
    assert code.count("NanoStruct(") == 2
    ns = {}
    exec(compile(code, README, "exec"), ns)
    layer = ns["pillar_layer"]
    # geo_params alone sets the geometry and the mesh sizes of the built-in circle
    assert (layer.period, layer.diameter1, layer.lc, layer.lc2) == (400.0, 160.0, 0.15, 2.0)
    T, R = float(ns["ds_spectra"]["T_total"].values[0]), float(ns["ds_spectra"]["R_total"].values[0])
    assert 0.0 < T < 1.0 and 0.0 < R < 1.0
    assert ns["abs_E"].shape == (50, 50) and np.isfinite(ns["abs_E"]).all()
    assert ns["ellipse_layer"].mesh is not None


def test_builtin_shape_geo_params():
    """Without geo_file, geo_params sets the matching arguments; unknown keys raise."""
    from EMUstack.backend import objects, materials
    kw = dict(background=materials.Material(1.0), inclusion_a=materials.Material(2.0),
              height_nm=100.0, make_mesh_now=False)
    a = objects.NanoStruct('2D_array', geo_params={'period': 400.0, 'diameter1': 160.0,
                                                   'lc': 0.15, 'lc2': 2.0, 'lc3': 3.0}, **kw)
    assert (a.period, a.diameter1, a.lc, a.lc2, a.lc3) == (400.0, 160.0, 0.15, 2.0, 3.0)
    # legacy names, and geo_params wins over the keyword arguments
    b = objects.NanoStruct('2D_array', 300.0, diameter1=100.0, lc_bkg=0.2,
                           geo_params={'d_in_nm': 400.0, 'a1': 160.0, 'lc': 0.15}, **kw)
    assert (b.period, b.diameter1, b.lc) == (400.0, 160.0, 0.15)
    with pytest.raises(ValueError, match="diamter1"):
        objects.NanoStruct('2D_array', geo_params={'period': 400.0, 'diamter1': 160.0}, **kw)


def readme_legacy_example_code():
    text = open(README).read()
    text = text[text.index("### 1. Script Syntax Compatibility"):text.index("### 2. Legacy Mesh Assets")]
    return "\n".join(re.findall(r"```python\n(.*?)```", text, re.S))


@pytest.mark.skipif(not HAVE_GMSH, reason="gmsh not installed or cannot start")
def test_readme_legacy_example_runs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(compile(readme_legacy_example_code(), README, "exec"), ns)
    assert len(ns["stacks_list"]) == 3
    assert not [f for f in os.listdir(tmp_path) if f.endswith((".pdf", ".txt"))]  # no file dumps
