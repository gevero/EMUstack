"""The code in docs/source/tutorial.rst runs as written."""
import os
import re

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
