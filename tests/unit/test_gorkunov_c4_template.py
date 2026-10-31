"""Tests for the C4-symmetric Gorkunov qBIC template (four bar dimers per cell)."""
import numpy as np
import ngsolve as ngs
import pytest

from EMUstack.backend.data.meshes import chiral_qbic_gorkunov_c4_msh_template as c4_cell


def _outline(inc, n=400):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    (cx, cy), rx, ry, a = inc["center"], inc["rx"], inc["ry"], inc["angle"]
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.c_[cx + x * np.cos(a) - y * np.sin(a), cy + x * np.sin(a) + y * np.cos(a)]


@pytest.mark.parametrize("theta_deg", [0.0, 3.0])
def test_bars_are_c4_symmetric_and_do_not_touch(theta_deg):
    incs = c4_cell.bar_inclusions(theta_deg=theta_deg)
    assert len(incs) == 8

    # Rotating the cell by 90 degrees about its centre maps the set of bars onto itself.
    def key(inc):
        return (round(inc["center"][0], 9), round(inc["center"][1], 9),
                round(np.mod(inc["angle"], np.pi), 9))
    rotated = []
    for inc in incs:
        x, y = inc["center"][0] - 0.5, inc["center"][1] - 0.5
        rotated.append({"center": (0.5 - y, 0.5 + x), "angle": inc["angle"] + np.pi / 2})
    assert sorted(map(key, incs)) == sorted(map(key, rotated))

    # All bars stay inside the cell and keep a gap to each other, also across cell edges.
    pts = [_outline(inc) for inc in incs]
    allp = np.vstack(pts)
    assert allp.min() > 0.0 and allp.max() < 1.0
    for i in range(8):
        for j in range(8):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    if i == j and sx == 0 and sy == 0:
                        continue
                    d = np.min(np.linalg.norm(pts[i][:, None] - (pts[j] + [sx, sy])[None], axis=2))
                    assert d * 480.0 > 5.0


def test_parallel_bars_keep_the_paper_gap():
    incs = c4_cell.bar_inclusions(theta_deg=0.0)
    d = np.min(np.linalg.norm(_outline(incs[0])[:, None] - _outline(incs[1])[None], axis=2))
    assert d * 480.0 == pytest.approx(15.0, abs=0.05)


def test_mesh_has_the_bar_area():
    mesh = c4_cell.build_unit_cell(theta_deg=3.0)
    bars = [m for m in mesh.GetMaterials() if m != "bg"]
    area = ngs.Integrate(ngs.CoefficientFunction(1.0), mesh,
                         definedon=mesh.Materials("|".join(bars)))
    expected = 8 * np.pi * (35.0 / 480.0) * (110.0 / 480.0)
    assert area == pytest.approx(expected, rel=2e-3)


def test_include_flags_keep_one_bar_per_dimer():
    assert len(c4_cell.bar_inclusions(include_bar2=False)) == 4
    assert len(c4_cell.bar_inclusions(include_bar1=False)) == 4


def test_rectangular_lattice_is_rejected():
    with pytest.raises(ValueError):
        c4_cell.build_unit_cell(period=480.0, period_y=500.0)
