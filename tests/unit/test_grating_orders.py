"""Plane-wave (grating order) selection shared by thin films and FEM layers.

Thin films select the orders with the periods in nm, FEM layers with the periods in
units of the period. Both must keep the same orders, or the stack cannot be built.
"""
import numpy as np
import pytest

from EMUstack.backend import objects, materials
from EMUstack.backend.legacy_sort import grating_orders_2d
from EMUstack.backend.stack import Stack


def legacy_orders(max_order, dx, dy):
    """The selection of legacy EMUstack (pw_ordering.f, calc_2d_grating_orders)."""
    p = np.arange(-max_order, max_order + 1)
    pys, pxs = np.meshgrid(p, p)
    keep = (pxs / dx)**2 + (pys / dy)**2 <= (max_order / max(dx, dy))**2
    return pxs[keep], pys[keep]


def same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("period, max_order", [(480.0, 5), (217.0, 3), (480.0, 10)])
def test_nm_and_normalized_units_agree(period, max_order):
    # Legacy kept different orders in the two unit systems for these cases (on Linux; the
    # cut falls on the boundary, so it depends on the last bit of floating-point division).
    assert same(grating_orders_2d(max_order, period, period), grating_orders_2d(max_order, 1.0, 1.0))


def test_boundary_orders_are_kept():
    pxs, pys = grating_orders_2d(5, 480.0, 480.0)
    assert len(pxs) == 81  # integer pairs with px**2 + py**2 <= 25
    kept = set(zip(pxs.tolist(), pys.tolist()))
    assert {(3, 4), (4, 3), (-3, -4), (5, 0), (0, -5)} <= kept


@pytest.mark.parametrize("period, max_order", [
    (165.0, 2), (600.0, 2), (600.0, 4), (600.0, 15), (940.0, 5), (300.0, 1), (660.0, 0), (0.012, 2),
])
def test_legacy_parity_configurations_are_unchanged(period, max_order):
    # Periods and orders of the legacy parity examples: same orders, same order.
    for dx, dy in ((period, period), (1.0, 1.0)):
        assert same(grating_orders_2d(max_order, dx, dy), legacy_orders(max_order, dx, dy))


def test_rectangular_lattice_matches_legacy_where_legacy_ran():
    for max_order in range(0, 8):
        for dx, dy in ((600.0, 400.0), (300.0, 450.0), (500.0, 700.0)):
            nm, norm = legacy_orders(max_order, dx, dy), legacy_orders(max_order, 1.0, dy / dx)
            if same(nm, norm):
                assert same(grating_orders_2d(max_order, dx, dy), nm)
            assert same(grating_orders_2d(max_order, dx, dy), grating_orders_2d(max_order, 1.0, dy / dx))


def test_stack_runs_where_legacy_stopped():
    # 217 nm period, pw = 3: legacy dropped the (+-3, 0), (0, +-3) orders in nm only.
    period = 217.0
    light = objects.Light(500.0, max_order_PWs=3, theta=0.0, phi=0.0)
    air = objects.ThinFilm(period, height_nm='semi_inf', material=materials.Air)
    holes = objects.NanoStruct('2D_array', period, 100.0, height_nm=50.0, mesher='netgen',
                               background=materials.Material(2.0), inclusion_a=materials.Air,
                               lc_bkg=0.2, lc2=1.0)
    modes = [air.calc_modes(light), holes.calc_modes(light), air.calc_modes(light)]
    assert modes[0].structure.num_pw_per_pol == 29
    stack = Stack(modes)
    stack.calc_scat(pol='TE')
    t, r = float(stack.t_list[-1]), float(stack.r_list[-1])
    assert 0.0 < t < 1.0 and 0.0 < r < 1.0 and t + r <= 1.0 + 1e-6  # lossless: t + r = 1
