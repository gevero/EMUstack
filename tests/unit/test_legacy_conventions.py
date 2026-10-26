"""Conventions the port copies from legacy EMUstack for parity (see benchmarks/parity)."""
import os
import shutil

import numpy as np
import pytest

from EMUstack.backend import paths
from EMUstack.backend.legacy_sort import legacy_argsort, z_indexx

# Expected orders were produced by NumPy 1.24 (oracle image) and by the compiled legacy
# z_indexx.f; equal keys are where the algorithms differ from a stable sort.
ARGSORT_CASES = [
    ([-0.7, 1.5, 0.4, -0.7, 1.1, -0.1, 0.6, -0.6, -0.6, 0.7, 0.1, -0.7, -0.8, 0.7, 1.8, -0.8, 0.7,
      -2.8, -1.3, 0.7, 1.6, 0.1, 0.6, 0.3, 1.0, -1.0, 0.7, -0.4, -0.3, 0.0, -0.2, -0.7],
     [17, 18, 25, 15, 12, 11, 0, 31, 3, 7, 8, 27, 28, 30, 5, 29, 10, 21, 23, 2, 6, 22, 26, 19, 16,
      13, 9, 24, 4, 1, 20, 14]),
    ([1.7, -0.5, -0.1, -1.0, -1.3, 1.1, -0.7, 0.3, -1.4, -0.7, 0.5, 0.5, 0.2, 0.4, 0.2, 1.0, 1.7,
      0.5, -0.1, -1.0, -0.5, -0.3, 1.4, -1.6, -0.8, -1.0, 0.8, -0.7, -1.9],
     [28, 23, 8, 4, 3, 25, 19, 24, 6, 9, 27, 1, 20, 21, 2, 18, 14, 12, 7, 13, 17, 11, 10, 26, 15,
      5, 22, 16, 0]),
]
Z_INDEXX_CASES = [
    ([0.5 - 0.75j, -0.25 - 0.75j, -0.5, 0.5, -0.5j, -0.5 + 0.5j, -0.5 + 0.5j, -0.75 - 0.5j, 0j,
      -0.25j, 0.5, 0.5 + 0.25j, 0.5],
     [7, 12, 2, 3, 10, 11, 8, 6, 5, 9, 4, 0, 1]),
    ([-0.5 + 0.5j, -0.25 + 0.25j, -0.5 + 0.5j, -0.5j, -0.5j, -0.25 + 0.25j, -0.75j, 0.5 - 0.75j,
      -0.5j, 0.5 - 0.75j, 0.5 - 0.5j, 0.25 - 0.75j],
     [0, 1, 10, 5, 2, 4, 8, 3, 7, 9, 11, 6]),
]


@pytest.mark.parametrize('keys, expected', ARGSORT_CASES)
def test_legacy_argsort_matches_numpy_1_24(keys, expected):
    assert list(legacy_argsort(keys)) == expected


@pytest.mark.parametrize('kz, expected', Z_INDEXX_CASES)
def test_z_indexx_matches_fortran(kz, expected):
    assert list(z_indexx(kz)) == expected


def test_ez_conventions():
    from EMUstack.backend import field_eval
    prop_minus = np.array([0.3 - 0.1j, 1.0 + 0.2j])
    kz = np.array([2.0 + 0.1j, 0.5j])
    np.testing.assert_allclose(field_eval.ez_coefficient(prop_minus, kz, physical=False), -prop_minus / kz)
    np.testing.assert_allclose(field_eval.ez_coefficient(prop_minus, kz, physical=True), -1j * kz * prop_minus)
    assert field_eval.ez_convention == 'physical'


def test_conv_gmsh_reproduces_legacy_mesh(tmp_path):
    """The legacy 2D ellipse template meshed as legacy's conv_gmsh does (simo_042 cell):
    legacy's .mail has 1161 nodes and 568 triangles."""
    from EMUstack.backend import meshing, objects
    if not (shutil.which('gmsh') or os.environ.get('GMSH_BIN')):
        pytest.skip('gmsh not available')
    old = objects.msh_location
    objects.msh_location = str(tmp_path) + '/'
    try:
        ns = objects.NanoStruct('2D_array', 165, 140, inc_shape='ellipse', ellipticity=80.0 / 140.0,
                                height_nm=60, make_mesh_now=True, force_mesh=True, lc_bkg=0.2, lc2=5.0)
    finally:
        objects.msh_location = old
    assert (ns.n_msh_pts, ns.n_msh_el) == (1161, 568)
    npt, nel, pts, idfn, table_nod, type_el = meshing.parse_mail_data(str(tmp_path / ns.mesh_file))
    assert sorted(set(idfn)) == [0, 1, 2, 3, 4] and sorted(set(type_el)) == [1, 2]


def test_1d_refuses_more_modes_than_unknowns():
    """Legacy ARPACK stops (info=-3) when 2.5*num_BMs+3 Krylov vectors exceed the unknowns."""
    from EMUstack.backend import objects, materials
    g = objects.NanoStruct('1D_array', 760, 190, height_nm=150, background=materials.Material(1.46),
                           inclusion_a=materials.Material(3.61), loss=True, lc_bkg=0.1, plt_msh=False)
    light = objects.Light(1600.0, max_order_PWs=2, theta=0.0, phi=0.0)
    # fem_1d imports `modes` as a top-level module, so match the message, not the class
    with pytest.raises(Exception, match='mesh too coarse'):
        g.calc_modes(light, num_BMs=60)


def test_physical_ez_satisfies_divergence_1d():
    """In a homogeneous layer div E = dEx/dx + i ky Ey + i beta Ez = 0. The physical
    convention's Ez = i beta ez satisfies it; legacy's ez / beta does not."""
    from EMUstack.backend import objects, materials
    g = objects.NanoStruct('1D_array', 600, 300, height_nm=200, background=materials.Material(1.5),
                           inclusion_a=materials.Material(1.5), loss=True, lc_bkg=0.005,
                           plotting_fields=True, plt_msh=False)
    sim = g.calc_modes(objects.Light(900.0, max_order_PWs=2, theta=30.0, phi=20.0), num_BMs=16)
    sol = np.asarray(sim.sol1)                     # (component, node: end, end, middle, mode, element)
    x = np.asarray(sim.structure.x_arr).ravel()
    ky = sim.k_pll_norm()[1]
    e = 10
    xs = np.array([x[2 * e], 0.5 * (x[2 * e] + x[2 * e + 2]), x[2 * e + 2]])
    for n in range(4):
        beta = sim.k_z[n]
        ex, ey, ez = (sol[c, [0, 2, 1], n, e] for c in range(3))
        dex = np.polyval(np.polyder(np.polyfit(xs, ex, 2)), xs[1])
        ez_true = 1j * (dex + 1j * ky * ey[1]) / beta
        np.testing.assert_allclose(ez_true, 1j * beta * ez[1], rtol=1e-3)
        assert abs(ez_true - ez[1] / beta) > 0.5 * abs(ez_true)
