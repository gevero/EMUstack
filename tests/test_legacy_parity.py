"""Parity with legacy EMUstack (Fortran oracle) on small, fast scenarios.

References: tests/ref/parity/<case>.npz, produced from the oracle image by
benchmarks/parity/make_test_refs.py. Each scenario below is the corresponding
legacy example / parity case reduced to plain function form; the full set of
examples is compared by benchmarks/parity (see PARITY_RESULTS.md).
"""
import os

import numpy as np
import pytest

from EMUstack.backend import objects, materials, plotting
from EMUstack.backend.stack import Stack

REF = os.path.join(os.path.dirname(__file__), 'ref', 'parity')


def load_ref(case):
    path = os.path.join(REF, case + '.npz')
    if not os.path.exists(path):
        pytest.skip('missing reference %s' % path)
    return np.load(path)


def flat(v):
    if isinstance(v, list):
        return np.concatenate([np.ravel(np.asarray(x)) for x in v]) if v else np.zeros(0)
    return np.ravel(np.asarray(v))


def order_free(a):
    """Eigenvalues and sorted |entries|: unchanged by a common reordering of plane waves."""
    a = np.asarray(a)
    return np.sort_complex(np.linalg.eigvals(a)), np.sort(np.abs(a).ravel())


def check_stacks(case, stacks, tol, net='exact'):
    ref = load_ref(case)
    names = list(ref['stack_names'])
    checked = 0
    for name, stack in stacks.items():
        if name not in names:
            continue
        i = names.index(name)
        for q in ('t_list', 'r_list', 'a_list'):
            np.testing.assert_allclose(np.real(flat(getattr(stack, q))), ref['%s_%d' % (q, i)],
                                       rtol=0, atol=tol, err_msg='%s %s %s' % (case, name, q))
        for q in ('R_net', 'T_net'):
            key = '%s_%d' % (q, i)
            if key not in ref.files:
                continue
            if net == 'exact':
                np.testing.assert_allclose(np.asarray(getattr(stack, q)), ref[key], rtol=1e-9, atol=10 * tol)
            else:  # near-tied plane waves may be ordered differently (NumPy sin/cos last bit)
                for a, b in zip(order_free(getattr(stack, q)), order_free(ref[key])):
                    np.testing.assert_allclose(a, b, rtol=1e-9, atol=10 * tol)
        checked += 1
    assert checked > 0, 'no stack of %s compared' % case
    return ref


def test_thin_film_mirror_simo_021():
    """Thin films with a 100 um Ag mirror (legacy simo_021): analytic layers."""
    light_list = [objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)
                  for wl in np.linspace(400, 800, 4)]
    period = 300
    sup = objects.ThinFilm(period, height_nm='semi_inf', material=materials.Air)
    tf2 = objects.ThinFilm(period, height_nm=5e6, material=materials.InP, loss=False)
    tf3 = objects.ThinFilm(period, height_nm=52, material=materials.Si_a)
    mirror = objects.ThinFilm(period, height_nm=1e5, material=materials.Ag)
    sub = objects.ThinFilm(period, height_nm='semi_inf', material=materials.Air)
    stacks = {}
    for i, light in enumerate(light_list):
        s = Stack((sub.calc_modes(light), mirror.calc_modes(light), tf3.calc_modes(light),
                   tf2.calc_modes(light), sup.calc_modes(light)))
        s.calc_scat(pol='TM')
        stacks['stacks_list[%d]' % i] = s
    # 5e-12: newer numpy/BLAS builds differ from the oracle by up to ~3e-12 here
    check_stacks('simo_021', stacks, 5e-12)


def test_1d_grating_simo_030():
    """Lamellar grating, 1D FEM (legacy simo_030): legacy 1D inclusion rule."""
    light_list = [objects.Light(wl, max_order_PWs=5, theta=0.0, phi=0.0)
                  for wl in np.linspace(400, 800, 2)]
    period = 300
    sup = objects.ThinFilm(period, height_nm='semi_inf', world_1d=True, material=materials.Air)
    sub = objects.ThinFilm(period, height_nm='semi_inf', world_1d=True, material=materials.Air)
    grating = objects.NanoStruct('1D_array', period, int(round(0.75 * period)), height_nm=2900,
                                 background=materials.Material(1.46 + 0.0j),
                                 inclusion_a=materials.Material(5.0 + 0.0j), loss=True, lc_bkg=0.0051)
    stacks = {}
    for i, light in enumerate(light_list):
        s = Stack((sub.calc_modes(light), grating.calc_modes(light), sup.calc_modes(light)))
        s.calc_scat(pol='TE')
        stacks['stacks_list[%d]' % i] = s
    check_stacks('simo_030', stacks, 1e-8)


def test_stacked_1d_gratings_oblique_tc_4o():
    """Two touching 1D gratings at 28 deg (test_case_4 geometry, oblique)."""
    light = objects.Light(800.0, max_order_PWs=6, theta=28.0, phi=0.0)
    period = 760
    sup = objects.ThinFilm(period, height_nm='semi_inf', world_1d=True, material=materials.Air, loss=False)
    sub = objects.ThinFilm(period, height_nm='semi_inf', world_1d=True, material=materials.Air, loss=False)
    g1 = objects.NanoStruct('1D_array', period, diameter1=int(round(0.25 * period)),
                            diameter2=int(round(0.25 * period)), height_nm=150,
                            inclusion_a=materials.Material(1.46 + 0.0j), inclusion_b=materials.Material(1.46 + 0.0j),
                            background=materials.Material(3.61 + 0.0j), loss=True, lc_bkg=0.005)
    g2 = objects.NanoStruct('1D_array', period, int(round(0.25 * period)), height_nm=900,
                            background=materials.Material(3.61 + 0.0j), inclusion_a=materials.Material(1.46 + 0.0j),
                            loss=True, lc_bkg=0.005)
    s = Stack((sub.calc_modes(light), g1.calc_modes(light), g2.calc_modes(light), sup.calc_modes(light)))
    s.calc_scat(pol='TE')
    # 6e-9 from the Fortran reference on Linux, 2.4e-8 on macOS (Accelerate LAPACK)
    check_stacks('tc_4o', {'stack': s}, 1e-7)


def _au_ellipse_holes():
    period, diam1, diam2 = 165, 140, 60
    ns = objects.NanoStruct('2D_array', period, diam1, inc_shape='ellipse',
                            ellipticity=float(diam1 - diam2) / float(diam1), height_nm=60,
                            inclusion_a=materials.Air, background=materials.Au_drude, loss=True,
                            make_mesh_now=True, force_mesh=True, lc_bkg=0.2, lc2=5.0)
    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air, loss=True)
    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air, loss=False)
    return ns, sup, sub


def test_2d_oblique_all_polarisations_tc_042pol():
    """simo_042 geometry at theta = phi = 45 deg, TE/TM/RCP/LCP (2D FEM, legacy mesh)."""
    ns, sup, sub = _au_ellipse_holes()
    stacks = {}
    for i, wl in enumerate(np.linspace(300, 1000, 21)[[0, 10, 20]]):
        light = objects.Light(wl, theta=45, phi=45, max_order_PWs=2)
        sim_sup, sim_ns, sim_sub = sup.calc_modes(light), ns.calc_modes(light, num_BMs=50), sub.calc_modes(light)
        for pol, name in (('TE', 'stacks_te'), ('TM', 'stacks_tm'), ('R Circ', 'stacks_rcp'), ('L Circ', 'stacks_lcp')):
            s = Stack((sim_sub, sim_ns, sim_sup))
            s.calc_scat(pol=pol)
            stacks['%s[%d]' % (name, i)] = s
    check_stacks('tc_042pol', stacks, 1e-8, net='order_free')


def test_2d_fields_oblique_tc_f2do(monkeypatch):
    """Au disks at theta = 30, phi = 20 deg: spectra and in-plane E maps, with legacy's Ez
    formula (the default is the physical Ez; field_eval is imported under two names)."""
    import sys
    for name in ('field_eval', 'EMUstack.backend.field_eval'):
        mod = sys.modules.get(name) or __import__(name, fromlist=['x'])
        monkeypatch.setattr(mod, 'ez_convention', 'legacy')
    light = objects.Light(634.6456692913387, max_order_PWs=2, theta=30.0, phi=20.0)
    ns = objects.NanoStruct('2D_array', 600, 200.0, height_nm=100, inclusion_a=materials.Au,
                            background=materials.Air, loss=True, inc_shape='circle', plotting_fields=True,
                            plot_real=1, make_mesh_now=True, force_mesh=True, lc_bkg=0.12, lc2=5.0, lc3=3.0)
    sup = objects.ThinFilm(period=600, height_nm='semi_inf', material=materials.Air, loss=False)
    sub = objects.ThinFilm(period=600, height_nm='semi_inf', material=materials.Air, loss=False)
    s = Stack((sub.calc_modes(light), ns.calc_modes(light), sup.calc_modes(light)))
    s.calc_scat(pol='TM')
    ref = check_stacks('tc_f2do', {'stack': s}, 1e-8)
    n = 61
    # legacy evaluated the gmsh cell at y in [-1, 0]; the interpolators now use y in [0, 1]
    # for every mesher, so the same physical points are at y + 1
    xs, ys = np.linspace(0.005, 0.995, n), np.linspace(-0.995, -0.005, n) + 1.0
    vx, vy = np.repeat(xs, n), np.tile(ys, n)
    for z in (0.1, 50.0):
        ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE = plotting.fields_interpolator_in_plane(s, lay_interest=1, z_value=z)
        tag = 'z01' if z == 0.1 else 'z50'
        for comp, (re, im) in (('Ex', (ReEx, ImEx)), ('Ey', (ReEy, ImEy)), ('Ez', (ReEz, ImEz))):
            key = 'field_%s_%s' % (comp, tag)
            if key not in ref.files:
                continue
            got = np.asarray(re(vx, vy)) + 1j * np.asarray(im(vx, vy))
            want = ref[key]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6 * np.max(np.abs(want)), err_msg=key)
