"""Unit tests for local magnetic field (H) evaluation, optical chirality, and chiral qBIC template."""
import sys
import os
import numpy as np
import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend"))
mesh_dir = os.path.join(backend_dir, "data", "meshes")
for d in (backend_dir, mesh_dir):
    if d not in sys.path:
        sys.path.insert(0, d)

import objects
import materials
import mode_calcs
import field_eval
import plotting
from stack import Stack


def compute_optical_chirality(Ex, Ey, Ez, Hx, Hy, Hz, k0=1.0, E0=1.0):
    """Local helper for chirality testing: C = -0.5*k0*Im(E* . H), C_hat = C / |C_CPL|."""
    E_dot_H = np.conj(Ex) * Hx + np.conj(Ey) * Hy + np.conj(Ez) * Hz
    C = -0.5 * float(k0) * np.imag(E_dot_H)
    C_cpl = 0.5 * float(k0) * (abs(E0)**2)
    C_hat = C / max(C_cpl, 1e-15)
    return C, C_hat


def test_plane_wave_cpl_chirality_calibration():
    """Verify that pure circular plane waves in free space give exactly C_hat = +/- 1 via the helper function."""
    k0 = 2.0 * np.pi / 0.7  # wl = 700 nm in normalized units

    # In Tang & Cohen / Lipkin convention:
    # Forward propagating plane wave: H = (1 / (i k0)) curl(E) = (-Ey, Ex, 0)
    # Right Circular Polarization: E = [1, -1j, 0] / sqrt(2)
    # => H = [1j, 1, 0] / sqrt(2)
    # E* . H = (1 * 1j + 1j * 1) / 2 = 1j => Im(E* . H) = +1
    # C = - (k0 / 2) * (+1) = - k0 / 2, C_hat = -1.0
    # Left Circular Polarization: E = [1, +1j, 0] / sqrt(2)
    # => H = [-1j, 1, 0] / sqrt(2)
    # E* . H = (1 * (-1j) + (-1j) * 1) / 2 = -1j => Im(E* . H) = -1
    # C = - (k0 / 2) * (-1) = + k0 / 2, C_hat = +1.0
    
    # LCP
    E_lcp = np.array([1.0, 1j, 0.0]) / np.sqrt(2.0)
    H_lcp = np.array([-1j, 1.0, 0.0]) / np.sqrt(2.0)
    C_lcp, C_hat_lcp = compute_optical_chirality(
        E_lcp[0], E_lcp[1], E_lcp[2],
        H_lcp[0], H_lcp[1], H_lcp[2],
        k0=k0, E0=1.0
    )
    assert np.isclose(C_hat_lcp, 1.0, atol=1e-12)

    # RCP
    E_rcp = np.array([1.0, -1j, 0.0]) / np.sqrt(2.0)
    H_rcp = np.array([1j, 1.0, 0.0]) / np.sqrt(2.0)
    C_rcp, C_hat_rcp = compute_optical_chirality(
        E_rcp[0], E_rcp[1], E_rcp[2],
        H_rcp[0], H_rcp[1], H_rcp[2],
        k0=k0, E0=1.0
    )
    assert np.isclose(C_hat_rcp, -1.0, atol=1e-12)

    # Linearly polarized light (achiral)
    E_lp = np.array([1.0, 0.0, 0.0])
    H_lp = np.array([0.0, 1.0, 0.0])
    C_lp, C_hat_lp = compute_optical_chirality(
        E_lp[0], E_lp[1], E_lp[2],
        H_lp[0], H_lp[1], H_lp[2],
        k0=k0, E0=1.0
    )
    assert np.isclose(C_hat_lp, 0.0, atol=1e-12)


def test_faraday_curl_consistency_mock():
    """Verify Faraday's law H = (1 / (i k0)) curl(E) in field_value_plane."""
    nel = 2
    n_pts = 6
    num_bms = 1
    k0 = 2.0 * np.pi / 0.5  # wl = 500 nm
    kz = np.array([k0])

    table_nod = np.array([[1, 2], [2, 3], [3, 4], [0, 0], [0, 0], [0, 0]])
    type_el = np.array([1, 1])
    eps = np.array([1.0, 1.0])
    x_arr = np.zeros((2, n_pts))

    # sol1 tensor of shape (6, 13, num_BMs, nel)
    # Mode 0: Ex = 1.0, Ey = 0.0, Ez = 0.0
    # From Faraday: Hx = 0, Hy = (kz / k0) * ux = 1.0, Hz = 0
    sol1 = np.zeros((6, 13, num_bms, nel), dtype=complex)
    sol1[0, :, 0, :] = 1.0  # ux
    sol1[1, :, 0, :] = 0.0  # uy
    sol1[2, :, 0, :] = 0.0  # z
    sol1[3, :, 0, :] = 0.0  # hx_term = -uy - d_y z
    sol1[4, :, 0, :] = 1.0  # hy_term = ux + d_x z
    sol1[5, :, 0, :] = 0.0  # hz_term = curl(u)

    # Forward propagating mode only: coef_down = 1.0, coef_up = 0.0
    vec_coef = np.array([1.0, 0.0])
    h_norm = 1.0
    z_val = 0.5

    m_E, m_H = field_eval.field_value_plane(
        num_BMs=1, n_msh_el=nel, n_msh_pts=n_pts, nnodes=6, nb_typ_el=2,
        table_nod=table_nod, type_el=type_el, eps=eps, x_arr=x_arr,
        k_z=kz, sol1=sol1, vec_coef=vec_coef, h_normed=h_norm, z_value=z_val,
        k0=k0, return_H=True
    )

    prop_expected = np.exp(1j * k0 * z_val)
    np.testing.assert_allclose(m_E[0, 0, 0], prop_expected, rtol=1e-10)
    np.testing.assert_allclose(m_E[0, 0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(m_H[0, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(m_H[0, 0, 1], prop_expected, rtol=1e-10)


def test_faraday_curl_with_longitudinal_gradient_and_backward_wave():
    """Verify Faraday's law H = (1 / (i k0)) curl(E) with non-zero d_x z, d_y z, curl(u), and forward+backward waves."""
    nel = 1
    n_pts = 6
    num_bms = 1
    k0 = 2.0 * np.pi / 0.6  # wl = 600 nm
    kz_val = 0.8 * k0
    kz = np.array([kz_val])

    table_nod = np.array([[1], [2], [3], [4], [5], [6]])
    type_el = np.array([1])
    eps = np.array([1.0])
    x_arr = np.zeros((2, n_pts))

    ux = 1.2 + 0.3j
    uy = -0.5 + 0.8j
    z_pot = 0.4 - 0.2j
    gzx = 0.7 + 0.1j    # d_x z
    gzy = -0.3 + 0.5j   # d_y z
    curl_u = 1.5 - 0.4j # curl_perp(u)

    # Faraday modal terms:
    # Hx_term = -uy - d_y z
    # Hy_term = ux + d_x z
    # Hz_term = curl_perp(u)
    hx_term = -uy - gzy
    hy_term = ux + gzx
    hz_term = curl_u

    sol1 = np.zeros((6, 13, num_bms, nel), dtype=complex)
    sol1[0, :, 0, 0] = ux
    sol1[1, :, 0, 0] = uy
    sol1[2, :, 0, 0] = z_pot
    sol1[3, :, 0, 0] = hx_term
    sol1[4, :, 0, 0] = hy_term
    sol1[5, :, 0, 0] = hz_term

    # Both forward and backward propagating coefficients
    c_down = 1.0 + 0.2j
    c_up = 0.3 - 0.1j
    vec_coef = np.array([c_down, c_up])
    h_norm = 1.0
    z_val = 0.35

    m_E, m_H = field_eval.field_value_plane(
        num_BMs=1, n_msh_el=nel, n_msh_pts=n_pts, nnodes=6, nb_typ_el=1,
        table_nod=table_nod, type_el=type_el, eps=eps, x_arr=x_arr,
        k_z=kz, sol1=sol1, vec_coef=vec_coef, h_normed=h_norm, z_value=z_val,
        k0=k0, return_H=True
    )

    prop_plus = c_down * np.exp(1j * kz_val * z_val) + c_up * np.exp(1j * kz_val * (h_norm - z_val))
    prop_minus = c_down * np.exp(1j * kz_val * z_val) - c_up * np.exp(1j * kz_val * (h_norm - z_val))

    # Analytical E:
    E_expected = np.array([
        prop_plus * ux,
        prop_plus * uy,
        prop_minus * (-1j * kz_val * z_pot)
    ])
    np.testing.assert_allclose(m_E[0, 0, :], E_expected, rtol=1e-12)

    # Analytical H:
    # Hx = (1 / (i k0)) (d_y Ez - d_z Ey)
    # d_y Ez = prop_minus * (-1j * kz * gzy)
    # d_z Ey = (i * kz * prop_minus) * uy
    # d_y Ez - d_z Ey = -i * kz * prop_minus * (uy + gzy)
    # Hx = (1 / (i k0)) * (-i * kz * prop_minus * (uy + gzy)) = (kz / k0) * prop_minus * (-uy - gzy)
    Hx_expected = (kz_val / k0) * prop_minus * (-uy - gzy)

    # Hy = (1 / (i k0)) (d_z Ex - d_x Ez)
    # d_z Ex = (i * kz * prop_minus) * ux
    # d_x Ez = prop_minus * (-1j * kz * gzx)
    # d_z Ex - d_x Ez = i * kz * prop_minus * (ux + gzx)
    # Hy = (1 / (i k0)) * (i * kz * prop_minus * (ux + gzx)) = (kz / k0) * prop_minus * (ux + gzx)
    Hy_expected = (kz_val / k0) * prop_minus * (ux + gzx)

    # Hz = (1 / (i k0)) curl_perp(E) = (1 / (i k0)) * prop_plus * curl_u
    Hz_expected = (1.0 / (1j * k0)) * prop_plus * curl_u

    H_expected = np.array([Hx_expected, Hy_expected, Hz_expected])
    np.testing.assert_allclose(m_H[0, 0, :], H_expected, rtol=1e-12)


def test_chiral_qbic_template_generation():
    """Verify chiral qBIC Netgen geometry template generates a valid mesh."""
    import chiral_qbic_template
    mesh = chiral_qbic_template.build_unit_cell(
        period=700.0,
        alpha=0.15,
        theta0=0.25,
        rx=0.09,
        ry=0.28,
        gap=0.36,
        lc=0.15,
        lc2=2.0
    )
    assert mesh is not None
    assert mesh.nv > 10
    assert mesh.ne > 10
    # Materials must contain bg and inc2
    mats = set(mesh.GetMaterials())
    assert "bg" in mats
    assert "inc2" in mats


def test_nanostruct_sol1_and_interpolators_full_stack():
    """Full simulation verifying sol1 modal reconstruction, backward compatibility, and FieldBundle."""
    period = 700.0
    wl = 700.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    # Chiral qBIC layer with plotting_fields enabled
    layer_struct = objects.NanoStruct(
        '2D_array', period, 200.0,
        geo_file='chiral_qbic_template.py',
        geo_params={'alpha': 0.15, 'lc': 0.20, 'lc2': 1.5},
        background=materials.Material(1.0),
        inclusion_a=materials.Material(3.5),
        height_nm=300.0,
        plotting_fields=1
    )

    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))
    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))

    sim_layer = layer_struct.calc_modes(light, num_BMs=16)
    sim_sup = sup.calc_modes(light)
    sim_sub = sub.calc_modes(light)

    # Assert sol1 is computed and has 6 components
    assert sim_layer.sol1 is not None
    assert sim_layer.sol1.shape[0] == 6
    assert sim_layer.sol1.shape[2] == 16
    # Check that electric and magnetic modal terms are non-zero
    assert np.any(np.abs(sim_layer.sol1[0]) > 0)
    assert np.any(np.abs(sim_layer.sol1[1]) > 0)
    assert np.any(np.abs(sim_layer.sol1[3]) > 0)
    assert np.any(np.abs(sim_layer.sol1[4]) > 0)

    # Build stack and compute scattering under RCP
    stk = Stack([sim_sub, sim_layer, sim_sup])
    stk.calc_scat(pol='RCP')
    assert len(stk.r_list) > 0
    assert len(stk.t_list) > 0

    # 1. Backward compatibility: return_H=False returns 7-tuple
    legacy_tuple = plotting.fields_interpolator_in_plane(stk, lay_interest=1, z_value=150.0, return_H=False)
    assert len(legacy_tuple) == 7
    ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE = legacy_tuple
    assert callable(AbsE)
    abs_e_val = AbsE(0.5, 0.5)
    assert np.isfinite(abs_e_val)

    # 2. Enhanced mode: return_H=True returns FieldBundle (14-tuple)
    bundle = plotting.fields_interpolator_in_plane(stk, lay_interest=1, z_value=150.0, return_H=True)
    assert isinstance(bundle, plotting.FieldBundle)
    assert len(bundle) == 14

    # Can be unpacked as 14-tuple
    rEx, iEx, rEy, iEy, rEz, iEz, aE, rHx, iHx, rHy, iHy, rHz, iHz, aH = bundle
    assert callable(aH)

    # Test evaluations at unit cell center
    e_val = bundle.Ex(0.5, 0.5)
    h_val = bundle.Hx(0.5, 0.5)
    abs_h_val = bundle.AbsH(0.5, 0.5)
    sz_val = bundle.Sz(0.5, 0.5)

    assert np.isfinite(e_val)
    assert np.isfinite(h_val)
    assert np.isfinite(abs_h_val)
    assert abs_h_val >= 0.0
    assert np.isfinite(sz_val)

    # 3. Test backward-compatibility alias return_h=True
    bundle_lower = plotting.fields_interpolator_in_plane(stk, lay_interest=1, z_value=150.0, return_h=True)
    assert isinstance(bundle_lower, plotting.FieldBundle)
    assert np.isclose(bundle_lower.AbsH(0.5, 0.5), abs_h_val, rtol=1e-12)

    # 4. Helper function compute_optical_chirality operating on returned E and H
    k0 = 2.0 * np.pi / (wl / float(period))
    c_val, c_hat_val = compute_optical_chirality(
        bundle.Ex(0.5, 0.5), bundle.Ey(0.5, 0.5), bundle.Ez(0.5, 0.5),
        bundle.Hx(0.5, 0.5), bundle.Hy(0.5, 0.5), bundle.Hz(0.5, 0.5),
        k0=k0, E0=1.0
    )
    assert np.isfinite(c_val)
    assert np.isfinite(c_hat_val)


def test_circular_dichroism_qbic():
    """Verify that chiral qBIC metasurface produces circular dichroism CD = T_LCP - T_RCP != 0."""
    period = 700.0
    wl = 700.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    layer_struct = objects.NanoStruct(
        '2D_array', period, 200.0,
        geo_file='chiral_qbic_template.py',
        geo_params={'alpha': 0.15, 'lc': 0.20, 'lc2': 1.5},
        background=materials.Material(1.0),
        inclusion_a=materials.Material(3.5),
        height_nm=300.0
    )

    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))
    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))

    sim_layer = layer_struct.calc_modes(light, num_BMs=16)
    sim_sup = sup.calc_modes(light)
    sim_sub = sub.calc_modes(light)

    stk_rcp = Stack([sim_sub, sim_layer, sim_sup])
    stk_rcp.calc_scat(pol='RCP')
    T_rcp = float(stk_rcp.t_list[-1])

    stk_lcp = Stack([sim_sub, sim_layer, sim_sup])
    stk_lcp.calc_scat(pol='LCP')
    T_lcp = float(stk_lcp.t_list[-1])

    cd = T_lcp - T_rcp
    # Asymmetric structure must have non-zero circular dichroism
    assert abs(cd) > 1e-4, f"Expected non-zero Circular Dichroism, got {cd}"

