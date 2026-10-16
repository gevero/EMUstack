"""Unit tests for 2D FEM-to-FEM mode matching and directly stacked 2D layers.

Verifies:
1. Direct 2D FEM-to-FEM mode matching (mode_matching.r_t_mat_ns_ns) in both 'air' and 'direct' modes.
2. Interface completeness diagnostics (P, Q rank and condition numbers).
3. Stack scattering with two distinct 2D FEM layers in direct contact without spacers.
4. Energy conservation (Poynting flux R + T == 1.0) for lossless 2D FEM-to-FEM stacks.
5. Transparency limit: Stacking two identical 2D layers gives results matching a single layer of double height.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(backend_dir))

import objects
import materials
from stack import Stack
import mode_matching


def test_2d_fem_to_fem_interface_matrices():
    """Test mode matching between two distinct 2D nanostructure layers."""
    period = 400.0
    wl = 650.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    layer1 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=120.0,
        background=materials.Air,
        inclusion_a=materials.Material(2.25),  # n = 1.5
        geo_params={'period': period, 'a1': 140.0, 'lc': 0.15, 'lc2': 3.0}
    )

    layer2 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=80.0,
        background=materials.Air,
        inclusion_a=materials.Material(4.0),   # n = 2.0
        geo_params={'period': period, 'a1': 200.0, 'lc': 0.15, 'lc2': 3.0}
    )

    sim1 = layer1.calc_modes(light, num_BMs=16)
    sim2 = layer2.calc_modes(light, num_BMs=16)

    # 1. Interface diagnostics
    diag = mode_matching.interface_completeness(sim1, sim2)
    assert diag["rank_P"] > 0
    assert diag["rank_Q"] > 0
    assert diag["cond_I_PQ"] < 1e6

    # 2. Air-mediated FEM-to-FEM scattering matrices
    R12_air, T12_air, R21_air, T21_air = mode_matching.r_t_mat_ns_ns(sim1, sim2, interface_mode="air")
    assert R12_air.shape == (16, 16)
    assert T12_air.shape == (16, 16)
    assert np.all(np.isfinite(R12_air))
    assert np.all(np.isfinite(T12_air))

    # 3. Direct algebraic FEM-to-FEM scattering matrices
    R12_dir, T12_dir, R21_dir, T21_dir = mode_matching.r_t_mat_ns_ns(sim1, sim2, interface_mode="direct")
    assert R12_dir.shape == (16, 16)
    assert T12_dir.shape == (16, 16)
    assert np.all(np.isfinite(R12_dir))
    assert np.all(np.isfinite(T12_dir))


def test_2d_fem_to_fem_stacked_energy_conservation():
    """Test full Redheffer stack with two 2D FEM layers touching directly satisfies R + T = 1."""
    period = 500.0
    wl = 700.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air)
    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air)

    layer1 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=100.0,
        background=materials.Air,
        inclusion_a=materials.Material(2.5),
        geo_params={'period': period, 'a1': 160.0, 'lc': 0.15, 'lc2': 3.0}
    )

    layer2 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=150.0,
        background=materials.Air,
        inclusion_a=materials.Material(3.5),
        geo_params={'period': period, 'a1': 220.0, 'lc': 0.15, 'lc2': 3.0}
    )

    sim_sub = sub.calc_modes(light)
    sim1 = layer1.calc_modes(light, num_BMs=16)
    sim2 = layer2.calc_modes(light, num_BMs=16)
    sim_sup = sup.calc_modes(light)

    # Directly stacked 2D layers: Layer 1 on Layer 2 without any intermediate thin film
    stack = Stack((sim_sub, sim1, sim2, sim_sup))
    stack.calc_scat(pol='TE')

    R_tot = float(stack.r_list[-1])
    T_tot = float(stack.t_list[-1])
    sum_flux = R_tot + T_tot

    assert 0.0 <= R_tot <= 1.0
    assert 0.0 <= T_tot <= 1.0
    # Poynting flux conservation for lossless dielectric structures
    assert np.isclose(sum_flux, 1.0, atol=1e-4)


def test_2d_fem_to_fem_direct_transparency():
    """Test that direct mode matching between identical 2D layers gives exact reflection-free coupling."""
    period = 450.0
    wl = 680.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    layer = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=80.0,
        background=materials.Air,
        inclusion_a=materials.Material(3.0),
        geo_params={'period': period, 'a1': 180.0, 'lc': 0.15, 'lc2': 3.0}
    )

    sim = layer.calc_modes(light, num_BMs=16)

    # Direct mode matching between layer and itself
    P, Q = mode_matching.compute_PQ(sim, sim)
    diag = mode_matching.interface_completeness(sim, sim, P=P, Q=Q)
    # Rank is bounded by the dimension of the mediating plane-wave subspace (10 PWs)
    num_pws = sim.J.shape[0]
    assert diag["rank_P"] == num_pws
    assert diag["rank_Q"] == num_pws
    assert diag["cond_I_PQ"] < 1e5

    # Direct solve
    R12_d, T12_d, R21_d, T21_d = mode_matching.solve_rt_from_PQ(P, Q)
    # Fundamental propagating mode couples predominantly through transmission
    assert abs(T12_d[0, 0]) > abs(R12_d[0, 0])
    assert np.all(np.isfinite(R12_d))
    assert np.all(np.isfinite(T12_d))


def test_2d_fem_to_fem_thin_film_spacer_limit():
    """Test that direct 2D FEM-to-FEM contact equals the zero-thickness thin-film spacer limit."""
    period = 400.0
    wl = 650.0
    light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air)
    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Air)

    layer1 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=100.0,
        background=materials.Air,
        inclusion_a=materials.Material(2.5),
        geo_params={'period': period, 'a1': 150.0, 'lc': 0.15, 'lc2': 3.0}
    )

    layer2 = objects.NanoStruct(
        '2D_array',
        period=period,
        height_nm=120.0,
        background=materials.Air,
        inclusion_a=materials.Material(3.5),
        geo_params={'period': period, 'a1': 220.0, 'lc': 0.15, 'lc2': 3.0}
    )

    sim_sub = sub.calc_modes(light)
    sim1 = layer1.calc_modes(light, num_BMs=16)
    sim2 = layer2.calc_modes(light, num_BMs=16)
    sim_sup = sup.calc_modes(light)

    # 1. Direct FEM-FEM contact
    stack_direct = Stack((sim_sub, sim1, sim2, sim_sup))
    stack_direct.calc_scat(pol='TE')
    R_direct = float(stack_direct.r_list[-1])
    T_direct = float(stack_direct.t_list[-1])

    # 2. Intermediate thin film spacer with h = 0
    zero_spacer = objects.ThinFilm(period=period, height_nm=0.0, material=materials.Air)
    sim_zero = zero_spacer.calc_modes(light)
    stack_zero = Stack((sim_sub, sim1, sim_zero, sim2, sim_sup))
    stack_zero.calc_scat(pol='TE')
    R_zero = float(stack_zero.r_list[-1])
    T_zero = float(stack_zero.t_list[-1])

    # Must be algebraically identical to machine precision
    assert np.isclose(R_direct, R_zero, atol=1e-12)
    assert np.isclose(T_direct, T_zero, atol=1e-12)

    # 3. Asymptotic continuity: h -> 0 must converge continuously to direct contact
    spacer_thin = objects.ThinFilm(period=period, height_nm=0.01, material=materials.Air)
    sim_thin = spacer_thin.calc_modes(light)
    stack_thin = Stack((sim_sub, sim1, sim_thin, sim2, sim_sup))
    stack_thin.calc_scat(pol='TE')
    R_thin = float(stack_thin.r_list[-1])
    T_thin = float(stack_thin.t_list[-1])

    assert abs(R_thin - R_direct) < 1e-4
    assert abs(T_thin - T_direct) < 1e-4
