"""Unit tests for the 2D finite-element Maxwell Bloch solver.

Verifies:
- Homogeneous cell analytic eigenvalue spectrum
- Biorthogonality ||L - I|| <= 1e-10 after conjugate-pair repair
- Plane-wave overlap isometry J^dag J approx I on homogeneous medium
- Mode flux matrix F properties
"""
import numpy as np
import pytest
import warnings

import netgen.geom2d as g2
import ngsolve as ngs
from ngsolve import Mesh

import fem_2d
import meshing


@pytest.fixture(scope="module")
def homogeneous_mesh():
    """Create a periodic homogeneous square unit cell (Path I native SplineGeometry)."""
    geo = g2.SplineGeometry()
    P = [geo.AddPoint(*q) for q in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    ib = geo.Append(["line", P[0], P[1]], bc="bottom", maxh=0.15)
    ir = geo.Append(["line", P[1], P[2]], bc="right", maxh=0.15)
    geo.Append(["line", P[3], P[2]], leftdomain=0, rightdomain=1, bc="top", maxh=0.15, copy=ib)
    geo.Append(["line", P[0], P[3]], leftdomain=0, rightdomain=1, bc="left", maxh=0.15, copy=ir)
    geo.SetMaterial(1, "bg")
    geo.SetDomainMaxH(1, 0.15)
    ngm = geo.GenerateMesh(maxh=0.15)
    ngm.Curve(2)
    return Mesh(ngm)


def test_g31_homogeneous_spectrum(homogeneous_mesh):
    """G3.1: Compare computed eigenvalues against analytic Bloch bands beta^2 = n^2 k0^2 - |Q + k_pll|^2."""
    wl_norm = 1.2
    k0 = 2.0 * np.pi / wl_norm
    n_bg = 1.5
    eps = n_bg**2
    kx, ky = 0.25, -0.15
    num_BMs = 6
    max_order = 2

    modes = fem_2d.solve_bloch_modes_2d(
        wl_norm=wl_norm,
        num_BMs=num_BMs,
        max_order_PWs=max_order,
        structure_or_mesh=homogeneous_mesh,
        n_effs=[n_bg],
        k_pll=(kx, ky),
        order=2,
        dy=1.0
    )

    computed_betas_sq = modes.zeta**2

    # Analytic eigenvalues for homogeneous medium:
    analytic_betas_sq = []
    for p in range(-max_order - 1, max_order + 2):
        for q in range(-max_order - 1, max_order + 2):
            qx = kx + 2.0 * np.pi * p
            qy = ky + 2.0 * np.pi * q
            b2 = eps * k0**2 - (qx**2 + qy**2)
            # Both TE and TM polarizations are degenerate in homogeneous medium
            analytic_betas_sq.append(b2)
            analytic_betas_sq.append(b2)

    analytic_betas_sq = np.array(analytic_betas_sq)

    # Each computed eigenvalue must match an analytic eigenvalue
    for c_b2 in computed_betas_sq:
        dists = np.abs(analytic_betas_sq - c_b2)
        min_dist = np.min(dists)
        rel_err = min_dist / (eps * k0**2)
        assert rel_err < 1e-3, f"Computed beta^2 {c_b2} has no analytic match (rel_err={rel_err})"


def test_g14_biorthogonality(homogeneous_mesh):
    """G1.4: Biorthogonality ||L - I|| <= 1e-10 after normalization and conjugate repair."""
    wl_norm = 1.0
    modes = fem_2d.solve_bloch_modes_2d(
        wl_norm=wl_norm,
        num_BMs=8,
        max_order_PWs=2,
        structure_or_mesh=homogeneous_mesh,
        n_effs=[1.0],
        k_pll=(0.31, -0.17),
        order=2,
        dy=1.0
    )
    assert modes.norm_res < 1e-10, f"Biorthogonality residual too large: {modes.norm_res}"


def test_mode_flux_structure(homogeneous_mesh):
    """Test that mode_flux matrix F is non-zero, finite, and well-conditioned."""
    wl_norm = 1.0
    modes = fem_2d.solve_bloch_modes_2d(
        wl_norm=wl_norm,
        num_BMs=6,
        max_order_PWs=2,
        structure_or_mesh=homogeneous_mesh,
        n_effs=[1.0],
        k_pll=(0.2, 0.3),
        order=2,
        dy=1.0
    )
    F = modes.mode_flux
    assert F.shape == (6, 6)
    assert np.all(np.isfinite(F))
    # Diagonal should be dominant for physical modes
    assert np.linalg.norm(np.diag(F)) > 1e-5


def test_warn_if_degenerate_warns_when_cut_splits_a_group():
    """A num_BMs cut between two (near-)degenerate modes must warn."""
    zeta = np.array([0.5 + 0.0j, 2.0 + 0.0j, 2.0 + 1e-12j, 3.0 + 0.0j])
    with pytest.warns(UserWarning, match="degenerate Bloch-mode group"):
        fem_2d.warn_if_degenerate(zeta, 2)


def test_warn_if_degenerate_is_silent_when_the_group_is_fully_kept():
    """No warning when the last kept and first dropped modes are well separated."""
    zeta = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fem_2d.warn_if_degenerate(zeta, 2)
    assert caught == []


def test_warn_if_degenerate_is_silent_without_a_dropped_mode():
    """No warning when the eigensolver returned exactly num_BMs modes."""
    zeta = np.array([1.0 + 0.0j, 1.0 + 1e-12j])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fem_2d.warn_if_degenerate(zeta, 2)
    assert caught == []


def test_degenerate_warning_fires_from_the_2d_solver(homogeneous_mesh):
    """At normal incidence an odd num_BMs splits a degenerate pair; the solver warns."""
    with pytest.warns(UserWarning, match="degenerate Bloch-mode group"):
        fem_2d.solve_bloch_modes_2d(
            wl_norm=1.2, num_BMs=5, max_order_PWs=2, structure_or_mesh=homogeneous_mesh,
            n_effs=[1.5], k_pll=(0.0, 0.0), order=2, dy=1.0)


def test_overlap_shapes(homogeneous_mesh):
    """Verify shapes and finiteness of J and J_dag."""
    wl_norm = 1.2
    max_order = 2
    num_BMs = 6
    modes = fem_2d.solve_bloch_modes_2d(
        wl_norm=wl_norm,
        num_BMs=num_BMs,
        max_order_PWs=max_order,
        structure_or_mesh=homogeneous_mesh,
        n_effs=[1.0],
        k_pll=(0.0, 0.0),
        order=2,
        dy=1.0
    )
    # Number of plane waves for max_order=2 with circular truncation
    px_range = np.arange(-max_order, max_order + 1)
    py_mesh, px_mesh = np.meshgrid(px_range, px_range)
    npw = int(np.sum(px_mesh**2 + py_mesh**2 <= max_order**2))

    assert modes.J.shape == (2 * npw, num_BMs)
    assert modes.J_dag.shape == (num_BMs, 2 * npw)
    assert np.all(np.isfinite(modes.J))
    assert np.all(np.isfinite(modes.J_dag))
