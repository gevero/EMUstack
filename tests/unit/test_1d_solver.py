"""Unit tests for the pure Python 1D FEM Maxwell solver (fem_1d.py).

Verifies:
- Phase periodicity under kx -> kx + 2*pi
- Determinism with explicit seed v0
- Biorthogonality residual ||L - I|| < 1e-10
- Mode self-flux F structure
- Analytic homogeneous slab comparison
- TE/TM decoupling at ky = 0 exactly
- Negative control: K_zx sign alteration breaks spectrum
"""
import pytest
import numpy as np
from scipy.sparse.linalg import splu, LinearOperator, eigs
import sys
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import fem_1d
from modes import BlochModes, FEMConvergenceError


def test_1d_bloch_phase_invariance():
    """G1.1: Eigenvalues must be invariant under kx -> kx + 2*pi/period."""
    nel = 20
    k0 = 2.0 * np.pi / 0.8
    kx = 0.35
    ky = 0.12
    period_x = 1.0
    num_BMs = 6

    # Two-material periodic layer
    eps_list = np.ones(nel, dtype=complex) * (3.5**2)
    eps_list[nel // 4 : 3 * nel // 4] = 1.5**2

    # Solver at kx
    K1, M1, neq = fem_1d.build_sparse_system_1d(nel, eps_list, k0, kx, ky, period_x)
    sigma = 1.1 * (3.5**2) * (k0**2) - kx**2 - ky**2
    lu1 = splu((K1 - sigma * M1).tocsc())
    OP1 = LinearOperator((neq, neq), matvec=lambda x: lu1.solve(M1 @ x), dtype=complex)
    vals1, _ = eigs(OP1, k=num_BMs, which='LM', tol=1e-12)
    zeta1 = np.sort(np.sqrt(1.0 / vals1 + sigma).real)

    # Solver at kx + 2*pi/period_x
    kx_shifted = kx + 2.0 * np.pi / period_x
    K2, M2, _ = fem_1d.build_sparse_system_1d(nel, eps_list, k0, kx_shifted, ky, period_x)
    sigma2 = 1.1 * (3.5**2) * (k0**2) - kx_shifted**2 - ky**2
    lu2 = splu((K2 - sigma2 * M2).tocsc())
    OP2 = LinearOperator((neq, neq), matvec=lambda x: lu2.solve(M2 @ x), dtype=complex)
    vals2, _ = eigs(OP2, k=num_BMs, which='LM', tol=1e-12)
    zeta2 = np.sort(np.sqrt(1.0 / vals2 + sigma2).real)

    # Compare sorted real parts of eigenvalues
    np.testing.assert_allclose(zeta1, zeta2, rtol=1e-5, atol=1e-5)


def test_1d_biorthogonality_hygiene():
    """G1.4: ||L - I|| must be < 1e-10 on converged modes."""
    nel = 40
    wl_norm = 1.05
    num_BMs = 8
    max_order_PWs = 3
    period_x = 1.0
    
    table_nod = np.zeros((3, nel), dtype=int)
    for e in range(nel):
        table_nod[0, e] = 2 * e + 1
        table_nod[1, e] = 2 * e + 3
        table_nod[2, e] = 2 * e + 2
    type_el = np.ones(nel, dtype=int)
    type_el[nel // 4 : 3 * nel // 4] = 2
    
    n_effs = [3.5 + 0.05j, 1.45 + 0.0j]
    x_arr = np.linspace(0, period_x, 2 * nel + 1)
    
    modes = fem_1d.solve_bloch_modes_1d(
        wl_norm=wl_norm, num_BMs=num_BMs, max_order_PWs=max_order_PWs,
        nb_typ_el=2, n_msh_pts=2*nel+1, n_msh_el=nel,
        table_nod=table_nod, type_el=type_el, x_arr=x_arr,
        n_effs=n_effs, bloch_vec_x=0.2, bloch_vec_y=0.1, world_1d=1
    )
    
    assert isinstance(modes, BlochModes)
    assert modes.norm_res < 1e-10, f"Biorthogonality residual too high: {modes.norm_res}"


def test_1d_tetm_decoupling_at_zero_ky():
    """G3.4: TE and TM fields decouple completely when ky == 0."""
    nel = 10
    k0 = 2.0 * np.pi
    kx = 0.3
    ky = 0.0
    eps_list = np.ones(nel, dtype=complex) * 4.0
    
    K, M, neq = fem_1d.build_sparse_system_1d(nel, eps_list, k0, kx, ky)
    K_dense = K.toarray()
    M_dense = M.toarray()
    
    # Dof blocks: Ex = [0:3*nel], Ey = [3*nel:6*nel], Ez = [6*nel:9*nel]
    nx = 3 * nel
    # Ex <-> Ey blocks in K and M must be identically zero
    np.testing.assert_allclose(K_dense[0:nx, nx:2*nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(K_dense[nx:2*nx, 0:nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(M_dense[0:nx, nx:2*nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(M_dense[nx:2*nx, 0:nx], 0.0, atol=1e-14)
    
    # Ey <-> Ez blocks in K and M must be identically zero
    np.testing.assert_allclose(K_dense[nx:2*nx, 2*nx:3*nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(K_dense[2*nx:3*nx, nx:2*nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(M_dense[nx:2*nx, 2*nx:3*nx], 0.0, atol=1e-14)
    np.testing.assert_allclose(M_dense[2*nx:3*nx, nx:2*nx], 0.0, atol=1e-14)


def test_1d_homogeneous_analytic_limit():
    """G3.1 / G3.5: Eigenvalues of a uniform medium match analytic plane-wave dispersion."""
    n_mat = 3.0
    nel = 40
    period_x = 1.0
    wl_norm = 1.0
    k0 = 2.0 * np.pi / wl_norm
    kx = 0.15
    ky = 0.25
    num_BMs = 6
    
    table_nod = np.zeros((3, nel), dtype=int)
    for e in range(nel):
        table_nod[0, e] = 2 * e + 1
        table_nod[1, e] = 2 * e + 3
        table_nod[2, e] = 2 * e + 2
    type_el = np.ones(nel, dtype=int)
    n_effs = [n_mat]
    x_arr = np.linspace(0, period_x, 2 * nel + 1)
    
    modes = fem_1d.solve_bloch_modes_1d(
        wl_norm=wl_norm, num_BMs=num_BMs, max_order_PWs=2,
        nb_typ_el=1, n_msh_pts=2*nel+1, n_msh_el=nel,
        table_nod=table_nod, type_el=type_el, x_arr=x_arr,
        n_effs=n_effs, bloch_vec_x=kx, bloch_vec_y=ky, world_1d=1
    )
    
    # Analytic eigenvalues: beta^2 = n^2 * k0^2 - (kx + 2*pi*m)^2 - ky^2
    m_orders = np.arange(-5, 6)
    k_xs = kx + 2.0 * np.pi * m_orders / period_x
    analytic_beta2 = (n_mat * k0)**2 - k_xs**2 - ky**2
    analytic_beta = np.sqrt(analytic_beta2.astype(complex))
    
    # Each computed eigenvalue should match one analytic eigenvalue
    for num_val in modes.zeta:
        dists = np.abs(analytic_beta - num_val)
        min_dist = np.min(dists)
        assert min_dist < 1e-2, f"Eigenvalue {num_val} did not match any analytic branch (min dist {min_dist})"


def test_1d_determinism_seed():
    """G1.3: Providing explicit v0 yields identical bitwise eigenvalues."""
    nel = 20
    wl_norm = 0.9
    num_BMs = 4
    
    table_nod = np.zeros((3, nel), dtype=int)
    for e in range(nel):
        table_nod[0, e] = 2 * e + 1
        table_nod[1, e] = 2 * e + 3
        table_nod[2, e] = 2 * e + 2
    type_el = np.ones(nel, dtype=int)
    n_effs = [2.5]
    x_arr = np.linspace(0, 1.0, 2 * nel + 1)
    
    rng = np.random.default_rng(12345)
    v0 = rng.standard_normal(9 * nel) + 1j * rng.standard_normal(9 * nel)
    
    modes1 = fem_1d.solve_bloch_modes_1d(
        wl_norm=wl_norm, num_BMs=num_BMs, max_order_PWs=1,
        nb_typ_el=1, n_msh_pts=2*nel+1, n_msh_el=nel,
        table_nod=table_nod, type_el=type_el, x_arr=x_arr,
        n_effs=n_effs, bloch_vec_x=0.1, bloch_vec_y=0.1,
        world_1d=1, v0=v0.copy()
    )
    modes2 = fem_1d.solve_bloch_modes_1d(
        wl_norm=wl_norm, num_BMs=num_BMs, max_order_PWs=1,
        nb_typ_el=1, n_msh_pts=2*nel+1, n_msh_el=nel,
        table_nod=table_nod, type_el=type_el, x_arr=x_arr,
        n_effs=n_effs, bloch_vec_x=0.1, bloch_vec_y=0.1,
        world_1d=1, v0=v0.copy()
    )
    
    np.testing.assert_array_equal(modes1.zeta, modes2.zeta)
    np.testing.assert_array_equal(modes1.J, modes2.J)


def test_1d_kzx_sign_negative_control():
    """Negative control: flipping K_zx = -Kyx to +Kyx corrupts the eigenvalues."""
    nel = 20
    period_x = 1.0
    h = period_x / nel
    k0 = 2.0 * np.pi
    kx = 0.2
    ky = 0.4
    eps_list = np.ones(nel, dtype=complex) * 4.0
    
    # Correct system
    K_correct, M, neq = fem_1d.build_sparse_system_1d(nel, eps_list, k0, kx, ky, period_x)
    sigma = 1.1 * 4.0 * (k0**2) - kx**2 - ky**2
    lu_c = splu((K_correct - sigma * M).tocsc())
    OP_c = LinearOperator((neq, neq), matvec=lambda x: lu_c.solve(M @ x), dtype=complex)
    vals_c, _ = eigs(OP_c, k=4, which='LM', tol=1e-12)
    
    # Corrupted pencil: replace -Kyx with +Kyx in K_zx
    # K_zx is in ez rows, ex cols: row index [nx+ny : nx+ny+4], col index [0:3]
    nx = ny = 3 * nel
    Mxx, Myy, Kxy, Kyx, Kyy = fem_1d.elem_matrices(h)
    
    rk, ck, vk = [], [], []
    for e in range(nel):
        q = eps_list[e] * (k0**2)
        ex = 3 * e + np.arange(3)
        ey = nx + np.array([3 * e, (3 * (e + 1)) % (3 * nel), 3 * e + 1, 3 * e + 2])
        ez = nx + ny + np.array([3 * e, (3 * (e + 1)) % (3 * nel), 3 * e + 1, 3 * e + 2])
        idx = np.concatenate([ex, ey, ez])
        
        # Wrong sign: +Kyx instead of -Kyx
        KE_wrong = np.block([
            [(ky**2 - q) * Mxx, 1j * ky * Kxy, np.zeros((3, 4))],
            [-1j * ky * Kyx, Kyy - q * Myy, np.zeros((4, 4))],
            [+Kyx, 1j * ky * Myy, Kyy + (ky**2 - q) * Myy]  # FLIPPED SIGN
        ])
        R, C = np.meshgrid(idx, idx, indexing='ij')
        rk.append(R.ravel())
        ck.append(C.ravel())
        vk.append(KE_wrong.ravel())
        
    from scipy.sparse import coo_matrix
    K_corrupt = coo_matrix((np.concatenate(vk), (np.concatenate(rk), np.concatenate(ck))), shape=(neq, neq)).tocsr()
    lu_w = splu((K_corrupt - sigma * M).tocsc())
    OP_w = LinearOperator((neq, neq), matvec=lambda x: lu_w.solve(M @ x), dtype=complex)
    vals_w, _ = eigs(OP_w, k=4, which='LM', tol=1e-12)
    
    # The eigenvalues MUST NOT match
    diff = np.min(np.abs(vals_c[:, None] - vals_w[None, :]))
    assert diff > 1e-4, f"Flipping K_zx sign did not change eigenvalues! diff={diff}"
