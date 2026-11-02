"""Pure Python 1D Finite Element Maxwell Solver.

Implements coupled wave equations using discontinuous P2 (Ex) + continuous periodic P3 (Ey, Ez) elements
using exact rational element matrices and SciPy sparse shift-invert (splu + eigs).

NO NGSolve or Netgen dependency (enforced by DoD-3).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, LinearOperator, eigs
# Inside the EMUstack package, import the package modules (one copy of each).
# Bare names are only for legacy scripts that put backend/ on sys.path.
if __package__:
    from .modes import BlochModes, FEMConvergenceError
    from .legacy_sort import z_indexx
else:
    from modes import BlochModes, FEMConvergenceError
    from legacy_sort import z_indexx


def elem_matrices(h):
    """Exact rational element matrices."""
    Mxx = (h / 30.0) * np.array([
        [4.0, -1.0, 2.0],
        [-1.0, 4.0, 2.0],
        [2.0, 2.0, 16.0]
    ])
    Myy = (h / 1680.0) * np.array([
        [128.0, 19.0, 99.0, -36.0],
        [19.0, 128.0, -36.0, 99.0],
        [99.0, -36.0, 648.0, -81.0],
        [-36.0, 99.0, -81.0, 648.0]
    ])
    Kxy = (1.0 / 120.0) * np.array([
        [-83.0, -7.0, 99.0, -9.0],
        [7.0, 83.0, 9.0, -99.0],
        [-44.0, 44.0, -108.0, 108.0]
    ])
    Kyx = Kxy.T
    Kyy = (1.0 / (40.0 * h)) * np.array([
        [148.0, -13.0, -189.0, 54.0],
        [-13.0, 148.0, 54.0, -189.0],
        [-189.0, 54.0, 432.0, -297.0],
        [54.0, -189.0, -297.0, 432.0]
    ])
    return Mxx, Myy, Kxy, Kyx, Kyy


def build_sparse_system_1d(nel, eps_list, k0, kx, ky, period_x=1.0):
    """Assemble the 1D Maxwell eigenvalue system (K, M) as SciPy CSR matrices.
    
    Total equations neq = 9 * nel.
    """
    h = period_x / nel
    neq = 9 * nel
    nx = ny = 3 * nel
    Mxx, Myy, Kxy, Kyx, Kyy = elem_matrices(h)
    
    rk, ck, vk = [], [], []
    rm, cm, vm = [], [], []
    ph = np.exp(1j * kx * period_x)
    
    Z34 = np.zeros((3, 4))
    Z43 = np.zeros((4, 3))
    Z44 = np.zeros((4, 4))
    
    for e in range(nel):
        q = eps_list[e] * (k0**2)
        ex = 3 * e + np.arange(3)
        ey = nx + np.array([3 * e, (3 * (e + 1)) % (3 * nel), 3 * e + 1, 3 * e + 2])
        ez = nx + ny + np.array([3 * e, (3 * (e + 1)) % (3 * nel), 3 * e + 1, 3 * e + 2])
        idx = np.concatenate([ex, ey, ez])
        
        p = np.ones(11, complex)
        if e == nel - 1:
            p[4] = ph
            p[8] = ph
            
        KE = np.block([
            [(ky**2 - q) * Mxx, 1j * ky * Kxy, Z34],
            [-1j * ky * Kyx, Kyy - q * Myy, Z44],
            [-Kyx, 1j * ky * Myy, Kyy + (ky**2 - q) * Myy]
        ])
        ME = np.block([
            [-Mxx, Z34, Kxy],
            [Z43, -Myy, 1j * ky * Myy],
            [Z43, Z44, Z44]
        ])
        
        # Apply periodic phase factor (Eq. 1395)
        # z_phase_fact = val_exp[jtest] * conj(val_exp[itrial])
        # In python: KE[i, j] *= p[j] * conj(p[i])
        KE = p.conj()[:, None] * KE * p[None, :]
        ME = p.conj()[:, None] * ME * p[None, :]
        
        R, C = np.meshgrid(idx, idx, indexing='ij')
        rk.append(R.ravel())
        ck.append(C.ravel())
        vk.append(KE.ravel())
        
        rm.append(R.ravel())
        cm.append(C.ravel())
        vm.append(ME.ravel())
        
    K = coo_matrix((np.concatenate(vk), (np.concatenate(rk), np.concatenate(ck))), shape=(neq, neq)).tocsr()
    M = coo_matrix((np.concatenate(vm), (np.concatenate(rm), np.concatenate(cm))), shape=(neq, neq)).tocsr()
    return K, M, neq


def vector_p2_exp_1d(xmin, xmax, alpha):
    """Analytic integral of P2 Lagrange basis * exp(i alpha x) on [xmin, xmax]."""
    h = xmax - xmin
    ah = alpha * h
    z1 = np.exp(1j * alpha * xmin)
    z2 = np.exp(1j * alpha * xmax)
    if abs(ah) > 1e-3:
        fact = alpha**3 * h**2
        v0 = (1j * (-4.0 + ah**2 - 3.0 * ah * 1j) * z1 + (ah + 4.0 * 1j) * z2) / fact
        v1 = ((ah - 4.0 * 1j) * z1 - 1j * (-4.0 + ah**2 + 3.0 * ah * 1j) * z2) / fact
        v2 = (-4.0 * (2.0 * 1j * (-z1 + z2) + ah * (z1 + z2))) / fact
        return np.array([v0, v1, v2])
    else:
        z3 = np.exp(1j * alpha * (xmax + xmin) / 2.0)
        fact = 120.0 / h
        return np.array([(20.0 + ah**2) * z1, (20.0 + ah**2) * z2, -2.0 * (-40.0 + ah**2) * z3]) / fact


def vector_p3_exp_1d(xmin, xmax, alpha):
    """Analytic integral of P3 Lagrange basis * exp(i alpha x) on [xmin, xmax]."""
    h = xmax - xmin
    ah = alpha * h
    z1 = np.exp(1j * alpha * xmin)
    z2 = np.exp(1j * alpha * xmax)
    if abs(ah) > 1e-3:
        fact = 2.0 * alpha**4 * h**3
        v0 = ((-54.0 + 11.0 * ah**2 - 36.0 * ah * 1j + 2.0 * ah**3 * 1j) * z1 - 2.0 * (-27.0 + ah**2 + 9.0 * ah * 1j) * z2) / fact
        v1 = (-2.0 * (-27.0 + ah**2 - 9.0 * ah * 1j) * z1 + (-54.0 + 11.0 * ah**2 + 36.0 * ah * 1j - 2.0 * ah**3 * 1j) * z2) / fact
        v2 = 9.0 * (-2.0 * (-9.0 + ah**2 - 5.0 * ah * 1j) * z1 + (-18.0 + ah**2 + 8.0 * ah * 1j) * z2) / fact
        v3 = -9.0 * ((18.0 - ah**2 + 8.0 * ah * 1j) * z1 + 2.0 * (-9.0 + ah**2 + 5.0 * ah * 1j) * z2) / fact
        return np.array([v0, v1, v2, v3])
    else:
        z3 = np.exp(1j * alpha * (xmax + 2.0 * xmin) / 3.0)
        z4 = np.exp(1j * alpha * (2.0 * xmax + xmin) / 3.0)
        fact = 240.0 / h
        v0 = (30.0 - ah**2 + 4.0 * ah * 1j) * z1 / fact
        v1 = (30.0 - ah**2 - 4.0 * ah * 1j) * z2 / fact
        v2 = (90.0 + ah**2 - 12.0 * ah * 1j) * z3 / fact
        v3 = (90.0 + ah**2 + 12.0 * ah * 1j) * z4 / fact
        return np.array([v0, v1, v2, v3])


def pw_ordering_1d(neq_PW, period_x, kx, ky, k0, max_order_PWs):
    """Sort plane wave orders in descending order of Re(beta_z**2)."""
    alpha_list = []
    beta_z_list = []
    for px in range(-max_order_PWs, max_order_PWs + 1):
        alpha = kx + 2.0 * np.pi * px / period_x
        alpha_list.append(alpha)
        beta_z_list.append(np.sqrt(k0**2 - alpha**2 - ky**2 + 0j))
        
    alpha_arr = np.array(alpha_list)
    beta_z_arr = np.array(beta_z_list)
    
    # legacy pw_ordering_1d.f: z_indexx, descending Re(beta_z**2), same order for ties
    order = z_indexx(beta_z_arr)
    inv_order = np.zeros_like(order)
    inv_order[order] = np.arange(len(order))
    return order, inv_order, alpha_arr


def calc_sol_local(vecs, nel, kx, period_x):
    """Extract local element solution arrays sol(11, nbm, nel) with phase applied."""
    nbm = vecs.shape[1]
    nx = ny = 3 * nel
    sol = np.zeros((11, nbm, nel), dtype=complex)
    ph = np.exp(1j * kx * period_x)
    
    for e in range(nel):
        sol[0:3, :, e] = vecs[3*e : 3*e+3, :]
        sol[3, :, e] = vecs[nx + 3*e, :]
        sol[4, :, e] = vecs[nx + (3*(e+1)) % (3*nel), :]
        sol[5, :, e] = vecs[nx + 3*e + 1, :]
        sol[6, :, e] = vecs[nx + 3*e + 2, :]
        
        sol[7, :, e] = vecs[nx + ny + 3*e, :]
        sol[8, :, e] = vecs[nx + ny + (3*(e+1)) % (3*nel), :]
        sol[9, :, e] = vecs[nx + ny + 3*e + 1, :]
        sol[10, :, e] = vecs[nx + ny + 3*e + 2, :]
        
        if e == nel - 1:
            sol[4, :, e] *= ph
            sol[8, :, e] *= ph
    return sol


def compute_overlaps_1d(sol_primal, sol_adjoint, nel, table_nod, x_arr, period_x, kx, ky, k0, max_order_PWs):
    """Compute J and J_dag overlap matrices using analytic Fourier integrals."""
    nbm = sol_primal.shape[1]
    neq_PW = 2 * max_order_PWs + 1
    
    order, inv_order, alpha_arr = pw_ordering_1d(neq_PW, period_x, kx, ky, k0, max_order_PWs)
    
    overlap_E = np.zeros((neq_PW, nbm), dtype=complex)
    overlap_K = np.zeros((neq_PW, nbm), dtype=complex)
    overlap_dag_E = np.zeros((nbm, neq_PW), dtype=complex)
    overlap_dag_K = np.zeros((nbm, neq_PW), dtype=complex)
    
    for e in range(nel):
        # 1-based table_nod to 0-based coordinates
        n1 = int(table_nod[0, e]) - 1
        n2 = int(table_nod[1, e]) - 1
        xmin = x_arr[n1]
        xmax = x_arr[n2]
        
        for idx_px, px in enumerate(range(-max_order_PWs, max_order_PWs + 1)):
            s2 = inv_order[idx_px]
            alpha = alpha_arr[idx_px]
            
            denom = np.sqrt(alpha**2 + ky**2)
            if denom != 0.0:
                norm = 1.0 / denom
                PlaneW_RK = np.array([alpha * norm, ky * norm])
                PlaneW_RE = np.array([ky * norm, -alpha * norm])
            else:
                PlaneW_RK = np.array([0.0, 1.0])
                PlaneW_RE = np.array([1.0, 0.0])
                
            # For J: exp(-i alpha x)
            p2_exp_J = vector_p2_exp_1d(xmin, xmax, -alpha)
            p3_exp_J = vector_p3_exp_1d(xmin, xmax, -alpha)
            
            # For J_dag: exp(+i alpha x)
            p2_exp_dag = vector_p2_exp_1d(xmin, xmax, alpha)
            p3_exp_dag = vector_p3_exp_1d(xmin, xmax, alpha)
            
            for m in range(nbm):
                phi_x_J = np.sum(sol_primal[0:3, m, e] * p2_exp_J)
                phi_y_J = np.sum(sol_primal[3:7, m, e] * p3_exp_J)
                
                overlap_K[s2, m] += phi_x_J * PlaneW_RK[0] + phi_y_J * PlaneW_RK[1]
                overlap_E[s2, m] += phi_x_J * PlaneW_RE[0] + phi_y_J * PlaneW_RE[1]
                
                phi_x_dag = np.sum(sol_adjoint[0:3, m, e] * p2_exp_dag)
                phi_y_dag = np.sum(sol_adjoint[3:7, m, e] * p3_exp_dag)
                
                overlap_dag_K[m, s2] += phi_x_dag * PlaneW_RK[0] + phi_y_dag * PlaneW_RK[1]
                overlap_dag_E[m, s2] += phi_x_dag * PlaneW_RE[0] + phi_y_dag * PlaneW_RE[1]
                
    J = np.vstack([overlap_E, overlap_K])
    J_dag = np.hstack([overlap_dag_E, overlap_dag_K])
    return J, J_dag, order, inv_order


def embed_1d_to_2d(J_1d, J_dag_1d, neq_PW_1d, neq_PW_2d, nbm, period_x, kx, ky, k0, max_order_PWs, inv_order_1d):
    """Embed 1D plane wave overlaps into 2D plane wave basis (pw_matrix_1d_to_2d)."""
    beta_z_2d = []
    pw_1d_to_2d = {}
    
    s = 0
    for px in range(-max_order_PWs, max_order_PWs + 1):
        for py in range(-max_order_PWs, max_order_PWs + 1):
            if (px / period_x)**2 + (py / period_x)**2 <= (max_order_PWs / period_x)**2:
                alpha = kx + 2.0 * np.pi * px / period_x
                beta = ky + 2.0 * np.pi * py / period_x
                beta_z_2d.append(np.sqrt(k0**2 - alpha**2 - beta**2 + 0j))
                if py == 0:
                    pw_1d_to_2d[px + max_order_PWs] = s
                s += 1
                
    beta_z_2d = np.array(beta_z_2d)
    order_2d = z_indexx(beta_z_2d)  # legacy pw_matrix_1d_to_2d.f
    inv_order_2d = np.zeros_like(order_2d)
    inv_order_2d[order_2d] = np.arange(len(order_2d))
    
    J_2d = np.zeros((2 * neq_PW_2d, nbm), dtype=complex)
    J_dag_2d = np.zeros((nbm, 2 * neq_PW_2d), dtype=complex)
    
    for ipw in range(neq_PW_1d):
        if ipw in pw_1d_to_2d:
            s_1d = inv_order_1d[ipw]
            s2 = pw_1d_to_2d[ipw]
            s_2d = inv_order_2d[s2]
            
            J_2d[s_2d, :] = J_1d[s_1d, :]
            J_2d[s_2d + neq_PW_2d, :] = J_1d[s_1d + neq_PW_1d, :]
            
            J_dag_2d[:, s_2d] = J_dag_1d[:, s_1d]
            J_dag_2d[:, s_2d + neq_PW_2d] = J_dag_1d[:, s_1d + neq_PW_1d]
            
    return J_2d, J_dag_2d


def solve_bloch_modes_1d(
    wl_norm, num_BMs, max_order_PWs, nb_typ_el,
    n_msh_pts, n_msh_el, table_nod, type_el,
    x_arr, itermax=30, FEM_debug=0,
    mesh_file=None, n_effs=None,
    bloch_vec_x=0.0, bloch_vec_y=0.0, shift=None,
    plotting_fields=False, plot_real=False,
    plot_imag=False, plot_abs=False,
    num_pw_per_pol=1, num_pw_per_pol_2d=1, world_1d=1,
    v0=None
):
    """Solve 1D Bloch modes (pure Python replacement for calc_modes_1d).
    
    Returns:
        BlochModes object (also unpacks as 6-tuple for legacy compatibility).
    """
    period_x = 1.0
    max_order_PWs = int(max_order_PWs)
    k0 = 2.0 * np.pi / wl_norm
    nel = n_msh_el
    
    # 1. Material indices per element
    n_effs_arr = np.asarray(n_effs)
    eps_list = np.zeros(nel, dtype=complex)
    for e in range(nel):
        tag = int(type_el[e]) - 1
        eps_list[e] = n_effs_arr[tag]**2
        
    n_max = np.max(np.real(n_effs_arr))
    if shift is None:
        sigma = 1.1 * max(n_max, 1.0)**2 * k0**2 - bloch_vec_x**2 - bloch_vec_y**2
    else:
        sigma = shift
        
    # 2. Primal assembly & solve
    K_pri, M_pri, neq = build_sparse_system_1d(nel, eps_list, k0, bloch_vec_x, bloch_vec_y, period_x)
    # Krylov size of legacy EMUstack (py_calc_modes_1d.f). Legacy ARPACK stops with
    # info=-3 when it exceeds the number of unknowns: the mesh is too coarse for num_BMs.
    ncv = 2 * num_BMs + num_BMs // 2 + 3
    if ncv > neq:
        raise FEMConvergenceError(
            f"1D mesh too coarse for num_BMs={num_BMs}: the eigensolver needs "
            f"{ncv} Krylov vectors but the mesh has only {neq} unknowns "
            f"(legacy EMUstack: ARPACK info=-3). Use a smaller lc_bkg or fewer modes.")
    
    if v0 is None:
        rng = np.random.default_rng(42)
        v0_use = rng.standard_normal(neq) + 1j * rng.standard_normal(neq)
    else:
        v0_use = v0
        
    lu_pri = splu((K_pri - sigma * M_pri).tocsc())
    OP_pri = LinearOperator((neq, neq), matvec=lambda x: lu_pri.solve(M_pri @ x), dtype=complex)
    vals_pri, vecs_pri = eigs(OP_pri, k=num_BMs, which='LM', ncv=ncv, maxiter=itermax, tol=0.0, v0=v0_use)
    
    zeta_pri = np.sqrt(1.0 / vals_pri + sigma)
    # Branch selection: |Im beta| < tol => Re > 0; else Im > 0
    for i in range(num_BMs):
        if abs(zeta_pri[i].imag) < 1e-8:  # legacy threshold
            if zeta_pri[i].real < 0:
                zeta_pri[i] = -zeta_pri[i]
        elif zeta_pri[i].imag < 0:
            zeta_pri[i] = -zeta_pri[i]
            
    order_pri = z_indexx(zeta_pri)  # legacy mode order
    zeta_pri = zeta_pri[order_pri]
    vecs_pri = vecs_pri[:, order_pri]
    
    # 3. Adjoint solve at (-kx, -ky)
    K_adj, M_adj, _ = build_sparse_system_1d(nel, eps_list, k0, -bloch_vec_x, -bloch_vec_y, period_x)
    lu_adj = splu((K_adj - sigma * M_adj).tocsc())
    OP_adj = LinearOperator((neq, neq), matvec=lambda x: lu_adj.solve(M_adj @ x), dtype=complex)
    vals_adj, vecs_adj = eigs(OP_adj, k=num_BMs, which='LM', ncv=ncv, maxiter=itermax, tol=0.0, v0=v0_use)
    
    zeta_adj = np.sqrt(1.0 / vals_adj + sigma)
    for i in range(num_BMs):
        if abs(zeta_adj[i].imag) < 1e-8:  # legacy threshold
            if zeta_adj[i].real < 0:
                zeta_adj[i] = -zeta_adj[i]
        elif zeta_adj[i].imag < 0:
            zeta_adj[i] = -zeta_adj[i]
            
    order_adj = z_indexx(zeta_adj)  # legacy mode order
    zeta_adj = zeta_adj[order_adj]
    vecs_adj = vecs_adj[:, order_adj]
    
    # 4. Extract local solutions
    sol_pri = calc_sol_local(vecs_pri, nel, bloch_vec_x, period_x)
    sol_adj = calc_sol_local(vecs_adj, nel, -bloch_vec_x, period_x)
    
    # 5. Bilinear pairing matrix B:
    # L(adj m, pri n) = (zeta_n / k0) * integral_0^1 [Ex_m Ex_n + Ey_m Ey_n - Ex_m dx(Ez_n) - i*ky*Ey_m Ez_n] dx
    h = period_x / nel
    Mxx, Myy, Kxy, Kyx, _ = elem_matrices(h)
    
    B_el = np.zeros((11, 11), dtype=complex)
    B_el[0:3, 0:3] = Mxx
    B_el[0:3, 7:11] = -Kxy
    B_el[3:7, 3:7] = Myy
    B_el[3:7, 7:11] = -1j * bloch_vec_y * Myy
    
    L = np.zeros((num_BMs, num_BMs), dtype=complex)
    for e in range(nel):
        for m in range(num_BMs):
            v_adj = sol_adj[:, m, e]
            for n in range(num_BMs):
                v_pri = sol_pri[:, n, e]
                L[m, n] += v_adj @ B_el @ v_pri
                
    for n in range(num_BMs):
        L[:, n] *= (zeta_pri[n] / k0)
        
    # 6. Conjugate-pair repair & normalization
    swaps = 0
    for j in range(num_BMs):
        if abs(L[j, j]) <= 1e-7:
            for j2 in range(j + 1, num_BMs):
                if abs(L[j2, j]) > 1e-7:
                    # Swap adjoint columns j and j2
                    vecs_adj[:, [j, j2]] = vecs_adj[:, [j2, j]]
                    sol_adj[:, [j, j2], :] = sol_adj[:, [j2, j], :]
                    zeta_adj[[j, j2]] = zeta_adj[[j2, j]]
                    L[[j, j2], :] = L[[j2, j], :]
                    swaps += 1
                    break
                    
    scales = np.zeros(num_BMs, dtype=complex)
    for i in range(num_BMs):
        z_norm = np.sqrt(L[i, i])
        if abs(z_norm) <= 1e-8:
            raise FEMConvergenceError(f"1D Mode {i} self-coupling norm near zero: {abs(z_norm)}")
        scale = 1.0 / z_norm
        scales[i] = scale
        vecs_pri[:, i] *= scale
        sol_pri[:, i, :] *= scale
        vecs_adj[:, i] *= scale
        sol_adj[:, i, :] *= scale
        
    L_normed = np.diag(scales) @ L @ np.diag(scales)
    norm_res = float(np.linalg.norm(L_normed - np.eye(num_BMs)))
    
    # Mode self-flux F (primal in both slots)
    F = np.zeros((num_BMs, num_BMs), dtype=complex)
    for e in range(nel):
        for m in range(num_BMs):
            v_m = sol_pri[:, m, e]
            for n in range(num_BMs):
                v_n = sol_pri[:, n, e]
                F[m, n] += v_m @ B_el @ v_n
    for n in range(num_BMs):
        F[:, n] *= (zeta_pri[n] / k0)
    
    # 7. sol_P2 reconstruction (3, 3, nval, nel)
    P3_mid_mode_value = np.array([-1.0/16.0, -1.0/16.0, 9.0/16.0, 9.0/16.0])
    sol_P2 = np.zeros((3, 3, num_BMs, nel), dtype=complex)
    for e in range(nel):
        for m in range(num_BMs):
            # Ex: left, right, mid
            sol_P2[0, :, m, e] = sol_pri[0:3, m, e]
            # Ey: left, right
            sol_P2[1, 0, m, e] = sol_pri[3, m, e]
            sol_P2[1, 1, m, e] = sol_pri[4, m, e]
            sol_P2[1, 2, m, e] = np.sum(sol_pri[3:7, m, e] * P3_mid_mode_value)
            # Ez: left, right
            sol_P2[2, 0, m, e] = sol_pri[7, m, e]
            sol_P2[2, 1, m, e] = sol_pri[8, m, e]
            sol_P2[2, 2, m, e] = np.sum(sol_pri[7:11, m, e] * P3_mid_mode_value)
            
    # 8. Overlaps J and J_dag
    J_1d, J_dag_1d, order_1d, inv_order_1d = compute_overlaps_1d(
        sol_pri, sol_adj, nel, table_nod, x_arr, period_x, bloch_vec_x, bloch_vec_y, k0, max_order_PWs
    )
    
    J_2d, J_dag_2d = None, None
    if world_1d == 0:
        J_2d, J_dag_2d = embed_1d_to_2d(
            J_1d, J_dag_1d, num_pw_per_pol, num_pw_per_pol_2d, num_BMs, period_x,
            bloch_vec_x, bloch_vec_y, k0, max_order_PWs, inv_order_1d
        )
        
    modes = BlochModes(
        zeta=zeta_pri,
        J=J_2d if world_1d == 0 else J_1d,
        J_dag=J_dag_2d if world_1d == 0 else J_dag_1d,
        mode_flux=F,
        norm_res=norm_res,
        pairs=np.arange(num_BMs),
        cluster=np.zeros(num_BMs, dtype=int),
        solver_info={"swaps": swaps, "ncv": ncv, "neq": neq},
        sol1=sol_P2,
    )
    modes._J_1d = J_1d
    modes._J_dag_1d = J_dag_1d
    modes._J_2d = J_2d
    modes._J_dag_2d = J_dag_2d
    return modes
