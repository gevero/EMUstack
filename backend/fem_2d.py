"""2D Finite Element Maxwell Bloch Solver using NGSolve (PORTING_PLAN §6).

Solves the coupled de Rham pencil on Periodic(HCurl(p) * H1(p+1)):
K u = lambda M u with lambda = zeta^2.

Bloch formulation: quasi-periodic space via ngsolve.Periodic (Option A in §6.1),
assembled with plain curl/grad.
"""
import os
import time
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import scipy.sparse as sp

import netgen.geom2d as g2
import ngsolve as ngs
from ngsolve import Mesh, HCurl, H1, GridFunction, BilinearForm

from modes import BlochModes, FEMConvergenceError
import meshing
import paths


def assemble_pencil_2d(
    mesh: ngs.Mesh,
    dy: float,
    eps_cf: ngs.CoefficientFunction,
    k0: float,
    kx: float,
    ky: float,
    order: int = 2,
    period_x: float = 1.0
) -> Tuple[BilinearForm, BilinearForm, ngs.comp.FESpace, ngs.comp.FESpace]:
    """Assemble the Maxwell weak form (K, M) on the quasi-periodic product space (§6.2).
    
    Option A (§6.1): quasi-periodic space via ngsolve.Periodic, plain curl/grad.
    """
    fes = HCurl(mesh, order=order, complex=True) * H1(mesh, order=order + 1, complex=True)
    phase = [np.exp(+1j * kx * period_x), np.exp(+1j * ky * dy)]
    space = ngs.Periodic(fes, phase=phase)
    assert space.FreeDofs().NumSet() < fes.ndof, "Periodic() left the space unconstrained"

    (u, z), (v, w) = space.TnT()

    K = BilinearForm(space, symmetric=False)
    K += (
        ngs.curl(u) * ngs.Conj(ngs.curl(v))
        - k0**2 * eps_cf * (u[0] * ngs.Conj(v[0]) + u[1] * ngs.Conj(v[1]))
        + u[0] * ngs.Conj(ngs.grad(w)[0]) + u[1] * ngs.Conj(ngs.grad(w)[1])
        - (ngs.grad(z)[0] * ngs.Conj(ngs.grad(w)[0]) + ngs.grad(z)[1] * ngs.Conj(ngs.grad(w)[1]))
        + k0**2 * eps_cf * z * ngs.Conj(w)
    ) * ngs.dx

    M = BilinearForm(space, symmetric=False)
    M += (
        -(u[0] * ngs.Conj(v[0]) + u[1] * ngs.Conj(v[1]))
        + (ngs.grad(z)[0] * ngs.Conj(v[0]) + ngs.grad(z)[1] * ngs.Conj(v[1]))
    ) * ngs.dx

    K.Assemble()
    M.Assemble()
    return K, M, space, fes


def assemble_bilinear_B(
    mesh: ngs.Mesh,
    space_pri: Optional[ngs.comp.FESpace] = None,
    space_adj: Optional[ngs.comp.FESpace] = None,
    order: int = 2
) -> sp.csr_matrix:
    """Assemble un-conjugated bilinear form B on trialspace=space_pri, testspace=space_adj (§6.4, Eq. 26).
    
    B = int_Omega mu^-1 (u - grad z) . v dA
    """
    if space_pri is not None and space_adj is not None:
        (u, z) = space_pri.TrialFunction()
        (v, w) = space_adj.TestFunction()
        B_form = BilinearForm(trialspace=space_pri, testspace=space_adj)
    else:
        fes = HCurl(mesh, order=order, complex=True) * H1(mesh, order=order + 1, complex=True)
        (u, z), (v, w) = fes.TnT()
        B_form = BilinearForm(fes, symmetric=False)

    # NGSolve conjugates test functions in complex mode; pass ngs.Conj(v) to keep it bilinear
    B_form += (
        (u[0] - ngs.grad(z)[0]) * ngs.Conj(v[0])
        + (u[1] - ngs.grad(z)[1]) * ngs.Conj(v[1])
    ) * ngs.dx
    B_form.Assemble()

    vals, col, row = B_form.mat.CSR()
    return sp.csr_matrix((np.array(vals), np.array(col), np.array(row)), shape=B_form.mat.shape)


def solve_pencil_modes(
    K: BilinearForm,
    M: BilinearForm,
    space: ngs.comp.FESpace,
    sigma: complex,
    numeig: int,
    k0: float,
    max_order_PWs: int,
    n_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve K u = lambda M u with ArnoldiSolver, filter infinite & spurious modes, and sort."""
    gfs = [GridFunction(space) for _ in range(numeig)]
    lam = np.array(ngs.ArnoldiSolver(K.mat, M.mat, space.FreeDofs(), [g.vec for g in gfs], shift=sigma))

    K_vals, K_col, K_row = K.mat.CSR()
    K_csr = sp.csr_matrix((np.array(K_vals), np.array(K_col), np.array(K_row)), shape=K.mat.shape)
    M_vals, M_col, M_row = M.mat.CSR()
    M_csr = sp.csr_matrix((np.array(M_vals), np.array(M_col), np.array(M_row)), shape=M.mat.shape)

    # Infinite eigenvalue cap (§6.3 step 1)
    lam_cap = 1.5 * (n_max * k0)**2 + 200.0 + (2.0 * np.pi * max(max_order_PWs, 2))**2

    valid_zeta = []
    valid_coeffs = []
    valid_res = []

    for i in range(len(lam)):
        l = lam[i]
        if np.isnan(l) or np.isinf(l) or abs(l) > lam_cap:
            continue

        c = np.array(gfs[i].vec.data)
        Kc = K_csr @ c
        Mc = M_csr @ c
        norm_kc = np.linalg.norm(Kc)
        norm_mc = np.linalg.norm(Mc)
        denom = norm_kc + norm_mc
        if denom < 1e-14:
            continue
        res = np.linalg.norm(Kc - l * Mc) / denom

        if res > 1e-3:  # Spurious mode filter (§6.3 step 2)
            continue

        z = np.sqrt(l)
        # Branch selection (§6.3 step 3, Eq. 22)
        if abs(z.imag) <= 1e-8 * k0:
            if z.real < 0:
                z = -z
                c = -c
        elif z.imag < 0:
            z = -z
            c = -c

        valid_zeta.append(z)
        valid_coeffs.append(c)
        valid_res.append(res)

    if len(valid_zeta) == 0:
        return np.array([]), np.array([]), np.array([])

    valid_zeta = np.array(valid_zeta)
    valid_coeffs = np.column_stack(valid_coeffs)
    valid_res = np.array(valid_res)

    # Sorting (§6.3 step 4 / z_indexx.f): descending Re(zeta^2) = Re^2 - Im^2
    sort_order = np.argsort(-np.real(valid_zeta**2), kind='stable')

    return valid_zeta[sort_order], valid_coeffs[:, sort_order], valid_res[sort_order]


def compute_overlaps_2d(
    mesh: ngs.Mesh,
    Phi_pri: np.ndarray,
    Phi_adj: np.ndarray,
    k0: float,
    kx: float,
    ky: float,
    dy: float,
    max_order_PWs: int,
    order: int = 2,
    period_x: float = 1.0,
    space_pri: Optional[ngs.comp.FESpace] = None,
    space_adj: Optional[ngs.comp.FESpace] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute projection overlaps J (divided by dy) and J_dag (un-normalized) via LinearForm assembly on periodic spaces (§6.5)."""
    if space_pri is None or space_adj is None:
        fes = HCurl(mesh, order=order, complex=True) * H1(mesh, order=order + 1, complex=True)
        space_pri = ngs.Periodic(fes, phase=[np.exp(+1j * kx * period_x), np.exp(+1j * ky * dy)])
        space_adj = ngs.Periodic(fes, phase=[np.exp(-1j * kx * period_x), np.exp(-1j * ky * dy)])

    px_range = np.arange(-max_order_PWs, max_order_PWs + 1)
    py_mesh, px_mesh = np.meshgrid(px_range, px_range)
    mask = (px_mesh / period_x)**2 + (py_mesh / dy)**2 <= (max_order_PWs / max(period_x, dy))**2

    pxs = px_mesh[mask]
    pys = py_mesh[mask]
    alphas = kx + 2.0 * np.pi * pxs / period_x
    betas = ky + 2.0 * np.pi * pys / dy

    kz_unsrt = np.sqrt(k0**2 - alphas**2 - betas**2 + 0j)
    # Air-reference sort order
    sort_pw = np.argsort(-kz_unsrt.real + kz_unsrt.imag, kind='stable')
    alphas = alphas[sort_pw]
    betas = betas[sort_pw]
    Qs = np.sqrt(alphas**2 + betas**2)

    npw = len(alphas)
    nbm = Phi_pri.shape[1]

    (v_pri, w_pri) = space_pri.TestFunction()
    (v_adj, w_adj) = space_adj.TestFunction()

    LF_pri_E, LF_pri_K = [], []
    LF_adj_E, LF_adj_K = [], []

    for s in range(npw):
        alpha, beta, Q = alphas[s], betas[s], Qs[s]
        if Q > 1e-12:
            pe_x, pe_y = beta / Q, -alpha / Q
            pk_x, pk_y = alpha / Q, beta / Q
        else:
            pe_x, pe_y = 1.0, 0.0
            pk_x, pk_y = 0.0, 1.0

        ph_minus = ngs.exp(-1j * (alpha * ngs.x + beta * ngs.y))
        ph_plus  = ngs.exp(+1j * (alpha * ngs.x + beta * ngs.y))

        # Primal: plane wave has exp(-i (alpha x + beta y))
        lf_pE = ngs.LinearForm(space_pri)
        lf_pE += (pe_x * ph_minus * v_pri[0] + pe_y * ph_minus * v_pri[1]) * ngs.dx
        lf_pE.Assemble()
        LF_pri_E.append(np.array(lf_pE.vec.data))

        lf_pK = ngs.LinearForm(space_pri)
        lf_pK += (pk_x * ph_minus * v_pri[0] + pk_y * ph_minus * v_pri[1]) * ngs.dx
        lf_pK.Assemble()
        LF_pri_K.append(np.array(lf_pK.vec.data))

        # Adjoint: plane wave has exp(+i (alpha x + beta y))
        lf_aE = ngs.LinearForm(space_adj)
        lf_aE += (pe_x * ph_plus * v_adj[0] + pe_y * ph_plus * v_adj[1]) * ngs.dx
        lf_aE.Assemble()
        LF_adj_E.append(np.array(lf_aE.vec.data))

        lf_aK = ngs.LinearForm(space_adj)
        lf_aK += (pk_x * ph_plus * v_adj[0] + pk_y * ph_plus * v_adj[1]) * ngs.dx
        lf_aK.Assemble()
        LF_adj_K.append(np.array(lf_aK.vec.data))

    mat_LF_pri = np.vstack([LF_pri_E, LF_pri_K])
    mat_LF_adj = np.vstack([LF_adj_E, LF_adj_K])

    J = (mat_LF_pri @ Phi_pri) / dy
    J_dag = Phi_adj.T @ mat_LF_adj.T

    return J, J_dag


def solve_bloch_modes_2d(
    wl_norm: float,
    num_BMs: int,
    max_order_PWs: int,
    structure_or_mesh: Any,
    n_effs: Any,
    k_pll: Tuple[float, float] = (0.0, 0.0),
    shift: Optional[float] = None,
    order: int = 2,
    plotting_fields: bool = False,
    num_pw_per_pol: Optional[int] = None,
    **kwargs
) -> BlochModes:
    """Solve 2D Bloch modes via NGSolve (PORTING_PLAN §6)."""
    t_start = time.perf_counter()
    k0 = 2.0 * np.pi / wl_norm
    kx, ky = float(k_pll[0]), float(k_pll[1])

    # 1. Resolve mesh & geometry
    structure = None
    if isinstance(structure_or_mesh, ngs.Mesh):
        mesh = structure_or_mesh
        period_x = kwargs.get("period_x", 1.0)
        dy = kwargs.get("dy", 1.0)
    else:
        structure = structure_or_mesh
        period_x = 1.0
        dy = float(structure.period_y) / float(structure.period)
        if hasattr(structure, "mesh") and structure.mesh is not None:
            mesh = structure.mesh
            if not hasattr(structure, "table_nod") or structure.table_nod is None:
                structure.n_msh_pts = mesh.nv
                structure.n_msh_el = mesh.ne
                structure.x_arr = np.array([v.point[:2] for v in mesh.vertices]).T
                t_nod = np.zeros((6, mesh.ne), dtype=int)
                t_el = np.zeros(mesh.ne, dtype=int)
                for i_el, el in enumerate(mesh.Elements(ngs.VOL)):
                    verts = [v.nr + 1 for v in el.vertices]
                    t_nod[:len(verts), i_el] = verts
                    t_el[i_el] = 1 if el.mat == 'bg' else (int(el.mat[3:]) if el.mat.startswith('inc') and el.mat[3:].isdigit() else 2)
                structure.table_nod = t_nod
                structure.type_el = t_el
        elif getattr(structure, "mesh_file", "").endswith(".mail"):
            mail_name = structure.mesh_file
            candidate_paths = [
                os.path.join(paths.msh_path, mail_name),
                os.path.join(paths.template_path, mail_name),
                os.path.join(paths.data_path, "meshes", mail_name)
            ]
            mail_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    mail_path = p
                    break
            if mail_path is None:
                raise FileNotFoundError(f"Could not locate mesh file: {mail_name}")
            mesh, table_nod, type_el, pts, npt, nel = meshing.read_mail_file(mail_path)
            structure.mesh = mesh
            structure.table_nod = table_nod
            structure.type_el = type_el
            structure.x_arr = pts
            structure.n_msh_pts = npt
            structure.n_msh_el = nel
        else:
            # Build native Netgen unit cell
            inc_shape = getattr(structure, "inc_shape", "circle")
            d1 = getattr(structure, "diameter1", 0.0) / float(structure.period)
            d2 = getattr(structure, "diameter2", 0.0) / float(structure.period)
            posx = getattr(structure, "posx", 0.0) / float(structure.period)
            posy = getattr(structure, "posy", 0.0) / float(structure.period)
            center = (0.5 + posx, dy / 2.0 + posy)

            inclusions = []
            if inc_shape in ("circle", "ellipse") and d1 > 0:
                inclusions.append({"kind": inc_shape, "c": center, "r": d1 / 2.0, "tag": 2})
            elif inc_shape in ("square", "rectangle") and d1 > 0:
                inclusions.append({"kind": inc_shape, "c": center, "s": d1, "tag": 2})
            elif inc_shape == "ring" and d1 > 0:
                inclusions.append({"kind": "ring", "c": center, "r_out": d1 / 2.0, "r_in": d2 / 2.0, "tag": 2})
            elif inc_shape == "dimer" and d1 > 0:
                gap = getattr(structure, "gap", 0.0) / float(structure.period)
                inclusions.append({"kind": "dimer", "c": center, "r": d1 / 2.0, "gap": gap, "tag": 2})

            refine = meshing.Refine(
                bg=getattr(structure, "lc", 0.1),
                inc=getattr(structure, "lc", 0.1) * getattr(structure, "lc2", 1.0)
            )
            mesh = meshing.unit_cell(dy, inclusions, refine=refine, curve_order=order, period_x=period_x)
            structure.mesh = mesh
            structure.n_msh_pts = mesh.nv
            structure.n_msh_el = mesh.ne
            structure.x_arr = np.array([v.point[:2] for v in mesh.vertices]).T
            t_nod = np.zeros((6, mesh.ne), dtype=int)
            t_el = np.zeros(mesh.ne, dtype=int)
            for i_el, el in enumerate(mesh.Elements(ngs.VOL)):
                verts = [v.nr + 1 for v in el.vertices]
                t_nod[:len(verts), i_el] = verts
                t_el[i_el] = 1 if el.mat == 'bg' else (int(el.mat[3:]) if el.mat.startswith('inc') and el.mat[3:].isdigit() else 2)
            structure.table_nod = t_nod
            structure.type_el = t_el

    # 2. Material mapping
    n_effs_list = list(n_effs) if hasattr(n_effs, "__iter__") else [n_effs]
    n_bg = n_effs_list[0]
    eps_dict = {
        "bg": ngs.CF(complex(n_bg**2)),
        "default": ngs.CF(complex(n_bg**2))
    }
    for tag_idx, n_val in enumerate(n_effs_list[1:], start=2):
        cf_val = ngs.CF(complex(n_val**2))
        eps_dict[f"inc{tag_idx}"] = cf_val
        eps_dict[f"inc{tag_idx}_in"] = cf_val
        eps_dict[f"inc{tag_idx}_out"] = cf_val
    if len(n_effs_list) > 1:
        eps_dict["inc"] = ngs.CF(complex(n_effs_list[1]**2))

    eps_cf = mesh.MaterialCF(eps_dict)
    n_max = float(np.max(np.abs(n_effs_list)))
    if shift is None:
        sigma = 1.1 * max(n_max, 1.0)**2 * k0**2 - kx**2 - ky**2
    else:
        sigma = shift

    # 3. Assemble pencils
    t_asm = time.perf_counter()
    K_pri, M_pri, space_pri, fes = assemble_pencil_2d(mesh, dy, eps_cf, k0, kx, ky, order=order, period_x=period_x)
    K_adj, M_adj, space_adj, _ = assemble_pencil_2d(mesh, dy, eps_cf, k0, -kx, -ky, order=order, period_x=period_x)
    B_csr = assemble_bilinear_B(mesh, space_pri, space_adj, order=order)
    time_asm = time.perf_counter() - t_asm

    # 4. Arnoldi solve with padding and completeness check
    t_arnoldi = time.perf_counter()
    numeig = max(num_BMs * 3, num_BMs + 40)

    zeta_pri, Phi_pri, res_pri = solve_pencil_modes(
        K_pri, M_pri, space_pri, sigma, numeig, k0, max_order_PWs, n_max
    )
    zeta_adj, Phi_adj, res_adj = solve_pencil_modes(
        K_adj, M_adj, space_adj, sigma, numeig, k0, max_order_PWs, n_max
    )

    if len(zeta_pri) < num_BMs or len(zeta_adj) < num_BMs:
        # Retry once with expanded numeig
        numeig_retry = int(numeig * 1.5)
        zeta_pri, Phi_pri, res_pri = solve_pencil_modes(
            K_pri, M_pri, space_pri, sigma, numeig_retry, k0, max_order_PWs, n_max
        )
        zeta_adj, Phi_adj, res_adj = solve_pencil_modes(
            K_adj, M_adj, space_adj, sigma, numeig_retry, k0, max_order_PWs, n_max
        )
        if len(zeta_pri) < num_BMs or len(zeta_adj) < num_BMs:
            raise FEMConvergenceError(
                f"Insufficient converged modes: requested {num_BMs}, "
                f"obtained primal {len(zeta_pri)}, adjoint {len(zeta_adj)}"
            )

    # Nearest-neighbor mode pairing between primal and adjoint eigenspectra (§3.3)
    paired_adj = []
    avail = set(range(len(zeta_adj)))
    for n in range(num_BMs):
        m_best = min(avail, key=lambda m: abs(zeta_pri[n] - zeta_adj[m]))
        paired_adj.append(m_best)
        avail.remove(m_best)

    zeta_pri = zeta_pri[:num_BMs]
    Phi_pri = Phi_pri[:, :num_BMs]
    zeta_adj = zeta_adj[paired_adj]
    Phi_adj = Phi_adj[:, paired_adj]
    time_arnoldi = time.perf_counter() - t_arnoldi

    # 5. Biorthogonal pairing and normalization (§3.3, §6.4)
    # L_mn = (zeta_n / k0) * Phi_adj[m]^T @ B @ Phi_pri[n]
    L = (1.0 / k0) * (Phi_adj.T @ B_csr @ Phi_pri) @ np.diag(zeta_pri)

    # Conjugate-pair repair (§3.3, §7.6 step 6)
    swaps = 0
    for j in range(num_BMs):
        if abs(L[j, j]) <= 1e-7:
            for j2 in range(j + 1, num_BMs):
                if abs(L[j2, j]) > 1e-7:
                    Phi_adj[:, [j, j2]] = Phi_adj[:, [j2, j]]
                    zeta_adj[[j, j2]] = zeta_adj[[j2, j]]
                    swaps += 1
                    L = (1.0 / k0) * (Phi_adj.T @ B_csr @ Phi_pri) @ np.diag(zeta_pri)
                    break

    # Algebraic biorthogonal repair Phi_adj <- Phi_adj @ L^-T (§3.3, G1.4)
    try:
        Phi_adj = Phi_adj @ np.linalg.inv(L).T
        L = (1.0 / k0) * (Phi_adj.T @ B_csr @ Phi_pri) @ np.diag(zeta_pri)
    except Exception:
        diag_L = np.diag(L)
        for m in range(num_BMs):
            if abs(diag_L[m]) > 1e-12:
                Phi_adj[:, m] /= diag_L[m]
        L = (1.0 / k0) * (Phi_adj.T @ B_csr @ Phi_pri) @ np.diag(zeta_pri)

    norm_res = float(np.linalg.norm(L - np.eye(num_BMs)))

    # 6. Mode self-flux matrix F (primal in both slots, §3.6)
    bform_pri = ngs.BilinearForm(space_pri, symmetric=False)
    (u_p, z_p) = space_pri.TrialFunction()
    (v_p, w_p) = space_pri.TestFunction()
    bform_pri += (
        (u_p[0] - ngs.grad(z_p)[0]) * ngs.Conj(v_p[0])
        + (u_p[1] - ngs.grad(z_p)[1]) * ngs.Conj(v_p[1])
    ) * ngs.dx
    bform_pri.Assemble()
    v_f, c_f, r_f = bform_pri.mat.CSR()
    B_pri_csr = sp.csr_matrix((np.array(v_f), np.array(c_f), np.array(r_f)), shape=bform_pri.mat.shape)
    mode_flux = -(1.0 / k0) * (Phi_pri.T @ B_pri_csr @ Phi_pri) @ np.diag(zeta_pri)

    # 7. Overlaps J and J_dag
    t_ov = time.perf_counter()
    J, J_dag = compute_overlaps_2d(
        mesh, Phi_pri, Phi_adj, k0, kx, ky, dy, max_order_PWs, order=order, period_x=period_x,
        space_pri=space_pri, space_adj=space_adj
    )
    time_ov = time.perf_counter() - t_ov

    # 8. Mode polarization / energy fractions (§9.4)
    mode_pol = np.zeros((4, num_BMs), dtype=float)
    fes_E = HCurl(mesh, order=order, complex=True)
    for n in range(num_BMs):
        gf_n = GridFunction(fes_E)
        gf_n.vec.data = Phi_pri[:fes_E.ndof, n]
        dot_E = gf_n[0] * ngs.Conj(gf_n[0]) + gf_n[1] * ngs.Conj(gf_n[1])
        try:
            e_tot = ngs.Integrate(eps_cf * dot_E, mesh)
            e_inc = ngs.Integrate(eps_cf * dot_E, mesh, definedon=mesh.Materials("inc.*"))
            mode_pol[3, n] = float(np.real(e_inc / max(e_tot, 1e-15)))
        except Exception:
            pass

    # 9. Sol1 tensor reconstruction if requested (§9.1)
    sol1 = None
    if plotting_fields and structure is not None and hasattr(structure, "table_nod") and structure.table_nod is not None:
        nel = structure.n_msh_el
        sol1 = np.zeros((6, 13, num_BMs, nel), dtype=complex)
        
        # Precompute Netgen mesh points for unique mesh vertices
        pts = []
        for p in structure.x_arr.T:
            px = float(p[0])
            py = float(p[1])
            try:
                pts.append(mesh(px, py))
            except Exception:
                px_c = np.clip(px, 1e-7, period_x - 1e-7)
                py_c = np.clip(py, 1e-7, dy - 1e-7)
                try:
                    pts.append(mesh(px_c, py_c))
                except Exception:
                    pts.append(None)

        for n in range(num_BMs):
            gf_n = GridFunction(fes)
            gf_n.vec.data = Phi_pri[:, n]
            u_n, z_n = gf_n.components
            grad_z = ngs.grad(z_n)
            curl_u = ngs.curl(u_n)

            v_vals = []
            for pt in pts:
                if pt is None:
                    v_vals.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    continue
                try:
                    val_u = u_n(pt)
                    val_z = z_n(pt)
                    val_gz = grad_z(pt)
                    val_cu = curl_u(pt)

                    ux = complex(val_u[0])
                    uy = complex(val_u[1])
                    zv = complex(val_z)
                    gzx = complex(val_gz[0])
                    gzy = complex(val_gz[1])
                    cu_val = complex(val_cu[0]) if isinstance(val_cu, (tuple, list)) else complex(val_cu)

                    hx_term = -uy - gzy
                    hy_term = ux - gzx
                    hz_term = cu_val
                    v_vals.append((ux, uy, zv, hx_term, hy_term, hz_term))
                except Exception:
                    v_vals.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

            for iel in range(nel):
                for inod in range(min(6, structure.table_nod.shape[0])):
                    g_idx = structure.table_nod[inod, iel] - 1
                    if 0 <= g_idx < len(v_vals):
                        val = v_vals[g_idx]
                        sol1[0, inod, n, iel] = val[0]
                        sol1[1, inod, n, iel] = val[1]
                        sol1[2, inod, n, iel] = val[2]
                        sol1[3, inod, n, iel] = val[3]
                        sol1[4, inod, n, iel] = val[4]
                        sol1[5, inod, n, iel] = val[5]

    # Degenerate cluster labelling (§6.3 step 5)
    cluster = np.zeros(num_BMs, dtype=int)
    c_id = 0
    for i in range(num_BMs):
        if i > 0 and abs(zeta_pri[i] - zeta_pri[i - 1]) > 1e-5 * k0:
            c_id += 1
        cluster[i] = c_id

    solver_info = {
        "ndof": fes.ndof,
        "numeig_requested": num_BMs,
        "norm_res": norm_res,
        "swaps": swaps,
        "time_assembly": time_asm,
        "time_arnoldi": time_arnoldi,
        "time_overlaps": time_ov,
        "total_time": time.perf_counter() - t_start
    }

    return BlochModes(
        zeta=zeta_pri,
        J=J,
        J_dag=J_dag,
        mode_flux=mode_flux,
        norm_res=norm_res,
        pairs=np.arange(num_BMs),
        cluster=cluster,
        solver_info=solver_info,
        sol1=sol1,
        mode_pol=mode_pol
    )
