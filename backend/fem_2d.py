"""2D Finite Element Maxwell Bloch Solver using NGSolve.

Solves the coupled Maxwell eigenvalue equation on Periodic(HCurl(p) * H1(p+1)):
K u = lambda M u with lambda = zeta^2.

Bloch formulation: quasi-periodic space via ngsolve.Periodic,
assembled with mixed vector-scalar finite elements.
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
from legacy_sort import z_indexx
import paths


def assemble_system_2d(
    mesh: ngs.Mesh,
    dy: float,
    eps_cf: ngs.CoefficientFunction,
    k0: float,
    kx: float,
    ky: float,
    order: int = 2,
    period_x: float = 1.0
) -> Tuple[BilinearForm, BilinearForm, ngs.comp.FESpace, ngs.comp.FESpace]:
    """Assemble the Maxwell weak form (K, M) on the quasi-periodic product space.
    
    Quasi-periodic space via ngsolve.Periodic, plain curl/grad.
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
    """Assemble the un-conjugated pairing B between adjoint (rows) and primal (columns) modes.

    B = int_Omega (u - grad z) . v dA, with u, z primal and v adjoint.

    On a quasi-periodic space NGSolve folds the boundary copy of a test dof into its
    partner with the conjugate of the space's phase (right for sesquilinear forms).
    The adjoint field has phase conj(p), so its rows must be assembled on a space with
    phase p: the primal space. space_adj is accepted for the call signature only.
    """
    if space_pri is not None:
        (u, z) = space_pri.TrialFunction()
        (v, w) = space_pri.TestFunction()
        B_form = BilinearForm(trialspace=space_pri, testspace=space_pri)
    else:
        fes = HCurl(mesh, order=order, complex=True) * H1(mesh, order=order + 1, complex=True)
        (u, z), (v, w) = fes.TnT()
        B_form = BilinearForm(fes, symmetric=False)

    # basis functions are real, so this is the plain bilinear product
    B_form += (
        (u[0] - ngs.grad(z)[0]) * v[0]
        + (u[1] - ngs.grad(z)[1]) * v[1]
    ) * ngs.dx
    B_form.Assemble()

    vals, col, row = B_form.mat.CSR()
    return sp.csr_matrix((np.array(vals), np.array(col), np.array(row)), shape=B_form.mat.shape)


# 2D eigensolver:
# 'arpack'  - SciPy ARPACK in shift-invert mode with legacy's settings (py_calc_modes.f /
#             valpr_64: the num_BMs eigenvalues nearest the shift, Krylov size
#             2*num_BMs + num_BMs/2 + 3, tol = 0). Default.
# 'ngsolve' - NGSolve ArnoldiSolver (no restarts; can leave some modes unconverged).
eigensolver = 'arpack'


def _scipy_csr(form) -> sp.csr_matrix:
    vals, col, row = form.mat.CSR()
    return sp.csr_matrix((np.array(vals), np.array(col), np.array(row)), shape=form.mat.shape)


def solve_system_modes_arpack(
    K: BilinearForm,
    M: BilinearForm,
    space: ngs.comp.FESpace,
    sigma: complex,
    num_BMs: int,
    maxiter: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The num_BMs eigenpairs of K u = lambda M u nearest sigma, by ARPACK shift-invert.

    Same return convention as solve_system_modes: nearest first, legacy branch for zeta.
    Raises FEMConvergenceError if ARPACK does not converge all of them.
    """
    from scipy.sparse.linalg import splu, LinearOperator, eigs, ArpackNoConvergence
    K_csr, M_csr = _scipy_csr(K), _scipy_csr(M)
    free = np.array(list(space.FreeDofs()), dtype=bool)
    idx = np.flatnonzero(free)
    n = len(idx)
    ncv = 2 * num_BMs + num_BMs // 2 + 3
    if ncv > n:
        raise FEMConvergenceError(
            f"2D mesh too coarse for num_BMs={num_BMs}: the eigensolver needs {ncv} Krylov "
            f"vectors but the mesh has only {n} unknowns (legacy EMUstack: ARPACK info=-3).")
    Kf = K_csr[idx][:, idx].tocsc()
    Mf = M_csr[idx][:, idx].tocsc()
    lu = splu((Kf - sigma * Mf).tocsc())
    OP = LinearOperator((n, n), matvec=lambda x: lu.solve(Mf @ x), dtype=complex)
    rng = np.random.default_rng(42)
    v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    try:
        mu, vecs = eigs(OP, k=num_BMs, which='LM', ncv=ncv, tol=0.0, maxiter=maxiter, v0=v0)
    except ArpackNoConvergence as e:
        raise FEMConvergenceError(
            f"ARPACK converged {len(e.eigenvalues)} of {num_BMs} Bloch modes") from e
    lam = sigma + 1.0 / mu

    zetas, coeffs, residuals = [], [], []
    for i in np.argsort(np.abs(lam - sigma), kind='stable'):
        c = np.zeros(K_csr.shape[0], dtype=complex)
        c[idx] = vecs[:, i]
        Kc, Mc = K_csr @ c, M_csr @ c
        denom = np.linalg.norm(Kc) + np.linalg.norm(Mc)
        z = np.sqrt(lam[i])
        if abs(z.imag) < 1e-8:  # legacy branch: forward propagating / decaying
            if z.real < 0:
                z, c = -z, -c
        elif z.imag < 0:
            z, c = -z, -c
        zetas.append(z)
        coeffs.append(c)
        residuals.append(np.linalg.norm(Kc - lam[i] * Mc) / max(denom, 1e-300))
    return np.array(zetas), np.column_stack(coeffs), np.array(residuals)


def solve_system_modes(
    K: BilinearForm,
    M: BilinearForm,
    space: ngs.comp.FESpace,
    sigma: complex,
    numeig: int,
    k0: float,
    max_order_PWs: int,
    n_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve K u = lambda M u with ArnoldiSolver around the shift sigma.

    Returns all finite eigenpairs ordered by distance to sigma (nearest first), as
    ARPACK in shift-invert mode delivers them to legacy EMUstack, with zeta = sqrt(lambda)
    on the legacy branch and the relative residual of each pair. max_order_PWs and n_max
    are kept for the call signature.
    """
    gfs = [GridFunction(space) for _ in range(numeig)]
    lam = np.array(ngs.ArnoldiSolver(K.mat, M.mat, space.FreeDofs(), [g.vec for g in gfs], shift=sigma))

    K_vals, K_col, K_row = K.mat.CSR()
    K_csr = sp.csr_matrix((np.array(K_vals), np.array(K_col), np.array(K_row)), shape=K.mat.shape)
    M_vals, M_col, M_row = M.mat.CSR()
    M_csr = sp.csr_matrix((np.array(M_vals), np.array(M_col), np.array(M_row)), shape=M.mat.shape)

    zetas, coeffs, residuals = [], [], []
    for i in np.argsort(np.abs(lam - sigma), kind='stable'):
        l = lam[i]
        if not np.isfinite(l):
            continue
        c = np.array(gfs[i].vec.data)
        Kc = K_csr @ c
        Mc = M_csr @ c
        denom = np.linalg.norm(Kc) + np.linalg.norm(Mc)
        if denom < 1e-14:
            continue
        z = np.sqrt(l)
        # Legacy branch: forward propagating (Re > 0) or forward decaying (Im > 0)
        if abs(z.imag) < 1e-8:
            if z.real < 0:
                z, c = -z, -c
        elif z.imag < 0:
            z, c = -z, -c
        zetas.append(z)
        coeffs.append(c)
        residuals.append(np.linalg.norm(Kc - l * Mc) / denom)

    if not zetas:
        return np.array([]), np.array([]), np.array([])
    return np.array(zetas), np.column_stack(coeffs), np.array(residuals)


def select_modes(zeta, Phi, res, num_BMs, tol=1e-3):
    """Legacy EMUstack mode set: the num_BMs eigenpairs nearest the shift (input order),
    sorted by decreasing Re(zeta^2) as z_indexx.f does. Returns None if fewer than
    num_BMs are available or any of them is not converged to tol."""
    if len(zeta) < num_BMs or np.max(res[:num_BMs]) > tol:
        return None
    zeta, Phi, res = zeta[:num_BMs], Phi[:, :num_BMs], res[:num_BMs]
    order = z_indexx(zeta)  # legacy z_indexx.f
    return zeta[order], Phi[:, order], res[order]


# Quadrature for the mode/plane-wave overlap integrals:
# 'legacy'  - the 16-point degree-8 rule of legacy EMUstack (quad_triangle.f), default;
# 'ngsolve' - NGSolve's default rule for the integrand.
overlap_quadrature = 'legacy'


def _legacy_triangle_rule() -> "ngs.IntegrationRule":
    """quad_triangle.f: weights sum to 1/2 on the reference triangle. The rule is fully
    symmetric, so it does not depend on the reference vertex order."""
    a1, b1 = 4.592925882927229e-1, 8.141482341455413e-2
    a2, b2 = 5.054722831703103e-2, 8.989055433659379e-1
    a3, b3 = 1.705693077517601e-1, 6.588613844964797e-1
    c1, c2, c3 = 7.284923929554041e-1, 2.63112829634638689e-1, 8.394777409957211e-3
    w0 = 1.443156076777862e-1 / 2.0
    w1, w2, w3 = 2.852749028018549e-1 / 6.0, 9.737549286959440e-2 / 6.0, 3.096521116041552e-1 / 6.0
    w4 = 1.633818850466092e-1 / 12.0
    pts = [(1.0 / 3.0, 1.0 / 3.0),
           (a1, a1), (a1, b1), (b1, a1),
           (a2, a2), (a2, b2), (b2, a2),
           (a3, a3), (a3, b3), (b3, a3),
           (c1, c2), (c2, c1), (c2, c3), (c3, c2), (c3, c1), (c1, c3)]
    wts = [w0] + [w1] * 3 + [w2] * 3 + [w3] * 3 + [w4] * 6
    return ngs.IntegrationRule(points=pts, weights=wts)


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
    space_adj: Optional[ngs.comp.FESpace] = None,
    quadrature: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute projection overlaps J (divided by dy) and J_dag (un-normalized) via LinearForm assembly on periodic spaces."""
    quadrature = quadrature or overlap_quadrature
    if quadrature == 'legacy':
        dx = ngs.dx(intrules={ngs.TRIG: _legacy_triangle_rule()})
    elif quadrature == 'ngsolve':
        dx = ngs.dx
    else:
        raise ValueError("quadrature must be 'legacy' or 'ngsolve', got %r" % quadrature)
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
    # legacy pw_ordering.f: z_indexx on -Re(k_z^2), same order for ties
    sort_pw = z_indexx(kz_unsrt)
    alphas = alphas[sort_pw]
    betas = betas[sort_pw]
    Qs = np.sqrt(alphas**2 + betas**2)

    npw = len(alphas)
    nbm = Phi_pri.shape[1]

    # Un-conjugated integrals of a field with phase p need test functions from the space
    # with phase conj(p) (NGSolve folds periodic test dofs with the conjugate phase):
    # primal modes (phase p) use the adjoint space and adjoint modes the primal space.
    (v_pri, w_pri) = space_adj.TestFunction()
    (v_adj, w_adj) = space_pri.TestFunction()

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
        lf_pE = ngs.LinearForm(space_adj)
        lf_pE += (pe_x * ph_minus * v_pri[0] + pe_y * ph_minus * v_pri[1]) * dx
        lf_pE.Assemble()
        LF_pri_E.append(np.array(lf_pE.vec.data))

        lf_pK = ngs.LinearForm(space_adj)
        lf_pK += (pk_x * ph_minus * v_pri[0] + pk_y * ph_minus * v_pri[1]) * dx
        lf_pK.Assemble()
        LF_pri_K.append(np.array(lf_pK.vec.data))

        # Adjoint: plane wave has exp(+i (alpha x + beta y))
        lf_aE = ngs.LinearForm(space_pri)
        lf_aE += (pe_x * ph_plus * v_adj[0] + pe_y * ph_plus * v_adj[1]) * dx
        lf_aE.Assemble()
        LF_adj_E.append(np.array(lf_aE.vec.data))

        lf_aK = ngs.LinearForm(space_pri)
        lf_aK += (pk_x * ph_plus * v_adj[0] + pk_y * ph_plus * v_adj[1]) * dx
        lf_aK.Assemble()
        LF_adj_K.append(np.array(lf_aK.vec.data))

    mat_LF_pri = np.vstack([LF_pri_E, LF_pri_K])
    mat_LF_adj = np.vstack([LF_adj_E, LF_adj_K])

    J = (mat_LF_pri @ Phi_pri) / dy
    J_dag = Phi_adj.T @ mat_LF_adj.T

    return J, J_dag


def mode_fields_at_nodes(mesh, Phi, table_nod, x_arr, kx, ky, period_x, dy, order=2):
    """Mode fields at the 6 nodes of each element, as legacy array_sol.f stores them.

    Elements are straight (legacy forces info_curved = 0), so the nodes are the vertices
    and the straight edge midpoints (1-2, 2-3, 3-1) of each .mail element, and every
    element is evaluated on its own side (the normal E component jumps across material
    interfaces). The field is evaluated on the quasi-periodic space, so dofs on the
    identified boundaries carry their Bloch phase.

    Returns sol1 with shape (6, 13, nval, nel): [Ex, Ey, Ez-variable, Hx, Hy, Hz] terms at
    nodes 0..5; slots 6..12 (legacy: P3 values of Ez) are left zero.
    """
    phase = [np.exp(+1j * kx * period_x), np.exp(+1j * ky * dy)]
    space = ngs.Periodic(HCurl(mesh, order=order, complex=True), phase=phase) * \
        ngs.Periodic(H1(mesh, order=order + 1, complex=True), phase=phase)
    table_nod = np.asarray(table_nod)
    x_arr = np.asarray(x_arr)
    xy = x_arr.T if x_arr.shape[0] == 2 else x_arr
    nel = table_nod.shape[1]
    nval = Phi.shape[1]

    verts = xy[table_nod[0:3] - 1]                       # (3, nel, 2) .mail vertices

    # the same 6 points of every NGSolve element, then matched to the targets
    ref = ngs.IntegrationRule(points=[(1, 0), (0, 1), (0, 0), (0.5, 0.5), (0, 0.5), (0.5, 0)],
                              weights=[1.0] * 6)
    mpts = mesh.MapToAllElements(ref, ngs.VOL)
    px = np.asarray(ngs.x(mpts)).reshape(-1, 6)
    py = np.asarray(ngs.y(mpts)).reshape(-1, 6)
    if px.shape[0] != nel:
        raise ValueError("mesh has %d elements, table_nod %d" % (px.shape[0], nel))
    pts = np.stack([px, py], axis=-1)                     # (nel_ng, 6, 2)
    cen_ng = pts[:, :3].mean(axis=1)
    cen_mail = verts.mean(axis=0)
    el_map = np.arange(nel)
    if np.max(np.abs(cen_ng - cen_mail)) > 1e-9:
        from scipy.spatial import cKDTree
        el_map = cKDTree(cen_ng).query(cen_mail)[1]       # .mail element -> NGSolve element
    # vertices by position; each edge midpoint is the mapped midpoint of the same two
    # reference vertices (on curved netgen elements it lies on the curve)
    pick = np.zeros((6, nel), dtype=int)                  # flat index into the mapped points
    ref_of_local = np.zeros((3, nel), dtype=int)          # local vertex -> reference vertex
    for i in range(3):
        d = np.linalg.norm(pts[el_map][:, :3, :] - verts[i][:, None, :], axis=-1)   # (nel, 3)
        j = np.argmin(d, axis=1)
        if np.max(d[np.arange(nel), j]) > 1e-9:
            raise ValueError("element vertices do not match the mesh")
        ref_of_local[i] = j
        pick[i] = el_map * 6 + j
    ref_mid = {frozenset((0, 1)): 3, frozenset((1, 2)): 4, frozenset((2, 0)): 5}
    for i, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        j = np.array([ref_mid[frozenset((ra, rb))] for ra, rb in zip(ref_of_local[a], ref_of_local[b])])
        pick[3 + i] = el_map * 6 + j

    sol1 = np.zeros((6, 13, nval, nel), dtype=complex)
    gf = GridFunction(space)
    u_n, z_n = gf.components
    cfs = [u_n, z_n, ngs.grad(z_n), ngs.curl(u_n)]
    for n in range(nval):
        gf.vec.FV().NumPy()[:] = Phi[:, n]
        u, z, gz, cu = (np.asarray(cf(mpts)) for cf in cfs)
        u = u.reshape(-1, 2); gz = gz.reshape(-1, 2); z = z.ravel(); cu = cu.ravel()
        for i in range(6):
            k = pick[i]
            sol1[0, i, n] = u[k, 0]
            sol1[1, i, n] = u[k, 1]
            sol1[2, i, n] = z[k]
            sol1[3, i, n] = -u[k, 1] - gz[k, 1]
            sol1[4, i, n] = u[k, 0] + gz[k, 0]
            sol1[5, i, n] = cu[k]
    return sol1


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
    """Solve 2D Bloch modes via NGSolve."""
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

            mesh = meshing.unit_cell(
                dy,
                inclusions,
                lc=getattr(structure, "lc", 0.1),
                lc2=getattr(structure, "lc2", 1.0),
                lc3=getattr(structure, "lc3", 1.0),
                lc4=getattr(structure, "lc4", 1.0),
                lc5=getattr(structure, "lc5", 1.0),
                lc6=getattr(structure, "lc6", 1.0),
                curve_order=order,
                period_x=period_x,
            )
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

    # 3. Assemble system
    t_asm = time.perf_counter()
    K_pri, M_pri, space_pri, fes = assemble_system_2d(mesh, dy, eps_cf, k0, kx, ky, order=order, period_x=period_x)
    K_adj, M_adj, space_adj, _ = assemble_system_2d(mesh, dy, eps_cf, k0, -kx, -ky, order=order, period_x=period_x)
    B_csr = assemble_bilinear_B(mesh, space_pri, space_adj, order=order)
    time_asm = time.perf_counter() - t_asm

    # 4. Eigen-solve around the shift; keep the num_BMs modes nearest to it (legacy)
    t_arnoldi = time.perf_counter()
    if eigensolver == 'arpack':
        pri = select_modes(*solve_system_modes_arpack(K_pri, M_pri, space_pri, sigma, num_BMs), num_BMs)
        adj = select_modes(*solve_system_modes_arpack(K_adj, M_adj, space_adj, sigma, num_BMs), num_BMs)
    elif eigensolver == 'ngsolve':
        numeig = max(num_BMs * 3, num_BMs + 40)
        for attempt in range(2):
            pri = select_modes(*solve_system_modes(
                K_pri, M_pri, space_pri, sigma, numeig, k0, max_order_PWs, n_max), num_BMs)
            adj = select_modes(*solve_system_modes(
                K_adj, M_adj, space_adj, sigma, numeig, k0, max_order_PWs, n_max), num_BMs)
            if pri is not None and adj is not None:
                break
            numeig = int(numeig * 1.5)  # retry once with a larger Krylov space
    else:
        raise ValueError("eigensolver must be 'arpack' or 'ngsolve', got %r" % eigensolver)
    if pri is None or adj is None:
        raise FEMConvergenceError(
            f"Could not converge the {num_BMs} Bloch modes nearest the shift "
            f"(primal {'ok' if pri is not None else 'failed'}, "
            f"adjoint {'ok' if adj is not None else 'failed'})"
        )
    zeta_pri, Phi_pri, res_pri = pri
    zeta_adj, Phi_adj, res_adj = adj

    # Nearest-neighbor mode pairing between primal and adjoint eigenspectra
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

    # 5. Biorthogonal pairing and normalization
    # L_mn = (zeta_n / k0) * Phi_adj[m]^T @ B @ Phi_pri[n]
    L = (1.0 / k0) * (Phi_adj.T @ B_csr @ Phi_pri) @ np.diag(zeta_pri)

    # Conjugate-pair repair
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

    # Algebraic biorthogonal repair Phi_adj <- Phi_adj @ L^-T
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

    # 6. Mode self-flux matrix F (primal in both slots)
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

    # 8. Mode polarization / energy fractions
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

    # 9. Mode fields at the 6 nodes of every element (legacy array_sol.f layout)
    sol1 = None
    if plotting_fields and structure is not None and getattr(structure, "table_nod", None) is not None:
        sol1 = mode_fields_at_nodes(mesh, Phi_pri, structure.table_nod, structure.x_arr,
                                    kx, ky, period_x, dy, order)

    # Degenerate cluster labelling
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
