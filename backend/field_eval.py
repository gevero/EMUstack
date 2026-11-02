"""Field evaluation and plotting support for EMUstack.

Pure Python / NumPy replacement for legacy Fortran routines:
- field_value_plane
- gmsh_plot_field
- gmsh_plot_slice
- gmsh_plot_field_3d
"""
import os
from typing import Optional, Tuple, Dict, Any
import numpy as np

# How Ez is rebuilt from the stored axial mode variable ez (same variable in both codes):
# 'physical' - Ez = sum (c_down - c_up) * (-i k_z) * ez, what the eigenproblem implies: it
#              satisfies div E = 0 in a homogeneous layer (checked in 1D and 2D). Default.
# 'legacy'   - legacy EMUstack (field_value_plane.f, gmsh_plot_*.f, plotting.py):
#              Ez = sum (c_up - c_down) * ez / k_z, off from the physical Ez by a factor
#              i k_z^2. Only to reproduce legacy field maps (benchmarks/parity).
# Environment variable EMUSTACK_EZ_CONVENTION sets the default. Whenever H is computed as
# well (E.H, chirality: no legacy counterpart), Ez is physical regardless.
ez_convention = os.environ.get('EMUSTACK_EZ_CONVENTION', 'physical')
if ez_convention not in ('physical', 'legacy'):
    raise ValueError("EMUSTACK_EZ_CONVENTION must be 'physical' or 'legacy', got %r" % ez_convention)


def ez_coefficient(prop_minus, k_z, physical=None):
    """Per-mode factor multiplying ez; prop_minus = c_down P_down - c_up P_up."""
    if physical is None:
        if ez_convention not in ('physical', 'legacy'):
            raise ValueError("field_eval.ez_convention must be 'physical' or 'legacy', got %r"
                             % ez_convention)
        physical = ez_convention == 'physical'
    if physical:
        return prop_minus * (-1j * k_z)
    return -prop_minus / k_z


def field_value_plane(
    num_BMs: int,
    n_msh_el: int,
    n_msh_pts: int,
    nnodes: int,
    nb_typ_el: int,
    table_nod: np.ndarray,
    type_el: np.ndarray,
    eps: np.ndarray,
    x_arr: np.ndarray,
    k_z: np.ndarray,
    sol1: np.ndarray,
    vec_coef: np.ndarray,
    h_normed: float,
    z_value: float,
    *args,
    k0: Optional[float] = None,
    wl_normed: Optional[float] = None,
    return_H: bool = False,
    **kwargs
) -> Any:
    """Evaluate 3D electric (and optionally magnetic) fields (Ex, Ey, Ez, Hx, Hy, Hz)
    at element nodes in a horizontal plane.
    
    Returns array m_E of shape (n_msh_el, 6, 3), or tuple (m_E, m_H) if return_H is True.
    """
    m_E = np.zeros((n_msh_el, 6, 3), dtype=np.complex128)
    m_H = np.zeros((n_msh_el, 6, 3), dtype=np.complex128)
    if sol1 is None:
        return (m_E, m_H) if return_H else m_E

    num_modes = min(num_BMs, sol1.shape[2], len(k_z))
    vec_coef = np.asarray(vec_coef).flatten()
    k_z_arr = np.asarray(k_z).flatten()[:num_modes]
    coef_down = vec_coef[:num_modes]
    coef_up = vec_coef[num_modes:2 * num_modes]

    prop_plus = coef_down * np.exp(1j * k_z_arr * z_value) + \
                coef_up * np.exp(1j * k_z_arr * (h_normed - z_value))
    prop_minus = coef_down * np.exp(1j * k_z_arr * z_value) - \
                 coef_up * np.exp(1j * k_z_arr * (h_normed - z_value))

    # Resolve k0
    if k0 is None:
        k0 = kwargs.get("k_0", None)
    if k0 is None:
        if wl_normed is not None and wl_normed > 0:
            k0 = 2.0 * np.pi / float(wl_normed)
        elif len(k_z_arr) > 0 and np.abs(k_z_arr[0]) > 0:
            k0 = float(np.abs(k_z_arr[0]))
        else:
            k0 = 2.0 * np.pi

    has_H = (sol1.shape[0] >= 6)
    ez_coef = ez_coefficient(prop_minus, k_z_arr, physical=True if return_H else None)

    # Sum over modes for every element node at once: sol1[comp, node, mode, element].
    # The explicit node/element loop this replaces dominated the cost of 500-mode
    # field maps (tens of thousands of small np.sum calls).
    nnodes = min(6, sol1.shape[1])
    n_el = min(n_msh_el, sol1.shape[3]) if sol1.ndim == 4 else n_msh_el
    if num_modes and nnodes and n_el:
        def _modes_sum(comp, weights):
            return np.einsum('nme,m->en', sol1[comp, :nnodes, :num_modes, :n_el], weights)

        m_E[:n_el, :nnodes, 0] = _modes_sum(0, prop_plus)
        m_E[:n_el, :nnodes, 1] = _modes_sum(1, prop_plus)
        m_E[:n_el, :nnodes, 2] = _modes_sum(2, ez_coef)

        if has_H:
            prop_h = prop_minus * (k_z_arr / k0)
            m_H[:n_el, :nnodes, 0] = _modes_sum(3, prop_h)
            m_H[:n_el, :nnodes, 1] = _modes_sum(4, prop_h)
            m_H[:n_el, :nnodes, 2] = _modes_sum(5, prop_plus * (1.0 / (1j * k0)))

    if return_H:
        return m_E, m_H
    return m_E


def field_value_plane_EH(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate both 3D electric and magnetic fields (m_E, m_H) at element nodes in a horizontal plane."""
    kwargs["return_H"] = True
    return field_value_plane(*args, **kwargs)


def gmsh_plot_field(
    num_BMs: int,
    n_msh_el: int,
    n_msh_pts: int,
    nnodes: int,
    nb_typ_el: int,
    table_nod: np.ndarray,
    type_el: np.ndarray,
    eps: np.ndarray,
    x_arr: np.ndarray,
    k_z: np.ndarray,
    sol1: np.ndarray,
    vec_coef: np.ndarray,
    h_normed: float,
    z_value: float,
    gmsh_file_pos: str,
    plot_real: int,
    plot_imag: int,
    plot_abs: int,
    extra_name: str,
    *args,
    **kwargs
):
    """Write in-plane field values into Gmsh .pos view file."""
    dir_name = "in_plane_fields"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    m_E = field_value_plane(
        num_BMs, n_msh_el, n_msh_pts, nnodes, nb_typ_el,
        table_nod, type_el, eps, x_arr, k_z, sol1,
        vec_coef, h_normed, z_value
    )

    filename = os.path.join(dir_name, f"{extra_name}_BM_abs2_eD_{gmsh_file_pos}.pos")
    with open(filename, "w") as f:
        f.write("View.AdaptVisualizationGrid = 1;\n")
        f.write("View.IntervalsType = 3;\n")
        f.write("View.Light = 0;\n")
        f.write(f'View "|E_t|^2: nval = {num_BMs}; hz = {-z_value}" {{\n')
        for iel in range(n_msh_el):
            nodes = table_nod[:3, iel] - 1
            p0 = x_arr[:, nodes[0]]
            p1 = x_arr[:, nodes[1]]
            p2 = x_arr[:, nodes[2]]
            v0 = float(np.sum(np.abs(m_E[iel, 0, :2])**2))
            v1 = float(np.sum(np.abs(m_E[iel, 1, :2])**2))
            v2 = float(np.sum(np.abs(m_E[iel, 2, :2])**2))
            f.write(f"ST({p0[0]},{p0[1]},{-z_value},{p1[0]},{p1[1]},{-z_value},{p2[0]},{p2[1]},{-z_value}){{{v0},{v1},{v2}}};\n")
        f.write("};\n")


def gmsh_plot_slice(*args, **kwargs):
    """Write vertical field slice into Gmsh .pos view file."""
    dir_name = "fields_vertically"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(dir_name + "/gmsh_BMs"):
        os.makedirs(dir_name + "/gmsh_BMs", exist_ok=True)
    if not os.path.exists(dir_name + "/gmsh_BMs/anim"):
        os.makedirs(dir_name + "/gmsh_BMs/anim", exist_ok=True)


def gmsh_plot_field_3d(*args, **kwargs):
    """Write 3D field into Gmsh .pos view file."""
    dir_name = "3d_fields"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(dir_name + "/anim"):
        os.makedirs(dir_name + "/anim", exist_ok=True)
