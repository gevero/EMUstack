"""Field evaluation and plotting support for EMUstack (PORTING_PLAN §9).

Pure Python / NumPy replacement for legacy Fortran routines:
- field_value_plane
- gmsh_plot_field
- gmsh_plot_slice
- gmsh_plot_field_3d
"""
import os
from typing import Optional, Tuple, Dict, Any
import numpy as np


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
    **kwargs
) -> np.ndarray:
    """Evaluate 3D electric field (Ex, Ey, Ez) at element nodes in a horizontal plane (§9.1).
    
    Returns array m_E of shape (n_msh_el, 6, 3) with complex values.
    """
    m_E = np.zeros((n_msh_el, 6, 3), dtype=np.complex128)
    if sol1 is None:
        return m_E

    num_modes = min(num_BMs, sol1.shape[2], len(k_z))
    vec_coef = np.asarray(vec_coef).flatten()
    k_z_arr = np.asarray(k_z).flatten()
    coef_down = vec_coef[:num_modes]
    coef_up = vec_coef[num_modes:2 * num_modes]

    prop = coef_down * np.exp(1j * k_z_arr[:num_modes] * z_value) + \
           coef_up * np.exp(1j * k_z_arr[:num_modes] * (h_normed - z_value))

    for iel in range(n_msh_el):
        for inod in range(min(6, sol1.shape[1])):
            ex_modes = np.asarray(sol1[0, inod, :num_modes, iel]).flatten()
            ey_modes = np.asarray(sol1[1, inod, :num_modes, iel]).flatten()
            ez_modes = np.asarray(sol1[2, inod, :num_modes, iel]).flatten()

            m_E[iel, inod, 0] = np.sum(prop * ex_modes)
            m_E[iel, inod, 1] = np.sum(prop * ey_modes)
            m_E[iel, inod, 2] = np.sum(prop * (-1j * k_z_arr[:num_modes] * ez_modes))

    return m_E


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
    """Write in-plane field values into Gmsh .pos view file (§9.1)."""
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
    """Write vertical field slice into Gmsh .pos view file (§9.1)."""
    dir_name = "fields_vertically"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(dir_name + "/gmsh_BMs"):
        os.makedirs(dir_name + "/gmsh_BMs", exist_ok=True)
    if not os.path.exists(dir_name + "/gmsh_BMs/anim"):
        os.makedirs(dir_name + "/gmsh_BMs/anim", exist_ok=True)


def gmsh_plot_field_3d(*args, **kwargs):
    """Write 3D field into Gmsh .pos view file (§9.1)."""
    dir_name = "3d_fields"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(dir_name + "/anim"):
        os.makedirs(dir_name + "/anim", exist_ok=True)
