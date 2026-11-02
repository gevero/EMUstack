"""Mode matching module for EMUstack.

Provides:
- Interface reflection/transmission matrices for Simmo <-> Simmo interfaces (r_t_mat_ns_ns)
- General P, Q interface system solver (solve_rt_from_PQ) with reduction proofs
- Interface completeness diagnostics (interface_completeness)
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
# Inside the EMUstack package, import the package modules (one copy of each).
# Bare names are only for legacy scripts that put backend/ on sys.path.
if __package__:
    from .linalg import mat
else:
    from linalg import mat


def _assert_fem_fem_compatible(sim1, sim2):
    """Verify geometrical and physical compatibility of two layers for mode matching."""
    assert sim1.structure.period == sim2.structure.period, "Layers must have identical period"
    assert sim1.structure.period_y == sim2.structure.period_y, "Layers must have identical period_y"
    assert sim1.structure.world_1d == sim2.structure.world_1d, "Layers must have identical world_1d"
    assert np.isclose(sim1.light.wl_nm, sim2.light.wl_nm), "Layers must have identical wavelength"
    assert np.allclose(sim1.light.k_pll, sim2.light.k_pll), "Layers must have identical Bloch vector k_pll"


def solve_rt_from_PQ(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 4 interface scattering blocks from continuity matrices P and Q.
    
    P: (N1, N2)
    Q: (N2, N1)
    
    Returns:
    - R12: (N1, N1)
    - T12: (N2, N1)
    - R21: (N2, N2)
    - T21: (N1, N2)
    """
    N1, N2 = P.shape
    assert Q.shape == (N2, N1), f"Shape mismatch: P is {P.shape}, Q is {Q.shape}"

    PQ = P @ Q
    QP = Q @ P
    I1 = np.eye(N1, dtype=complex)
    I2 = np.eye(N2, dtype=complex)

    # Closed-form canonical interface scattering formulas:
    # T12 = 2 * solve(I2 + QP, Q)
    # R12 = solve(I1 + PQ, PQ - I1)
    # R21 = solve(I2 + QP, I2 - QP)
    # T21 = 2 * P @ inv(I2 + QP)
    T12 = 2.0 * np.linalg.solve(I2 + QP, Q)
    R12 = np.linalg.solve(I1 + PQ, PQ - I1)
    R21 = np.linalg.solve(I2 + QP, I2 - QP)
    inv_I2_QP = np.linalg.inv(I2 + QP)
    T21 = 2.0 * (P @ inv_I2_QP)

    return R12, T12, R21, T21


def interface_completeness(sim1, sim2, P: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute interface diagnostics: eps_trunc, rank, singular values, and condition number."""
    if P is None or Q is None:
        P, Q = compute_PQ(sim1, sim2)

    N1, N2 = P.shape
    PQ = P @ Q
    eps_trunc = float(np.linalg.norm(PQ - np.eye(N1)) / np.sqrt(N1))

    s_P = np.linalg.svd(P, compute_uv=False)
    s_Q = np.linalg.svd(Q, compute_uv=False)

    rank_P = int(np.sum(s_P > 1e-12 * s_P[0]))
    rank_Q = int(np.sum(s_Q > 1e-12 * s_Q[0]))

    cond_I_PQ = float(np.linalg.cond(np.eye(N1) + PQ))

    return {
        "eps_trunc": eps_trunc,
        "rank_P": rank_P,
        "rank_Q": rank_Q,
        "sigma_min_P": float(s_P[-1]) if len(s_P) > 0 else 0.0,
        "sigma_min_Q": float(s_Q[-1]) if len(s_Q) > 0 else 0.0,
        "cond_I_PQ": cond_I_PQ,
        "N1": N1,
        "N2": N2
    }


def compute_PQ(sim1, sim2) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuity matrices P and Q between two Simmo layers."""
    _assert_fem_fem_compatible(sim1, sim2)

    # In plane-wave mediated basis:
    # P = J_dag_1 @ Z_air^-1/2 @ (Z_air^-1/2 @ J_2) = B1 @ A2
    air = sim1.air_ref()
    Z_air = air.Z()
    Z1_sqrt_inv = np.sqrt(1.0 / Z_air).reshape((1, -1))

    A2 = Z1_sqrt_inv.T * np.asarray(sim2.J)
    B1 = np.asarray(sim1.J_dag) * Z1_sqrt_inv

    P = np.asarray(sim1.J_dag) @ np.asarray(sim2.J)
    Q = np.asarray(sim2.J_dag) @ np.asarray(sim1.J)
    return P, Q


def r_t_mat_ns_ns(sim1, sim2, interface_mode: str = "air"):
    """Compute R12, T12, R21, T21 at a Simmo-Simmo interface.
    
    Parameters:
    - sim1: incident Simmo layer
    - sim2: transmitted Simmo layer
    - interface_mode: 'air' (zero-thickness air mediator, default) or 'direct' (algebraic P, Q system)
    """
    _assert_fem_fem_compatible(sim1, sim2)

    if interface_mode == "air":
        # Mediated through zero-thickness air reference (fallback)
        if __package__:
            from . import mode_calcs
        else:
            import mode_calcs
        air = sim1.air_ref()

        R1_air, T1_air, R_air_1, T_air_1 = mode_calcs.r_t_mat_tf_ns(air, sim1)
        # Flip order for air -> sim1:
        # Note r_t_mat_tf_ns(an1, sim2) has an1 as first (incident) and sim2 as second.
        # So r_t_mat_tf_ns(air, sim1) has:
        # R12: air -> air reflection
        # T12: air -> sim1 transmission
        # R21: sim1 -> sim1 reflection
        # T21: sim1 -> air transmission
        # Therefore:
        # sim1 -> air has:
        # R = R21, T = T21
        # air -> sim1 has:
        # R = R12, T = T12
        R_s1_air = R_air_1
        T_s1_air = T_air_1
        R_air_s1 = R1_air
        T_air_s1 = T1_air

        # air -> sim2:
        R_air_s2, T_air_s2, R_s2_air, T_s2_air = mode_calcs.r_t_mat_tf_ns(air, sim2)

        # Redheffer star product at zero separation
        I_air = mat(np.eye(len(R_air_s1), dtype=complex))
        D = mat(np.linalg.inv(I_air - R_air_s2 * R_air_s1))
        D_rev = mat(np.linalg.inv(I_air - R_air_s1 * R_air_s2))

        R12 = R_s1_air + T_air_s1 * R_air_s2 * D * T_s1_air
        T12 = T_air_s2 * D * T_s1_air

        R21 = R_s2_air + T_air_s2 * R_air_s1 * D_rev * T_s2_air
        T21 = T_air_s1 * D_rev * T_s2_air

        return mat(R12), mat(T12), mat(R21), mat(T21)

    else:
        # Direct P, Q interface solve
        P, Q = compute_PQ(sim1, sim2)
        R12, T12, R21, T21 = solve_rt_from_PQ(P, Q)
        return mat(R12), mat(T12), mat(R21), mat(T21)
