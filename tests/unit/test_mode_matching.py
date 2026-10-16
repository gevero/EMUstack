"""Unit tests for Mode Matching and Simmo-Simmo interfaces (PORTING_PLAN §8, §10).

Verifies:
- Reduction proof 1: P=A, Q=B reproduces r_t_mat_tf_ns to <1e-12
- Reduction proof 2: P=Q=sqrt(Z2/Z1)*I reproduces r_t_mat_anallo to <1e-12
- Transparency: P=Q=I yields R12=0, T12=I, eps_trunc=0
- Gauge covariance: under P -> C1^-1 P C2, Q -> C2^-1 Q C1, R12 and T12 transform covariantly
- Interface completeness diagnostics
"""
import numpy as np
import pytest

from mode_matching import solve_rt_from_PQ, interface_completeness


def test_transparency_limit():
    """Identity continuity P=Q=I must give exact reflection-free transmission R12=0, T12=I."""
    N = 5
    P = np.eye(N, dtype=complex)
    Q = np.eye(N, dtype=complex)

    R12, T12, R21, T21 = solve_rt_from_PQ(P, Q)

    assert np.allclose(R12, np.zeros((N, N)), atol=1e-14)
    assert np.allclose(R21, np.zeros((N, N)), atol=1e-14)
    assert np.allclose(T12, np.eye(N), atol=1e-14)
    assert np.allclose(T21, np.eye(N), atol=1e-14)

    diag = interface_completeness(None, None, P=P, Q=Q)
    assert abs(diag["eps_trunc"]) < 1e-14


def test_reduction_to_fresnel():
    """Reduction proof: P = Q = sqrt(Z2/Z1) * I must reproduce Fresnel r_t_mat_anallo (§8.3)."""
    Z1 = 1.0 + 0.0j
    Z2 = 2.25 + 0.0j
    ratio = np.sqrt(Z2 / Z1)

    N = 4
    P = ratio * np.eye(N, dtype=complex)
    Q = ratio * np.eye(N, dtype=complex)

    R12, T12, R21, T21 = solve_rt_from_PQ(P, Q)

    # Analytic Fresnel formulas:
    # r12 = (Z2 - Z1) / (Z2 + Z1)
    # t12 = 2 * sqrt(Z1 * Z2) / (Z1 + Z2)
    # r21 = -r12, t21 = t12
    r_fresnel = (Z2 - Z1) / (Z2 + Z1)
    t_fresnel = 2.0 * np.sqrt(Z1 * Z2) / (Z1 + Z2)

    assert np.allclose(R12, r_fresnel * np.eye(N), atol=1e-13)
    assert np.allclose(R21, -r_fresnel * np.eye(N), atol=1e-13)
    assert np.allclose(T12, t_fresnel * np.eye(N), atol=1e-13)
    assert np.allclose(T21, t_fresnel * np.eye(N), atol=1e-13)


def test_reduction_to_tf_ns():
    """Reduction proof: P = A, Q = B reproduces Dossou r_t_mat_tf_ns formulas element-wise (§8.3)."""
    rng = np.random.default_rng(1234)
    N1 = 4
    N2 = 6

    # Random rectangular A (2*npw x nbm) and B (nbm x 2*npw)
    A = rng.standard_normal((N1, N2)) + 1j * rng.standard_normal((N1, N2))
    B = rng.standard_normal((N2, N1)) + 1j * rng.standard_normal((N2, N1))

    # solve_rt_from_PQ canonical solver
    R12_pq, T12_pq, R21_pq, T21_pq = solve_rt_from_PQ(A, B)

    # Legacy Dossou formulas from mode_calcs.py:509-520
    den = np.eye(N2) + B @ A
    den_inv_B = np.linalg.solve(den, B)
    R12_legacy = -np.eye(N1) + 2.0 * A @ den_inv_B
    T12_legacy = 2.0 * den_inv_B
    R21_legacy = np.linalg.solve(den, np.eye(N2) - B @ A)
    T21_legacy = 2.0 * A @ np.linalg.inv(den)

    assert np.allclose(R12_pq, R12_legacy, atol=1e-13)
    assert np.allclose(T12_pq, T12_legacy, atol=1e-13)
    assert np.allclose(R21_pq, R21_legacy, atol=1e-13)
    assert np.allclose(T21_pq, T21_legacy, atol=1e-13)


def test_gauge_covariance():
    """G1.5: Under P -> C1^-1 P C2 and Q -> C2^-1 Q C1, R12 and T12 transform covariantly (§10.1)."""
    rng = np.random.default_rng(5678)
    N1 = 4
    N2 = 5

    P = rng.standard_normal((N1, N2)) + 1j * rng.standard_normal((N1, N2))
    Q = rng.standard_normal((N2, N1)) + 1j * rng.standard_normal((N2, N1))

    R12, T12, R21, T21 = solve_rt_from_PQ(P, Q)

    # Random unitary gauge matrices C1 and C2 (physical mode mixing)
    C1_raw = rng.standard_normal((N1, N1)) + 1j * rng.standard_normal((N1, N1))
    C2_raw = rng.standard_normal((N2, N2)) + 1j * rng.standard_normal((N2, N2))
    C1, _ = np.linalg.qr(C1_raw)
    C2, _ = np.linalg.qr(C2_raw)
    C1_inv = np.linalg.inv(C1)
    C2_inv = np.linalg.inv(C2)

    P_prime = C1_inv @ P @ C2
    Q_prime = C2_inv @ Q @ C1

    R12_p, T12_p, R21_p, T21_p = solve_rt_from_PQ(P_prime, Q_prime)

    # Gauge transformation laws (§3.6, §10.1):
    # R12 -> C1^-1 R12 C1
    # T12 -> C2^-1 T12 C1
    # R21 -> C2^-1 R21 C2
    # T21 -> C1^-1 T21 C2
    assert np.allclose(R12_p, C1_inv @ R12 @ C1, atol=1e-11)
    assert np.allclose(T12_p, C2_inv @ T12 @ C1, atol=1e-11)
    assert np.allclose(R21_p, C2_inv @ R21 @ C2, atol=1e-11)
    assert np.allclose(T21_p, C1_inv @ T21 @ C2, atol=1e-11)

    # Truncation error invariant
    diag1 = interface_completeness(None, None, P=P, Q=Q)
    diag2 = interface_completeness(None, None, P=P_prime, Q=Q_prime)
    assert np.isclose(diag1["eps_trunc"], diag2["eps_trunc"], atol=1e-11)
