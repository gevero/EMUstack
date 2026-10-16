"""Exact rational element matrix verification (PORTING_PLAN §7.2).

Proves that Mxx, Myy, Kxy, Kyx, Kyy closed-form rational matrices match
independent high-order Gauss-Legendre numerical quadrature to machine precision.
"""
import numpy as np
from scipy.integrate import fixed_quad


def lagrange_p2(x, h):
    """P2 Lagrange polynomials at x in [0, h] with nodes [0, h, h/2]."""
    # N1 at 0: (x - h)*(x - h/2) / (h * h/2) = 2*(x - h)*(x - h/2)/h^2
    # N2 at h: x*(x - h/2) / (h * h/2) = 2*x*(x - h/2)/h^2
    # N3 at h/2: 4*x*(h - x)/h^2
    N1 = 2 * (x - h) * (x - 0.5 * h) / (h**2)
    N2 = 2 * x * (x - 0.5 * h) / (h**2)
    N3 = 4 * x * (h - x) / (h**2)
    return np.array([N1, N2, N3])


def lagrange_p3(x, h):
    """P3 Lagrange polynomials at x in [0, h] with nodes [0, h, h/3, 2h/3]."""
    x1, x2, x3, x4 = 0.0, h, h / 3.0, 2.0 * h / 3.0
    N1 = (x - x2) * (x - x3) * (x - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
    N2 = (x - x1) * (x - x3) * (x - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
    N3 = (x - x1) * (x - x2) * (x - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
    N4 = (x - x1) * (x - x2) * (x - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))
    return np.array([N1, N2, N3, N4])


def d_lagrange_p3(x, h):
    """Derivatives of P3 Lagrange polynomials at x."""
    x1, x2, x3, x4 = 0.0, h, h / 3.0, 2.0 * h / 3.0
    # derivative of product (x-a)(x-b)(x-c) is (x-b)(x-c) + (x-a)(x-c) + (x-a)(x-b)
    dN1 = ((x - x3) * (x - x4) + (x - x2) * (x - x4) + (x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
    dN2 = ((x - x3) * (x - x4) + (x - x1) * (x - x4) + (x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
    dN3 = ((x - x2) * (x - x4) + (x - x1) * (x - x4) + (x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
    dN4 = ((x - x2) * (x - x3) + (x - x1) * (x - x3) + (x - x1) * (x - x2)) / ((x4 - x1) * (x4 - x2) * (x4 - x3))
    return np.array([dN1, dN2, dN3, dN4])


def test_rational_element_matrices():
    """Verify closed rational element matrices against Gauss-Legendre quadrature across h."""
    for h in [0.01, 0.1, 0.25, 1.0]:
        # Rational definitions (§7.2):
        Mxx_rational = (h / 30.0) * np.array([
            [4.0, -1.0, 2.0],
            [-1.0, 4.0, 2.0],
            [2.0, 2.0, 16.0]
        ])
        
        Myy_rational = (h / 1680.0) * np.array([
            [128.0, 19.0, 99.0, -36.0],
            [19.0, 128.0, -36.0, 99.0],
            [99.0, -36.0, 648.0, -81.0],
            [-36.0, 99.0, -81.0, 648.0]
        ])
        
        Kxy_rational = (1.0 / 120.0) * np.array([
            [-83.0, -7.0, 99.0, -9.0],
            [7.0, 83.0, 9.0, -99.0],
            [-44.0, 44.0, -108.0, 108.0]
        ])
        
        Kyx_rational = Kxy_rational.T
        
        Kyy_rational = (1.0 / (40.0 * h)) * np.array([
            [148.0, -13.0, -189.0, 54.0],
            [-13.0, 148.0, 54.0, -189.0],
            [-189.0, 54.0, 432.0, -297.0],
            [54.0, -189.0, -297.0, 432.0]
        ])
        
        # Quadrature computation (10-point Gauss-Legendre is exact for polynomials up to degree 19)
        Mxx_quad = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                val, _ = fixed_quad(lambda x: lagrange_p2(x, h)[i] * lagrange_p2(x, h)[j], 0, h, n=10)
                Mxx_quad[i, j] = val
        assert np.max(np.abs(Mxx_quad - Mxx_rational)) < 1e-13 * h
        
        Myy_quad = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                val, _ = fixed_quad(lambda x: lagrange_p3(x, h)[i] * lagrange_p3(x, h)[j], 0, h, n=10)
                Myy_quad[i, j] = val
        assert np.max(np.abs(Myy_quad - Myy_rational)) < 1e-13 * h
        
        Kxy_quad = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                val, _ = fixed_quad(lambda x: lagrange_p2(x, h)[i] * d_lagrange_p3(x, h)[j], 0, h, n=10)
                Kxy_quad[i, j] = val
        assert np.max(np.abs(Kxy_quad - Kxy_rational)) < 1e-13
        
        Kyy_quad = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                val, _ = fixed_quad(lambda x: d_lagrange_p3(x, h)[i] * d_lagrange_p3(x, h)[j], 0, h, n=10)
                Kyy_quad[i, j] = val
        assert np.max(np.abs(Kyy_quad - Kyy_rational)) < 1e-12 / h
