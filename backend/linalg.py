"""Linear algebra helpers preserving the matrix boundary.

Under NumPy >= 2.0, np.mat was removed. np.asmatrix / np.matrix remain.
This module provides standard helpers for the Stack / mode_calcs matrix boundary.
"""
import numpy as np


def mat(x, dtype=None):
    """Convert input to np.matrix."""
    if dtype is not None:
        return np.asmatrix(x, dtype=dtype)
    return np.asmatrix(x)


def h(m):
    """Hermitian conjugate of matrix or ndarray."""
    if isinstance(m, np.matrix):
        return m.H
    return np.conjugate(np.swapaxes(m, -1, -2))


def inv_solve(A, B):
    """Solve A X = B."""
    return np.linalg.solve(A, B)
