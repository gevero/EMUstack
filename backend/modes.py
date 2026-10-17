"""Mode set container (PORTING_PLAN §4.2).

Returned by both 1D and 2D solvers, providing a unified container for
eigenvalues, overlap matrices, self-flux, and diagnostic metadata.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


class FEMConvergenceError(RuntimeError):
    """Raised when eigensolver fails to converge sufficient physical modes."""
    pass


@dataclass
class BlochModes:
    """Bloch mode set returned by fem_1d and fem_2d solvers."""
    zeta: np.ndarray                       # (nbm,) complex
    J: np.ndarray                          # (2*npw, nbm) complex
    J_dag: np.ndarray                      # (nbm, 2*npw) complex
    mode_flux: np.ndarray                  # (nbm, nbm) complex F
    norm_res: float                        # ||L - I|| after biorthogonalization
    pairs: np.ndarray                      # (nbm,) int
    cluster: np.ndarray                    # (nbm,) int
    solver_info: Dict[str, Any] = field(default_factory=dict)
    sol1: Optional[np.ndarray] = None      # 2D: (6,13,nbm,nel) | 1D: (3,3,nbm,nel)
    mode_pol: Optional[np.ndarray] = None  # (4, nbm)
    _J_1d: Optional[np.ndarray] = None
    _J_dag_1d: Optional[np.ndarray] = None

    def __iter__(self):
        j1 = self._J_1d if self._J_1d is not None else self.J
        j1_dag = self._J_dag_1d if self._J_dag_1d is not None else self.J_dag
        return iter((self.zeta, j1, j1_dag, self.J, self.J_dag, self.sol1))
