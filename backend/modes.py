"""Mode set container.

Returned by both 1D and 2D solvers, providing a unified container for
eigenvalues, overlap matrices, self-flux, and diagnostic metadata.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import warnings
import numpy as np


class FEMConvergenceError(RuntimeError):
    """Raised when eigensolver fails to converge sufficient physical modes."""
    pass


def warn_if_degenerate(zeta, num_BMs, rtol=1e-6):
    """Warn when the num_BMs cut falls inside a degenerate group of Bloch modes.

    At high symmetry, modes come in (near-)degenerate pairs or groups. Keeping only
    part of a group breaks the lattice symmetry and can change symmetry-protected
    quantities (e.g. |T_RCP - T_LCP|). Needs the first dropped eigenvalue, so the
    solvers request one eigenpair more than num_BMs before truncating.
    """
    if num_BMs < 1 or len(zeta) <= num_BMs:
        return
    kept, dropped = zeta[num_BMs - 1], zeta[num_BMs]
    diff = abs(kept - dropped)
    scale = max(abs(kept), abs(dropped))
    if scale == 0.0:
        degenerate = diff == 0.0
    else:
        degenerate = diff <= rtol * scale
    if degenerate:
        warnings.warn(
            "num_BMs=%d cuts through a degenerate Bloch-mode group: the last kept mode "
            "(zeta=%s) and the first dropped mode (zeta=%s) differ by %.3g (relative %.1e). "
            "Increase num_BMs so the whole group is kept; otherwise symmetry-protected "
            "quantities such as |T_RCP - T_LCP| can change."
            % (num_BMs, format(kept, '.6g'), format(dropped, '.6g'), diff,
               diff / scale if scale else 0.0),
            stacklevel=3,
        )


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
