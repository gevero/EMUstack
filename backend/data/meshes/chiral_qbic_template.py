"""Netgen 2D SplineGeometry Template for a Chiral quasi-BIC (qBIC) Unit Cell.

Implements a broken-symmetry dielectric dimer metasurface unit cell
supporting symmetry-protected BICs transitioning into high-Q chiral quasi-BICs
under non-zero geometric asymmetry alpha.

References:
- Koshelev, Kivshar et al., Science 367, 288-292 (2020)
- Gorkunov et al., Phys. Rev. Lett. 125, 093901 (2020)
- Overvig et al., Phys. Rev. Lett. 126, 073001 (2021)
"""
from typing import Optional, Dict, Any, List
import numpy as np
import ngsolve as ngs
import meshing


def build_unit_cell(
    period: float = 700.0,
    period_y: Optional[float] = None,
    alpha: float = 0.15,
    theta0: float = 0.25,
    rx: float = 0.09,
    ry: float = 0.28,
    gap: float = 0.36,
    lc: float = 0.10,
    lc2: float = 2.5,
    lc3: float = 1.0,
    curve_order: int = 2,
    **kwargs
) -> ngs.Mesh:
    """Build a periodic 2D unit cell mesh for a chiral qBIC metasurface.
    
    Parameters:
    - period: unit cell lattice period along x in nm (default 700 nm)
    - period_y: unit cell lattice period along y in nm (defaults to period)
    - alpha: asymmetry parameter breaking in-plane mirror symmetry (tilt asymmetry).
             alpha = 0 gives symmetric BIC; alpha != 0 gives chiral qBIC.
    - theta0: base tilt angle in radians (~14.3 degrees)
    - rx: semi-minor axis of resonators (normalized to period)
    - ry: semi-major axis of resonators (normalized to period)
    - gap: center-to-center distance between the two dimer resonators along x
    - lc: background mesh size (relative to period)
    - lc2: inclusion refinement factor
    - lc3: center refinement factor
    - curve_order: Netgen polynomial curvature order (>= 2)
    """
    py = period if period_y is None else period_y
    dy = float(py) / float(period)

    # Resonator 1: tilted at +theta0 + alpha
    # Resonator 2: tilted at -theta0
    c1 = (0.5 - gap / 2.0, dy / 2.0)
    c2 = (0.5 + gap / 2.0, dy / 2.0)

    angle1 = float(theta0 + alpha)
    angle2 = float(-theta0)

    inclusions = [
        {
            "kind": "ellipse",
            "c": c1,
            "rx": float(rx),
            "ry": float(ry),
            "angle": angle1,
            "tag": 2,
            "lc": lc2,
        },
        {
            "kind": "ellipse",
            "c": c2,
            "rx": float(rx),
            "ry": float(ry),
            "angle": angle2,
            "tag": 2,
            "lc": lc2,
        }
    ]

    return meshing.unit_cell(
        dy, inclusions, lc=lc,
        curve_order=curve_order, period_x=1.0
    )


def build_geometry(*args, **kwargs) -> ngs.Mesh:
    """Alias for build_unit_cell to support both naming conventions."""
    return build_unit_cell(*args, **kwargs)
