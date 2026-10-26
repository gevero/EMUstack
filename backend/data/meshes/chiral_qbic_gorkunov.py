"""Native Netgen unit-cell template for the Gorkunov chiral qBIC design.

Reference: M. V. Gorkunov, A. A. Antonov, Yu. S. Kivshar,
"Metasurfaces with Maximum Chirality Empowered by Bound States in the Continuum",
Phys. Rev. Lett. 125, 093903 (2020).

The unit cell contains a dimer of identical dielectric bars (elliptical cross
section l x w), separated by an in-plane gap, rotated by +/- theta/2 and offset
vertically by d.  The vertical offset is represented by splitting the structure
into three z-invariant layers (see the notebook): a 10 nm layer with bar 1 only,
a 90 nm layer with both bars, and a 10 nm layer with bar 2 only.

Parameters are in nm; the mesh is built in normalized [0,1]^2 coordinates.
"""
from typing import Optional, Dict, Any
import numpy as np
import ngsolve as ngs
try:
    from EMUstack.backend import meshing
except ImportError:
    import meshing


def build_unit_cell(
    period: float = 480.0,
    period_y: Optional[float] = None,
    theta: float = 3.5,
    d: float = 10.0,
    l: float = 220.0,
    w: float = 70.0,
    gap: float = 15.0,
    include_bar1: bool = True,
    include_bar2: bool = True,
    rot1: Optional[float] = None,
    rot2: Optional[float] = None,
    lc: float = 0.08,
    lc2: float = 2.0,
    lc3: float = 1.0,
    curve_order: int = 2,
    **kwargs
) -> ngs.Mesh:
    """Build the normalized periodic unit-cell mesh for one z-slice."""
    py = period if period_y is None else period_y
    dy = float(py) / float(period)
    a = float(period)

    theta_rad = np.deg2rad(float(theta))
    if rot1 is None:
        rot1 = +0.5 * theta_rad
    if rot2 is None:
        rot2 = -0.5 * theta_rad

    # normalized geometry
    rx = 0.5 * float(w) / a
    ry = 0.5 * float(l) / a
    sep = (float(w) + float(gap)) / a
    c1 = (0.5 - 0.5 * sep, 0.5 * dy)
    c2 = (0.5 + 0.5 * sep, 0.5 * dy)

    inclusions = []
    if include_bar1:
        inclusions.append({"kind": "ellipse", "c": c1, "rx": rx, "ry": ry,
                           "angle": float(rot1), "tag": 2})
    if include_bar2:
        inclusions.append({"kind": "ellipse", "c": c2, "rx": rx, "ry": ry,
                           "angle": float(rot2), "tag": 2})

    inc_h = lc / lc2 if lc2 > 1.0 else lc
    refine = meshing.Refine(bg=lc, inc=inc_h)
    return meshing.unit_cell(dy, inclusions, refine=refine,
                             curve_order=curve_order, period_x=1.0)


def build_geometry(*args, **kwargs) -> ngs.Mesh:
    return build_unit_cell(*args, **kwargs)
