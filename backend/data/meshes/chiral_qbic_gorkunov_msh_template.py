"""Native Netgen unit-cell template for the Gorkunov chiral qBIC design.

Reference: M. V. Gorkunov, A. A. Antonov, Yu. S. Kivshar,
"Metasurfaces with Maximum Chirality Empowered by Bound States in the Continuum",
Phys. Rev. Lett. 125, 093903 (2020).

The unit cell contains a dimer of identical dielectric bars (elliptical cross
section length x width), separated by an in-plane gap, rotated by
+/- theta_deg/2 and offset vertically by z_offset.  The vertical offset is
represented by splitting the structure into three z-invariant layers (see the
notebook): a 10 nm layer with bar 1 only, a 90 nm layer with both bars, and a
10 nm layer with bar 2 only.

Parameters are in nm and degrees; the mesh is built in normalized [0,1]^2
coordinates. angle1/angle2 override the bar rotations, in radians.
The old names l, w, d, theta, rot1 and rot2 still work.
"""
from typing import Optional, Dict, Any
import numpy as np
import ngsolve as ngs
try:
    from EMUstack.backend import meshing
except ImportError:
    import meshing


@meshing.legacy_params(theta="theta_deg", d="z_offset", l="length", w="width",
                        rot1="angle1", rot2="angle2")
def build_unit_cell(
    period: float = 480.0,
    period_y: Optional[float] = None,
    theta_deg: float = 3.5,
    z_offset: float = 10.0,
    length: float = 220.0,
    width: float = 70.0,
    gap: float = 15.0,
    include_bar1: bool = True,
    include_bar2: bool = True,
    angle1: Optional[float] = None,
    angle2: Optional[float] = None,
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

    theta_rad = np.deg2rad(float(theta_deg))
    if angle1 is None:
        angle1 = +0.5 * theta_rad
    if angle2 is None:
        angle2 = -0.5 * theta_rad

    # normalized geometry
    rx = 0.5 * float(width) / a
    ry = 0.5 * float(length) / a
    sep = (float(width) + float(gap)) / a
    c1 = (0.5 - 0.5 * sep, 0.5 * dy)
    c2 = (0.5 + 0.5 * sep, 0.5 * dy)

    inclusions = []
    if include_bar1:
        inclusions.append({"kind": "ellipse", "center": c1, "rx": rx, "ry": ry,
                           "angle": float(angle1), "tag": 2, "lc": lc2})
    if include_bar2:
        inclusions.append({"kind": "ellipse", "center": c2, "rx": rx, "ry": ry,
                           "angle": float(angle2), "tag": 2, "lc": lc2})

    return meshing.unit_cell(inclusions, dy=dy, lc=lc, curve_order=curve_order)


def build_geometry(*args, **kwargs) -> ngs.Mesh:
    """Old name of build_unit_cell."""
    return build_unit_cell(*args, **kwargs)
