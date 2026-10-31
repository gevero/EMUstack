"""Native Netgen unit-cell template for the C4-symmetric Gorkunov qBIC metasurface.

Reference: M. V. Gorkunov, A. A. Antonov, Yu. S. Kivshar,
"Metasurfaces with Maximum Chirality Empowered by Bound States in the Continuum",
Phys. Rev. Lett. 125, 093903 (2020), arXiv:2006.15884, Fig. 2.

This is the unit cell used in the paper. The square lattice (period ``a``) holds
four identical dimers of elliptical bars, related by 90-degree rotations about
the cell centre, so the lattice keeps the C4 symmetry that makes its normal-
incidence transmission independent of polarization. In the paper's drawing the
cell is a diamond; here it is rotated by 45 degrees so that its edges lie along
x and y, which leaves every spectrum unchanged.

Within each dimer the two bars (``length`` x ``width``, ``gap`` apart when
parallel) are rotated in opposite directions by ``theta_deg``, the paper's
theta: the angle between the two bars is ``2 * theta_deg``. Note that
``chiral_qbic_gorkunov_msh_template.py`` (one dimer per cell) instead uses
``theta_deg`` for the full angle between its bars.

The out-of-plane offset ``d`` of the paper is built from z-invariant layers:
``include_bar1`` / ``include_bar2`` keep only the bar rotated by +theta / -theta
in every dimer. For d = 0 a single layer with both bars is exact.

Parameters are in nm and degrees; the mesh is built in normalized [0,1]^2
coordinates.
"""
from typing import Dict, List, Optional

import numpy as np
import ngsolve as ngs
try:
    from EMUstack.backend import meshing
except ImportError:
    import meshing


def bar_inclusions(
    period: float = 480.0,
    theta_deg: float = 0.0,
    length: float = 220.0,
    width: float = 70.0,
    gap: float = 15.0,
    include_bar1: bool = True,
    include_bar2: bool = True,
    lc2: float = 2.0,
    **kwargs,
) -> List[Dict]:
    """Return the ``meshing.unit_cell`` inclusion dicts of all bars (normalized units).

    Also handy for drawing the bar outlines over field maps.
    """
    a = float(period)
    theta = np.deg2rad(float(theta_deg))
    rx = 0.5 * float(width) / a
    ry = 0.5 * float(length) / a
    half_sep = 0.5 * (float(width) + float(gap)) / a
    # Distance of each dimer centre from the cell centre: the four dimers sit in
    # the four quarters of the cell.
    s = 1.0 / (2.0 * np.sqrt(2.0))

    # One dimer, drawn with its bars along y, centred at (0, s) in the paper's frame.
    bars = []
    if include_bar1:
        bars.append((-half_sep, +theta))
    if include_bar2:
        bars.append((+half_sep, -theta))

    inclusions = []
    for k in range(4):
        # Rotate the dimer by k * 90 deg (C4 copies), then by -45 deg into the cell frame.
        phi = k * np.pi / 2.0 - np.pi / 4.0
        c, sn = np.cos(phi), np.sin(phi)
        for dx, ang in bars:
            x, y = dx, s
            centre = (0.5 + x * c - y * sn, 0.5 + x * sn + y * c)
            inclusions.append({"kind": "ellipse", "center": centre, "rx": rx, "ry": ry,
                               "angle": ang + phi, "tag": 2, "lc": lc2})
    return inclusions


def build_unit_cell(
    period: float = 480.0,
    period_y: Optional[float] = None,
    theta_deg: float = 0.0,
    length: float = 220.0,
    width: float = 70.0,
    gap: float = 15.0,
    include_bar1: bool = True,
    include_bar2: bool = True,
    lc: float = 0.12,
    lc2: float = 2.0,
    curve_order: int = 2,
    **kwargs,
) -> ngs.Mesh:
    """Build the normalized periodic unit-cell mesh (square lattice only)."""
    if period_y is not None and float(period_y) != float(period):
        raise ValueError("the C4 Gorkunov cell needs a square lattice (period_y == period)")
    inclusions = bar_inclusions(period=period, theta_deg=theta_deg, length=length,
                                width=width, gap=gap, include_bar1=include_bar1,
                                include_bar2=include_bar2, lc2=lc2)
    return meshing.unit_cell(inclusions, lc=lc, curve_order=curve_order)


def build_geometry(*args, **kwargs) -> ngs.Mesh:
    """Alias of build_unit_cell."""
    return build_unit_cell(*args, **kwargs)
