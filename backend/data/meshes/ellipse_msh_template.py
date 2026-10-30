"""Netgen 2D SplineGeometry Template for an Elliptical Inclusion.

Companion to ellipse_msh_template.geo.
Demonstrates clean, in-memory Netgen CSG meshing with exact quadratic
rational Bézier splines (curve order >= 2) and rotation angle transformation.
"""
from typing import Optional
import numpy as np
import ngsolve as ngs
import meshing


def build_unit_cell(
    period: float = 400.0,
    period_y: Optional[float] = None,
    a1: Optional[float] = None,
    ellipticity: Optional[float] = None,
    angle: Optional[float] = None,
    lc: float = 0.12,
    lc2: float = 5.0,
    lc3: float = 1.0,
) -> ngs.Mesh:
    """Builds a 2D periodic unit cell with a rotated elliptical inclusion in Netgen.

    Parameters:
    - period: unit cell lattice period along x in nm
    - period_y: unit cell lattice period along y in nm (defaults to period)
    - a1: major diameter in nm
    - ellipticity: (rx - ry) / rx
    - angle: rotation angle in radians
    - lc: background mesh size (relative to period)
    - lc2: boundary/inclusion mesh scaling factor (maxh = lc / lc2)
    - lc3: optional additional mesh scaling factor
    """
    py = period if period_y is None else period_y
    dy = float(py) / float(period)

    if a1 is not None:
        rx = float(a1) / (2.0 * float(period))
    else:
        rx = 150.0 / float(period)

    if ellipticity is not None:
        ry = rx * (1.0 - float(ellipticity))
    else:
        ry = rx * 0.5

    if angle is not None:
        ang_rad = float(angle)
    else:
        ang_rad = 0.0

    center = (0.5, dy / 2.0)

    inclusions = [{
        "kind": "ellipse",
        "center": center,
        "rx": rx,
        "ry": ry,
        "angle": ang_rad,
        "tag": 2,
        "lc": lc2,
    }]

    return meshing.unit_cell(inclusions, dy=dy, lc=lc)
