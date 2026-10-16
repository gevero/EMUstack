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
    a1: float = 300.0,
    ellipticity: float = 0.5,
    angle: float = 0.0,
    lc: float = 0.12,
    lc2: float = 5.0,
    lc3: float = 1.0
) -> ngs.Mesh:
    """Builds a 2D periodic unit cell with a rotated elliptical inclusion in Netgen.
    
    Parameters:
    - period: unit cell lattice period along x in nm
    - period_y: unit cell lattice period along y in nm (defaults to period)
    - a1: major diameter in nm
    - ellipticity: (rx - ry) / rx, where ry = rx * (1 - ellipticity)
    - angle: rotation angle in radians
    - lc: background mesh size (relative to period)
    - lc2: boundary refinement factor (mesh size = lc / lc2)
    - lc3: center refinement factor
    """
    py = period if period_y is None else period_y
    dy = float(py) / float(period)
    rx = float(a1) / (2.0 * float(period))
    ry = rx * (1.0 - float(ellipticity))
    center = (0.5, dy / 2.0)
    
    inclusions = [{
        "kind": "ellipse",
        "c": center,
        "rx": rx,
        "ry": ry,
        "angle": float(angle),
        "tag": 2
    }]
    
    inc_h = lc / lc2 if lc2 > 1.0 else lc
    refine = meshing.Refine(bg=lc, inc=inc_h)
    return meshing.unit_cell(dy, inclusions, refine=refine, curve_order=2, period_x=1.0)
