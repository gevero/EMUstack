"""Netgen 2D SplineGeometry Template for an Elliptical Inclusion.

Companion to ellipse_msh_template.geo, with the same parameter names.
Demonstrates clean, in-memory Netgen CSG meshing with exact quadratic
rational Bézier splines (curve order >= 2) and rotation angle transformation.
"""
from typing import Optional
import ngsolve as ngs
try:
    from EMUstack.backend import meshing
except ImportError:
    import meshing


@meshing.legacy_params(a1="diameter1")
def build_unit_cell(
    period: float = 400.0,
    period_y: Optional[float] = None,
    diameter1: Optional[float] = None,
    ellipticity: Optional[float] = None,
    angle: Optional[float] = None,
    lc: float = 0.12,
    lc2: float = 5.0,
    lc3: float = 1.0,
) -> ngs.Mesh:
    """Builds a 2D periodic unit cell with a rotated elliptical inclusion in Netgen.

    Parameters (lengths in nm):
    - period: unit cell lattice period along x
    - period_y: unit cell lattice period along y (defaults to period)
    - diameter1: major diameter (old name: a1), default 300 nm
    - ellipticity: (rx - ry) / rx, default 0.5
    - angle: rotation angle in radians
    - lc: background mesh size (relative to period)
    - lc2: boundary/inclusion mesh scaling factor (maxh = lc / lc2)
    - lc3: optional additional mesh scaling factor
    """
    py = period if period_y is None else period_y
    dy = float(py) / float(period)

    if diameter1 is not None:
        rx = float(diameter1) / (2.0 * float(period))
    else:
        rx = 150.0 / float(period)

    if ellipticity is not None:
        ry = rx * (1.0 - float(ellipticity))
    else:
        ry = rx * 0.5

    ang_rad = float(angle) if angle is not None else 0.0

    inclusions = [{
        "kind": "ellipse",
        "center": (0.5, dy / 2.0),
        "rx": rx,
        "ry": ry,
        "angle": ang_rad,
        "tag": 2,
        "lc": lc2,
    }]

    return meshing.unit_cell(inclusions, dy=dy, lc=lc)


def build_geometry(*args, **kwargs) -> ngs.Mesh:
    """Old name of build_unit_cell."""
    return build_unit_cell(*args, **kwargs)
