"""Geometry and Meshing Module for EMUstack.

Provides:
- Built-in Netgen Meshing: Native Netgen SplineGeometry cell builder (unit_cell) with periodic boundaries and exact Bézier curved inclusions.
- Gmsh Template Import: Gmsh MSH 2.2 importer (import_gmsh_mesh) with 1-based AddPointIdentification.
- .mail file reader and importer (read_mail_file) converting legacy .mail to linear MSH 2.2.
- Periodic Boundary Coupling: algebraic constraint operator (bloch_constraint).
- Bloch product spaces (bloch_spaces) with mandatory assertions.

Naming is shared by the three ways of describing a unit cell (see the
"Templates & naming" section of the docs):

- ``.geo`` (gmsh) and ``.py`` (netgen) templates are named
  ``<shape>_msh_template.<ext>`` and take the same parameter names, in nm:
  ``period``, ``period_y``, ``diameter1``, ``width``, ``thickness``, ...
  Angles are in radians unless the name ends in ``_deg``.
- ``unit_cell`` takes inclusion dicts in normalized coordinates with the
  matching keys: ``kind``, ``center``, ``radius``, ``width``, ``angle``, ...

Old file names, old parameter names and old dict keys keep working through the
alias tables below.
"""
import os
import re
import sys
import tempfile
import shutil
import subprocess
import inspect
import functools
import warnings
import importlib.util
from typing import List, Tuple, Dict, Any, Optional, Iterable

import numpy as np
import netgen.geom2d as g2
from netgen.meshing import IdentificationType
from netgen.read_gmsh import ReadGmsh
import ngsolve as ngs
from ngsolve import Mesh, HCurl, H1
from ngsolve.fem import NODE_TYPE


# ---------------------------------------------------------------------------
# Shared names
# ---------------------------------------------------------------------------

# gmsh Physical Line numbers used by every .geo template and .mail file.
BOUNDARY_NAMES = {1: "top", 2: "bottom", 3: "left", 4: "right"}
# Material names: domain 1 is the background, domain n >= 2 is inclusion n.
BACKGROUND_NAME = "bg"


def inclusion_name(tag: int) -> str:
    """Material name of inclusion domain ``tag`` (``2 -> "inc2"``)."""
    return f"inc{tag}"


TEMPLATE_SUFFIX = "_msh_template"

# Old template file names -> current names. Old names keep resolving everywhere.
LEGACY_TEMPLATE_NAMES = {
    "ring1_msh_template.geo": "ring_msh_template.geo",
    "rect1_msh_template.geo": "rectangle_msh_template.geo",
    "rect_shell1_msh_template.geo": "rectangle_shell_msh_template.geo",
    "cross1_msh_template.geo": "cross_msh_template.geo",
    "cross_shell1_msh_template.geo": "cross_shell_msh_template.geo",
    "dimer1_msh_template.geo": "dimer_msh_template.geo",
    "square_dimer1_msh_template.geo": "square_dimer_msh_template.geo",
    "square_shell_dimer1_msh_template.geo": "square_shell_dimer_msh_template.geo",
    "1_strip_msh_template.geo": "strip_msh_template.geo",
    "1_2strip_msh_template.geo": "double_strip_msh_template.geo",
    "chiral_qbic_template.py": "chiral_qbic_msh_template.py",
    "chiral_qbic_gorkunov.py": "chiral_qbic_gorkunov_msh_template.py",
}

# Template parameter names shared by all templates: canonical -> legacy spellings.
# Names that only one template uses declare their legacy spelling inside that
# template, on the placeholder line: ``width = 0; // nm (legacy: a1)``.
PARAM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "period": ("d_in_nm",),
    "period_y": ("dy_in_nm",),
    "thickness": ("t",),
    **{f"diameter{k}": (f"a{k}",) for k in range(1, 17)},
}


def _build_alias_groups(aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Tuple[str, ...]]:
    """Map every name (canonical or legacy) to its other spellings, canonical first."""
    groups: Dict[str, Tuple[str, ...]] = {}
    for canonical, legacy in aliases.items():
        names = (canonical,) + tuple(legacy)
        for name in names:
            groups[name] = tuple(n for n in names if n != name)
    return groups


_PARAM_EQUIVALENTS = _build_alias_groups(PARAM_ALIASES)


def param_equivalents(name: str) -> Tuple[str, ...]:
    """Other spellings of a template parameter (``"a1" -> ("diameter1",)``)."""
    return _PARAM_EQUIVALENTS.get(name, ())


# A .geo placeholder is ``name = 0;`` (a value) or ``name = lc/1;`` (a mesh
# refinement divisor), optionally followed by ``// ... (legacy: old_name)``.
_GEO_PLACEHOLDER = re.compile(
    r"(?<![\w.])(?P<name>[A-Za-z_]\w*)(?P<sep>[ \t]*=[ \t]*)(?P<val>0|lc/1)[ \t]*;"
    r"(?P<rest>[^\n]*)"
)
_GEO_LEGACY_TAG = re.compile(r"\(legacy:\s*([^)]*)\)")


def geo_placeholders(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Find the placeholders of a .geo template.

    Returns ``(placeholders, legacy)``: ``placeholders`` maps each name to
    ``"value"`` (``name = 0;``) or ``"lc"`` (``name = lc/1;``), and ``legacy``
    maps each legacy name declared with ``(legacy: ...)`` to its current name.
    """
    placeholders: Dict[str, str] = {}
    legacy: Dict[str, str] = {}
    for m in _GEO_PLACEHOLDER.finditer(text):
        name = m.group("name")
        placeholders.setdefault(name, "lc" if m.group("val") == "lc/1" else "value")
        tag = _GEO_LEGACY_TAG.search(m.group("rest"))
        if tag:
            for old in re.split(r"[,\s]+", tag.group(1).strip()):
                if old:
                    legacy[old] = name
    return placeholders, legacy


def _geo_target(key: str, placeholders: Dict[str, str], legacy: Dict[str, str]) -> Optional[str]:
    """Placeholder that ``key`` fills: exact name first, then its other spellings."""
    if key in placeholders:
        return key
    for alt in param_equivalents(key):
        if alt in placeholders:
            return alt
    for alt in (key,) + param_equivalents(key):
        if alt in legacy:
            return legacy[alt]
    return None


def fill_geo_template(
    text: str,
    params: Dict[str, Any],
    fmt: str = "%f",
    strict: bool = True,
    optional: Iterable[str] = (),
    template: str = "template",
) -> str:
    """Fill the placeholders of a .geo template.

    Each key fills the placeholder of the same name. If the template has no
    such placeholder, the key's other spellings are tried: the shared ones in
    ``PARAM_ALIASES`` (``d_in_nm`` <-> ``period``, ``a1`` <-> ``diameter1``, ...)
    and the ``(legacy: ...)`` tags of the template itself. So old keys fill
    renamed templates, and new keys fill old custom templates.

    ``name = 0;`` becomes ``name = <value>;`` and ``name = lc/1;`` becomes
    ``name = lc/<value>;``. Numbers are written with ``fmt``; strings are
    written as they are. When two keys fill the same placeholder, the later
    one wins.

    A key that matches no placeholder raises ``KeyError`` when ``strict`` is
    set, unless it is listed in ``optional``; otherwise it is ignored.
    """
    placeholders, legacy = geo_placeholders(text)
    optional = set(optional)
    values: Dict[str, Any] = {}
    for key, value in params.items():
        target = _geo_target(key, placeholders, legacy)
        if target is None:
            if strict and key not in optional:
                expected = f"{key} = lc/1;" if key.startswith("lc") and key[2:].isdigit() else f"{key} = 0;"
                raise KeyError(
                    f"Parameter '{key}' in geo_params was not found in template '{template}'. "
                    f"Expected placeholder '{expected}'."
                )
            continue
        values[target] = value

    def _sub(m: "re.Match[str]") -> str:
        name = m.group("name")
        if name not in values:
            return m.group(0)
        value = values[name]
        number = value if isinstance(value, str) else fmt % value
        if m.group("val") == "lc/1":
            number = "lc/" + number
        return f"{name}{m.group('sep')}{number};{m.group('rest')}"

    return _GEO_PLACEHOLDER.sub(_sub, text)


def legacy_params(**renames: str):
    """Let a template builder accept old keyword names after a rename.

    ``@legacy_params(a1="diameter1")`` above ``def build_unit_cell(..., diameter1=None, ...)``
    turns a call with ``a1=300`` into ``diameter1=300``. Passing both names raises
    ``TypeError``. The renames are stored on the function as ``legacy_params``, so
    ``mesh_from_py`` accepts the old names in ``geo_params`` too. It only renames;
    values are not converted. See "Renaming a .py Template Parameter Safely" in the
    meshing guide.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{fn.__name__}() got both '{old}' and its new name '{new}'")
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)
        wrapper.legacy_params = dict(renames)
        return wrapper
    return decorator


# unit_cell inclusion keys: legacy key -> canonical key.
INCLUSION_KEY_ALIASES = {
    "shape": "kind",
    "c": "center",
    "r": "radius",
    "w": "width",
    "h": "height",
    "s": "side",
    "rot": "angle_deg",
    "r_out": "radius_out",
    "r_in": "radius_in",
}
INCLUSION_KEYS = frozenset({
    "kind", "center", "radius", "rx", "ry", "width", "height", "side",
    "angle", "angle_deg", "radius_out", "radius_in", "gap", "sub_kind",
    "tag", "lc", "maxh", "nseg",
})


def _normalize_inclusion(inc: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``inc`` with canonical keys. A canonical key wins over its alias."""
    out = {k: v for k, v in inc.items() if k in INCLUSION_KEYS}
    for old, new in INCLUSION_KEY_ALIASES.items():
        if old in inc and new not in out:
            out[new] = inc[old]
    unknown = sorted(k for k in inc if k not in INCLUSION_KEYS and k not in INCLUSION_KEY_ALIASES)
    if unknown:
        warnings.warn(
            f"unit_cell(): ignoring unknown inclusion key(s) {unknown}. "
            f"Valid keys: {sorted(INCLUSION_KEYS)}.",
            UserWarning,
            stacklevel=3,
        )
    return out


def arc_segments(r: float, c: Tuple[float, float], nseg: int = 12):
    """Generate nseg angle intervals (a0, a1) for a full circle/ellipse."""
    dtheta = 2.0 * np.pi / nseg
    for i in range(nseg):
        yield i * dtheta, (i + 1) * dtheta


def arc_control_points(
    geo: g2.SplineGeometry,
    kind: str,
    c: Tuple[float, float],
    a0: float,
    a1: float,
    r: Optional[float] = None,
    rx: Optional[float] = None,
    ry: Optional[float] = None,
    angle: float = 0.0
) -> Tuple[int, int, int]:
    """Calculate Bézier control points for quadratic rational arc."""
    dtheta = a1 - a0
    mid_theta = (a0 + a1) / 2.0
    sec_half = 1.0 / np.cos(dtheta / 2.0)

    if kind == "circle":
        r_val = r if r is not None else rx
        ax = r_val
        by = r_val
    elif kind == "ellipse":
        ax = rx if rx is not None else r
        by = ry if ry is not None else r
    else:
        raise ValueError(f"Unknown curved inclusion kind: {kind}")

    x0, y0 = ax * np.cos(a0), by * np.sin(a0)
    xc, yc = ax * sec_half * np.cos(mid_theta), by * sec_half * np.sin(mid_theta)
    x1, y1 = ax * np.cos(a1), by * np.sin(a1)

    if angle != 0.0:
        ca, sa = np.cos(angle), np.sin(angle)
        x0, y0 = x0 * ca - y0 * sa, x0 * sa + y0 * ca
        xc, yc = xc * ca - yc * sa, xc * sa + yc * ca
        x1, y1 = x1 * ca - y1 * sa, x1 * sa + y1 * ca

    p0 = (c[0] + x0, c[1] + y0)
    p_ctrl = (c[0] + xc, c[1] + yc)
    p1 = (c[0] + x1, c[1] + y1)

    pt0 = geo.AddPoint(*p0)
    pt_ctrl = geo.AddPoint(*p_ctrl)
    pt1 = geo.AddPoint(*p1)
    return pt0, pt_ctrl, pt1


def unit_cell(
    inclusions: Any,
    dx: Any = 1.0,
    dy: float = 1.0,
    lc: float = 0.1,
    curve_order: int = 2,
    nseg: int = 24,
    **lc_kwargs: Any,
) -> ngs.Mesh:
    """Build a periodic 2D unit cell mesh using native Netgen SplineGeometry.

    All lengths are normalized to the period along x. The old call
    ``unit_cell(dy, inclusions, period_x=...)`` still works.

    Parameters
    ----------
    inclusions : list of dict
        One dict per inclusion. Keys (old spellings in brackets still work):

        - ``"kind"`` [``"shape"``]: ``"circle"`` (default), ``"ellipse"``,
          ``"square"``, ``"rectangle"``, ``"ring"`` or ``"dimer"``
        - ``"center"`` [``"c"``]: ``(x, y)``, default ``(dx/2, dy/2)``
        - circle/ellipse: ``"radius"`` [``"r"``] or ``"rx"``, ``"ry"``
        - square/rectangle: ``"side"`` [``"s"``] or ``"width"``, ``"height"``
          [``"w"``, ``"h"``]
        - ring: ``"radius_out"``, ``"radius_in"`` [``"r_out"``, ``"r_in"``]
        - dimer: ``"radius"``, ``"gap"`` and ``"sub_kind"`` (``"circle"`` or a
          square with ``"side"``/``"width"``/``"height"``)
        - ``"angle"``: rotation in radians, or ``"angle_deg"`` [``"rot"``] in degrees
        - ``"tag"``: material tag ``n >= 2`` (domain ``"inc<n>"``, default 2)
        - ``"lc"``: refinement factor, ``maxh = lc / lc_n`` (finer when > 1),
          or ``"maxh"`` directly; ``"nseg"``: arcs per curved boundary
    dx, dy : float, default 1.0
        Unit-cell width and height (``dy = period_y / period``).
    lc : float, default 0.1
        Background mesh size (domain ``"bg"``).
    curve_order : int, default 2
        Netgen geometric curvature order (>= 2).
    nseg : int, default 24
        Default number of arcs per curved boundary (at least 12).
    **lc_kwargs : float
        ``lc2, lc3, ..., lc<n>``: refinement factor of inclusions with
        ``tag = n`` that do not set their own ``"lc"``.
    """
    # Old positional form unit_cell(dy, inclusions, ...)
    if isinstance(inclusions, (int, float)) and isinstance(dx, list):
        dy = float(inclusions)
        inclusions = dx
        dx = 1.0

    if "period_x" in lc_kwargs:
        dx = float(lc_kwargs.pop("period_x"))
    default_nseg = int(nseg)

    bg_h = float(lc)
    if bg_h <= 0.0:
        raise ValueError(f"lc must be positive, got {lc}")

    lc_factors: Dict[int, Optional[float]] = {2: 2.0}
    for k, v in lc_kwargs.items():
        if k.startswith("lc") and k[2:].isdigit() and int(k[2:]) >= 2:
            lc_factors[int(k[2:])] = None if v is None else float(v)
        else:
            raise TypeError(
                f"unit_cell() got an unexpected keyword argument '{k}'. "
                "Extra keyword arguments must be mesh scaling factors of the form 'lc<n>' (e.g. lc2, lc3, ...)."
            )

    default_lc2 = lc_factors.get(2)
    if default_lc2 is None:
        default_lc2 = 2.0

    def _factor_to_maxh(factor: Optional[float]) -> float:
        f = float(default_lc2) if factor is None else float(factor)
        if f <= 0.0:
            raise ValueError(f"lc scaling factor must be positive, got {f}")
        return bg_h / f

    def _default_inc_h(tag: int) -> float:
        return _factor_to_maxh(lc_factors.get(tag))

    geo = g2.SplineGeometry()
    L = float(dx)
    H = float(dy)

    # Cell boundary points
    P = [geo.AddPoint(*q) for q in [(0, 0), (L, 0), (L, H), (0, H)]]

    # Periodic boundary edges
    # left is primary, right is copy (Ident 1: x)
    i_l = geo.Append(["line", P[0], P[3]], leftdomain=0, rightdomain=1, bc="left", maxh=bg_h)
    geo.Append(["line", P[1], P[2]], leftdomain=1, rightdomain=0, bc="right", maxh=bg_h, copy=i_l)

    # bottom is primary, top is copy (Ident 2: y)
    i_b = geo.Append(["line", P[0], P[1]], leftdomain=1, rightdomain=0, bc="bottom", maxh=bg_h)
    geo.Append(["line", P[3], P[2]], leftdomain=0, rightdomain=1, bc="top", maxh=bg_h, copy=i_b)

    # Inclusions
    tags = {1}
    domain_maxh: Dict[int, float] = {}
    for inc in inclusions:
        inc = _normalize_inclusion(inc)
        kind = inc.get("kind", "circle")
        tag = inc.get("tag", 2)
        tags.add(tag)
        name = inclusion_name(tag)
        c = inc.get("center", (L / 2.0, H / 2.0))
        if inc.get("lc") is not None:
            maxh_inc = _factor_to_maxh(inc["lc"])
        elif inc.get("maxh") is not None:
            maxh_inc = float(inc["maxh"])
        else:
            maxh_inc = _default_inc_h(tag)
        domain_maxh[tag] = min(domain_maxh.get(tag, maxh_inc), maxh_inc)
        nseg_inc = max(12, int(inc.get("nseg", default_nseg)))

        if kind in ("circle", "ellipse"):
            r = inc.get("radius", None)
            rx = inc.get("rx", r)
            ry = inc.get("ry", r)
            if "angle" in inc:
                ang = float(inc["angle"])
            else:
                ang = float(inc.get("angle_deg", 0.0)) * np.pi / 180.0
            for a0, a1 in arc_segments(r=r, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, kind, c, a0, a1, r=r, rx=rx, ry=ry, angle=ang)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)

        elif kind in ("square", "rectangle"):
            side_default = inc.get("side", 0.2)
            w = float(inc.get("width", side_default))
            h = float(inc.get("height", side_default))
            pts_rect = [
                geo.AddPoint(c[0] - w / 2.0, c[1] - h / 2.0),
                geo.AddPoint(c[0] + w / 2.0, c[1] - h / 2.0),
                geo.AddPoint(c[0] + w / 2.0, c[1] + h / 2.0),
                geo.AddPoint(c[0] - w / 2.0, c[1] + h / 2.0),
            ]
            geo.Append(["line", pts_rect[0], pts_rect[1]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
            geo.Append(["line", pts_rect[1], pts_rect[2]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
            geo.Append(["line", pts_rect[2], pts_rect[3]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
            geo.Append(["line", pts_rect[3], pts_rect[0]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)

        elif kind == "ring":
            r_out = inc["radius_out"]
            r_in = inc["radius_in"]
            # Outer circle: leftdomain = tag, rightdomain = 1
            for a0, a1 in arc_segments(r=r_out, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, "circle", c, a0, a1, r=r_out)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=f"{name}_out", maxh=maxh_inc)
            # Inner circle: leftdomain = 1, rightdomain = tag
            for a0, a1 in arc_segments(r=r_in, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, "circle", c, a0, a1, r=r_in)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=1, rightdomain=tag, bc=f"{name}_in", maxh=maxh_inc)

        elif kind == "dimer":
            # Two inclusions with a gap
            sub_kind = inc.get("sub_kind", "circle")
            r = inc.get("radius", 0.1)
            gap = inc.get("gap", 0.05)
            shift_x = r + gap / 2.0
            c1 = (c[0] - shift_x, c[1])
            c2 = (c[0] + shift_x, c[1])
            for center in (c1, c2):
                if sub_kind in ("circle", "ellipse"):
                    for a0, a1 in arc_segments(r=r, c=center, nseg=nseg_inc):
                        pt0, pt_ctrl, pt1 = arc_control_points(geo, sub_kind, center, a0, a1, r=r)
                        geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
                else:
                    side_default = inc.get("side", 2 * r)
                    w = float(inc.get("width", side_default))
                    h = float(inc.get("height", side_default))
                    pts_rect = [
                        geo.AddPoint(center[0] - w / 2.0, center[1] - h / 2.0),
                        geo.AddPoint(center[0] + w / 2.0, center[1] - h / 2.0),
                        geo.AddPoint(center[0] + w / 2.0, center[1] + h / 2.0),
                        geo.AddPoint(center[0] - w / 2.0, center[1] + h / 2.0),
                    ]
                    geo.Append(["line", pts_rect[0], pts_rect[1]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
                    geo.Append(["line", pts_rect[1], pts_rect[2]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
                    geo.Append(["line", pts_rect[2], pts_rect[3]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)
                    geo.Append(["line", pts_rect[3], pts_rect[0]], leftdomain=tag, rightdomain=1, bc=name, maxh=maxh_inc)

    geo.SetMaterial(1, BACKGROUND_NAME)
    geo.SetDomainMaxH(1, bg_h)
    for tag in tags:
        if tag != 1:
            geo.SetMaterial(tag, inclusion_name(tag))
            geo.SetDomainMaxH(tag, domain_maxh.get(tag, _default_inc_h(tag)))

    global_maxh = max([bg_h] + list(domain_maxh.values()))
    ngm = geo.GenerateMesh(maxh=global_maxh)
    ngm.Curve(curve_order)
    mesh = Mesh(ngm)

    # Geometric and periodic boundary sanity assertions
    assert ngm.GetNrIdentifications() == 4, "periodic identifications missing"
    assert ngm.GetCurveOrder() >= 2, "curved geometry not activated"
    assert len(mesh.GetPeriodicNodePairs(NODE_TYPE.VERTEX)) > 0, "no periodic vertex pairs found"
    cell_area = ngs.Integrate(1.0, mesh)
    expected_area = L * H
    assert abs(cell_area - expected_area) < 1e-11 * expected_area, f"cell area wrong: got {cell_area}, expected {expected_area}"
    return mesh


def bloch_spaces(
    mesh: ngs.Mesh,
    dy: float,
    order: int,
    kx: float,
    ky: float,
    period_x: float = 1.0
) -> ngs.comp.FESpace:
    """Create periodic HCurl * H1 product Bloch space with phase convention exp(-i k.r)."""
    fes = HCurl(mesh, order=order, complex=True) * H1(mesh, order=order + 1, complex=True)
    phase = [np.exp(-1j * kx * period_x), np.exp(-1j * ky * dy)]
    space = ngs.Periodic(fes, phase=phase)
    assert space.FreeDofs().NumSet() < fes.ndof, "Periodic() left the space unconstrained"
    return space


# Dimension of each MSH 2.2 element type (points, lines, surfaces, volumes).
_MSH2_ELEMENT_DIM = {15: 0, 1: 1, 8: 1, 26: 1, 27: 1, 28: 1,
                     2: 2, 3: 2, 9: 2, 10: 2, 16: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2,
                     4: 3, 5: 3, 6: 3, 7: 3, 11: 3, 12: 3, 13: 3, 14: 3, 17: 3, 18: 3, 19: 3}


def _prepare_msh_for_netgen(msh_text: str) -> Optional[str]:
    """Make an MSH 2.2 mesh safe for netgen's ReadGmsh.

    ReadGmsh needs a name for every physical group, and it numbers domains and
    boundaries by gmsh's elementary tags; with more elementary surfaces than
    boundaries it reads out of bounds (random crashes). So, as ``mail_to_msh22``
    does, every element gets its physical tag as elementary tag, and unnamed
    groups get names: boundary lines after the cell side they lie on
    (``BOUNDARY_NAMES``, or ``bnd<N>`` inside the cell), surfaces ``bg`` (tag 1)
    or ``inc<N>``. Returns None when the mesh needs no change.
    """
    lines = msh_text.split("\n")

    def section(name):
        try:
            start = lines.index("$" + name)
        except ValueError:
            return None, None
        return start, lines.index("$End" + name, start)

    names: Dict[Tuple[int, int], str] = {}
    ps, pe = section("PhysicalNames")
    if ps is not None:
        for line in lines[ps + 2:pe]:
            dim, tag, name = line.split(maxsplit=2)
            names[(int(dim), int(tag))] = name.strip()

    ns, ne_ = section("Nodes")
    es, ee = section("Elements")
    if ns is None or es is None:
        return None
    xy = {}
    for line in lines[ns + 2:ne_]:
        parts = line.split()
        xy[int(parts[0])] = (float(parts[1]), float(parts[2]))

    used: Dict[Tuple[int, int], set] = {}
    retagged = False
    for i in range(es + 2, ee):
        parts = [int(v) for v in lines[i].split()]
        dim = _MSH2_ELEMENT_DIM.get(parts[1])
        if dim is None or parts[2] == 0:
            continue
        used.setdefault((dim, parts[3]), set()).update(parts[3 + parts[2]:])
        if parts[2] >= 2 and parts[4] != parts[3]:
            parts[4] = parts[3]
            lines[i] = " ".join(str(v) for v in parts)
            retagged = True

    missing = [key for key in used if key not in names]
    if not missing and not retagged:
        return None

    pts = np.array(list(xy.values()))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    tol = 1e-6 * max(hi - lo)
    sides = {"left": (0, lo[0]), "right": (0, hi[0]), "bottom": (1, lo[1]), "top": (1, hi[1])}

    for dim, tag in sorted(missing):
        if dim == 1:
            nodes = np.array([xy[n] for n in used[(dim, tag)]])
            on_side = {side: np.sum(np.abs(nodes[:, axis] - value) < tol) for side, (axis, value) in sides.items()}
            side = max(on_side, key=on_side.get)
            name = side if on_side[side] > len(nodes) / 2 else f"bnd{tag}"
        elif dim == 2:
            name = BACKGROUND_NAME if tag == 1 else inclusion_name(tag)
        else:
            name = f"group{dim}_{tag}"
        names[(dim, tag)] = f'"{name}"'
    if not missing:
        return "\n".join(lines)

    block = ["$PhysicalNames", str(len(names))]
    block += [f"{dim} {tag} {name}" for (dim, tag), name in sorted(names.items())]
    block.append("$EndPhysicalNames")
    if ps is None:
        _, fe = section("MeshFormat")
        lines[fe + 1:fe + 1] = block
    else:
        lines[ps:pe + 1] = block
    return "\n".join(lines)


def import_gmsh_mesh(
    msh_path: str,
    period_x: float = 1.0,
    period_y: Optional[float] = None,
    method: str = "IIA",
    curve_order: int = 2
) -> ngs.Mesh:
    """Import a legacy Gmsh MSH 2.2 mesh and configure 2D periodic identifications.

    Parameters:
    - msh_path: path to .msh file (must be MSH 2.2, linear elements)
    - period_x: cell width in x
    - period_y: cell height in y (if None, determined from mesh coordinates)
    - method: 'IIA' (native 1-based AddPointIdentification) or 'IIB' (unconstrained for bloch_constraint)
    - curve_order: curve order to apply (>= 2)
    """
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"MSH file not found: {msh_path}")

    # Inspect file format and element types
    with open(msh_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(500)
    if "$MeshFormat" in head:
        lines = head.splitlines()
        for idx_l, line in enumerate(lines):
            if line.strip() == "$MeshFormat" and idx_l + 1 < len(lines):
                version_str = lines[idx_l + 1].split()[0]
                if version_str.startswith("4"):
                    raise ValueError(
                        f"MSH 4.x format detected ({version_str}) in {msh_path}. "
                        "Netgen requires legacy Gmsh MSH 2.2 format (-format msh2 -order 1)."
                    )

    # Read with Netgen, which needs a name for every physical group
    with open(msh_path, "r", encoding="utf-8", errors="ignore") as f:
        named = _prepare_msh_for_netgen(f.read())
    if named is None:
        ngm = ReadGmsh(msh_path)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".msh", delete=False) as f:
            f.write(named)
            named_path = f.name
        try:
            ngm = ReadGmsh(named_path)
        finally:
            os.remove(named_path)
    # Ensure mesh dimension is 2D for planar unit cells
    ngm.dim = 2
    if curve_order >= 2:
        ngm.Curve(curve_order)

    if method == "IIA":
        # Identify periodic boundaries geometrically
        P = np.array([[p[0], p[1], p[2]] for p in ngm.Points()])
        x_min, x_max = P[:, 0].min(), P[:, 0].max()
        y_min, y_max = P[:, 1].min(), P[:, 1].max()

        tol = 1e-5 * max(x_max - x_min, y_max - y_min)
        lo = [i for i, p in enumerate(P) if abs(p[0] - x_min) < tol]
        ri = [i for i, p in enumerate(P) if abs(p[0] - x_max) < tol]
        bo = [i for i, p in enumerate(P) if abs(p[1] - y_min) < tol]
        to = [i for i, p in enumerate(P) if abs(p[1] - y_max) < tol]

        assert len(lo) == len(ri), f"Non-matching boundary node count on x-faces: {len(lo)} != {len(ri)}"
        assert len(bo) == len(to), f"Non-matching boundary node count on y-faces: {len(bo)} != {len(to)}"

        x_pairs = list(zip(sorted(lo, key=lambda i: P[i][1]), sorted(ri, key=lambda i: P[i][1])))
        y_pairs = list(zip(sorted(bo, key=lambda i: P[i][0]), sorted(to, key=lambda i: P[i][0])))
        # Paired nodes must face each other, or the Bloch conditions would join wrong points.
        dy_max = max((abs(P[a][1] - P[b][1]) for a, b in x_pairs), default=0.0)
        dx_max = max((abs(P[a][0] - P[b][0]) for a, b in y_pairs), default=0.0)
        assert dy_max < tol and dx_max < tol, (
            "Boundary nodes on opposite faces of the unit cell do not match (offset up to "
            f"{max(dx_max, dy_max):.3g}); the mesh is not periodic.")

        # 1-based identification numbers: 1 for x-pairs, 2 for y-pairs
        for p1, p2 in x_pairs:
            ngm.AddPointIdentification(p1 + 1, p2 + 1, 1, IdentificationType.PERIODIC)
        for p1, p2 in y_pairs:
            ngm.AddPointIdentification(p1 + 1, p2 + 1, 2, IdentificationType.PERIODIC)

    mesh = Mesh(ngm)
    assert mesh.dim == 2, f"Mesh dimension must be 2, got {mesh.dim}"
    return mesh


def parse_mail_data(mail_path: str):
    """Parse .mail file into raw numpy tables (table_nod, type_el, x_arr, npt, nel)."""
    with open(mail_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    npt, nel = int(header[0]), int(header[1])

    pts = np.zeros((2, npt), dtype=float)
    idfn = np.zeros(npt, dtype=int)
    for i in range(1, npt + 1):
        parts = lines[i].split()
        pts[0, i - 1] = float(parts[1])
        pts[1, i - 1] = float(parts[2])
        idfn[i - 1] = int(parts[3])

    tokens = []
    for line in lines[npt + 1:]:
        tokens.extend(line.split())

    assert len(tokens) == nel * 8, f"Expected {nel * 8} element tokens in {mail_path}, got {len(tokens)}"
    table_nod = np.zeros((6, nel), dtype=int)
    type_el = np.zeros(nel, dtype=int)

    for e in range(nel):
        idx = int(tokens[e * 8])
        nodes = [int(x) for x in tokens[e * 8 + 1 : e * 8 + 7]]
        typ = int(tokens[e * 8 + 7])
        table_nod[:, e] = nodes
        type_el[e] = typ

    return npt, nel, pts, idfn, table_nod, type_el


def mail_to_msh22(mail_path: str) -> str:
    """Convert .mail file to Gmsh MSH 2.2 linear triangles string."""
    npt, nel, pts, idfn, table_nod, type_el = parse_mail_data(mail_path)

    # Determine unique boundary line segments from triangle corners
    bnd_dict = {}
    for e in range(nel):
        corners = table_nod[0:3, e]
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[0])
        ]
        for a, b in edges:
            ia = idfn[a - 1]
            ib = idfn[b - 1]
            if ia != 0 and ib != 0:
                tag = ia if ia == ib else (ia if ia != 0 else ib)
                pair = (min(a, b), max(a, b))
                if pair not in bnd_dict:
                    bnd_dict[pair] = tag

    unique_types = sorted(set(type_el))
    out = [
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat",
        f"$PhysicalNames\n{4 + len(unique_types)}",
        *(f'1 {num} "{side}"' for num, side in BOUNDARY_NAMES.items()),
    ]
    for typ in unique_types:
        name = BACKGROUND_NAME if typ == 1 else inclusion_name(typ)
        out.append(f'2 {typ} "{name}"')
    out.append("$EndPhysicalNames")

    out.append(f"$Nodes\n{npt}")
    for i in range(npt):
        out.append(f"{i + 1} {pts[0, i]:.16e} {pts[1, i]:.16e} 0.0")
    out.append("$EndNodes")

    total_elements = len(bnd_dict) + nel
    out.append(f"$Elements\n{total_elements}")
    elm_id = 1
    for (a, b), tag in bnd_dict.items():
        out.append(f"{elm_id} 1 2 {tag} {tag} {a} {b}")
        elm_id += 1
    for e in range(nel):
        typ = type_el[e]
        c = table_nod[0:3, e]
        out.append(f"{elm_id} 2 2 {typ} {typ} {c[0]} {c[1]} {c[2]}")
        elm_id += 1
    out.append("$EndElements\n")
    return "\n".join(out)


def read_mail_file(mail_path: str, method: str = "IIA") -> Tuple[ngs.Mesh, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Read a .mail file, convert to linear MSH 2.2, import into Netgen/NGSolve, and return mesh + arrays."""
    npt, nel, pts, idfn, table_nod, type_el = parse_mail_data(mail_path)
    msh_content = mail_to_msh22(mail_path)

    with tempfile.NamedTemporaryFile("w", suffix=".msh", delete=False) as f:
        f.write(msh_content)
        tmp_path = f.name

    try:
        mesh = import_gmsh_mesh(tmp_path, method=method)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return mesh, table_nod, type_el, pts, npt, nel


def _gmsh_binary() -> str:
    for cand in (os.environ.get("GMSH_BIN"), shutil.which("gmsh"),
                 os.path.expanduser("~/.local/bin/gmsh"), os.path.expanduser("~/bin/gmsh")):
        if cand and os.path.exists(cand):
            return cand
    raise EnvironmentError(
        "Gmsh binary not found. Install Gmsh, set GMSH_BIN, or use mesher='netgen'."
    )


# Seconds gmsh may take for one mesh. gmsh never returns on some invalid geometries
# (e.g. lines crossing an inclusion), so without a limit the caller would hang forever.
gmsh_timeout = float(os.environ.get("EMUSTACK_GMSH_TIMEOUT", "600"))


# gmsh is a plain child process: if the Python process is killed (SIGKILL/SIGTERM),
# gmsh would keep running (forever on invalid geometry). On Linux, start gmsh through
# a tiny launcher that sets PR_SET_PDEATHSIG, so the kernel kills gmsh when its parent
# dies. A preexec_fn would do the same but is unsafe in threaded programs, so the
# launcher runs in a fresh interpreter and then execs gmsh.
_PDEATHSIG_LAUNCHER = (
    "import ctypes, os, signal, sys\n"
    "try:\n"
    "    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, SIGKILL\n"
    "except Exception:\n"
    "    pass\n"
    "# Exit if the parent died before prctl took effect; a legitimate parent may be PID 1.\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(1)\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)


def _gmsh_command(args: List[str]) -> List[str]:
    """gmsh command line, with a parent-death-signal launcher on Linux."""
    binary = _gmsh_binary()
    if sys.platform.startswith("linux") and sys.executable and os.path.exists(sys.executable):
        return [sys.executable, "-c", _PDEATHSIG_LAUNCHER, str(os.getpid()), binary] + list(args)
    if sys.platform == "win32" and binary.lower().endswith((".bat", ".cmd")):
        # The gmsh pip wheel installs gmsh.bat, which runs `python gmsh` through cmd.exe.
        # Killing cmd.exe on timeout would leave that python running and holding our pipes,
        # so run the script next to it with this interpreter, as one process.
        script = os.path.splitext(binary)[0]
        if os.path.isfile(script) and sys.executable:
            return [sys.executable, script] + list(args)
    return [binary] + list(args)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and, on Windows, every process it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.kill()


def _run_gmsh(args: List[str], geo: str) -> subprocess.CompletedProcess:
    """Run gmsh with `args`; stop it and raise if it takes longer than `gmsh_timeout`."""
    cmd = _gmsh_command(args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace")
    try:
        out, err = proc.communicate(timeout=gmsh_timeout)
    except BaseException as exc:  # timeout, Ctrl-C, ...: never leave gmsh running
        _kill_tree(proc)
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass  # a grandchild still holds the pipes; gmsh itself is dead
        if not isinstance(exc, subprocess.TimeoutExpired):
            raise
        raise RuntimeError(
            "gmsh did not finish meshing %s within %g s and was stopped. The geometry is "
            "probably invalid, e.g. inclusions that overlap or cross other lines: check the "
            "template parameters. For a very fine mesh, raise the limit with "
            "meshing.gmsh_timeout or EMUSTACK_GMSH_TIMEOUT." % (geo, gmsh_timeout)) from None
    res = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if res.returncode != 0 and "cannot open shared object file" in res.stderr:
        raise RuntimeError(
            "gmsh could not start: a system library is missing.\n%s\n"
            "The gmsh pip package needs OpenGL and X11 client libraries, which minimal Linux "
            "systems (containers, headless servers) lack. On Debian/Ubuntu:\n"
            "    sudo apt-get install %s\n"
            "or use mesher='netgen'." % (res.stderr.strip().splitlines()[-1], GMSH_SYSTEM_LIBS))
    return res


# Debian/Ubuntu packages the gmsh pip wheel needs on a minimal Linux system.
GMSH_SYSTEM_LIBS = "libglu1-mesa libgl1 libgomp1 libxcursor1 libxft2 libxinerama1"


_gmsh_works = None


def gmsh_available() -> bool:
    """True if a gmsh binary is found and starts (checked once per process)."""
    global _gmsh_works
    if _gmsh_works is None:
        try:
            res = subprocess.run(_gmsh_command(["-version"]), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=60)
            _gmsh_works = res.returncode == 0
        except (EnvironmentError, subprocess.TimeoutExpired):
            _gmsh_works = False
    return _gmsh_works


def conv_gmsh(root: str) -> Tuple[int, int]:
    """Mesh `root`.geo with gmsh and write `root`.mail, as legacy EMUstack's conv_gmsh.f does.

    Same gmsh command line as legacy (6-node triangles, MSH 2.2 set by the templates).
    The .mail holds every node with the physical tag of the boundary line it lies on
    (0 inside), and every triangle with its physical surface tag. Unlike legacy, nodes
    keep gmsh's order instead of being renumbered for bandwidth; the mesh is the same.
    Returns (number of nodes, number of triangles).
    """
    geo, msh, mail = root + ".geo", root + ".msh", root + ".mail"
    tmp_msh = msh + ".%d.tmp" % os.getpid()
    try:
        res = _run_gmsh(["-0", "-2", "-order", "2", "-v", "0", "-format", "msh2", "-o", tmp_msh, geo], geo)
    except RuntimeError:
        if os.path.exists(tmp_msh):
            os.remove(tmp_msh)
        raise
    if res.returncode != 0 or not os.path.exists(tmp_msh):
        raise RuntimeError("gmsh failed on %s (exit %d):\n%s" % (geo, res.returncode, res.stderr))

    with open(tmp_msh, encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes")
    npt = int(lines[i + 1])
    index = {}  # gmsh node id -> 1-based .mail index
    xy = np.zeros((npt, 2))
    for k in range(npt):
        parts = lines[i + 2 + k].split()
        index[int(parts[0])] = k + 1
        xy[k] = float(parts[1]), float(parts[2])
    i = lines.index("$Elements")
    idfn = np.zeros(npt, dtype=int)
    tris, tri_tags = [], []
    for k in range(int(lines[i + 1])):
        parts = [int(p) for p in lines[i + 2 + k].split()]
        typ, ntags = parts[1], parts[2]
        tag, nodes = parts[3], [index[n] for n in parts[3 + ntags:]]
        if typ == 8:  # 3-node line: its nodes get the line's physical tag
            for n in nodes:
                idfn[n - 1] = tag
        elif typ == 9:  # 6-node triangle
            tris.append(nodes)
            tri_tags.append(tag)
        elif typ != 15:  # points are ignored, anything else is unexpected
            raise ValueError("conv_gmsh: unexpected gmsh element type %d in %s" % (typ, msh))

    out = ["%9d %9d" % (npt, len(tris))]
    out += ["  %7d %24.15E %24.15E %5d" % (k + 1, xy[k, 0], xy[k, 1], idfn[k]) for k in range(npt)]
    out += ["%d %s %d" % (e + 1, " ".join(str(n) for n in tris[e]), tri_tags[e])
            for e in range(len(tris))]
    tmp_mail = mail + ".%d.tmp" % os.getpid()
    with open(tmp_mail, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp_msh, msh)  # atomic: parallel runs may build the same mesh
    os.replace(tmp_mail, mail)
    return npt, len(tris)


def resolve_mesh_file_path(filename: str) -> str:
    """Find a .geo, .py, .msh or .mail file.

    Looks in the current directory first, then in ``backend/data/meshes``.
    A template can be given by its full name (``ring_msh_template.geo``), its
    old name (``ring1_msh_template.geo``) or its short name (``ring.geo``).
    """
    if os.path.exists(filename):
        return os.path.abspath(filename)
    base = os.path.basename(filename)
    candidate = os.path.join(TEMPLATE_DIR, base)
    if os.path.exists(candidate):
        return candidate
    for name in _template_names(base)[1:]:
        for path in (name, os.path.join(TEMPLATE_DIR, name)):
            if os.path.exists(path):
                return os.path.abspath(path)
    raise FileNotFoundError(f"Geometry / mesh file not found: {filename}")


# Folder of the mesh templates shipped with EMUstack (inside the installed package).
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meshes")


def _template_names(base: str) -> List[str]:
    """`base` followed by the names it may stand for: long form and old (legacy) names."""
    stem, ext = os.path.splitext(base)
    names = [base]
    if ext in (".geo", ".py") and not stem.endswith(TEMPLATE_SUFFIX):
        # short name: ring.geo -> ring_msh_template.geo; ring_msh.geo -> ring_msh_template.geo
        long_stem = stem + "_template" if stem.endswith("_msh") else stem + TEMPLATE_SUFFIX
        names.append(long_stem + ext)
    for name in list(names):
        if name in LEGACY_TEMPLATE_NAMES:
            names.append(LEGACY_TEMPLATE_NAMES[name])
    return names


def list_templates() -> List[str]:
    """File names of the mesh templates shipped with EMUstack.

    ``.geo`` templates are meshed with gmsh, ``.py`` templates with netgen. Pass a name
    (or its short form, e.g. ``'ellipse.geo'``) as ``NanoStruct(geo_file=...)``.
    """
    return sorted(f for f in os.listdir(TEMPLATE_DIR)
                  if f.endswith(TEMPLATE_SUFFIX + ".geo") or f.endswith(TEMPLATE_SUFFIX + ".py"))


def copy_template(name: str, dest: str = ".", overwrite: bool = False) -> str:
    """Copy a shipped template to `dest` (a folder or a file name) and return the new path.

    Use it to start your own geometry from a shipped one. Files in the working directory are
    found before the shipped templates, so after ``copy_template('ellipse.geo')``,
    ``geo_file='ellipse_msh_template.geo'`` meshes your edited copy.
    """
    for candidate in _template_names(os.path.basename(name)):
        src = os.path.join(TEMPLATE_DIR, candidate)
        if os.path.exists(src):
            break
    else:
        raise FileNotFoundError(
            f"No shipped template named {name!r}; see meshing.list_templates().")
    target = os.path.join(dest, os.path.basename(src)) if os.path.isdir(dest) else dest
    if os.path.exists(target) and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it.")
    shutil.copyfile(src, target)
    return os.path.abspath(target)


def mesh_from_geo(
    geo_file: str,
    geo_params: Optional[Dict[str, Any]] = None,
    period: float = 1.0,
    period_y: Optional[float] = None,
    lc: float = 0.1,
    lc2: float = 1.0,
    lc3: float = 1.0,
    curve_order: int = 2
) -> ngs.Mesh:
    """Universal Gmsh .geo loader with parameter substitution and MSH 2.2 ingestion."""
    if geo_params is None:
        geo_params = {}

    resolved_path = resolve_mesh_file_path(geo_file)
    with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
        geo_text = f.read()

    period_val = float(geo_params.get("period", geo_params.get("d_in_nm", period)))
    period_y_val = geo_params.get("period_y", geo_params.get("dy_in_nm", period_y))
    py = period_val if period_y_val is None else float(period_y_val)

    # Arguments fill the template when present; geo_params must all be used.
    base: Dict[str, Any] = {"period": period_val, "period_y": py}
    if "lc" not in geo_params:
        base["lc"] = lc
    if "lc2" not in geo_params and lc2 != 1.0:
        base["lc2"] = lc2
    if "lc3" not in geo_params and lc3 != 1.0:
        base["lc3"] = lc3
    user = {k: v for k, v in geo_params.items() if k not in ("period", "period_y", "d_in_nm", "dy_in_nm")}
    geo_text = fill_geo_template(geo_text, {**base, **user}, fmt="%.16e",
                                 optional=base.keys(), template=geo_file)

    # Name the boundaries and the first two domains for netgen.
    for num, side in BOUNDARY_NAMES.items():
        geo_text = geo_text.replace(f"Physical Line({num}) =", f'Physical Line("{side}") =')
    geo_text = geo_text.replace("Physical Surface(1) =", f'Physical Surface("{BACKGROUND_NAME}") =')
    geo_text = geo_text.replace("Physical Surface(2) =", f'Physical Surface("{inclusion_name(2)}") =')

    _gmsh_binary()  # fail early with a clear message if gmsh is missing

    with tempfile.NamedTemporaryFile("w", suffix=".geo", delete=False) as f_geo:
        f_geo.write(geo_text)
        f_geo.flush()
        tmp_geo = f_geo.name

    tmp_msh = tmp_geo.replace(".geo", ".msh")
    try:
        res = _run_gmsh(["-2", "-format", "msh2", "-order", "1", tmp_geo, "-o", tmp_msh], geo_file)
        if res.returncode != 0:
            raise RuntimeError(f"Gmsh meshing failed with exit code {res.returncode}:\n{res.stderr}")
        mesh = import_gmsh_mesh(tmp_msh, curve_order=curve_order)
    finally:
        if os.path.exists(tmp_geo):
            os.remove(tmp_geo)
        if os.path.exists(tmp_msh):
            os.remove(tmp_msh)

    return mesh


def mesh_from_py(
    py_file: str,
    geo_params: Optional[Dict[str, Any]] = None,
    period: float = 1.0,
    period_y: Optional[float] = None,
    lc: float = 0.1,
    lc2: float = 1.0,
    lc3: float = 1.0,
    curve_order: int = 2,
    **lc_kwargs: float,
) -> ngs.Mesh:
    """Universal Netgen Python .py geometry loader with signature validation."""
    if geo_params is None:
        geo_params = {}

    resolved_path = resolve_mesh_file_path(py_file)
    spec = importlib.util.spec_from_file_location("custom_netgen_geo", resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python geometry module: {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    builder = getattr(mod, "build_unit_cell", None) or getattr(mod, "build_geometry", None)
    if builder is None:
        raise AttributeError(
            f"Python geometry template '{py_file}' must define 'build_unit_cell' or 'build_geometry'."
        )

    py = period if period_y is None else period_y
    merged_params: Dict[str, Any] = {
        "period": period,
        "period_y": py,
        "lc": lc,
        "lc2": lc2,
        "lc3": lc3,
    }
    for k, v in lc_kwargs.items():
        if k.startswith("lc") and k[2:].isdigit() and int(k[2:]) >= 2:
            merged_params[k] = v
        else:
            raise TypeError(f"mesh_from_py() got an unexpected keyword argument '{k}'")

    # Give each parameter to the builder under a name it accepts: its own name,
    # the new name of a @legacy_params keyword, or another spelling from
    # PARAM_ALIASES (d_in_nm -> period, diameter1 <-> a1, ...).
    sig = inspect.signature(builder)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    explicit = {
        name for name, p in sig.parameters.items()
        if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    renamed = getattr(builder, "legacy_params", {})

    def _target(key: str) -> Optional[str]:
        if key in explicit:
            return key
        if key in renamed:
            return renamed[key]
        for alt in param_equivalents(key):
            if alt in explicit:
                return alt
        return None

    def _ordered(params: Dict[str, Any]):
        # A key the builder takes as is wins over another spelling of it.
        return sorted(params.items(), key=lambda kv: kv[0] in explicit)

    p_in = dict(geo_params)
    if not has_var_kw:
        for k in p_in:
            if _target(k) is None:
                raise TypeError(
                    f"Unexpected parameter '{k}' in geo_params for template '{py_file}'. "
                    f"Valid parameters: {sorted(explicit | set(renamed))}"
                )

    call_params: Dict[str, Any] = {}
    for params in (merged_params, p_in):
        for k, v in _ordered(params):
            target = _target(k)
            if target is not None:
                call_params[target] = v
            elif has_var_kw:
                # Into **kwargs, under every spelling, as older builders read old names.
                call_params[k] = v
                for alt in param_equivalents(k):
                    call_params.setdefault(alt, v)
    merged_params.update(p_in)

    result = builder(**call_params)
    if isinstance(result, ngs.Mesh):
        return result
    elif isinstance(result, g2.SplineGeometry):
        ngm = result.GenerateMesh(maxh=float(merged_params["lc"]))
        if curve_order >= 2:
            ngm.Curve(curve_order)
        return Mesh(ngm)
    elif isinstance(result, (tuple, list)):
        if len(result) == 2 and isinstance(result[0], (int, float)):
            dy_val, inclusions = result
            extra_lc = {
                k: float(v)
                for k, v in merged_params.items()
                if k.startswith("lc") and k[2:].isdigit() and int(k[2:]) >= 2 and v is not None
            }
            return unit_cell(
                inclusions,
                dy=float(dy_val),
                lc=float(merged_params["lc"]),
                curve_order=curve_order,
                **extra_lc,
            )
    raise TypeError(
        f"Builder function in '{py_file}' returned unrecognized object type {type(result)}. "
        "Expected ngsolve.Mesh, netgen.geom2d.SplineGeometry, or (dy, inclusions) tuple."
    )
