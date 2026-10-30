"""Geometry and Meshing Module for EMUstack.

Provides:
- Built-in Netgen Meshing: Native Netgen SplineGeometry cell builder (unit_cell) with periodic boundaries and exact Bézier curved inclusions.
- Gmsh Template Import: Gmsh MSH 2.2 importer (import_gmsh_mesh) with 1-based AddPointIdentification.
- .mail file reader and importer (read_mail_file) converting legacy .mail to linear MSH 2.2.
- Periodic Boundary Coupling: algebraic constraint operator (bloch_constraint).
- Bloch product spaces (bloch_spaces) with mandatory assertions.
"""
import os
import tempfile
import shutil
import subprocess
import inspect
import importlib.util
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import netgen.geom2d as g2
from netgen.meshing import IdentificationType
from netgen.read_gmsh import ReadGmsh
import ngsolve as ngs
from ngsolve import Mesh, HCurl, H1
from ngsolve.fem import NODE_TYPE


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
    dy: float,
    inclusions: List[Dict[str, Any]],
    lc: float = 0.1,
    lc2: float = 2.0,
    nseg: int = 12,
    curve_order: int = 2,
    period_x: float = 1.0,
    **lc_kwargs: Optional[float],
) -> ngs.Mesh:
    """Build a periodic 2D unit cell mesh using native Netgen SplineGeometry.

    Parameters
    ----------
    dy : float
        Cell height / period_x (normalized cell height).
    inclusions : list of dict
        List of inclusion dictionaries specifying geometries (circle, ellipse,
        square, rectangle, ring, dimer). Each dictionary may also specify
        ``"tag": int`` (subdomain material tag) and/or ``"lc": float`` (a mesh
        scaling factor ``lc_n > 0`` that sets the inclusion's Netgen ``maxh``
        to ``lc / lc_n``, producing a finer mesh when ``lc_n > 1`` and a
        coarser mesh when ``0 < lc_n < 1``).
    lc : float, default=0.1
        Background mesh element size (domain tag 1, ``"bg"``) in normalized
        unit-cell coordinates.
    lc2 : float, default=2.0
        Default mesh scaling factor for inclusions (``maxh = lc / lc2``).
    nseg : int, default=12
        Default number of quadratic rational Bézier spline segments per closed
        curved perimeter (circle/ellipse/ring).
    curve_order : int, default=2
        Netgen geometric curvature order (>= 2).
    period_x : float, default=1.0
        Normalized cell width.
    **lc_kwargs : float, optional
        Arbitrary additional scaling factors ``lc3, lc4, ..., lc_n`` mapped by
        default to subdomain ``tag = 3, 4, ..., n`` (``maxh = lc / lc_n``) when
        an inclusion dictionary does not specify its own ``"lc"`` key.
    """
    bg_h = float(lc)
    if bg_h <= 0.0:
        raise ValueError(f"lc must be positive, got {lc}")

    lc_factors: Dict[int, Optional[float]] = {2: lc2}
    for k, v in lc_kwargs.items():
        if k.startswith("lc") and k[2:].isdigit() and int(k[2:]) >= 2:
            lc_factors[int(k[2:])] = None if v is None else float(v)
        else:
            raise TypeError(
                f"unit_cell() got an unexpected keyword argument '{k}'. "
                "Extra keyword arguments must be mesh scaling factors of the form 'lc<n>' (e.g. lc3, lc4, ...)."
            )

    def _factor_to_maxh(factor: Optional[float]) -> float:
        f = float(lc2) if factor is None else float(factor)
        if f <= 0.0:
            raise ValueError(f"lc scaling factor must be positive, got {f}")
        return bg_h / f

    def _default_inc_h(tag: int) -> float:
        return _factor_to_maxh(lc_factors.get(tag))

    geo = g2.SplineGeometry()
    L = period_x
    H = dy

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
        kind = inc.get("kind", "circle")
        tag = inc.get("tag", 2)
        tags.add(tag)
        c = inc.get("c", (L / 2.0, H / 2.0))
        if inc.get("lc") is not None:
            maxh_inc = _factor_to_maxh(inc["lc"])
        elif inc.get("maxh") is not None:
            maxh_inc = float(inc["maxh"])
        else:
            maxh_inc = _default_inc_h(tag)
        domain_maxh[tag] = min(domain_maxh.get(tag, maxh_inc), maxh_inc)
        nseg_inc = max(12, inc.get("nseg", nseg))

        if kind in ("circle", "ellipse"):
            r = inc.get("r", None)
            rx = inc.get("rx", r)
            ry = inc.get("ry", r)
            ang = inc.get("angle", 0.0)
            for a0, a1 in arc_segments(r=r, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, kind, c, a0, a1, r=r, rx=rx, ry=ry, angle=ang)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)

        elif kind in ("square", "rectangle"):
            w = inc.get("w", inc.get("s", 0.2))
            h = inc.get("h", inc.get("s", 0.2))
            pts_rect = [
                geo.AddPoint(c[0] - w / 2.0, c[1] - h / 2.0),
                geo.AddPoint(c[0] + w / 2.0, c[1] - h / 2.0),
                geo.AddPoint(c[0] + w / 2.0, c[1] + h / 2.0),
                geo.AddPoint(c[0] - w / 2.0, c[1] + h / 2.0),
            ]
            geo.Append(["line", pts_rect[0], pts_rect[1]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
            geo.Append(["line", pts_rect[1], pts_rect[2]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
            geo.Append(["line", pts_rect[2], pts_rect[3]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
            geo.Append(["line", pts_rect[3], pts_rect[0]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)

        elif kind == "ring":
            r_out = inc["r_out"]
            r_in = inc["r_in"]
            # Outer circle: leftdomain = tag, rightdomain = 1
            for a0, a1 in arc_segments(r=r_out, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, "circle", c, a0, a1, r=r_out)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=f"inc{tag}_out", maxh=maxh_inc)
            # Inner circle: leftdomain = 1, rightdomain = tag
            for a0, a1 in arc_segments(r=r_in, c=c, nseg=nseg_inc):
                pt0, pt_ctrl, pt1 = arc_control_points(geo, "circle", c, a0, a1, r=r_in)
                geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=1, rightdomain=tag, bc=f"inc{tag}_in", maxh=maxh_inc)

        elif kind == "dimer":
            # Two inclusions with a gap
            sub_kind = inc.get("sub_kind", "circle")
            r = inc.get("r", 0.1)
            gap = inc.get("gap", 0.05)
            shift_x = r + gap / 2.0
            c1 = (c[0] - shift_x, c[1])
            c2 = (c[0] + shift_x, c[1])
            for center in (c1, c2):
                if sub_kind in ("circle", "ellipse"):
                    for a0, a1 in arc_segments(r=r, c=center, nseg=nseg_inc):
                        pt0, pt_ctrl, pt1 = arc_control_points(geo, sub_kind, center, a0, a1, r=r)
                        geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
                else:
                    w = inc.get("w", inc.get("s", 2 * r))
                    h = inc.get("h", inc.get("s", 2 * r))
                    pts_rect = [
                        geo.AddPoint(center[0] - w / 2.0, center[1] - h / 2.0),
                        geo.AddPoint(center[0] + w / 2.0, center[1] - h / 2.0),
                        geo.AddPoint(center[0] + w / 2.0, center[1] + h / 2.0),
                        geo.AddPoint(center[0] - w / 2.0, center[1] + h / 2.0),
                    ]
                    geo.Append(["line", pts_rect[0], pts_rect[1]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
                    geo.Append(["line", pts_rect[1], pts_rect[2]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
                    geo.Append(["line", pts_rect[2], pts_rect[3]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)
                    geo.Append(["line", pts_rect[3], pts_rect[0]], leftdomain=tag, rightdomain=1, bc=f"inc{tag}", maxh=maxh_inc)

    geo.SetMaterial(1, "bg")
    geo.SetDomainMaxH(1, bg_h)
    for tag in tags:
        if tag != 1:
            geo.SetMaterial(tag, f"inc{tag}")
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
    with open(msh_path, "r", errors="ignore") as f:
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

    # Read with Netgen
    ngm = ReadGmsh(msh_path)
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
    with open(mail_path, "r") as f:
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
        '1 1 "top"',
        '1 2 "bottom"',
        '1 3 "left"',
        '1 4 "right"',
    ]
    for typ in unique_types:
        name = "bg" if typ == 1 else f"inc{typ}"
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
    cmd = [_gmsh_binary(), "-0", "-2", "-order", "2", "-v", "0", "-format", "msh2",
           "-o", tmp_msh, geo]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(tmp_msh):
        raise RuntimeError("gmsh failed on %s (exit %d):\n%s" % (geo, res.returncode, res.stderr))

    with open(tmp_msh) as f:
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
    with open(tmp_mail, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp_msh, msh)  # atomic: parallel runs may build the same mesh
    os.replace(tmp_mail, mail)
    return npt, len(tris)


def resolve_mesh_file_path(filename: str) -> str:
    """Locate a .geo, .py, .msh, or .mail file in CWD or the dedicated meshes folder."""
    if os.path.exists(filename):
        return os.path.abspath(filename)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(base_dir, "data", "meshes")
    candidate = os.path.join(mesh_dir, os.path.basename(filename))
    if os.path.exists(candidate):
        return candidate
    if not filename.endswith("_template.geo") and filename.endswith(".geo"):
        tmpl = os.path.basename(filename)[:-4] + "_template.geo"
        for path in (tmpl, os.path.join(mesh_dir, tmpl)):
            if os.path.exists(path):
                return os.path.abspath(path)
    raise FileNotFoundError(f"Geometry / mesh file not found: {filename}")


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
    with open(resolved_path, "r", errors="ignore") as f:
        geo_text = f.read()

    period_val = float(geo_params.get("period", geo_params.get("d_in_nm", period)))
    period_y_val = geo_params.get("period_y", geo_params.get("dy_in_nm", period_y))
    py = period_val if period_y_val is None else float(period_y_val)

    geo_text = geo_text.replace("d_in_nm = 0;", f"d_in_nm = {period_val:.16e};")
    geo_text = geo_text.replace("dy_in_nm = 0;", f"dy_in_nm = {py:.16e};")
    if "lc" not in geo_params:
        geo_text = geo_text.replace("lc = 0;", f"lc = {lc:.16e};")
    if "lc2" not in geo_params and lc2 != 1.0:
        geo_text = geo_text.replace("lc2 = lc/1;", f"lc2 = lc/{lc2:.16e};")
    if "lc3" not in geo_params and lc3 != 1.0:
        geo_text = geo_text.replace("lc3 = lc/1;", f"lc3 = lc/{lc3:.16e};")

    for key, val in geo_params.items():
        if key in ("period", "period_y", "d_in_nm", "dy_in_nm"):
            continue
        if key.startswith("lc") and key not in ("lc", "lc2", "lc3"):
            str_in = f"{key} = lc/1;"
            str_out = f"{key} = lc/{val:.16e};"
        elif key in ("lc", "lc2", "lc3"):
            if key == "lc":
                str_in = "lc = 0;"
                str_out = f"lc = {val:.16e};"
            else:
                str_in = f"{key} = lc/1;"
                str_out = f"{key} = lc/{val:.16e};"
        else:
            str_in = f"{key} = 0;"
            str_out = f"{key} = {val:.16e};"

        if str_in in geo_text:
            geo_text = geo_text.replace(str_in, str_out)
        else:
            raise KeyError(
                f"Parameter '{key}' in geo_params was not found in template '{geo_file}'. "
                f"Expected placeholder '{str_in}'."
            )

    geo_text = geo_text.replace('Physical Line(1) =', 'Physical Line("left") =')
    geo_text = geo_text.replace('Physical Line(2) =', 'Physical Line("bottom") =')
    geo_text = geo_text.replace('Physical Line(3) =', 'Physical Line("right") =')
    geo_text = geo_text.replace('Physical Line(4) =', 'Physical Line("top") =')
    geo_text = geo_text.replace('Physical Surface(1) =', 'Physical Surface("bg") =')
    geo_text = geo_text.replace('Physical Surface(2) =', 'Physical Surface("inc2") =')

    gmsh_bin = (
        os.environ.get("GMSH_BIN")
        or shutil.which("gmsh")
        or os.path.expanduser("~/.local/bin/gmsh")
        or os.path.expanduser("~/bin/gmsh")
    )
    if not gmsh_bin or not os.path.exists(gmsh_bin):
        raise EnvironmentError(
            "Gmsh binary not found. Please install Gmsh or use a native Netgen .py geometry template."
        )

    with tempfile.NamedTemporaryFile("w", suffix=".geo", delete=False) as f_geo:
        f_geo.write(geo_text)
        f_geo.flush()
        tmp_geo = f_geo.name

    tmp_msh = tmp_geo.replace(".geo", ".msh")
    cmd = [gmsh_bin, "-2", "-format", "msh2", "-order", "1", tmp_geo, "-o", tmp_msh]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    # Handle known aliases between .geo and .py conventions
    p_in = dict(geo_params)
    if "d_in_nm" in p_in and "period" not in p_in:
        p_in["period"] = p_in.pop("d_in_nm")
    else:
        p_in.pop("d_in_nm", None)

    if "dy_in_nm" in p_in and "period_y" not in p_in:
        p_in["period_y"] = p_in.pop("dy_in_nm")
    else:
        p_in.pop("dy_in_nm", None)

    if "diameter1" in p_in and "a1" not in p_in:
        p_in["a1"] = p_in.pop("diameter1")
    else:
        p_in.pop("diameter1", None)

    merged_params.update(p_in)

    sig = inspect.signature(builder)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_kw:
        valid_keys = set(sig.parameters.keys())
        for k in p_in.keys():
            if k not in valid_keys:
                raise TypeError(
                    f"Unexpected parameter '{k}' in geo_params for template '{py_file}'. "
                    f"Valid parameters: {sorted(valid_keys)}"
                )

    call_params = {}
    if has_var_kw:
        call_params = merged_params
    else:
        for k, v in merged_params.items():
            if k in sig.parameters:
                call_params[k] = v

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
                if k.startswith("lc") and k[2:].isdigit() and int(k[2:]) >= 3 and v is not None
            }
            return unit_cell(
                float(dy_val),
                inclusions,
                lc=float(merged_params["lc"]),
                lc2=float(merged_params["lc2"]),
                curve_order=curve_order,
                **extra_lc,
            )
    raise TypeError(
        f"Builder function in '{py_file}' returned unrecognized object type {type(result)}. "
        "Expected ngsolve.Mesh, netgen.geom2d.SplineGeometry, or (dy, inclusions) tuple."
    )
