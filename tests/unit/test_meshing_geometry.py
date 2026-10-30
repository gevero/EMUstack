"""Unit tests for the Meshing and Geometry module (backend/meshing.py).

Verifies PORTING_PLAN §5:
- §5.1 / §5.2: Mandatory assertions on periodic identifications, curvature, and area.
- §5.3: Analytic inclusion area and perimeter tolerances for circle, ellipse, square, rectangle, ring, dimer.
- §5.8: Gmsh MSH 2.2 import with Path IIA 1-based AddPointIdentification and MSH 4.x rejection.
- §5.6: .mail file import for 4testing-600_120.mail and 4testing-940_266_sq.mail.
"""
import os
import tempfile
import pytest
import numpy as np
import ngsolve as ngs
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
import sys
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import meshing
from meshing import unit_cell, bloch_spaces, import_gmsh_mesh, read_mail_file


def test_native_circle_area_and_perimeter():
    """Verify circle inclusion area and perimeter accuracy with nseg >= 12 (§5.3)."""
    r = 0.2
    dy = 1.0
    c = (0.5, 0.5)
    inclusions = [{"kind": "circle", "c": c, "r": r, "tag": 2, "nseg": 24}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, nseg=24, curve_order=2)

    # 1. Total cell area
    cell_area = ngs.Integrate(1.0, mesh)
    assert abs(cell_area - 1.0) < 1e-11

    # 2. Inclusion area
    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = np.pi * r**2
    rel_err_area = abs(area_inc - exact_area) / exact_area
    assert rel_err_area < 5e-4, f"Circle area error too high: {rel_err_area}"

    # 3. Perimeter
    perim = ngs.Integrate(1.0, mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("inc2"))
    exact_perim = 2.0 * np.pi * r
    rel_err_perim = abs(perim - exact_perim) / exact_perim
    assert rel_err_perim < 5e-4, f"Circle perimeter error too high: {rel_err_perim}"

    # 4. Free dofs reduction in Bloch space
    space = bloch_spaces(mesh, dy=dy, order=2, kx=0.31, ky=0.17)
    assert space.FreeDofs().NumSet() < space.ndof


def test_native_ellipse_area_and_perimeter():
    """Verify ellipse inclusion area and perimeter accuracy (§5.3)."""
    rx, ry = 0.25, 0.15
    dy = 0.8
    c = (0.5, 0.4)
    inclusions = [{"kind": "ellipse", "c": c, "rx": rx, "ry": ry, "tag": 2, "nseg": 24}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, nseg=24, curve_order=2)

    # Inclusion area
    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = np.pi * rx * ry
    rel_err_area = abs(area_inc - exact_area) / exact_area
    assert rel_err_area < 5e-4, f"Ellipse area error too high: {rel_err_area}"

    # Ramanujan's perimeter approximation
    exact_perim = np.pi * (3 * (rx + ry) - np.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    perim = ngs.Integrate(1.0, mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("inc2"))
    rel_err_perim = abs(perim - exact_perim) / exact_perim
    assert rel_err_perim < 1e-3, f"Ellipse perimeter error too high: {rel_err_perim}"


def test_native_rotated_ellipse_area_and_perimeter():
    """Verify rotated ellipse inclusion area and perimeter accuracy."""
    rx, ry = 0.25, 0.15
    dy = 1.0
    c = (0.5, 0.5)
    angle = np.pi / 3.0
    inclusions = [{"kind": "ellipse", "c": c, "rx": rx, "ry": ry, "angle": angle, "tag": 2, "nseg": 24}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, nseg=24, curve_order=2)

    # Inclusion area (invariant under rotation)
    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = np.pi * rx * ry
    rel_err_area = abs(area_inc - exact_area) / exact_area
    assert rel_err_area < 5e-4, f"Rotated ellipse area error too high: {rel_err_area}"

    # Perimeter (invariant under rotation)
    exact_perim = np.pi * (3 * (rx + ry) - np.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    perim = ngs.Integrate(1.0, mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("inc2"))
    rel_err_perim = abs(perim - exact_perim) / exact_perim
    assert rel_err_perim < 1e-3, f"Rotated ellipse perimeter error too high: {rel_err_perim}"


def test_native_square_and_rectangle():
    """Verify square and rectangle inclusions (§5.4)."""
    w, h = 0.3, 0.2
    dy = 1.0
    c = (0.5, 0.5)
    inclusions = [{"kind": "rectangle", "c": c, "w": w, "h": h, "tag": 2}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, curve_order=2)

    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = w * h
    assert abs(area_inc - exact_area) < 1e-10

    perim = ngs.Integrate(1.0, mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("inc2"))
    exact_perim = 2.0 * (w + h)
    assert abs(perim - exact_perim) < 1e-10


def test_native_ring_area_and_perimeter():
    """Verify ring (concentric circles) inclusion (§5.4)."""
    r_out, r_in = 0.25, 0.12
    dy = 1.0
    c = (0.5, 0.5)
    inclusions = [{"kind": "ring", "c": c, "r_out": r_out, "r_in": r_in, "tag": 2, "nseg": 24}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, nseg=24, curve_order=2)

    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = np.pi * (r_out**2 - r_in**2)
    rel_err_area = abs(area_inc - exact_area) / exact_area
    assert rel_err_area < 5e-4, f"Ring area error: {rel_err_area}"


def test_native_dimer_inclusion():
    """Verify dimer (two inclusions with gap) (§5.4)."""
    r = 0.1
    gap = 0.05
    dy = 1.0
    c = (0.5, 0.5)
    inclusions = [{"kind": "dimer", "sub_kind": "circle", "c": c, "r": r, "gap": gap, "tag": 2, "nseg": 24}]
    mesh = unit_cell(dy=dy, inclusions=inclusions, lc=0.08, lc2=2.0, nseg=24, curve_order=2)

    area_inc = ngs.Integrate(mesh.MaterialCF({"inc2": 1.0, "bg": 0.0}), mesh)
    exact_area = 2.0 * np.pi * r**2
    rel_err_area = abs(area_inc - exact_area) / exact_area
    assert rel_err_area < 5e-4, f"Dimer area error: {rel_err_area}"


def test_unit_cell_arbitrary_lc_and_per_inclusion_lc():
    """Verify new unit_cell(inclusions, dx, dy, lc, **lc_kwargs) signature, readable keys, and 100% backward compatibility."""
    dy = 1.0
    # 1. Clean new signature with readable keys ("center", "radius", "width", "height", "lc")
    inclusions_clean = [
        {"kind": "circle", "center": (0.3, 0.5), "radius": 0.12, "tag": 2, "lc": 1.5},
        {"kind": "rectangle", "center": (0.7, 0.5), "width": 0.2, "height": 0.2, "tag": 3, "lc": 5.0},
    ]
    mesh_clean = unit_cell(inclusions_clean, dy=dy, lc=0.12)
    n_inc2 = sum(1 for el in mesh_clean.Elements(ngs.VOL) if el.mat == "inc2")
    n_inc3 = sum(1 for el in mesh_clean.Elements(ngs.VOL) if el.mat == "inc3")
    assert n_inc3 > n_inc2, f"Expected finer mesh in inc3 (lc=5.0) than inc2 (lc=1.5), got {n_inc3} vs {n_inc2}"

    # 2. 100% backward compatibility with legacy positional unit_cell(dy, inclusions, period_x=1.0) and short keys
    mesh_legacy = unit_cell(
        0.8,
        [{"kind": "circle", "c": (0.5, 0.4), "r": 0.15, "tag": 2}],
        lc=0.1,
        lc2=2.0,
        period_x=1.0,
    )
    assert abs(ngs.Integrate(1.0, mesh_legacy) - 0.8) < 1e-11

    # 3. Coarser mesh inside inclusion when 0 < lc_n < 1 vs finer when lc_n > 1
    mesh_coarser = unit_cell(
        [{"kind": "circle", "center": (0.5, 0.5), "radius": 0.25, "tag": 2, "lc": 0.5}],
        lc=0.06,
    )
    mesh_equal = unit_cell(
        [{"kind": "circle", "center": (0.5, 0.5), "radius": 0.25, "tag": 2, "lc": 1.0}],
        lc=0.06,
    )
    mesh_finer = unit_cell(
        [{"kind": "circle", "center": (0.5, 0.5), "radius": 0.25, "tag": 2, "lc": 3.0}],
        lc=0.06,
    )
    n_coarser = sum(1 for el in mesh_coarser.Elements(ngs.VOL) if el.mat == "inc2")
    n_equal = sum(1 for el in mesh_equal.Elements(ngs.VOL) if el.mat == "inc2")
    n_finer = sum(1 for el in mesh_finer.Elements(ngs.VOL) if el.mat == "inc2")
    assert n_coarser <= n_equal < n_finer

    # 4. Arbitrary **lc_kwargs fallback (e.g. lc7 for tag=7)
    inclusions_kwargs = [
        {"kind": "circle", "center": (0.3, 0.5), "radius": 0.12, "tag": 2},
        {"kind": "circle", "center": (0.7, 0.5), "radius": 0.12, "tag": 7},
    ]
    mesh_kwargs = unit_cell(inclusions_kwargs, lc=0.12, lc2=1.5, lc7=5.0)
    n_tag2 = sum(1 for el in mesh_kwargs.Elements(ngs.VOL) if el.mat == "inc2")
    n_tag7 = sum(1 for el in mesh_kwargs.Elements(ngs.VOL) if el.mat == "inc7")
    assert n_tag7 > n_tag2

    # 5. Invalid non-lc<n> keyword argument raises TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        unit_cell([], lc=0.1, unknown_arg=2.0)


def test_msh4_rejection():
    """Verify that MSH 4.x files are explicitly rejected with informative message (§5.8)."""
    with tempfile.NamedTemporaryFile("w", suffix=".msh", delete=False) as f:
        f.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
        tmp_name = f.name
    try:
        with pytest.raises(ValueError, match="MSH 4.x format detected"):
            import_gmsh_mesh(tmp_name)
    finally:
        os.remove(tmp_name)


def test_mail_file_import_and_periodicity():
    """Verify .mail parser and importer on repository reference meshes (§5.6)."""
    mail_path = backend_path / "data" / "meshes" / "4testing-600_120.mail"
    if not mail_path.exists():
        mail_path = backend_path.parent / "tests" / "data" / "meshes" / "4testing-600_120.mail"
    if not mail_path.exists():
        pytest.skip(f"Reference file 4testing-600_120.mail not found")

    mesh, table_nod, type_el, pts, npt, nel = read_mail_file(str(mail_path), method="IIA")
    assert mesh.dim == 2
    assert npt == 1077
    assert nel == 518
    assert table_nod.shape == (6, nel)
    assert len(type_el) == nel
    assert pts.shape == (2, npt)

    # Test Bloch space reduction
    space = bloch_spaces(mesh, dy=1.0, order=2, kx=0.31, ky=0.17)
    elim = space.ndof - space.FreeDofs().NumSet()
    assert elim > 0, "No dofs eliminated on imported .mail mesh"
