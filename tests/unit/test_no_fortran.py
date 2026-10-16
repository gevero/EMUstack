"""Verification of DoD gates and zero Fortran backend invariants (PORTING_PLAN §1.2, §12).

Checks:
- DoD-1: Zero active Fortran imports or calls in backend
- DoD-2: fem_2d imports only ngsolve, netgen, numpy, scipy, and internal backend modules
- DoD-3: fem_1d has zero ngsolve/netgen imports
- DoD-8: Zero active os.system calls in backend
- Ban on IdentifyPeriodicBoundaries (buggy in Netgen 6.2.2608)
- .mail files are read-only input
"""
import ast
from pathlib import Path
import re
import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"


def test_dod3_fem1d_no_ngsolve():
    """DoD-3: fem_1d.py must not import ngsolve or netgen."""
    fem1d_path = _BACKEND_DIR / "fem_1d.py"
    assert fem1d_path.exists(), "fem_1d.py missing"
    content = fem1d_path.read_text()
    assert "ngsolve" not in content, "fem_1d.py must not import ngsolve"
    assert "netgen" not in content, "fem_1d.py must not import netgen"


def test_dod2_fem2d_allowed_imports():
    """DoD-2: fem_2d imports only ngsolve, netgen, numpy, scipy, and stdlib/backend modules."""
    fem2d_path = _BACKEND_DIR / "fem_2d.py"
    assert fem2d_path.exists(), "fem_2d.py missing"
    tree = ast.parse(fem2d_path.read_text())

    allowed_modules = {
        "os", "time", "typing", "sys", "math",
        "numpy", "scipy", "netgen", "ngsolve",
        "modes", "meshing", "paths", "linalg"
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                assert root_pkg in allowed_modules, f"Disallowed import {alias.name} in fem_2d.py"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg in allowed_modules, f"Disallowed import from {node.module} in fem_2d.py"


def test_no_identify_periodic_boundaries():
    """IdentifyPeriodicBoundaries is banned because it marks identifications UNDEFINED in Netgen 6.2.2608."""
    for py_file in _BACKEND_DIR.glob("*.py"):
        content = py_file.read_text()
        assert "IdentifyPeriodicBoundaries" not in content, (
            f"{py_file.name} contains banned Netgen API 'IdentifyPeriodicBoundaries'. "
            "Use 1-based AddPointIdentification instead (§5.8)."
        )


def test_no_fortran_calls_in_backend():
    """DoD-1: No active Fortran imports or EMUstack runtime entry points in backend."""
    forbidden_pattern = re.compile(
        r"^(?!\s*#).*?\b(from\s+fortran\s+import|import\s+fortran|EMUstack\.(calc_modes|conv_gmsh|gmsh_plot|field_value))\b"
    )

    violations = []
    for py_file in _BACKEND_DIR.glob("*.py"):
        lines = py_file.read_text().splitlines()
        for idx, line in enumerate(lines, start=1):
            if forbidden_pattern.search(line):
                violations.append(f"{py_file.name}:{idx}: {line.strip()}")

    assert len(violations) == 0, f"Found forbidden Fortran calls in backend:\n" + "\n".join(violations)


def test_dod8_no_active_os_system():
    """DoD-8: No active os.system calls in backend."""
    active_system_pattern = re.compile(r"^(?!\s*#).*?\bos\.system\s*\(")
    violations = []
    for py_file in _BACKEND_DIR.glob("*.py"):
        lines = py_file.read_text().splitlines()
        for idx, line in enumerate(lines, start=1):
            if active_system_pattern.search(line):
                violations.append(f"{py_file.name}:{idx}: {line.strip()}")

    assert len(violations) == 0, f"Found active os.system calls in backend:\n" + "\n".join(violations)


def test_mail_is_read_only():
    """Meshing and objects must only read .mail files, never write them."""
    write_mail_pattern = re.compile(r'open\s*\([^)]*?\.mail["\']\s*,\s*["\'][wa]')
    for py_file in _BACKEND_DIR.glob("*.py"):
        content = py_file.read_text()
        assert not write_mail_pattern.search(content), f"{py_file.name} appears to write a .mail file!"


def test_dod1_fortran_directory_and_source_purged():
    """DoD-1: backend/fortran/ directory is removed and zero .f/.f90/.pyf/.a files remain."""
    fortran_dir = _BACKEND_DIR / "fortran"
    assert not fortran_dir.exists(), f"Legacy Fortran directory {fortran_dir} must not exist (DoD-1)"

    root_dir = _BACKEND_DIR.parent
    banned_extensions = {".f", ".f90", ".pyf", ".a"}
    found_files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix in banned_extensions:
            found_files.append(str(p.relative_to(root_dir)))

    assert len(found_files) == 0, f"Found legacy Fortran/compiled archive files (DoD-1):\n" + "\n".join(found_files)
