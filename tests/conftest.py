"""Pytest configuration and shared fixtures for EMUstack tests."""
import sys
from pathlib import Path
import pytest

_tests_dir = Path(__file__).resolve().parent
_backend_dir = _tests_dir.parent / "backend"
_root_dir = _tests_dir.parent

# The test_case_*.py files are written like legacy scripts (bare module names, backend/ on
# sys.path); the unit tests import the package (from EMUstack.backend import ...).
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))


@pytest.fixture(scope="session")
def oracle_id():
    from refdata import ORACLE_IMAGE_ID
    return ORACLE_IMAGE_ID


@pytest.fixture(scope="session", autouse=True)
def cleanup_calculation_outputs():
    """Ensure any temporary test output files (*_stack0001.txt, etc.) are removed after pytest finishes."""
    yield
    from EMUstack.backend import plotting
    plotting.clear_previous()



_mesh_tmp = None


def pytest_configure(config):
    """gmsh meshes go to a temporary folder, not into the repository. Set before collection:
    the legacy test_case_*.py scripts mesh at import time."""
    global _mesh_tmp
    import tempfile
    _mesh_tmp = tempfile.TemporaryDirectory(prefix="emustack_meshes_")
    for meshing in _meshing_modules():
        meshing.mesh_dir = _mesh_tmp.name


def pytest_unconfigure(config):
    for meshing in _meshing_modules():
        meshing.mesh_dir = None
    if _mesh_tmp is not None:
        _mesh_tmp.cleanup()


def _meshing_modules():
    """The package's meshing module and the copy that legacy-style scripts import by bare name."""
    from EMUstack.backend import meshing as package_meshing
    import meshing as legacy_meshing
    return {package_meshing, legacy_meshing}
