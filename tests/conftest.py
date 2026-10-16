"""Pytest configuration and shared fixtures for EMUstack tests."""
import sys
from pathlib import Path
import pytest

_tests_dir = Path(__file__).resolve().parent
_backend_dir = _tests_dir.parent / "backend"
_root_dir = _tests_dir.parent

if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))


@pytest.fixture(scope="session")
def oracle_id():
    from refdata import ORACLE_IMAGE_ID
    return ORACLE_IMAGE_ID
