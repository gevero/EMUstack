"""Oracle reference data loader and provenance constants.

Authoritative provenance:
Container image: emustack:latest / emustack:oracle-77696a0d73de
"""
from pathlib import Path
import numpy as np

ORACLE_IMAGE_ID = "77696a0d73de7e711695269d289f41034e4462717c85aa8ce9a2f9e6bd9f30e4"
ORACLE_ARCHIVE_SHA256 = "974bcd77a2a95c006b3463d4512dacbd7161a246090bb3e64830caa04429e3fa"

_REF_DIR = Path(__file__).resolve().parent / "ref"


def load_reference_data(case_name):
    """Load pickled reference stack_list from tests/ref/<case_name>.npz.
    
    Handles Python 2 latin1 encoding safely.
    """
    path = _REF_DIR / f"{case_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    data = np.load(path, allow_pickle=True, encoding="latin1")
    return data["stack_list"]


def load_reference_txt(case_name, filename):
    """Load numerical .txt reference file from tests/ref/<case_name>/<filename>."""
    path = _REF_DIR / case_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    return np.loadtxt(path)
