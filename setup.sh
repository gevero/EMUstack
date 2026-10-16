#!/usr/bin/env bash
# Modernized EMUstack installation and verification script (Python 3.11+, Zero Fortran)
set -euo pipefail

echo "====================================================================="
echo "   EMUstack: Finite Element Maxwell Mode-Matching Solver (v2.0)      "
echo "====================================================================="

# Check Python environment
PYTHON_BIN="${PYTHON:-python3}"
echo "Using Python: $($PYTHON_BIN --version) at $(which $PYTHON_BIN)"

# Install package in editable mode with modern build backend
echo "Installing emustack-fem in editable mode..."
$PYTHON_BIN -m pip install --no-build-isolation -e .

# Run pytest test suite
echo "Running pytest test suite..."
$PYTHON_BIN -m pytest tests/ -v --tb=short

echo ""
echo "====================================================================="
echo "        EMUstack modernized suite successfully verified!             "
echo "  Zero Fortran dependencies. Powered by Netgen/NGSolve & SciPy.      "
echo "====================================================================="
