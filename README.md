# EMUstack: Finite Element Maxwell Mode-Matching Solver

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Backend: Netgen/NGSolve](https://img.shields.io/badge/FEM%20Backend-Netgen%2FNGSolve%206.2-orange.svg)](https://ngsolve.org/)
[![Zero Fortran](https://img.shields.io/badge/Fortran-0%25%20(Pure%20Python%20%2B%20NGSolve)-brightgreen.svg)](#modernization)

EMUstack is an open-source simulation package for calculating light propagation, reflection, transmission, and absorption through multi-layered stacks of dispersive, lossy, nanostructured optical media. It implements a generalised scattering matrix method, extending the physical intuition of thin film optics to complex nanostructured systems.

---

## What's New in EMUstack v2.0

* **100% Zero Fortran**: Completely eliminated all legacy Fortran 90 routines, `f2py` bindings, and static library dependencies.
* **Modern FEM Engines**:
  * **2D FEM**: Built on [Netgen / NGSolve](https://ngsolve.org/) (v6.2+), using higher-order vector elements ($H(\text{curl})$ and $H^1$) with exact curved Bézier geometry.
  * **1D FEM**: Pure Python/SciPy implementation with exact rational element matrices and sparse shift-invert eigensolvers.
* **Dual Geometry Pipeline**: Native in-memory Netgen CSG (Path I) and Gmsh `.geo` / MSH 2.2 import with 1-based periodic node identification (Path II).
* **Direct FEM $\leftrightarrow$ FEM Mode Matching**: Inter-layer scattering matrices are computed directly between structured layers without requiring artificial thin-film air spacers.
* **Modern Python Packaging**: Standard PEP 517/518/621 `pyproject.toml`, installable via `pip install -e .`.

---

## Installation

### Requirements

* Python **>= 3.11**
* `numpy>=1.26`, `scipy>=1.11`, `matplotlib>=3.8`
* `ngsolve>=6.2.2600`, `netgen-mesher>=6.2.2600`
* Optional: `gmsh` for Path II mesh generation from `.geo` templates

### Install with pip

```bash
# Editable install from repo
pip install -e .

# With optional high-performance MKL Pardiso solvers
pip install -e ".[fast]"

# With test suite dependencies
pip install -e ".[dev]"
```

---

## Quickstart

```python
import numpy as np
from EMUstack import objects, materials, stack

# Define incident light (wavelength sweep 400 - 800 nm, TE pol)
light = objects.Light(wl_nm=np.linspace(400, 800, 41), pol='TE', theta=0.0)

# Materials
air = materials.Material(1.0)
glass = materials.Material(1.5)
si = materials.Si_c

# 1D grating: period = 400 nm, duty cycle 50%
grating = objects.NanoStruct(
    periodicity='1D_array',
    period=400.0,
    diameter1=200.0,
    background=air,
    inclusion_a=si,
    thickness=150.0,
    num_modes=15,
)

# Stack layers: Air -> Grating -> Glass substrate
sim_stack = stack.Stack([
    (air, 0.0),
    (grating, 150.0),
    (glass, 0.0),
], light=light)

# Spectra
R = sim_stack.R()
T = sim_stack.T()
A = 1.0 - R - T
```

---

## Package Structure

* `backend/`
  * `objects.py`: High-level nanostructure, thin film, light, and stack classes.
  * `fem_2d.py`: 2D Maxwell de Rham pencil solver powered by NGSolve.
  * `fem_1d.py`: 1D coupled pencil solver in pure Python/SciPy.
  * `meshing.py`: Dual geometry pipeline (Netgen CSG Path I, Gmsh MSH 2.2 Path II).
  * `mode_matching.py`: Direct FEM $\leftrightarrow$ FEM interface scattering matrices.
  * `mode_calcs.py`: Modal solver dispatchers and Bloch mode containers.
  * `stack.py`: Scattering matrix cascading (Redheffer star product).
  * `materials.py`: Dispersive optical constant tables and Drude models.
  * `plotting.py`: Near-field spatial distributions and cross-sectional visualizations.
  * `data/`: Tabulated refractive index data (62 materials) and mesh templates.
* `examples/`: Extensive simulation tutorials (`simo_010` to `simo_091`).
* `examples_netgen/`: Modern native Netgen CSG examples.
* `tests/`: Comprehensive pytest verification suite.

---

## Testing

Run the automated test suite with pytest:

```bash
pytest tests/ -v
```

---

## License

EMUstack is free software released under the **GNU General Public License v3 or later (GPL-3.0-or-later)**. See `LICENSE.txt` for details.
