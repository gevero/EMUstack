# EMUstack: Finite-Element & Scattering-Matrix Electromagnetic Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/gevero/EMUstack/blob/master/LICENSE.txt)
[![GitHub Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-blue.svg)](https://gevero.github.io/EMUstack/)
[![FEM Engine: NGSolve / Netgen](https://img.shields.io/badge/FEM%20Engine-NGSolve%20%2F%20Netgen-orange.svg)](https://ngsolve.org/)
[![Pure Python 1D FEM](https://img.shields.io/badge/1D%20FEM-Pure%20Python%20%2F%20SciPy-brightgreen.svg)](#1d-and-2d-mode-solvers)

**EMUstack** is an open-source simulation package for calculating electromagnetic wave propagation, transmission, reflection, and absorption in multilayered nanostructured optical media. It combines high-order finite-element eigenmode solvers with generalized scattering-matrix (S-matrix) methods, bringing the physical intuition and computational speed of thin-film optics to complex periodic nanostructures.

---

## What EMUstack Is & How It Works

EMUstack solves electromagnetic scattering through multilayered stacks in two decoupled stages:

```
Incident Light (Plane Wave / Circular Polarization)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Layer 1: Semi-infinite Substrate (ThinFilm)            │  <- Analytic plane waves (diffraction orders)
├────────────────────────────────────────────────────────┤
│ Layer 2: Periodic Nanostructure (NanoStruct)          │  <- Finite-element Bloch modes (NGSolve/SciPy)
├────────────────────────────────────────────────────────┤
│ Layer 3: Semi-infinite Superstrate (ThinFilm)          │  <- Analytic plane waves (diffraction orders)
└────────────────────────────────────────────────────────┘
       │
       ▼
Transmitted, Reflected, & Absorbed Flux + 3D Near-Fields
```

1. **Step 1: Eigenmode Calculation in Each Layer**
   - **Homogeneous Layers (`ThinFilm`):** Fields are represented analytically as a spatial Fourier series of plane-wave diffraction orders (`Anallo`).
   - **Periodic Nanostructures (`NanoStruct`):** Transverse unit-cell fields are computed numerically as Bloch modes (`Simmo`) using high-order finite elements. For 1D gratings, pure Python/SciPy solves the coupled wave equations. For 2D arrays, NGSolve solves the Maxwell eigenvalue equation using mixed vector-scalar finite elements.
2. **Step 2: Scattering-Matrix Layer Stacking**
   - Inter-layer transmission and reflection matrices ($R_{ij}, T_{ij}$) are calculated by matching transverse field continuities across each interface.
   - Layers are cascaded using numerically stable Redheffer star-product recursion. Mode propagation through layer thicknesses is evaluated analytically via $\exp(i k_z d)$, allowing thickness sweeps and parameter studies with zero re-meshing or re-solving of eigenmodes.

---

## Installation

EMUstack v2.0 is a pure Python package. It requires **no Fortran compilers**, **no C++ build tools**, and **no SuiteSparse compilation**.

### Prerequisites
* **Python:** 3.10, 3.11, 3.12, 3.13, or 3.14.
* **Core Dependencies:** `numpy>=1.22`, `scipy>=1.9`, `matplotlib>=3.6`, `ngsolve>=6.2.2401`, `netgen-mesher>=6.2.2401`, `xarray>=2023.1`, `pyyaml>=6.0`, `gmsh>=4.11` (the default mesher).
* **Optional (installed by `.[all]`):** Jupyter, `adaptive`, `loky`, pytest, Sphinx.

### From PyPI
```bash
pip install emustack            # core: FEM solvers, gmsh and netgen meshers
pip install "emustack[all]"     # + notebooks, tests, docs
```

### From Source
The repository includes the example notebooks and the test suite:
```bash
git clone https://github.com/gevero/EMUstack.git
cd EMUstack
```
The commands below install EMUstack from the clone **with all optional packages** (`.[all]`), so the mesher, the example notebooks, the tests and the docs build all work right away.

### Environment Setup & Installation

We recommend installing EMUstack into an isolated environment using Conda, standard `venv`, or `uv`:

#### Option 1: Dedicated Conda Environment + Pip
```bash
conda create -n emustack python=3.12 -y
conda activate emustack
pip install --upgrade pip
pip install -e ".[all]"
```

#### Option 2: Dedicated Virtual Environment (`venv`) + Pip
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[all]"
```

#### Option 3: Dedicated Virtual Environment with `uv`
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

### Smaller Installs
`[all]` is a shortcut for every optional group. To install only part of it:
```bash
pip install -e .                    # core: gmsh mesher plus netgen
pip install -e ".[notebooks]"       # + Jupyter, nbconvert, adaptive, loky
pip install -e ".[dev]"             # + pytest, ruff
pip install -e ".[docs]"            # + Sphinx, sphinx-rtd-theme, nbsphinx
```
(Use `uv pip install` in place of `pip install` with uv.) A system Gmsh also works: EMUstack checks `GMSH_BIN`, then `gmsh` on your `PATH`.

On a minimal Linux system (container, headless server) the NGSolve and gmsh wheels need a few system libraries (no display): `sudo apt-get install libopenblas0 libglu1-mesa libgl1 libgomp1 libxcursor1 libxft2 libxinerama1`. Without the gmsh ones, use `mesher='netgen'`.

---

## Quickstart Tutorials

### 1. Simulating a 2D Silicon Nanopillar Array

```python
import numpy as np
from EMUstack.backend import objects, materials, stack, plotting

# 1. Incident illumination (wavelength 600 nm, normal incidence)
wl = 600.0
period = 400.0
light = objects.Light(wl_nm=wl, max_order_PWs=1, theta=0.0, phi=0.0)

# 2. Layer materials and structures
substrate = objects.ThinFilm(period=period, material=materials.Material(1.5), height_nm='semi_inf')
superstrate = objects.ThinFilm(period=period, material=materials.Material(1.0), height_nm='semi_inf')

# Silicon cylinders, 160 nm in diameter and 150 nm tall. geo_params holds the geometry
# (lengths in nm) and the mesh sizes: lc for the background, lc / lc2 at the cylinder edge.
pillar_layer = objects.NanoStruct(
    '2D_array',
    geo_params={'period': period, 'diameter1': 160.0, 'lc': 0.15, 'lc2': 2.0},
    background=materials.Material(1.0),
    inclusion_a=materials.Si_c,
    height_nm=150.0,
    plotting_fields=1,  # keep the mode fields for the field maps in example 3
)

# 3. Compute modes and build scattering stack
modes_sub = substrate.calc_modes(light)
modes_sup = superstrate.calc_modes(light)
modes_meta = pillar_layer.calc_modes(light, num_BMs=20)

sim_stack = stack.Stack((modes_sub, modes_meta, modes_sup))
sim_stack.calc_scat(pol='TE')

# 4. Extract total reflectance, transmittance, and absorptance into xarray
ds_spectra = plotting.spectra_to_xarray([sim_stack], wavelengths=[wl])
print(f"Total Transmission: {float(ds_spectra['T_total'].values[0]):.4f}")
print(f"Total Reflection:   {float(ds_spectra['R_total'].values[0]):.4f}")
```

### 2. Custom 2D Geometries with Python & Gmsh Templates

EMUstack supports both pure-Python Netgen geometry templates (`*.py`) and classic Gmsh scripts (`*.geo`) located in the working directory or `backend/data/meshes/` (e.g. `ellipse_msh_template.py`, `ellipse_msh_template.geo`, `chiral_qbic_gorkunov_msh_template.py`). Templates are named `<shape>_msh_template.<ext>`, and `.geo` and `.py` templates take the same parameter names (`period`, `period_y`, `diameter1`, `width`, `thickness`, ...; lengths in nm, angles in radians). Old file and parameter names such as `ring1_msh_template.geo`, `d_in_nm` or `a1` still work.

In a Python template, `build_unit_cell(...)` receives the parameters from `geo_params`, normalizes physical dimensions by `period` onto the dimensionless unit cell $[0, 1] \times [0, d_y]$, defines a list of `inclusions` (with `"tag": 2` mapped to `inclusion_a`, `"tag": 3` to `inclusion_b`, etc., and `"tag": 1` reserved for `background`), and calls `meshing.unit_cell()` to build a curved second-order periodic `ngsolve.Mesh` in memory:

```python
from EMUstack.backend import objects, materials

ellipse_layer = objects.NanoStruct(
    '2D_array',
    background=materials.Material(1.0),
    inclusion_a=materials.Material(2.0),
    height_nm=150.0,
    geo_file='ellipse_msh_template.py',  # or 'ellipse_msh_template.geo'
    geo_params={'period': 400.0, 'diameter1': 100.0, 'ellipticity': 0.5, 'angle': 0.0, 'lc': 0.15, 'lc2': 2.0}
)
```

For a complete, step-by-step guide comparing the Gmsh (`.geo`) and Netgen (`.py`) pipelines—with line-by-line walkthroughs of `ellipse_msh_template.geo`, `ellipse_msh_template.py`, and `chiral_qbic_gorkunov_msh_template.py`—see the **[2D Geometry & Meshing Guide](https://gevero.github.io/EMUstack/meshing.html)** in the documentation.

### 3. Evaluating Electric & Magnetic Near-Fields, Poynting Vector, and Optical Chirality

When a layer is created with `plotting_fields=1` (as in example 1), EMUstack provides complete 3D electric and magnetic field interpolators via `FieldBundle`. They take in-plane coordinates normalized to the period (0 to 1):

```python
# Evaluate fields at mid-height (z = 75 nm) in the nanostructured layer (index 1)
bundle = plotting.fields_interpolator_in_plane(sim_stack, lay_interest=1, z_value=75.0, return_H=True)

# Grid over one unit cell, in units of the period
u = np.linspace(0.0, 1.0, 50)
U, V = np.meshgrid(u, u)

# Electric and magnetic fields, Poynting flux, and optical chirality enhancement
abs_E = bundle.AbsE(U, V)
sz_flux = bundle.Sz(U, V)
chirality = bundle.chirality(U, V)
c_hat = bundle.chirality_enhancement(U, V)

# Export a labeled dataset (x, y in nm) directly to xarray
ds_field = bundle.to_xarray(u, u, period=period)
```

---

## Interactive Tutorial Notebooks

EMUstack v2.0 provides **45 didactic Jupyter Notebooks** organized into two dedicated suites:
- **`EMUstack/examples/` (22 Notebooks):** The original EMUstack examples with their legacy inputs (a `LEGACY_PARAMS` switch selects lighter sampling where the legacy inputs take hours), meshed with the legacy gmsh templates.
- **`EMUstack/examples_netgen/` (23 Notebooks):** Demonstrates the native in-memory Netgen CSG workflow with pure Python geometry templates.

| Notebook | Topic & Physical System | `examples/` (Gmsh) | `examples_netgen/` (Netgen) |
| :--- | :--- | :---: | :---: |
| `simo_010-single_interface.ipynb` | Analytical Fresnel equations & diffraction order setup | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_010-single_interface.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_010-single_interface.ipynb) |
| `simo_011-single_interface-dispersive.ipynb` | Dispersive refractive index interpolation (materials.Si_c) | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_011-single_interface-dispersive.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_011-single_interface-dispersive.ipynb) |
| `simo_020-thin_film_multilayered_stack.ipynb` | Fabry–Pérot interference and layer-by-layer absorption | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_020-thin_film_multilayered_stack.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_020-thin_film_multilayered_stack.ipynb) |
| `simo_021-thin_film_mirror.ipynb` | Metallic back-reflector and substrate loss handling | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_021-thin_film_mirror.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_021-thin_film_mirror.ipynb) |
| `simo_030-1D_grating.ipynb` | 1D lamellar diffraction grating and polarization (TE/TM) | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_030-1D_grating.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_030-1D_grating.ipynb) |
| `simo_031-1D_grating-2_inclusions.ipynb` | 1D multi-inclusion unit cell with interleaved elements | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_031-1D_grating-2_inclusions.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_031-1D_grating-2_inclusions.ipynb) |
| `simo_032-1D_grating-3_inclusions.ipynb` | 3-element 1D unit cell with complex spacing rules | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_032-1D_grating-3_inclusions.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_032-1D_grating-3_inclusions.ipynb) |
| `simo_040-2D_array.ipynb` | 2D silicon nanowire array for photovoltaics ($J_{\text{sc}}$, $\eta$) | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_040-2D_array.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_040-2D_array.ipynb) |
| `simo_041-combining_1D_and_2D_array.ipynb` | Hybrid stack combining 1D lamellar grating and 2D array | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_041-combining_1D_and_2D_array.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_041-combining_1D_and_2D_array.ipynb) |
| `simo_042-eliptical_holes-CD.ipynb` | Elliptical nanohole array and circular dichroism | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_042-eliptical_holes-CD.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_042-eliptical_holes-CD.ipynb) |
| `simo_050-plotting_fields_1d.ipynb` | 1D Bloch mode profiles and vertical $(x, z)$ cross-sections | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_050-plotting_fields_1d.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_050-plotting_fields_1d.ipynb) |
| `simo_051-plotting_fields_2d.ipynb` | 2D near-field visualization in $(x, y)$ and $(x, z)$ planes | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_051-plotting_fields_2d.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_051-plotting_fields_2d.ipynb) |
| `simo_052-plotting_amplitudes.ipynb` | Modal amplitudes, plane-wave order coupling, evanescent FOM | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_052-plotting_amplitudes.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_052-plotting_amplitudes.ipynb) |
| `simo_053-plotting_k_space_amps.ipynb` | In-plane momentum excitation spectrum ($k_\parallel / k_0$) | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_053-plotting_k_space_amps.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_053-plotting_k_space_amps.ipynb) |
| `simo_060-shear_transformations.ipynb` | Lateral unit-cell translation via shear operators | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_060-shear_transformations.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_060-shear_transformations.ipynb) |
| `simo_070-ultrathin_limit.ipynb` | Fundamental 50% absorption limit in ultrathin coherent films | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_070-ultrathin_limit.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_070-ultrathin_limit.ipynb) |
| `simo_071-many_substrates.ipynb` | Modal reuse across 100 substrates & thickness averaging | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_071-many_substrates.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_071-many_substrates.ipynb) |
| `simo_080-convergence-stacked_gratings.ipynb` | Convergence benchmark of stacked gratings (Handmer et al.) | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_080-convergence-stacked_gratings.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_080-convergence-stacked_gratings.ipynb) |
| `simo_090-EOT.ipynb` | Extraordinary optical transmission through subwavelength holes | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_090-EOT.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_090-EOT.ipynb) |
| `simo_091-slab_mode_finding.ipynb` | Guided slab mode dispersion finding via determinant zeros | [View](https://github.com/gevero/EMUstack/blob/master/examples/simo_091-slab_mode_finding.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/simo_091-slab_mode_finding.ipynb) |
| `Au Nanodisk Array with Interpolators.ipynb` | Plasmonic gold nanodisk array & 3-plane field maps (`xy`, `xz`, `yz`) | [View](https://github.com/gevero/EMUstack/blob/master/examples/Au%20Nanodisk%20Array%20with%20Interpolators.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/Au%20Nanodisk%20Array%20with%20Interpolators.ipynb) |
| `Elliptical Nanohole Array (custom target interface).ipynb` | Custom geometry template tutorial & 3-plane maps (`xy`, `xz`, `yz`) | [View](https://github.com/gevero/EMUstack/blob/master/examples/Elliptical%20Nanohole%20Array%20%28custom%20target%20interface%29.ipynb) | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/Elliptical%20Nanohole%20Array%20%28custom%20target%20interface%29.ipynb) |
| `Chiral_qBIC_Optical_Chirality.ipynb` | BIC to quasi-BIC in a bar-dimer metasurface (Gorkunov et al., Fig. 3a, [520, 650] nm), field & optical chirality enhancement $\hat{C}$ at the q-BIC | — | [View](https://github.com/gevero/EMUstack/blob/master/examples_netgen/Chiral_qBIC_Optical_Chirality.ipynb) |

---

## Retrocompatibility & Migration Guide (v1.x -> v2.0)

EMUstack v2.0 is engineered with strict **backward compatibility** for existing user simulation scripts and geometry assets, allowing a seamless upgrade from legacy Fortran EMUstack (v1.x):

### 1. Script Syntax Compatibility
Scripts written for legacy EMUstack run with only their imports changed. Core classes and methods keep their signatures and behaviour. This is legacy `examples/simo_040-2D_array.py`, shortened:
```python
import numpy as np
from EMUstack.backend import objects, materials, plotting
from EMUstack.backend.stack import Stack

# One Light object per wavelength, as in legacy EMUstack
wavelengths = np.linspace(310, 1127, 3)
light_list = [objects.Light(wl, max_order_PWs=2, theta=0.0, phi=0.0) for wl in wavelengths]

period = 600
superstrate = objects.ThinFilm(period, height_nm='semi_inf', material=materials.Air, loss=False)
substrate = objects.ThinFilm(period, height_nm='semi_inf', material=materials.SiO2, loss=False)
# Positional arguments: periodicity, period, diameter1
NW_array = objects.NanoStruct('2D_array', period, 120, height_nm=2330,
    inclusion_a=materials.Si_c, background=materials.Air, loss=True,
    make_mesh_now=True, force_mesh=True, lc_bkg=0.1, lc2=2.0)

def simulate_stack(light):
    sim_superstrate = superstrate.calc_modes(light)
    sim_substrate = substrate.calc_modes(light)
    sim_NWs = NW_array.calc_modes(light, num_BMs=200)
    stack = Stack((sim_substrate, sim_NWs, sim_superstrate))  # bottom to top
    stack.calc_scat(pol='TE')
    return stack

stacks_list = [simulate_stack(light) for light in light_list]  # or multiprocessing Pool.map
plotting.t_r_a_plots(stacks_list, active_layer_nu=1, J_sc=True)  # returns figures, writes no files
```

### 2. Legacy Mesh Assets & Formats
All legacy mesh assets remain fully functional:
- **Gmsh `.geo` scripts:** Parameterized `.geo` templates in `backend/data/meshes/` run via Gmsh whenever `objects.default_mesher = 'gmsh'` (the default).
- **Pre-generated `.mail` & `.msh` files:** Existing offline meshes load directly with zero conversion.

### 3. Configurable Compatibility Flags
For users requiring exact numerical reproduction of legacy Fortran runs, EMUstack v2.0 provides fine-grained runtime switches:

| Subsystem | Legacy Default | Modern Alternative | Configuration |
|---|---|---|---|
| **2D Mesh Engine** | Gmsh with `.geo` templates | Native in-memory Netgen CSG | `objects.default_mesher = 'gmsh'` (or `'netgen'`) |
| **2D Eigensolver** | ARPACK shift-invert | NGSolve Arnoldi iteration | `fem_2d.eigensolver = 'arpack'` (or `'ngsolve'`) |
| **Overlap Quadrature** | Legacy 16-point rule | High-order NGSolve integration | `fem_2d.overlap_quadrature = 'legacy'` (or `'ngsolve'`) |
| **$E_z$ Normalization** | Legacy unscaled convention | Physical divergence-free $\nabla\cdot\mathbf{E}=0$ | `field_eval.ez_convention = 'physical'` (or `'legacy'`) |

> **Note on $E_z$ in Field Maps:**  
> Legacy EMUstack's Fortran routine (`field_value_plane.f`) contained an unphysical factor of $i k_z^2$ in $E_z$. In v2.0, the default is `field_eval.ez_convention = 'physical'`, ensuring $\nabla \cdot \mathbf{E} = 0$. Setting `field_eval.ez_convention = 'legacy'` (or setting `EMUSTACK_EZ_CONVENTION=legacy` in the shell) restores exact bitwise parity with legacy field maps. Magnetic fields ($\mathbf{H}$), Poynting vectors ($\mathbf{S}$), and optical chirality density always use the physical formulation.

### 4. Non-Polluting Plotting Defaults
In legacy EMUstack, plotting routines wrote `.pdf`, `.png`, and `.txt` files directly to disk by default. In v2.0, all plotting routines return Matplotlib `Figure` objects for clean interactive rendering in Jupyter notebooks, defaulting to:
```python
save_pdf = False
save_txt = False
save_files = False
```
Legacy file dumps can be re-enabled at any time by explicitly passing `save_pdf=True` or `save_txt=True`.

---

## What Changed in v2.0 (Summary of Modernization)

1. **Zero Fortran & Pure-Python Setup:**
   - No compiled Fortran (`f2py`, `gfortran`, `ARPACK`) or C wrappers: every dependency installs from pip wheels for Python 3.10–3.14. Tested on Linux; the dependencies also ship wheels for macOS and Windows, where EMUstack has not been tested yet.
2. **Modern Finite-Element Solvers:**
   - Pure-Python 1D FEM using high-order polynomial elements with coupled TE/TM states for conical incidence.
   - NGSolve 2D FEM utilizing mixed vector-scalar spaces ($H(\text{curl}) \times H^1$) with exact tangential continuity and zero spurious eigenmodes.
3. **In-Memory Netgen CSG Geometry:**
   - Parameterized Python geometry templates (`*_msh_template.py`) enabling curved second-order elements with zero disk I/O.
4. **Full Near-Field & Chirality Suite:**
   - Faraday reconstruction of true 3D magnetic fields $\mathbf{H}$.
   - Fast `FieldBundle` interpolators for in-plane slices ($xy$), cross-sections ($xz$, $yz$), Poynting vector flux, Lipkin optical chirality density $C$, and normalized chirality enhancement $\hat{C}$.
5. **Vectorized Labeled Export:**
   - Native integration with `xarray.Dataset`, packaging multi-frequency spectra and 3D electromagnetic near-fields with physical coordinates and metadata.
6. **Parallel Adaptive Sampling:**
   - First-class compatibility with `adaptive.Learner1D` and process pools (`loky`) for high-throughput spectral discovery.

---

## History & Credits

EMUstack was created by Björn Sturmberg, Kokou Dossou and Felix Lawrence (with Christopher Poulton, Ross McPhedran, Martijn de Sterke and Lindsay Botten) as a Fortran/Python package. Version 1, with its Fortran finite-element backend, lives at [bjornsturmberg/EMUstack](https://github.com/bjornsturmberg/EMUstack) ([v1 documentation](https://emustack.readthedocs.io/)). Version 2 is a pure-Python port maintained by Giovanni Pellegrini at [gevero/EMUstack](https://github.com/gevero/EMUstack); it reproduces version 1's results (see the parity tests) and keeps its scripts working.

If you use EMUstack in published work, please cite the original paper: B. C. P. Sturmberg et al., "EMUstack: an open source route to insightful electromagnetic computation via the Bloch mode scattering matrix method", *Computer Physics Communications* **202**, 276–286 (2016), doi:[10.1016/j.cpc.2015.12.022](https://doi.org/10.1016/j.cpc.2015.12.022).

---

## License

EMUstack is licensed under the GNU General Public License v3.0 or later ([GPL-3.0-or-later](https://github.com/gevero/EMUstack/blob/master/LICENSE.txt)).
