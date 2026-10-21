# EMUstack: Finite-Element & Scattering-Matrix Electromagnetic Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
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
* **Core Dependencies:** `numpy>=1.22`, `scipy>=1.9`, `matplotlib>=3.6`, `ngsolve>=6.2.2401`, `netgen-mesher>=6.2.2401`, `xarray>=2023.1`.

### Install from Repository
```bash
# Clone the repository
git clone https://github.com/EMUstack/EMUstack.git
cd EMUstack

# Editable installation
pip install -e .

# Optional extras
pip install -e ".[notebooks]"   # Jupyter, nbconvert, adaptive
pip install -e ".[dev]"         # Pytest, ruff
pip install -e ".[docs]"        # Sphinx, sphinx-rtd-theme
```

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

pillar_layer = objects.NanoStruct(
    '2D_array',
    period=period,
    diameter1=160.0,
    background=materials.Material(1.0),
    inclusion_a=materials.Si_c,
    height_nm=150.0,
    geo_params={'period': period, 'a1': 160.0, 'lc': 0.15, 'lc2': 2.0}
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

### 2. Custom 2D Geometries with Python Templates

EMUstack supports parameterized Python geometry templates located in `backend/data/meshes/` (e.g. `ellipse_msh_template.py`, `chiral_qbic_template.py`):

```python
from EMUstack.backend import objects, materials

ellipse_layer = objects.NanoStruct(
    '2D_array',
    period=400.0,
    diameter1=200.0,
    background=materials.Material(1.0),
    inclusion_a=materials.Material(2.0),
    height_nm=150.0,
    geo_file='ellipse_msh_template.py',
    geo_params={'period': 400.0, 'a1': 100.0, 'ellipticity': 0.5, 'angle': 0.0, 'lc': 0.15, 'lc2': 2.0}
)
```

### 3. Evaluating Electric & Magnetic Near-Fields, Poynting Vector, and Optical Chirality

When computing modes with `plotting_fields=1`, EMUstack provides complete 3D electric and magnetic field interpolators via `FieldBundle`:

```python
# Evaluate fields at mid-height (z = 75 nm) in nanostructured layer (index 1)
bundle = plotting.fields_interpolator_in_plane(sim_stack, lay_interest=1, z_value=75.0, return_H=True)

# Grid evaluation
x = np.linspace(-period/2, period/2, 50)
y = np.linspace(-period/2, period/2, 50)
X, Y = np.meshgrid(x, y)

# Electric and magnetic fields, Poynting flux, and optical chirality enhancement
abs_E = bundle.AbsE(X, Y)
sz_flux = bundle.Sz(X, Y)
chirality = bundle.chirality(X, Y)
c_hat = bundle.chirality_enhancement(X, Y)

# Export labeled dataset directly to xarray
ds_field = bundle.to_xarray(x, y)
```

---

## Interactive Tutorial Notebooks

EMUstack v2.0 provides **45 didactic Jupyter Notebooks** organized into two dedicated suites:
- **`EMUstack/examples/` (22 Notebooks):** Demonstrates the classic Gmsh `.geo` and pre-generated `.mail` mesh workflow.
- **`EMUstack/examples_netgen/` (23 Notebooks):** Demonstrates the native in-memory Netgen CSG workflow with pure Python geometry templates.

| Notebook | Topic & Physical System | `examples/` (Gmsh) | `examples_netgen/` (Netgen) |
| :--- | :--- | :---: | :---: |
| `simo_010-single_interface.ipynb` | Analytical Fresnel equations & diffraction order setup | [View](examples/simo_010-single_interface.ipynb) | [View](examples_netgen/simo_010-single_interface.ipynb) |
| `simo_011-single_interface-dispersive.ipynb` | Dispersive refractive index interpolation (materials.Si_c) | [View](examples/simo_011-single_interface-dispersive.ipynb) | [View](examples_netgen/simo_011-single_interface-dispersive.ipynb) |
| `simo_020-thin_film_multilayered_stack.ipynb` | Fabry–Pérot interference and layer-by-layer absorption | [View](examples/simo_020-thin_film_multilayered_stack.ipynb) | [View](examples_netgen/simo_020-thin_film_multilayered_stack.ipynb) |
| `simo_021-thin_film_mirror.ipynb` | Metallic back-reflector and substrate loss handling | [View](examples/simo_021-thin_film_mirror.ipynb) | [View](examples_netgen/simo_021-thin_film_mirror.ipynb) |
| `simo_030-1D_grating.ipynb` | 1D lamellar diffraction grating and polarization (TE/TM) | [View](examples/simo_030-1D_grating.ipynb) | [View](examples_netgen/simo_030-1D_grating.ipynb) |
| `simo_031-1D_grating-2_inclusions.ipynb` | 1D multi-inclusion unit cell with interleaved elements | [View](examples/simo_031-1D_grating-2_inclusions.ipynb) | [View](examples_netgen/simo_031-1D_grating-2_inclusions.ipynb) |
| `simo_032-1D_grating-3_inclusions.ipynb` | 3-element 1D unit cell with complex spacing rules | [View](examples/simo_032-1D_grating-3_inclusions.ipynb) | [View](examples_netgen/simo_032-1D_grating-3_inclusions.ipynb) |
| `simo_040-2D_array.ipynb` | 2D silicon nanowire array for photovoltaics ($J_{\text{sc}}$, $\eta$) | [View](examples/simo_040-2D_array.ipynb) | [View](examples_netgen/simo_040-2D_array.ipynb) |
| `simo_041-combining_1D_and_2D_array.ipynb` | Hybrid stack combining 1D lamellar grating and 2D array | [View](examples/simo_041-combining_1D_and_2D_array.ipynb) | [View](examples_netgen/simo_041-combining_1D_and_2D_array.ipynb) |
| `simo_042-eliptical_holes-CD.ipynb` | Elliptical nanohole array and circular dichroism | [View](examples/simo_042-eliptical_holes-CD.ipynb) | [View](examples_netgen/simo_042-eliptical_holes-CD.ipynb) |
| `simo_050-plotting_fields_1d.ipynb` | 1D Bloch mode profiles and vertical $(x, z)$ cross-sections | [View](examples/simo_050-plotting_fields_1d.ipynb) | [View](examples_netgen/simo_050-plotting_fields_1d.ipynb) |
| `simo_051-plotting_fields_2d.ipynb` | 2D near-field visualization in $(x, y)$ and $(x, z)$ planes | [View](examples/simo_051-plotting_fields_2d.ipynb) | [View](examples_netgen/simo_051-plotting_fields_2d.ipynb) |
| `simo_052-plotting_amplitudes.ipynb` | Modal amplitudes, plane-wave order coupling, evanescent FOM | [View](examples/simo_052-plotting_amplitudes.ipynb) | [View](examples_netgen/simo_052-plotting_amplitudes.ipynb) |
| `simo_053-plotting_k_space_amps.ipynb` | In-plane momentum excitation spectrum ($k_\parallel / k_0$) | [View](examples/simo_053-plotting_k_space_amps.ipynb) | [View](examples_netgen/simo_053-plotting_k_space_amps.ipynb) |
| `simo_060-shear_transformations.ipynb` | Lateral unit-cell translation via shear operators | [View](examples/simo_060-shear_transformations.ipynb) | [View](examples_netgen/simo_060-shear_transformations.ipynb) |
| `simo_070-ultrathin_limit.ipynb` | Fundamental 50% absorption limit in ultrathin coherent films | [View](examples/simo_070-ultrathin_limit.ipynb) | [View](examples_netgen/simo_070-ultrathin_limit.ipynb) |
| `simo_071-many_substrates.ipynb` | Modal reuse across 100 substrates & thickness averaging | [View](examples/simo_071-many_substrates.ipynb) | [View](examples_netgen/simo_071-many_substrates.ipynb) |
| `simo_080-convergence-stacked_gratings.ipynb` | Convergence benchmark of stacked gratings (Handmer et al.) | [View](examples/simo_080-convergence-stacked_gratings.ipynb) | [View](examples_netgen/simo_080-convergence-stacked_gratings.ipynb) |
| `simo_090-EOT.ipynb` | Extraordinary optical transmission through subwavelength holes | [View](examples/simo_090-EOT.ipynb) | [View](examples_netgen/simo_090-EOT.ipynb) |
| `simo_091-slab_mode_finding.ipynb` | Guided slab mode dispersion finding via determinant zeros | [View](examples/simo_091-slab_mode_finding.ipynb) | [View](examples_netgen/simo_091-slab_mode_finding.ipynb) |
| `Au Nanodisk Array with Interpolators.ipynb` | Plasmonic gold nanodisk array & 3-plane field maps (`xy`, `xz`, `yz`) | [View](examples/Au%20Nanodisk%20Array%20with%20Interpolators.ipynb) | [View](examples_netgen/Au%20Nanodisk%20Array%20with%20Interpolators.ipynb) |
| `Elliptical Nanohole Array (custom target interface).ipynb` | Custom geometry template tutorial & 3-plane maps (`xy`, `xz`, `yz`) | [View](examples/Elliptical%20Nanohole%20Array%20(custom%20target%20interface).ipynb) | [View](examples_netgen/Elliptical%20Nanohole%20Array%20(custom%20target%20interface).ipynb) |
| `Chiral_qBIC_Optical_Chirality.ipynb` | Chiral quasi-BIC metasurface & 3-plane optical chirality $\hat{C}$ | — | [View](examples_netgen/Chiral_qBIC_Optical_Chirality.ipynb) |

---

## What Changed in v2.0 (Summary of Modernization)

1. **Meshing:**
   - In-memory 1D discretization and 2D curved Netgen CSG meshing with second-order elements.
   - Parameterized Python geometry templates (`*_template.py`) alongside legacy Gmsh `.geo` and `.mail` meshes, all consolidated in `EMUstack/backend/data/meshes/`.
2. **1D and 2D Mode Solvers:**
   - Pure-Python NGSolve/SciPy finite-element solvers eliminating all compiled legacy Fortran binaries.
   - High-order polynomial elements in 1D (including coupled TE/TM equations at conical incidence) and mixed vector-scalar finite elements in 2D ($H(\text{curl}) \times H^1$) enforcing exact tangential continuity without spurious modes.
3. **Direct Layer-to-Layer Mode Matching:**
   - Vectorized analytical plane-wave overlap integrals (`J_overlap`) and direct nanostructure-to-nanostructure (`Simmo`-to-`Simmo`) mode coupling without artificial air gaps.
4. **Full Near-Field & Chirality Evaluation:**
   - Reconstruction of 3D magnetic fields from Faraday's law.
   - `FieldBundle` interpolators supporting in-plane slices, fast vectorized vertical cross-sections (`xz`, `yz`), optical chirality density $C$, normalized optical chirality enhancement $\hat{C}$, and Poynting vector flux.
   - Automatic dataset export to labeled `xarray.Dataset` structures.
5. **Notebook-First Examples & Clean Plotting:**
   - `plotting.py` functions return Matplotlib `Figure` objects directly for interactive inline notebook rendering, defaulting to `save_pdf=False` and `save_txt=False` to prevent disk pollution.

---

## License

EMUstack is licensed under the GNU General Public License v3.0 or later ([GPL-3.0-or-later](LICENSE)).
