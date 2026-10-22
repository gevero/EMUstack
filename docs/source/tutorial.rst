Tutorials and Examples Guide
==============================

EMUstack v2.0 provides an interactive, notebook-first workflow. All examples are organized into two distinct directories to demonstrate the two supported meshing paradigms:

1. **Classic Mesh Workflow (``EMUstack/examples/`` — 22 Notebooks):**
   Demonstrates the classic EMUstack workflow where 2D nanostructured layers either load pre-generated meshes (``.mail``, ``.msh``) or invoke Gmsh on parameterized ``.geo`` templates from ``EMUstack/backend/data/meshes/``.

2. **Native In-Memory Netgen Workflow (``EMUstack/examples_netgen/`` — 23 Notebooks):**
   Demonstrates the modern pure-Python workflow where 2D nanostructured layers generate curved quadratic meshes directly in memory via ``geo_params`` dictionaries or Python geometry templates (``*_template.py``) from ``EMUstack/backend/data/meshes/``, requiring zero external Gmsh installation or disk files.

.. note::
   Every tutorial notebook is self-contained and pre-executed with inline figures and xarray outputs. They can be viewed directly on GitHub or run interactively in Jupyter Lab / Notebook.

Simulation Workflow
-------------------

Simulations in EMUstack follow a concise, 5-step pattern:

1. Define Illumination
~~~~~~~~~~~~~~~~~~~~~~

Configure the incident wavelength range, polarization, and angle of incidence:

.. code-block:: python

    import numpy as np
    from EMUstack.backend import objects, materials, stack, plotting

    # Wavelength sweep (e.g. 500 nm to 800 nm, 61 points)
    light = objects.Light(
        wl_nm=np.linspace(500.0, 800.0, 61),
        theta=0.0,
        phi=0.0,
        max_order_PWs=3,
    )

2. Define Materials and Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create homogeneous thin films and periodic nanostructures:

.. code-block:: python

    # Homogeneous substrate (semi-infinite glass)
    glass = materials.Material(1.52)
    sub = objects.ThinFilm(period=600.0, material=glass, loss=False)

    # 2D Periodic nanostructure (Silicon nanopillars)
    # Native Netgen workflow (examples_netgen):
    meta = objects.NanoStruct(
        period=600.0,
        background=materials.Air,
        inclusion_a=materials.Si_c,
        thickness=200.0,
        geo_params={'period': 600.0, 'a1': 240.0, 'lc': 0.1, 'lc2': 2.0},
    )

    # Superstrate (semi-infinite air)
    sup = objects.ThinFilm(period=600.0, material=materials.Air, loss=False)

3. Compute Eigenmodes
~~~~~~~~~~~~~~~~~~~~~

Solve for the electromagnetic modes of each layer across all wavelengths:

.. code-block:: python

    modes_sub = sub.calc_modes(light)
    modes_meta = meta.calc_modes(light, num_BMs=30)
    modes_sup = sup.calc_modes(light)

4. Stack and Calculate Scattering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble the stack and cascade the layers using scattering matrices:

.. code-block:: python

    sim_stack = stack.Stack((modes_sub, modes_meta, modes_sup))
    sim_stack.calc_scat(pol='TE')

5. Visualize and Export Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot spectra or inspect local fields inline without disk side effects:

.. code-block:: python

    # Export spectra to xarray Dataset
    ds = plotting.spectra_to_xarray([sim_stack])

    # Evaluate 3D electromagnetic near-fields on an in-plane slice (xy)
    bundle = plotting.fields_interpolator_in_plane(
        sim_stack, lay_interest=1, z_value=100.0, return_H=True
    )
    Ex, Ey, Ez = bundle.Ex(X, Y), bundle.Ey(X, Y), bundle.Ez(X, Y)

    # Evaluate vertical cross-sections (xz or yz) across the full stack
    cross_bundle = plotting.fields_interpolator_cross_section(
        sim_stack, plane="xz", slice_coord=300.0
    )

Available Tutorial Notebooks
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - Notebook Name
     - In ``examples/``
     - In ``examples_netgen/``
     - Physics Topic
   * - ``simo_010-single_interface.ipynb``
     - Yes
     - Yes
     - Single Non-Dispersive Interface: Fresnel transmission/reflection.
   * - ``simo_011-single_interface-dispersive.ipynb``
     - Yes
     - Yes
     - Single Dispersive Interface: Tabulated refractive indices (Si).
   * - ``simo_020-thin_film_multilayered_stack.ipynb``
     - Yes
     - Yes
     - Multilayered Thin-Film Stack: Fabry–Pérot resonance.
   * - ``simo_021-thin_film_mirror.ipynb``
     - Yes
     - Yes
     - Metallic Back-Reflector: Modeling lossy finite mirrors.
   * - ``simo_030-1D_grating.ipynb``
     - Yes
     - Yes
     - 1D Lamellar Grating: TE vs TM diffraction spectra.
   * - ``simo_031-1D_grating-2_inclusions.ipynb``
     - Yes
     - Yes
     - Multi-Inclusion 1D Grating: Asymmetric diffraction profiles.
   * - ``simo_032-1D_grating-3_inclusions.ipynb``
     - Yes
     - Yes
     - 3-Element 1D Grating: Resonance engineering.
   * - ``simo_040-2D_array.ipynb``
     - Yes
     - Yes
     - 2D Nanowire Array: Photovoltaic absorption and omega-k plots.
   * - ``simo_041-combining_1D_and_2D_array.ipynb``
     - Yes
     - Yes
     - Hybrid 1D/2D Stack: Direct Simmo-to-Simmo mode matching.
   * - ``simo_042-eliptical_holes-CD.ipynb``
     - Yes
     - Yes
     - Elliptical Nanohole Array: Circular dichroism under CPL.
   * - ``simo_050-plotting_fields_1d.ipynb``
     - Yes
     - Yes
     - 1D Bloch Modes & Field Maps: Near-field visualization.
   * - ``simo_051-plotting_fields_2d.ipynb``
     - Yes
     - Yes
     - 2D Array Near-Fields: In-plane and vertical cuts.
   * - ``simo_052-plotting_amplitudes.ipynb``
     - Yes
     - Yes
     - Mode Amplitudes: Bloch and plane-wave coupling strengths.
   * - ``simo_053-plotting_k_space_amps.ipynb``
     - Yes
     - Yes
     - In-Plane Momentum Spectrum: Guided-mode coupling.
   * - ``simo_060-shear_transformations.ipynb``
     - Yes
     - Yes
     - Lateral Layer Shifts: S-matrix shear transformations.
   * - ``simo_070-ultrathin_limit.ipynb``
     - Yes
     - Yes
     - Ultrathin Film Limit: Coherent 50% absorption bound.
   * - ``simo_071-many_substrates.ipynb``
     - Yes
     - Yes
     - Substrate Sweeps: Reusing layer modes across 100 substrates.
   * - ``simo_080-convergence-stacked_gratings.ipynb``
     - Yes
     - Yes
     - Grating Convergence: Replication of Handmer et al. (2010).
   * - ``simo_090-EOT.ipynb``
     - Yes
     - Yes
     - Extraordinary Optical Transmission: Liu & Lalanne (2008).
   * - ``simo_091-slab_mode_finding.ipynb``
     - Yes
     - Yes
     - Guided Slab Modes: Metamaterial dispersion finding.
   * - ``Au Nanodisk Array with Interpolators.ipynb``
     - Yes
     - Yes
     - Showcase: 3-plane (xy, xz, yz) near-fields of plasmonic disks.
   * - ``Elliptical Nanohole Array (custom target interface).ipynb``
     - Yes
     - Yes
     - Showcase: Custom unit-cell templates & 3-plane field maps.
   * - ``Chiral_qBIC_Optical_Chirality.ipynb``
     - No
     - Yes
     - Flagship: Chiral quasi-BIC and 3-plane optical chirality maps.

Native Netgen CSG Tutorials (In-Memory Meshing)
-----------------------------------------------

.. toctree::
   :maxdepth: 1

   examples_netgen/Chiral_qBIC_Optical_Chirality.ipynb
   examples_netgen/Au Nanodisk Array with Interpolators.ipynb
   examples_netgen/Elliptical Nanohole Array (custom target interface).ipynb
   examples_netgen/simo_010-single_interface.ipynb
   examples_netgen/simo_011-single_interface-dispersive.ipynb
   examples_netgen/simo_020-thin_film_multilayered_stack.ipynb
   examples_netgen/simo_021-thin_film_mirror.ipynb
   examples_netgen/simo_030-1D_grating.ipynb
   examples_netgen/simo_031-1D_grating-2_inclusions.ipynb
   examples_netgen/simo_032-1D_grating-3_inclusions.ipynb
   examples_netgen/simo_040-2D_array.ipynb
   examples_netgen/simo_041-combining_1D_and_2D_array.ipynb
   examples_netgen/simo_042-eliptical_holes-CD.ipynb
   examples_netgen/simo_050-plotting_fields_1d.ipynb
   examples_netgen/simo_051-plotting_fields_2d.ipynb
   examples_netgen/simo_052-plotting_amplitudes.ipynb
   examples_netgen/simo_053-plotting_k_space_amps.ipynb
   examples_netgen/simo_060-shear_transformations.ipynb
   examples_netgen/simo_070-ultrathin_limit.ipynb
   examples_netgen/simo_071-many_substrates.ipynb
   examples_netgen/simo_080-convergence-stacked_gratings.ipynb
   examples_netgen/simo_090-EOT.ipynb
   examples_netgen/simo_091-slab_mode_finding.ipynb

Classic Gmsh & Pre-Generated Mesh Tutorials
-------------------------------------------

.. toctree::
   :maxdepth: 1

   examples/Au Nanodisk Array with Interpolators.ipynb
   examples/Elliptical Nanohole Array (custom target interface).ipynb
   examples/simo_010-single_interface.ipynb
   examples/simo_011-single_interface-dispersive.ipynb
   examples/simo_020-thin_film_multilayered_stack.ipynb
   examples/simo_021-thin_film_mirror.ipynb
   examples/simo_030-1D_grating.ipynb
   examples/simo_031-1D_grating-2_inclusions.ipynb
   examples/simo_032-1D_grating-3_inclusions.ipynb
   examples/simo_040-2D_array.ipynb
   examples/simo_041-combining_1D_and_2D_array.ipynb
   examples/simo_042-eliptical_holes-CD.ipynb
   examples/simo_050-plotting_fields_1d.ipynb
   examples/simo_051-plotting_fields_2d.ipynb
   examples/simo_052-plotting_amplitudes.ipynb
   examples/simo_053-plotting_k_space_amps.ipynb
   examples/simo_060-shear_transformations.ipynb
   examples/simo_070-ultrathin_limit.ipynb
   examples/simo_071-many_substrates.ipynb
   examples/simo_080-convergence-stacked_gratings.ipynb
   examples/simo_090-EOT.ipynb
   examples/simo_091-slab_mode_finding.ipynb

.. toctree::
   :hidden:

   screen_sesh
