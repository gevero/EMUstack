# Changelog

## Unreleased

### Fixed
- On Windows, a gmsh run that hit `meshing.gmsh_timeout` hung instead of stopping: the pip
  `gmsh.bat` wrapper started a second process that kept running. gmsh now runs as one process,
  and the whole process tree is killed on timeout.
- The test suite runs on macOS and Windows in CI.

## 2.0.0rc1 (2026-11-04)

A pure-Python port of EMUstack. Version 1 (Fortran backend) is tagged `v1-fortran` and lives at
[bjornsturmberg/EMUstack](https://github.com/bjornsturmberg/EMUstack).

### Changed
- **No Fortran.** 2D Bloch modes are solved with NGSolve/Netgen finite elements, 1D modes with
  NumPy/SciPy. Everything installs from pip wheels (`pip install emustack`); nothing is compiled.
- **Same results as version 1.** A parity suite compares spectra, `R_net`/`T_net` and fields
  with the Fortran version on the example cases (`tests/test_legacy_parity.py`).
- **Old scripts keep working** with only their imports changed: same classes, signatures and
  defaults, including old mesh-template file names, `.geo` placeholder names and `geo_params` keys.
- **Plotting returns figures** and writes no files unless asked (`save_pdf=True`, `save_txt=True`).
- **E_z in field maps** uses the physical convention (div E = 0) by default;
  `field_eval.ez_convention = 'legacy'` restores version 1's maps.

### Added
- In-memory Netgen meshing with curved second-order elements and Python geometry templates
  (`*_msh_template.py`), next to the gmsh `.geo` templates (still the default mesher).
- One naming scheme for `.geo` and `.py` templates and `meshing.unit_cell`, with the old names
  kept as aliases.
- Magnetic fields, Poynting vector and optical chirality from `FieldBundle` field interpolators,
  in-plane and in vertical cross-sections; spectra and fields export to `xarray`.
- Example notebooks for both meshers, including a chiral quasi-BIC metasurface (Gorkunov et al.)
  sampled with `adaptive` in parallel.
- `geo_params` alone describes a built-in shape (period, diameters, mesh sizes, ...); unknown
  keys raise an error.
- gmsh is stopped after `meshing.gmsh_timeout` seconds (600 by default) and, on Linux, when the
  Python process dies.
- `meshing.list_templates()` and `meshing.copy_template()`: the 30 shipped templates are found by
  file name from any directory; a copy in the working directory takes priority.

### Fixed
- Module-level settings (`fem_2d.eigensolver`, `fem_2d.overlap_quadrature`,
  `field_eval.ez_convention`, `meshing.gmsh_timeout`) were ignored when set through
  `from EMUstack.backend import ...`, because the backend was loaded twice under two names.
  Each module now exists once, whether imported from the package or by bare name from a legacy
  script, and `import EMUstack` no longer changes `sys.path` (it hid users' own modules named
  `plotting`, `stack`, ...).
- In-plane field interpolators take x and y from 0 to 1 (y up to `period_y/period`) for every
  mesher. Before, layers meshed by gmsh needed y from -1 to 0, so field maps drawn on 0..1
  came out blank.
- `fields_interpolator_in_plane` returned `None` and printed a message on any `ValueError`
  (inherited from version 1, meant for 1D layers); errors on 2D layers now surface. An invalid
  `field_eval.ez_convention` set at run time raises instead of meaning 'legacy'.
- The 16-inclusion supercell template meshed part of the cell twice; it is now a 4x4 grid
  (also broken in version 1).
- The strip shapes hang gmsh when the inclusion is not narrower than the strip (as in
  version 1); such parameters are now rejected. The square strip template got the boundary
  lines the new mesh import needs.
- Cached mesh names of the 9- and 16-inclusion supercells ignored diameters 6 to 16 (also in
  version 1).

### Known issues
- Only tested on Linux. On a minimal Linux system the NGSolve and gmsh wheels need
  `libopenblas0 libglu1-mesa libgl1 libgomp1 libxcursor1 libxft2 libxinerama1`.
- Too few Bloch modes (`num_BMs`) can break the symmetry of highly symmetric cells; check a
  symmetry-protected quantity (for example |T_RCP - T_LCP| in an achiral C4 cell) when you change
  it.
