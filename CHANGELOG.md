# Changelog

## 2.0.0 (unreleased)

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

### Fixed
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
