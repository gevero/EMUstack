"""Helper for test_import_styles.py: apply each documented switch, print what it changes.

Usage: python _switch_probe.py package|legacy BACKEND_DIR
"""
import json
import sys

import numpy as np

style, backend_dir = sys.argv[1], sys.argv[2]
if style == 'package':
    from EMUstack.backend import objects, materials, plotting, fem_2d, field_eval, meshing
    from EMUstack.backend.stack import Stack
else:  # a legacy script: backend/ on sys.path, bare module names
    sys.path.append(backend_dir)
    import objects, materials, plotting, fem_2d, field_eval, meshing
    from stack import Stack


def layer():
    return objects.NanoStruct('2D_array', 400.0, 160.0, height_nm=100.0, plotting_fields=1,
                              inclusion_a=materials.Material(2.0), lc_bkg=0.15, lc2=2.0)


def solve():
    light = objects.Light(600.0, max_order_PWs=1, theta=0.0, phi=0.0)
    film = objects.ThinFilm(400.0, height_nm='semi_inf', material=materials.Air)
    s = Stack([film.calc_modes(light), layer().calc_modes(light, num_BMs=40), film.calc_modes(light)])
    s.calc_scat(pol='TE')
    re_ez = plotting.fields_interpolator_in_plane(s, lay_interest=1, z_value=50.0)[4]
    u = np.linspace(0.1, 0.9, 9)
    U, V = np.meshgrid(u, u)
    ez = np.ma.filled(np.ma.masked_invalid(np.asarray(re_ez(U, V), dtype=float)), np.nan)
    return {'T': float(s.t_list[-1]), 'ReEz': float(np.nansum(np.abs(ez)))}


def mesh_kind():
    m = layer().mesh
    ys = [v.point[1] for v in m.vertices]
    return 'y>=0' if min(ys) > -1e-9 else 'y<=0'


def gmsh_stopped():
    try:
        layer()
        return False
    except RuntimeError as e:
        return 'did not finish' in str(e)


out = {'default': solve(), 'mesh_default': mesh_kind()}
fem_2d.eigensolver = 'ngsolve'
out['eigensolver'] = solve()
fem_2d.eigensolver = 'arpack'
fem_2d.overlap_quadrature = 'ngsolve'
out['overlap_quadrature'] = solve()
fem_2d.overlap_quadrature = 'legacy'
field_eval.ez_convention = 'legacy'
out['ez_convention'] = solve()
field_eval.ez_convention = 'physical'
objects.default_mesher = 'netgen'
out['default_mesher'] = mesh_kind()
objects.default_mesher = 'gmsh'
meshing.gmsh_timeout = 1e-3
out['gmsh_timeout'] = gmsh_stopped()
print('PROBE ' + json.dumps(out))
