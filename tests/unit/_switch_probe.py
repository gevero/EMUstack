"""Helper for test_import_styles.py: check that each documented switch is read by the code.

Each switch is set the documented way, to a value the code must reject (or, for
default_mesher, to the other mesher); the probe records what happened. A switch that is not
read lets the calculation run as if nothing was set.

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


def solve(fields=False):
    light = objects.Light(600.0, max_order_PWs=1, theta=0.0, phi=0.0)
    film = objects.ThinFilm(400.0, height_nm='semi_inf', material=materials.Air)
    s = Stack([film.calc_modes(light), layer().calc_modes(light, num_BMs=52), film.calc_modes(light)])
    s.calc_scat(pol='TE')
    if fields:  # the plain interpolators use field_eval.ez_convention
        plotting.fields_interpolator_in_plane(s, lay_interest=1, z_value=50.0)
    return float(s.t_list[-1])


def outcome(fn):
    try:
        fn()
        return 'ran'
    except (ValueError, RuntimeError) as e:
        return type(e).__name__ + ': ' + str(e).split(',')[0]


def mesh_frame():
    mesh = layer().mesh  # keep a reference: NGSolve's vertex iterator does not keep the mesh alive
    ys = [v.point[1] for v in mesh.vertices]
    return 'y>=0' if min(ys) > -1e-9 else 'y<=0'


out = {'T': solve(fields=True), 'mesh': mesh_frame()}
fem_2d.eigensolver = 'bogus'
out['eigensolver'] = outcome(solve)
fem_2d.eigensolver = 'arpack'
fem_2d.overlap_quadrature = 'bogus'
out['overlap_quadrature'] = outcome(solve)
fem_2d.overlap_quadrature = 'legacy'
field_eval.ez_convention = 'bogus'
out['ez_convention'] = outcome(lambda: solve(fields=True))
field_eval.ez_convention = 'physical'
objects.default_mesher = 'netgen'
out['default_mesher'] = mesh_frame()
objects.default_mesher = 'gmsh'
meshing.gmsh_timeout = 1e-3
out['gmsh_timeout'] = outcome(layer)
print('PROBE ' + json.dumps(out))
