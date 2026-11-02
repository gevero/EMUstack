"""Unit tests for the Universal Custom Geometry Pipeline (.geo and .py).

Verifies:
1. mesh_from_geo: Compiles and imports arbitrary Gmsh .geo templates into NGSolve.
2. mesh_from_py: Loads and builds arbitrary Netgen .py geometry templates in-memory.
3. Typo detection: Raises KeyError on unrecognized parameters in .geo templates.
4. Parameter validation: Raises TypeError on unrecognized parameters in .py templates.
5. Pure dictionary configuration: NanoStruct accepts all parameters via geo_params.
6. Dictionary override: geo_params strictly overrides legacy arguments without conflicts.
"""
import pytest
import numpy as np
import ngsolve as ngs
import sys
from pathlib import Path


from EMUstack.backend import meshing, objects, materials


def test_mesh_from_geo_universal():
    """Test loading and compiling ellipse_msh_template.geo via universal Gmsh bridge."""
    geo_params = {
        'a1': 280.0,
        'ellipticity': 0.4,
        'angle': np.pi / 4.0,
        'lc': 0.15,
        'lc2': 4.0,
        'lc3': 1.0,
    }
    mesh = meshing.mesh_from_geo(
        "ellipse_msh_template.geo",
        geo_params=geo_params,
        period=450.0,
        period_y=450.0
    )
    assert isinstance(mesh, ngs.Mesh)
    assert mesh.dim == 2
    assert mesh.ne > 0
    assert mesh.nv > 0
    assert "bg" in mesh.GetMaterials()
    assert "inc2" in mesh.GetMaterials()


def test_mesh_from_py_universal():
    """Test loading and building ellipse_msh_template.py via universal Netgen bridge."""
    geo_params = {
        'a1': 280.0,
        'ellipticity': 0.4,
        'angle': np.pi / 4.0,
        'lc': 0.15,
        'lc2': 4.0,
        'lc3': 1.0,
    }
    mesh = meshing.mesh_from_py(
        "ellipse_msh_template.py",
        geo_params=geo_params,
        period=450.0,
        period_y=450.0
    )
    assert isinstance(mesh, ngs.Mesh)
    assert mesh.dim == 2
    assert mesh.ne > 0
    assert mesh.nv > 0
    assert "bg" in mesh.GetMaterials()
    assert "inc2" in mesh.GetMaterials()


def test_geo_typo_raises_key_error():
    """Test that a typo in geo_params for a .geo template raises KeyError."""
    bad_params = {
        'a1': 280.0,
        'raduis': 100.0,  # Typo! Not present in template
    }
    with pytest.raises(KeyError, match="Parameter 'raduis' in geo_params was not found"):
        meshing.mesh_from_geo(
            "ellipse_msh_template.geo",
            geo_params=bad_params,
            period=450.0
        )


def test_py_unexpected_param_raises_type_error():
    """Test that an unexpected parameter in geo_params for a .py template raises TypeError."""
    bad_params = {
        'a1': 280.0,
        'unsupported_param_xyz': 42.0,
    }
    with pytest.raises(TypeError, match="Unexpected parameter 'unsupported_param_xyz'"):
        meshing.mesh_from_py(
            "ellipse_msh_template.py",
            geo_params=bad_params,
            period=450.0
        )


def test_nanostruct_pure_dict_configuration():
    """Test creating a NanoStruct layer purely via geo_params with zero legacy arguments (.py and .geo)."""
    for template in ('ellipse_msh_template.py', 'ellipse_msh_template.geo'):
        ns = objects.NanoStruct(
            '2D_array',
            height_nm=60.0,
            inclusion_a=materials.Air,
            background=materials.Material(3.5),
            geo_file=template,
            geo_params={
                'period': 500.0,
                'period_y': 500.0,
                'a1': 200.0,
                'ellipticity': 0.3,
                'angle': 0.2,
                'lc': 0.12,
                'lc2': 3.0,
            }
        )
        assert ns.period == 500.0
        assert ns.period_y == 500.0
        assert ns.mesh is not None
        assert ns.mesh.ne > 0


def test_nanostruct_dict_overrides_legacy():
    """Test that geo_params strictly overrides legacy positional/keyword arguments."""
    ns = objects.NanoStruct(
        '2D_array',
        period=400.0,
        diameter1=100.0,
        height_nm=50.0,
        geo_file='ellipse_msh_template.py',
        geo_params={
            'period': 600.0,  # Overrides period=400.0
            'a1': 350.0,      # Overrides diameter1=100.0
            'ellipticity': 0.5,
        }
    )
    assert ns.period == 600.0
    harvested = ns.harvest_geo_params()
    assert harvested['a1'] == 350.0
    assert harvested['period'] == 600.0
