"""NGSolve API ground-truth capability tests.

Validates each verified API row against the pinned NGSolve version in this environment.
"""
import pytest
import numpy as np
import scipy.sparse as sp
import ngsolve as ngs
import netgen.geom2d as g2
from netgen.meshing import IdentificationType


@pytest.fixture
def unit_square_mesh():
    """Create a periodic unit cell mesh with SplineGeometry."""
    geo = g2.SplineGeometry()
    pts = [geo.AddPoint(*q) for q in [(0, 0), (1, 0), (1, 0.8), (0, 0.8)]]
    bot = geo.Append(["line", pts[0], pts[1]], bc="bottom", maxh=0.15)
    rgt = geo.Append(["line", pts[1], pts[2]], bc="right", maxh=0.15)
    geo.Append(["line", pts[3], pts[2]], leftdomain=0, rightdomain=1, bc="top", maxh=0.15, copy=bot)
    geo.Append(["line", pts[0], pts[3]], leftdomain=0, rightdomain=1, bc="left", maxh=0.15, copy=rgt)
    geo.SetMaterial(1, "bg")
    ngm = geo.GenerateMesh(maxh=0.15)
    ngm.Curve(2)
    return ngs.Mesh(ngm)


def test_spaces(unit_square_mesh):
    """Row 1-3: HCurl, H1, Product space, and TnT."""
    mesh = unit_square_mesh
    fes_curl = ngs.HCurl(mesh, order=2, complex=True)
    fes_h1 = ngs.H1(mesh, order=3, complex=True)
    fes = fes_curl * fes_h1
    
    assert fes_curl.ndof > 0
    assert fes_h1.ndof > 0
    assert fes.ndof == fes_curl.ndof + fes_h1.ndof
    
    (u, z), (v, w) = fes.TnT()
    assert u is not None and z is not None
    assert v is not None and w is not None


def test_periodic_space_reduction(unit_square_mesh):
    """Row 4: Periodic wrapper reduces free dofs on product space."""
    mesh = unit_square_mesh
    fes = ngs.HCurl(mesh, order=2, complex=True) * ngs.H1(mesh, order=3, complex=True)
    kx, ky = 0.31, 0.17
    phase_x = np.exp(-1j * kx * 1.0)
    phase_y = np.exp(-1j * ky * 0.8)
    fes_per = ngs.Periodic(fes, phase=[phase_x, phase_y])
    
    free_dofs = fes_per.FreeDofs().NumSet()
    assert free_dofs < fes.ndof, "Periodic space failed to reduce free dofs"


def test_periodic_mesh_identifications(unit_square_mesh):
    """Row 5-6: SplineGeometry periodic identifications and node pairs."""
    mesh = unit_square_mesh
    assert mesh.ngmesh.GetNrIdentifications() == 4
    pairs = mesh.GetPeriodicNodePairs(ngs.fem.NODE_TYPE.VERTEX)
    assert len(pairs) > 0


def test_forms_and_partial_derivatives(unit_square_mesh):
    """Row 7, 10, 15: BilinearForm with Conj, grad(w)[i] partial derivatives."""
    mesh = unit_square_mesh
    fes = ngs.HCurl(mesh, order=2, complex=True) * ngs.H1(mesh, order=3, complex=True)
    (u, z), (v, w) = fes.TnT()
    
    a = ngs.BilinearForm(fes, symmetric=False)
    a += (ngs.curl(u) * ngs.Conj(ngs.curl(v))
          + u[0] * ngs.Conj(ngs.grad(w)[0]) + u[1] * ngs.Conj(ngs.grad(w)[1])
          - (ngs.grad(z)[0] * ngs.Conj(ngs.grad(w)[0]) + ngs.grad(z)[1] * ngs.Conj(ngs.grad(w)[1]))
          ) * ngs.dx
    a.Assemble()
    assert a.mat.shape == (fes.ndof, fes.ndof)


def test_scalar_integrate(unit_square_mesh):
    """Row 8: Scalar moments of FEM fields via ngs.Integrate."""
    mesh = unit_square_mesh
    fes = ngs.H1(mesh, order=1, complex=False)
    gfu = ngs.GridFunction(fes)
    gfu.Set(1.0)
    
    area = ngs.Integrate(gfu, mesh)
    # 1.0 * 0.8 = 0.8
    assert abs(area - 0.8) < 1e-10


def test_material_cf(unit_square_mesh):
    """Row 10: MaterialCF requires string keys."""
    mesh = unit_square_mesh
    mat_cf = mesh.MaterialCF({"bg": ngs.CF(3.4**2), "default": ngs.CF(1.0)})
    assert mat_cf is not None


def test_csr_export(unit_square_mesh):
    """Row 16: Export BilinearForm to scipy CSR."""
    mesh = unit_square_mesh
    fes = ngs.H1(mesh, order=1, complex=True)
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes)
    a += (ngs.grad(u)[0] * ngs.Conj(ngs.grad(v)[0]) + u * ngs.Conj(v)) * ngs.dx
    a.Assemble()
    
    vals, col, row = a.mat.CSR()
    mat_sp = sp.csr_matrix((np.array(vals), np.array(col), np.array(row)), shape=a.mat.shape)
    assert mat_sp.shape == (fes.ndof, fes.ndof)
    assert mat_sp.nnz > 0


def test_curved_geometry(unit_square_mesh):
    """Row 18: Curve activation on netgen mesh."""
    mesh = unit_square_mesh
    assert mesh.ngmesh.GetCurveOrder() >= 2


def test_arnoldi_solver(unit_square_mesh):
    """Row 14: ArnoldiSolver on periodic space."""
    mesh = unit_square_mesh
    fes = ngs.H1(mesh, order=1, complex=True)
    fes_per = ngs.Periodic(fes, phase=[1.0, 1.0])
    u, v = fes_per.TnT()
    
    a = ngs.BilinearForm(fes_per)
    a += (ngs.grad(u)[0]*ngs.Conj(ngs.grad(v)[0]) + ngs.grad(u)[1]*ngs.Conj(ngs.grad(v)[1])) * ngs.dx
    m = ngs.BilinearForm(fes_per)
    m += u * ngs.Conj(v) * ngs.dx
    a.Assemble()
    m.Assemble()
    
    gfs = [ngs.GridFunction(fes_per) for _ in range(3)]
    vals = ngs.ArnoldiSolver(a.mat, m.mat, fes_per.FreeDofs(), [g.vec for g in gfs], shift=1.0)
    assert len(vals) == 3


def test_absent_apis(unit_square_mesh):
    """Verify absent/non-functional APIs as listed in §2.2."""
    mesh = unit_square_mesh
    assert not hasattr(mesh, "EnablePeriodic")
    assert not hasattr(ngs, "MatrixInverse")
    assert not hasattr(ngs, "XYZ")
    assert not hasattr(ngs, "Diff")
    assert not hasattr(ngs, "diff")
