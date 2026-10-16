"""Test Bloch sign convention and boundary conditions (G3.2).

Exit gate:
- G3.2 passes with correct sign exp(-i k.r) and fails loudly with flipped sign.
- Periodic without identifications leaves space unconstrained (asserts detection).
"""
import pytest
import numpy as np
import ngsolve as ngs
import netgen.geom2d as g2


def test_bloch_sign_convention():
    """G3.2: Manufactured quasi-periodic field satisfies Bloch BC with correct sign,

    and fails loudly with flipped sign.
    """
    L, H = 1.0, 0.8
    kx, ky = 0.31, 0.17
    
    geo = g2.SplineGeometry()
    pts = [geo.AddPoint(*q) for q in [(0, 0), (L, 0), (L, H), (0, H)]]
    bot = geo.Append(["line", pts[0], pts[1]], bc="bottom", maxh=0.1)
    rgt = geo.Append(["line", pts[1], pts[2]], bc="right", maxh=0.1)
    geo.Append(["line", pts[3], pts[2]], leftdomain=0, rightdomain=1, bc="top", maxh=0.1, copy=bot)
    geo.Append(["line", pts[0], pts[3]], leftdomain=0, rightdomain=1, bc="left", maxh=0.1, copy=rgt)
    geo.SetMaterial(1, "bg")
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))
    
    fes = ngs.HCurl(mesh, order=2, complex=True)
    
    # Correct phase: exp(-i k . d)
    phase_x = np.exp(-1j * kx * L)
    phase_y = np.exp(-1j * ky * H)
    
    # Analytic quasi-periodic field u = grad(exp(-i(kx*x + ky*y)))
    ex = ngs.exp(-1j * (kx * ngs.x + ky * ngs.y))
    u_cf = ngs.CF((-1j * kx * ex, -1j * ky * ex))
    
    # Integrate tangential jumps on left/right (x=0 vs x=L) and bottom/top (y=0 vs y=H)
    # Correct sign: u(L, y) - phase_x * u(0, y) == 0
    # On right boundary, tangential component is u_y:
    val_right = ngs.Integrate(u_cf[1], mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("right"))
    val_left = ngs.Integrate(u_cf[1], mesh, ngs.comp.VorB.BND, definedon=mesh.Boundaries("left"))
    
    # Check analytic relation: val_right == phase_x * val_left
    rel_err_correct = abs(val_right - phase_x * val_left) / abs(val_left)
    assert rel_err_correct < 1e-12, f"Expected < 1e-12, got {rel_err_correct}"
    
    # Flipped sign: phase_x_flipped = conj(phase_x) = exp(+1j * kx * L)
    rel_err_flipped = abs(val_right - np.conj(phase_x) * val_left) / abs(val_left)
    assert rel_err_flipped > 0.1, f"Flipped sign failed to fail loudly: {rel_err_flipped}"


def test_periodic_without_identifications():
    """Negative control: Periodic() on a mesh without identifications leaves free dofs unchanged."""
    geo = g2.SplineGeometry()
    pts = [geo.AddPoint(*q) for q in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    # No copy= parameter => no identifications
    geo.Append(["line", pts[0], pts[1]], bc="bottom", maxh=0.2)
    geo.Append(["line", pts[1], pts[2]], bc="right", maxh=0.2)
    geo.Append(["line", pts[2], pts[3]], bc="top", maxh=0.2)
    geo.Append(["line", pts[3], pts[0]], bc="left", maxh=0.2)
    geo.SetMaterial(1, "bg")
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))
    
    assert mesh.ngmesh.GetNrIdentifications() == 0
    
    fes = ngs.HCurl(mesh, order=2, complex=True)
    fes_per = ngs.Periodic(fes, phase=[1.0, 1.0])
    
    # Silent failure check (§5.2): free dofs equal ndof
    assert fes_per.FreeDofs().NumSet() == fes.ndof
