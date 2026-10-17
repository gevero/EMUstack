#!/usr/bin/env python
"""
simo_100-chiral_qbic_chirality.py: Simulation of a Chiral quasi-BIC (qBIC) Metasurface.

Demonstrates:
1. High-Q chiral quasi-Bound State in the Continuum (qBIC) using broken-symmetry tilted dielectric dimers.
2. Circular Dichroism (CD = T_LCP - T_RCP) and sharp Fano resonances in reflectance and transmittance.
3. Full electromagnetic near-field reconstruction evaluating both E and H fields.
4. Large, uniform local optical chirality enhancement C_hat(x, y) = C(x, y) / |C_CPL| inside the unit cell.

References:
- Tang & Cohen, Science 332, 333-336 (2011) [Optical Chirality]
- Koshelev, Kivshar et al., Science 367, 288-292 (2020) [Subwavelength Dielectric Resonators and BICs]
- Gorkunov et al., Phys. Rev. Lett. 125, 093901 (2020) [Metasurfaces with Giant Circular Dichroism]
- Overvig et al., Phys. Rev. Lett. 126, 073001 (2021) [Chiral quasi-Bound States in the Continuum]
"""
import os
import sys
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
mesh_dir = os.path.join(backend_dir, "data", "meshes")
for d in (backend_dir, mesh_dir):
    if d not in sys.path:
        sys.path.insert(0, d)

import objects
import materials
import plotting
import field_eval
from stack import Stack

def main():
    start_time = time.time()
    plotting.clear_previous()

    print("=" * 70)
    print("EMUstack Chiral quasi-BIC (qBIC) & Optical Chirality Simulation")
    print("=" * 70)

    # 1. Metasurface and Light Parameters
    period = 700.0        # Unit cell period in nm
    height = 300.0        # Metasurface thickness in nm
    alpha = 0.15          # Tilt asymmetry breaking in-plane mirror symmetry
    num_bms = 18          # Number of Bloch modes

    # Wavelength grid spanning the qBIC resonance
    wavelengths = np.linspace(685.0, 725.0, 9)
    print(f"Lattice period: {period} nm, Thickness: {height} nm, Asymmetry alpha: {alpha}")
    print(f"Simulating {len(wavelengths)} spectral points from {wavelengths[0]} nm to {wavelengths[-1]} nm...")

    # 2. Build Layers
    layer_struct = objects.NanoStruct(
        '2D_array', period, 200.0,
        geo_file='chiral_qbic_template.py',
        geo_params={'alpha': alpha, 'theta0': 0.25, 'rx': 0.09, 'ry': 0.28, 'gap': 0.36, 'lc': 0.18, 'lc2': 1.8},
        background=materials.Material(1.0),
        inclusion_a=materials.Material(3.5),  # High-index dielectric (e.g. silicon / TiO2)
        height_nm=height,
        plotting_fields=1
    )
    sup = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))
    sub = objects.ThinFilm(period=period, height_nm='semi_inf', material=materials.Material(1.0))

    # 3. Spectral Sweep for Circular Polarizations (RCP and LCP)
    T_rcp_list = []
    R_rcp_list = []
    T_lcp_list = []
    R_lcp_list = []
    res_stack_rcp = None
    res_wl = 700.0

    for idx, wl in enumerate(wavelengths):
        t0 = time.time()
        light = objects.Light(wl, max_order_PWs=1, theta=0.0, phi=0.0)

        # Solve modes
        sim_layer = layer_struct.calc_modes(light, num_BMs=num_bms)
        sim_sup = sup.calc_modes(light)
        sim_sub = sub.calc_modes(light)

        # RCP evaluation
        stk_rcp = Stack([sim_sub, sim_layer, sim_sup])
        stk_rcp.calc_scat(pol='RCP')
        T_rcp = float(stk_rcp.t_list[-1])
        R_rcp = float(stk_rcp.r_list[-1])
        T_rcp_list.append(T_rcp)
        R_rcp_list.append(R_rcp)

        # LCP evaluation
        stk_lcp = Stack([sim_sub, sim_layer, sim_sup])
        stk_lcp.calc_scat(pol='LCP')
        T_lcp = float(stk_lcp.t_list[-1])
        R_lcp = float(stk_lcp.r_list[-1])
        T_lcp_list.append(T_lcp)
        R_lcp_list.append(R_lcp)

        cd = T_lcp - T_rcp
        print(f"[{idx+1}/{len(wavelengths)}] wl = {wl:.1f} nm: T_RCP = {T_rcp:.4f}, T_LCP = {T_lcp:.4f}, CD = {cd:+.4f} (dt = {time.time()-t0:.2f}s)")

        # Save the resonance stack at 700 nm for near-field visualization
        if abs(wl - res_wl) < 1.0 or res_stack_rcp is None:
            res_stack_rcp = stk_rcp
            res_wl = wl

    T_rcp_arr = np.array(T_rcp_list)
    R_rcp_arr = np.array(R_rcp_list)
    T_lcp_arr = np.array(T_lcp_list)
    R_lcp_arr = np.array(R_lcp_list)
    CD_arr = T_lcp_arr - T_rcp_arr

    # 4. Near-Field Evaluation (E, H, and Optical Chirality)
    print("\nEvaluating 2D near-field electromagnetic fields and optical chirality at resonance...")
    z_eval = height / 2.0  # Center plane of the metasurface
    bundle = plotting.fields_interpolator_in_plane(res_stack_rcp, lay_interest=1, z_value=z_eval, return_H=True)

    grid_n = 60
    xg = np.linspace(0.02, 0.98, grid_n)
    yg = np.linspace(0.02, 0.98, grid_n)
    X, Y = np.meshgrid(xg, yg)

    E_norm = bundle.AbsE(X, Y)
    H_norm = bundle.AbsH(X, Y)
    C_hat = bundle.C_hat(X, Y)

    max_E2 = float(np.nanmax(E_norm**2))
    max_H2 = float(np.nanmax(H_norm**2))
    max_Chat = float(np.nanmax(np.abs(C_hat)))
    print(f"Near-field peaks: max |E|^2 = {max_E2:.2f}, max |H|^2 = {max_H2:.2f}, max |C_hat| = {max_Chat:.2f}")

    # 5. Plotting & Artifact Generation
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Figure 1: Spectral Response and Circular Dichroism
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.plot(wavelengths, T_rcp_arr, 'r-o', lw=2, label=r'$T_{\mathrm{RCP}}$')
    ax1.plot(wavelengths, T_lcp_arr, 'b-s', lw=2, label=r'$T_{\mathrm{LCP}}$')
    ax1.plot(wavelengths, R_rcp_arr, 'r--', lw=1.5, label=r'$R_{\mathrm{RCP}}$')
    ax1.plot(wavelengths, R_lcp_arr, 'b--', lw=1.5, label=r'$R_{\mathrm{LCP}}$')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Transmittance / Reflectance', fontsize=12)
    ax1.set_title('Chiral qBIC Spectra', fontsize=13)
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(wavelengths, CD_arr, 'm-^', lw=2.5, label=r'$\mathrm{CD} = T_{\mathrm{LCP}} - T_{\mathrm{RCP}}$')
    ax2.axhline(0, color='gray', linestyle=':', lw=1)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Circular Dichroism (CD)', fontsize=12)
    ax2.set_title('Circular Dichroism across Resonance', fontsize=13)
    ax2.legend(loc='best', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    spec_path = os.path.join(output_dir, "chiral_qbic_spectra.png")
    fig1.savefig(spec_path)
    plt.close(fig1)
    print(f"Saved spectral plot to: {spec_path}")

    # Figure 2: Near-Field Maps (|E|^2, |H|^2, and Optical Chirality Enhancement C_hat)
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    im0 = axes[0].imshow(E_norm**2, origin='lower', extent=[0, period, 0, period], cmap='inferno')
    axes[0].set_title(r'$|\mathbf{E}|^2 / |E_0|^2$', fontsize=13)
    axes[0].set_xlabel('x (nm)', fontsize=11)
    axes[0].set_ylabel('y (nm)', fontsize=11)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(H_norm**2, origin='lower', extent=[0, period, 0, period], cmap='magma')
    axes[1].set_title(r'$|\mathbf{H}|^2 / |H_0|^2$', fontsize=13)
    axes[1].set_xlabel('x (nm)', fontsize=11)
    axes[1].set_ylabel('y (nm)', fontsize=11)
    plt.colorbar(im1, ax=axes[1])

    vmax_c = max(float(np.nanpercentile(np.abs(C_hat), 98)), 1.0)
    im2 = axes[2].imshow(C_hat, origin='lower', extent=[0, period, 0, period], cmap='coolwarm', vmin=-vmax_c, vmax=vmax_c)
    axes[2].set_title(r'Chirality Enhancement $\hat{C} = C / |C_{\mathrm{CPL}}| $', fontsize=13)
    axes[2].set_xlabel('x (nm)', fontsize=11)
    axes[2].set_ylabel('y (nm)', fontsize=11)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    nearfield_path = os.path.join(output_dir, "chiral_qbic_nearfields.png")
    fig2.savefig(nearfield_path)
    plt.close(fig2)
    print(f"Saved near-field map to: {nearfield_path}")

    elapsed = time.time() - start_time
    print(f"\nSimulation successfully completed in {elapsed:.2f} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    main()
