"""
    test_case_3_simple_mk_msh.py is a simulation example for EMUstack.

    Copyright (C) 2015  Bjorn Sturmberg, Kokou Dossou, Felix Lawrence

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
Test simulation of a relatively simple structure;
a dilute InP nanowire array.
Tests the creation of new .mail file via Gmsh.
"""

import time
import datetime
import numpy as np
import sys
from multiprocessing import Pool
sys.path.append("../backend/")

import objects
import materials
import plotting
from stack import *
import testing
from numpy.testing import assert_allclose as assert_ac
from numpy.testing import assert_equal



def setup_module(module):
    ################ Light parameters #####################

    # Set up light objects
    wavelengths = np.array([700])
    light_list  = [objects.Light(wl, max_order_PWs = 2, theta = 0.0, phi = 0.0) for wl in wavelengths]
    light = light_list[0]

    #period must be consistent throughout simulation!!!
    period = 500
    NW_diameter = 120
    num_BMs = 40
    NW_array = objects.NanoStruct('2D_array', period, NW_diameter, height_nm = 2330,
        inclusion_a = materials.InP, background = materials.Air,
        loss = True, make_mesh_now = True, force_mesh = True,
        lc_bkg = 0.07, lc2= 1.5, lc3= 2.0)

    superstrate  = objects.ThinFilm(period = period, height_nm = 'semi_inf',
        material = materials.Air, loss = False)

    substrate = objects.ThinFilm(period = period, height_nm = 'semi_inf',
        material = materials.SiO2, loss = False)

    ################ Evaluate each layer individually ##############
    sim_superstrate = superstrate.calc_modes(light)
    sim_NW_array = NW_array.calc_modes(light, num_BMs = num_BMs)
    sim_substrate = substrate.calc_modes(light)

    stack = Stack((sim_substrate, sim_NW_array, sim_superstrate))
    stack.calc_scat(pol = 'TE')
    module.stack_list = [stack]

    plotting.t_r_a_write_files(stack_list, wavelengths)


    # # SAVE DATA AS REFERENCE
    # # Only run this after changing what is simulated - this
    # # generates a new set of reference answers to check against
    # # in the future
    # testing.save_reference_data("case_3", stack_list)



from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent

def results_match_reference(filename):
    rtol = 1e-3
    atol = 1e-1
    ref_path = _THIS_DIR / "ref/case_3" / filename
    reference = np.loadtxt(ref_path)
    result    = np.loadtxt(filename)
    np.testing.assert_allclose(result, reference, rtol, atol, filename)

def test_txt_results():
    result_files = (
        "Absorptance_stack0001.txt",
        "Lay_Absorb_0_stack0001.txt",
        "Lay_Trans_0_stack0001.txt",
        "Reflectance_stack0001.txt",
        "Transmittance_stack0001.txt",
        )
    for f in result_files:
        results_match_reference(f)

def test_stack_list_matches_saved(casefile_name = 'case_3'):
    rtol = 1.5e-1
    atol = 1.5e-0
    ref_path = _THIS_DIR / f"ref/{casefile_name}.npz"
    ref = np.load(ref_path, allow_pickle=True, encoding='latin1')
    assert_equal(len(stack_list), len(ref['stack_list']))
    for stack, rstack in zip(stack_list, ref['stack_list']):
        assert_equal(len(stack.layers), len(rstack['layers']))
        lbl_s = "wl = %f, " % stack.layers[0].light.wl_nm
        for i, (lay, rlay) in enumerate(zip(stack.layers, rstack['layers'])):
            lbl_l = lbl_s + "lay %i, " % i
            assert_ac(np.asarray(lay.R12), np.asarray(rlay['R12']), rtol, atol, lbl_l + 'R12')
            # Per Section 3.6, individual T12 and T21 scale with mode gauge C, while T21 @ T12 is gauge-invariant
            assert_ac(np.asarray(lay.T21 @ lay.T12), np.asarray(rlay['T21'] @ rlay['T12']), rtol, atol, lbl_l + 'T21_T12')
            assert_ac(np.asarray(lay.k_z), np.asarray(rlay['k_z']), rtol, atol, lbl_l + 'k_z')
            #TODO: assert_ac(lay.sol1, rlay['sol1'])
        assert_ac(np.asarray(stack.R_net), np.asarray(rstack['R_net']), rtol, atol, lbl_s + 'R_net')
        assert_ac(np.asarray(stack.T_net), np.asarray(rstack['T_net']), rtol, atol, lbl_s + 'T_net')

if __name__ == '__main__':
    setup_module(sys.modules[__name__])
    test_txt_results()
    test_stack_list_matches_saved()
    print("test_case_3 passed successfully!")

