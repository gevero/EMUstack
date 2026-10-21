"""
    plotting.py is a subroutine of EMUstack that contains numerous plotting
    routines.

    Copyright (C) 2015  Bjorn Sturmberg, Kokou Dossou, Felix Lawrence

    EMUstack is free software: you can redistribute it and/or modify
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
import os
import subprocess
import numpy as np
from numpy import sqrt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from EMUstack.backend import paths
try:
    from EMUstack.backend import objects, mode_calcs
except ImportError:
    import objects
    import mode_calcs


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}

# font = {'size'   : 14}
# matplotlib.rc('font', **font)
linesstrength = 2.5
title_font = 10


#### Natural constants ########################################################
ASTM15_tot_I   = 900.084            # Integral ASTM 1.5 solar irradiance W/m**2
Plancks_h      = 6.62606957*1e-34   # Planck's constant
speed_c        = 299792458          # Speed of light in vacuum
charge_e       = 1.602176565*1e-19  # Charge of an electron
###############################################################################


#### Short utility functions ##################################################
def clear_previous():
    """ Delete all files of specified type as well as field directories. """

    devnull = open(os.devnull, 'wb')

    type_list = ['*.npz', '*.pdf', '*.txt', '*.gif', '*.png', '*.log', '*.svg',
    'fields_vertically -r', 'in_plane_fields -r', 'Bloch_fields -r',
    'field_values -r', '3d_fields -r']
    for typ in type_list:
        try:
            files_rm = 'rm %s'% typ
            subprocess.call(files_rm, shell = True, stderr=devnull)
        except:
            pass


def zeros_int_str(zero_int):
    """ Convert integer into string with '0' in place of ' '. """
    # if zero_int == 0:
    #     fmt_string = '0000'
    # else:
    #     string = '%4.0f' % zero_int
    #     fmt_string = string.replace(' ','0')
    string = '%4.0f' % zero_int
    fmt_string = string.replace(' ','0')
    return fmt_string


def tick_function(energies):
    """ Convert energy in eV into wavelengths in nm """
    wls = Plancks_h*speed_c/(energies*charge_e)*1e9
    return wls


def _is_simmo(layer):
    return isinstance(layer, mode_calcs.Simmo) or type(layer).__name__ == "Simmo"


def _is_anallo(layer):
    return isinstance(layer, mode_calcs.Anallo) or type(layer).__name__ == "Anallo"


def _is_nanostruct(struct):
    return isinstance(struct, objects.NanoStruct) or type(struct).__name__ == "NanoStruct"


def max_n(stacks_list):
    """ Find maximum refractive index n in stacks_list. """
    ns = []
    for s in stacks_list:
        for l in s.layers:
            if _is_anallo(l):
                ns.append(l.n())
            if _is_simmo(l):
                wl = l.light.wl_nm
                ns.append(l.structure.background.n(wl))
                ns.append(l.structure.inclusion_a.n(wl))
                ns.append(l.structure.inclusion_b.n(wl))
    return np.real(np.max(ns))


def gen_params_string(stack, layer=1):
    """ Generate the string of simulation info that is to be printed \
        at the top of plots.
    """
    param_layer = stack[0].layers[layer].structure
    # Plot t,r,a for each layer & total, then save each to text files
    if _is_nanostruct(param_layer):
        params_2_print = 'ff = %5.3f, '% param_layer.ff
        params_2_print += 'd = %(period)d, a1 = %(diameter)d, '% {
        'period'        : param_layer.period, 'diameter' : param_layer.diameter1,}
        if param_layer.diameter2 != 0: params_2_print += 'a2 = %(rad)d '% {'rad' : param_layer.diameter2,}
        if param_layer.periodicity == '2D_array':
            if param_layer.diameter3  != 0: params_2_print += 'a3 = %(rad)d '%    {'rad' : param_layer.diameter3,}
            if param_layer.diameter4  != 0: params_2_print += 'a4 = %(rad)d '%    {'rad' : param_layer.diameter4,}
            if param_layer.diameter5  != 0: params_2_print += '\na5 = %(rad)d '%  {'rad' : param_layer.diameter5,}
            if param_layer.diameter6  != 0: params_2_print += 'a6 = %(rad)d '%    {'rad' : param_layer.diameter6,}
            if param_layer.diameter7  != 0: params_2_print += 'a7 = %(rad)d '%    {'rad' : param_layer.diameter7,}
            if param_layer.diameter8  != 0: params_2_print += 'a8 = %(rad)d '%    {'rad' : param_layer.diameter8,}
            if param_layer.diameter9  != 0: params_2_print += 'a9 = %(rad)d \n'%  {'rad' : param_layer.diameter9,}
            if param_layer.diameter10 != 0: params_2_print += 'a10 = %(rad)d '%   {'rad' : param_layer.diameter10,}
            if param_layer.diameter11 != 0: params_2_print += 'a11 = %(rad)d '%   {'rad' : param_layer.diameter11,}
            if param_layer.diameter12 != 0: params_2_print += 'a12 = %(rad)d '%   {'rad' : param_layer.diameter12,}
            if param_layer.diameter13 != 0: params_2_print += 'a13 = %(rad)d '%   {'rad' : param_layer.diameter13,}
            if param_layer.diameter14 != 0: params_2_print += 'a14 = %(rad)d '%   {'rad' : param_layer.diameter14,}
            if param_layer.diameter15 != 0: params_2_print += 'a15 = %(rad)d '%   {'rad' : param_layer.diameter15,}
            if param_layer.diameter16 != 0: params_2_print += 'a16 = %(rad)d \n'% {'rad' : param_layer.diameter16,}
            if param_layer.inc_shape == 'square': params_2_print += '\nSquare ins '
            if param_layer.inc_shape == 'ellipse': params_2_print += '\nEllipticity = %(rad)5.3f '% {'rad' : param_layer.ellipticity}
        elif param_layer.periodicity == '1D_array':
            params_2_print += ''
        params_2_print += '%(BMs)dBMs, PW_radius = %(PWs)d, '% \
        {'BMs' : stack[0].layers[layer].num_BMs, \
        'PWs' : stack[0].layers[layer].max_order_PWs}
    else:
        params_2_print = 'PW_radius = %(PWs)d, ' \
        % {'PWs' : stack[0].layers[layer].max_order_PWs, }

    # light = stack[0].layers[layer].light
    # k_pll = light.k_pll * param_layer.period
    # params_2_print += r'$k_\parallel d$ = '
    # tmp3 = '(%(kx)1.4f, %(ky)1.4f), '% {'kx' : k_pll[0], 'ky' : k_pll[1]}
    # params_2_print += r'$\theta$ = %(theta)6.2f, $\phi$ = %(phi)6.2f, '% {
    # 'theta' : light.theta,'phi' : light.phi, }

    return params_2_print
###############################################################################


#### Standard plotting of spectra #############################################
def t_r_a_plots(stacks_list, xvalues=None, xlabel='', params_layer=1,
    active_layer_nu=1, stack_label=1, ult_eta=False, J_sc=False,
    weight_spec=False, extinct=False, add_height=0, add_name='',
    save_pdf=False, save_txt=False, plot_layers=False, set_y_lim=True, label_eV=False):
    """ Plot t, r, a for each layer & in total.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            xvalues  (list): The values stacks_list is to be plotted as a \
                function of.

            params_layer  (int): The index in stacks_list of the layer for \
                which the geometric parameters are put in the title of \
                the plots.

            active_layer_nu  (int): The index in stacks_list (from bottom) \
                of the layer for which the ult_eta and/or J_sc are calculated.

            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            ult_eta  (bool): If True; calculate the 'ultimate efficiency'.

            J_sc  (bool): If True; calculate the idealised short circuit \
                current.

            weight_spec  (bool): If True; plot t, r, a spectra weighted by \
                the ASTM 1.5 solar spectrum.

            extinct  (bool): If True; calculate the extinction ratio in \
                transmission.

            add_height  (float): Print the heights of :Stack: layer in title.

            add_name  (str): Add add_name to title.

            save_pdf  (bool): If True; save spectra as pdf files. \
                False by default.

            save_txt  (bool): If True; save spectra data to text \
                files. False by default.

            plot_layers (bool): If True; plot per-layer spectra even if save_pdf is False.

            set_y_lim  (bool): If True; set y limits to (0,1).

            label_eV  (bool): If True; add energy label in eV.
    """

    height_list = stacks_list[0].heights_nm()[::-1]
    params_2_print = gen_params_string(stacks_list, params_layer)
    params_2_print += '\n'r'$h_t,...,h_b$ = '
    params_2_print += ''.join('%4f, ' % num for num in height_list)

    if xvalues is None:
        if stacks_list[0].layers[0].light.wl_nm != stacks_list[-1].layers[0].light.wl_nm:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
        elif set(stacks_list[0].layers[0].light.k_pll) != set(stacks_list[-1].layers[0].light.k_pll):
            xvalues = [np.sqrt(s.layers[0].light.k_pll[0]**2 + s.layers[0].light.k_pll[1]**2) for s in stacks_list]
            xlabel = r'$|k_\parallel|$'
        else:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
            print("t_r_a_plots is guessing you have a single wavelength, else specify xvalues.")

    if add_height!=0: add_name += "_" + zeros_int_str(add_height)
    stack_label = zeros_int_str(stack_label)

    a_list = []
    t_list = []
    r_list = []
    for stack in stacks_list:
        a_list.extend(stack.a_list)
        t_list.extend(stack.t_list)
        r_list.extend(stack.r_list)

    layers_steps = len(stacks_list[0].layers) - 1
    a_tot = []
    t_tot = []
    r_tot = []
    for i in range(len(xvalues)):
        a_tot.append(float(np.asarray(a_list[layers_steps-1+(i*layers_steps)]).flat[0]))
        t_tot.append(float(np.asarray(t_list[layers_steps-1+(i*layers_steps)]).flat[0]))
        r_tot.append(float(np.asarray(r_list[layers_steps-1+(i*layers_steps)] if len(r_list) > layers_steps-1+(i*layers_steps) else r_list[-1]).flat[0]))

    if ult_eta is True or J_sc is True:
        active_abs = []
        active_layer_nu = len(stacks_list[0].layers) - active_layer_nu - 1
        if not 0 < active_layer_nu < len(stacks_list[0].layers)-1:
            raise ValueError("active_layer_nu must refer to a finite layer.")
        for i in range(len(xvalues)):
            active_abs.append(float(np.asarray(a_list[active_layer_nu - 1 +
                              i*layers_steps]).flat[0]))
        out = []

    if ult_eta is True:
        Efficiency = ult_efficiency(active_abs, xvalues,
                                    params_2_print=params_2_print,
                                    stack_label=stack_label, add_name=add_name,
                                    save_txt=save_txt)
        params_2_print += r'$\eta$ = %(Efficiency)6.2f' % {'Efficiency':
                                                           Efficiency*100, }
        params_2_print += ' %'
        out.append(Efficiency)

    if J_sc is True:
        J = J_short_circuit(active_abs, xvalues,
                            params_2_print=params_2_print,
                            stack_label=stack_label, add_name=add_name,
                            save_txt=save_txt)
        params_2_print += r'$J_{sc}$ = %(J)6.2f'% {'J' : J, }
        params_2_print += r' mA/cm$^2$'
        out.append(J)

    fig_layers = []
    if save_txt is True or save_pdf is True or plot_layers is True:
        total_h = sum(stacks_list[0].heights_nm()) # look at first wl result to find h.
        # Plot t,r,a for each layer.
        f_a = layers_plot('Lay_Absorb', a_list, xvalues, xlabel, total_h,
                    params_2_print, stack_label, add_name, save_pdf, save_txt,
                    label_eV, set_y_lim)
        f_t = layers_plot('Lay_Trans',  t_list, xvalues, xlabel, total_h,
                    params_2_print, stack_label, add_name, save_pdf, save_txt,
                    label_eV, set_y_lim)
        f_r = layers_plot('Lay_Reflec', r_list, xvalues, xlabel, total_h,
                    params_2_print, stack_label, add_name, save_pdf, save_txt,
                    label_eV, set_y_lim)
        fig_layers = [f_a, f_t, f_r]

    # Plot total t,r,a on a single plot.
    plot_name = 'Total_Spectra'
    fig_tot = total_tra_plot(plot_name, a_tot, t_tot, r_tot, xvalues, xlabel,
                   params_2_print, stack_label, add_name, label_eV,
                   set_y_lim, save_pdf=save_pdf)

    if weight_spec == True:
        # Plot totals weighted by solar irradiance.
        Irrad_spec_file = os.path.join(paths.data_path, 'ASTMG173.txt')
        i_data = np.loadtxt(Irrad_spec_file)
        i_spec = np.interp(xvalues, i_data[:,0], i_data[:,3])
        bandgap_wl = xvalues[-1]
        weighting = i_spec/i_spec.max()*(xvalues/bandgap_wl)
        a_weighted = a_tot*weighting
        t_weighted = t_tot*weighting
        r_weighted = r_tot*weighting
        plot_name = 'Weighted_Total_Spectra'
        total_tra_plot(plot_name, a_weighted, t_weighted, r_weighted, xvalues,
                       xlabel, params_2_print, stack_label, add_name, label_eV,
                       set_y_lim, save_pdf=save_pdf)

    if extinct == True:
        extinction_plot(t_tot, xvalues, params_2_print, stack_label, add_name, save_pdf=save_pdf)

    if J_sc == True or ult_eta == True:
        return out
    elif plot_layers:
        return [fig_tot] + [f for f in fig_layers if f is not None]
    else:
        return fig_tot


def layers_plot(spectra_name, spec_list, xvalues, xlabel, total_h,
                params_2_print, stack_label, add_name, save_pdf=False, save_txt=False,
                label_eV=False, set_y_lim=True):
    """ Plots one type of spectrum across all layers.

        Is called from t_r_a_plots.
    """

    fig = plt.figure(num=None, figsize=(9, 12))
    nu_layers = int(len(spec_list)/len(xvalues))
    h_array = np.ones(len(xvalues))*total_h
    for i in range(nu_layers):
        layer_spec = []
        for wl in range(len(xvalues)):
            layer_spec = np.append(layer_spec, spec_list[wl*nu_layers + i])
        av_array = list(zip(xvalues, layer_spec, h_array))
        ax1 = fig.add_subplot(nu_layers, 1, i+1)
        ax1.plot(xvalues, layer_spec, linewidth=linesstrength)
        if label_eV == True:
            ax2 = ax1.twiny()
            new_tick_values = np.linspace(10, 0.5, 20)
            new_tick_locations = tick_function(new_tick_values)
            new_tick_labels = ["%.1f" % z for z in new_tick_values]
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(new_tick_labels)
            ax2.set_xlim((xvalues[0], xvalues[-1]))
        if i == 0:
            if xlabel == r'$\lambda$ (nm)' and label_eV == True:
                ax2.set_xlabel('Energy (eV)')
        elif i == nu_layers-1:
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Total')
        if i != 0 and label_eV == True:
            ax2.set_xticklabels( () )
        if i != nu_layers-1:
            ax1.set_xticklabels( () )
        if spectra_name == 'Lay_Absorb':
            if i == 0: ax1.set_ylabel('Top Layer')
            if i == nu_layers - 2: ax1.set_ylabel('Bottom Layer')
            suptitle_w_params = 'Absorptance in each layer' + add_name + \
            '\n' + params_2_print
            plt.suptitle(suptitle_w_params, fontsize = title_font)
            lay_spec_name = 'Lay_Absorb'
            if i == nu_layers-1:
                ax1.set_ylabel('Total')
                lay_spec_name = 'Absorptance'
                if set_y_lim==True:
                    plt.ylim((0, 1))
        elif spectra_name == 'Lay_Trans':
            if i == 0: ax1.set_ylabel('Top Layer')
            if i == nu_layers-2: ax1.set_ylabel('Bottom Layer')
            suptitle_w_params = 'Transmittance in each layer'+add_name+'\n'+params_2_print
            plt.suptitle(suptitle_w_params,fontsize=title_font)
            lay_spec_name = 'Lay_Trans'
            if i == nu_layers-1:
                ax1.set_ylabel('Total')
                lay_spec_name = 'Transmittance'
                if set_y_lim==True:
                    plt.ylim((0, 1))
        elif spectra_name == 'Lay_Reflec':
            if i == 0: ax1.set_ylabel('Top Layer')
            if i == nu_layers-3: ax1.set_ylabel('Bottom Layer')
            if i == nu_layers-2: ax1.set_ylabel('Substrate')
            suptitle_w_params = 'Reflectance in each layer'+add_name+'\n'+params_2_print
            plt.suptitle(suptitle_w_params,fontsize=title_font)
            lay_spec_name = 'Lay_Reflec'
            if i == nu_layers-1:
                ax1.set_ylabel('Total')
                lay_spec_name = 'Reflectance'
                if set_y_lim==True:
                    plt.ylim((0, 1))
        if len(xvalues) > 1 and xvalues[0] != xvalues[-1]:
            ax1.set_xlim((xvalues[0], xvalues[-1]))

        if save_txt == True:
            if i != nu_layers-1:
                np.savetxt('%(s)s_%(i)i_stack%(bon)s%(add)s.txt'% \
                    {'s' : lay_spec_name, 'i' : i, 'bon' : stack_label, \
                    'add' : add_name}, av_array, fmt = '%18.11f')
            else:
                np.savetxt('%(s)s_stack%(bon)s%(add)s.txt'% \
                    {'s' : lay_spec_name, 'bon' : stack_label, \
                    'add' : add_name}, av_array, fmt = '%18.11f')
        if save_pdf == True:
            plt.savefig('%(s)s_stack%(bon)s%(add)s'% \
                {'s': spectra_name, 'bon': stack_label, 'add' : add_name})
    if save_pdf:
        plt.close(fig)
        return None
    return fig


def total_tra_plot(plot_name, a_spec, t_spec, r_spec, xvalues, xlabel,
    params_2_print, stack_label, add_name, label_eV=False, set_y_lim=True, save_pdf=False):
    """ Plots total t, r, a spectra on one plot.

        Is called from t_r_a_plots, t_r_a_plots_subs
    """

    fig = plt.figure(num=None, figsize=(9, 12))
    has_range = (len(xvalues) > 1 and xvalues[0] != xvalues[-1])

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(xvalues, a_spec, linewidth=linesstrength)
    ax1.set_ylabel('Absorptance')
    if has_range:
        ax1.set_xlim((xvalues[0], xvalues[-1]))
    if label_eV == True:
        ax2 = ax1.twiny()
        new_tick_values = np.linspace(10, 0.5, 20)
        new_tick_locations = tick_function(new_tick_values)
        new_tick_labels = ["%.1f" % z for z in new_tick_values]
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(new_tick_labels)
        if has_range:
            ax2.set_xlim((xvalues[0], xvalues[-1]))
        if xlabel == r'$\lambda$ (nm)':
            ax2.set_xlabel('Energy (eV)')
    if set_y_lim==True:
        plt.ylim((0, 1))

    ax1 = fig.add_subplot(3, 1, 2)
    ax1.plot(xvalues, t_spec, linewidth=linesstrength)
    ax1.set_ylabel('Transmittance')
    if has_range:
        ax1.set_xlim((xvalues[0], xvalues[-1]))
    if label_eV == True:
        ax2 = ax1.twiny()
        ax2.set_xticklabels( () )
        ax2.set_xticks(new_tick_locations)
        if has_range:
            ax2.set_xlim((xvalues[0], xvalues[-1]))
    if set_y_lim==True:
        plt.ylim((0, 1))

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(xvalues, r_spec, linewidth=linesstrength)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Reflectance')
    if has_range:
        ax1.set_xlim((xvalues[0], xvalues[-1]))
    if label_eV == True:
        ax2 = ax1.twiny()
        ax2.set_xticklabels( () )
        ax2.set_xticks(new_tick_locations)
        if has_range:
            ax2.set_xlim((xvalues[0], xvalues[-1]))
    if set_y_lim==True:
        plt.ylim((0, 1))

    plt.suptitle(params_2_print, fontsize=title_font)
    # plt.suptitle(plot_name+add_name+'\n'+params_2_print)
    if save_pdf:
        plt.savefig('%(s)s_stack%(bon)s%(add)s'% \
                    {'s': plot_name, 'bon': stack_label, 'add': add_name})
        plt.close(fig)
        return None
    return fig
###############################################################################


#### Plot spectra indicating Wood anomalies in substrate ######################
def t_r_a_plots_subs(stacks_list, wavelengths, period, sub_n,
    params_layer=1, active_layer_nu=1, stack_label=1, ult_eta=False,
    J_sc=False, weight_spec=False, extinct=False, add_name='',
    save_pdf=False, save_txt=False):
    """ Plot t, r, a indicating Wood anomalies in substrate for each layer \
        & total.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            wavelengths  (list): The wavelengths corresponding to stacks_list.

            period  (float): Period of :Stack:s.

            sub_n  (float): Refractive index of the substrate in which Wood \
                anomalies are considered.

        Keyword Args:
            params_layer  (int): The index in stacks_list of the layer for \
                which the geometric parameters are put in the title of \
                the plots.

            active_layer_nu  (int): The index in stacks_list (from bottom) \
                of the layer for which the ult_eta and/or J_sc are calculated.

            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            ult_eta  (bool): If True, calculate the 'ultimate efficiency'.

            J_sc  (bool): If True, calculate the idealised short circuit \
                current.

            weight_spec  (bool): If True, plot t, r, a spectra weighted by \
                the ASTM 1.5 solar spectrum.

            extinct  (bool): If True, calculate the extinction ratio in \
                transmission.

            add_name  (str): Add add_name to title.

            save_pdf (bool): If True, save plot as PDF. False by default.

            save_txt (bool): If True, save spectra to txt. False by default.
    """

    height_list = stacks_list[0].heights_nm()[::-1]
    params_2_print = gen_params_string(stacks_list, params_layer)
    params_2_print += '\n'r'$h_t,...,h_b$ = '
    params_2_print += ''.join('%4d, ' % num for num in height_list)

    stack_label = zeros_int_str(stack_label)

    a_list = []
    t_list = []
    r_list = []
    for stack in stacks_list:
        a_list.extend(stack.a_list)
        t_list.extend(stack.t_list)
        r_list.extend(stack.r_list)

    layers_steps = len(stacks_list[0].layers) - 1
    a_tot      = []
    t_tot      = []
    r_tot      = []
    for i in range(len(wavelengths)):
        a_tot.append(float(a_list[layers_steps-1+(i*layers_steps)]))
        t_tot.append(float(t_list[layers_steps-1+(i*layers_steps)]))
        r_tot.append(float(r_list[layers_steps-1+(i*layers_steps)] if len(r_list) > layers_steps-1+(i*layers_steps) else r_list[-1]))

    out = []
    if ult_eta == True or J_sc == True:
        active_abs = []
        active_layer_nu = len(stacks_list[0].layers) - active_layer_nu - 1
        if not 0 < active_layer_nu < len(stacks_list[0].layers)-1:
            raise ValueError("active_layer_nu must refer to a finite layer.")
        for i in range(len(wavelengths)):
            active_abs.append(float(a_list[active_layer_nu - 1 + \
                i*layers_steps]))

    if ult_eta == True:
        Efficiency = ult_efficiency(active_abs, wavelengths,
                                    params_2_print=params_2_print,
                                    stack_label=stack_label, add_name=add_name,
                                    save_txt=save_txt)
        params_2_print += r'$\eta$ = %(Efficiency)6.2f'% {'Efficiency' : Efficiency*100, }
        params_2_print += ' %'
        out.append(Efficiency)

    if J_sc == True:
        J = J_short_circuit(active_abs, wavelengths,
                            params_2_print=params_2_print,
                            stack_label=stack_label, add_name=add_name,
                            save_txt=save_txt)
        params_2_print += r'$J_{sc}$ = %(J)6.2f'% {'J' : J, }
        params_2_print += r' mA/cm$^2$'
        out.append(J)

    # Plot total t,r,a on a single plot indicating Wood anomalies
    plot_name = 'Total_Spectra_subs'
    fig = total_tra_plot_subs(plot_name, a_tot, t_tot, r_tot, wavelengths, params_2_print,
      stack_label, add_name, period, sub_n, save_pdf=save_pdf)

    if weight_spec == True:
        # Also plot totals weighted by solar irradiance
        Irrad_spec_file = os.path.join(paths.data_path, 'ASTMG173.txt')
        i_data       = np.loadtxt(Irrad_spec_file)
        i_spec       = np.interp(wavelengths, i_data[:,0], i_data[:,3])
        bandgap_wl   = wavelengths[-1]
        weighting = i_spec/i_spec.max()*(wavelengths/bandgap_wl)
        a_weighted = a_tot*weighting
        t_weighted = t_tot*weighting
        r_weighted = r_tot*weighting
        plot_name = 'Weighted_Total_Spectra_subs'
        total_tra_plot_subs(plot_name, a_weighted, t_weighted, r_weighted, wavelengths,
            params_2_print, stack_label, add_name, period, sub_n, save_pdf=save_pdf)

    if extinct == True:
        extinction_plot(t_tot, wavelengths, params_2_print, stack_label, add_name, save_pdf=save_pdf)
    if ult_eta == True or J_sc == True:
        return out
    return fig


def total_tra_plot_subs(plot_name, a_spec, t_spec, r_spec, wavelengths,
    params_2_print, stack_label, add_name, period, sub_n, save_pdf=False):
    """ Plots total t, r, a spectra with lines at first 6 Wood anomalies.

        Is called from t_r_a_plots_subs
    """

    t_WA_01 = sub_n * period
    t_WA_11 = sub_n * period / np.sqrt(2)
    t_WA_20 = sub_n * period / 2
    t_WA_21 = sub_n * period / np.sqrt(5)
    t_WA_22 = sub_n * period / np.sqrt(8)
    t_WA_30 = sub_n * period / 3

    r_WA_01 = period
    r_WA_11 = period / np.sqrt(2)
    r_WA_20 = period / 2
    r_WA_21 = period / np.sqrt(5)
    r_WA_22 = period / np.sqrt(8)

    sub_line01 = [t_WA_01, t_WA_01]
    sub_line11 = [t_WA_11, t_WA_11]
    sub_line20 = [t_WA_20, t_WA_20]
    sub_line21 = [t_WA_21, t_WA_21]
    sub_line22 = [t_WA_22, t_WA_22]
    sub_line30 = [t_WA_30, t_WA_30]
    sup_line01 = [r_WA_01, r_WA_01]
    sup_line11 = [r_WA_11, r_WA_11]
    sup_line20 = [r_WA_20, r_WA_20]
    sup_line21 = [r_WA_21, r_WA_21]
    sup_line22 = [r_WA_22, r_WA_22]
    v_line = [0, 1]

    fig = plt.figure(num=None, figsize=(9, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(wavelengths, a_spec)
    ax1.plot(sub_line01, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line11, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line20, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line21, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line22, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line30, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sup_line01, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line11, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line20, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line21, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line22, v_line, 'r', linewidth=linesstrength)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Absorptance')
    ax1.set_xlim((wavelengths[0], wavelengths[-1]))
    ax2 = ax1.twiny()
    new_tick_values = np.linspace(10, 0.5, 20)
    new_tick_locations = tick_function(new_tick_values)
    new_tick_labels = ["%.1f" % z for z in new_tick_values]
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(new_tick_labels)
    ax2.set_xlim((wavelengths[0], wavelengths[-1]))
    ax2.set_xlabel('Energy (eV)')
    plt.ylim((0, 1))
    ax1 = fig.add_subplot(3, 1, 2)
    ax1.plot(wavelengths, t_spec)
    ax1.plot(sub_line01, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line11, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line20, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line21, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line22, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line30, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sup_line01, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line11, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line20, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line21, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line22, v_line, 'r', linewidth=linesstrength)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transmittance')
    ax1.set_xlim((wavelengths[0], wavelengths[-1]))
    ax2 = ax1.twiny()
    ax2.set_xticklabels( () )
    ax2.set_xticks(new_tick_locations)
    ax2.set_xlim((wavelengths[0], wavelengths[-1]))
    plt.ylim((0, 1))
    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(wavelengths, r_spec)
    ax1.plot(sub_line01, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line11, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line20, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line21, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line22, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sub_line30, v_line, 'k', linewidth=linesstrength)
    ax1.plot(sup_line01, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line11, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line20, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line21, v_line, 'r', linewidth=linesstrength)
    ax1.plot(sup_line22, v_line, 'r', linewidth=linesstrength)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_xlim((wavelengths[0], wavelengths[-1]))
    ax2 = ax1.twiny()
    ax2.set_xticklabels( () )
    ax2.set_xticks(new_tick_locations)
    ax2.set_xlim((wavelengths[0], wavelengths[-1]))
    plt.ylim((0, 1))
    plt.suptitle(params_2_print,fontsize=title_font)
    if save_pdf:
        plt.savefig('%(s)s_stack%(bon)s__%(add)s'% {'s' : plot_name, 'bon' : stack_label,'add' : add_name})
        fig.clf()
        plt.close(fig)
        return None
    return fig
###############################################################################


#### Save J_sc & ult efficiency w/o spectra ###################################
def J_sc_eta_NO_plots(stacks_list, wavelengths, params_layer=1,
    active_layer_nu=1, stack_label=1, add_name=''):
    """ Calculate J_sc & ultimate efficiency but do not save or plot spectra.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            wavelengths  (list): The wavelengths corresponding to stacks_list.

        Keyword Args:
            params_layer  (int): The index in stacks_list of the layer for \
                which the geometric parameters are put in the title of \
                the plots.

            active_layer_nu  (int): The index in stacks_list (from bottom) \
                of the layer for which the ult_eta and/or J_sc are calculated.

            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            add_name  (str): Add add_name to title.
    """

    height_list = stacks_list[0].heights_nm()[::-1]
    params_2_print = gen_params_string(stacks_list, params_layer)
    params_2_print += '\n'r'$h_t,...,h_b$ = '
    params_2_print += ''.join('%4d, ' % num for num in height_list)

    stack_label = zeros_int_str(stack_label)

    a_list = []
    for stack in stacks_list:
        a_list.extend(stack.a_list)

    layers_steps = len(stacks_list[0].layers) - 1
    active_abs = []
    active_layer_nu = len(stacks_list[0].layers) - active_layer_nu - 1
    if not 0 < active_layer_nu < len(stacks_list[0].layers)-1:
        raise ValueError("active_layer_nu must refer to a finite layer.")
    for i in range(len(wavelengths)):
        active_abs.append(float(a_list[active_layer_nu - 1 + i*layers_steps]))

    Efficiency = ult_efficiency(active_abs, wavelengths,
                                params_2_print=params_2_print,
                                stack_label=stack_label, add_name=add_name)

    J = J_short_circuit(active_abs, wavelengths,
                        params_2_print=params_2_print,
                        stack_label=stack_label, add_name=add_name)
    return
###############################################################################


#### Saving spectra to files ##################################################
def t_r_a_write_files(stacks_list, wavelengths, stack_label=1,
    add_name=''):
    """ Save t, r, a for each layer & total in text files.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            wavelengths  (list): The wavelengths corresponding to stacks_list.

        Keyword Args:
            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            add_name  (str): Add add_name to title.
    """
    stack_label = zeros_int_str(stack_label)

    a_list = []
    t_list = []
    r_list = []
    for stack in stacks_list:
        a_list.extend(stack.a_list)
        t_list.extend(stack.t_list)
        r_list.extend(stack.r_list)

    layers_steps = len(stacks_list[0].layers) - 1
    a_tot      = []
    t_tot      = []
    r_tot      = []
    for i in range(len(wavelengths)):
        a_tot.append(float(a_list[layers_steps-1+(i*layers_steps)]))
        t_tot.append(float(t_list[layers_steps-1+(i*layers_steps)]))
        r_tot.append(float(r_list[i]))

    total_h = sum(stacks_list[0].heights_nm()) # look at first wl result to find h.
    layers_print('Lay_Absorb', a_list, wavelengths, total_h, stack_label)
    layers_print('Lay_Trans',  t_list, wavelengths, total_h, stack_label)
    layers_print('Lay_Reflec', r_list, wavelengths, total_h, stack_label)


def layers_print(spectra_name, spec_list, wavelengths, total_h,
    stack_label=1, add_name=''):
    """ Save spectra to text files.

        Is called from t_r_a_write_files.
    """

    nu_layers = int(len(spec_list)/len(wavelengths))
    h_array = np.ones(len(wavelengths))*total_h
    for i in range(nu_layers):
        layer_spec = []
        for wl in range(len(wavelengths)):
            layer_spec = np.append(layer_spec,spec_list[wl*nu_layers + i])
        if spectra_name == 'Lay_Absorb':
            lay_spec_name = 'Lay_Absorb'
            if i == nu_layers-1:
                lay_spec_name = 'Absorptance'
        elif spectra_name == 'Lay_Trans':
            lay_spec_name = 'Lay_Trans'
            if i == nu_layers-1:
                lay_spec_name = 'Transmittance'
        elif spectra_name == 'Lay_Reflec':
            lay_spec_name = 'Lay_Reflec'
            if i == nu_layers-1:
                lay_spec_name = 'Reflectance'
        av_array = list(zip(wavelengths, layer_spec, h_array))

        if i != nu_layers-1:
            np.savetxt('%(s)s_%(i)i_stack%(bon)s%(add)s.txt'% {'s' : lay_spec_name, 'i' : i,
                'bon' : stack_label,'add' : add_name}, av_array, fmt = '%18.11f')
        else:
            np.savetxt('%(s)s_stack%(bon)s%(add)s.txt'% {'s' : lay_spec_name,
                'bon' : stack_label,'add' : add_name}, av_array, fmt = '%18.11f')
###############################################################################


#### Plot spectra on other scales #############################################
def extinction_plot(t_spec, wavelengths, params_2_print, stack_label,
    add_name, save_pdf=False):
    """ Plot extinction ratio in transmission extinct = log_10(1/t). """

    fig = plt.figure(num=None, figsize=(9, 12), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1,1,1)
    extinciton = np.log10(1.0/np.array(t_spec))
    ax1.plot(wavelengths, extinciton, linewidth=linesstrength)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Extinction')
    plot_name = 'Extinction'
    plt.suptitle(plot_name + add_name + '\n' + params_2_print, \
        fontsize = title_font)
    if save_pdf:
        plt.savefig('%(s)s_stack%(bon)s_%(add)s'% \
            {'s' : plot_name, 'bon' : stack_label,'add' : add_name})
        fig.clf()
        plt.close(fig)
        return None
    return fig


def EOT_plot(stacks_list, wavelengths=None, pol='TM', params_layer=1,
    add_name='', savetxt=False, save_pdf=False):
    """ Plot T_{00} as in Martin-Moreno PRL 86 2001.

        Args:
            pol  (str): Take the (0,0) component for TE->TE scattering
                or the (0,0) component for TM->TM scattering.
    """
    if wavelengths is None:
        wavelengths = [s.layers[0].light.wl_nm for s in stacks_list]
    wavelengths = np.asarray(wavelengths)

    height_list = stacks_list[0].heights_nm()[::-1]
    params_2_print = gen_params_string(stacks_list, params_layer)
    params_2_print += r'$h_t,...,h_b$ = '
    params_2_print += ''.join('%4d, ' % num for num in height_list)

    if pol == 'TM':
        num_pw_per_pol = stacks_list[0].layers[-1].structure.num_pw_per_pol
    else:
        num_pw_per_pol = 0

    T_00 = []
    for i in range(len(wavelengths)):
        t00  = stacks_list[i].T_net[num_pw_per_pol,num_pw_per_pol]
        t00c = t00.conjugate()
        T_00.append(np.real(t00*t00c))

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(wavelengths, T_00, linewidth = linesstrength)
    # ax1.set_xlabel('Wavelength (nm)')
    ax1.set_xlabel(r'$\lambda/a$')
    ax1.set_ylabel(r'T$_{00}$')
    # plt.ylim((0, 0.3))

    R_00 = []
    for i in range(len(wavelengths)):
        r00  = stacks_list[i].R_net[num_pw_per_pol,num_pw_per_pol]
        r00c = r00.conjugate()
        R_00.append(np.real(r00*r00c))

    ax1 = fig.add_subplot(2,1,2)
    ax1.plot(wavelengths, R_00, linewidth=linesstrength)
    # ax1.set_xlabel('Wavelength (nm)')
    ax1.set_xlabel(r'$\lambda/a$')
    ax1.set_ylabel(r'R$_{00}$')
    # plt.ylim((0.2, 1.0))
    plot_name = 'EOT'
    plt.suptitle(params_2_print,fontsize=title_font)
    if save_pdf:
        plt.savefig('%(s)s%(add)s'% {'s' : plot_name, 'add' : add_name})
        fig.clf()
        plt.close(fig)

    if savetxt == True:
        np.savetxt('Trans_%(s)s%(add)s.txt'% {'s' : plot_name, 'add' : add_name}, \
        list(zip(wavelengths, T_00)), fmt = '%10.6f')
        np.savetxt('Refl_%(s)s%(add)s.txt'% {'s' : plot_name, 'add' : add_name}, \
        list(zip(wavelengths, R_00)), fmt = '%10.6f')

    if not save_pdf:
        return fig
###############################################################################


#### Calculate short circuit current and ultimate efficiency ##################
def J_short_circuit(active_abs, wavelengths, params_2_print='',
                    stack_label='', add_name='', save_txt=False):
    """ Calculate the short circuit current J_sc under ASTM 1.5 illumination.
        Assuming every absorbed photon produces a pair of charge carriers.
    """

    Irrad_spec_file = os.path.join(paths.data_path, 'ASTMG173.txt')
    i_data = np.loadtxt(Irrad_spec_file)
    i_spec = np.interp(wavelengths, i_data[:, 0], i_data[:, 3])
    expression = i_spec*active_abs*wavelengths
    _trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    integral_tmp = _trapz_fn(expression, x=wavelengths)
    J = (charge_e/(Plancks_h*speed_c)) * integral_tmp * 1e-10  # in mA/cm^2
    nums_2_print = params_2_print.split()
    if save_txt:
        np.savetxt('J_sc_stack%(bon)s%(add)s.txt' % {'bon': stack_label,
                   'add': add_name}, np.array([J]), fmt='%10.6f')
    return J


def ult_efficiency(active_abs, wavelengths, bandgap_wl=None,
                   params_2_print='', stack_label='', add_name='', save_txt=False):
    """ Calculate the photovoltaic ultimate efficiency achieved in the specified active layer.

        For definition see `Sturmberg et al., Optics Express, Vol. 19, Issue S5, pp. A1067-A1081 (2011)\
         <http://dx.doi.org/10.1364/OE.19.0A1067>`_.

        Args:
            bandgap_wl  (float): allows you to set the wavelength equivalent
                to the bandgap. Else it is assumed to be the maximum wavelength
                simulated.
    """
    if bandgap_wl is None:
        bandgap_wl = np.max(wavelengths)
    Irrad_spec_file = os.path.join(paths.data_path, 'ASTMG173.txt')
    i_data       = np.loadtxt(Irrad_spec_file)
    i_spec       = np.interp(wavelengths, i_data[:,0], i_data[:,3])
    expression   = i_spec*active_abs*wavelengths
    _trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    integral_tmp = _trapz_fn(expression, x=wavelengths)
    Efficiency   = integral_tmp/(bandgap_wl*ASTM15_tot_I)
    nums_2_print = params_2_print.split()
    if save_txt:
        np.savetxt('Efficiency_stack%(bon)s%(add)s.txt'% {'bon' : stack_label,'add' : add_name},\
            np.array([Efficiency]), fmt = '%8.6f')
    return Efficiency
###############################################################################


#### Plot dispersion diagrams & field concentrations function of wavelength ###
def omega_plot(stacks_list, wavelengths=None, params_layer=1, stack_label=1, save_pdf=False):
    """ Plots the dispersion diagram of each layer in one plot. \
        k_z has units nm^-1.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            wavelengths  (list): The wavelengths corresponding to stacks_list.

        Keyword Args:
            params_layer  (int): The index in stacks_list of the layer for \
                which the geometric parameters are put in the title of \
                the plots.

            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            save_pdf  (bool): If True, save plots to PDF files. Defaults to False.
    """
    if wavelengths is None:
        wavelengths = [s.layers[0].light.wl_nm for s in stacks_list]
    wavelengths = np.asarray(wavelengths)

    params_2_print = gen_params_string(stacks_list, params_layer)
    stack_label = zeros_int_str(stack_label)
    period = np.array(stacks_list[0].layers[0].structure.period)
    normed_omegas = 1/wavelengths*period

    num_layers = len(stacks_list[0].layers)
    fig1 = plt.figure(num=None, figsize=(9, 12), dpi=80, facecolor='w', edgecolor='k')
    fig2 = plt.figure(num=None, figsize=(9, 12), dpi=80, facecolor='w', edgecolor='k')
    fig3 = plt.figure(num=None, figsize=(9, 12), dpi=80, facecolor='w', edgecolor='k')
    fig4 = plt.figure(num=None, figsize=(9, 12), dpi=80, facecolor='w', edgecolor='k')
    for l in range(num_layers):
        ax1 = fig1.add_subplot(num_layers,1,num_layers-l)
        ax2 = fig2.add_subplot(num_layers,1,num_layers-l)
        ax3 = fig3.add_subplot(num_layers,1,num_layers-l)
        ax4 = fig4.add_subplot(num_layers,1,num_layers-l)
        for i in range(len(wavelengths)):
            k_zs = stacks_list[i].layers[l].k_z
            real_k_zs = []
            imag_k_zs = []
            for k in k_zs:
                if np.real(k) > np.imag(k): #alternatively np.imag(k)< small
                    real_k_zs.append(np.real(k))
                    imag_k_zs.append(np.imag(k))
            wl = np.ones(len(real_k_zs))*wavelengths[i]
            ax1.plot(wl,real_k_zs,'bo', linewidth=linesstrength)
            wl = np.ones(len(imag_k_zs))*wavelengths[i]
            ax2.plot(wl,imag_k_zs,'ro', linewidth=linesstrength)
            om = np.ones(len(real_k_zs))*normed_omegas[i]
            ax3.plot(real_k_zs,om,'bo', linewidth=linesstrength)
            om = np.ones(len(imag_k_zs))*normed_omegas[i]
            ax4.plot(imag_k_zs, om,'ro', linewidth=linesstrength)
        ax1.set_ylabel(r'Real $k_z$ (d)'), ax2.set_ylabel(r'Imaginary $k_z$ (d)')
        ax3.set_ylabel(r'Frequency ($\omega$d/2$\pi$c)'), ax4.set_ylabel(r'Frequency ($\omega$d/2$\pi$c)')
        if l == 0:
            ax1.set_ylabel('Bottom Layer'), ax2.set_ylabel('Bottom Layer')
            ax3.set_ylabel('Bottom Layer'), ax4.set_ylabel('Bottom Layer')
            ax3.set_xlabel(r'Real $k_z$ (d)'), ax4.set_xlabel(r'Imaginary $k_z$ (d)')
            ax1.set_xlabel('Wavelength (nm)'), ax2.set_xlabel('Wavelength (nm)')
        if l == num_layers-1:
            ax1.set_ylabel('Top Layer'), ax2.set_ylabel('Top Layer')
            ax3.set_ylabel('Top Layer'), ax4.set_ylabel('Top Layer')
        ax2.set_xlim((wavelengths[0], wavelengths[-1]))
        ax1.set_xlim((wavelengths[0], wavelengths[-1]))
        # Plot the (dispersive) light line in homogeneous layers.
        if _is_anallo(stacks_list[0].layers[l]):
            ns = [stacks_list[i].layers[l].n() for i in range(len(wavelengths))]
            ax3.plot(np.real(ns)*normed_omegas, normed_omegas,'k', linewidth=linesstrength)
    fig1.suptitle(r'Real $k_z$'+params_2_print+'\n',fontsize=title_font)
    fig2.suptitle(r'Imaginary $k_z$'+params_2_print+'\n',fontsize=title_font)
    fig3.suptitle(r'Real $k_z$'+params_2_print+'\n',fontsize=title_font)
    fig4.suptitle(r'Imaginary $k_z$'+params_2_print+'\n',fontsize=title_font)
    if save_pdf:
        fig1.savefig('Disp_Diagram_Re_stack%(bon)s'% {'bon' : stack_label}, bbox_inches='tight')
        fig2.savefig('Disp_Diagram_Im_stack%(bon)s'% {'bon' : stack_label}, bbox_inches='tight')
        fig3.savefig('Disp_Diagram_w(k)_Re_stack%(bon)s'% {'bon' : stack_label}, bbox_inches='tight')
        fig4.savefig('Disp_Diagram_w(k)_Im_stack%(bon)s'% {'bon' : stack_label}, bbox_inches='tight')
    return fig1, fig2, fig3, fig4


def E_conc_plot(stacks_list, which_layer, which_modes, wavelengths,
                params_layer=1, stack_label=1, save_pdf=False):
    """ Plots the energy concentration (epsilon E_cyl / epsilon E_cell) of given layer.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            which_layer  (int): The index in stacks_list of the layer for \
                which the energy concentration is to be calculated.

            which_modes  (list): Indices of Bloch modes for which to calculate \
                the energy concentration.

            wavelengths  (list): The wavelengths corresponding to stacks_list.

        Keyword Args:
            params_layer  (int): The index in stacks_list of the layer for \
                which the geometric parameters are put in the title of \
                the plots.

            stack_label  (int): Label to differentiate plots of different \
                :Stack:s.

            save_pdf  (bool): If True, save plot to PDF file. Defaults to False.
    """

    params_2_print = gen_params_string(stacks_list, params_layer)
    stack_label = zeros_int_str(stack_label)
    if _is_simmo(stacks_list[0].layers[which_layer]):
        fig1 = plt.figure(num=None, figsize=(9, 4), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig1.add_subplot(1,1,1)
        for i in range(len(wavelengths)):
            for mode in which_modes:
                E_conc_tmp = np.real(stacks_list[i].layers[which_layer].mode_pol[3,mode])
                ax1.plot(wavelengths[i],E_conc_tmp,'bo', linewidth=linesstrength)
        plt.xlim((wavelengths[0], wavelengths[-1]))
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel(r'$\epsilon |E_{cyl}| / \epsilon |E_{cell}|$')
        fig1.suptitle('Energy Concentration = ' + r'$E_{cyl} / E_{cell}$' + '\n' + \
            params_2_print, fontsize=title_font)
        if save_pdf:
            fig1.savefig('Energy_Concentration_stack%(bon)s'% {'bon' : stack_label}, \
                bbox_inches='tight')
        return fig1
    else:
        print("\nERROR: plotting.E_conc_plot; \n" + \
            "Can only calculate energy concentration in NanoStruct layers.")
        print(repr(stacks_list[0].layers[which_layer]))
        return None
###############################################################################


#### Visualise scattering Matrices ############################################
def vis_scat_mats(scat_mat, nu_prop_PWs=0, wl=None, add_name='',
                  max_scale=None, save_pdf=False):
    """ Plot given scattering matrix as greyscale images.

        Args:
            scat_mat  (np.matrix): A scattering matrix, which is \
                organised as \
                | TE -> TE | TM -> TE | \
                | TE -> TM | TM -> TM |

        Keyword Args:
            nu_prop_PWs  (int): Number of propagating PWs.

            wl  (int): Index in case of calling in a loop.

            add_name  (str): Add add_name to title.

            max_scale  (float): Limit maximum amplitude shown.

            save_pdf  (bool): If True, save plot to PDF file. Defaults to False.
    """

    fig = plt.figure(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    if wl != None: add_name = '-wl%(wl)i-%(ti)s' % {'wl' : wl, 'ti' : add_name}

    for i in [1,2]:
        ax1 = fig.add_subplot(1,2,i)
        if i==1: image = abs(np.real(scat_mat))
        if i==2: image = abs(np.imag(scat_mat))
        if max_scale != None:
            mat = ax1.matshow(image, vmax = max_scale, cmap=plt.cm.gray)
        else:
            mat = ax1.matshow(image, cmap=plt.cm.gray)
        scat_mat_dim_x = np.shape(scat_mat)[0]
        scat_mat_dim_y = np.shape(scat_mat)[1]
        half_dim_x = scat_mat_dim_x/2-0.5
        half_dim_y = scat_mat_dim_y/2-0.5
        ax1.plot([-0.5, scat_mat_dim_x-0.5],[half_dim_y,half_dim_y],'w', linewidth=2)
        ax1.plot([half_dim_x,half_dim_x],[-0.5, scat_mat_dim_y-0.5],'w', linewidth=2)
        ax1.axis([-0.5, scat_mat_dim_y-0.5, scat_mat_dim_x-0.5,  -0.5])
        ax1.set_xticks([half_dim_y/2, scat_mat_dim_y-half_dim_y/2],['TE', 'TM'])
        ax1.set_yticks([half_dim_x/2, scat_mat_dim_x-half_dim_x/2],['TE', 'TM'])
        ax1.plot([-0.5, scat_mat_dim_y-0.5],[nu_prop_PWs-0.5,nu_prop_PWs-0.5],'r', linewidth=1)
        ax1.plot([nu_prop_PWs-0.5,nu_prop_PWs-0.5],[-0.5, scat_mat_dim_x-0.5],'r', linewidth=1)
        ax1.plot([-0.5, scat_mat_dim_y-0.5],[half_dim_x+nu_prop_PWs,half_dim_x+nu_prop_PWs],'r', linewidth=1)
        ax1.plot([half_dim_y+nu_prop_PWs,half_dim_y+nu_prop_PWs],[-0.5, scat_mat_dim_x-0.5],'r', linewidth=1)
        cbar = fig.colorbar(mat,extend='neither')
        if i==1: cbar.ax.set_ylabel('|Real(matrix)|', fontsize=14)
        if i==2: cbar.ax.set_ylabel('|Imag(matrix)|', fontsize=14)
        ax1.set_xticklabels('')
        ax1.set_yticklabels('')
        ax1.set_xlabel('TE        |        TM \n Incoming Orders', fontsize=14)
        ax1.set_ylabel('Outgoing Orders \nTM       |         TE', fontsize=14)

    plt.suptitle('Scattering Matrices' + add_name)
    if save_pdf:
        plt.savefig('Scat_mat' + add_name)
        fig.clf()
    return fig


def vis_matrix(scat_mat, add_name='', max_scale=None, only_real=False, save_pdf=False):
    """ Plot given matrix as a greyscale image.

        Args:
            scat_mat  (np.matrix): A matrix.

        Keyword Args:
            add_name  (str): Add add_name to title.

            max_scale  (float): Limit maximum amplitude shown.

            only_real  (bool): Only plot the real part of matrix.

            save_pdf  (bool): If True, save plot to PDF file. Defaults to False.
    """

    fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

    if only_real == True: real_im = [1]
    else: real_im = [1,2]

    for i in real_im:
        ax1 = fig.add_subplot(1,len(real_im),i)
        if i==1: image = abs(np.real(scat_mat))
        if i==2: image = abs(np.imag(scat_mat))
        if max_scale != None:
            mat = ax1.matshow(image, vmax = max_scale, cmap=plt.cm.gray)
        else:
            mat = ax1.matshow(image, cmap=plt.cm.gray)
        cbar = fig.colorbar(mat,extend='neither')
        if i==1: cbar.ax.set_ylabel('|Re(matrix)|', fontsize=14)
        if i==2: cbar.ax.set_ylabel('|Im(matrix)|', fontsize=14)
        ax1.set_xticklabels('')
        ax1.set_yticklabels('')

    plt.suptitle(add_name)
    if save_pdf:
        plt.savefig('Matrix' + add_name)
        fig.clf()
    return fig
###############################################################################


#### Plot PW amplitudes function k-vector #####################################
def t_func_k_plot_1D(stacks_list, lay_interest=0, pol='TE', save_pdf=False):
    """ PW amplitudes in transmission as a function of their in-plane k-vector.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            lay_interest  (int): The index in stacks_list of the layer in \
                which amplitudes are calculated.

            pol  (str): Include transmission in Which polarisation.

            save_pdf  (bool): If True, save plot to PDF file. Defaults to False.
    """
    fig = plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1,1,1)

    # vec_coef sorted from top of stack, everything else is sorted from bottom
    vec_index = len(stacks_list[0].layers) - lay_interest - 1

    store_alphas = []
    store_k_trans = []
    for stack in stacks_list:
        n_PW_p_pols = stack.layers[0].structure.num_pw_per_pol
        sort_order = stack.layers[lay_interest].sort_order

        select_trans  = abs(stack.vec_coef_down[vec_index][0:n_PW_p_pols])
        store_alphas  = np.append(store_alphas,stack.layers[lay_interest].alphas[sort_order])
        store_k_trans = np.append(store_k_trans,select_trans)

    sort_indices = np.argsort(store_alphas)
    plot_alphas = store_alphas[sort_indices]
    plot_k_trans = store_k_trans[sort_indices]

    ax1.plot(plot_alphas,plot_k_trans, linewidth=1.5)

    min_k_label = np.max(np.abs(plot_alphas))
    k0 = abs(stack.layers[lay_interest].k())
    min_k_l_k0  = np.max(np.abs(plot_alphas))/k0

    n_H = max_n(stacks_list)
    new_tick_values = [-min_k_label, -n_H*k0, -k0, 0, k0, n_H*k0, min_k_label]
    new_tick_labels = [r"$-%ik_0$"%min_k_l_k0,r'$-n_Hk_0$',r'$-k_0$',r'0',
        r'$k_0$',r'$n_Hk_0$',r"$%ik_0$"%min_k_l_k0]
    ax1.set_xticks(new_tick_values)
    ax1.set_xticklabels(new_tick_labels)
    ax1.set_xlim(-min_k_label,min_k_label)

    ax1.set_xlabel(r'$k_\parallel$')
    ax1.set_ylabel(r'$|E|$')
    if save_pdf:
        plt.savefig('k_vector_excitation-lay_%s' % lay_interest, bbox_inches='tight')
        fig.clf()
    return fig
###############################################################################


#### Plot amplitudes of modes #################################################
def BM_amplitudes(stacks_list, xvalues=None, chosen_BMs=None,
                  lay_interest=1, up_and_down=True, add_height=None, add_name='',
                  save_pdf=False, save_npz=False, save_txt=False):
    """ Plot the amplitudes of Bloch modes in selected layer.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            xvalues  (list): The values stacks_list is to be plotted as a \
                function of.

            chosen_BMs  (list): Bloch Modes to include, identified by their \
                indices in the scattering matrices (order most propagating \
                to most evanescent) eg. [0,2,4].

            lay_interest  (int): The index in stacks_list of the layer in \
                which amplitudes are calculated.

            up_and_down  (bool): Average the amplitudes of up & downward \
                propagating modes. Else include only downward in all layers\
                except for the superstrate where include only upward.

            add_height  (float): Print the heights of :Stack: layer in title.

            add_name  (str): Add add_name to title.

            save_pdf  (bool): If True, save plot as PDF. False by default.

            save_npz  (bool): If True, saves lists of BM amplitudes to file.

            save_txt  (bool): If True, saves lists of BM amps to txt file.
    """

    fig = plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1, 1, 1)
    # vec_coef sorted from top of stack, everything else is sorted from bottom
    vec_index = len(stacks_list[-1].layers) - lay_interest - 1

    xlabel = 'xvalues'
    if xvalues is None:
        if stacks_list[0].layers[0].light.wl_nm != stacks_list[-1].layers[0].light.wl_nm:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
        elif set(stacks_list[0].layers[0].light.k_pll) != set(stacks_list[-1].layers[0].light.k_pll):
            xvalues = [np.sqrt(s.layers[0].light.k_pll[0]**2 + s.layers[0].light.k_pll[1]**2) for s in stacks_list]
            xlabel = r'$|k_\parallel|$'
        else:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
            print("BM_amplitudes is guessing you have a single wavelength, else specify xvalues.")

    if chosen_BMs is None:
        chosen_BMs = range(stacks_list[-1].layers[lay_interest].num_BMs)
    try:
        save_trans = []
        for BM in chosen_BMs:
            store_trans = []
            for stack in stacks_list:
                if not _is_simmo(stack.layers[lay_interest]):
                    raise ValueError

                trans = np.abs(stack.vec_coef_down[vec_index][BM])
                if up_and_down is True:
                    trans += np.abs(stack.vec_coef_up[vec_index][BM])
                    store_trans = np.append(store_trans, trans / 2)
                else:
                    store_trans = np.append(store_trans, trans)
            ax1.plot(xvalues, store_trans, label="BM %i" % BM)
            if save_npz is True or save_txt is True:
                save_trans.append(store_trans)

        if add_height is not None:
            add_name += '_' + zeros_int_str(add_height)
        add_name = str(lay_interest) + add_name

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.set_ylabel('BM Amplitude')
        ax1.set_xlabel(xlabel)
        plt.suptitle(add_name, fontsize=title_font)

        if save_pdf:
            plt.savefig('BM_amplitudes-lay_%s' % add_name,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.clf()

        if save_npz is True:
            np.savez('BM_amplitudes-lay_%s' % add_name, save_trans=save_trans)

        if save_txt is True:
            np.savetxt('BM_amplitudes-lay_%s.txt' % add_name,
                       save_trans, fmt='%18.11f')

        if not save_pdf:
            return fig

    except ValueError:
        print("BM_amplitudes only works in NanoStruct layers."
              "\nPlease select lay_interest !=%i.\n" % lay_interest)
        return None


def PW_amplitudes(stacks_list, xvalues=None, chosen_PWs=None,
                  lay_interest=0, up_and_down=True, add_height=None, add_name='',
                  save_pdf=False, save_npz=False, save_txt=False):
    """ Plot the amplitudes of plane wave orders in selected layer.

        Assumes dealing with 1D grating and only have 1D diffraction orders.
        Takes the average of up & downward propagating modes.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            xvalues  (list): The values stacks_list is to be plotted as a \
                function of.

            chosen_PWs  (list): PW diffraction orders to include. \
                eg. [-1,0,2]. If 'None' are given all are plotted.

            lay_interest  (int): The index in stacks_list of the layer in \
                which amplitudes are calculated.

            up_and_down  (bool): Average the amplitudes of up & downward \
                propagating modes. Else include only downward in all layers\
                except for the superstrate where include only upward.

            add_height  (float): Print the heights of :Stack: layer in title.

            add_name  (str): Add add_name to title.

            save_pdf  (bool): If True, save plot to PDF. False by default.

            save_npz  (bool): If True, saves lists of PW amplitudes to file.

            save_txt  (bool): If True, saves lists of PW amps to txt file.
    """

    fig = plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1, 1, 1)
    # vec_coef sorted from top of stack, everything else is sorted from bottom
    vec_index = len(stacks_list[-1].layers) - lay_interest - 1
    if chosen_PWs is None:
        max_ords = stacks_list[0].layers[-1].max_order_PWs
        chosen_PWs = np.arange(-max_ords, max_ords + 1)

    xlabel = 'xvalues'
    if xvalues is None:
        if stacks_list[0].layers[0].light.wl_nm != stacks_list[-1].layers[0].light.wl_nm:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
        elif set(stacks_list[0].layers[0].light.k_pll) != set(stacks_list[-1].layers[0].light.k_pll):
            xvalues = [np.sqrt(s.layers[0].light.k_pll[0]**2 + s.layers[0].light.k_pll[1]**2) for s in stacks_list]
            xlabel = r'$|k_\parallel|$'
        else:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
            print("PW_amplitudes is guessing you have a single wavelength, else specify xvalues.")

    try:
        save_trans = []
        for pxs in chosen_PWs:
            store_trans = []
            for stack in stacks_list:
                if not _is_anallo(stack.layers[lay_interest]):
                    raise ValueError

                k0 = stack.layers[0].k()
                n_PW_p_pols = stack.layers[0].structure.num_pw_per_pol
                alpha0, beta0 = stack.layers[0].k_pll_norm()
                alphas = alpha0 + pxs * 2 * np.pi
                on_axis_kzs = sqrt(k0**2 - alphas**2 - beta0**2)
                full_k_space = stack.layers[0].k_z
                one_pol_k_space = full_k_space[0:n_PW_p_pols]

                ix = np.isin(one_pol_k_space.ravel(), on_axis_kzs).reshape(one_pol_k_space.shape)
                axis_indices = np.ravel(np.array(np.where(ix))).astype(int)
                # Substrate - only ever take downward propagating.
                if vec_index == len(stacks_list[-1].layers) - 1:
                    trans = np.abs(stack.vec_coef_down[vec_index][axis_indices]).reshape(-1,)
                    trans += np.abs(stack.vec_coef_down[vec_index][n_PW_p_pols+axis_indices]).reshape(-1,)
                    store_trans = np.append(store_trans, trans)
                elif vec_index == 0:
                    trans = np.abs(stack.vec_coef_up[vec_index][axis_indices]).reshape(-1,)
                    trans += np.abs(stack.vec_coef_up[vec_index][n_PW_p_pols+axis_indices]).reshape(-1,)
                    if up_and_down is True:
                        trans += np.abs(stack.vec_coef_down[vec_index][axis_indices]).reshape(-1,)
                        trans += np.abs(stack.vec_coef_down[vec_index][n_PW_p_pols+axis_indices]).reshape(-1,)
                        store_trans = np.append(store_trans, trans / 2)
                    else:
                        store_trans = np.append(store_trans, trans)
                else:
                    trans = np.abs(stack.vec_coef_down[vec_index][axis_indices]).reshape(-1,)
                    trans += np.abs(stack.vec_coef_down[vec_index][n_PW_p_pols+axis_indices]).reshape(-1,)
                    if up_and_down is True:
                        trans += np.abs(stack.vec_coef_up[vec_index][axis_indices]).reshape(-1,)
                        trans += np.abs(stack.vec_coef_up[vec_index][n_PW_p_pols+axis_indices]).reshape(-1,)
                        store_trans = np.append(store_trans, trans / 2)
                    else:
                        store_trans = np.append(store_trans, trans)
            ax1.plot(xvalues, store_trans, label="m = %i" % pxs)
            if save_npz is True or save_txt is True:
                save_trans.append(store_trans)

        if add_height is not None:
            add_name += '_' + zeros_int_str(add_height)
        add_name = str(lay_interest) + add_name

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.set_ylabel(r'$|E|$')
        ax1.set_xlabel(xlabel)
        plt.suptitle(add_name, fontsize=title_font)

        if save_pdf:
            plt.savefig('PW_amplitudes-lay_%s' % add_name,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.clf()

        if save_npz is True:
            np.savez('PW_amplitudes-lay_%s' % add_name, save_trans=save_trans)

        if save_txt is True:
            np.savetxt('PW_amplitudes-lay_%s.txt' % add_name,
                       save_trans, fmt='%18.11f')

        if not save_pdf:
            return fig

    except ValueError:
        print("PW_amplitudes only works in ThinFilm layers."
              "\nPlease select lay_interest !=%i.\n" % lay_interest)
        return None


def evanescent_merit(stacks_list, xvalues=None, chosen_PWs=None,
                     lay_interest=0, add_height=None, add_name='',
                     save_pdf=False, save_txt=False):
    """ Plot a figure of merit for the 'evanescent-ness' of excited fields.

        Assumes dealing with 1D grating and only have 1D diffraction orders.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            xvalues  (list): The values stacks_list is to be plotted as a \
                function of.

            chosen_PWs  (list): PW diffraction orders to include. \
                eg. [-1,0,2].

            lay_interest  (int): The index in stacks_list of the layer in \
                which amplitudes are calculated.

            add_height  (float): Print the heights of :Stack: layer in title.

            add_name  (str): Add add_name to title.

            save_pdf  (bool): If True, save plot to PDF. False by default.

            save_txt  (bool): If True, saves average value of \
                mean PW order to file.
    """
    vec_index = len(stacks_list[-1].layers) - lay_interest - 1
    if chosen_PWs is None:
        max_ords = stacks_list[0].layers[-1].max_order_PWs
        chosen_PWs = np.arange(-max_ords, max_ords + 1)
    n_H = max_n(stacks_list)

    xlabel = 'xvalues'
    if xvalues is None:
        if stacks_list[0].layers[0].light.wl_nm != stacks_list[-1].layers[0].light.wl_nm:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
        elif set(stacks_list[0].layers[0].light.k_pll) != set(stacks_list[-1].layers[0].light.k_pll):
            xvalues = [np.sqrt(s.layers[0].light.k_pll[0]**2 + s.layers[0].light.k_pll[1]**2) for s in stacks_list]
            xlabel = r'$|k_\parallel|$'
        else:
            xvalues = [s.layers[0].light.wl_nm for s in stacks_list]
            xlabel = r'$\lambda$ (nm)'
            print("evanescent_merit is guessing you have a single wavelength, else specify xvalues.")

    store_m_p = []
    store_m_ne = []
    store_m_fe = []
    store_x_p = []
    store_x_ne = []
    store_x_fe = []
    store_tot_amps = []
    store_mean_ev = []
    s = 0
    try:
        for stack in stacks_list:
            if not _is_anallo(stack.layers[lay_interest]):
                raise ValueError

            merit_prop = 0.0
            merit_near_ev = 0.0
            merit_far_ev = 0.0
            sum_p_amps = 0.0
            sum_amps = 0.0
            for pxs in chosen_PWs:
                k0 = stack.layers[-1].k()
                k = stack.layers[lay_interest].k()
                n_PW_p_pols = stack.layers[lay_interest].structure.num_pw_per_pol
                alpha0, beta0 = stack.layers[lay_interest].k_pll_norm()
                alphas = alpha0 + pxs * 2 * np.pi
                this_k_pll2 = alphas**2 + beta0**2
                on_axis_kzs = sqrt(k**2 - alphas**2 - beta0**2)
                full_k_space = stack.layers[lay_interest].k_z
                one_pol_k_space = full_k_space[0:n_PW_p_pols]

                ix = np.isin(one_pol_k_space.ravel(), on_axis_kzs).reshape(one_pol_k_space.shape)
                axis_indices = np.ravel(np.array(np.where(ix))).astype(int)

                trans = np.abs(stack.vec_coef_down[vec_index][axis_indices])
                trans += np.abs(stack.vec_coef_down[vec_index][n_PW_p_pols + axis_indices])

                if np.abs(this_k_pll2) < np.abs(k**2):
                    merit_prop += trans
                if np.abs(k**2) < np.abs(this_k_pll2) < np.abs((n_H * k0)**2):
                    merit_near_ev += trans
                if np.abs(this_k_pll2) > np.abs((n_H * k0)**2):
                    merit_far_ev += trans
                sum_p_amps += np.abs(pxs) * trans
                sum_amps += np.abs(stack.vec_coef_down[vec_index][axis_indices])**2 + np.abs(stack.vec_coef_down[vec_index][n_PW_p_pols + axis_indices])**2

            if merit_prop != 0.0:
                store_m_p = np.append(store_m_p, merit_prop)
                store_x_p = np.append(store_x_p, xvalues[s])
            if merit_near_ev != 0.0:
                store_m_ne = np.append(store_m_ne, merit_near_ev)
                store_x_ne = np.append(store_x_ne, xvalues[s])
            if merit_far_ev != 0.0:
                store_m_fe = np.append(store_m_fe, merit_far_ev)
                store_x_fe = np.append(store_x_fe, xvalues[s])
            s += 1
            store_tot_amps = np.append(store_tot_amps, sum_amps)
            store_mean_ev = np.append(store_mean_ev, sum_p_amps / sum_amps)

        if add_height is not None:
            add_name += '_' + zeros_int_str(add_height)
        add_name = str(lay_interest) + add_name

        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(2, 1, 1)
        if len(store_m_p) != 0:
            ax1.plot(store_x_p, store_m_p, 'b', label="prop")
        if len(store_m_ne) != 0:
            ax1.plot(store_x_ne, store_m_ne, 'g', label="near ev")
        if len(store_m_fe) != 0:
            ax1.plot(store_x_fe, store_m_fe, 'r', label="far ev")
        ax1.plot(xvalues, store_mean_ev, 'k', label=r'$\Sigma|p| |a_p| / \Sigma|a_p|$')
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(xvalues, store_tot_amps, 'k', label="prop")

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, ncol=4, loc='upper center',
                         bbox_to_anchor=(0.5, 1.6))
        ax1.set_ylabel('Ev FoM')
        ax1.set_xticklabels(())
        ax2.set_ylabel(r'$\Sigma|a_p|^2$')
        ax2.set_xlabel(xlabel)
        plt.suptitle(add_name, fontsize=title_font)

        if save_pdf:
            plt.savefig('evanescent_merit-lay_%s' % add_name,
                        bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            fig.clf()

        if save_txt is True:
            av_diff = [np.mean(store_m_p)]
            np.savetxt('prop_diff_order-lay_%s.txt' % add_name,
                       av_diff, fmt='%18.11f')
            av_diff = [np.mean(store_m_ne)]
            np.savetxt('near_ev_diff_order-lay_%s.txt' % add_name,
                       av_diff, fmt='%18.11f')
            av_diff = [np.mean(store_m_fe)]
            np.savetxt('far_ev_diff_order-lay_%s.txt' % add_name,
                       av_diff, fmt='%18.11f')
            av_diff = [np.mean(store_mean_ev)]
            np.savetxt('average_diff_order-lay_%s.txt' % add_name,
                       av_diff, fmt='%18.11f')

        if not save_pdf:
            return fig

    except ValueError:
        print("evanescent_merit only works in ThinFilm layers."
              "\nPlease select lay_interest !=%i.\n" % lay_interest)
        return None
###############################################################################


#### Field plotting routines ##################################################
def fields_in_plane(stacks_list, lay_interest=1, z_values=[0.1, 3.6],
                    nu_calc_pts=51, save_pdf=False, save_pos=False):
    """
    Plot fields in the x-y plane at chosen values of z.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            lay_interest  (int): the index of the layer considered within \
                the stack.

            z_values  (float): distance in nm from bottom surface of layer \
                at which to calculate fields. If layer is semi-inf substrate \
                then z_value is distance from top of this layer (i.e. bottom \
                interface of stack).

            nu_calc_pts  (int): fields are calculated over a mesh of \
                nu_calc_pts * nu_calc_pts points.

            save_pdf  (bool): If True, save plots to PDF. Defaults to False.
            save_pos  (bool): If True, save Gmsh .pos files. Defaults to False.
    """
    import field_eval

    dir_name = "in_plane_fields"
    if (save_pdf or save_pos) and not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # always make odd
    if nu_calc_pts % 2 == 0:
        nu_calc_pts += 1

    figs = []
    stack_num = 0
    for pstack in stacks_list:
        z_values = np.array(z_values)/float(pstack.layers[lay_interest].structure.period)
        num_lays = len(pstack.layers)
        # If selected z location is within a NanoStruct layer
        # plot fields in Bloch Mode Basis using fortran routine.
        if _is_simmo(pstack.layers[lay_interest]):
            meat = pstack.layers[lay_interest]
            try:
                if not meat.structure.periodicity == '2D_array':
                    raise ValueError
                gmsh_file_pos = meat.structure.mesh_file[0:-5]
                eps = meat.n_effs**2 if meat.n_effs is not None else np.ones(meat.structure.nb_typ_el)
                h_normed = float(meat.structure.height_nm)/float(meat.structure.period)
                for z_value in z_values:
                    # Fortran routine naturally measure z top down
                    z_value = h_normed - z_value
                    wl_normed = pstack.layers[lay_interest].wl_norm()

                    nnodes = 6
                    if meat.E_H_field == 1:
                        EH_name = "E_"
                    else:
                        EH_name = "H_"
                    extra_name = EH_name + "Lay" + zeros_int_str(lay_interest)

                    # vec_coef sorted from top of stack, everything else is sorted from bottom
                    vec_index = num_lays - lay_interest - 1
                    vec_coef = np.concatenate((pstack.vec_coef_down[vec_index],
                                              pstack.vec_coef_up[vec_index]))

                    if save_pos:
                        field_eval.gmsh_plot_field(meat.num_BMs,
                                                 meat.n_msh_el, meat.n_msh_pts,
                                                 nnodes, meat.structure.nb_typ_el,
                                                 meat.table_nod, meat.type_el, eps,
                                                 meat.x_arr, meat.k_z, meat.sol1,
                                                 vec_coef, h_normed, z_value,
                                                 gmsh_file_pos,
                                                 meat.structure.plot_real,
                                                 meat.structure.plot_imag,
                                                 meat.structure.plot_abs,
                                                 extra_name)

            except ValueError:
                print("fields_in_plane cannot plot fields in 1D-arrays."\
                "\nPlease select a different lay_interest.\n")


        # If selected z location is within a ThinFilm layer
        # plot fields in Plane Wave Basis using python routine.
        else:
            wl = np.around(pstack.layers[-1].light.wl_nm,decimals=2)
            pw = pstack.layers[-1].max_order_PWs
            period = pstack.layers[-1].structure.period
            heights_list = []
            name_lay = ''

            for i in range(num_lays):
                if i == 0 or i == num_lays-1:pass
                else: heights_list.append(pstack.layers[i].structure.height_nm)
                try:
                    pstack.layers[i].structure.diameter1
                    diameter = pstack.layers[i].structure.diameter1
                except:pass

            if lay_interest == 0: name_lay = "0_Substrate"
            elif lay_interest == num_lays-1: name_lay = "%(i)s_Superstrate" %{'i':num_lays-1}
            else: name_lay = "Thin_Film_%(i)s" %{'i':lay_interest}

            x_range = np.linspace(0.0,1.0,nu_calc_pts)
            y_range = np.linspace(0.0,1.0,nu_calc_pts)
            z_range = np.array(z_values)

            s = pstack.layers[lay_interest].sort_order
            alpha_unsrt = np.array(pstack.layers[lay_interest].alphas)
            beta_unsrt = np.array(pstack.layers[lay_interest].betas)
            alpha = alpha_unsrt[s]
            if pstack.layers[lay_interest].structure.world_1d == True:
                beta = beta_unsrt
            else:
                beta = beta_unsrt[s]
            gamma = np.array(pstack.layers[lay_interest].calc_kz())
            n = pstack.layers[lay_interest].n()
            PWordtot = pstack.layers[lay_interest].structure.num_pw_per_pol
            prop = 2*(gamma.imag == 0).sum()
            evan = 2*PWordtot - prop

            vec_coef_down = np.array(pstack.vec_coef_down[num_lays-1-lay_interest]).flatten()
            vec_coef_down_TE = vec_coef_down[0:PWordtot]
            vec_coef_down_TM = vec_coef_down[PWordtot::]
            if lay_interest == 0:
                vec_coef_up = np.zeros((2*PWordtot), dtype = 'complex')
            else:
                vec_coef_up = np.array(pstack.vec_coef_up[num_lays-1-lay_interest]).flatten()

            vec_coef_up_TE = vec_coef_up[0:PWordtot]
            vec_coef_up_TM = vec_coef_up[PWordtot::]

            norm = np.sqrt(alpha**2+beta**2)
            k = np.sqrt(alpha**2+beta**2+gamma**2)
            chi_TE = np.sqrt((n*gamma)/k)
            chi_TM = np.sqrt((n*k)/gamma)
            E_TE_x = beta/norm
            E_TE_y = -1*alpha/norm
            E_TE_z = np.array(np.zeros(np.size(E_TE_x)))
            E_TM_x = alpha/norm
            E_TM_y = beta/norm
            E_TM_z = -1*norm/gamma
            k = np.around(k,decimals=4)
            n = np.around(n,decimals=4)

            eta_TE_x_down = (vec_coef_down_TE*E_TE_x)/chi_TE
            eta_TE_y_down = (vec_coef_down_TE*E_TE_y)/chi_TE
            eta_TE_z_down = (vec_coef_down_TE*E_TE_z)/chi_TE
            eta_TM_x_down = (vec_coef_down_TM*E_TM_x)/chi_TM
            eta_TM_y_down = (vec_coef_down_TM*E_TM_y)/chi_TM
            eta_TM_z_down = (vec_coef_down_TM*E_TM_z)/chi_TM

            eta_TE_x_up = (vec_coef_up_TE*E_TE_x)/chi_TE
            eta_TE_y_up = (vec_coef_up_TE*E_TE_y)/chi_TE
            eta_TE_z_up = (vec_coef_up_TE*E_TE_z)/chi_TE
            eta_TM_x_up = (vec_coef_up_TM*E_TM_x)/chi_TM
            eta_TM_y_up = (vec_coef_up_TM*E_TM_y)/chi_TM
            eta_TM_z_up = (vec_coef_up_TM*E_TM_z)/chi_TM

            x_axis = np.zeros((nu_calc_pts,nu_calc_pts), dtype = 'float')
            y_axis = np.zeros((nu_calc_pts,nu_calc_pts), dtype = 'float')

            if lay_interest == 0: z_range = -1*z_range
            else: z_range = np.abs(z_range)

            x1 = x_range
            y1 = y_range
            z1 = z_range
            (y_axis,x_axis) = np.meshgrid(y1,x1)

            E_TE_x_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')
            E_TE_y_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')
            E_TE_z_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')
            E_TM_x_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')
            E_TM_y_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')
            E_TM_z_array = np.zeros((np.size(x1),np.size(y1),np.size(z1)), dtype = 'complex')

            for z in range(np.size(z1)):
                for y in range(np.size(y1)):
                    for x in range(np.size(x1)):
                        if pstack.layers[lay_interest].structure.height_nm == 'semi_inf':
                            expo_down = np.exp(1j*(alpha*x1[x]+beta*y1[y]-gamma*z1[z]))
                        elif z1[z] <= float(pstack.layers[lay_interest].structure.height_nm)/period:
                            expo_down = np.exp(1j*(alpha*x1[x]+beta*y1[y]-gamma*(z1[z]-float(pstack.layers[lay_interest].structure.height_nm)/period)))
                        else:
                            raise ValueError("fields_in_plane: z_value exceeds thickness of layer, reduce it. ")
                        expo_up = np.exp(1j*(alpha*x1[x]+beta*y1[y]+gamma*z1[z]))

                        E_TE_x = np.sum(eta_TE_x_down*expo_down + eta_TE_x_up*expo_up)
                        E_TE_y = np.sum(eta_TE_y_down*expo_down + eta_TE_y_up*expo_up)
                        E_TE_z = np.sum(eta_TE_z_down*expo_down + eta_TE_z_up*expo_up)
                        E_TM_x = np.sum(eta_TM_x_down*expo_down + eta_TM_x_up*expo_up)
                        E_TM_y = np.sum(eta_TM_y_down*expo_down + eta_TM_y_up*expo_up)
                        E_TM_z = np.sum(eta_TM_z_down*expo_down + eta_TM_z_up*expo_up)
                        E_TE_x_array[x,y,z] = E_TE_x
                        E_TE_y_array[x,y,z] = E_TE_y
                        E_TE_z_array[x,y,z] = E_TE_z
                        E_TM_x_array[x,y,z] = E_TM_x
                        E_TM_y_array[x,y,z] = E_TM_y
                        E_TM_z_array[x,y,z] = E_TM_z
            E_x_array = E_TE_x_array + E_TM_x_array
            E_y_array = E_TE_y_array + E_TM_y_array
            E_z_array = E_TE_z_array + E_TM_z_array
            E_tot_array = np.sqrt(E_x_array*np.conj(E_x_array) + E_y_array*np.conj(E_y_array) + E_z_array*np.conj(E_z_array))

            E_super = [E_x_array, E_y_array, E_z_array, E_tot_array]
            E_sup_labels = [' E_x', ' E_y', ' E_z', ' |E|',]

            # Save figures
            for re_im in ['real','imag']:
                for z_of_xy in range(np.size(z1)):
                    fig1 = plt.figure(num=None, figsize=(12,21), dpi=80, facecolor='w', edgecolor='k')
                    for i in range(len(E_super)):
                        ax1 = fig1.add_subplot(4,1,i+1)
                        if re_im == 'real':
                            E_slice = np.real(E_super[i][:,:,z_of_xy])
                        elif re_im == 'imag':
                            E_slice = np.imag(E_super[i][:,:,z_of_xy])
                        x_min = x1[0]
                        x_max = x1[-1]
                        y_min = y1[0]
                        y_max = y1[-1]
                        cmap = plt.get_cmap('jet')
                        CS = plt.contourf(x_axis,y_axis,E_slice,35,cmap=cmap)
                        plt.axis([x_min,x_max,y_min,y_max])
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel(re_im + E_sup_labels[i])
                        ax1.set_xlabel('x (d)')
                        ax1.set_ylabel('y (d)')
                        ax1.axis('scaled')
                        ax1.set_ylim((y_min,y_max))

                    plt.suptitle(r'%(name)s ' + '\n' + r'%(p)s_E_xy_slice, z = %(z_pos)s, heights = %(h)s ' + '\n' + \
                        r'$\lambda$ = %(wl)f nm, period = %(d)f, PW = %(pw)i,' % \
                        {'name' : name_lay, 'h':heights_list, 'p' : re_im, 'z_pos' : z1[z_of_xy],'wl' : wl, \
                        'd' : period, 'pw' : pw} + '\n'
                        + '# prop. ords = %(prop)s, # evan. ords = %(evan)s, n = %(n)s,k = %(k)s'\
                        % {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                    if save_pdf:
                        plt.savefig('%(dir_name)s/stack_%(stack_num)s_%(p)s_E_%(name)s_slice=%(z_pos)s_wl=%(wl)s.pdf'% \
                            {'dir_name' : dir_name, 'wl' : wl, 'p' : re_im, 'z_pos' : z1[z_of_xy], \
                            'name' : name_lay,'stack_num':stack_num})
                        fig1.clf()
                        plt.close(fig1)
                    else:
                        figs.append(fig1)

        stack_num += 1
    return figs

            # ## Old fortran plane wave plotting
            # extra_name = EH_name + "Lay" + zeros_int_str(TF_lay)
            # select_h = 0.0
            # lat_vec = [[1.0, 0.0], [0.0, 1.0]]
            # bun = pstack.layers[TF_lay] # superstrate # substrate
            # n_eff_0 = bun.n()
            # neq_PW = bun.structure.num_pw_per_pol
            # bloch_vec = bun.k_pll_norm()
            # ordre_ls = bun.max_order_PWs
            # index_pw = bun.sort_order
            # index_pw_inv = np.zeros(shape=(np.shape(index_pw)),dtype='int')
            # for s in range(len(index_pw)):
            #     s2 = index_pw[s]
            #     index_pw_inv[s2] = s
            # index_pw_inv += 1 # convert to fortran indices (starting from 1)

            # # vec_coef sorted from top of stack, everything else is sorted from bottom
            # vec_index = len(pstack.layers) - TF_lay - 1
            # vec_coef_down = pstack.vec_coef_down[vec_index]
            # if TF_lay == 0: vec_coef_up = np.zeros(shape=(np.shape(vec_coef_down)),dtype='complex128')
            # else: vec_coef_up = pstack.vec_coef_up[vec_index]



            # # # Semi-inf case
            # # vec_coef_up = meat.R12[:,0] # TE polarisation
            # # vec_coef_up = meat.R12[:,neq_PW] # TM polarisation

            # # vec_coef_down = np.zeros(shape=(np.shape(vec_coef_up)),dtype='complex128')
            # # vec_coef_down[neq_PW] = 1.0


class GridInterpolatorWrapper:
    """Wrapper around scipy.interpolate.RegularGridInterpolator so it can be called as f(u, v)."""
    def __init__(self, rgi):
        self.rgi = rgi

    def __call__(self, u, v):
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)
        b_shape = np.broadcast_shapes(u_arr.shape, v_arr.shape)
        u_b = np.broadcast_to(u_arr, b_shape)
        v_b = np.broadcast_to(v_arr, b_shape)
        pts = np.stack([u_b, v_b], axis=-1)
        res = self.rgi(pts)
        if b_shape == ():
            return res.item()
        return res


class FieldBundle:
    """Container for 2D electromagnetic field (E and H) interpolators.

    Can be unpacked as a 14-tuple:
    (ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE, ReHx, ImHx, ReHy, ImHy, ReHz, ImHz, AbsH)
    or accessed via named attributes.
    """
    def __init__(
        self,
        ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE,
        ReHx, ImHx, ReHy, ImHy, ReHz, ImHz, AbsH,
        k0=None,
        interface_z_nm=None,
        plane=None,
        slice_coord=None,
        u_grid=None,
        v_grid=None,
    ):
        self.ReEx = ReEx
        self.ImEx = ImEx
        self.ReEy = ReEy
        self.ImEy = ImEy
        self.ReEz = ReEz
        self.ImEz = ImEz
        self.AbsE = AbsE
        self.ReHx = ReHx
        self.ImHx = ImHx
        self.ReHy = ReHy
        self.ImHy = ImHy
        self.ReHz = ReHz
        self.ImHz = ImHz
        self.AbsH = AbsH
        self.k0 = k0
        self.interface_z_nm = interface_z_nm
        self.plane = plane
        self.slice_coord = slice_coord
        self.u_grid = u_grid
        self.v_grid = v_grid

    def Ex(self, x, y):
        return self.ReEx(x, y) + 1j * self.ImEx(x, y)

    def Ey(self, x, y):
        return self.ReEy(x, y) + 1j * self.ImEy(x, y)

    def Ez(self, x, y):
        return self.ReEz(x, y) + 1j * self.ImEz(x, y)

    def Hx(self, x, y):
        return self.ReHx(x, y) + 1j * self.ImHx(x, y)

    def Hy(self, x, y):
        return self.ReHy(x, y) + 1j * self.ImHy(x, y)

    def Hz(self, x, y):
        return self.ReHz(x, y) + 1j * self.ImHz(x, y)

    def Sz(self, x, y):
        """Evaluate z-component of time-averaged Poynting vector: Sz = 0.5 * Re(Ex * Hy^* - Ey * Hx^*)."""
        ex = self.Ex(x, y)
        ey = self.Ey(x, y)
        hx = self.Hx(x, y)
        hy = self.Hy(x, y)
        return 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))

    def Poynting(self, u, v):
        """Evaluate 3D time-averaged Poynting vector (Sx, Sy, Sz) = 0.5 * Re(E x H*)."""
        ex = self.Ex(u, v)
        ey = self.Ey(u, v)
        ez = self.Ez(u, v)
        hx = self.Hx(u, v)
        hy = self.Hy(u, v)
        hz = self.Hz(u, v)
        sx = 0.5 * np.real(ey * np.conj(hz) - ez * np.conj(hy))
        sy = 0.5 * np.real(ez * np.conj(hx) - ex * np.conj(hz))
        sz = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
        return (sx, sy, sz)

    def chirality(self, u, v):
        """Evaluate optical chirality density C = -0.5 * k0 * Im(E* . H)."""
        if self.k0 is None:
            raise ValueError("k0 is required to compute optical chirality density.")
        ex = self.Ex(u, v)
        ey = self.Ey(u, v)
        ez = self.Ez(u, v)
        hx = self.Hx(u, v)
        hy = self.Hy(u, v)
        hz = self.Hz(u, v)
        edotH = np.conj(ex) * hx + np.conj(ey) * hy + np.conj(ez) * hz
        res = -0.5 * float(self.k0) * np.imag(edotH)
        if isinstance(res, np.ma.MaskedArray):
            return res.filled(np.nan)
        return res

    def chirality_enhancement(self, u, v):
        """Evaluate local optical chirality enhancement normalized to CPL: C_hat = -Im(E* . H)."""
        ex = self.Ex(u, v)
        ey = self.Ey(u, v)
        ez = self.Ez(u, v)
        hx = self.Hx(u, v)
        hy = self.Hy(u, v)
        hz = self.Hz(u, v)
        edotH = np.conj(ex) * hx + np.conj(ey) * hy + np.conj(ez) * hz
        res = -np.imag(edotH)
        if isinstance(res, np.ma.MaskedArray):
            return res.filled(np.nan)
        return res

    def to_xarray(
        self,
        u_1d,
        v_1d,
        dims=("y", "x"),
        period=1.0,
        fixed_coord=None,
        include_H=True,
        include_chirality=True,
    ):
        import sys
        if "dask" not in sys.modules:
            try:
                import yaml
            except ImportError:
                sys.modules["dask"] = None
                sys.modules["dask.base"] = None
        import xarray as xr

        u_arr = np.asarray(u_1d, dtype=float)
        v_arr = np.asarray(v_1d, dtype=float)
        U, V = np.meshgrid(u_arr, v_arr)

        ex = self.Ex(U, V)
        ey = self.Ey(U, V)
        ez = self.Ez(U, V)
        abs_e = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)

        data_vars = {
            "Ex": (dims, ex),
            "Ey": (dims, ey),
            "Ez": (dims, ez),
            "AbsE": (dims, abs_e),
        }

        if include_H:
            hx = self.Hx(U, V)
            hy = self.Hy(U, V)
            hz = self.Hz(U, V)
            abs_h = np.sqrt(np.abs(hx)**2 + np.abs(hy)**2 + np.abs(hz)**2)
            sx, sy, sz = self.Poynting(U, V)
            data_vars.update({
                "Hx": (dims, hx),
                "Hy": (dims, hy),
                "Hz": (dims, hz),
                "AbsH": (dims, abs_h),
                "Sx": (dims, sx),
                "Sy": (dims, sy),
                "Sz": (dims, sz),
            })

            if include_chirality:
                c_enh = self.chirality_enhancement(U, V)
                data_vars["chirality_enhancement"] = (dims, c_enh)
                if self.k0 is not None:
                    c_dens = self.chirality(U, V)
                    data_vars["chirality"] = (dims, c_dens)

        coords = {
            dims[0]: v_arr,
            dims[1]: u_arr,
        }
        attrs = {
            "period": period,
        }
        fc = fixed_coord if fixed_coord is not None else self.slice_coord
        if fc is not None:
            attrs["fixed_coord"] = fc
        if self.interface_z_nm is not None:
            attrs["interface_z_nm"] = np.asarray(self.interface_z_nm).tolist()

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    def __iter__(self):
        return iter((
            self.ReEx, self.ImEx, self.ReEy, self.ImEy, self.ReEz, self.ImEz, self.AbsE,
            self.ReHx, self.ImHx, self.ReHy, self.ImHy, self.ReHz, self.ImHz, self.AbsH
        ))

    def __getitem__(self, idx):
        return tuple(self)[idx]

    def __len__(self):
        return 14


# field interpolators for Nanostruct layers
def fields_interpolator_in_plane(pstack, lay_interest=1, z_value=0.1, return_H=False, return_h=False):
    """
    Returns linear interpolators in the x-y plane at chosen value of z.
    
    If return_H is False (default):
        Returns 7-tuple: (ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE).
    If return_H is True (or return_h is True):
        Returns FieldBundle containing 14 interpolators for E and H.

    Requires matplotlib v1.3.0 or later.

        Args:
            pstack: Stack object (not a list!!!) containing data to plot.

        Keyword Args:
            lay_interest  (int): the index of the layer considered within \
                the stack.

            z_value  (float): distance in nm from bottom surface of layer \
                at which to calculate fields. If layer is semi-inf substrate \
                then z_value is distance from top of this layer (i.e. bottom \
                interface of stack).
                
            return_H (bool): If True, compute both E and H fields, returning \
                a FieldBundle instance (which can also be unpacked as a 14-tuple).
            return_h (bool): Alias for return_H.
    """
    return_H = bool(return_H or return_h)
    import field_eval

    num_lays = len(pstack.layers)
    # If selected z location is within a NanoStruct layer
    # plot fields in Bloch Mode Basis using fortran routine.
    if _is_simmo(pstack.layers[lay_interest]):
        meat = pstack.layers[lay_interest]

        try:
            if not meat.structure.periodicity == '2D_array':
                raise ValueError
            eps = meat.n_effs**2 if meat.n_effs is not None else np.ones(meat.structure.nb_typ_el)
            h_normed = float(meat.structure.height_nm)/float(meat.structure.period)

            # fortran routine naturally measure z top down
            z_value = z_value/float(meat.structure.period)
            z_value = h_normed - z_value
            wl_normed = pstack.layers[lay_interest].wl_norm()
            k0 = 2.0 * np.pi / float(wl_normed)

            nnodes = 6

            # vec_coef sorted from top of stack, everything else is sorted from bottom
            vec_index = num_lays - lay_interest - 1
            vec_coef = np.concatenate((pstack.vec_coef_down[vec_index], pstack.vec_coef_up[vec_index]))

            # piling up of all the bloch modes to get all the fields
            if return_H:
                m_E, m_H = field_eval.field_value_plane(
                    meat.num_BMs, meat.n_msh_el, meat.n_msh_pts, nnodes,
                    meat.structure.nb_typ_el, meat.table_nod, meat.type_el,
                    eps, meat.x_arr, meat.k_z, meat.sol1,
                    vec_coef, h_normed, z_value, k0=k0, wl_normed=wl_normed, return_H=True
                )
            else:
                m_E = field_eval.field_value_plane(
                    meat.num_BMs, meat.n_msh_el, meat.n_msh_pts, nnodes,
                    meat.structure.nb_typ_el, meat.table_nod, meat.type_el,
                    eps, meat.x_arr, meat.k_z, meat.sol1,
                    vec_coef, h_normed, z_value, k0=k0, wl_normed=wl_normed, return_H=False
                )
                m_H = None

            # unrolling data for the interpolators
            table_nod = meat.table_nod.T
            x_arr = meat.x_arr.T

            has_6_nodes = bool((table_nod[:, 3] > 0).any()) if table_nod.shape[1] >= 4 else False

            if has_6_nodes:
                # dense triangulation with multiple points (6-node quadratic elements)
                v_x6p = np.zeros(6*meat.n_msh_el)
                v_y6p = np.zeros(6*meat.n_msh_el)
                v_Ex6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                v_Ey6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                v_Ez6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                if return_H:
                    v_Hx6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                    v_Hy6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                    v_Hz6p = np.zeros(6*meat.n_msh_el, dtype=np.complex128)
                v_triang6p = []

                i = 0
                for i_el in np.arange(meat.n_msh_el):
                    idx = np.arange(6*i_el, 6*(i_el+1))
                    triangles = [[idx[0], idx[3], idx[5]],
                                 [idx[1], idx[4], idx[3]],
                                 [idx[2], idx[5], idx[4]],
                                 [idx[3], idx[4], idx[5]]]
                    v_triang6p.extend(triangles)

                    for i_node in np.arange(6):
                        i_ex = table_nod[i_el, i_node]-1
                        v_x6p[i] = x_arr[i_ex, 0]
                        v_y6p[i] = x_arr[i_ex, 1]
                        v_Ex6p[i] = m_E[i_el, i_node, 0]
                        v_Ey6p[i] = m_E[i_el, i_node, 1]
                        v_Ez6p[i] = m_E[i_el, i_node, 2]
                        if return_H:
                            v_Hx6p[i] = m_H[i_el, i_node, 0]
                            v_Hy6p[i] = m_H[i_el, i_node, 1]
                            v_Hz6p[i] = m_H[i_el, i_node, 2]
                        i += 1

                v_E6p = np.sqrt(np.abs(v_Ex6p)**2 +
                                np.abs(v_Ey6p)**2 +
                                np.abs(v_Ez6p)**2)

                # dense triangulation with unique points
                v_triang1p = []
                for i_el in np.arange(meat.n_msh_el):
                    triangles = [[table_nod[i_el,0]-1,table_nod[i_el,3]-1,table_nod[i_el,5]-1],
                                 [table_nod[i_el,1]-1,table_nod[i_el,4]-1,table_nod[i_el,3]-1],
                                 [table_nod[i_el,2]-1,table_nod[i_el,5]-1,table_nod[i_el,4]-1],
                                 [table_nod[i_el,3]-1,table_nod[i_el,4]-1,table_nod[i_el,5]-1]]
                    v_triang1p.extend(triangles)

                # triangulations
                triang6p = matplotlib.tri.Triangulation(v_x6p,v_y6p,v_triang6p)
                triang1p = matplotlib.tri.Triangulation(x_arr[:,0],x_arr[:,1],v_triang1p)

                # building interpolators: triang1p for the finder, triang6p for the values
                finder = matplotlib.tri.TrapezoidMapTriFinder(triang1p)
                ReEx = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ex6p.real,trifinder=finder)
                ImEx = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ex6p.imag,trifinder=finder)
                ReEy = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ey6p.real,trifinder=finder)
                ImEy = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ey6p.imag,trifinder=finder)
                ReEz = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ez6p.real,trifinder=finder)
                ImEz = matplotlib.tri.LinearTriInterpolator(triang6p,v_Ez6p.imag,trifinder=finder)
                AbsE = matplotlib.tri.LinearTriInterpolator(triang6p,v_E6p,trifinder=finder)

                if return_H:
                    v_H6p = np.sqrt(np.abs(v_Hx6p)**2 + np.abs(v_Hy6p)**2 + np.abs(v_Hz6p)**2)
                    ReHx = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hx6p.real, trifinder=finder)
                    ImHx = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hx6p.imag, trifinder=finder)
                    ReHy = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hy6p.real, trifinder=finder)
                    ImHy = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hy6p.imag, trifinder=finder)
                    ReHz = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hz6p.real, trifinder=finder)
                    ImHz = matplotlib.tri.LinearTriInterpolator(triang6p, v_Hz6p.imag, trifinder=finder)
                    AbsH = matplotlib.tri.LinearTriInterpolator(triang6p, v_H6p, trifinder=finder)
                    return FieldBundle(
                        ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE,
                        ReHx, ImHx, ReHy, ImHy, ReHz, ImHz, AbsH,
                        k0=k0
                    )

                return ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE
            else:
                # 3-node linear triangles
                v_Ex_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                v_Ey_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                v_Ez_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                if return_H:
                    v_Hx_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                    v_Hy_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                    v_Hz_nod = np.zeros(meat.n_msh_pts, dtype=np.complex128)
                counts = np.zeros(meat.n_msh_pts)
                v_triang = []
                for i_el in range(meat.n_msh_el):
                    tri = [table_nod[i_el, 0] - 1, table_nod[i_el, 1] - 1, table_nod[i_el, 2] - 1]
                    v_triang.append(tri)
                    for inod in range(3):
                        node_idx = table_nod[i_el, inod] - 1
                        v_Ex_nod[node_idx] += m_E[i_el, inod, 0]
                        v_Ey_nod[node_idx] += m_E[i_el, inod, 1]
                        v_Ez_nod[node_idx] += m_E[i_el, inod, 2]
                        if return_H:
                            v_Hx_nod[node_idx] += m_H[i_el, inod, 0]
                            v_Hy_nod[node_idx] += m_H[i_el, inod, 1]
                            v_Hz_nod[node_idx] += m_H[i_el, inod, 2]
                        counts[node_idx] += 1
                c = np.maximum(counts, 1)
                v_Ex_nod /= c
                v_Ey_nod /= c
                v_Ez_nod /= c
                v_E_nod = np.sqrt(np.abs(v_Ex_nod)**2 + np.abs(v_Ey_nod)**2 + np.abs(v_Ez_nod)**2)

                triang = matplotlib.tri.Triangulation(x_arr[:, 0], x_arr[:, 1], v_triang)
                ReEx = matplotlib.tri.LinearTriInterpolator(triang, v_Ex_nod.real)
                ImEx = matplotlib.tri.LinearTriInterpolator(triang, v_Ex_nod.imag)
                ReEy = matplotlib.tri.LinearTriInterpolator(triang, v_Ey_nod.real)
                ImEy = matplotlib.tri.LinearTriInterpolator(triang, v_Ey_nod.imag)
                ReEz = matplotlib.tri.LinearTriInterpolator(triang, v_Ez_nod.real)
                ImEz = matplotlib.tri.LinearTriInterpolator(triang, v_Ez_nod.imag)
                AbsE = matplotlib.tri.LinearTriInterpolator(triang, v_E_nod)

                if return_H:
                    v_Hx_nod /= c
                    v_Hy_nod /= c
                    v_Hz_nod /= c
                    v_H_nod = np.sqrt(np.abs(v_Hx_nod)**2 + np.abs(v_Hy_nod)**2 + np.abs(v_Hz_nod)**2)
                    ReHx = matplotlib.tri.LinearTriInterpolator(triang, v_Hx_nod.real)
                    ImHx = matplotlib.tri.LinearTriInterpolator(triang, v_Hx_nod.imag)
                    ReHy = matplotlib.tri.LinearTriInterpolator(triang, v_Hy_nod.real)
                    ImHy = matplotlib.tri.LinearTriInterpolator(triang, v_Hy_nod.imag)
                    ReHz = matplotlib.tri.LinearTriInterpolator(triang, v_Hz_nod.real)
                    ImHz = matplotlib.tri.LinearTriInterpolator(triang, v_Hz_nod.imag)
                    AbsH = matplotlib.tri.LinearTriInterpolator(triang, v_H_nod)
                    return FieldBundle(
                        ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE,
                        ReHx, ImHx, ReHy, ImHy, ReHz, ImHz, AbsH,
                        k0=k0
                    )

                return ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE

        except ValueError as e:
            print(e)
            print("fields_in_plane cannot plot fields in 1D-arrays."\
                  "\nPlease select a different lay_interest.\n")


    # If selected z location is within a ThinFilm layer
    # plot fields in Plane Wave Basis using python routine.
    else:
        raise Exception("Not a NanoStruct Layer")


def fields_interpolator_cross_section(
    pstack,
    plane="xz",
    slice_coord=0.5,
    lay_interest=None,
    n_horiz=151,
    n_vert_per_layer=61,
    semi_inf_height=0.5,
    return_H=True,
    return_h=False,
):
    """Returns 2D cross-section interpolators (FieldBundle) along the xz or yz plane.

    Inside any EMUstack layer, the electromagnetic field is separable:
    E(x, y, z) = sum_m [ P^+_m(z) e_{t,m}(x, y) + P^-_m(z) z_hat (-i k_{z,m} e_{z,m}(x, y)) ]
    allowing fast, vectorized cross-section evaluation via 1D line interpolation
    combined with matrix multiplication across z.

    Args:
        pstack: Evaluated Stack object.

    Keyword Args:
        plane (str): 'xz' (default) or 'yz'.
        slice_coord (float): Fixed coordinate value in [0, 1] (y for 'xz', x for 'yz').
        lay_interest (int or None): Specific layer index, or None to evaluate and
            stitch all layers in the stack.
        n_horiz (int): Number of horizontal grid points along the cut.
        n_vert_per_layer (int): Number of vertical grid points per layer.
        semi_inf_height (float): Normalized vertical extent for semi-infinite substrate/superstrate.
        return_H (bool): If True (default), returns FieldBundle (14 interpolators).
        return_h (bool): Alias for return_H.

    Returns:
        FieldBundle (if return_H=True) or 7-tuple (if return_H=False).
    """
    return_H = bool(return_H or return_h)
    from scipy.interpolate import RegularGridInterpolator

    num_lays = len(pstack.layers)
    if lay_interest is not None:
        layers_to_eval = [int(lay_interest)]
    else:
        layers_to_eval = list(range(num_lays))

    u_line = np.linspace(0.0, 1.0, n_horiz)
    if plane.lower() == "xz":
        x_line = u_line
        y_line = np.full_like(u_line, slice_coord)
    elif plane.lower() == "yz":
        x_line = np.full_like(u_line, slice_coord)
        y_line = u_line
    else:
        raise ValueError(f"Unknown plane: {plane}. Must be 'xz' or 'yz'.")

    # Determine heights and interface positions in nm
    period = float(pstack.layers[0].structure.period)
    wl_normed = float(pstack.layers[0].wl_norm())
    k0 = 2.0 * np.pi / wl_normed

    # Calculate global z offsets for all layers in stack
    layer_heights_nm = []
    for l_idx in range(num_lays):
        lay_obj = pstack.layers[l_idx]
        if lay_obj.structure.height_nm == "semi_inf":
            layer_heights_nm.append(float(semi_inf_height) * period)
        else:
            layer_heights_nm.append(float(lay_obj.structure.height_nm))

    # Interface z coordinates in nm (stack ordered 0=bottom to num_lays-1=top)
    interface_z_nm = []
    current_z = 0.0
    for h in layer_heights_nm[:-1]:
        current_z += h
        interface_z_nm.append(current_z)

    z_slices = []
    Ex_slices = []
    Ey_slices = []
    Ez_slices = []
    Hx_slices = []
    Hy_slices = []
    Hz_slices = []

    for l_idx in layers_to_eval:
        layer = pstack.layers[l_idx]
        h_nm = layer_heights_nm[l_idx]
        h_normed = h_nm / period
        vec_index = num_lays - l_idx - 1

        if lay_interest is not None:
            z_start = 0.0
            z_end = h_nm
        else:
            z_start = sum(layer_heights_nm[:l_idx])
            z_end = z_start + h_nm

        # Avoid exact duplicate coordinate on layer interfaces for RegularGridInterpolator
        if lay_interest is None and l_idx < num_lays - 1:
            z_local_nm = np.linspace(0.0, h_nm - 1e-5, n_vert_per_layer)
        else:
            z_local_nm = np.linspace(0.0, h_nm, n_vert_per_layer)
        z_global_nm = z_start + z_local_nm
        z_slices.append(z_global_nm)

        if _is_simmo(layer):
            if layer.structure.periodicity == "2D_array":
                num_modes = min(layer.num_BMs, layer.sol1.shape[2], len(layer.k_z))
                k_z_arr = np.asarray(layer.k_z).flatten()[:num_modes]
                c_down = np.asarray(pstack.vec_coef_down[vec_index]).flatten()[:num_modes]
                c_up = np.asarray(pstack.vec_coef_up[vec_index]).flatten()[:num_modes]

                table_nod = layer.table_nod.T
                x_arr = layer.x_arr.T
                n_msh_el = layer.n_msh_el
                has_6_nodes = bool((table_nod[:, 3] > 0).any()) if table_nod.shape[1] >= 4 else False

                M_line = np.zeros((6, n_horiz, num_modes), dtype=complex)
                qx = np.clip(x_line, 1e-6, 1.0 - 1e-6)
                qy = np.clip(y_line, 1e-6, 1.0 - 1e-6)

                if has_6_nodes:
                    v_x6p = np.zeros(6 * n_msh_el)
                    v_y6p = np.zeros(6 * n_msh_el)
                    v_triang6p = []
                    v_triang1p = []
                    for i_el in range(n_msh_el):
                        idx = np.arange(6 * i_el, 6 * (i_el + 1))
                        v_triang6p.extend([[idx[0], idx[3], idx[5]], [idx[1], idx[4], idx[3]], [idx[2], idx[5], idx[4]], [idx[3], idx[4], idx[5]]])
                        for inod in range(6):
                            iex = table_nod[i_el, inod] - 1
                            v_x6p[6 * i_el + inod] = x_arr[iex, 0]
                            v_y6p[6 * i_el + inod] = x_arr[iex, 1]
                        v_triang1p.extend([[table_nod[i_el, 0]-1, table_nod[i_el, 3]-1, table_nod[i_el, 5]-1],
                                           [table_nod[i_el, 1]-1, table_nod[i_el, 4]-1, table_nod[i_el, 3]-1],
                                           [table_nod[i_el, 2]-1, table_nod[i_el, 5]-1, table_nod[i_el, 4]-1],
                                           [table_nod[i_el, 3]-1, table_nod[i_el, 4]-1, table_nod[i_el, 5]-1]])

                    triang6p = matplotlib.tri.Triangulation(v_x6p, v_y6p, v_triang6p)
                    triang1p = matplotlib.tri.Triangulation(x_arr[:, 0], x_arr[:, 1], v_triang1p)
                    finder = matplotlib.tri.TrapezoidMapTriFinder(triang1p)
                    t_indices = finder(qx, qy)

                    bad = np.where(t_indices == -1)[0]
                    if len(bad) > 0:
                        tri_centers_x = np.mean(x_arr[np.array(v_triang1p), 0], axis=1)
                        tri_centers_y = np.mean(x_arr[np.array(v_triang1p), 1], axis=1)
                        for b in bad:
                            dists = (tri_centers_x - qx[b])**2 + (tri_centers_y - qy[b])**2
                            t_indices[b] = np.argmin(dists)

                    for k in range(n_horiz):
                        t_idx = t_indices[k]
                        tri = triang6p.triangles[t_idx]
                        x0_pt, y0_pt = v_x6p[tri[0]], v_y6p[tri[0]]
                        x1_pt, y1_pt = v_x6p[tri[1]], v_y6p[tri[1]]
                        x2_pt, y2_pt = v_x6p[tri[2]], v_y6p[tri[2]]
                        det = (y1_pt - y2_pt)*(x0_pt - x2_pt) + (x2_pt - x1_pt)*(y0_pt - y2_pt)
                        if abs(det) < 1e-14:
                            w0, w1, w2 = 1.0/3, 1.0/3, 1.0/3
                        else:
                            w0 = ((y1_pt - y2_pt)*(qx[k] - x2_pt) + (x2_pt - x1_pt)*(qy[k] - y2_pt)) / det
                            w1 = ((y2_pt - y0_pt)*(qx[k] - x2_pt) + (x0_pt - x2_pt)*(qy[k] - y2_pt)) / det
                            w2 = 1.0 - w0 - w1
                        iel0, inod0 = tri[0] // 6, tri[0] % 6
                        iel1, inod1 = tri[1] // 6, tri[1] % 6
                        iel2, inod2 = tri[2] // 6, tri[2] % 6
                        M_line[:, k, :] = (
                            w0 * layer.sol1[:6, inod0, :num_modes, iel0] +
                            w1 * layer.sol1[:6, inod1, :num_modes, iel1] +
                            w2 * layer.sol1[:6, inod2, :num_modes, iel2]
                        )
                else:
                    v_triang = []
                    v_sol1_nod = np.zeros((6, layer.n_msh_pts, num_modes), dtype=complex)
                    counts = np.zeros(layer.n_msh_pts)
                    for i_el in range(n_msh_el):
                        tri = [table_nod[i_el, 0] - 1, table_nod[i_el, 1] - 1, table_nod[i_el, 2] - 1]
                        v_triang.append(tri)
                        for inod in range(3):
                            node_idx = table_nod[i_el, inod] - 1
                            v_sol1_nod[:, node_idx, :] += layer.sol1[:6, inod, :num_modes, i_el]
                            counts[node_idx] += 1
                    c = np.maximum(counts, 1)[:, None]
                    v_sol1_nod /= c[None, :, :]

                    triang = matplotlib.tri.Triangulation(x_arr[:, 0], x_arr[:, 1], v_triang)
                    finder = matplotlib.tri.TrapezoidMapTriFinder(triang)
                    t_indices = finder(qx, qy)
                    bad = np.where(t_indices == -1)[0]
                    if len(bad) > 0:
                        tri_centers_x = np.mean(x_arr[np.array(v_triang), 0], axis=1)
                        tri_centers_y = np.mean(x_arr[np.array(v_triang), 1], axis=1)
                        for b in bad:
                            dists = (tri_centers_x - qx[b])**2 + (tri_centers_y - qy[b])**2
                            t_indices[b] = np.argmin(dists)

                    for k in range(n_horiz):
                        t_idx = t_indices[k]
                        tri = triang.triangles[t_idx]
                        x0_pt, y0_pt = x_arr[tri[0], 0], x_arr[tri[0], 1]
                        x1_pt, y1_pt = x_arr[tri[1], 0], x_arr[tri[1], 1]
                        x2_pt, y2_pt = x_arr[tri[2], 0], x_arr[tri[2], 1]
                        det = (y1_pt - y2_pt)*(x0_pt - x2_pt) + (x2_pt - x1_pt)*(y0_pt - y2_pt)
                        if abs(det) < 1e-14:
                            w0, w1, w2 = 1.0/3, 1.0/3, 1.0/3
                        else:
                            w0 = ((y1_pt - y2_pt)*(qx[k] - x2_pt) + (x2_pt - x1_pt)*(qy[k] - y2_pt)) / det
                            w1 = ((y2_pt - y0_pt)*(qx[k] - x2_pt) + (x0_pt - x2_pt)*(qy[k] - y2_pt)) / det
                            w2 = 1.0 - w0 - w1
                        M_line[:, k, :] = (
                            w0 * v_sol1_nod[:, tri[0], :] +
                            w1 * v_sol1_nod[:, tri[1], :] +
                            w2 * v_sol1_nod[:, tri[2], :]
                        )

                z_top = (h_nm - z_local_nm) / period
                z_top = z_top[None, :]
                kz_col = k_z_arr[:, None]
                P_plus = c_down[:, None] * np.exp(1j * kz_col * z_top) + c_up[:, None] * np.exp(1j * kz_col * (h_normed - z_top))
                P_minus = c_down[:, None] * np.exp(1j * kz_col * z_top) - c_up[:, None] * np.exp(1j * kz_col * (h_normed - z_top))

                Ex_lay = M_line[0] @ P_plus
                Ey_lay = M_line[1] @ P_plus
                Ez_lay = M_line[2] @ ((-1j * kz_col) * P_minus)
                Hx_lay = M_line[3] @ (((kz_col / k0)) * P_minus)
                Hy_lay = M_line[4] @ (((kz_col / k0)) * P_minus)
                Hz_lay = M_line[5] @ ((1.0 / (1j * k0)) * P_plus)
            else:
                # 1D_array
                struct = layer.structure
                num_modes = min(layer.num_BMs, layer.sol1.shape[2], len(layer.k_z))
                k_z_arr = np.asarray(layer.k_z).flatten()[:num_modes]
                c_down = np.asarray(pstack.vec_coef_down[vec_index]).flatten()[:num_modes]
                c_up = np.asarray(pstack.vec_coef_up[vec_index]).flatten()[:num_modes]

                from scipy.interpolate import interp1d
                x_mesh = struct.x_arr[:struct.n_msh_pts]
                M_line = np.zeros((6, n_horiz, num_modes), dtype=complex)
                for BM in range(num_modes):
                    for comp in range(min(6, layer.sol1.shape[0])):
                        val_mesh = np.zeros(struct.n_msh_pts, dtype=complex)
                        val_mesh[0] = layer.sol1[comp, 0, BM, 0]
                        for x_el in range(struct.n_msh_el - 1):
                            val_mesh[2 * x_el + 1] = layer.sol1[comp, 1, BM, x_el]
                            val_mesh[2 * x_el + 2] = layer.sol1[comp, 2, BM, x_el]
                        val_mesh[2 * (struct.n_msh_el - 1) + 1] = layer.sol1[comp, 1, BM, -1]
                        val_mesh[2 * (struct.n_msh_el - 1) + 2] = layer.sol1[comp, 2, BM, -1]
                        f_interp = interp1d(x_mesh, val_mesh, bounds_error=False, fill_value="extrapolate")
                        M_line[comp, :, BM] = f_interp(x_line)

                z_top = (h_nm - z_local_nm) / period
                z_top = z_top[None, :]
                kz_col = k_z_arr[:, None]
                P_plus = c_down[:, None] * np.exp(1j * kz_col * z_top) + c_up[:, None] * np.exp(1j * kz_col * (h_normed - z_top))
                P_minus = c_down[:, None] * np.exp(1j * kz_col * z_top) - c_up[:, None] * np.exp(1j * kz_col * (h_normed - z_top))

                Ex_lay = M_line[0] @ P_plus
                Ey_lay = M_line[1] @ P_plus
                Ez_lay = M_line[2] @ ((-1j * kz_col) * P_minus)
                Hx_lay = M_line[3] @ (((kz_col / k0)) * P_minus)
                Hy_lay = M_line[4] @ (((kz_col / k0)) * P_minus)
                Hz_lay = M_line[5] @ ((1.0 / (1j * k0)) * P_plus)
        else:
            # ThinFilm / Anallo
            alphas = np.array(layer.alphas)[layer.sort_order]
            betas = np.array(layer.betas)[layer.sort_order] if not layer.structure.world_1d else np.full_like(alphas, float(layer.betas))
            gammas = np.array(layer.calc_kz())
            n = layer.n()
            num_pws = layer.structure.num_pw_per_pol

            vec_down = np.asarray(pstack.vec_coef_down[vec_index]).flatten()
            c_down_TE = vec_down[:num_pws]
            c_down_TM = vec_down[num_pws:2 * num_pws]

            if l_idx == 0:
                c_up_TE = np.zeros(num_pws, dtype=complex)
                c_up_TM = np.zeros(num_pws, dtype=complex)
            else:
                vec_up = np.asarray(pstack.vec_coef_up[vec_index]).flatten()
                c_up_TE = vec_up[:num_pws]
                c_up_TM = vec_up[num_pws:2 * num_pws]

            norm = np.sqrt(alphas**2 + betas**2)
            norm[norm == 0] = 1e-15
            k_mag = np.sqrt(alphas**2 + betas**2 + gammas**2)
            chi_TE = np.sqrt((n * gammas) / k_mag)
            chi_TM = np.sqrt((n * k_mag) / gammas)

            e_TE = np.stack([betas / norm, -alphas / norm, np.zeros_like(alphas)], axis=0)
            e_TM = np.stack([alphas / norm, betas / norm, -norm / gammas], axis=0)

            phase_xy = np.exp(1j * (alphas[:, None] * x_line[None, :] + betas[:, None] * y_line[None, :]))
            z_norm = z_local_nm / period

            if l_idx == 0:
                prop_down = np.exp(-1j * gammas[:, None] * (h_normed - z_norm[None, :]))
                prop_up = np.zeros_like(prop_down)
            elif l_idx == num_lays - 1:
                prop_down = np.exp(-1j * gammas[:, None] * z_norm[None, :])
                prop_up = np.exp(1j * gammas[:, None] * z_norm[None, :])
            else:
                prop_down = np.exp(-1j * gammas[:, None] * (h_normed - z_norm[None, :]))
                prop_up = np.exp(1j * gammas[:, None] * z_norm[None, :])

            A_TE_down = (c_down_TE / chi_TE)[:, None, None] * prop_down[:, None, :] * phase_xy[:, :, None]
            A_TE_up = (c_up_TE / chi_TE)[:, None, None] * prop_up[:, None, :] * phase_xy[:, :, None]
            A_TM_down = (c_down_TM / chi_TM)[:, None, None] * prop_down[:, None, :] * phase_xy[:, :, None]
            A_TM_up = (c_up_TM / chi_TM)[:, None, None] * prop_up[:, None, :] * phase_xy[:, :, None]

            E_down_m = A_TE_down[None, :, :, :] * e_TE[:, :, None, None] + A_TM_down[None, :, :, :] * e_TM[:, :, None, None]
            E_up_m = A_TE_up[None, :, :, :] * e_TE[:, :, None, None] + A_TM_up[None, :, :, :] * e_TM[:, :, None, None]

            E_tot = np.sum(E_down_m + E_up_m, axis=1)
            Ex_lay = E_tot[0]
            Ey_lay = E_tot[1]
            Ez_lay = E_tot[2]

            Hx_down = betas[None, :, None, None] * E_down_m[2:3] - (-gammas[None, :, None, None]) * E_down_m[1:2]
            Hy_down = (-gammas[None, :, None, None]) * E_down_m[0:1] - alphas[None, :, None, None] * E_down_m[2:3]
            Hz_down = alphas[None, :, None, None] * E_down_m[1:2] - betas[None, :, None, None] * E_down_m[0:1]

            Hx_up = betas[None, :, None, None] * E_up_m[2:3] - gammas[None, :, None, None] * E_up_m[1:2]
            Hy_up = gammas[None, :, None, None] * E_up_m[0:1] - alphas[None, :, None, None] * E_up_m[2:3]
            Hz_up = alphas[None, :, None, None] * E_up_m[1:2] - betas[None, :, None, None] * E_up_m[0:1]

            Hx_lay = np.sum(Hx_down + Hx_up, axis=1)[0] / k0
            Hy_lay = np.sum(Hy_down + Hy_up, axis=1)[0] / k0
            Hz_lay = np.sum(Hz_down + Hz_up, axis=1)[0] / k0

        Ex_slices.append(Ex_lay)
        Ey_slices.append(Ey_lay)
        Ez_slices.append(Ez_lay)
        Hx_slices.append(Hx_lay)
        Hy_slices.append(Hy_lay)
        Hz_slices.append(Hz_lay)

    z_all = np.concatenate(z_slices)
    Ex_all = np.concatenate(Ex_slices, axis=1)
    Ey_all = np.concatenate(Ey_slices, axis=1)
    Ez_all = np.concatenate(Ez_slices, axis=1)
    Hx_all = np.concatenate(Hx_slices, axis=1)
    Hy_all = np.concatenate(Hy_slices, axis=1)
    Hz_all = np.concatenate(Hz_slices, axis=1)

    AbsE_all = np.sqrt(np.abs(Ex_all)**2 + np.abs(Ey_all)**2 + np.abs(Ez_all)**2)
    AbsH_all = np.sqrt(np.abs(Hx_all)**2 + np.abs(Hy_all)**2 + np.abs(Hz_all)**2)

    def _make_interp(arr):
        rgi = RegularGridInterpolator((u_line, z_all), arr, bounds_error=False, fill_value=None)
        return GridInterpolatorWrapper(rgi)

    ReEx = _make_interp(Ex_all.real)
    ImEx = _make_interp(Ex_all.imag)
    ReEy = _make_interp(Ey_all.real)
    ImEy = _make_interp(Ey_all.imag)
    ReEz = _make_interp(Ez_all.real)
    ImEz = _make_interp(Ez_all.imag)
    AbsE = _make_interp(AbsE_all)

    if not return_H:
        return ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE

    ReHx = _make_interp(Hx_all.real)
    ImHx = _make_interp(Hx_all.imag)
    ReHy = _make_interp(Hy_all.real)
    ImHy = _make_interp(Hy_all.imag)
    ReHz = _make_interp(Hz_all.real)
    ImHz = _make_interp(Hz_all.imag)
    AbsH = _make_interp(AbsH_all)

    return FieldBundle(
        ReEx, ImEx, ReEy, ImEy, ReEz, ImEz, AbsE,
        ReHx, ImHx, ReHy, ImHy, ReHz, ImHz, AbsH,
        k0=k0,
        interface_z_nm=interface_z_nm if lay_interest is None else [0.0, layer_heights_nm[lay_interest]],
        plane=plane.lower(),
        slice_coord=slice_coord,
        u_grid=u_line,
        v_grid=z_all,
    )


def spectra_to_xarray(stacks_list, wavelengths=None):
    """Extract T_total, R_total, A_total, and per-layer T, R, A into an xarray.Dataset.

    Args:
        stacks_list (list or Stack): List of evaluated Stack objects (or single Stack).
        wavelengths (array-like, optional): Wavelength coordinates. Defaults to stack wavelengths.

    Returns:
        xarray.Dataset: Labeled dataset with coordinates 'wavelength' (nm) and 'layer_index'.
    """
    import sys
    if "dask" not in sys.modules:
        try:
            import yaml
        except ImportError:
            sys.modules["dask"] = None
            sys.modules["dask.base"] = None
    import xarray as xr

    if not isinstance(stacks_list, (list, tuple)):
        stacks_list = [stacks_list]

    if wavelengths is None:
        wavelengths = [s.layers[0].light.wl_nm for s in stacks_list]
    wavelengths = np.asarray(wavelengths, dtype=float)

    n_wl = len(stacks_list)
    nu_layers = len(stacks_list[0].layers) - 1

    t_tot = np.zeros(n_wl, dtype=float)
    r_tot = np.zeros(n_wl, dtype=float)
    a_tot = np.zeros(n_wl, dtype=float)

    t_layers = np.zeros((nu_layers, n_wl), dtype=float)
    r_layers = np.zeros((nu_layers, n_wl), dtype=float)
    a_layers = np.zeros((nu_layers, n_wl), dtype=float)

    for w_idx, s in enumerate(stacks_list):
        t_tot[w_idx] = float(np.asarray(s.t_list[-1]).flat[0])
        r_tot[w_idx] = float(np.asarray(s.r_list[-1]).flat[0])
        a_tot[w_idx] = float(np.asarray(s.a_list[-1]).flat[0])
        for lay_idx in range(nu_layers):
            if lay_idx < len(s.t_list):
                t_layers[lay_idx, w_idx] = float(np.asarray(s.t_list[lay_idx]).flat[0])
            if lay_idx < len(s.r_list):
                r_layers[lay_idx, w_idx] = float(np.asarray(s.r_list[lay_idx]).flat[0])
            if lay_idx < len(s.a_list):
                a_layers[lay_idx, w_idx] = float(np.asarray(s.a_list[lay_idx]).flat[0])

    layer_indices = np.arange(nu_layers)

    ds = xr.Dataset(
        data_vars={
            "T_total": (["wavelength"], t_tot),
            "R_total": (["wavelength"], r_tot),
            "A_total": (["wavelength"], a_tot),
            "T_layer": (["layer_index", "wavelength"], t_layers),
            "R_layer": (["layer_index", "wavelength"], r_layers),
            "A_layer": (["layer_index", "wavelength"], a_layers),
        },
        coords={
            "wavelength": wavelengths,
            "layer_index": layer_indices,
        },
        attrs={
            "description": "EMUstack scattering and absorption spectra",
        },
    )
    return ds


def fields_vertically(stacks_list, factor_pts_vert=31, nu_pts_hori=41,
                      semi_inf_height=1.0, gradient=None, no_incoming=False,
                      add_name='', force_eq_ratio=False, colour_res=30,
                      plot_boundaries=True, save_pdf=False, save_png=False):
    """
    Plot fields in the x-y plane at chosen values of z, where z is \
    calculated from the bottom of chosen layer.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            factor_pts_vert  (int): sampling factor for fields vertically. \
            Calculated as factor_pts_vert * (epsilon*h/wl).

            nu_pts_hori  (int): in-plane fields are calculated over a mesh of \
                nu_pts_hori * nu_pts_hori points.

            semi_inf_height  (float): distance to which fields are plotting \
                in semi-infinite (sub)superstrates.

            gradient  (float): further slices calculated with given gradient \
                and -gradient. It is entitled 'specified_diagonal_slice'.\
                These slices are only calculated for ThinFilm layers.

            no_incoming  (bool): if True, plots fields in superstrate in the \
                absence of the incident driving field (i.e. only showing \
                upward propagating scattered field).

            add_name  (str): concatenate add_name to title.

            force_eq_ratio  (bool): each layer plotted on equal space.

            colour_res  (int): number of colour intervals to use.

            plot_boundaries  (bool): plot boundaries of inclusions.

            save_pdf (bool): If True, save plots as PDF. False by default.

            save_png (bool): If True, save plots as PNG. False by default.
    """

    import field_eval

    dir_name = "fields_vertically"
    if (save_pdf or save_png) and not os.path.exists(dir_name):
        os.mkdir(dir_name)

    figs = []
    # always make odd
    if nu_pts_hori % 2 == 0: nu_pts_hori += 1

    stack_num = 0
    for pstack in stacks_list:
        num_lays  = len(pstack.layers)
        period    = pstack.layers[0].structure.period
        pw        = pstack.layers[0].max_order_PWs
        wl_normed = pstack.layers[0].wl_norm()

        # Stack layers on top of one another to find heights of interfaces.
        ind_h_list = pstack.heights_norm()
        h_list = [-semi_inf_height, 0]
        for i in range(len(ind_h_list)):
            h_list.append(h_list[-1]+ind_h_list[i])
        h_list.append(h_list[-1]+semi_inf_height)
        ind_h_list.append(semi_inf_height)
        ind_h_list.insert(0, semi_inf_height)
        min_h = np.min(ind_h_list)
        h_ratio = [h/min_h for h in ind_h_list[::-1]]

        layer = pstack.layers[0]
        if isinstance(layer,mode_calcs.Anallo):
            if layer.structure.world_1d == True:
                slice_types = ['xz']
            else:
                slice_types = ['xz']#,'yz','diag+','diag-'] #rest not implemented
                if gradient != None: slice_types.append(('special+','special-'))
        else:
            if layer.structure.periodicity == '1D_array':
                slice_types = ['xz']
            else:
                slice_types = ['xz']#,'yz','diag+','diag-'] #rest not implemented
                if gradient != None: slice_types.append(('special+','special-'))

        for sli in slice_types:
            E_fields = ['Re(E_x)', 'Im(E_x)', 'Re(E_y)', 'Im(E_y)', 'Re(E_z)',
                        'Im(E_z)', 'Re(E)',  'eps_abs(E)']
            E_fields_tot = []
            epsE_fields_tot = []
            for E in E_fields:
                fig = plt.figure(figsize=(9, 12))
                if force_eq_ratio == True:
                    h_ratio = [1 for h in ind_h_list[::-1]]
                    gs = gridspec.GridSpec(num_lays, 1, wspace=0.0, hspace=0.0,
                                           width_ratios=[1],
                                           height_ratios=h_ratio)
                else:
                    gs = gridspec.GridSpec(num_lays, 1, wspace=0.0, hspace=0.0,
                                           width_ratios=[1],
                                           height_ratios=h_ratio)
                # Find maximum field value to normalise plots.
                max_E = None
                min_E = None
                for max_E_field in [0, 1]:
                    # Iterate through layers
                    for lay in range(num_lays):
                        layer = pstack.layers[lay]
                        ax1 = plt.subplot(gs[num_lays-lay-1])
                        if isinstance(layer, mode_calcs.Simmo):
                            eps = layer.n_effs**2
                        else:
                            eps = layer.n()**2
                        nu_pts_vert = int(np.round(factor_pts_vert*(np.real(np.max(eps))*ind_h_list[lay] / wl_normed)))
                        if nu_pts_vert < 11: nu_pts_vert = 11
                        if lay == 0:
                            z_range = np.linspace(h_list[lay],0.0,nu_pts_vert)
                        else:
                            z_range = np.linspace(0.0, ind_h_list[lay], nu_pts_vert)
                        # vec_coef sorted from top, everything else sorted from bottom
                        vec_index = num_lays - lay - 1

                        # If NanoStruct layer
                        if isinstance(layer, mode_calcs.Simmo):
                            name_lay = "%i_NanoStruct"% lay
                            gmsh_file_pos = 'stack_%(stack_num)s_lay_%(name)s_'% \
                                            {'name': name_lay, 'stack_num': stack_num}

                            vec_coef_fem = np.concatenate((pstack.vec_coef_down[vec_index],
                                                          pstack.vec_coef_up[vec_index]))

                            # If 2D_array plot fields using fortran routine.
                            if layer.structure.periodicity == '2D_array':
                                if max_E_field == 0 and E == 'Re(E_x)':
                                    E_fields_tot.append(None)
                                    epsE_fields_tot.append(None)
                                elif max_E_field == 1:
                                    if (save_pdf or save_png):
                                        if not os.path.exists(dir_name+"/gmsh_BMs"):
                                            os.mkdir(dir_name+"/gmsh_BMs")
                                        if not os.path.exists(dir_name+"/gmsh_BMs/anim"):
                                            os.mkdir(dir_name+"/gmsh_BMs/anim")

                                        # TODO fix the scaling issue in 2D FEM plotting!
                                        # Then remove these arbitrary scaling params.
                                        scale_plot = 2.0
                                        shift_x_plot = -.5
                                        h_normed = float(layer.structure.height_nm)/float(layer.structure.period)
                                        shift_v_plot = h_normed*0.75
                                        nnodes = 6
                                        field_eval.gmsh_plot_slice(layer.E_H_field, layer.num_BMs,
                                            layer.n_msh_el, layer.n_msh_pts, nnodes, layer.type_el,
                                            layer.structure.nb_typ_el, eps, layer.table_nod,
                                            layer.x_arr, layer.k_z, layer.sol1, vec_coef_fem,
                                            h_normed, wl_normed, gmsh_file_pos,
                                            scale_plot, shift_v_plot, shift_x_plot)

                                    z_plot = np.linspace(h_list[lay], h_list[lay+1],2)
                                    (y_axis_plot, x_axis) = np.meshgrid(z_plot,[0,1])

                                    CS = plt.contourf(x_axis, y_axis_plot,[[0,0],[0,0]],
                                        cmap=plt.cm.binary, vmin=0, vmax=1)
                                    CS = plt.contourf(x_axis, y_axis_plot,[[0,0],[0,0]],
                                        cmap=plt.cm.binary, vmin=0, vmax=1)
                                    CS.set_clim(0, 1)
                                    if abs(ind_h_list[lay]) < 0.05 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        ax1.yaxis.set_ticks([tick_half])
                                        ticktitles = ['%3.2f' % tick_half]
                                        ax1.yaxis.set_ticklabels(ticktitles)
                                    elif abs(ind_h_list[lay]) < 0.25 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        tick_int = ind_h_list[lay]/4.
                                        ticks = [tick_half-tick_int, tick_half, tick_half+tick_int]
                                        ax1.yaxis.set_ticks(ticks)
                                        ticktitles = []
                                        [ticktitles.append('%3.2f' % tick) for tick in ticks]
                                        ax1.yaxis.set_ticklabels(ticktitles)

                            # If 1D_array plot fields as follows.
                            else:
                                struct = layer.structure
                                E_slice = np.zeros((struct.n_msh_pts,nu_pts_vert), dtype = 'complex128')

                                boundary = []
                                for i in range(len(struct.type_el) - 1):
                                    if struct.type_el[i] != struct.type_el[i+1]:
                                        boundary.append(struct.x_arr[2*(i+1)])

                                if E[-3:] != '(E)':
                                    if E[5] == 'x': comp = 0
                                    if E[5] == 'y': comp = 1
                                    if E[5] == 'z': comp = 2
                                    ### sol1 = sol_P2([Ex,Ey,Ez],P2_interpolation_points,nval,nel)
                                    for BM in range(layer.num_BMs):
                                        BM_sol = layer.sol1[comp,:,BM,:]
                                        beta = layer.k_z[BM]
                                        for h in range(len(z_range)):
                                            hz = z_range[h]
                                            P_down = np.exp(1j*beta*(ind_h_list[lay]-hz))   # Introduce Propagation in -z
                                            P_up = np.exp(1j*beta*hz) # Introduce Propagation in +z

                                            coef_down = vec_coef_fem[BM] * P_down
                                            coef_up = vec_coef_fem[BM+layer.num_BMs] * P_up
                                            if E[5] == 'z':
                                                coef_tot = (coef_up - coef_down)/beta # Taking into account the change of variable for Ez
                                            else:
                                                coef_tot = coef_up + coef_down
                                            coef_tot = coef_tot[0,0]
                                            E_slice[0,h] += BM_sol[0,0] * coef_tot
                                            for x in range(struct.n_msh_el - 1):
                                                E_slice[2*x+1,h] += BM_sol[1,x] * coef_tot
                                                # E_slice[2*x+2,h] += (BM_sol[2,x] + BM_sol[0,x+1] / 2.) * coef_tot
                                                E_slice[2*x+2,h] += (BM_sol[2,x]) * coef_tot
                                            E_slice[2*x+3,h] += BM_sol[1,-1] * coef_tot
                                            E_slice[2*x+4,h] += BM_sol[2,-1] * coef_tot

                                    if max_E_field == 0 and E[0] == 'R':
                                        if E[5] == 'x':
                                            E_fields_tot.append(E_slice*np.conj(E_slice))
                                            epsE_fields_tot.append(np.zeros(np.shape(E_slice)))
                                        elif E[5] == 'y' or E[5] == 'z':
                                            E_fields_tot[lay] += E_slice*np.conj(E_slice)
                                elif E == 'eps_abs(E)':
                                    if max_E_field == 0:
                                        type_el = np.vstack((struct.type_el, struct.type_el)).reshape((-1,),order='F')
                                        # type_el = np.append(type_el, type_el[-1])
                                        type_el = np.append(type_el[-1], type_el)
                                        type_el[type_el == 1] = np.abs(eps[0])
                                        type_el[type_el == 2] = np.abs(eps[1])
                                        type_el = np.diag(type_el)
                                        epsE_fields_tot[lay] = np.dot(type_el, (E_fields_tot[lay]))
                                    else:
                                        E_slice = epsE_fields_tot[lay]
                                elif E[-3:] == '(E)' and max_E_field == 1:
                                    E_slice = np.sqrt(E_fields_tot[lay])

                                if E[0] == 'R' or E[0] == 'e':
                                    E_slice = np.real(E_slice)
                                elif E[0] == 'I':
                                    E_slice = np.imag(E_slice)

                                if E[-3:] != '(E)':  # Re(x,y,z), Im(x,y,z)
                                    max_E_lay = np.max(E_slice)
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(E_slice)
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay
                                elif E == 'Re(E)':
                                    max_E_lay = np.max(np.real(np.sqrt(E_fields_tot[lay])))
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(np.real(np.sqrt(E_fields_tot[lay])))
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay
                                else:
                                    max_E_lay = np.max(np.abs(epsE_fields_tot[lay]))
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(np.abs(epsE_fields_tot[lay]))
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay

                                if lay == 0:
                                    z_plot = np.linspace(h_list[lay], 0.0, nu_pts_vert)
                                else:
                                    z_plot = np.linspace(h_list[lay], h_list[lay+1],nu_pts_vert)
                                (y_axis_plot, x_axis) = np.meshgrid(z_plot,struct.x_arr)

                                if max_E_field == 1:
                                    if E == 'eps_abs(E)' or E == 'Re(E)':
                                        choice_cmap = plt.cm.hot
                                    else:
                                        choice_cmap = plt.cm.jet
                                    CS = plt.contourf(x_axis, y_axis_plot,
                                                      E_slice, colour_res,
                                                      cmap=choice_cmap,
                                                      vmin=min_E, vmax=max_E)
                                    CS = plt.contourf(x_axis, y_axis_plot,
                                                      E_slice, colour_res,
                                                      cmap=choice_cmap,
                                                      vmin=min_E, vmax=max_E)
                                    CS.set_clim(min_E, max_E)
                                    ax1.set_xlim((x_range[0], x_range[-1]))
                                    if abs(ind_h_list[lay]) < 0.05 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        ax1.yaxis.set_ticks([tick_half])
                                        ticktitles = ['%3.2f' % tick_half]
                                        ax1.yaxis.set_ticklabels(ticktitles)
                                    elif abs(ind_h_list[lay]) < 0.25 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        tick_int = ind_h_list[lay]/4.
                                        ticks = [tick_half-tick_int, tick_half, tick_half+tick_int]
                                        ax1.yaxis.set_ticks(ticks)
                                        ticktitles = []
                                        [ticktitles.append('%3.2f' % tick) for tick in ticks]
                                        ax1.yaxis.set_ticklabels(ticktitles)

                                    if plot_boundaries == True:
                                        start, end = ax1.get_ylim()
                                        for b in boundary:
                                            ax1.plot([b, b], [start, end], 'w',
                                                     linewidth=2)



                        # ThinFilm layer, using Plane Wave basis.
                        else:
                            x_range = np.linspace(0.0, 1.0, nu_pts_hori)
                            y_range = np.linspace(0.0, 1.0, nu_pts_hori)

                            s = layer.sort_order
                            alpha_unsrt = np.array(layer.alphas)
                            beta_unsrt = np.array(layer.betas)
                            alpha = alpha_unsrt[s]
                            if layer.structure.world_1d is True:
                                beta = beta_unsrt
                            else:
                                beta = beta_unsrt[s]
                            gamma = np.array(layer.calc_kz())
                            n = layer.n()
                            num_pws = layer.structure.num_pw_per_pol

                            if lay == num_lays-1 and no_incoming == True:
                                vec_coef_down = np.zeros((2*num_pws), dtype='complex')
                            else:
                                vec_coef_down = np.array(pstack.vec_coef_down[vec_index]).flatten()
                            vec_coef_down_TE = vec_coef_down[0:num_pws]
                            vec_coef_down_TM = vec_coef_down[num_pws::]

                            if lay == 0:
                                vec_coef_up = np.zeros((2*num_pws), dtype='complex')
                            else:
                                vec_coef_up = np.array(pstack.vec_coef_up[vec_index]).flatten()
                            vec_coef_up_TE = vec_coef_up[0:num_pws]
                            vec_coef_up_TM = vec_coef_up[num_pws::]

                            norm = np.sqrt(alpha**2+beta**2)
                            k = np.sqrt(alpha**2+beta**2+gamma**2)
                            chi_TE = np.sqrt((n*gamma)/k)
                            chi_TM = np.sqrt((n*k)/gamma)
                            E_TE_x = beta/norm
                            E_TE_y = -1*alpha/norm
                            E_TE_z = np.array(np.zeros(np.size(E_TE_x)))
                            E_TM_x = alpha/norm
                            E_TM_y = beta/norm
                            E_TM_z = -1*norm/gamma

                            if E[-3:] != '(E)':
                                if E[5] == 'x':
                                    TE_coef = E_TE_x
                                    TM_coef = E_TM_x
                                elif E[5] == 'y':
                                    TE_coef = E_TE_y
                                    TM_coef = E_TM_y
                                elif E[5] == 'z':
                                    TE_coef = E_TE_z
                                    TM_coef = E_TM_z

                            eta_TE_down = (vec_coef_down_TE*TE_coef)/chi_TE
                            eta_TM_down = (vec_coef_down_TM*TM_coef)/chi_TM
                            eta_TE_up = (vec_coef_up_TE*TE_coef)/chi_TE
                            eta_TM_up = (vec_coef_up_TM*TM_coef)/chi_TM

                            x_axis = np.zeros((nu_pts_hori, nu_pts_hori), dtype='float')
                            y_axis = np.zeros((nu_pts_hori, nu_pts_hori), dtype='float')

                            if sli == 'xz':
                                x1 = x_range
                                if layer.structure.world_1d == True:
                                    y1 = np.array([0])
                                else:
                                    y1 = np.array([0, 0.5])
                                (y_axis, x_axis) = np.meshgrid(z_range, x1)
                            elif sli == 'yz':
                                x1 = np.array([0, 0.5])
                                y1 = y_range
                                (y_axis, x_axis) = np.meshgrid(z_range, y1)
                            elif sli == 'diag+':
                                x1 = x_range
                                y1 = np.array([0])
                                y2 = x_range
                                (y_axis, x_axis) = np.meshgrid(z_range, sqrt(2)*x1)
                            elif sli == 'diag-':
                                x1 = x_range[::-1]
                                y1 = np.array([0])
                                y2 = x_range
                                (y_axis, x_axis) = np.meshgrid(z_range, sqrt(2)*x1[::-1])
                            elif sli == 'special+':
                                x1 = x_range
                                y1 = np.array([0])
                                y2 = gradient*x_range
                                for i in range(np.size(y2)):
                                    if y2[i] > 1:
                                        y2 = np.resize(y2, (i,))
                                        x1 = np.resize(x1, (i,))
                                        break
                                (y_axis, x_axis) = np.meshgrid(z_range,sqrt(1+gradient**2)*x1)
                            elif sli == 'special-':
                                x1 = x_range[::-1]
                                y1 = np.array([0])
                                y2 = gradient*x_range
                                for i in range(np.size(y2)):
                                    if y2[i] > 1:
                                        y2 = np.resize(y2,(i,))
                                        x1 = np.resize(x1,(i,))
                                        break
                                (y_axis, x_axis) = np.meshgrid(z_range, sqrt(1+gradient**2)*(x1[::-1]-x1[-1]))

                            E_field = np.zeros((np.size(x1), np.size(y1),
                                               np.size(z_range)), dtype='complex')

                            for z in range(np.size(z_range)):
                                for y in range(np.size(y1)):
                                    for x in range(np.size(x1)):
                                        if sli == 'diag+' or sli == 'diag-' or sli == 'special+' or sli == 'special-':
                                            if layer.structure.height_nm == 'semi_inf':
                                                expo_down = np.exp(1j*(alpha*x1[x]+beta*y2[x]-gamma*z_range[z]))
                                            else:
                                                expo_down = np.exp(1j*(alpha*x1[x]+beta*y2[x]-gamma*(z_range[z]-ind_h_list[lay])))
                                            expo_up = np.exp(1j*(alpha*x1[x]+beta*y2[x]+gamma*z_range[z]))
                                        else:
                                            if lay == 0 or lay == num_lays-1:
                                                expo_down = np.exp(1j*(alpha*x1[x]+beta*y1[y]-gamma*z_range[z]))
                                            else:
                                                expo_down = np.exp(1j*(alpha*x1[x]+beta*y1[y]-gamma*(z_range[z]-ind_h_list[lay])))
                                            expo_up = np.exp(1j*(alpha*x1[x]+beta*y1[y]+gamma*z_range[z]))

                                        E_field[x,y,z] = np.sum(eta_TE_down*expo_down + eta_TE_up*expo_up) + np.sum(eta_TM_down*expo_down + eta_TM_up*expo_up)

                            if lay == 0:
                                z_plot = np.linspace(h_list[lay],0.0,nu_pts_vert)
                            else:
                                z_plot = np.linspace(h_list[lay],h_list[lay+1],nu_pts_vert)
                            (y_axis_plot, x_axis) = np.meshgrid(z_plot, x_range)

                            if sli == 'xz':
                                E_slice = E_field[:, 0, :]
                                if max_E_field == 0:
                                    if E[-3:] != '(E)' and E[0] == 'R':
                                        if E[5] == 'x':
                                            E_fields_tot.append(E_slice*np.conj(E_slice))
                                            epsE_fields_tot.append(np.zeros(np.shape(E_slice)))
                                        elif E[5] == 'y' or E[5] == 'z':
                                            E_fields_tot[lay] += E_slice*np.conj(E_slice)
                                    elif E == 'eps_abs(E)':
                                        epsE_fields_tot[lay] = np.real(eps) * (E_fields_tot[lay])
                                elif max_E_field == 1 and E == 'eps_abs(E)':
                                    E_slice = epsE_fields_tot[lay]
                                elif max_E_field == 1 and E[-3:] == '(E)':
                                    E_slice = np.sqrt(E_fields_tot[lay])

                                if E[0] == 'R' or E[0] == 'e':
                                    E_slice = np.real(E_slice)
                                elif E[0] == 'I':
                                    E_slice = np.imag(E_slice)

                                if E[-3:] != '(E)':
                                    max_E_lay = np.max(E_slice)
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(E_slice)
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay
                                elif E == 'Re(E)':
                                    max_E_lay = np.max(np.real(np.sqrt(E_fields_tot[lay])))
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(np.real(np.sqrt(E_fields_tot[lay])))
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay
                                else:
                                    max_E_lay = np.max(np.real(epsE_fields_tot[lay]))
                                    if max_E is None or max_E_lay > max_E:
                                        max_E = max_E_lay
                                    min_E_lay = np.min(np.real(epsE_fields_tot[lay]))
                                    if min_E is None or min_E_lay < min_E:
                                        min_E = min_E_lay

                                if max_E_field == 1:
                                    if E == 'eps_abs(E)' or E == 'Re(E)':
                                        choice_cmap = plt.cm.hot
                                    else:
                                        choice_cmap = plt.cm.jet
                                    CS = plt.contourf(x_axis,y_axis_plot,E_slice,
                                        colour_res, cmap=choice_cmap, vmin=min_E, vmax=max_E)
                                    # hack to remove white contour lines
                                    CS = plt.contourf(x_axis,y_axis_plot,E_slice,
                                        colour_res, cmap=choice_cmap, vmin=min_E, vmax=max_E)
                                    CS.set_clim(min_E,max_E)
                                    ax1.set_xlim((x_range[0],x_range[-1]))
                                    if abs(ind_h_list[lay]) < 0.05 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        ax1.yaxis.set_ticks([tick_half])
                                        ticktitles = ['%3.2f' % tick_half]
                                        ax1.yaxis.set_ticklabels(ticktitles)
                                    elif abs(ind_h_list[lay]) < 0.25 * np.sum(ind_h_list):
                                        tick_half = (h_list[lay]+h_list[lay+1])/2.
                                        tick_int = ind_h_list[lay]/4.
                                        ticks = [tick_half-tick_int, tick_half, tick_half+tick_int]
                                        ax1.yaxis.set_ticks(ticks)
                                        ticktitles = []
                                        [ticktitles.append('%3.2f' % tick) for tick in ticks]
                                        ax1.yaxis.set_ticklabels(ticktitles)

                            # elif sli == 'yz':
                            #     E_slice = E_field[0, :, :]
                            #     if max_E_field == 0:
                            #         if E[-3:] != '(E)' and E[0] == 'R':
                            #             if E[5] == 'x':
                            #                 E_fields_tot.append(E_slice*np.conj(E_slice))
                            #                 epsE_fields_tot.append(np.zeros(np.shape(E_slice)))
                            #             elif E[5] == 'y' or E[5] == 'z':
                            #                 E_fields_tot[lay] += E_slice*np.conj(E_slice)
                            #         elif E == 'eps_abs(E)':
                            #             epsE_fields_tot[lay] = np.real(eps) * (E_fields_tot[lay])
                            #     elif max_E_field == 1 and E == 'eps_abs(E)':
                            #         E_slice = epsE_fields_tot[lay]
                            #     elif max_E_field == 1 and E[-3:] == '(E)':
                            #         E_slice = np.sqrt(E_fields_tot[lay])

                            #     if E[0] == 'R' or E[0] == 'e':
                            #         E_slice = np.real(E_slice)
                            #     elif E[0] == 'I':
                            #         E_slice = np.imag(E_slice)

                            #     if E[-3:] != '(E)':
                            #         max_E_lay = np.max(E_slice)
                            #         if max_E_lay > max_E or max_E == None:
                            #             max_E = max_E_lay
                            #         min_E_lay = np.min(E_slice)
                            #         if min_E_lay < min_E or min_E == None:
                            #             min_E = min_E_lay
                            #     elif E == 'Re(E)':
                            #         max_E_lay = np.max(np.real(np.sqrt(E_fields_tot[lay])))
                            #         if max_E_lay > max_E or max_E == None:
                            #             max_E = max_E_lay
                            #         min_E_lay = np.min(np.real(np.sqrt(E_fields_tot[lay])))
                            #         if min_E_lay < min_E or min_E == None:
                            #             min_E = min_E_lay
                            #     else:
                            #         max_E_lay = np.max(np.real(epsE_fields_tot[lay]))
                            #         if max_E_lay > max_E or max_E == None:
                            #             max_E = max_E_lay
                            #         min_E_lay = np.min(np.real(epsE_fields_tot[lay]))
                            #         if min_E_lay < min_E or min_E == None:
                            #             min_E = min_E_lay

                            #     if max_E_field == 1:
                            #         if E == 'eps_abs(E)' or E == 'Re(E)':
                            #             choice_cmap = plt.cm.hot
                            #         else:
                            #             choice_cmap = plt.cm.jet
                            #         CS = plt.contourf(x_axis, y_axis_plot, E_slice,
                            #             colour_res, cmap=choice_cmap, vmin=min_E, vmax=max_E)
                            #         CS.set_clim(min_E, max_E)
                            #         ax1.set_xlim((x_range[0], x_range[-1]))
                            #         if abs(ind_h_list[lay]) < 0.05 * np.sum(ind_h_list):
                            #             tick_half = (h_list[lay]+h_list[lay+1])/2.
                            #             ax1.yaxis.set_ticks([tick_half])
                            #             ticktitles = ['%3.2f' % tick_half]
                            #             ax1.yaxis.set_ticklabels(ticktitles)
                            #         elif abs(ind_h_list[lay]) < 0.25 * np.sum(ind_h_list):
                            #             tick_half = (h_list[lay]+h_list[lay+1])/2.
                            #             tick_int = ind_h_list[lay]/4.
                            #             ticks = [tick_half-tick_int, tick_half, tick_half+tick_int]
                            #             ax1.yaxis.set_ticks(ticks)
                            #             ticktitles = []
                            #             [ticktitles.append('%3.2f' % tick) for tick in ticks]
                            #             ax1.yaxis.set_ticklabels(ticktitles)

                        if lay == 0 and sli == 'xz':
                            ax1.set_xlabel('x (d)')
                        elif lay == 0 and sli == 'yz':
                            ax1.set_xlabel('y (d)')
                        else:
                            ax1.set_xticklabels(())

                cax = fig.add_axes([0.6, 0.1, 0.03, 0.8])
                cb = fig.colorbar(plt.contourf([0, 1], [0, 1], [[min_E, min_E],
                                  [max_E, max_E]], colour_res,
                                  cmap=choice_cmap, vmin=min_E, vmax=max_E),
                                  cax=cax)
                if hasattr(cb, 'set_clim'):
                    cb.set_clim(min_E, max_E)
                elif hasattr(cb, 'mappable') and hasattr(cb.mappable, 'set_clim'):
                    cb.mappable.set_clim(min_E, max_E)
                if E == 'eps_abs(E)':
                    cax.set_ylabel(r'$|\epsilon|$ |E|$^2$')
                elif E == 'Re(E)':
                    cax.set_ylabel(r'|E|')
                else:
                    cax.set_ylabel(E)

                if save_pdf:
                    plt.savefig('%(dir_name)s/stack_%(stack_num)s_E_%(slice)s_slice_%(comp)s%(add)s.pdf'% \
                        {'dir_name': dir_name, 'comp': E, 'slice': sli,\
                         'stack_num': stack_num, 'add': add_name},
                            bbox_inches='tight')
                    plt.close(fig)
                elif save_png:
                    plt.savefig('%(dir_name)s/stack_%(stack_num)s_E_%(slice)s_slice_%(comp)s%(add)s.png'% \
                        {'dir_name': dir_name, 'comp': E, 'slice': sli,\
                         'stack_num': stack_num, 'add': add_name},
                            bbox_inches='tight')
                    plt.close(fig)
                else:
                    figs.append(fig)

                            # elif sli == 'yz':
                            #     for x_of_yz in range(np.size(x1)):
                            #         fig1 = plt.figure(num=None, figsize=(12, 21), dpi=80, facecolor='w', edgecolor='k')
                            #         for i in range(len(E_super)):
                            #             ax1 = fig1.add_subplot(4, 1, i+1)
                            #             if re_im == 'real':
                            #                 E_slice = np.real(E_super[i][x_of_yz,:,:])
                            #             elif re_im == 'imag':
                            #                 E_slice = np.imag(E_super[i][x_of_yz,:,:])
                            #             y_min = y_range[0]
                            #             y_max = y_range[-1]
                            #             z_min = z_range[0]
                            #             z_max = z_range[-1]
                            #             cmap = plt.get_cmap('jet')
                            #             CS = plt.contourf(x_axis,y_axis,E_slice,15,cmap=cmap)
                            #             plt.axis([y_min,y_max,z_min,z_max])
                            #             cbar = plt.colorbar()
                            #             cbar.ax.set_ylabel(r'%(p)s E_x'%{'p':p})
                            #             ax1.set_xlabel('y (d)')
                            #             ax1.set_ylabel('z (d)')
                            #             if scale_axis == True: ax1.axis('scaled')
                            #             ax1.xaxis.set_ticks([x_min,x_max])
                            #             ax1.set_ylim((z_min,z_max))
                            #             if np.abs(z_max-z_min) < x_max: ax1.yaxis.set_ticks([z_min,z_max])

                                    # plt.suptitle('%(name)s \n E_yz_slice_%(p)s, x = %(x_pos)s, heights = %(h)s \n \
                                    #     $\lambda$ = %(wl)snm, period = %(d)f, PW = %(pw)i, %(add)s' % \
                                    #     {'name' : name_lay, 'h':heights_list, 'p' : p, 'x_pos' : x1[x_of_yz],'wl' : wl, \
                                    #     'd' : period, 'pw' : pw, 'add' : add_name} + '\n'
                                    #     + '# prop. ords = %(prop)s, # evan. ords = %(evan)s , \
                                    #     n = %(n)s, k = %(k)s' % {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                                    # plt.savefig('%(dir_name)s/stack_%(stack_num)s_lay_%(name)s_E_yz_slice=%(x_pos)s_wl=%(wl)s_%(p)s%(add)s.pdf'% \
                                    #     {'dir_name' : dir_name, 'p':p, 'wl' : wl, 'x_pos' : x1[x_of_yz],\
                                    #     'name' : name_lay,'stack_num':stack_num, 'add' : add_name})

                            # elif sli == 'diag+':
                            #     diag = 1
                            #     fig1 = plt.figure(num=None, figsize=(12,21), dpi=80, facecolor='w', edgecolor='k')
                            #     for i in range(len(E_super)):
                            #         ax1 = fig1.add_subplot(4,1,i+1)
                            #         if re_im == 'real':
                            #             E_slice = np.real(E_super[i][:,0,:])
                            #         elif re_im == 'imag':
                            #             E_slice = np.imag(E_super[i][:,0,:])
                            #         y_min = y_range[0]
                            #         y_max = np.around(np.sqrt(2)*y_range[-1],decimals=2)
                            #         z_min = z1[0]
                            #         z_max = z1[-1]
                            #         cmap = plt.get_cmap('jet')
                            #         CS = plt.contourf(x_axis,y_axis,E_slice,15,cmap=cmap)
                            #         plt.axis([y_min,y_max,z_min,z_max])
                            #         cbar = plt.colorbar()
                            #         cbar.ax.set_ylabel(r'%(p)s E_x'%{'p':p})
                            #         ax1.set_xlabel('x=%(diag)sy (d)'%{'diag':diag})
                            #         ax1.set_ylabel('z (d)')
                            #         if scale_axis == True: ax1.axis('scaled')
                            #         ax1.xaxis.set_ticks([y_min,y_max])
                            #         ax1.set_xlim((y_min,y_max))
                            #         ax1.set_ylim((z_min,z_max))
                            #         if np.abs(z_max-z_min) < x_max: ax1.yaxis.set_ticks([z_min,z_max])

                            #     # plt.suptitle('%(name)s \n E_diagonal_slice_%(p)s, y = %(diag)sx, heights = %(h)s \n\
                            #     #     $\lambda$ = %(wl)f, period = %(d)f, PW = %(pw)i, %(add)s' % \
                            #     #     {'name' : name_lay, 'h':heights_list, 'p':p,'diag' : diag,'wl' : wl, 'd' : period, \
                            #     #     'pw' : pw, 'add' : add_name} + '\n'
                            #     #     + '# prop. ords = %(prop)s, # evan. ords = %(evan)s , n = %(n)s, k = %(k)s' % \
                            #     #     {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                            #     # plt.savefig('%(dir_name)s/stack_%(stack_num)s_lay_%(name)s_E_diag_slice_y=%(diag)sx_wl=%(wl)s_%(p)s%(add)s.pdf'% \
                            #     #     {'dir_name' : dir_name, 'p':p,'wl' : wl, 'diag' : diag, \
                            #     #     'name' : name_lay,'stack_num':stack_num, 'add' : add_name})

                            # elif sli == 'diag-':
                            #     diag = -1
                            #     fig1 = plt.figure(num=None, figsize=(12,21), dpi=80, facecolor='w', edgecolor='k')
                            #     for i in range(len(E_super)):
                            #         ax1 = fig1.add_subplot(4,1,i+1)
                            #         if re_im == 'real':
                            #             E_slice = np.real(E_super[i][:,0,:])
                            #         elif re_im == 'imag':
                            #             E_slice = np.imag(E_super[i][:,0,:])
                            #         y_min = y_range[0]
                            #         y_max = np.around(np.sqrt(2)*y_range[-1],decimals=2)
                            #         z_min = z1[0]
                            #         z_max = z1[-1]
                            #         cmap = plt.get_cmap('jet')
                            #         CS = plt.contourf(x_axis,y_axis,E_slice,15,cmap=cmap)
                            #         plt.axis([y_min,y_max,z_min,z_max])
                            #         cbar = plt.colorbar()
                            #         cbar.ax.set_ylabel(r'%(p)s E_x'%{'p':p})
                            #         ax1.set_xlabel('x=%(diag)sy (d)'%{'diag':diag})
                            #         ax1.set_ylabel('z (d)')
                            #         if scale_axis == True: ax1.axis('scaled')
                            #         ax1.xaxis.set_ticks([y_min,y_max])
                            #         ax1.set_xlim((y_min,y_max))
                            #         ax1.set_ylim((z_min,z_max))
                            #         if np.abs(z_max-z_min) < x_max: ax1.yaxis.set_ticks([z_min,z_max])

                            #     # plt.suptitle('%(name)s \n E_diagonal_slice_%(p)s, y = %(diag)sx, heights = %(h)s \n\
                            #     #     $\lambda$ = %(wl)f nm, period = %(d)f, PW = %(pw)i, %(add)s' % \
                            #     #         {'name' : name_lay, 'h':heights_list,'diag' : diag, 'p' : p,'wl' : wl, \
                            #     #         'd' : period, 'pw' : pw, 'add' : add_name} + '\n'  +
                            #     #     '# prop. ords = %(prop)s, # evan. ords = %(evan)s , n = %(n)s, k = %(k)s' % \
                            #     #     {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                            #     # plt.savefig('%(dir_name)s/stack_%(stack_num)s_lay_%(name)s_E_diag_slice_y=%(diag)sx_wl=%(wl)s_%(p)s%(add)s.pdf'% \
                            #     #     {'dir_name' : dir_name, 'p':p,'wl' : wl, 'diag' : diag, \
                            #     #     'name' : name_lay,'stack_num':stack_num, 'add' : add_name})

                            # elif sli == 'special+':
                            #     diag = 1
                            #     fig1 = plt.figure(num=None, figsize=(12,21), dpi=80, facecolor='w', edgecolor='k')
                            #     for i in range(len(E_super)):
                            #         ax1 = fig1.add_subplot(4,1,i+1)
                            #         if re_im == 'real':
                            #             E_slice = np.real(E_super[i][:,0,:])
                            #         elif re_im == 'imag':
                            #             E_slice = np.imag(E_super[i][:,0,:])
                            #         y_min = y_range[0]
                            #         y_max = np.around(sqrt(1+gradient**2)*x1[-1],decimals=2)
                            #         z_min = z1[0]
                            #         z_max = z1[-1]
                            #         cmap = plt.get_cmap('jet')
                            #         CS = plt.contourf(x_axis,y_axis,E_slice,15,cmap=cmap)
                            #         plt.axis([y_min,y_max,z_min,z_max])
                            #         cbar = plt.colorbar()
                            #         cbar.ax.set_ylabel(r'%(p)s E_x'%{'p':p})
                            #         ax1.set_xlabel('y=%(diag)sx (d)'%{'diag':diag*gradient})
                            #         ax1.set_ylabel('z (d)')
                            #         if scale_axis == True: ax1.axis('scaled')
                            #         ax1.xaxis.set_ticks([y_min,y_max])
                            #         ax1.set_ylim((z_min,z_max))
                            #         ax1.set_xlim((y_min,y_max))
                            #         if np.abs(z_max-z_min) < x_max: ax1.yaxis.set_ticks([z_min,z_max])

                            #     # plt.suptitle('%(name)s \n E_specified_diagonal_slice_%(p)s, y = %(diag*gradient)sx, (x,y):(0,0) to (%(x)s,1), heights = %(h)s \n\
                            #     #     $\lambda$ = %(wl)f, period = %(d)f, PW = %(pw)i, %(add)s' % \
                            #     #     {'name' : name_lay, 'h':heights_list,'diag*gradient':diag*gradient, 'p':p,'wl' : wl, 'd' : period, \
                            #     #     'pw' : pw, 'x':x1[-1], 'add' : add_name} + '\n' +
                            #     #     '# prop. ords = %(prop)s, # evan. ords = %(evan)s , n = %(n)s, k = %(k)s' % \
                            #     #     {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                            #     # plt.savefig('%(dir_name)s/stack_%(stack_num)s_lay_%(name)s_E_specified_diagonal_slice_y=%(diag*gradient)sx_wl=%(wl)s_%(p)s%(add)s.pdf'% \
                            #     #     {'dir_name' : dir_name, 'diag*gradient':diag*gradient, 'p':p,'wl' : wl,\
                            #     #     'name' : name_lay,'stack_num':stack_num, 'add' : add_name})

                            # elif sli == 'special-':
                            #     diag = -1
                            #     fig1 = plt.figure(num=None, figsize=(12,21), dpi=80, facecolor='w', edgecolor='k')
                            #     for i in range(len(E_super)):
                            #         ax1 = fig1.add_subplot(4,1,i+1)
                            #         if re_im == 'real':
                            #             E_slice = np.real(E_super[i][:,0,:])
                            #         elif re_im == 'imag':
                            #             E_slice = np.imag(E_super[i][:,0,:])
                            #         y_min = y_range[0]
                            #         y_max = np.around(sqrt(1+gradient**2)*(x1[0]-x1[-1]),decimals=2)
                            #         z_min = z1[0]
                            #         z_max = z1[-1]
                            #         cmap = plt.get_cmap('jet')
                            #         CS = plt.contourf(x_axis,y_axis,E_slice,15,cmap=cmap)
                            #         plt.axis([y_min,y_max,z_min,z_max])
                            #         cbar = plt.colorbar()
                            #         cbar.ax.set_ylabel(r'%(p)s E_x'%{'p':p})
                            #         ax1.set_xlabel('y=%(diag)sx (d)'%{'diag':diag*gradient})
                            #         ax1.set_ylabel('z (d)')
                            #         if scale_axis == True: ax1.axis('scaled')
                            #         ax1.xaxis.set_ticks([y_min,y_max])
                            #         ax1.set_ylim((z_min,z_max))
                            #         ax1.set_xlim((y_min,y_max))
                            #         if np.abs(z_max-z_min) < x_max: ax1.yaxis.set_ticks([z_min,z_max])

                            #     # plt.suptitle('%(name)s \n E_specified_diagonal_slice_%(p)s, y = %(diag*gradient)sx, (x,y):(1,0) to (%(x)s,1), heights = %(h)s \n\
                            #     #     $\lambda$ = %(wl)f, period = %(d)f, PW = %(pw)i, %(add)s' % \
                            #     #     {'name' : name_lay, 'h':heights_list,'diag*gradient':diag*gradient, 'p':p,'wl' : wl, 'd' : period, \
                            #     #     'pw' : pw,'x':x1[-1], 'add' : add_name} + '\n' +
                            #     #     '# prop. ords = %(prop)s, # evan. ords = %(evan)s , n = %(n)s, k = %(k)s' % \
                            #     #     {'evan' : evan, 'prop' : prop, 'n' : n, 'k' : k[0]})
                            #     # plt.savefig('%(dir_name)s/stack_%(stack_num)s_lay_%(name)s_E_specified_diagonal_slice_y=%(diag*gradient)sx_wl=%(wl)s_%(p)s%(add)s.pdf'% \
                            #     #     {'dir_name' : dir_name, 'diag*gradient':diag*gradient, 'p':p,'wl' : wl,\
                            #     #     'name' : name_lay,'stack_num':stack_num, 'add' : add_name})

        stack_num += 1

    return figs


def field_values(stacks_list, lay_interest=0, xyz_values=[(0.1,0.1,0.1)], save_files=False):
    """
    Save electric field values at given x-y-z points. Points must be within \
    a ThinFilm layer. In txt file fields are given as \
    Re(Ex) Im(Ex) Re(Ey) Im(Ey) Re(Ez) Im(Ez) ``|E|``

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            lay_interest  (int): the index of the layer considered within \
                the stack. Must be a ThinFilm layer.

            xyz_values  (list): list of distances in normalised units of (d) \
                from top surface of layer at which to calculate fields. \
                For semi-inf substrate then z_value is distance from \
                top of this layer (i.e. bottom interface of stack).

            save_files (bool): If True, save npz and txt files to disk. False by default.
    """

    dir_name = 'field_values'
    if save_files and not os.path.exists(dir_name):
        os.mkdir(dir_name)

    stack_num = 0
    super_points = []
    for pstack in stacks_list:
        try:
            if not isinstance(pstack.layers[lay_interest],mode_calcs.Anallo):
                raise ValueError

            num_lays = len(pstack.layers)
            wl = np.around(pstack.layers[-1].light.wl_nm,decimals=2)

            if lay_interest == 0: name_lay = "0_Substrate"
            elif lay_interest == num_lays-1: name_lay = "%i_Superstrate" % (num_lays-1)
            else: name_lay = "%i_Thin_Film" % lay_interest

            n = pstack.layers[lay_interest].n()
            period = pstack.layers[-1].structure.period
            PWordtot = pstack.layers[lay_interest].structure.num_pw_per_pol
            s = pstack.layers[lay_interest].sort_order
            alpha_unsrt = np.array(pstack.layers[lay_interest].alphas)
            beta_unsrt = np.array(pstack.layers[lay_interest].betas)
            alpha = alpha_unsrt[s]
            if pstack.layers[lay_interest].structure.world_1d is True:
                beta = beta_unsrt
            else:
                beta = beta_unsrt[s]
            gamma = np.array(pstack.layers[lay_interest].calc_kz())

            vec_coef_down = np.array(pstack.vec_coef_down[num_lays-1-lay_interest]).flatten()
            vec_coef_down_TE = vec_coef_down[0:PWordtot]
            vec_coef_down_TM = vec_coef_down[PWordtot::]
            if lay_interest == 0:
                vec_coef_up = np.zeros((2*PWordtot), dtype = 'complex')
            else:
                vec_coef_up = np.array(pstack.vec_coef_up[num_lays-1-lay_interest]).flatten()

            vec_coef_up_TE = vec_coef_up[0:PWordtot]
            vec_coef_up_TM = vec_coef_up[PWordtot::]

            norm = np.sqrt(alpha**2+beta**2)
            k = np.sqrt(alpha**2+beta**2+gamma**2)
            chi_TE = np.sqrt((n*gamma)/k)
            chi_TM = np.sqrt((n*k)/gamma)
            E_TE_x = beta/norm
            E_TE_y = -1*alpha/norm
            E_TE_z = np.array(np.zeros(np.size(E_TE_x)))
            E_TM_x = alpha/norm
            E_TM_y = beta/norm
            E_TM_z = -1*norm/gamma

            eta_TE_x_down = (vec_coef_down_TE*E_TE_x)/chi_TE
            eta_TE_y_down = (vec_coef_down_TE*E_TE_y)/chi_TE
            eta_TE_z_down = (vec_coef_down_TE*E_TE_z)/chi_TE
            eta_TM_x_down = (vec_coef_down_TM*E_TM_x)/chi_TM
            eta_TM_y_down = (vec_coef_down_TM*E_TM_y)/chi_TM
            eta_TM_z_down = (vec_coef_down_TM*E_TM_z)/chi_TM

            eta_TE_x_up = (vec_coef_up_TE*E_TE_x)/chi_TE
            eta_TE_y_up = (vec_coef_up_TE*E_TE_y)/chi_TE
            eta_TE_z_up = (vec_coef_up_TE*E_TE_z)/chi_TE
            eta_TM_x_up = (vec_coef_up_TM*E_TM_x)/chi_TM
            eta_TM_y_up = (vec_coef_up_TM*E_TM_y)/chi_TM
            eta_TM_z_up = (vec_coef_up_TM*E_TM_z)/chi_TM

            calc_E_TE_x_array = np.zeros(len(xyz_values),dtype='complex')
            calc_E_TE_y_array = np.zeros(len(xyz_values),dtype='complex')
            calc_E_TE_z_array = np.zeros(len(xyz_values),dtype='complex')
            calc_E_TM_x_array = np.zeros(len(xyz_values),dtype='complex')
            calc_E_TM_y_array = np.zeros(len(xyz_values),dtype='complex')
            calc_E_TM_z_array = np.zeros(len(xyz_values),dtype='complex')

            for i in range(len(xyz_values)):

                (x1,y1,z1) = np.array(xyz_values[i])#/float(pstack.layers[lay_interest].structure.period)

                if pstack.layers[lay_interest].structure.world_1d == True: y1 = 0

                if lay_interest == 0: z1 = -1*z1
                else: z1 = np.abs(z1)

                if pstack.layers[lay_interest].structure.height_nm == 'semi_inf':
                    calc_expo_down = np.exp(1j*(alpha*x1+beta*y1-gamma*z1))
                else:
                    calc_expo_down = np.exp(1j*(alpha*x1+beta*y1-gamma*(z1-float(pstack.layers[lay_interest].structure.height_nm)/period)))
                calc_expo_up = np.exp(1j*(alpha*x1+beta*y1+gamma*z1))

                calc_E_TE_x = np.sum(eta_TE_x_down*calc_expo_down + eta_TE_x_up*calc_expo_up)
                calc_E_TE_y = np.sum(eta_TE_y_down*calc_expo_down + eta_TE_y_up*calc_expo_up)
                calc_E_TE_z = np.sum(eta_TE_z_down*calc_expo_down + eta_TE_z_up*calc_expo_up)
                calc_E_TM_x = np.sum(eta_TM_x_down*calc_expo_down + eta_TM_x_up*calc_expo_up)
                calc_E_TM_y = np.sum(eta_TM_y_down*calc_expo_down + eta_TM_y_up*calc_expo_up)
                calc_E_TM_z = np.sum(eta_TM_z_down*calc_expo_down + eta_TM_z_up*calc_expo_up)
                calc_E_TE_x_array[i] = calc_E_TE_x
                calc_E_TE_y_array[i] = calc_E_TE_y
                calc_E_TE_z_array[i] = calc_E_TE_z
                calc_E_TM_x_array[i] = calc_E_TM_x
                calc_E_TM_y_array[i] = calc_E_TM_y
                calc_E_TM_z_array[i] = calc_E_TM_z
            calc_E_x_array = calc_E_TE_x_array + calc_E_TM_x_array
            calc_E_y_array = calc_E_TE_y_array + calc_E_TM_y_array
            calc_E_z_array = calc_E_TE_z_array + calc_E_TM_z_array
            calc_E_tot_array = np.sqrt(calc_E_x_array*np.conj(calc_E_x_array) \
                                +calc_E_y_array*np.conj(calc_E_y_array)+calc_E_z_array*np.conj(calc_E_z_array))

            if save_files:
                np.savez('%(dir_name)s/%(stack_num)s_E_calc_points_%(name)s_wl_%(wl)s'% \
                                    {'dir_name':dir_name, 'wl':wl,'name' : name_lay,'stack_num':stack_num},\
                                    calc_E_x_array=calc_E_x_array,calc_E_y_array=calc_E_y_array,\
                                    calc_E_z_array=calc_E_z_array,calc_E_tot_array=calc_E_tot_array)

                np.savetxt('%(dir_name)s/%(stack_num)s_E_calc_points_%(name)s_wl_%(wl)s.txt'% \
                                    {'dir_name':dir_name, 'wl':wl,'name' : name_lay,'stack_num':stack_num},\
                                    np.array([np.real(calc_E_x_array), np.imag(calc_E_x_array), np.real(calc_E_y_array),\
                                    np.imag(calc_E_y_array), np.real(calc_E_z_array), np.imag(calc_E_z_array), np.real(calc_E_tot_array)]))
            stack_num += 1
            super_points.append([calc_E_x_array,calc_E_y_array,calc_E_z_array,calc_E_tot_array])

        except ValueError:
            print("field_values can only calculate field values in ThinFilms."\
            "\nPlease select a different lay_interest.\n")

        return super_points


def fields_3d(stacks_list, lay_interest=1):
    """
    Plot fields in 3D using gmsh.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            lay_interest  (int): the index of the layer considered within \
            the stack.
    """
    import field_eval
    import subprocess

    nnodes=6
    dir_name = "3d_fields"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name = "3d_fields/anim"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    stack_num = 0
    for pstack in stacks_list:
        try:
            if not isinstance(pstack.layers[lay_interest],mode_calcs.Simmo):
                raise ValueError

            meat = pstack.layers[lay_interest]
            if not meat.structure.periodicity == '2D_array':
                raise ValueError

            gmsh_file_pos = meat.structure.mesh_file[0:-5]

            # vec_coef sorted from top of stack, everything else is sorted from bottom
            vec_index = len(pstack.layers) - lay_interest - 1

            vec_coef = np.concatenate((pstack.vec_coef_down[vec_index],pstack.vec_coef_up[vec_index]))
            # vec_coef_up = np.zeros(shape=(np.shape(pstack.vec_coef_down[vec_index])),dtype='complex128')
            # vec_coef = np.concatenate((pstack.vec_coef_down[vec_index],vec_coef_up))
            h_normed = float(meat.structure.height_nm)/float(meat.structure.period)
            wl_normed = pstack.layers[lay_interest].wl_norm()

            layer_name = 'Lay_' + zeros_int_str(lay_interest) + 'Stack_' + str(stack_num)

            field_eval.gmsh_plot_field_3d(wl_normed, h_normed, meat.num_BMs,
                meat.E_H_field, meat.n_msh_el, meat.n_msh_pts,
                nnodes, meat.type_el, meat.structure.nb_typ_el, meat.table_nod,
                meat.k_z, meat.sol1, vec_coef, meat.x_arr, gmsh_file_pos, layer_name)

            stack_num += 1
        except ValueError:
            print("fields_3d can only plot 3D fields within 2D_array "\
            "Nanostruct layers. \nPlease select a different lay_interest.\n")


def Bloch_fields_1d(stacks_list, lay_interest=None, save_pdf=False):
    """
    Plot Bloch mode fields along the x axis.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

        Keyword Args:
            lay_interest  (int): the index of the layer considered within \
                the stack. Must be a 1D_array NanoStruct layer. By default \
                routine finds all such layers.

            save_pdf (bool): If True, save figures as PDF files. False by default.
    """

    dir_name = "Bloch_fields"
    if save_pdf and not os.path.exists(dir_name):
        os.mkdir(dir_name)

    figs = []

    stack_num = 0
    for pstack in stacks_list:
        # by default find all 1D layers
        if lay_interest is None:
            lay_interest = []
            for l in range(len(pstack.layers)):
                if isinstance(pstack.layers[l],mode_calcs.Simmo):
                    if pstack.layers[l].structure.periodicity == '1D_array':
                        lay_interest.append(l)
        else: lay_interest = [lay_interest]

        num_lays = len(pstack.layers)
        for lay in lay_interest:
            try:
                meat = pstack.layers[lay]
                if not isinstance(meat,mode_calcs.Simmo):
                    raise ValueError
                if not meat.structure.periodicity == '1D_array':
                    raise ValueError

                eps = meat.n_effs**2

                struct = meat.structure
                struct_type_el = np.asarray(struct.type_el, dtype=int)
                boundary = []
                for i in range(len(struct_type_el) - 1):
                    if struct_type_el[i] != struct_type_el[i+1]:
                        boundary.append(struct.x_arr[2*(i+1)])

                fields = ['Re(E_x)', 'Im(E_x)', 'Re(E_y)', 'Im(E_y)', 'Re(E_z)', 'Im(E_z)']
                for BM in range(meat.num_BMs):
                    fig = plt.figure(figsize=(12, 12))
                    plot_sol_abs = np.zeros(struct.n_msh_pts)
                    plot_sol_eps_abs = np.zeros(struct.n_msh_pts)
                    # plot_sol_2_FT    = np.zeros(struct.n_msh_pts)
                    for i in range(len(fields)):
                        E = fields[i]
                        # sol_P2([Ex,Ey,Ez],P2_interpolation_points,nval,nel)
                        BM_sol = meat.sol1[i // 2, :, BM, :]
                        if E[0] == 'R': plot_sol = [np.real(BM_sol[0,0])]
                        else: plot_sol = [np.imag(BM_sol[0,0])]
                        plot_sol_abs[0] += np.real(np.sqrt((BM_sol[0,0])* np.conj(BM_sol[0,0])))
                        plot_sol_eps_abs[0] += np.real(eps[struct_type_el[0]-1] * np.sqrt((BM_sol[0,0])* np.conj(BM_sol[0,0])))
                        for x in range(struct.n_msh_el - 1):
                            if E[0] == 'R':
                                plot_sol.append(np.real(BM_sol[1,x]))
                                plot_sol.append(np.real(BM_sol[2,x] + BM_sol[0,x+1]) / 2)
                            else:
                                plot_sol.append(np.imag(BM_sol[1,x]))
                                plot_sol.append(np.imag(BM_sol[2,x] + BM_sol[0,x+1]) / 2)
                            plot_sol_abs[2*x+1] += np.real(np.sqrt((BM_sol[1,x])* np.conj(BM_sol[1,x])))
                            plot_sol_abs[2*x+2] += np.real(np.sqrt(((BM_sol[2,x] + BM_sol[0,x+1]) / 2)* np.conj((BM_sol[2,x] + BM_sol[0,x+1]) / 2)))
                            plot_sol_eps_abs[2*x+1] += np.real(eps[struct_type_el[x]-1] * np.sqrt((BM_sol[1,x])* np.conj(BM_sol[1,x])))
                            plot_sol_eps_abs[2*x+2] += np.real(eps[struct_type_el[x]-1] * np.sqrt(((BM_sol[2,x] + BM_sol[0,x+1]) / 2)* np.conj((BM_sol[2,x] + BM_sol[0,x+1]) / 2)))

                        if E[0] == 'R':
                            plot_sol.append(np.real(BM_sol[1,-1]))
                            plot_sol.append(np.real(BM_sol[2,-1]))
                        else:
                            plot_sol.append(np.imag(BM_sol[1,-1]))
                            plot_sol.append(np.imag(BM_sol[2,-1]))
                        plot_sol_abs[-2] += np.real(np.sqrt((BM_sol[1,x])* np.conj(BM_sol[1,x])))
                        plot_sol_abs[-1] += np.real(np.sqrt((BM_sol[2,x])* np.conj(BM_sol[2,x])))
                        plot_sol_eps_abs[-2] += np.real(eps[struct_type_el[-1]-1] * np.sqrt((BM_sol[1,x])* np.conj(BM_sol[1,x])))
                        plot_sol_eps_abs[-1] += np.real(eps[struct_type_el[-1]-1] * np.sqrt((BM_sol[2,x])* np.conj(BM_sol[2,x])))

                        ax1 = fig.add_subplot(6, 2, i+1)
                        ax1.plot(struct.x_arr, plot_sol)
                        ax1.set_ylabel(E)
                        if E == 'Re(E_x)' or E == 'Im(E_x)':
                            ax1.xaxis.set_label_position("top")
                            ax1.xaxis.tick_top()
                            ax1.xaxis.set_ticks_position('both')
                            ax1.set_xlabel('x (d)')
                        else:
                            ax1.set_xticklabels( () )
                        start, end = ax1.get_ylim()
                        ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                        for b in boundary:
                            ax1.plot([b,b],[start,end],'k')
                        if i%2 == 1:
                            ax1.yaxis.set_label_position("right")
                            ax1.yaxis.tick_right()
                            ax1.yaxis.set_ticks_position('both')

                        if E == 'Re(E_x)': plot_sol_2_FTx = plot_sol
                        if E == 'Im(E_x)': plot_sol_2_FT2x = plot_sol
                        if E == 'Re(E_y)': plot_sol_2_FT = plot_sol
                        if E == 'Im(E_y)': plot_sol_2_FT2 = plot_sol
                    ax1 = fig.add_subplot(6, 2, 7)
                    ax1.plot(struct.x_arr, plot_sol_abs)
                    ax1.set_xticklabels( () )
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    for b in boundary:
                        ax1.plot([b,b],[start,end],'k')
                    ax1.set_ylabel(r'|E|')
                    ax1 = fig.add_subplot(6, 2, 8)
                    ax1.plot(struct.x_arr, plot_sol_eps_abs)
                    ax1.set_xticklabels( () )
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    for b in boundary:
                        ax1.plot([b,b],[start,end],'k')
                    ax1.yaxis.set_label_position("right")
                    ax1.yaxis.tick_right()
                    ax1.yaxis.set_ticks_position('both')
                    ax1.set_ylabel(r'Re($\epsilon$) |E|')

                    ax1 = fig.add_subplot(6, 2, 9)
                    xshape = struct.x_arr.size
                    # plot_sol_2_FT = np.sin(3*np.linspace(0,2*np.pi,xshape))
                    FT = np.fft.fft(plot_sol_2_FTx)
                    freqs = np.fft.fftfreq(xshape,struct.x_arr[1])
                    freqs = np.fft.fftshift(freqs)
                    FT = np.fft.fftshift(FT)
                    ax1.plot(freqs, np.abs(FT),'ro', linewidth=3.0)
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    ax1.set_ylabel(r'FT Re(E$_x$)')
                    ax1.set_xlim((0,10))

                    ax1 = fig.add_subplot(6, 2, 10)
                    xshape = struct.x_arr.size
                    FT = np.fft.fft(plot_sol_2_FT2x)
                    freqs = np.fft.fftfreq(xshape,struct.x_arr[1])
                    freqs = np.fft.fftshift(freqs)
                    FT = np.fft.fftshift(FT)
                    ax1.plot(freqs, np.abs(FT),'ro', linewidth=3.0)
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    ax1.set_ylabel(r'FT Im(E$_x$)')
                    ax1.set_xlim((0,10))
                    ax1.yaxis.set_label_position("right")
                    ax1.yaxis.tick_right()
                    ax1.yaxis.set_ticks_position('both')

                    ax1 = fig.add_subplot(6, 2, 11)
                    xshape = struct.x_arr.size
                    FT = np.fft.fft(plot_sol_2_FT)
                    freqs = np.fft.fftfreq(xshape,struct.x_arr[1])
                    freqs = np.fft.fftshift(freqs)
                    FT = np.fft.fftshift(FT)
                    ax1.plot(freqs, np.abs(FT),'ro', linewidth=3.0)
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    ax1.set_ylabel(r'FT Re(E$_y$)')
                    ax1.set_xlim((0,10))
                    ax1.set_xlabel('Diffraction order')

                    ax1 = fig.add_subplot(6, 2, 12)
                    xshape = struct.x_arr.size
                    FT = np.fft.fft(plot_sol_2_FT2)
                    freqs = np.fft.fftfreq(xshape,struct.x_arr[1])
                    freqs = np.fft.fftshift(freqs)
                    FT = np.fft.fftshift(FT)
                    ax1.plot(freqs, np.abs(FT),'ro', linewidth=3.0)
                    start, end = ax1.get_ylim()
                    ax1.yaxis.set_ticks(np.linspace(start, end, 3))
                    ax1.set_ylabel(r'FT Im(E$_y$)')
                    ax1.set_xlim((0,10))
                    ax1.set_xlabel('Diffraction order')
                    ax1.yaxis.set_label_position("right")
                    ax1.yaxis.tick_right()
                    ax1.yaxis.set_ticks_position('both')

                    name_lay = "layer-%(lay)i-BM_%(BM)s"% \
                        {'lay' : lay, 'BM' : zeros_int_str(BM)}

                    plt.suptitle(r'BM %(BM)i, k$_z$ = %(re)7.3f + %(im)7.3fi (d)'% \
                        {'re' : np.real(meat.k_z[BM]), 'im' : np.imag(meat.k_z[BM]), 'BM' : BM})
                    if save_pdf:
                        plt.savefig(dir_name + '/' + name_lay, bbox_inches = 'tight')
                        plt.close(fig)
                    else:
                        figs.append(fig)

            except ValueError:
                print("fields_1d can only plot 1D fields of 1D_array "\
                "NanoStruct layers. \nPlease select different lay_interest.\n")

    return figs

###############################################################################

#### Fabry-Perot resonances ###################################################
def Fabry_Perot_res(stacks_list, freq_list, kx_list, f_0, k_0, lay_interest=1, save_pdf=False):
    """ Calculate the Fabry-Perot resonance condition for a resonances within a layer.

        This is equivalent to finding the slab waveguide modes of the layer.

        Args:
            stacks_list  (list): Stack objects containing data to plot.

            freq_list  (list): Frequencies included.

            kx_list  (list): In-plane wavenumbers included.

            f_0  (list): Frequency w.r.t. which axis is normalised.

            k_0  (list): In-plane wavenumber w.r.t. which axis is normalised.

        Keyword Args:
            lay_interest  (int): The index in stacks_list of the layer of\
            which F-P resonances are calculated.

            save_pdf (bool): If True, save plot as PDF. False by default.
    """

    n_freqs = len(freq_list)
    n_kxs = len(kx_list)
    # assuming all stacks have equal height
    height = stacks_list[-1].heights_nm()[lay_interest-1]
    period = stacks_list[-1].period
    num_BMss = stacks_list[-1].layers[lay_interest].num_BMs
    I_mat = np.matrix(np.eye(num_BMss),dtype='D')

    plot_mat = np.zeros(shape=(n_freqs,n_kxs),dtype='complex128')
    for i in range(n_kxs):
        kx_slice = stacks_list[i*n_freqs:(i+1)*n_freqs]
        j = 0
        for stack in kx_slice:
            P = stack.layers[lay_interest].prop_fwd(height/period)
            R21 = stack.layers[lay_interest].R21
            FP_term = np.linalg.det(I_mat - R21*P*R21*P)
            plot_mat[j, i] = FP_term
            j += 1
    image = np.log(np.abs(plot_mat))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cax = ax1.imshow(image, cmap=plt.cm.gray_r, interpolation='none')

    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    shape = np.shape(plot_mat)
    majorLocator = MultipleLocator(shape[1]-1)
    ax1.xaxis.set_major_locator(majorLocator)
    majorLocator = MultipleLocator(shape[0]-1)
    ax1.yaxis.set_major_locator(majorLocator)
    xlims = [kx_list[0]/k_0, kx_list[-1]/k_0]
    ax1.set_xticklabels(xlims)
    ylims = [freq_list[0]/f_0, freq_list[-1]/f_0]
    ax1.set_yticklabels(ylims)

    cbar = fig.colorbar(cax)
    cbar.set_label(r'$ln(|I - R_{21}PR_{21}P|)$',size=18)
    ax1.set_xlabel(r'$k_\perp/k_0$')
    ax1.set_ylabel(r'$f/f_0$')
    ax1.axis('image')
    if save_pdf:
        plt.savefig('Fabry-Perot_resonances', bbox_inches = 'tight')
        fig.clf()
    return fig
###############################################################################
