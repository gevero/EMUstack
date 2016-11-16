import numpy as np
import objects
import mode_calcs
from scipy.interpolate import interp2d, RectBivariateSpline

def fields_1d_vertical_slice(pstack, lay_interest=1,
                             x_min=0.0,x_max=1.0,
                             z_min=0.0,z_max=1.0,
                             n_step=51,
                             semi_inf_height=1.0, gradient=None, no_incoming=False):
    """
    Plot fields in the x-y plane at chosen values of z, where z is \
    calculated from the bottom of chosen layer.

        Args:
            'stack' = actual EMUstack stack object (EMUstack object)

        Keyword Args:
            n_step  (int): sampling density

            'lay_interest' = number of the layer of interest (structured)

            'x_min,x_max,y_min,y_max,y_min,z_max' = box boundaries normalized to
                                                    lattice x-pitch

            gradient  (float): further slices calculated with given gradient \
                and -gradient. It is entitled 'specified_diagonal_slice'.\
                These slices are only calculated for ThinFilm layers.

            no_incoming  (bool): if True, plots fields in superstrate in the \
                absence of the incident driving field (i.e. only showing \
                upward propagating scattered field).
    """

    from fortran import EMUstack

    # defining horizontal and vertical dimensions from n_step
    # these are the resolution of my final output
    nu_pts_hori = n_step
    nu_pts_vert = n_step

    # always make odd
    if nu_pts_hori % 2 == 0:
        nu_pts_hori += 1
    if nu_pts_vert % 2 == 0:
        nu_pts_vert += 1

    # getting pstack and calculation parameters
    num_lays = len(pstack.layers)
    period = pstack.layers[0].structure.period
    pw = pstack.layers[0].max_order_PWs
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

    # removed all the options for the selection of slice type
    # and keep the only usefule kind of slice type, i.e. 'xz'
    # incidentally I shall remove the loop over slice types
    sli = 'xz'

    # selecting the right layer and defining meshing points
    lay = lay_interest
    layer = pstack.layers[lay]
    struct  = layer.structure

    # redefining vertical resolution and spacings
    if isinstance(layer,mode_calcs.Simmo):
        eps = layer.n_effs**2
    else:
        eps = layer.n()**2

    if lay == 0:
        z_range = np.linspace(h_list[lay],0.0,nu_pts_vert)
    else:
        z_range = np.linspace(0.0,ind_h_list[lay],nu_pts_vert)
    # vec_coef sorted from top, everything else sorted from bottom
    vec_index = num_lays - lay - 1

    vec_coef_fem = np.concatenate((pstack.vec_coef_down[vec_index],
        pstack.vec_coef_up[vec_index]))

    E_fields = ['Re(E_x)', 'Im(E_x)', 'Re(E_y)', 'Im(E_y)', 'Re(E_z)', 'Im(E_z)', 'Re(E)',  'eps_abs(E)']
    m_E = np.zeros((8,struct.n_msh_pts,nu_pts_vert))
    m_E_out = np.zeros((8,n_step,n_step))
    for i_E,E in enumerate(E_fields):

        # Find maximum field value to normalise plots.
        for max_E_field in [0,1]:

            # If 1D_array plot fields as follows.
            try:
                # works only for 1d layers
                if not layer.structure.periodicity == '1D_array':
                    raise ValueError
                struct  = layer.structure
                E_slice = np.zeros((struct.n_msh_pts,nu_pts_vert), dtype = 'complex128')

                boundary = []
                for i in range(len(struct.type_el) - 1):
                    if struct.type_el[i] != struct.type_el[i+1]:
                        boundary.append(struct.x_arr[2*(i+1)])

                # trying to understand in a very clumsy way which field component
                # I am dealing with
                if E[-3:] != '(E)':  # I am NOT dealing with 'Re(E)' or 'eps_abs(E)'
                    if E[5] == 'x':
                        comp = 0
                    if E[5] == 'y':
                        comp = 1
                    if E[5] == 'z':
                        comp = 2

                    # piling up the modes to compute the field
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
                            # print('n_elements',struct.n_msh_el - 1)
                            for x in range(struct.n_msh_el - 1):
                                E_slice[2*x+1,h] += BM_sol[1,x] * coef_tot
                                # E_slice[2*x+2,h] += (BM_sol[2,x] + BM_sol[0,x+1] / 2.) * coef_tot
                                E_slice[2*x+2,h] += (BM_sol[2,x]) * coef_tot
                            E_slice[2*x+3,h] += BM_sol[1,-1] * coef_tot
                            E_slice[2*x+4,h] += BM_sol[2,-1] * coef_tot
                            # print('x,2*x+3,2*x+4',x,2*x+3,2*x+4)

                    # if real field and first pass
                    if max_E_field == 0 and E[0] == 'R':
                        if E[5] == 'x':
                            E_fields_tot = E_slice*np.conj(E_slice)
                            epsE_fields_tot = np.zeros(np.shape(E_slice))
                        elif E[5] == 'y' or E[5] == 'z':
                            E_fields_tot += E_slice*np.conj(E_slice)

                # computing field module multiplied by epsilon
                elif E == 'eps_abs(E)':

                    # first pass: result storage in epsE_fields_tot[lay]
                    if max_E_field == 0:
                        type_el = np.vstack((struct.type_el,struct.type_el)).reshape((-1,),order='F')
                        type_el = np.append(type_el,type_el[-1])
                        type_el[type_el == 1] = np.real(eps[0])
                        type_el[type_el == 2] = np.real(eps[1])
                        type_el = np.diag(type_el)
                        epsE_fields_tot = np.dot(type_el,(E_fields_tot))
                    else:
                        E_slice = epsE_fields_tot

                # if 'Re(E)' or 'eps_abs(E)' and second pass
                elif E[-3:] == '(E)' and max_E_field == 1:
                    E_slice = np.sqrt(E_fields_tot)

                # if Re or eps
                if E[0] == 'R' or E[0] == 'e':
                    E_slice = np.real(E_slice)
                elif E[0] == 'I':  # instead if imag
                    E_slice = np.imag(E_slice)

                # field storage
                m_E[i_E,:,:] = np.real(E_slice)

            except ValueError as e:
                print(e)
                print("fields_1d_vertical_slice plots fields only in 1D-arrays."\
                      "\nPlease select a different lay_interest.\n")


    # raw coords as input for the interpolator
    v_x_raw = np.linspace(0.0,1.0,struct.n_msh_pts)
    v_z_raw = np.linspace(0.0,1.0,nu_pts_vert)

    # xyz simple coordinates
    v_x = np.linspace(x_min,x_max,n_step)
    v_z = np.linspace(z_min,z_max,n_step)

    # loop to create the appropriate interpolators
    for i_E,m_E_slice in enumerate(m_E):
        m_swap = m_E[i_E,:,:]
        # f_E = interp2d(v_x_raw,v_z_raw,m_swap)
        f_E = RectBivariateSpline(v_x_raw,v_z_raw,m_swap)
        m_E_out[i_E,:,:] = f_E(v_x,v_z)

    return v_x, v_z, m_E_out, m_E_out**2
