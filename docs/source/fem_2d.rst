2D FEM Mode Solver
====================

2D Mesh
--------

2D FEM meshes can be created natively via NGSolve/Netgen or using Gmsh.
Templates for Gmsh inclusion shapes are stored in ``backend/data/meshes/``. For an up to date list of templates see the 'inc_shape' entry in the NanoStruct docstring. 

An advantage of using the FEM to calculate the modes of layers is that there is absolutely no constraints on the content of the unit cell. If you wish to create a different structure this can be done using Netgen native builders or Gmsh.

Note that the area of the unit cell must always be unity! This has been assumed throughout the theoretical derivations.

Note that the Physical Surface numbers set in the .geo file must map onto the following array in EMUstack ::

    self.n_effs = np.array([st.background.n(wl), st.inclusion_a.n(wl),
                            st.inclusion_b.n(wl), st.inclusion_c.n(wl),
                            st.inclusion_d.n(wl), st.inclusion_e.n(wl)])

So, for example, Physical Surface(1) is st.background and Physical Surface(2) is inclusion_a.


FEM Solver
----------

The 2D mode solver uses NGSolve to assemble Maxwell curl-curl / grad-div generalized eigenvalue pencils on periodic Nedelec (HCurl) and Lagrange (H1) finite element spaces. Convergence failures raise ``FEMConvergenceError`` rather than aborting.