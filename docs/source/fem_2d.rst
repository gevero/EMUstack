2D Finite Element Mode Solver
==============================

Overview
--------

EMUstack computes the electromagnetic Bloch modes of 2D periodic nanostructures using the high-performance finite element library `NGSolve <https://ngsolve.org/>`_. 

The Maxwell eigenvalue equation:

.. math::

    \mathbf{K} \mathbf{u} = \beta^2 \mathbf{M} \mathbf{u}

is discretized on a quasi-periodic product space using mixed vector-scalar finite elements:

* **Nédélec edge elements** (:math:`H(\text{curl})`) for the transverse electric field :math:`\mathbf{E}_t`, which naturally enforce continuity of tangential electric fields across material boundaries while allowing normal field jumps.
* **Lagrange nodal elements** (:math:`H^1`) for the longitudinal electric field :math:`E_z`.

This mixed formulation completely avoids non-physical spurious modes without artificial penalty terms. The eigenvalue problem is solved via shift-invert Arnoldi iteration using NGSolve's built-in sparse solvers. If the requested number of modes cannot be resolved to numerical tolerance, a descriptive ``FEMConvergenceError`` is raised.

Meshing Workflows
-----------------

EMUstack supports two meshing workflows for 2D nanostructured layers:

1. **Native Built-in Netgen Meshing (Pure Python):**
   Unit cells are built directly in memory using Netgen's 2D constructive solid geometry (CSG) tools or Python geometry templates (e.g., ``ellipse_msh_template.py``, ``chiral_qbic_template.py``). This workflow requires no external mesh generation binaries and supports curved second-order elements with exact boundary representation.

2. **Gmsh Template Import:**
   Legacy or complex geometries defined in Gmsh ``.geo`` scripts are meshed and imported via Gmsh MSH 2.2 format. Periodic boundary identifications are automatically applied to matching unit cell boundary edges.

Material Surface Mapping
------------------------

In 2D unit cells, domain materials are mapped to physical surface tags:

.. code-block:: python

    self.n_effs = np.array([
        st.background.n(wl),
        st.inclusion_a.n(wl),
        st.inclusion_b.n(wl),
        st.inclusion_c.n(wl),
        st.inclusion_d.n(wl),
        st.inclusion_e.n(wl)
    ])

Domain tag 1 corresponds to ``background``, domain tag 2 corresponds to ``inclusion_a``, and higher tags correspond to subsequent inclusions.