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

EMUstack supports two complete meshing pipelines for 2D nanostructured layers, both of which follow the same 6 stages (parameter injection, coordinate normalization to a dimensionless unit cell of width :math:`1` and height :math:`d_y = \text{period\_y} / \text{period}`, periodic outer boundary identification, second-order curved inclusion boundaries, subdomain material tagging, and mesh refinement verification):

1. **Native Built-in Netgen Meshing (Pure Python):**
   Unit cells are built directly in memory using Netgen's 2D constructive solid geometry (``netgen.geom2d.SplineGeometry``) with zero external binaries or temporary disk files. Within a ``.py`` template, you can either:

   * **Use the high-level helper** :func:`meshing.unit_cell` **(Route B1)** for unit cells composed of standard inclusion primitives (``circle``, ``ellipse``, ``square``, ``rectangle``, ``ring``, ``dimer``), such as ``ellipse_msh_template.py`` or ``chiral_qbic_gorkunov.py``, where :func:`meshing.unit_cell` automatically constructs the periodic box, 12-segment quadratic rational Bézier spline arcs (``spline3``), domain tags, and second-order curving (``ngm.Curve(2)``); or
   * **Construct a** ``netgen.geom2d.SplineGeometry`` **directly (Route B2)** when you need a fancier custom resonator shape (such as an L-antenna, split ring, crescent, or arbitrary polygon/spline).

2. **Gmsh Template Import:**
   Legacy or custom geometries defined in Gmsh ``.geo`` scripts (e.g., ``ellipse_msh_template.geo``) have their ``key = 0;`` placeholders substituted from ``geo_params``, are meshed in MSH 2.2 format, and have opposite outer boundary vertices paired via 1-based ``AddPointIdentification`` for Bloch periodicity.

For a complete, step-by-step walkthrough of both pipelines—including side-by-side line-by-line explanations of ``ellipse_msh_template.geo``, ``ellipse_msh_template.py``, and ``chiral_qbic_gorkunov.py``—see the :doc:`meshing`.

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