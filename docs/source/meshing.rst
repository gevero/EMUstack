2D Geometry & Meshing Guide (Gmsh & Netgen)
===========================================

EMUstack solves for the electromagnetic Bloch modes of 2D periodically structured layers (`NanoStruct('2D_array', ...)`) by discretizing the 2D unit cell into high-order finite elements in `NGSolve <https://ngsolve.org/>`_.

To support both legacy simulation scripts and pure-Python workflows, EMUstack provides **two complete 2D meshing pipelines** that converge into the same ``ngsolve.Mesh`` representation:

1. **The Gmsh Pipeline (Classic Workflow):**
   Reads a parameterized Gmsh ``.geo`` script (or a pre-generated ``.mail`` / ``.msh`` file), substitutes geometry parameters, invokes the external ``gmsh`` binary to triangulate the unit cell, and imports the resulting mesh into Netgen/NGSolve with periodic vertex identifications.
2. **The Netgen Pipeline (Native Pure-Python Workflow):**
   Executes a pure-Python geometry builder (either built-in parametric shapes or a custom ``*_template.py`` module such as ``ellipse_msh_template.py`` or ``chiral_qbic_gorkunov.py``) to construct a ``netgen.geom2d.SplineGeometry`` directly in memory with exact quadratic rational Bézier curves and periodic boundary identifications—requiring **zero external binaries and zero temporary disk files**.

.. note::
   **How EMUstack Selects the Pipeline:**

   * Passing a Python template ``geo_file='... .py'`` **always** runs the native Netgen pipeline.
   * For standard built-in shapes (``inc_shape='circle'``, ``'ellipse'``, ``'square'``, ``'rectangle'``, ``'ring'``, ``'dimer'``), the pipeline is controlled either globally via ``objects.default_mesher = 'gmsh'`` (default) or ``'netgen'``, or per layer via ``NanoStruct(..., mesher='netgen')``.
   * Passing a Gmsh script ``geo_file='... .geo'`` runs the Gmsh pipeline.

---

Part I: General Presentation & the 6 Core Concepts
--------------------------------------------------

Regardless of whether you use the **Gmsh pipeline** or the **Netgen pipeline**, building a valid 2D periodic unit-cell mesh in EMUstack follows the exact same **six conceptual stages**. Understanding these six concepts makes it straightforward to read, modify, or author geometry templates in either pipeline.

.. list-table:: Summary of the 6 Core Meshing Stages Across Both Pipelines
   :header-rows: 1
   :widths: 22 39 39

   * - Stage / Concept
     - Gmsh Pipeline (``.geo``)
     - Netgen Pipeline (``.py``)
   * - **1. Parameter Injection**
     - Textual substitution of placeholders (``key = 0;`` :math:`\to` ``key = val;``) in the ``.geo`` script via :func:`meshing.mesh_from_geo`.
     - Python function signature inspection of ``build_unit_cell(...)`` and keyword-argument passing via :func:`meshing.mesh_from_py`.
   * - **2. Coordinate Normalization**
     - Physical lengths in nm are divided by ``d_in_nm`` (:math:`a`) to span :math:`[0, 1] \times [-d_y, 0]`, centered at :math:`(0.5, -0.5\,d_y)`.
     - Physical lengths in nm are divided by ``period`` (:math:`a`) to span :math:`[0, 1] \times [0, d_y]`, centered at :math:`(0.5, 0.5\,d_y)`.
   * - **3. Periodic Outer Boundaries**
     - Outer edges tagged as ``Physical Line(1..4)``; opposite boundary vertices are paired geometrically upon import via ``AddPointIdentification``.
     - Left and bottom edges are defined first; right and top edges are appended with ``copy=i_l`` and ``copy=i_b`` to clone 1D boundary nodes.
   * - **4. Curved Inclusions**
     - ``Circle`` / ``Ellipsis`` arcs and ``Rotate`` commands in ``.geo``, meshed as 6-node quadratic triangles or curved on import.
     - Declarative ``inclusions`` dicts converted into 12 exact rational Bézier ``spline3`` arcs with off-curve tangent control points, curved via ``ngm.Curve(2)``.
   * - **5. Subdomain Material Tags**
     - ``Plane Surface`` + ``Physical Surface(1)`` (background) and ``Physical Surface(2..6)`` (inclusions).
     - Counter-clockwise curve orientation (``leftdomain=tag, rightdomain=1``) + ``geo.SetMaterial(1, "bg")`` and ``geo.SetMaterial(tag, "inc{tag}")``.
   * - **6. Refinement & Verification**
     - Point characteristic lengths ``lc`` (background), ``lc2 = lc/factor`` (inclusion boundary), and ``lc3`` (inclusion center).
     - ``meshing.Refine(bg=lc, inc=lc/lc2)`` + ``SetDomainMaxH``, followed by automatic checks on periodic pairs, curve order, and cell area.

Concept 1: Parameter Injection (``geo_file`` & ``geo_params``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you instantiate ``objects.NanoStruct('2D_array', period, ..., geo_file=..., geo_params={...})``, EMUstack locates ``geo_file`` (searching first in your current working directory and then in ``EMUstack/backend/data/meshes/``) and merges the layer's basic parameters (``period``, ``period_y``, ``diameter1``, ``lc``, ``lc2``, ``lc3``) with your custom ``geo_params`` dictionary:

* **In the Gmsh pipeline**, :func:`meshing.mesh_from_geo` reads the ``.geo`` file as text and replaces placeholder assignments such as ``d_in_nm = 0;``, ``a1 = 0;``, ``angle = 0;``, ``lc = 0;``, and ``lc2 = lc/1;`` with the numerical values from ``geo_params``. If a key in ``geo_params`` does not have a matching placeholder in the ``.geo`` file, a ``KeyError`` is raised immediately.
* **In the Netgen pipeline**, :func:`meshing.mesh_from_py` dynamically imports the ``.py`` module, inspects the Python signature of ``build_unit_cell(...)`` using ``inspect.signature``, verifies that every key in ``geo_params`` is a valid parameter of ``build_unit_cell`` (raising a ``TypeError`` on typos), and calls ``build_unit_cell(**call_params)``.

Concept 2: Coordinate Normalization to a Dimensionless Unit Cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All 2D meshes in EMUstack are constructed in **dimensionless coordinates** normalized by the unit-cell pitch along :math:`x` (``period`` :math:`= a`):

.. math::

   \tilde{x} = \frac{x}{a} \in [0, 1], \qquad
   \tilde{y} = \frac{y}{a} \in [0, d_y], \qquad
   d_y = \frac{\text{period\_y}}{\text{period}}.

Every physical dimension in nanometers—such as a cylinder diameter :math:`D`, ellipse semi-axes :math:`r_x, r_y`, resonator width :math:`w`, length :math:`l`, or inter-resonator gap :math:`g`—is divided by ``period`` before constructing the geometry.

**Why normalize?** In the 2D finite-element solver (:mod:`fem_2d`), the free-space wavenumber :math:`k_0 = 2\pi/\lambda` is scaled by ``period``. Because the mesh itself is dimensionless, **the unit cell is meshed only once** when ``NanoStruct`` is instantiated and is reused across all wavelengths in a spectral sweep without re-meshing.

Concept 3: Outer Unit-Cell Boundaries & Periodic Node Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enforce Bloch quasi-periodicity across the unit cell,

.. math::

   \mathbf{E}(x + a, y) = e^{-i k_x a}\,\mathbf{E}(x, y), \qquad
   \mathbf{E}(x, y + a d_y) = e^{-i k_y a d_y}\,\mathbf{E}(x, y),

NGSolve's ``Periodic(HCurl * H1, phase=...)`` space requires that opposite outer boundaries (**Left** :math:`\leftrightarrow` **Right** and **Bottom** :math:`\leftrightarrow` **Top**) have **identical 1D mesh discretizations** and explicit **periodic vertex identifications**:

* **In the Gmsh pipeline**, the four outer boundary edges are tagged as ``Physical Line(1..4)``. When :func:`meshing.import_gmsh_mesh` ingests the ``.msh`` file, it locates all vertices on :math:`x = x_{\min}` and :math:`x = x_{\max}` (and :math:`y = y_{\min}`, :math:`y = y_{\max}`), verifies that opposite faces have the exact same node count, sorts them by coordinate, and pairs them via 1-based ``ngm.AddPointIdentification(p1, p2, id_nr, IdentificationType.PERIODIC)``.
* **In the Netgen pipeline**, :func:`meshing.unit_cell` appends the **left** edge (:math:`\tilde{x}=0`) as primary segment ``i_l`` and then appends the **right** edge (:math:`\tilde{x}=1`) in the **same bottom-to-top direction** with ``copy=i_l``. Likewise, it appends the **bottom** edge (:math:`\tilde{y}=0`) as primary segment ``i_b`` and the **top** edge (:math:`\tilde{y}=d_y`) in the **same left-to-right direction** with ``copy=i_b``. The ``copy=`` argument instructs Netgen to clone the 1D mesh nodes from the primary edge onto its partner edge and automatically register the periodic vertex identifications.

Concept 4: Interior Inclusions & High-Order Curved Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximating curved dielectric or metallic resonators (such as circular nanopillars or tilted elliptical bars in a high-:math:`Q` quasi-BIC metasurface) with straight-edged polygons introduces artificial corner singularities and staircase errors. Both pipelines represent curved boundaries using **second-order isoparametric curved elements**:

* **In the Gmsh pipeline**, curved boundaries are defined using Gmsh's built-in ``Circle`` or ``Ellipsis`` arc primitives (optionally rotated via ``Rotate``) and meshed either as 6-node quadratic triangles (``gmsh -2 -order 2``) or curved to second order upon Netgen import (``ngm.Curve(2)``).
* **In the Netgen pipeline**, :func:`meshing.unit_cell` divides each circle or rotated ellipse into :math:`N_{\text{seg}} = 12` angular sectors :math:`[a_0, a_1]` (:math:`\Delta\theta = 30^\circ`). For each sector, :func:`meshing.arc_control_points` computes the start point :math:`\mathbf{P}_0`, the end point :math:`\mathbf{P}_1`, and the exact **off-curve tangent intersection control point** :math:`\mathbf{P}_{\text{ctrl}}` of a quadratic rational Bézier arc:

  .. math::

     \mathbf{P}_{\text{ctrl}} = \mathbf{c} +
     \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}
     \begin{pmatrix} r_x \sec(\Delta\theta / 2) \cos\theta_{\text{mid}} \\ r_y \sec(\Delta\theta / 2) \sin\theta_{\text{mid}} \end{pmatrix},
     \qquad \theta_{\text{mid}} = \frac{a_0 + a_1}{2},

  and appends ``["spline3", pt0, pt_ctrl, pt1]`` to the ``SplineGeometry``. Calling ``ngm.Curve(2)`` after mesh generation curves all triangle edges on the resonator boundary onto the exact analytical ellipse.

Concept 5: Subdomain Material Tagging (``bg``, ``inc2``, ``inc3``, ...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During the finite-element assembly (:mod:`fem_2d`), EMUstack maps each mesh subdomain to the refractive index of the corresponding material passed to ``objects.NanoStruct``:

.. list-table:: Subdomain Material Mapping in 2D Unit Cells
   :header-rows: 1
   :widths: 20 25 25 30

   * - Integer Tag
     - Netgen Material Name
     - Gmsh Physical Surface
     - ``NanoStruct`` Argument
   * - ``1``
     - ``"bg"``
     - ``Physical Surface(1)``
     - ``background``
   * - ``2``
     - ``"inc2"``
     - ``Physical Surface(2)``
     - ``inclusion_a``
   * - ``3``
     - ``"inc3"``
     - ``Physical Surface(3)``
     - ``inclusion_b``
   * - ``4``
     - ``"inc4"``
     - ``Physical Surface(4)``
     - ``inclusion_c``
   * - ``5``
     - ``"inc5"``
     - ``Physical Surface(5)``
     - ``inclusion_d``
   * - ``6``
     - ``"inc6"``
     - ``Physical Surface(6)``
     - ``inclusion_e``

* **In the Gmsh pipeline**, you explicitly group closed ``Line Loop`` definitions into ``Plane Surface`` entities (subtracting interior inclusion loops from the outer unit-cell loop) and assign ``Physical Surface(1)`` to the background and ``Physical Surface(2)`` to the inclusion.
* **In the Netgen pipeline**, 2D regions are defined topologically by **directed boundary curves** via ``leftdomain`` and ``rightdomain``: as you walk along a segment from its start point to its end point, ``leftdomain`` is the region ID to your left and ``rightdomain`` is the region ID to your right (where ``0`` denotes the exterior of the computational domain). Because inclusion splines are traversed **counter-clockwise**, the interior of the inclusion lies on the left (``leftdomain=tag``) and the surrounding background lies on the right (``rightdomain=1``).

Concept 6: Mesh Refinement Controls (``lc``, ``lc2``, ``lc3``) & Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh resolution is controlled in dimensionless unit-cell units (fractions of ``period``):

* ``lc`` (**background mesh size**): Maximum triangle edge length in the background medium (domain ``1``). For example, ``lc = 0.08`` produces roughly :math:`1 / 0.08 \approx 12\text{--}13` elements across the unit cell.
* ``lc2`` (**inclusion refinement factor**): Divides ``lc`` to set the maximum element size inside and along the boundary of inclusions:

  .. math::

     h_{\text{inc}} = \frac{\text{lc}}{\text{lc2}} \qquad (\text{when } \text{lc2} > 1).

  Higher-index dielectric or plasmonic inclusions have shorter internal wavelengths and sharper local fields, so setting ``lc2 = 2.0`` to ``5.0`` refines the resonator interior and boundary without over-refining the background.
* ``lc3`` (**center refinement factor**, Gmsh pipeline): Controls element size at a control point placed at the center of an inclusion (``lc / lc3``).

Once the mesh is generated in the Netgen pipeline, :func:`meshing.unit_cell` automatically runs **four sanity assertions**:

1. ``ngm.GetNrIdentifications() == 4`` (periodic boundary tables exist).
2. ``ngm.GetCurveOrder() >= 2`` (second-order curved elements are active).
3. ``len(mesh.GetPeriodicNodePairs(VERTEX)) > 0`` (opposite boundary vertices are paired).
4. :math:`\left|\int_{\Omega} 1\,dA - 1 \times d_y\right| < 10^{-11}` (exact unit-cell area conservation).

---

Part II: Step-by-Step Practical Examples
----------------------------------------

To see how the **6 Core Concepts** from Part I translate directly into code, we now walk step by step through two real templates included in ``EMUstack/backend/data/meshes/``:

* **Example A (Gmsh Pipeline):** ``ellipse_msh_template.geo`` — a rotated elliptical nanohole/nanopillar unit cell in Gmsh.
* **Example B (Netgen Pipeline):** ``ellipse_msh_template.py`` and ``chiral_qbic_gorkunov.py`` — the native pure-Python Netgen equivalents for a single rotated ellipse and for the 3-layer tilted elliptical-bar dimer used in the flagship **Chiral quasi-BIC (qBIC)** tutorial, followed by the internal steps of :func:`meshing.unit_cell`.

In both walkthroughs below, **Steps 1 through 6 correspond 1-to-1 to Concepts 1 through 6 from Part I**.

---

Example A: Step-by-Step Gmsh Template (``ellipse_msh_template.geo``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the step-by-step breakdown of ``EMUstack/backend/data/meshes/ellipse_msh_template.geo`` (for the general rotated case ``angle != 0``).

Step 1 (Concept 1 — Parameter Placeholders)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the top of ``ellipse_msh_template.geo``, every parameter that can be overridden by ``NanoStruct`` or ``geo_params`` is initialized with a ``0`` (or ``lc/1``) placeholder:

.. code-block:: c

   // Force Gmsh to output legacy MSH 2.2 format required by Netgen ReadGmsh
   Mesh.MshFileVersion = 2.2;

   d = 1;            // Normalized unit-cell width along x
   d_in_nm = 0;      // Replaced by period (in nm)
   dy_in_nm = 0;     // Replaced by period_y (in nm)
   a1 = 0;           // Replaced by major axis diameter (in nm)
   ellipticity = 0;  // Replaced by (rx - ry) / rx
   angle = 0;        // Replaced by rotation angle (in radians)

   lc = 0;           // Replaced by background element size
   lc2 = lc/1;       // Replaced by lc / lc2 (on ellipse boundary)
   lc3 = lc/1;       // Replaced by lc / lc3 (at ellipse center)

When you pass ``geo_params={'a1': 300, 'ellipticity': 0.5, 'angle': np.pi/3, 'lc': 0.08, 'lc2': 4.0}``, EMUstack replaces ``a1 = 0;`` with ``a1 = 300.0;``, ``lc2 = lc/1;`` with ``lc2 = lc/4.0;``, and so on.

Step 2 (Concept 2 — Coordinate Normalization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, the script converts the physical dimensions ``d_in_nm``, ``dy_in_nm``, and ``a1`` (in nanometers) into normalized unit-cell coordinates where the width along :math:`x` is ``d = 1``:

.. code-block:: c

   dy = dy_in_nm / d_in_nm;        // Normalized unit-cell height along y
   radius1 = (a1 / (2 * d_in_nm)) * d;  // Normalized semi-major axis rx
   hy = dy;
   hx = 0.0;

Step 3 (Concept 3 — Outer Unit-Cell Box & Physical Boundary Lines)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four corners of the unit cell :math:`[0, 1] \times [-d_y, 0]` are created with characteristic mesh size ``lc`` and connected into outer boundary lines ``Line(1..4)``, closed into ``Line Loop(9)``, and exported as ``Physical Line(1..4)``:

.. code-block:: c

   Point(1) = {0,  0,   0, lc};
   Point(2) = {0, -hy,  0, lc};
   Point(3) = {d, -hy,  0, lc};
   Point(4) = {d,  0,   0, lc};

   Line(1) = {1, 2};   // Left boundary   (x = 0)
   Line(2) = {2, 3};   // Bottom boundary (y = -dy)
   Line(3) = {3, 4};   // Right boundary  (x = 1)
   Line(4) = {4, 1};   // Top boundary    (y = 0)

   Line Loop(9) = {1, 2, 3, 4};

   Physical Line(1) = {1};
   Physical Line(2) = {2};
   Physical Line(3) = {3};
   Physical Line(4) = {4};

When the generated mesh is imported into NGSolve by :func:`meshing.import_gmsh_mesh`, the vertices on ``Line(1)`` and ``Line(3)`` (and on ``Line(2)`` and ``Line(4)``) are matched and paired via ``AddPointIdentification`` for Bloch periodicity.

Step 4 (Concept 4 — Rotated Elliptical Inclusion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ellipse center ``Point(5)`` is placed at :math:`(0.5, -0.5\,d_y)` with mesh size ``lc3``, and the four cardinal vertices ``Point(6..9)`` of the unrotated ellipse (with semi-axes :math:`r_x = \text{radius1}` and :math:`r_y = \text{radius1}\,(1 - \text{ellipticity})`) are created with boundary mesh size ``lc2``. Gmsh's ``Rotate`` command then rotates ``Point{6,7,8,9}`` around the center by ``angle`` (around the :math:`z`-axis ``{0, 0, 1}``), and four ``Ellipsis`` arcs join them into ``Line Loop(10)``:

.. code-block:: c

   // Center of the ellipse
   Point(5) = {d/2, -hy/2, 0, lc3};

   // Unrotated cardinal endpoints on the ellipse perimeter (mesh size lc2)
   Point(6) = {d/2,           -hy/2 + (radius1 - ellipticity*radius1), 0, lc2};
   Point(7) = {d/2 - radius1, -hy/2,                                   0, lc2};
   Point(8) = {d/2,           -hy/2 - (radius1 - ellipticity*radius1), 0, lc2};
   Point(9) = {d/2 + radius1, -hy/2,                                   0, lc2};

   // Rotate cardinal points around (0.5*d, -0.5*dy, 0) by angle (in radians)
   Rotate{{0, 0, 1}, {0.5 * d, -0.5 * dy, 0}, angle} {Point{6, 7, 8, 9};}

   // Connect the 4 elliptical quadrants: Ellipsis(id) = {start, center, maj_pt, end}
   Ellipsis(5) = {9, 5, 6, 6};
   Ellipsis(6) = {6, 5, 7, 7};
   Ellipsis(7) = {7, 5, 8, 8};
   Ellipsis(8) = {8, 5, 9, 9};

   Line Loop(10) = {5, 6, 7, 8};

Step 5 (Concept 5 — Domain Surfaces & Material Tags)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The background region is defined as ``Plane Surface(11) = {9, 10}`` (the outer unit-cell loop ``9`` minus the interior ellipse hole ``10``) and tagged as ``Physical Surface(1)`` (mapped to ``background``). The interior of the ellipse is defined as ``Plane Surface(12) = {10}`` and tagged as ``Physical Surface(2)`` (mapped to ``inclusion_a``):

.. code-block:: c

   Plane Surface(11) = {9, 10};   // Background region (unit cell minus ellipse)
   Plane Surface(12) = {10};      // Elliptical inclusion interior

   Physical Surface(1) = {11};    // Tag 1 -> background material
   Physical Surface(2) = {12};    // Tag 2 -> inclusion_a material

Step 6 (Concept 6 — Interior Mesh Control & Calling from ``NanoStruct``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, ``Point{5} In Surface{12};`` embeds the center point inside the ellipse surface so Gmsh respects ``lc3`` at the center of the inclusion:

.. code-block:: c

   Point{5} In Surface{12};

To use this Gmsh template in a simulation script:

.. code-block:: python

   import numpy as np
   from EMUstack.backend import objects, materials

   nh_layer = objects.NanoStruct(
       '2D_array',
       period=400.0,
       diameter1=300.0,
       ellipticity=0.5,
       inc_shape='custom',
       geo_file='ellipse_msh_template.geo',
       geo_params={
           'angle': np.pi / 3,
           'lc': 0.08,
           'lc2': 4.0,
           'lc3': 2.0,
       },
       background=materials.Au,
       inclusion_a=materials.Air,
       height_nm=50.0,
       mesher='gmsh',
   )

---

Example B: Step-by-Step Netgen Templates (``ellipse_msh_template.py`` & ``chiral_qbic_gorkunov.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let us walk through the **native Netgen pipeline**, using both the single rotated ellipse template (``ellipse_msh_template.py``) and the two-ellipse chiral quasi-BIC dimer template (``chiral_qbic_gorkunov.py``) from the flagship tutorial ``Chiral_qBIC_Optical_Chirality.ipynb``.

Notice how **Steps 1 through 6** below directly mirror the Gmsh steps above, while eliminating manual point/line IDs and external binary calls.

Step 1 (Concept 1 — Python Function Signature & Parameter Validation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Netgen geometry template is a standard Python file defining ``build_unit_cell(...)``. Here is the signature from ``chiral_qbic_gorkunov.py`` (which models two elliptical dielectric bars of length ``l``, width ``w``, separated by ``gap``, and rotated by :math:`\pm \theta/2`):

.. code-block:: python

   from typing import Optional
   import numpy as np
   import ngsolve as ngs
   from EMUstack.backend import meshing

   def build_unit_cell(
       period: float = 480.0,
       period_y: Optional[float] = None,
       theta: float = 3.5,          # Tilt angle in degrees
       d: float = 10.0,             # Vertical offset in nm
       l: float = 220.0,            # Bar length in nm (major axis 2*ry)
       w: float = 70.0,             # Bar width in nm  (minor axis 2*rx)
       gap: float = 15.0,           # Edge-to-edge gap between bars in nm
       include_bar1: bool = True,   # Toggle left bar (for 3-layer z-slicing)
       include_bar2: bool = True,   # Toggle right bar (for 3-layer z-slicing)
       rot1: Optional[float] = None,
       rot2: Optional[float] = None,
       lc: float = 0.08,            # Background mesh size (fraction of period)
       lc2: float = 2.0,            # Inclusion refinement factor (inc = lc / lc2)
       lc3: float = 1.0,
       curve_order: int = 2,        # Second-order curved elements
       **kwargs
   ) -> ngs.Mesh:

Similarly, in ``ellipse_msh_template.py``, the signature accepts ``period``, ``period_y``, ``a1`` (major diameter in nm), ``ellipticity``, and ``angle`` (in radians):

.. code-block:: python

   def build_unit_cell(
       period: float = 400.0,
       period_y: Optional[float] = None,
       a1: Optional[float] = None,
       ellipticity: Optional[float] = None,
       angle: Optional[float] = None,
       r1: Optional[float] = None,
       r2: Optional[float] = None,
       rot: Optional[float] = None,
       lc: float = 0.12,
       lc2: float = 5.0,
       lc3: float = 1.0,
   ) -> ngs.Mesh:

When ``NanoStruct`` calls :func:`meshing.mesh_from_py`, it inspects these parameters, translates legacy aliases automatically (``d_in_nm`` :math:`\to` ``period``, ``dy_in_nm`` :math:`\to` ``period_y``, ``diameter1`` :math:`\to` ``a1``), and passes your ``geo_params`` dictionary directly as keyword arguments.

Step 2 (Concept 2 — Coordinate Normalization to :math:`[0, 1] \times [0, d_y]`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside ``build_unit_cell``, all physical dimensions in nanometers are divided by ``period`` (:math:`a`) so the unit cell spans :math:`[0, 1] \times [0, d_y]` with its center at :math:`(0.5, 0.5\,d_y)`:

* **In** ``ellipse_msh_template.py`` **(single rotated ellipse at unit-cell center):**

  .. code-block:: python

     py = period if period_y is None else period_y
     dy = float(py) / float(period)

     # Normalize semi-major axis rx and semi-minor axis ry by period
     rx = float(a1) / (2.0 * float(period))
     ry = rx * (1.0 - float(ellipticity))
     ang_rad = float(angle)
     center = (0.5, dy / 2.0)

* **In** ``chiral_qbic_gorkunov.py`` **(two tilted elliptical bars separated along x):**

  .. code-block:: python

     py = period if period_y is None else period_y
     dy = float(py) / float(period)
     a = float(period)

     # Convert tilt angle from degrees to radians (+/- theta / 2)
     theta_rad = np.deg2rad(float(theta))
     rot1 = +0.5 * theta_rad if rot1 is None else rot1
     rot2 = -0.5 * theta_rad if rot2 is None else rot2

     # Normalize ellipse semi-axes rx, ry and center-to-center separation by period
     rx = 0.5 * float(w) / a
     ry = 0.5 * float(l) / a
     sep = (float(w) + float(gap)) / a

     # Normalized centers of left bar (c1) and right bar (c2)
     c1 = (0.5 - 0.5 * sep, 0.5 * dy)
     c2 = (0.5 + 0.5 * sep, 0.5 * dy)

Step 3 (Concept 3 — Outer Unit-Cell Box & Periodic ``copy=`` Boundaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``build_unit_cell`` calls ``meshing.unit_cell(dy, inclusions, refine=refine, curve_order=2)``, :func:`meshing.unit_cell` first constructs the outer unit-cell rectangle :math:`[0, 1] \times [0, d_y]` in ``netgen.geom2d.SplineGeometry`` and pairs opposite periodic faces using ``copy=``:

.. code-block:: python

   geo = g2.SplineGeometry()
   L = period_x   # 1.0
   H = dy         # period_y / period

   # 1. Add the 4 corner points of the normalized unit cell
   P = [geo.AddPoint(*q) for q in [(0, 0), (L, 0), (L, H), (0, H)]]

   # 2. Periodic x-boundaries: Left (P0 -> P3) is primary; Right (P1 -> P2) is its copy
   i_l = geo.Append(["line", P[0], P[3]], leftdomain=0, rightdomain=1, bc="left", maxh=refine.bg)
   geo.Append(["line", P[1], P[2]], leftdomain=1, rightdomain=0, bc="right", maxh=refine.bg, copy=i_l)

   # 3. Periodic y-boundaries: Bottom (P0 -> P1) is primary; Top (P3 -> P2) is its copy
   i_b = geo.Append(["line", P[0], P[1]], leftdomain=1, rightdomain=0, bc="bottom", maxh=refine.bg)
   geo.Append(["line", P[3], P[2]], leftdomain=0, rightdomain=1, bc="top", maxh=refine.bg, copy=i_b)

.. important::
   Notice the point ordering on periodic pairs:

   * ``left`` goes from ``P[0] (0,0)`` to ``P[3] (0,H)``, and ``right`` goes in the **same direction** from ``P[1] (L,0)`` to ``P[2] (L,H)`` with ``copy=i_l``.
   * Because ``left`` walks up the left border, the unit-cell interior (domain ``1``) is on the **right** (``leftdomain=0, rightdomain=1``). Because ``right`` walks up the right border, the unit-cell interior is on the **left** (``leftdomain=1, rightdomain=0``).

Step 4 (Concept 4 — Declarative Inclusions & Exact Rational Bézier Splines)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the template, each resonator is described as a clean Python dictionary inside the ``inclusions`` list:

* **Single rotated ellipse (``ellipse_msh_template.py``):**

  .. code-block:: python

     inclusions = [{
         "kind": "ellipse",
         "c": (0.5, dy / 2.0),
         "rx": rx,
         "ry": ry,
         "angle": ang_rad,
         "tag": 2,
     }]

* **Two-bar chiral qBIC dimer (``chiral_qbic_gorkunov.py``):**

  .. code-block:: python

     inclusions = []
     if include_bar1:
         inclusions.append({
             "kind": "ellipse",
             "c": c1,
             "rx": rx,
             "ry": ry,
             "angle": float(rot1),
             "tag": 2,
         })
     if include_bar2:
         inclusions.append({
             "kind": "ellipse",
             "c": c2,
             "rx": rx,
             "ry": ry,
             "angle": float(rot2),
             "tag": 2,
         })

  *(In the 3-layer qBIC metasurface, Layer 1 sets* ``include_bar1=True, include_bar2=False`` *for the bottom 10 nm offset slice, Layer 2 sets both* ``True`` *for the middle 90 nm overlap region, and Layer 3 sets* ``include_bar1=False, include_bar2=True`` *for the top 10 nm slice—all using this single template!)*

**What** :func:`meshing.unit_cell` **does with each ellipse dictionary:**
For each ellipse, :func:`meshing.unit_cell` loops over 12 angular intervals :math:`[a_0, a_1]` (:math:`30^\circ` each), calls :func:`meshing.arc_control_points` to compute the rotated endpoints ``pt0``, ``pt1`` and the off-curve tangent intersection control point ``pt_ctrl``, and appends a 3-point quadratic rational Bézier spline (``"spline3"``):

.. code-block:: python

   for a0, a1 in arc_segments(r=r, c=c, nseg=12):
       pt0, pt_ctrl, pt1 = arc_control_points(
           geo, kind="ellipse", c=c, a0=a0, a1=a1, rx=rx, ry=ry, angle=ang
       )
       geo.Append(
           ["spline3", pt0, pt_ctrl, pt1],
           leftdomain=tag,
           rightdomain=1,
           bc=f"inc{tag}",
           maxh=maxh_inc,
       )

All built-in shapes supported in the ``inclusions`` list of :func:`meshing.unit_cell` are summarized below:

.. list-table:: Built-in Inclusion Shapes Supported by ``meshing.unit_cell()``
   :header-rows: 1
   :widths: 16 44 40

   * - ``"kind"``
     - Required / Optional Keys
     - Description
   * - ``"circle"``
     - ``"c": (xc, yc)``, ``"r": radius``, ``"tag": int``
     - Circular inclusion of normalized radius ``r`` built from 12 ``spline3`` arcs.
   * - ``"ellipse"``
     - ``"c": (xc, yc)``, ``"rx": float``, ``"ry": float``, ``"angle": rad``, ``"tag": int``
     - Elliptical inclusion with semi-axes ``rx``, ``ry`` rotated counter-clockwise by ``angle`` (radians).
   * - ``"square"`` / ``"rectangle"``
     - ``"c": (xc, yc)``, ``"w": width``, ``"h": height`` (or ``"s": side``), ``"tag": int``
     - Axis-aligned rectangular or square inclusion centered at ``c``.
   * - ``"ring"``
     - ``"c": (xc, yc)``, ``"r_out": float``, ``"r_in": float``, ``"tag": int``
     - Annular ring between inner radius ``r_in`` (background) and outer radius ``r_out``.
   * - ``"dimer"``
     - ``"c": (xc, yc)``, ``"sub_kind": "circle"|"square"``, ``"r": float``, ``"gap": float``, ``"tag": int``
     - Symmetric pair of inclusions separated along :math:`x` by edge-to-edge ``gap``.

Step 5 (Concept 5 — Oriented Domain Tagging ``leftdomain`` / ``rightdomain`` & Materials)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because ``arc_segments`` steps counter-clockwise from :math:`0` to :math:`2\pi`, walking from ``pt0`` to ``pt1`` along each ``spline3`` arc keeps the **interior of the ellipse on the left** (``leftdomain=tag``, i.e. ``2``) and the **background medium on the right** (``rightdomain=1``).

After all inclusions are appended, :func:`meshing.unit_cell` assigns the domain names and maximum element sizes:

.. code-block:: python

   geo.SetMaterial(1, "bg")
   geo.SetDomainMaxH(1, refine.bg)
   for tag in tags:
       if tag != 1:
           geo.SetMaterial(tag, f"inc{tag}")
           geo.SetDomainMaxH(tag, refine.inc)

When ``NanoStruct`` builds the layer, ``"bg"`` (tag ``1``) is assigned the permittivity of ``background``, and ``"inc2"`` (tag ``2``) is assigned the permittivity of ``inclusion_a``. If your unit cell contains two *different* resonator materials, simply set ``"tag": 2`` on the first dictionary (mapped to ``inclusion_a``) and ``"tag": 3`` on the second dictionary (mapped to ``inclusion_b``).

Step 6 (Concept 6 — Mesh Refinement, Curving ``ngm.Curve(2)``, & Calling from ``NanoStruct``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the end of ``build_unit_cell``, the template computes the inclusion element size ``inc_h = lc / lc2``, packages it into :class:`meshing.Refine`, and delegates mesh generation and curving to :func:`meshing.unit_cell`:

.. code-block:: python

   inc_h = lc / lc2 if lc2 > 1.0 else lc
   refine = meshing.Refine(bg=lc, inc=inc_h)
   return meshing.unit_cell(dy, inclusions, refine=refine, curve_order=curve_order, period_x=1.0)

Inside :func:`meshing.unit_cell`, Netgen generates the 2D mesh, curves all boundary splines to polynomial order ``curve_order=2``, wraps it into ``ngsolve.Mesh``, and verifies the 4 topological invariants:

.. code-block:: python

   ngm = geo.GenerateMesh(maxh=refine.bg)
   ngm.Curve(curve_order)
   mesh = Mesh(ngm)

To use ``chiral_qbic_gorkunov.py`` in a simulation and visually inspect the generated mesh:

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.tri as mtri
   from EMUstack.backend import objects, materials

   period = 480.0
   mat_bars = materials.Material(4.0 + 0.01j)
   mat_air = materials.Material(1.0)

   # Build the middle overlap layer containing both tilted elliptical bars
   lay2 = objects.NanoStruct(
       '2D_array',
       period=period,
       diameter1=200.0,
       geo_file='chiral_qbic_gorkunov.py',
       geo_params={
           'period': period,
           'theta': 3.5,
           'include_bar1': True,
           'include_bar2': True,
           'lc': 0.15,
           'lc2': 1.5,
       },
       background=mat_air,
       inclusion_a=mat_bars,
       height_nm=90.0,
   )

   # Visually inspect the generated mesh and material tags (1 = bg, 2 = inc2)
   x_nm = lay2.x_arr[0] * period
   y_nm = lay2.x_arr[1] * period
   triangles = lay2.table_nod[:3].T - 1  # 0-based vertex indices
   triang = mtri.Triangulation(x_nm, y_nm, triangles)

   fig, ax = plt.subplots(figsize=(5, 5))
   ax.tripcolor(triang, facecolors=lay2.type_el, cmap='coolwarm', edgecolors='k', lw=0.4)
   ax.set_aspect('equal')
   ax.set_xlabel('x (nm)')
   ax.set_ylabel('y (nm)')
   ax.set_title(f'Chiral qBIC Unit Cell ({lay2.n_msh_pts} vertices, {lay2.n_msh_el} triangles)')

---

Advanced Recipe: Writing a Low-Level Custom ``SplineGeometry`` Template
-----------------------------------------------------------------------

If you want to model an arbitrary 2D polygon or custom curve (such as an L-shaped antenna, split-ring resonator, or crescent) that is not one of the built-in shapes in :func:`meshing.unit_cell`, your ``build_unit_cell(...)`` function in ``my_custom_template.py`` can construct a ``netgen.geom2d.SplineGeometry`` directly and return either the ``SplineGeometry`` or an ``ngsolve.Mesh``:

.. code-block:: python

   import netgen.geom2d as g2
   import ngsolve as ngs

   def build_unit_cell(
       period: float = 500.0,
       period_y: float = 500.0,
       arm_len: float = 300.0,
       arm_width: float = 100.0,
       lc: float = 0.10,
       lc2: float = 2.5,
       curve_order: int = 2,
   ) -> ngs.Mesh:
       dy = float(period_y) / float(period)
       L_arm = arm_len / period
       W_arm = arm_width / period
       inc_h = lc / lc2 if lc2 > 1.0 else lc

       geo = g2.SplineGeometry()

       # Step 3: Outer periodic box [0, 1] x [0, dy]
       P = [geo.AddPoint(*q) for q in [(0, 0), (1, 0), (1, dy), (0, dy)]]
       i_l = geo.Append(["line", P[0], P[3]], leftdomain=0, rightdomain=1, bc="left", maxh=lc)
       geo.Append(["line", P[1], P[2]], leftdomain=1, rightdomain=0, bc="right", maxh=lc, copy=i_l)
       i_b = geo.Append(["line", P[0], P[1]], leftdomain=1, rightdomain=0, bc="bottom", maxh=lc)
       geo.Append(["line", P[3], P[2]], leftdomain=0, rightdomain=1, bc="top", maxh=lc, copy=i_b)

       # Step 4 & 5: Counter-clockwise polygon for an L-shaped resonator (leftdomain=2, rightdomain=1)
       x0, y0 = 0.5 - L_arm / 2, 0.5 * dy - L_arm / 2
       pts_L = [
           (x0, y0),
           (x0 + L_arm, y0),
           (x0 + L_arm, y0 + W_arm),
           (x0 + W_arm, y0 + W_arm),
           (x0 + W_arm, y0 + L_arm),
           (x0, y0 + L_arm),
       ]
       Q = [geo.AddPoint(*p) for p in pts_L]
       for i in range(len(Q)):
           geo.Append(
               ["line", Q[i], Q[(i + 1) % len(Q)]],
               leftdomain=2,
               rightdomain=1,
               bc="inc2",
               maxh=inc_h,
           )

       # Step 6: Assign material names, domain maxh, generate mesh, and curve
       geo.SetMaterial(1, "bg")
       geo.SetDomainMaxH(1, lc)
       geo.SetMaterial(2, "inc2")
       geo.SetDomainMaxH(2, inc_h)

       ngm = geo.GenerateMesh(maxh=lc)
       ngm.Curve(curve_order)
       return ngs.Mesh(ngm)

---

``meshing`` Module API Reference
--------------------------------

.. automodule:: meshing
    :members:
    :undoc-members:
    :show-inheritance:
