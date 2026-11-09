2D Geometry & Meshing Guide (Gmsh & Netgen)
===========================================

EMUstack solves for the electromagnetic Bloch modes of 2D periodically structured layers (`NanoStruct('2D_array', ...)`) by discretizing the 2D unit cell into high-order finite elements in `NGSolve <https://ngsolve.org/>`_.

To support both legacy simulation scripts and pure-Python workflows, EMUstack provides **two complete 2D meshing pipelines** that converge into the same ``ngsolve.Mesh`` representation:

1. **The Gmsh Pipeline (Classic Workflow — ``.geo``):**
   Reads a parameterized Gmsh ``.geo`` script (or a pre-generated ``.mail`` / ``.msh`` file), substitutes geometry parameters from ``geo_params``, invokes the external ``gmsh`` binary to triangulate the unit cell, and imports the resulting mesh into Netgen/NGSolve with periodic vertex identifications.
2. **The Netgen Pipeline (Native Pure-Python Workflow — ``.py``):**
   Executes a pure-Python geometry builder (``build_unit_cell(...)`` in a ``*.py`` template) to construct a curved second-order periodic ``ngsolve.Mesh`` directly in memory—requiring **zero external binaries and zero temporary disk files**.

   Within a Netgen ``.py`` template, you can choose between **two routes** depending on the complexity of your resonator geometry:

   * **Netgen Route B1 — High-Level Helper** :func:`meshing.unit_cell` **(Standard Inclusions):**
     If your unit cell is composed of one or more **standard geometric primitives**—circles, rotated ellipses, squares, rectangles, concentric rings, or dimers (such as in ``ellipse_msh_template.py`` or the two tilted elliptical bars in ``chiral_qbic_gorkunov_msh_template.py``)—you do not need to write low-level Netgen CSG code. You simply normalize your dimensions, describe each inclusion as a Python dictionary (``{"kind": "ellipse", ...}``), and pass the list to :func:`meshing.unit_cell`, which builds the periodic box, curved splines, domain tags, and mesh automatically.
   * **Netgen Route B2 — Direct** ``netgen.geom2d.SplineGeometry`` **(Fancier / Arbitrary Custom Shapes):**
     If you need a **fancier or arbitrary custom shape** that is not one of the standard primitives supported by :func:`meshing.unit_cell` (for example: an L-shaped nanoantenna, a split-ring resonator with a cut gap, a crescent, a polygon, or freeform Bézier curves), you bypass :func:`meshing.unit_cell` and construct a ``netgen.geom2d.SplineGeometry`` directly inside ``build_unit_cell(...)``.

.. note::
   **How EMUstack Selects the Pipeline:**

   * Passing a Python template ``geo_file='... .py'`` **always** runs the native Netgen pipeline.
   * Passing a Gmsh script ``geo_file='... .geo'`` runs the Gmsh pipeline (unless overridden).
   * For built-in ``NanoStruct`` shapes without a ``geo_file`` (``inc_shape='circle'``, ``'ellipse'``, ``'square'``, ``'rectangle'``, ``'ring'``, ``'dimer'``), the pipeline is controlled either globally via ``objects.default_mesher = 'gmsh'`` (default) or ``'netgen'``, or per layer via ``NanoStruct(..., mesher='netgen')``.

.. _shipped-templates:

Using the Shipped Templates
---------------------------

EMUstack ships 30 mesh templates inside the package (``EMUstack/backend/data/meshes/`` in the
repository, inside ``site-packages`` once EMUstack is installed). You never need their path:
pass the **file name** as ``geo_file`` and the parameters as ``geo_params``:

.. code-block:: python

    from EMUstack.backend import objects, materials, meshing

    layer = objects.NanoStruct(
        '2D_array', geo_file='ellipse_msh_template.geo',   # or the short name 'ellipse.geo'
        geo_params={'period': 400.0, 'diameter1': 160.0, 'ellipticity': 0.3, 'angle': 0.5,
                    'lc': 0.1, 'lc2': 2.0},
        background=materials.Air, inclusion_a=materials.Si_c, height_nm=100.0)

    meshing.list_templates()        # names of all shipped templates

``geo_file`` is looked up in this order:

1. **Your working directory** (or the path you give), so your own files always win.
2. **The shipped templates**, by full name (``ellipse_msh_template.geo``), short name
   (``ellipse.geo``, ``ellipse.py``) or old v1 name (``ring1_msh_template.geo``).

The extension picks the mesher: ``.geo`` templates are meshed by gmsh, ``.py`` templates by
netgen.

**To start your own geometry from a shipped template**, copy it into your project and edit the
copy:

.. code-block:: python

    meshing.copy_template('ellipse.geo')              # writes ./ellipse_msh_template.geo
    meshing.copy_template('ellipse.geo', 'my_cell.geo')

Because the working directory is searched first, ``geo_file='ellipse_msh_template.geo'`` then
uses your edited copy. Give it a new name (``my_cell.geo``) to keep both. To read a shipped
template without copying it, ``meshing.resolve_mesh_file_path('ellipse.geo')`` returns its path.

Lengths are in nm and angles in radians unless the table says otherwise. Every template also
takes the mesh sizes ``lc`` (background) and ``lc2``, ``lc3``, ... (finer sizes ``lc/lcN`` at
the inclusions; see Concept 6).

.. list-table:: Templates shipped with EMUstack (``meshing.list_templates()``)
   :header-rows: 1
   :widths: 28 8 34 30

   * - Template
     - Mesher
     - Geometry
     - Parameters
   * - ``16_msh_template.geo``
     - gmsh
     - 4x4 grid of inclusions (circle, ellipse or square), numbered row by row from the top left.
     - ``period``, ``period_y``, ``diameter1`` ... ``diameter16``, ``ellipticity``, ``square``, ``lc``, ``posx``, ``posy``, ``lc2``, ``lc3``
   * - ``1_msh_template.geo``
     - gmsh
     - One inclusion: circle, ellipse (``ellipticity``) or square (``square = 1``).
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``square``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``2_msh_template.geo``
     - gmsh
     - Two circular inclusions per cell.
     - ``period``, ``period_y``, ``diameter1``, ``diameter2``, ``lc``, ``posx``, ``posy``, ``lc2``, ``lc3``, ``lc4``
   * - ``3_msh_template.geo``
     - gmsh
     - Three circular inclusions per cell.
     - ``period``, ``period_y``, ``diameter1``, ``diameter2``, ``diameter3``, ``lc``, ``lc2``, ``lc3``, ``lc4``, ``lc5``
   * - ``4_msh_template.geo``
     - gmsh
     - 2x2 grid of inclusions (circle, ellipse or square); ``posx``/``posy`` move them toward the centre.
     - ``period``, ``period_y``, ``diameter1`` ... ``diameter4``, ``ellipticity``, ``square``, ``lc``, ``posx``, ``posy``, ``lc2``, ``lc3``, ``lc4``, ``lc5``, ``lc6``
   * - ``9_msh_template.geo``
     - gmsh
     - 3x3 grid of circular inclusions.
     - ``period``, ``period_y``, ``diameter1`` ... ``diameter9``, ``lc``, ``posx``, ``posy``, ``lc2``, ``lc3``
   * - ``L_msh_template.geo``
     - gmsh
     - L-shaped inclusion with a rounded corner.
     - ``period``, ``period_y``, ``length``, ``width``, ``rounding``, ``lc``, ``lc2``
   * - ``SRR_msh_template.geo``
     - gmsh
     - Split-ring resonator with its open side at the top.
     - ``period``, ``period_y``, ``len_vertical``, ``len_horizontal``, ``width``, ``lc``, ``lc2``, ``lc3``
   * - ``chiral_qbic_gorkunov_c4_msh_template.py``
     - netgen
     - Four rotated bar dimers in a C4-symmetric cell (Gorkunov et al.).
     - ``period``, ``period_y``, ``theta_deg``, ``length``, ``width``, ``gap``, ``include_bar1``, ``include_bar2``, ``lc``, ``lc2``
   * - ``chiral_qbic_gorkunov_msh_template.py``
     - netgen
     - Bar dimer of Gorkunov et al.; ``theta_deg`` in degrees.
     - ``period``, ``period_y``, ``theta_deg``, ``z_offset``, ``length``, ``width``, ``gap``, ``include_bar1``, ``include_bar2``, ``angle1``, ``angle2``, ``lc``, ``lc2``, ``lc3``
   * - ``chiral_qbic_msh_template.py``
     - netgen
     - Chiral quasi-BIC cell; ``rx``, ``ry``, ``gap`` normalized, ``theta0`` in radians.
     - ``period``, ``period_y``, ``alpha``, ``theta0``, ``rx``, ``ry``, ``gap``, ``lc``, ``lc2``, ``lc3``
   * - ``chiral_slits_msh_template.geo``
     - gmsh
     - Chiral slit pattern, slits moved by ``offset_x``, ``offset_y``.
     - ``period``, ``period_y``, ``width``, ``height``, ``smooth``, ``offset_x``, ``offset_y``, ``lc``, ``lc2``, ``lc3``
   * - ``cross_msh_template.geo``
     - gmsh
     - Cross of size ``width`` x ``height``.
     - ``period``, ``period_y``, ``width``, ``height``, ``smooth``, ``lc``, ``lc2``, ``lc3``
   * - ``cross_shell_msh_template.geo``
     - gmsh
     - Cross-shaped shell of wall ``thickness``.
     - ``period``, ``period_y``, ``width``, ``height``, ``smooth``, ``thickness``, ``lc``, ``lc2``, ``lc3``
   * - ``dimer_msh_template.geo``
     - gmsh
     - Two circles separated by ``gap``.
     - ``period``, ``period_y``, ``diameter1``, ``diameter2``, ``gap``, ``ellipticity``, ``lc``, ``lc2``, ``lc3``
   * - ``double_strip_msh_template.geo``
     - gmsh
     - Inclusion inside two strips (``strip``, ``strip2``).
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``square``, ``strip``, ``strip2``, ``lc``, ``lc2``, ``lc3``
   * - ``egg_inclusion_msh_template.geo``
     - gmsh
     - Egg-shaped inclusion containing a circular inclusion of radius ``r_inc`` (lengths in nm).
     - ``period``, ``period_y``, ``r1``, ``r2``, ``r3``, ``angle``, ``shiftx``, ``shifty``, ``r_inc``, ``shift_inc_x``, ``shift_inc_y``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``egg_msh_template.geo``
     - gmsh
     - Egg-shaped inclusion made of arcs of radii ``r1``-``r3``, rotated by ``angle`` and moved by ``shiftx``/``shifty`` (lengths in nm).
     - ``period``, ``period_y``, ``r1``, ``r2``, ``r3``, ``angle``, ``shiftx``, ``shifty``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``elle_msh_template.geo``
     - gmsh
     - ELLE shape (Zanotto et al., Nanophotonics 8, 2291 (2019)); ``L1``-``L4`` in units of the period.
     - ``period``, ``period_y``, ``L1``, ``L2``, ``L3``, ``L4``
   * - ``ellipse_msh_template.geo``
     - gmsh
     - Rotated ellipse (``angle`` in radians).
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``angle``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``ellipse_msh_template.py``
     - netgen
     - Rotated ellipse (``angle`` in radians), curved second-order netgen mesh.
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``angle``, ``lc``, ``lc2``, ``lc3``
   * - ``hex_msh_template.geo``
     - gmsh
     - One inclusion in a hexagonal lattice.
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``square``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``rectangle_msh_template.geo``
     - gmsh
     - Rectangle ``width`` x ``height``, corners rounded by ``smooth``.
     - ``period``, ``period_y``, ``width``, ``height``, ``smooth``, ``lc``, ``lc2``, ``lc3``
   * - ``rectangle_shell_msh_template.geo``
     - gmsh
     - Rectangular shell of wall ``thickness``.
     - ``period``, ``period_y``, ``width``, ``height``, ``thickness``, ``ellipticity``, ``smooth``, ``lc``, ``lc2``, ``lc3``
   * - ``ring_msh_template.geo``
     - gmsh
     - Ring: outer ``diameter1``, inner ``diameter2``; ``xshift`` moves the hole.
     - ``period``, ``period_y``, ``diameter1``, ``diameter2``, ``ellipticity``, ``square``, ``lc``, ``xshift``, ``lc2``, ``lc3``
   * - ``row_2_msh_template.geo``
     - gmsh
     - Row of 2 inclusions; period, size and spacing are set inside the file (edit a copy).
     - ``ellipticity``, ``square``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``row_5_msh_template.geo``
     - gmsh
     - Row of 5 inclusions; period, size and spacing are set inside the file (edit a copy).
     - ``ellipticity``, ``square``, ``lc``, ``lc2``, ``lc3``, ``lc4``
   * - ``square_dimer_msh_template.geo``
     - gmsh
     - Two squares (sides ``side1``, ``side2``) separated by ``gap``.
     - ``period``, ``period_y``, ``side1``, ``side2``, ``gap``, ``ellipticity``, ``smooth``, ``lc``, ``lc2``, ``lc3``
   * - ``square_shell_dimer_msh_template.geo``
     - gmsh
     - Two rectangular shells of wall ``thickness``, separated by ``gap``.
     - ``period``, ``period_y``, ``width1``, ``width2``, ``height1``, ``height2``, ``gap``, ``thickness``, ``ellipticity``, ``smooth``, ``lc``, ``lc2``, ``lc3``
   * - ``strip_msh_template.geo``
     - gmsh
     - Inclusion inside a strip of width ``strip`` (``diameter1 < strip < period_y``).
     - ``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``square``, ``strip``, ``lc``, ``lc2``, ``lc3``

.. _mesh-folder:

Where the Meshes Are Written
----------------------------

With gmsh, ``NanoStruct`` writes three files per mesh: the ``.geo`` with the parameters filled
in, the ``.msh`` made by gmsh, and the ``.mail`` that EMUstack reads. They go in a folder named
``emustack_meshes``, so you can open them in gmsh to inspect the mesh:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Geometry
     - Meshes written to
   * - Built-in shape (``inc_shape='circle'``, ...) or a shipped template given by name
     - ``emustack_meshes/`` in the working directory (where the script or notebook runs)
   * - Your own ``.geo``, in the working directory or given with a relative or absolute path
     - ``emustack_meshes/`` next to that ``.geo``
   * - Any of the above, with ``meshing.mesh_dir`` set (or the ``EMUSTACK_MESH_DIR``
       environment variable)
     - that folder

.. code-block:: python

    from EMUstack.backend import meshing
    meshing.mesh_dir = '/scratch/me/meshes'   # every mesh goes here, e.g. on a cluster

    layer.mesh_dir                            # folder of this layer's mesh
    layer.mesh_file                           # its .mail file name

The file names hold the template name, the periods and the parameters, for example
``ellipse400_0_400_0__diameter1160_lc0_1.mail``. If the folder cannot be written (a read-only
shared folder, for example), the meshes go to your user cache folder instead
(``~/.cache/emustack/meshes`` on Linux, ``~/Library/Caches/emustack/meshes`` on macOS,
``%LOCALAPPDATA%\emustack\meshes`` on Windows), with a warning.

``NanoStruct`` meshes again on every run by default (``force_mesh=True``, as in version 1).
With ``force_mesh=False`` it reuses a mesh already in the folder, unless your ``.geo`` has
changed since. The netgen mesher (``.py`` templates, ``mesher='netgen'``) keeps meshes in memory
and writes no files.

---

Part I: General Presentation & the 6 Core Concepts
--------------------------------------------------

Regardless of whether you use the **Gmsh pipeline (``.geo``)** or the **Netgen pipeline (``.py``, via Route B1 or Route B2)**, building a valid 2D periodic unit-cell mesh in EMUstack always requires the exact same **six conceptual stages**.

.. list-table:: Summary of the 6 Core Meshing Stages Across Gmsh and Both Netgen Routes
   :header-rows: 1
   :widths: 19 27 27 27

   * - Stage / Concept
     - Gmsh Pipeline (``.geo``)
     - Netgen Route B1: ``meshing.unit_cell()`` (Standard Shapes)
     - Netgen Route B2: Direct ``SplineGeometry`` (Custom Shapes)
   * - **1. Parameter Injection**
     - Textual substitution of ``key = 0;`` placeholders in the ``.geo`` script from ``geo_params``.
     - Python signature inspection of ``build_unit_cell(...)`` and keyword passing from ``geo_params``.
     - Python signature inspection of ``build_unit_cell(...)`` and keyword passing from ``geo_params``.
   * - **2. Coordinate Normalization**
     - Physical lengths in nm are divided by ``period`` (:math:`a`) onto :math:`[0, 1] \times [-d_y, 0]`.
     - Physical lengths in nm are divided by ``period`` (:math:`a`) onto :math:`[0, 1] \times [0, d_y]`.
     - Physical lengths in nm are divided by ``period`` (:math:`a`) onto :math:`[0, 1] \times [0, d_y]`.
   * - **3. Periodic Outer Boundaries**
     - Outer edges tagged as ``Physical Line(1..4)``; opposite vertices paired on import via ``AddPointIdentification``.
     - **Automated by** ``unit_cell()`` (creates corners and pairs left/right and bottom/top edges with ``copy=``).
     - Append primary ``left`` and ``bottom`` lines, then append ``right`` (``copy=i_l``) and ``top`` (``copy=i_b``).
   * - **4. Interior Inclusions**
     - ``Circle`` / ``Ellipsis`` arcs and ``Rotate`` commands inside ``Line Loop`` definitions.
     - **Automated by** ``unit_cell()`` from declarative ``inclusions`` dicts (``circle``, ``ellipse``, ``square``, ``rectangle``, ``ring``, ``dimer``).
     - Append custom ``["line", p0, p1]`` segments and/or 3-point quadratic Bézier ``["spline3", p0, pc, p1]`` arcs.
   * - **5. Subdomain Material Tags**
     - ``Plane Surface`` + ``Physical Surface(1)`` (background) and ``Physical Surface(2..6)`` (inclusions).
     - **Automated by** ``unit_cell()`` using the ``"tag": 2..6`` key on each dictionary in ``inclusions``.
     - Set ``leftdomain=tag, rightdomain=1`` on counter-clockwise inclusion curves + ``geo.SetMaterial(...)``.
   * - **6. Refinement & Meshing**
     - Point sizes ``lc`` (background), ``lc2 = lc/factor`` (boundary), ``lc3`` (center) + ``gmsh`` binary call.
     - **Automated by** ``unit_cell()`` from ``lc`` (background) and ``"lc": lc_n`` on each inclusion (converted internally to Netgen ``maxh = lc / lc_n``) + ``Curve(2)`` + sanity checks.
     - Call ``SetDomainMaxH(..., maxh)``, ``geo.GenerateMesh(maxh=...)``, and ``ngm.Curve(2)``.

Concept 1: Parameter Injection (``geo_file`` & ``geo_params``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you instantiate ``objects.NanoStruct('2D_array', geo_file=..., geo_params={...})``, **all** geometric and mesh refinement parameters (``period``, ``period_y``, ``diameter1``, ``ellipticity``, ``angle``, ``lc``, ``lc2``, ``lc3``; the names are the same for ``.geo`` and ``.py`` templates, see :ref:`template-naming`) can be passed directly inside the ``geo_params`` dictionary—without repeating legacy positional arguments like ``period``, ``diameter1``, ``ellipticity``, or ``inc_shape='custom'``. EMUstack locates ``geo_file`` (first in your working directory, then among the shipped templates; see :ref:`shipped-templates`) and dispatches ``geo_params`` to the selected template:

* **In the Gmsh pipeline**, ``NanoStruct`` / :func:`meshing.mesh_from_geo` reads the ``.geo`` file as text and replaces placeholder assignments such as ``period = 0;``, ``period_y = 0;``, ``diameter1 = 0;``, ``ellipticity = 0;``, ``angle = 0;``, ``lc = 0;``, and ``lc2 = lc/1;`` with the numerical values from ``geo_params``. If a key in ``geo_params`` does not have a matching placeholder in the ``.geo`` file, a ``KeyError`` is raised immediately.
* **In the Netgen pipeline (both Routes B1 and B2)**, :func:`meshing.mesh_from_py` dynamically imports the ``.py`` module, inspects the Python signature of ``build_unit_cell(...)`` using ``inspect.signature``, verifies that every key in ``geo_params`` is a valid parameter of ``build_unit_cell`` (or an old name of one) (raising a ``TypeError`` on typos), and calls ``build_unit_cell(**call_params)``.

.. _template-naming:

Naming: One Vocabulary for ``.geo``, ``.py`` and ``unit_cell``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three ways of describing a unit cell use the same words for the same quantities:

* **Template files** are named ``<shape>_msh_template.geo`` (gmsh) or ``<shape>_msh_template.py`` (netgen), where ``<shape>`` is the ``NanoStruct(inc_shape=...)`` name when there is one (``ring``, ``rectangle``, ``cross_shell``, ...). ``N_msh_template.geo`` holds ``N`` circular inclusions per cell. A template can also be given by its short name: ``geo_file='ring.geo'``.
* **Template parameters** (``.geo`` placeholders, ``.py`` keyword arguments and ``geo_params`` keys) share one set of names. Lengths are in nm and angles in radians, unless the name ends in ``_deg``.
* **unit_cell inclusion keys** use the same words in normalized coordinates: ``kind`` (the same shape names), ``center``, ``radius``, ``width``, ``height``, ``side``, ``angle``/``angle_deg``, ``radius_out``/``radius_in``.

Old names keep working everywhere: old file names are looked up in ``meshing.LEGACY_TEMPLATE_NAMES``; old parameter names are listed in ``meshing.PARAM_ALIASES`` (shared by all templates) or in the ``(legacy: ...)`` comment of the template's placeholder line; old ``unit_cell`` keys are in ``meshing.INCLUSION_KEY_ALIASES``. A parameter name present in the template is always filled as is, so custom templates written with the old names behave exactly as before, and accept the new names too.

.. list-table:: Template names
   :header-rows: 1
   :widths: 30 40 30

   * - Template
     - Parameters
     - Old names
   * - all
     - ``period``, ``period_y``
     - ``d_in_nm``, ``dy_in_nm``
   * - ``1_``, ``2_``, ``3_``, ``4_``, ``9_``, ``16_``, ``hex_``, ``ellipse_``, ``dimer_``, ``strip_``, ``double_strip_``
     - ``diameter1``, ``diameter2``, ...
     - ``a1``, ``a2``, ... (``d1``, ``d2``, ... in ``16_``)
   * - ``ring_`` (was ``ring1_``)
     - ``diameter1`` (outer), ``diameter2`` (inner), ``xshift``
     - ``a1``, ``a2``, ``xshift_nm``
   * - ``rectangle_``, ``cross_`` (were ``rect1_``, ``cross1_``), ``chiral_slits_``
     - ``width``, ``height``; ``offset_x``, ``offset_y`` in ``chiral_slits_``
     - ``a1``, ``a2``; ``offset_x_nm``, ``offset_y_nm``
   * - ``rectangle_shell_``, ``cross_shell_`` (were ``rect_shell1_``, ``cross_shell1_``)
     - ``width``, ``height``, ``thickness``
     - ``a1``, ``b1`` / ``a2``, ``t``
   * - ``square_dimer_``, ``square_shell_dimer_`` (were ``..._dimer1_``)
     - ``side1``, ``side2`` / ``width1``, ``width2``, ``height1``, ``height2``, ``thickness``
     - ``a1``, ``a2`` / ``a1``, ``a2``, ``b1``, ``b2``, ``t``
   * - ``SRR_``
     - ``len_vertical``, ``len_horizontal``, ``width``
     - ``lvert_nm``, ``lhori_nm``, ``width_nm``
   * - ``L_``
     - ``length``, ``width``, ``rounding``
     - ``L_nm``, ``W_nm``, ``r``
   * - ``chiral_qbic_gorkunov_msh_template.py`` (was ``chiral_qbic_gorkunov.py``)
     - ``theta_deg``, ``length``, ``width``, ``z_offset``, ``angle1``, ``angle2``
     - ``theta``, ``l``, ``w``, ``d``, ``rot1``, ``rot2``
   * - ``chiral_qbic_gorkunov_c4_msh_template.py`` (new)
     - ``theta_deg`` (rotation of each bar, the paper's θ), ``length``, ``width``, ``gap``, ``include_bar1``, ``include_bar2``
     -
   * - ``chiral_qbic_msh_template.py`` (was ``chiral_qbic_template.py``)
     - unchanged; ``rx``, ``ry``, ``gap`` are normalized to the period
     -
   * - ``egg_``, ``egg_inclusion_``, ``elle_``
     - unchanged; their shape parameters are normalized to the period
     -

Mesh sizes keep their names everywhere: ``lc`` is the background size and ``lc2``, ``lc3``, ... are refinement divisors (``lcN = lc/1;`` in a ``.geo`` file).

A few templates have layout rules:

* ``4_``, ``9_`` and ``16_`` place their inclusions on a 2×2, 3×3 and 4×4 grid, numbered row by row from the top left. Inclusion 1 is ``inclusion_a``, all others ``inclusion_b``. ``posx`` and ``posy`` move the inclusions towards the cell centre.
* ``strip_`` puts the inclusion (``diameter1``) inside a strip of width ``strip`` (``diameter2`` for ``NanoStruct``) that runs along *x*, so the strip must be wider than the inclusion and narrower than the cell. ``double_strip_`` adds a band of total width ``strip2`` (``diameter3``) around it. ``NanoStruct`` checks these rules before meshing.

gmsh never returns on some invalid geometries, such as lines that cross an inclusion. EMUstack stops gmsh after ``meshing.gmsh_timeout`` seconds (default 600, or the ``EMUSTACK_GMSH_TIMEOUT`` environment variable) and raises an error that names the ``.geo`` file.

.. _legacy-params:

Renaming a ``.py`` Template Parameter Safely: ``@meshing.legacy_params``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``.geo`` template keeps its old names in a ``(legacy: ...)`` comment. A ``.py`` template is a Python function, so it needs a Python tool for the same job: the :func:`meshing.legacy_params` decorator.

**The problem.** Suppose a template was written with a cryptic keyword:

.. code-block:: python

   def build_unit_cell(period=400.0, a1=None, lc=0.12):
       ...

and you rename ``a1`` to the readable ``diameter1``. Every script that still calls ``build_unit_cell(a1=300)``, or passes ``geo_params={'a1': 300}`` to ``NanoStruct``, now fails with ``TypeError: unexpected keyword argument 'a1'``. Keeping both keywords in the signature would work, but the function would then have two parameters for one quantity, and every template would need the same "if a1 is given, copy it to diameter1" code.

**The solution.** Write the function with the new name only, and declare the old name once, above it:

.. code-block:: python

   from EMUstack.backend import meshing

   @meshing.legacy_params(a1="diameter1")      # old name="new name"
   def build_unit_cell(period=400.0, diameter1=None, lc=0.12):
       ...                                      # the body only ever sees diameter1

The decorator wraps the function in a thin layer that renames old keywords before the call. You can list several pairs, as the Gorkunov template does: ``@meshing.legacy_params(theta="theta_deg", l="length", w="width", d="z_offset", rot1="angle1", rot2="angle2")``.

**What happens on each call.** With the template above:

.. code-block:: python

   build_unit_cell(diameter1=300)            # new name: passed through unchanged
   build_unit_cell(a1=300)                   # old name: renamed to diameter1=300 first
   build_unit_cell(500, 300)                 # positional: untouched (period=500, diameter1=300)
   build_unit_cell(a1=300, diameter1=250)    # both: which one is meant? refuse instead of guessing
   # TypeError: build_unit_cell() got both 'a1' and its new name 'diameter1'

Step by step, the wrapper:

1. looks at the keyword arguments of the call;
2. for each ``old="new"`` pair, if ``old`` was passed, moves its value to ``new``, or raises the ``TypeError`` above if ``new`` was passed as well;
3. calls the real function, which never sees an old name.

**Under the hood.** The whole decorator is a dozen lines of :mod:`meshing`:

.. code-block:: python

   def legacy_params(**renames):
       def decorator(fn):
           @functools.wraps(fn)                     # keep name, docstring, signature
           def wrapper(*args, **kwargs):
               for old, new in renames.items():
                   if old in kwargs:
                       if new in kwargs:
                           raise TypeError(f"{fn.__name__}() got both '{old}' and its new name '{new}'")
                       kwargs[new] = kwargs.pop(old)
               return fn(*args, **kwargs)
           wrapper.legacy_params = dict(renames)    # lets mesh_from_py see the old names
           return wrapper
       return decorator

Two details make it fit the rest of EMUstack:

* ``functools.wraps`` keeps the function's name, docstring and signature. ``inspect.signature(build_unit_cell)`` still shows ``(period=400.0, diameter1=None, lc=0.12)``, so ``help()``, the API reference and tab completion show only the new names.
* ``wrapper.legacy_params`` records the renames (here ``{'a1': 'diameter1'}``). :func:`meshing.mesh_from_py` checks every ``geo_params`` key against the builder's signature and raises ``TypeError: Unexpected parameter ...`` on a typo. It reads this attribute, so it accepts ``'a1'`` as a valid key and passes it on as ``diameter1``.

**How a ``NanoStruct`` call reaches the template.** The rename works the same whichever way the template is called:

.. code-block:: text

   NanoStruct(geo_file='ellipse_msh_template.py', geo_params={'a1': 300, ...})
     -> meshing.mesh_from_py(...)                    'a1' is valid: listed in build_unit_cell.legacy_params
       -> build_unit_cell(diameter1=300, ...)        mesh_from_py already uses the new name

   import ellipse_msh_template                       (calling the template directly)
   ellipse_msh_template.build_unit_cell(a1=300)
     -> wrapper renames a1 -> diameter1
       -> build_unit_cell(diameter1=300)

**Recipe: renaming a parameter of your own template.**

1. Rename the parameter in the signature and in the body (``a1`` -> ``diameter1``).
2. Add ``@meshing.legacy_params(a1="diameter1")`` directly above ``def build_unit_cell``.
3. Mention the old name in the docstring, so readers know where it went.
4. Check both spellings give the same mesh, for example ``build_unit_cell(a1=300)`` and ``build_unit_cell(diameter1=300)``.

If a template defines ``build_geometry`` as an alias that calls ``build_unit_cell(*args, **kwargs)``, the alias inherits the renames.

**What it does not do.**

* **It renames, it does not convert.** The old and new keywords must mean exactly the same value in the same units. ``theta`` -> ``theta_deg`` is fine because both are degrees. If you change a parameter from degrees to radians, or from nm to a fraction of the period, a rename would silently give wrong geometry. Keep the old parameter and convert it in the body instead.
* **It only applies to keyword arguments.** Positional calls are untouched, so keep the parameter in the same position when you rename it.
* **It is only for ``.py`` templates.** ``.geo`` templates use the ``(legacy: ...)`` comment, ``unit_cell`` inclusion keys use ``meshing.INCLUSION_KEY_ALIASES``, and the names shared by all templates (``d_in_nm``, ``a1``, ...) are in ``meshing.PARAM_ALIASES``.

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
* **In the Netgen pipeline**, the **left** edge (:math:`\tilde{x}=0`) is appended as primary segment ``i_l`` and the **right** edge (:math:`\tilde{x}=1`) is appended in the **same bottom-to-top direction** with ``copy=i_l``. Likewise, the **bottom** edge (:math:`\tilde{y}=0`) is appended as primary segment ``i_b`` and the **top** edge (:math:`\tilde{y}=d_y`) in the **same left-to-right direction** with ``copy=i_b``. The ``copy=`` argument instructs Netgen to clone the 1D mesh nodes from the primary edge onto its partner edge and automatically register the periodic vertex identifications. *(When using Route B1, :func:`meshing.unit_cell` sets up this outer periodic box for you automatically; when using Route B2, you write these 5 lines at the top of your ``SplineGeometry``.)*

Concept 4: Interior Inclusions & High-Order Curved Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximating curved dielectric or metallic resonators (such as circular nanopillars or tilted elliptical bars in a high-:math:`Q` quasi-BIC metasurface) with straight-edged polygons introduces artificial corner singularities and staircase errors. Both pipelines represent curved boundaries using **second-order isoparametric curved elements**:

* **In the Gmsh pipeline**, curved boundaries are defined using Gmsh's built-in ``Circle`` or ``Ellipsis`` arc primitives (optionally rotated via ``Rotate``) and meshed either as 6-node quadratic triangles (``gmsh -2 -order 2``) or curved to second order upon Netgen import (``ngm.Curve(2)``).
* **In the Netgen pipeline**, curved boundaries are represented in ``netgen.geom2d.SplineGeometry`` using 3-point quadratic rational Bézier spline segments (``["spline3", pt0, pt_ctrl, pt1]``) together with ``ngm.Curve(2)``:

  * **In Route B1 (** :func:`meshing.unit_cell` **):** You simply specify a dictionary such as ``{"kind": "ellipse", "center": (xc, yc), "rx": rx, "ry": ry, "angle": alpha, "tag": 2, "lc": lc2}``. Internally, :func:`meshing.unit_cell` divides the circle or ellipse into :math:`N_{\text{seg}} = 24` angular sectors :math:`[a_0, a_1]` (:math:`\Delta\theta = 15^\circ`) and calls :func:`meshing.arc_control_points` to compute the start point :math:`\mathbf{P}_0`, the end point :math:`\mathbf{P}_1`, and the exact **off-curve tangent intersection control point** :math:`\mathbf{P}_{\text{ctrl}}`:

    .. math::

       \mathbf{P}_{\text{ctrl}} = \mathbf{c} +
       \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}
       \begin{pmatrix} r_x \sec(\Delta\theta / 2) \cos\theta_{\text{mid}} \\ r_y \sec(\Delta\theta / 2) \sin\theta_{\text{mid}} \end{pmatrix},
       \qquad \theta_{\text{mid}} = \frac{a_0 + a_1}{2}.

  * **In Route B2 (Direct** ``SplineGeometry`` **):** For arbitrary polygons or custom resonators, you append straight segments via ``geo.Append(["line", p0, p1], ...)`` and curved segments via ``geo.Append(["spline3", pt0, pt_ctrl, pt1], ...)`` (you can even call the helper :func:`meshing.arc_control_points` for circular/elliptical fillets or arcs inside your custom shape).

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
* **In the Netgen pipeline**, 2D regions are defined topologically by **directed boundary curves** via ``leftdomain`` and ``rightdomain``: as you walk along a segment from its start point to its end point, ``leftdomain`` is the region ID to your left and ``rightdomain`` is the region ID to your right (where ``0`` denotes the exterior of the computational domain). When an inclusion boundary is traversed **counter-clockwise**, its interior lies on the left (``leftdomain=tag``) and the surrounding background lies on the right (``rightdomain=1``). In **Route B1**, you simply set ``"tag": 2`` (or ``3..6``) in each inclusion dictionary and :func:`meshing.unit_cell` sets ``leftdomain``, ``rightdomain``, and ``geo.SetMaterial(...)`` automatically. In **Route B2**, you pass ``leftdomain=tag, rightdomain=1`` directly to ``geo.Append(...)`` and call ``geo.SetMaterial(...)``.

Concept 6: Mesh Refinement Controls (``lc, lc2, ..., lc_n`` vs. Netgen ``maxh``) & Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The General Role of ``lc`` and ``lc2, lc3, ..., lc_n``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mesh resolution in EMUstack is controlled in dimensionless unit-cell units (fractions of ``period``):

* ``lc`` (**baseline background mesh size**): Characteristic triangle edge length in the background medium (domain ``tag = 1``, ``"bg"``) and along the outer periodic unit-cell boundaries. For example, ``lc = 0.08`` produces roughly :math:`1 / 0.08 \approx 12\text{--}13` elements across the unit cell.
* ``lc2, lc3, ..., lc_n`` (**dimensionless local mesh-scaling factors**, :math:`\text{lc}_n > 0`): Scale ``lc`` to set the local target element size :math:`h_n` at specific geometric features:

  .. math::

     h_n = \frac{\text{lc}}{\text{lc}_n} \qquad (\text{for any } \text{lc}_n > 0).

  * :math:`\text{lc}_n > 1` :math:`\implies h_n < \text{lc}` (**finer mesh** than the background—for example, inside or along the boundary of a high-index dielectric or plasmonic resonator where internal wavelengths are shorter and local fields are sharper).
  * :math:`\text{lc}_n = 1` :math:`\implies h_n = \text{lc}` (**same mesh size** as the background).
  * :math:`0 < \text{lc}_n < 1` :math:`\implies h_n > \text{lc}` (**coarser mesh** than the background—for example, inside a low-index air hole etched in a high-index slab, or at the interior center of a large inclusion whose boundary is already finely resolved).

.. important::
   **Do** ``lc2, lc3, ..., lc_n`` **Always Map 1-to-1 to Subdomains or Inclusions? No.**

   While it is often convenient to think of ``lc_n`` as the refinement factor of an inclusion or material domain, **in general a mesh-scaling parameter** ``lc_n`` **can control any local geometric feature**:

   * **An embedded interior point:** In ``ellipse_msh_template.geo`` (Example A below), there is only **one** elliptical inclusion (``Physical Surface(2)``), yet there are **three** mesh parameters: ``lc`` on the outer unit-cell corners, ``lc2`` on the ellipse perimeter (``lc / lc2``), and ``lc3`` on an **isolated point embedded at the center of the ellipse** (``Point(5)`` via ``Point{5} In Surface{12};`` with size ``lc / lc3``) to grade the interior mesh smoothly between the boundary and the center.
   * **Individual inclusions (** ``"lc": lc_n`` **):** In a multi-resonator unit cell, you can pass as many ``lc_n`` parameters as you need (``lc2, lc3, lc4, ..., lc_n``) and assign ``"lc": lc_n`` directly inside each inclusion dictionary in :func:`meshing.unit_cell`—even when all inclusions share the same material ``"tag": 2``.
   * **Subdomain tags (** ``**lc_kwargs`` **):** If an inclusion dictionary omits ``"lc"``, :func:`meshing.unit_cell` also accepts arbitrary ``lc2=..., lc3=..., ..., lc_n=...`` keyword arguments (``**lc_kwargs``) and maps ``lc_n`` by default to subdomain ``tag = n`` (falling back to ``lc2``).

2. ``lc`` (Gmsh) vs. ``maxh`` (Netgen) — How the Two Meshers Differ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the hood, Gmsh and Netgen control mesh element sizes using different geometric primitives and terminology:

* **In Gmsh (** ``lc`` **on 0D Points):**
  Gmsh attaches a **characteristic length** (``lc``) to 0D ``Point`` entities (``Point(i) = {x, y, 0, lc_i};``) and interpolates element sizes across curves and surfaces. In ``.geo`` templates, EMUstack substitutes ``lc2 = lc/1;`` with ``lc2 = lc/val;``.
* **In Netgen (** ``maxh`` **on 1D Curves & 2D Domains):**
  Netgen's ``SplineGeometry`` does not assign characteristic lengths to points on spline curves; instead, it controls element sizes via a **maximum element size (** ``maxh`` **)** passed to 1D boundary segments (``geo.Append(..., maxh=...)``), 2D subdomains (``geo.SetDomainMaxH(tag, maxh)``), and ``geo.GenerateMesh(maxh=...)``.
* **How EMUstack Bridges** ``lc`` **and** ``maxh``:

  * **In Netgen Route B1 (** :func:`meshing.unit_cell` **):** You use EMUstack's uniform ``lc`` notation everywhere—writing ``"lc": lc2``, ``"lc": lc3``, ..., ``"lc": lc_n`` directly in each inclusion dictionary (or passing ``lc2, lc3, ..., lc_n`` via ``**lc_kwargs``)—and :func:`meshing.unit_cell` **internally converts** each factor ``lc_n`` into Netgen's ``maxh = lc / lc_n`` on both the boundary splines and the subdomain.
  * **In Netgen Route B2 (Direct** ``SplineGeometry`` **):** Because you call Netgen's native ``geo.Append(...)`` and ``geo.SetDomainMaxH(...)`` methods directly, you compute ``inc_maxh = lc / lc2`` yourself in Python and pass it to Netgen's ``maxh=`` keyword argument.

Once the mesh is generated in :func:`meshing.unit_cell`, it automatically runs **four sanity assertions**:

1. ``ngm.GetNrIdentifications() == 4`` (periodic boundary tables exist).
2. ``ngm.GetCurveOrder() >= 2`` (second-order curved elements are active).
3. ``len(mesh.GetPeriodicNodePairs(VERTEX)) > 0`` (opposite boundary vertices are paired).
4. :math:`\left|\int_{\Omega} 1\,dA - 1 \times d_y\right| < 10^{-11}` (exact unit-cell area conservation).

---

Part II: Step-by-Step Practical Examples
----------------------------------------

To reinforce the **6 Core Concepts** from Part I, we now examine three concrete examples where **each practical step (`Step 1` through `Step 6`) maps directly to `Concept 1` through `Concept 6` above**:

* **Example A (Gmsh Pipeline):** ``ellipse_msh_template.geo`` — a rotated elliptical nanohole/nanopillar unit cell in Gmsh, passing all parameters via ``geo_params``.
* **Example B1 (Netgen Route B1 — Standard Inclusions via** :func:`meshing.unit_cell` **):** ``ellipse_msh_template.py`` and ``chiral_qbic_gorkunov_msh_template.py`` — showing how standard inclusions (such as single or multiple rotated ellipses) are built effortlessly by passing an ``inclusions`` dictionary list to :func:`meshing.unit_cell`.
* **Example B2 (Netgen Route B2 — Fancier Custom Shapes via Direct** ``SplineGeometry`` **):** Walking through all 6 steps using ``netgen.geom2d.SplineGeometry`` directly when you need a geometry beyond the standard primitives of :func:`meshing.unit_cell`.

---

Example A: Step-by-Step Gmsh Template (``ellipse_msh_template.geo``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the step-by-step breakdown of ``EMUstack/backend/data/meshes/ellipse_msh_template.geo`` (for the general rotated case ``angle != 0``).

Step 1 (Concept 1 — Parameter Placeholders)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the top of ``ellipse_msh_template.geo``, every parameter that can be overridden by ``geo_params`` is initialized with a ``0`` (or ``lc/1``) placeholder:

.. code-block:: c

   // Force Gmsh to output legacy MSH 2.2 format required by Netgen ReadGmsh
   Mesh.MshFileVersion = 2.2;

   d = 1;            // Normalized unit-cell width along x
   period = 0;       // Replaced by period (in nm)        (legacy: d_in_nm)
   period_y = 0;     // Replaced by period_y (in nm)      (legacy: dy_in_nm)
   diameter1 = 0;    // Replaced by major diameter (in nm) (legacy: a1)
   ellipticity = 0;  // Replaced by (rx - ry) / rx
   angle = 0;        // Replaced by rotation angle (in radians)

   lc = 0;           // Replaced by background element size
   lc2 = lc/1;       // Replaced by lc / lc2 (on ellipse boundary)
   lc3 = lc/1;       // Replaced by lc / lc3 (at ellipse center)

When you pass ``geo_params={'period': 400.0, 'diameter1': 300.0, 'ellipticity': 0.5, 'angle': np.pi/3, 'lc': 0.08, 'lc2': 4.0, 'lc3': 2.0}``, EMUstack replaces ``period = 0;`` with ``period = 400.0;``, ``diameter1 = 0;`` with ``diameter1 = 300.0;``, ``lc2 = lc/1;`` with ``lc2 = lc/4.0;``, and so on. The ``(legacy: ...)`` comments record the old placeholder names, so ``geo_params={'d_in_nm': 400.0, 'a1': 300.0, ...}`` fills the same placeholders.

Step 2 (Concept 2 — Coordinate Normalization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, the script converts the physical dimensions ``period``, ``period_y``, and ``diameter1`` (in nanometers) into normalized unit-cell coordinates where the width along :math:`x` is ``d = 1``:

.. code-block:: c

   dy = period_y / period;                    // Normalized unit-cell height along y
   radius1 = (diameter1 / (2 * period)) * d;  // Normalized semi-major axis rx
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

To use this Gmsh template in a simulation script, pass **all** geometric and meshing parameters directly inside the ``geo_params`` dictionary:

.. code-block:: python

   import numpy as np
   from EMUstack.backend import objects, materials

   nh_layer = objects.NanoStruct(
      '2D_array',
      geo_file='ellipse_msh_template.geo',
      geo_params={
          'period': 400.0,
          'diameter1': 300.0,
          'ellipticity': 0.5,
          'angle': np.pi / 3,
          'lc': 0.08,
          'lc2': 4.0,
          'lc3': 2.0,
      },
      background=materials.Au,
      inclusion_a=materials.Air,
      height_nm=50.0,
   )

---

Example B1: Netgen Route B1 — Standard Inclusions via ``meshing.unit_cell()`` (``ellipse_msh_template.py`` & ``chiral_qbic_gorkunov_msh_template.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your unit cell consists of one or more **standard geometric primitives** (circles, rotated ellipses, squares, rectangles, concentric rings, or dimers), you do **not** need to construct a ``SplineGeometry`` by hand. Instead, you use :func:`meshing.unit_cell`, which automates **Steps 3, 4, 5, and 6** from a list of inclusion dictionaries.

Let us see how both ``ellipse_msh_template.py`` (single rotated ellipse) and ``chiral_qbic_gorkunov_msh_template.py`` (a two-ellipse quasi-BIC dimer) are implemented using this route. ``Chiral_qBIC_Optical_Chirality.ipynb`` uses ``chiral_qbic_gorkunov_c4_msh_template.py``, which builds the same kind of dimer four times per cell, rotated by 90° each time, as in the paper by Gorkunov *et al.*

Step 1 (Concept 1 — Python Function Signature & Parameter Validation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The template defines ``build_unit_cell(...)`` with explicit keyword arguments and defaults:

* **In** ``ellipse_msh_template.py`` **(single rotated ellipse):**

  .. code-block:: python

     from typing import Optional
     import ngsolve as ngs
     from EMUstack.backend import meshing

     @meshing.legacy_params(a1="diameter1")   # the old name a1 still works
     def build_unit_cell(
         period: float = 400.0,
         period_y: Optional[float] = None,
         diameter1: Optional[float] = None,
         ellipticity: Optional[float] = None,
         angle: Optional[float] = None,
         lc: float = 0.12,
         lc2: float = 5.0,
         lc3: float = 1.0,
     ) -> ngs.Mesh:

* **In** ``chiral_qbic_gorkunov_msh_template.py`` **(two tilted elliptical bars separated by a gap):**

  .. code-block:: python

     @meshing.legacy_params(theta="theta_deg", d="z_offset", l="length", w="width",
                            rot1="angle1", rot2="angle2")
     def build_unit_cell(
         period: float = 480.0,
         period_y: Optional[float] = None,
         theta_deg: float = 3.5,      # Tilt angle in degrees
         z_offset: float = 10.0,      # Vertical offset in nm
         length: float = 220.0,       # Bar length in nm (major axis 2*ry)
         width: float = 70.0,         # Bar width in nm  (minor axis 2*rx)
         gap: float = 15.0,           # Edge-to-edge gap between bars in nm
         include_bar1: bool = True,   # Toggle left bar (for 3-layer z-slicing)
         include_bar2: bool = True,   # Toggle right bar (for 3-layer z-slicing)
         angle1: Optional[float] = None,  # Bar rotations in radians
         angle2: Optional[float] = None,
         lc: float = 0.08,            # Background mesh size (fraction of period)
         lc2: float = 2.0,            # Inclusion refinement factor (inc = lc / lc2)
         lc3: float = 1.0,
         curve_order: int = 2,
         **kwargs
     ) -> ngs.Mesh:

Step 2 (Concept 2 — Coordinate Normalization to :math:`[0, 1] \times [0, d_y]`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, physical lengths in nanometers are divided by ``period`` (:math:`a`) so the unit cell spans :math:`[0, 1] \times [0, d_y]` with its center at :math:`(0.5, 0.5\,d_y)`:

* **In** ``ellipse_msh_template.py``:

  .. code-block:: python

     py = period if period_y is None else period_y
     dy = float(py) / float(period)

     rx = float(diameter1) / (2.0 * float(period))
     ry = rx * (1.0 - float(ellipticity))
     ang_rad = float(angle)
     center = (0.5, dy / 2.0)

* **In** ``chiral_qbic_gorkunov_msh_template.py``:

  .. code-block:: python

     py = period if period_y is None else period_y
     dy = float(py) / float(period)
     a = float(period)

     theta_rad = np.deg2rad(float(theta_deg))
     angle1 = +0.5 * theta_rad if angle1 is None else angle1
     angle2 = -0.5 * theta_rad if angle2 is None else angle2

     rx = 0.5 * float(width) / a
     ry = 0.5 * float(length) / a
     sep = (float(width) + float(gap)) / a
     c1 = (0.5 - 0.5 * sep, 0.5 * dy)
     c2 = (0.5 + 0.5 * sep, 0.5 * dy)

Steps 3, 4, 5, & 6 (Concepts 3–6 — Automated by ``meshing.unit_cell()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because both templates use standard elliptical inclusions, **Steps 3 through 6 require only a few lines of Python**:

1. Build the ``inclusions`` list of dictionaries—where each dictionary specifies the shape ``"kind"`` (**Step 4**), the material ``"tag"`` (**Step 5**), and the inclusion's mesh-scaling factor ``"lc": lc_n`` (**Step 6**, which :func:`meshing.unit_cell` converts internally to Netgen's ``maxh = lc / lc_n``):

   * **In** ``ellipse_msh_template.py``:

     .. code-block:: python

        inclusions = [{
            "kind": "ellipse",
            "center": center,
            "rx": rx,
            "ry": ry,
            "angle": ang_rad,
            "tag": 2,          # Maps to inclusion_a
            "lc": lc2,         # Converted internally to maxh = lc / lc2
        }]

   * **In** ``chiral_qbic_gorkunov_msh_template.py``:

     .. code-block:: python

        inclusions = []
        if include_bar1:
            inclusions.append({
                "kind": "ellipse", "center": c1, "rx": rx, "ry": ry,
                "angle": float(angle1), "tag": 2, "lc": lc2,
            })
        if include_bar2:
            inclusions.append({
                "kind": "ellipse", "center": c2, "rx": rx, "ry": ry,
                "angle": float(angle2), "tag": 2, "lc": lc2,
            })

2. Call :func:`meshing.unit_cell` with ``inclusions``, ``dy=dy``, and the background mesh size ``lc=lc`` (**Steps 3 & 6**):

   .. code-block:: python

      return meshing.unit_cell(inclusions, dy=dy, lc=lc)

   .. tip::
      **Using an arbitrary number of** ``lc_n`` **parameters (** ``"lc": lc_n`` **and** ``**lc_kwargs`` **):**

      * You can define as many ``lc_n`` parameters (``lc2, lc3, lc4, ..., lc_n``) in ``build_unit_cell(...)`` and ``geo_params`` as you have inclusions, and map ``"lc": lc_n`` directly inside each inclusion dictionary (where :math:`\text{lc}_n > 1` refines and :math:`0 < \text{lc}_n < 1` coarsens relative to ``lc``):

        .. code-block:: python

           inclusions = [
               {"kind": "ellipse", "center": c1, "rx": rx1, "ry": ry1, "tag": 2, "lc": lc2},
               {"kind": "circle",  "center": c2, "radius": r2,         "tag": 2, "lc": lc3},
               {"kind": "square",  "center": c3, "side": s3,           "tag": 3, "lc": lc4},
           ]
           return meshing.unit_cell(inclusions, dy=dy, lc=lc)

      * Alternatively, if your inclusions have distinct tags (``"tag": 2, 3, ..., n``) and omit the ``"lc"`` key in their dictionaries, you can pass ``lc2=lc2, lc3=lc3, lc4=lc4, ...`` directly as keyword arguments (``**lc_kwargs``) to :func:`meshing.unit_cell`.

What ``meshing.unit_cell()`` Supports vs. When to Take Route B2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All standard inclusion shapes supported out of the box by :func:`meshing.unit_cell` are listed below (old short keys are shown in brackets and still work):

.. list-table:: Standard Inclusion Shapes Supported by ``meshing.unit_cell()``
   :header-rows: 1
   :widths: 18 42 40

   * - ``"kind"``
     - Required / Optional Keys
     - Description
   * - ``"circle"``
     - ``"center": (xc, yc)`` [``"c"``], ``"radius": float`` [``"r"``], ``"tag": int``, ``"lc": lc_n``
     - Circular inclusion of normalized radius ``radius`` built from 24 ``spline3`` arcs.
   * - ``"ellipse"``
     - ``"center": (xc, yc)`` [``"c"``], ``"rx": float``, ``"ry": float``, ``"angle": rad`` or ``"angle_deg": deg`` [``"rot"``], ``"tag": int``, ``"lc": lc_n``
     - Elliptical inclusion with semi-axes ``rx``, ``ry`` rotated counter-clockwise by ``angle`` (radians).
   * - ``"square"`` / ``"rectangle"``
     - ``"center": (xc, yc)`` [``"c"``], ``"width": float``, ``"height": float`` [``"w"``, ``"h"``] or ``"side": float`` [``"s"``], ``"tag": int``, ``"lc": lc_n``
     - Axis-aligned rectangular or square inclusion centered at ``center``.
   * - ``"ring"``
     - ``"center": (xc, yc)`` [``"c"``], ``"radius_out": float``, ``"radius_in": float`` [``"r_out"``, ``"r_in"``], ``"tag": int``, ``"lc": lc_n``
     - Annular ring between inner radius ``radius_in`` (background) and outer radius ``radius_out``.
   * - ``"dimer"``
     - ``"center": (xc, yc)`` [``"c"``], ``"sub_kind": "circle"|"square"``, ``"radius": float`` [``"r"``], ``"gap": float``, ``"tag": int``, ``"lc": lc_n``
     - Symmetric pair of inclusions separated along :math:`x` by edge-to-edge ``gap``.

.. important::
   **The Exact Role and Boundary of** :func:`meshing.unit_cell`:

   * **Use Route B1 (** :func:`meshing.unit_cell` **)** whenever your unit cell consists of one or more non-overlapping inclusions drawn from the 6 standard shapes in the table above (in any combination of positions ``center``, sizes, rotation angles, and material tags ``2..6``).
   * **Use Route B2 (Direct** ``SplineGeometry`` **, shown next)** whenever you need a **fancier custom inclusion** that is *not* in the table above—for example, an L-shaped resonator, a split-ring with an angular cut, a crescent, a triangle/hexagonal prism, a rotated rectangle, or freeform splines.

Calling ``ellipse_msh_template.py`` and ``chiral_qbic_gorkunov_msh_template.py`` from ``NanoStruct``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run ``ellipse_msh_template.py``, you pass the **exact same** ``geo_params`` dictionary as in the Gmsh example—simply changing ``.geo`` to ``.py``:

.. code-block:: python

   import numpy as np
   from EMUstack.backend import objects, materials

   nh_layer = objects.NanoStruct(
       '2D_array',
       geo_file='ellipse_msh_template.py',
       geo_params={
           'period': 400.0,
           'diameter1': 300.0,
           'ellipticity': 0.5,
           'angle': np.pi / 3,
           'lc': 0.08,
           'lc2': 4.0,
           'lc3': 2.0,
       },
       background=materials.Au,
       inclusion_a=materials.Air,
       height_nm=50.0,
   )

Similarly, to build the chiral qBIC dimer layer with ``chiral_qbic_gorkunov_msh_template.py`` and plot the triangulated mesh:

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.tri as mtri
   from EMUstack.backend import objects, materials

   period = 480.0
   lay2 = objects.NanoStruct(
       '2D_array',
       geo_file='chiral_qbic_gorkunov_msh_template.py',
       geo_params={
           'period': period,
           'theta_deg': 3.5,
           'include_bar1': True,
           'include_bar2': True,
           'lc': 0.15,
           'lc2': 1.5,
       },
       background=materials.Material(1.0),
       inclusion_a=materials.Material(4.0 + 0.01j),
       height_nm=90.0,
   )

   # Visually inspect the generated mesh and material tags (1 = bg, 2 = inc2)
   x_nm = lay2.x_arr[0] * period
   y_nm = lay2.x_arr[1] * period
   triangles = lay2.table_nod[:3].T - 1
   triang = mtri.Triangulation(x_nm, y_nm, triangles)

   fig, ax = plt.subplots(figsize=(5, 5))
   ax.tripcolor(triang, facecolors=lay2.type_el, cmap='coolwarm', edgecolors='k', lw=0.4)
   ax.set_aspect('equal')
   ax.set_xlabel('x (nm)')
   ax.set_ylabel('y (nm)')
   ax.set_title(f'Chiral qBIC Unit Cell ({lay2.n_msh_pts} vertices, {lay2.n_msh_el} triangles)')

---

Example B2: Netgen Route B2 — Fancier Custom Shapes via Direct ``SplineGeometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you need a **fancier resonator geometry** that is not one of the standard primitives in :func:`meshing.unit_cell` (or if you want to understand what :func:`meshing.unit_cell` does under the hood), your ``build_unit_cell(...)`` function constructs a ``netgen.geom2d.SplineGeometry`` directly, performing **all 6 steps** explicitly in Python.

Below is the step-by-step walkthrough of how to build a custom ``SplineGeometry`` template (illustrated for an **L-shaped nanoantenna**, along with how curved ``"spline3"`` arcs are added).

Step 1 (Concept 1 — Python Function Signature)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define ``build_unit_cell(...)`` in your custom ``.py`` file (e.g., ``l_antenna_template.py``) with your custom geometric parameters and mesh controls:

.. code-block:: python

   import netgen.geom2d as g2
   import ngsolve as ngs
   from ngsolve.fem import NODE_TYPE

   def build_unit_cell(
       period: float = 500.0,
       period_y: float = 500.0,
       arm_len: float = 300.0,
       arm_width: float = 100.0,
       lc: float = 0.10,
       lc2: float = 2.5,
       curve_order: int = 2,
   ) -> ngs.Mesh:

Step 2 (Concept 2 — Coordinate Normalization to :math:`[0, 1] \times [0, d_y]`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Divide all physical dimensions in nanometers by ``period`` so the geometry lives inside :math:`[0, 1] \times [0, d_y]`:

.. code-block:: python

       dy = float(period_y) / float(period)
       L_arm = float(arm_len) / float(period)
       W_arm = float(arm_width) / float(period)
       inc_h = lc / lc2  # Convert EMUstack's (lc, lc2) into Netgen's maxh

Step 3 (Concept 3 — Outer Unit-Cell Box & Periodic ``copy=`` Boundaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate ``geo = g2.SplineGeometry()``, add the 4 corners of the normalized unit cell, and append the primary ``left`` and ``bottom`` edges followed by their periodic copies ``right`` (``copy=i_l``) and ``top`` (``copy=i_b``):

.. code-block:: python

       geo = g2.SplineGeometry()

       # 1. Add the 4 corner points of [0, 1] x [0, dy]
       P = [geo.AddPoint(*q) for q in [(0, 0), (1.0, 0), (1.0, dy), (0, dy)]]

       # 2. Periodic x-boundaries: Left (P0 -> P3) is primary; Right (P1 -> P2) is its copy
       i_l = geo.Append(["line", P[0], P[3]], leftdomain=0, rightdomain=1, bc="left", maxh=lc)
       geo.Append(["line", P[1], P[2]], leftdomain=1, rightdomain=0, bc="right", maxh=lc, copy=i_l)

       # 3. Periodic y-boundaries: Bottom (P0 -> P1) is primary; Top (P3 -> P2) is its copy
       i_b = geo.Append(["line", P[0], P[1]], leftdomain=1, rightdomain=0, bc="bottom", maxh=lc)
       geo.Append(["line", P[3], P[2]], leftdomain=0, rightdomain=1, bc="top", maxh=lc, copy=i_b)

.. note::
   Always ensure that ``left`` and ``right`` are traversed in the **same direction** (both bottom-to-top: ``P[0]->P[3]`` and ``P[1]->P[2]``), and that ``bottom`` and ``top`` are traversed in the **same direction** (both left-to-right: ``P[0]->P[1]`` and ``P[3]->P[2]``).

Step 4 (Concept 4 — Drawing the Custom Inclusion with ``"line"`` and ``"spline3"``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the vertices (and any spline control points) of your custom resonator in **counter-clockwise order**, and connect them with ``geo.Append(...)``:

* **For straight polygon edges (such as the 6 vertices of our L-shaped antenna):**
  Use ``["line", start_pt, end_pt]``:

  .. code-block:: python

       x0, y0 = 0.5 - L_arm / 2.0, 0.5 * dy - L_arm / 2.0
       pts_L = [
           (x0,         y0),
           (x0 + L_arm, y0),
           (x0 + L_arm, y0 + W_arm),
           (x0 + W_arm, y0 + W_arm),
           (x0 + W_arm, y0 + L_arm),
           (x0,         y0 + L_arm),
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

* **For curved boundary arcs (e.g., rounded corners, crescents, or split rings):**
  Use ``["spline3", pt0, pt_ctrl, pt1]``, where ``pt0`` and ``pt1`` are the arc endpoints and ``pt_ctrl`` is the off-curve tangent intersection control point. For circular or elliptical arcs spanning angle :math:`[a_0, a_1]`, you can call the built-in helper :func:`meshing.arc_control_points` directly:

  .. code-block:: python

       # Example: appending a curved circular/elliptical arc from angle a0 to a1
       pt0, pt_ctrl, pt1 = meshing.arc_control_points(
           geo, kind="circle", c=(0.5, 0.5 * dy), a0=a0, a1=a1, r=radius
       )
       geo.Append(["spline3", pt0, pt_ctrl, pt1], leftdomain=2, rightdomain=1, bc="inc2", maxh=inc_h)

Step 5 (Concept 5 — Oriented Domain Tagging ``leftdomain`` / ``rightdomain`` & Materials)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the vertices ``Q[0] -> Q[1] -> ... -> Q[5] -> Q[0]`` wind **counter-clockwise** around the L-antenna, the interior of the antenna is always to the left of each segment (``leftdomain=2``) and the background is to the right (``rightdomain=1``).

Next, assign the domain material names and maximum element sizes (`"bg"` for domain ``1`` and `"inc2"` for domain ``2``, or `"inc3".."inc6"` if you have multiple distinct inclusion materials):

.. code-block:: python

       geo.SetMaterial(1, "bg")
       geo.SetDomainMaxH(1, lc)
       geo.SetMaterial(2, "inc2")
       geo.SetDomainMaxH(2, inc_h)

Step 6 (Concept 6 — Mesh Generation, Curving ``ngm.Curve(2)``, & Calling from ``NanoStruct``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, generate the mesh, curve any spline boundaries to polynomial order ``curve_order=2``, and return the ``ngsolve.Mesh``:

.. code-block:: python

       ngm = geo.GenerateMesh(maxh=lc)
       ngm.Curve(curve_order)
       mesh = ngs.Mesh(ngm)

       # Optional sanity checks
       assert ngm.GetNrIdentifications() == 4, "periodic identifications missing"
       assert len(mesh.GetPeriodicNodePairs(NODE_TYPE.VERTEX)) > 0
       return mesh

You can then invoke ``l_antenna_template.py`` from ``objects.NanoStruct`` using ``geo_params`` in the exact same way as any other template:

.. code-block:: python

   from EMUstack.backend import objects, materials

   l_layer = objects.NanoStruct(
       '2D_array',
       geo_file='l_antenna_template.py',
       geo_params={
           'period': 500.0,
           'arm_len': 320.0,
           'arm_width': 90.0,
           'lc': 0.08,
           'lc2': 3.0,
       },
       background=materials.Air,
       inclusion_a=materials.Au,
       height_nm=60.0,
   )

---

``meshing`` Module API Reference
--------------------------------

.. automodule:: meshing
    :members:
    :undoc-members:
    :show-inheritance:
