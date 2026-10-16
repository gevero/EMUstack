Installation
================

.. contents::
   :local:

Which backend are you installing?
--------------------------------------

EMUstack solves a Bloch eigenproblem in each nanostructured layer. The legacy Fortran
backend has been completely replaced by modern pure Python and NGSolve backends. Everything else —
the plane-wave Resolutes, the Redheffer stack recursion, the public API — is preserved verbatim.

Installation is pure Python via ``pip install -e .`` within an environment providing ``ngsolve`` and standard scientific Python packages.

================================== ================================================================== ========================================================
backend                            2D FEM engine                                                      status
================================== ================================================================== ========================================================
**Legacy Fortran**                 hand-written ``P2``/``P3`` elements + bundled ARPACK + UMFPACK     **Purged**: completely removed from repository.
**NGSolve (2D)**                   ``ngsolve`` ``HCurl`` (Nedelec) x ``H1`` product space,            **Active**: pure Python/NGSolve 2D solver with
                                   built-in ``ArnoldiSolver`` + ``Periodic(phase=...)``               native Netgen and Gmsh MSH 2.2 import.
**Pure Python (1D)**               Coupled pencil on ``dP2(E_x) ⊕ P3(E_y) ⊕ P3(Ê_z)``                 **Active**: pure Python sparse CSR + SuperLU solver.
**Option B - FEniCSx**             ``dolfinx`` mixed ``N1curl`` x Lagrange space, UFL forms,          installable and API-verified on 0.11.0, but
                                   SLEPc eigenproblem via ``petsc4py``/``slepc4py`` (MPI + PETSc)     **not selected** and **not** specified: sections 5,
                                                                                                      6 and 9 of the plan would be rewritten, see
                                                                                                      'Differences between the two ports'
================================== ================================================================== ========================================================

.. note::

   Where ``PORTING_PLAN_EMUSTACK_FEM.md`` is referred to below, that is the backend-port
   specification that accompanies this checkout in the porting workspace (next to the ``EMUstack/``
   directory, not inside it). It is written for Option A. As the port lands, its sections are meant to
   move into this documentation tree (``docs/source/fem_2d.rst`` currently still documents the
   Fortran internals that the port deletes).

The 1D grating solver is pure Python/NumPy/SciPy in **both** options — no FEM library is required
for 1D, and no compiled extension at all is required after the port.

You install Option A (plus, optionally, the legacy backend for reference regeneration). Do not try to
install both options into one environment: Option B pins PETSc/SLEPc and NumPy 2 in ways that Option
A's dependency resolution does not expect.


Common requirements (both options)
--------------------------------------

* Python 3.11 or 3.12 (the tested interpreters are 3.11.10 and 3.12.3).
* NumPy ≥ 1.26, SciPy ≥ 1.11, Matplotlib ≥ 3.8.
* Under NumPy ≥ 2.0 the legacy code cannot be imported: ``np.mat``/``np.matrix`` were removed in
  NumPy 2.0. Both ports modernise this (§4.5 of ``PORTING_PLAN_EMUSTACK_FEM.md``); the legacy
  backend simply requires NumPy 1.x.
* ``gmsh`` **is** used under Option A: all existing examples keep their ``.geo`` templates and are
  meshed with ``gmsh -2 -format msh2 -order 1`` and then imported (§"Geometry and meshing under
  Option A"). New examples build geometry natively in netgen/OpenCASCADE instead. Option B needs
  ``gmsh`` for meshing in practice too.
* No Fortran compiler and no MPI in either port.

Verify your interpreter before installing anything::

    $ python -c "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"


Option A — NGSolve backend
--------------------------------------

Install

.. code-block:: console

    $ python -m venv ~/.venvs/emustack-ngs && source ~/.venvs/emustack-ngs/bin/activate
    $ pip install "ngsolve>=6.2.2608" "netgen-mesher>=6.2.2608" numpy scipy matplotlib
    $ pip install pytest            # optional mkl only if you want inverse="pardiso"

There is **no compiler step, no MPI, no PETSc**. Everything is a binary wheel. The platform matrix
of the pinned release (read from the PyPI file list) is:

.. code-block:: text

    ngsolve 6.2.2608: cp310 cp311 cp312 cp313 cp314
      macosx_10_15_universal2   (includes Apple Silicon)
      manylinux_2_28_x86_64
      win_amd64
      -> NO linux_aarch64 wheel. On ARM Linux, use Option B.

Licenses: ``ngsolve`` and ``netgen-mesher`` are LGPL-2.1-only, compatible with EMUstack's GPL-3.0.

Mandatory verification (every line was executed on 6.2.2608 before it was written into the porting
plan; if this snippet fails, stop — the backend you got is not the backend the code is written for):

.. code-block:: python

   import numpy as np, netgen.geom2d as g2, ngsolve as ngs
   from ngsolve import Mesh, HCurl, H1, GridFunction

   geo = g2.SplineGeometry()
   P = [geo.AddPoint(*q) for q in [(0,0), (1,0), (1,0.8), (0,0.8)]]
   ib = geo.Append(["line", P[0], P[1]], bc="bottom", maxh=0.2)
   ir = geo.Append(["line", P[1], P[2]], bc="right",  maxh=0.2)
   geo.Append(["line", P[2], P[3]], bc="top",   maxh=0.2, copy=ib)     # identifications
   geo.Append(["line", P[3], P[0]], bc="left",  maxh=0.2, copy=ir)
   ngm = geo.GenerateMesh(maxh=0.2); ngm.Curve(3)                       # Curve() is NOT optional
   mesh = Mesh(ngm)
   assert ngm.GetNrIdentifications() == 4
   assert ngm.GetCurveOrder() >= 2

   kx, ky, dy = 0.31, -0.17, 0.8
   fes = HCurl(mesh, order=2, complex=True) * H1(mesh, order=3, complex=True)
   sp = ngs.Periodic(fes, phase=[np.exp(-1j*kx), np.exp(-1j*ky*dy)])    # the Bloch space
   assert sp.FreeDofs().NumSet() < fes.ndof                             # periodicity really applied
   (u, z), (v, w) = sp.TnT()
   K = ngs.BilinearForm(sp, symmetric=False)
   K += (ngs.curl(u)*ngs.curl(v) - u[0]*ngs.Conj(v[0]) - u[1]*ngs.Conj(v[1]))*ngs.dx
   K.Assemble()
   gfs = [GridFunction(sp) for _ in range(4)]
   lam = np.array(ngs.ArnoldiSolver(K.mat, K.mat, sp.FreeDofs(), [g.vec for g in gfs], shift=1.0))
   print("OK ngsolve", ngs.__version__ if hasattr(ngs, "__version__") else "", len(lam))

Note the deliberate oddity: ``ArnoldiSolver(a, a, …)`` is used here only to prove the call surface.
The real pencil is in §6.2 of the porting plan. What this snippet pins are the four things that fail
silently otherwise: mesh identifications, ``Curve(order)``, ``Periodic`` reducing the free-DOF set,
and eigenvector extraction through ``GridFunction(...).vec`` (``bla.Vector``/``VectorC`` are rejected
by ``ArnoldiSolver``).

Measured cost on this class of machine (``HCurl(2) × H1(3)`` periodic product space, one circular
inclusion, single process) — the numbers the port's performance gates are written against:

.. code-block:: text

    ndof  numeig  npw    assemble   factor    Arnoldi   overlaps   total
     1597     20    40      0.006 s   0.479 s   0.669 s    0.253 s  1.42 s
     2419     40   113      0.007 s   0.747 s   2.207 s    2.151 s  5.13 s
     4369     40   113      0.032 s   2.435 s   4.967 s    2.316 s  9.78 s


Geometry and meshing under Option A
`````````````````````````````````````

Option A has two sanctioned geometry paths. Both end in a netgen mesh wrapped by ``ngsolve.Mesh``;
they differ in who produces the mesh and, more importantly, in how the Bloch condition is imposed.

**Path I -- native netgen geometry.** Used by new examples and by every physics regression test.
``netgen.geom2d.SplineGeometry`` with ``copy=<spline index>`` on the two periodic edge pairs, then
``ngm.Curve(order)`` (assert ``GetCurveOrder() >= 2``), then ``ngsolve.Mesh``, then
``ngsolve.Periodic(space, phase=[...])``. CAD-style shapes go through ``netgen.occ`` instead, where
faces and regions are named by us, so ``mesh.MaterialCF({...})`` works directly. This is the only path
on which ``Periodic`` does anything; the assertions ``ngm.GetNrIdentifications() == 4`` and
``Periodic(...).FreeDofs().NumSet() < space.ndof`` are mandatory on it.

**Path II -- gmsh, MSH 2.2.** Used by the existing examples, whose ``.geo`` templates and checked-in
meshes are gmsh artefacts::

    $ gmsh -2 cell.geo -format msh2 -order 1 -o cell.msh2.msh

and imported as follows (every line below was executed on ngsolve 6.2.2608 / gmsh 4.9.3):

.. code-block:: python

    import ngsolve as ngs
    from netgen.read_gmsh import ReadGmsh

    ngm = ReadGmsh("cell.msh2.msh")      # named physical groups required
    ngm.dim = 2                          # mandatory: imported planar meshes arrive as dim == 3
    ngm.Curve(2)                         # curvature, since the mesh is linear
    mesh = ngs.Mesh(ngm)
    print(mesh.dim, list(ngm.GetRegionNames(dim=2)))
    # 2 ['bg', 'inc', 'top', 'left', 'inc', 'inc', 'inc', 'inc']
    #   ^ region and boundary names share this inventory; read GetCD2Name/GetBCName separately

The four requirements above are measured, not stylistic:

* **``-format msh2``**: ``ReadGmsh`` cannot parse MSH 4.x at all
  (``ValueError: invalid literal for int() with base 10: '10 952 1 952'``), and it needs *named*
  physical groups -- numeric-only groups make it raise ``KeyError: 101``. With named groups the
  region names survive the import and ``MaterialCF`` resolves them.
* **``-order 1``**: importing 6-node (``P2``) triangles corrupts the DOF mapping on its own --
  measured on such an import, an ``H1(order=1)`` space lost 1036 of 1497 dofs with *no* periodic
  identifications present at all. Linear meshes pay an ``O(h^2)`` geometry error instead (measured on
  a 1 x 0.8 cell with a ``r = 0.1`` circle: +2.6e-4 relative on the cell area, -6.4e-3 relative on the
  inclusion area), which the port refines rather than argues about.
* **``ngm.dim = 2``**: left at 3, ``HCurl`` assembly dies with
  ``NgException: BaseVector::operator/=: division by zero``. ``SetDimensions()`` and
  ``Mesh.Use2D3D`` do not exist; setting ``dim`` on an already-wrapped ``ngsolve.Mesh`` does not fix
  the wrapper, so do it before constructing it. Measured refinement: the ``dim == 3`` state came from
  the ``P2`` import -- a ``-order 1`` import already reports ``dim == 2`` -- but the assert stays,
  because that is an accident of this version and not a documented guarantee.
* **no ``ngsolve.Periodic`` on Path II**: ``IdentifyPeriodicBoundaries`` reports success but marks
  almost every identification ``UNDEFINED`` (measured: exactly **1** ``PERIODIC`` of 41), so the
  quasi-periodic space silently stops being a Bloch space -- and the manual
  ``AddPointIdentification`` route segfaults the interpreter unless native geometry allocated the
  identification table. Path II imposes the Bloch phase as an explicit constraint operator on the
  assembled pencil instead (§5.9 of ``PORTING_PLAN_EMUSTACK_FEM.md``).

.. note::

   ``netgen.occ.LoadOCCGeometry("cell.step").GenerateMesh(maxh=0.06)`` also works and keeps exact
   geometry, which removes every spline-Bézier pitfall -- but gmsh 4.9.3 writes **no** physical-group
   names into STEP (the file contains zero occurrences of the group names; the loaded faces report
   ``name is None``), so every region arrives as ``'default'`` and materials must be recovered by
   geometric predicates. ``gmsh``'s BREP export is not loadable by this netgen
   (``Couldn't load OCC geometry``), and ``.unv`` / ``.vtk`` / ``.bdf`` are not readable by
   ``ngsolve.Mesh`` at all.


Option B — FEniCSx backend
--------------------------------------

Install (conda-forge, the vendor-recommended route; verified on this machine)

.. code-block:: console

    $ conda create -y -n emustack-fenicsx -c conda-forge \
          python=3.11 fenics-dolfinx mpi4py petsc4py slepc4py
    $ conda activate emustack-fenicsx

Debian/Ubuntu alternative (vendor-documented)

.. code-block:: console

    $ sudo add-apt-repository ppa:fenics-packages/fenics
    $ sudo apt update && sudo apt install fenicsx
    # complex-valued build (required here — see below):
    $ sudo apt install python3-dolfinx-complex
    $ PETSC_DIR=/usr/lib/petscdir/petsc-complex \
      SLEPC_DIR=/usr/lib/slepcdir/slepc-complex  python3 your_run.py

Other routes: Docker images from the FEniCS project, Spack, or a source build (C++20 compiler,
Basix, Boost, CMake ≥ 3.26, HDF5 built with MPI, MPI-3+). A ``pip`` wheel index
(``wheels.fenicsproject.org``) exists upstream but did **not** resolve by DNS on this machine, so do
not build the install story around ``pip install fenics-dolfinx``.

What a working install looks like (measured in this env, ``linux-64``):

.. code-block:: text

    dolfinx 0.11.0   basix 0.11.0   ffcx 0.11.0   ufl 2026.1.0
    mpi4py 4.1.2     petsc4py 3.25.6   slepc4py 3.25.2
    numpy 2.4.6      python 3.11.17        total env size ≈ 1.9 GB

License: ``fenics-dolfinx`` is LGPL-3.0-or-later (compatible with EMUstack's GPL-3.0).

**Complex scalar type is a build-time choice, not a runtime flag.** A default install gives
``PETSc.ScalarType == numpy.float64``, and assembling a form with a complex coefficient fails::

    ValueError: Unexpected complex value in real expression.

``dolfinx.fem.functionspace`` has no ``dtype``/``complex`` keyword. EMUstack needs complex numbers
(the Bloch phase, the complex propagation constant ζ, complex eigenvectors), so Option B must be
installed against a complex-PETSc/SLEPc build (``petsc-complex``/``slepc-complex`` on conda-forge,
``python3-dolfinx-complex`` + ``PETSC_DIR`` on Debian/Ubuntu) and every test runner, notebook kernel
and CI job must select it. A mixed real/complex environment produces confusing failures, not clean
errors.

Mandatory verification, in increasing order of "this is where an Option B port dies"

.. code-block:: python

   import numpy as np, dolfinx, dolfinx.fem, dolfinx.fem.petsc, basix.ufl as bu, ufl
   import mpi4py.MPI as MPI
   from petsc4py import PETSc

   mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,
                                        [np.array([0.,0.]), np.array([1.,0.8])], [8, 6])
   V = dolfinx.fem.functionspace(mesh, ("N1curl", 1))          # Nédélec 1st kind: OK (158 dofs)
   E1, E2 = bu.element("N1curl", "triangle", 1), bu.element("P", "triangle", 1)
   W = dolfinx.fem.functionspace(mesh, bu.mixed_element([E1, E2]))
   print("mixed Nédélec × Lagrange:", W.num_sub_spaces, "subspaces")      # -> 2, OK

   print("scalar type:", PETSc.ScalarType)                      # MUST be complex128 for EMUstack
   print("periodic support:", [a for a in dir(dolfinx.fem)
                               if "period" in a.lower() or "scroll" in a.lower()])
   import dolfinx.io
   print("gmsh input:", hasattr(dolfinx.io, "gmsh"), "| VTX:", dolfinx.io.has_adios2)
   import slepc4py.SLEPc as S
   print("ST types:", [m for m in dir(S.ST.Type) if not m.startswith("_")])   # SINVERT, not SHIFTINVERT

Verified outcomes of that snippet on dolfinx 0.11.0, and what each one means for the port:

* ``("N1curl", 1)`` and ``("N1curl", 2, (2,))`` element specifiers are accepted; ``("Nedelec", 1)``
  is **not** (``ValueError: Unknown element family``).
* The mixed Nédélec × Lagrange space the formulation needs is constructed with
  ``basix.ufl.mixed_element`` → ``dolfinx.fem.functionspace`` (``num_sub_spaces == 2``).
  ``basix.ufl.blocked_element`` **rejects** it: ``ValueError: Cannot create a blocked element
  containing a non-scalar element`` — so an Option B port has only the *unblocked* (interleaved dof)
  layout and cannot get the contiguous ``[transverse | longitudinal]`` block layout that makes the
  mode-vector extraction easy in Option A. Assembly of a curl–curl form on the mixed space was
  **not** verified in this session: it is the first thing to verify in an Option B port.
* ``dir(dolfinx)`` contains no eigen-solver: there is **no built-in eigensolver**. You write the
  SLEPc driver yourself (``EPS`` + ``ST`` with ``S.ST.Type.SINVERT`` — the member is spelled
  ``SINVERT``, there is no ``SHIFTINVERT`` in slepc4py 3.25.2), plus ``KSP``/``PF`` plumbing for the
  shift-invert factorisation, because ``PETSc.Mat.factorizer`` does not exist in petsc4py 3.25.6.
* Mesh input arrives through ``dolfinx.io.gmsh`` (there is **no** ``dolfinx.io.gmshio`` in 0.11), and
  output through ``VTKFile``/``XDMFFile``/``VTXWriter`` (``VTXWriter`` needs ADIOS2).
* Churn warning, measured on one minor release: ``FunctionSpace.dim`` is gone (dof counts come from
  ``V.dofmap.index_map``), ``dolfinx.io.gmshio`` is gone, ``ufl.MixedElement`` /
  ``basix.create_mixed_element`` are gone, and ``dolfinx.fem.dirichletbyscrolling`` does not exist in
  this build. Option B therefore pins ``dolfinx==0.11.*`` and treats an upgrade as a project.

**The hard part: quasi-periodicity.** There is no periodic or quasi-periodic boundary condition in
this dolfinx build at any layer — the string ``periodic`` does not appear in the installed Python
package, and ``strings`` finds zero matches in the compiled core. A FEniCS-based Bloch solver must
therefore implement the Bloch condition itself: geometrically identify the dofs on the two cell
boundaries, orient them consistently, and eliminate/expand them with the phase factor
``exp(i k·d)`` — for Nédélec elements, whose degrees of freedom are *tangential moments*, whose sign
follows edge orientation, so the identification must account for orientation or the constraint is
silently wrong. (This difficulty is corroborated in the upstream discussion of periodic constraints
for Nédélec spaces and the long-standing open feature request for topologically periodic meshes;
neither is something you can paper over with a penalty term at the accuracy EMUstack needs.)


Differences between the two port approaches
---------------------------------------------------------------

The two options are not two installers for the same design. They relocate *where the risk and the
engineering work live*.

================================ ===========================================================================
concern                          Option A (NGSolve) vs Option B (FEniCSx)
================================ ===========================================================================
Bloch quasi-periodicity          **A:** one primitive, ``ngs.Periodic(fes, phase=[e^{-ikx}, e^{-iky·dy}])``,
                                 verified to reduce the free-DOF set. **B:** nothing built in; you write DOF
                                 identification + orientation handling + constraint elimination for Nédélec
                                 moments. *This is the single largest difference between the two ports.*
Complex arithmetic               **A:** per-space ``complex=True``, native. **B:** build-time PETSc/SLEPc scalar
                                 type; you must install and select the complex variant, and every runner,
                                 notebook kernel and CI job has to pick it up.
Mesh and geometry                **A:** native ``SplineGeometry`` with ``copy=`` identifications and an explicit
                                 ``Curve(order)`` call; the port must *rebuild* every built-in shape and the
                                 shipped ``.geo`` templates become dead weight. **B:** keeps ``gmsh`` as the
                                 mesher, so the existing ``.geo`` templates and the ``custom``/``geo_file``
                                 user paths survive — a genuine, and the main, advantage of B. The price is
                                 that periodicity is no longer a mesh feature you can encode (``copy=``) and
                                 must be reconstructed geometrically by matching coordinates.
Eigensolver                      **A:** ``ngs.ArnoldiSolver(K, M, freedofs, vecs, shift=σ)`` — one call,
                                 documented semantics (returns eigenvalues nearest ``σ``, Krylov dimension
                                 ``2·numeig+1``), inverse selectable (``umfpack``, ``sparsecholesky``,
                                 ``pardiso`` via optional ``mkl``). **B:** hand-written SLEPc driver
                                 (``EPS``+``ST/SINVERT``+``KSP``/``PF`` reuse), which is also where the
                                 singular-mass-pencil handling (``M`` singular ⇒ infinite eigenvalues) has
                                 to be reasoned about explicitly rather than filtered afterwards.
Mixed product space              **A:** ``HCurl × H1`` product space; vectors are stored ``[HCurl | H1]``, which
                                 the port uses to pull the transverse field out for the plane-wave moments.
                                 **B:** ``basix.ufl.mixed_element`` gives an unblocked mixed space with
                                 interleaved dofs; blocked layout is refused for Nédélec elements
                                 (``Cannot create a blocked element containing a non-scalar element``).
                                 Different extraction code, different mode-vector→``sol1`` mapping.
Eigenvector / ``sol1`` mapping   Both backends permute dofs differently from the legacy ``.mail`` ordering,
                                 so the golden ``sol1`` node-order arrays must be captured per backend. B's
                                 interleaved layout makes the legacy field-array emulation slightly harder;
                                 A's block layout is the easier mapping of the two.
Field post-processing            **A:** NGSolve ``GridFunction`` + coefficient algebra; ``.pos``/``.geo`` legacy
                                 and both options' plotting rewrites need a writer of their own.
Parallelism                      **A:** shared-memory threads inside one process; tests are ordinary ``pytest``.
                                 **B:** MPI-first. Distributed memory is available if you ever need 3D at
                                 scale, but ``pytest`` + notebooks + containers all inherit ``mpiexec -n 1``,
                                 MPI-enabled HDF5, and FFCx JIT compilation with its own cache directory
                                 (first-call latency, ``--check``/cache handling in CI).
Dependency footprint             **A:** a few wheels, no toolchain, no MPI. **B:** ≈1.9 GB conda env
                                 (PETSc, SLEPc, MPI, HDF5/ADIOS2, FFCx, Basix, UFL).
Platform coverage                **A:** ``macosx_universal2`` (incl. Apple Silicon), ``manylinux_2_28_x86_64``,
                                 ``win_amd64``; **no linux-aarch64**. **B:** conda-forge covers more
                                 platforms than pip wheels do — check the current subdir list for
                                 ``fenics-dolfinx`` before committing if ARM Linux matters to you.
Version-churn risk               **A:** the API surface used by the port has been stable across the pinned
                                 releases. **B:** measured churn within one minor release (removed
                                 ``FunctionSpace.dim``, ``dolfinx.io.gmshio``, ``ufl.MixedElement``; missing
                                 ``dolfinx.fem.dirichletbyscrolling``). Pin tightly.
What the porting plan reuses     Sections 1–4 (physics contracts, NumPy/Py3 modernisation, gauge doctrine),
                                 §7 (the entire pure-Python 1D solver), §8 (the ``Simmo``↔``Simmo`` algebra,
                                 which is matrix algebra on modes and does not care who produced the modes),
                                 §10 (the verification tiers) and §12 (the Fortran-elimination gates) are
                                 backend-independent. §5 (geometry/meshing), §6 (the 2D solver) and §9 (field
                                 reconstruction/plotting) are written against Option A and would be rewritten
                                 for Option B.
Effect on verification           None of the gauge-invariant doctrine changes: Tier G1 (gauge), G2 (structural
                                 identities), G3 (analytic/manufactured solutions) and G4 (checked-in legacy
                                 references) apply to both, but every numerical gate must be *re-run* per
                                 backend, and G3.3 — which pins the ``J``/``J_dag`` conjugation convention
                                 against a closed-form case — becomes more important in Option B, because
                                 there the sign/orientation conventions of the constrained Nédélec dofs are
                                 locally invented rather than inherited from a library primitive.
================================ ===========================================================================

Neither option changes the 1D solver, the plane-wave Resolutes, the Redheffer recursion, the public
API, or the requirement that all Fortran be removed. The choice is about the 2D engine: which
conventions you get from a library and which you write yourself.

Summarised in one line each: **Option A buys you the Bloch condition and the eigensolver and charges
you for geometry; Option B buys you the geometry (your ``.geo`` files keep working) and charges you
for the Bloch condition, the complex build, and the eigensolver driver.**


The legacy Fortran backend
------------------------------

You need this only to reproduce or extend the reference data — the ported code never invokes it.
Prefer the container over building it: a working ``emustack:latest`` image (Python 3.12.7,
NumPy 1.24.4, the prebuilt ``EMUstack.cpython-312-…so``, ``gmsh`` included) runs the reference tests
in seconds on one core:

.. code-block:: console

    $ docker run --rm -w /home/EMUstack/backend emustack:latest \
        bash -lc 'python3 ../tests/test_case_1_simple_struct.py'      # ~2 s, exit 0
    $ ./research/check_oracle_provenance.sh                          # image == this checkout?

With the completion of the port, no Fortran compiler, SuiteSparse, UMFPACK wrapper,
or ``.mail`` files are required. EMUstack is installed directly with standard Python tools:

.. code-block:: console

    $ pip install -e .

All of the above disappears with the port: no compiler, no SuiteSparse, no UMFPACK backport, no
``.mail`` files, no ``gmsh`` subprocess (Option A) — which is one of the main reasons for doing it.


Choosing
------------

Choose **Option A (NGSolve)** if you want the shortest path to a maintained, dependency-light,
single-process EMUstack on x86-64 Linux, Windows or macOS, and you accept rebuilding the geometry
layer natively. This is the path ``PORTING_PLAN_EMUSTACK_FEM.md`` specifies in implementable detail.

Choose **Option B (FEniCSx)** if any of these are true for you: you need ARM Linux or distributed
memory; you want the UFL form language (and a route to 3D/generalised geometries) as the primary
authoring surface; or preserving the ``gmsh``/``.geo`` geometry front-end matters more to your users
than the risk of hand-deriving constrained quasi-periodic Nédélec spaces and a SLEPc driver.

If you choose Option B, do these three things before writing any EMUstack code, in this order, and
treat each as a gate that may fail: (1) install the **complex** PETSc/SLEPc build and prove
``PETSc.ScalarType`` is complex; (2) assemble the actual 2D pencil (§6.2 of the porting plan,
translated to UFL) on a ``basix.ufl.mixed_element`` space and check it against the closed-form
homogeneous-cell spectrum; (3) impose the Bloch quasi-periodic condition by explicit dof
identification on a manufactured solution and prove the sign convention (this is Tier G3.2, the
``~1e-6`` vs ``~10¹`` test). If (3) does not pass with an error you can drive to zero, Option B is
not viable at EMUstack's accuracy requirements, and Option A is the port.
