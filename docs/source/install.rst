Installation
============

Requirements
------------

EMUstack v2.0 is a pure Python package with modern finite-element solvers powered by SciPy and NGSolve. It requires **no Fortran compilers**, **no C++ build tools**, and **no SuiteSparse compilation**.

Prerequisites:

* **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14.
* **Core Scientific Libraries:** NumPy (>= 1.22), SciPy (>= 1.9), Matplotlib (>= 3.6), xarray (>= 2023.1).
* **Finite Element Engine:** NGSolve (>= 6.2.2401) and Netgen (>= 6.2.2401).
* **Mesher for the .geo templates:** Gmsh (>= 4.11), installed from PyPI by the full install below.

All of these are installed by pip; nothing has to be compiled.
Everything installs from pip wheels on Linux, macOS and Windows; the test suite runs on all
three.

Step 1: Create an Environment and Install EMUstack
--------------------------------------------------

We strongly recommend installing EMUstack inside its own environment. Pick one of the three
workflows below. Each installs ``emustack[all]`` from PyPI: the package with everything the
example notebooks need (Jupyter, ``adaptive``, ``loky``), plus the test and docs tools. The 30
mesh templates ship with the package. See :ref:`install-extras` for a smaller install.

Option A: Conda Environment + Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda create -n emustack python=3.12 -y
    conda activate emustack
    pip install "emustack[all]"

Option B: Python Virtual Environment (venv) + Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install "emustack[all]"

Option C: Virtual Environment with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    uv venv .venv --python 3.12
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install "emustack[all]"

Step 2 (Optional): Get the Example Notebooks
--------------------------------------------

The 45 example notebooks are in the GitHub repository, in ``examples/`` (gmsh meshes) and
``examples_netgen/`` (Netgen meshes). Clone it and open them with the environment from
step 1; nothing else needs installing:

.. code-block:: bash

    git clone https://github.com/gevero/EMUstack.git
    cd EMUstack
    jupyter lab examples/

Step 3 (Optional): Development Install
--------------------------------------

To change EMUstack itself, run the test suite or build this documentation, install the
clone in editable mode instead (from the repository root):

.. code-block:: bash

    pip install -e ".[all]"

.. _install-extras:

Choosing What to Install
------------------------

``[all]`` is a shortcut for the optional groups below. If you only need some of them,
install the core package and pick the groups you want (with pip or ``uv pip``):

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Group
     - Contents
     - Needed for
   * - *(core)*
     - NumPy, SciPy, Matplotlib, xarray, PyYAML, NGSolve/Netgen, Gmsh
     - Simulations with the default ``mesher='gmsh'`` or ``mesher='netgen'``
   * - ``gmsh``
     - Gmsh (kept as a backward-compatible alias; Gmsh is now a core dependency)
     - Older scripts that install ``emustack[gmsh]``
   * - ``notebooks``
     - Jupyter, nbconvert, ipykernel, ``adaptive``, ``loky``
     - The example notebooks (adaptive sampling, parallel sweeps)
   * - ``dev``
     - pytest, ruff
     - Running the test suite
   * - ``docs``
     - Sphinx, sphinx-rtd-theme, nbsphinx
     - Building this documentation

.. code-block:: bash

    pip install emustack                  # core only: solvers, gmsh and netgen meshers
    pip install "emustack[notebooks]"     # e.g. core + notebooks

.. note::

   Gmsh can also come from your system (``apt install gmsh``, ``conda install -c
   conda-forge gmsh``, or a binary from gmsh.info). EMUstack looks for it in this
   order: the ``GMSH_BIN`` environment variable, ``gmsh`` on your ``PATH``,
   ``~/.local/bin/gmsh``, ``~/bin/gmsh``.

.. note::

   On a minimal Linux system (a container, a headless server or an HPC node) the pip wheels
   can fail with ``cannot open shared object file``: NGSolve links against the system
   OpenBLAS, and gmsh against OpenGL and X11 client libraries. On Debian/Ubuntu install them
   with::

       sudo apt-get install libopenblas0 libglu1-mesa libgl1 libgomp1 libxcursor1 libxft2 libxinerama1

   No display is needed. Without the gmsh libraries you can still use ``mesher='netgen'``.

Verifying Installation
----------------------

Check that EMUstack imports and finds gmsh:

.. code-block:: bash

    python -c "from EMUstack.backend import meshing; print(meshing.gmsh_available())"

This prints ``True``. With a development install (step 3), also run the test suite from the
repository root:

.. code-block:: bash

    pytest -q

All tests should pass without errors or leftover files.
