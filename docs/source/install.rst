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

Installation via Pip
--------------------

From the repository root, install EMUstack together with all optional packages:

.. code-block:: bash

    pip install -e ".[all]"

This is the recommended install: everything in the repository then works out of the
box, i.e. the gmsh mesher, the example notebooks (Jupyter, ``adaptive``, ``loky``), the
test suite and the documentation build. See :ref:`install-extras` for a smaller install.

Setting Up a Dedicated Environment
-----------------------------------

We strongly recommend installing EMUstack inside an isolated environment. Below are instructions for three standard workflows:

Option A: Dedicated Conda Environment + Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate a clean Conda environment, then install EMUstack and its dependencies via pip:

.. code-block:: bash

    # 1. Create a dedicated environment (Python 3.11 or 3.12 recommended)
    conda create -n emustack python=3.12 -y

    # 2. Activate the environment
    conda activate emustack

    # 3. Upgrade pip and install EMUstack in editable mode
    pip install --upgrade pip
    pip install -e ".[all]"

Option B: Dedicated Python Virtual Environment (venv) + Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a lightweight virtual environment using Python's built-in ``venv`` module:

.. code-block:: bash

    # 1. Create a virtual environment in the repository folder
    python3 -m venv .venv

    # 2. Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # 3. Upgrade pip and install EMUstack with all optional packages
    pip install --upgrade pip
    pip install -e ".[all]"

Option C: Dedicated Virtual Environment with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use `uv <https://docs.astral.sh/uv/>`_ for extremely fast environment management:

.. code-block:: bash

    # 1. Create a virtual environment with uv
    uv venv .venv --python 3.12

    # 2. Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # 3. Install EMUstack with all optional packages
    uv pip install -e ".[all]"

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
     - NumPy, SciPy, Matplotlib, xarray, NGSolve/Netgen
     - Simulations with ``mesher='netgen'``
   * - ``gmsh``
     - Gmsh
     - The default ``mesher='gmsh'`` and the ``.geo`` templates
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

    pip install -e .                      # core only
    pip install -e ".[gmsh,notebooks]"    # e.g. core + gmsh + notebooks

.. note::

   Gmsh can also come from your system (``apt install gmsh``, ``conda install -c
   conda-forge gmsh``, or a binary from gmsh.info). EMUstack looks for it in this
   order: the ``GMSH_BIN`` environment variable, ``gmsh`` on your ``PATH``,
   ``~/.local/bin/gmsh``, ``~/bin/gmsh``.

Verifying Installation
----------------------

Verify your installation by running the test suite with pytest:

.. code-block:: bash

    pytest -v

All unit and integration tests should pass without errors or leftover files.
