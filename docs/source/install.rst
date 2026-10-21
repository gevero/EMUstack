Installation
============

Requirements
------------

EMUstack v2.0 is a pure Python package with modern finite-element solvers powered by SciPy and NGSolve. It requires **no Fortran compilers**, **no C++ build tools**, and **no SuiteSparse compilation**.

Prerequisites:

* **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14.
* **Core Scientific Libraries:** NumPy (>= 1.22), SciPy (>= 1.9), Matplotlib (>= 3.6), xarray (>= 2023.1).
* **Finite Element Engine:** NGSolve (>= 6.2.2401) and Netgen (>= 6.2.2401).

Installation via Pip
--------------------

To install EMUstack in editable development mode from the repository root:

.. code-block:: bash

    pip install -e .

Installing Optional Dependencies
--------------------------------

You can install optional extras depending on your workflow:

.. code-block:: bash

    # For running tutorial Jupyter notebooks
    pip install -e ".[notebooks]"

    # For running the test suite
    pip install -e ".[dev]"

    # For building documentation
    pip install -e ".[docs]"

    # For everything
    pip install -e ".[dev,notebooks,docs]"

Using Conda / Mamba
-------------------

If you prefer managing environments with Conda or Mamba:

.. code-block:: bash

    conda create -n fem-scattering python=3.11
    conda activate fem-scattering
    conda install -c conda-forge ngsolve netgen numpy scipy matplotlib xarray
    pip install -e .

Verifying Installation
----------------------

Verify your installation by running the test suite with pytest:

.. code-block:: bash

    pytest -v

All unit and integration tests should pass without errors or leftover files.
