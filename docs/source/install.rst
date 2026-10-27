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
    pip install -e .

Option B: Dedicated Python Virtual Environment (venv) + Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a lightweight virtual environment using Python's built-in ``venv`` module:

.. code-block:: bash

    # 1. Create a virtual environment in the repository folder
    python3 -m venv .venv

    # 2. Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # 3. Upgrade pip and install EMUstack in editable mode
    pip install --upgrade pip
    pip install -e .

Option C: Dedicated Virtual Environment with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use `uv <https://docs.astral.sh/uv/>`_ for extremely fast environment management:

.. code-block:: bash

    # 1. Create a virtual environment with uv
    uv venv .venv --python 3.12

    # 2. Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # 3. Install EMUstack with uv pip
    uv pip install -e .

Installing Optional Dependencies
--------------------------------

You can install optional dependency groups using pip or uv:

.. code-block:: bash

    # For running tutorial Jupyter notebooks (Jupyter, adaptive, etc.)
    pip install -e ".[notebooks]"       # or: uv pip install -e ".[notebooks]"

    # For developer tools and test suites (pytest, ruff)
    pip install -e ".[dev]"             # or: uv pip install -e ".[dev]"

    # For building documentation (Sphinx, sphinx-rtd-theme, nbsphinx)
    pip install -e ".[docs]"            # or: uv pip install -e ".[docs]"

    # For all extras simultaneously
    pip install -e ".[dev,notebooks,docs]"

Verifying Installation
----------------------

Verify your installation by running the test suite with pytest:

.. code-block:: bash

    pytest -v

All unit and integration tests should pass without errors or leftover files.
