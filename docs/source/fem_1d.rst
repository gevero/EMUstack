1D Finite Element Mode Solver
==============================

Overview
--------

EMUstack solves for the electromagnetic Bloch modes of 1D periodic lamellar gratings using a pure Python and SciPy finite element formulation. The coupled wave equations:

.. math::

    \mathbf{K} \mathbf{u} = \beta^2 \mathbf{M} \mathbf{u}

are assembled directly into SciPy CSR sparse matrices using exact rational element matrices. Discontinuous quadratic (:math:`P_2`) elements are used for the normal electric field component :math:`E_x` (allowing exact step jumps across material interfaces), while continuous periodic cubic (:math:`P_3`) elements are used for the tangential components :math:`E_y` and :math:`E_z`.

At normal incidence (:math:`k_y = 0`), the equations decouple into independent TE and TM polarizations. At conical incidence (:math:`k_y \ne 0`), the coupled system is solved simultaneously via sparse shift-invert Arnoldi iteration (SciPy ``splu`` and ``eigs``).

1D Unit Cell Geometry
---------------------

1D unit cells are defined via ``NanoStruct('1D_array', ...)``, which automatically generates the 1D mesh in memory. The parameter **'lc_bkg'** controls the mesh refinement, where **1 / lc_bkg** is the number of elements per unit cell.

Single Inclusion
~~~~~~~~~~~~~~~~

For a single inclusion centered in the unit cell::

    |                   period                    |
    |--------------------| 1 |--------------------|

The inclusion has width ``diameter1`` and material ``inclusion_a``, with the surrounding region made of ``background``.

Multiple Inclusions
~~~~~~~~~~~~~~~~~~~

For gratings with 2 inclusions, the spacing between inclusion surfaces is set by ``small_space``::

    |                   period                    |
      | small_space  |      
    |2|----------------|   1   |--------------| 2 |

Inclusion 1 is centered and made of ``inclusion_a``; higher-order inclusions are made of ``inclusion_b``.

For 3 or more inclusions, two spacing conventions are available:

1. **Equal center-to-center spacing** (default, ``edge_spacing=False``)::

    |       |   equally    |  separated   |       | 
    |--|    2    |------|  1  |---------| 3 |-----|

2. **Equal surface-to-surface spacing** (``edge_spacing=True``)::

    |  |       |equally |     |separated|   |      | 
    |--|   2   |--------|  1  |--------| 3 |------|

EMUstack supports up to 6 inclusions in a 1D unit cell.
