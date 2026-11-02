"""EMUstack: Finite Element Method mode matching solver for nanophotonics.

Import the modules from the package, for example
``from EMUstack.backend import objects, materials, plotting``. Importing EMUstack does
not change ``sys.path``, so each module exists once and module-level settings such as
``fem_2d.eigensolver`` or ``objects.default_mesher`` apply everywhere.
"""
from .backend import objects, materials, plotting, stack, mode_calcs, paths, meshing, modes
from .backend import fem_1d, fem_2d, mode_matching, field_eval
from .backend.objects import *
from .backend.materials import *
from .backend.mode_calcs import *
from .backend.stack import *
from .backend.plotting import *
