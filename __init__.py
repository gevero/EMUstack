"""EMUstack: Finite Element Method mode matching solver for nanophotonics."""
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from .backend import objects, materials, plotting, stack, mode_calcs, paths, meshing, modes
from .backend import fem_1d, fem_2d, mode_matching, field_eval
from .backend.objects import *
from .backend.materials import *
from .backend.mode_calcs import *
from .backend.stack import *
from .backend.plotting import *
