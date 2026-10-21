"""
    paths.py contais all the relevant paths for the operation of the
    code. When path are assigned their default values, EMUstack code
    can only be executed from 'EMUstack/subfolder'. An appropriate
    modification of the path values with absolute paths allows for
    code execution from an arbitrary folder.



    Copyright (C) 2015  Bjorn Sturmberg, Kokou Dossou, Felix Lawrence

    EMUstack is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path

_backend_dir = Path(__file__).resolve().parent

# where the backend code is placed
backend_path = str(_backend_dir)

# folder containing relevant data as the optical constants
data_path = str(_backend_dir / 'data') + '/'

# Dedicated directory where all .geo, *_template.py, .msh, and .mail files are stored
_msh_dir = _backend_dir / "data" / "meshes"
template_path = str(_msh_dir) + "/"
msh_path = str(_msh_dir) + "/"


