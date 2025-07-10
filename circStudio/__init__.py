"""Open-source software for actigraphy data analysis"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#
from . import analysis
from . import filters
from . import io
from . import sleep
from . import tests

__all__ = [
    "analysis",
    "filters",
    "io",
    "models",
    "sleep",
    "tests",
]

__version__ = '1.0.0'
