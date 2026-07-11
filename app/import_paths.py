import sys
from pathlib import Path

def _add_import_path(folder: Path) -> None:
    """
    Add a local folder to Python's import search path.

    Parameters
    ----------
    path:
        Folder that should be search when importing Python modules.

    Notes
    -----  
    ``sys.path.insert(0, ...)`` puts the folder at the beginning of the
    search path, so that local modules are found before any installed packages.
    """
    if folder.is_dir():
        sys.path.insert(0, str(folder))