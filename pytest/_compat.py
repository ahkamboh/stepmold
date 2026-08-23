"""Hand control to the real pytest if it is installed somewhere else on sys.path."""

import os
import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _other_paths():
    out = []
    for entry in sys.path:
        resolved = os.path.abspath(entry or os.getcwd())
        if resolved != _ROOT:
            out.append(entry)
    return out


def delegate():
    """Load and install the real pytest, or return None if there isn't one."""
    spec = PathFinder.find_spec("pytest", _other_paths())
    if spec is None or spec.loader is None:
        return None
    module = module_from_spec(spec)
    sys.modules["pytest"] = module
    spec.loader.exec_module(module)
    return module
