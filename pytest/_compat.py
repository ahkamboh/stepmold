"""Hand control to the real pytest if it is installed somewhere else on sys.path."""

import os
import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec

# realpath, not abspath: abspath does not resolve symlinks, so on macOS the same
# directory reachable as both /tmp/x and /private/tmp/x compares unequal. When the repo
# root then appears on sys.path under the other spelling, _other_paths() fails to filter
# this shim out, PathFinder finds it again, and importing pytest recurses until the
# stack dies. Normalising both sides makes the two spellings the same path.
_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _other_paths():
    out = []
    for entry in sys.path:
        resolved = os.path.realpath(entry or os.getcwd())
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
