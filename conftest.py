"""Present so the repo root lands on sys.path for both the real pytest and stepmold's shim."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
