"""
Path setup for spann3r_slam package.
Adds bundled model sources to ``sys.path`` so ``import dust3r`` and
``import spann3r`` resolve to the integrated upstream copy in this repository.
"""

import sys
import os

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spann3r_core = os.path.join(_root_dir, "spann3r_core")

# Insert at the beginning so our bundled copies take priority.
for _p in [_spann3r_core]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
