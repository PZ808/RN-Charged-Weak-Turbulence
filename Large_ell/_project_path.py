"""Locate the repository root so that `coupling_coefficients_stable` is importable.

The large-l programme lives in its own folder, which may sit at any depth below
the repository root that holds `coupling_coefficients_stable.py`.  Only the
`validate()` and cross-check paths need the project code, so the search is done
lazily at those call sites and never at import time of a working module.

Usage:

    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "coupling_coefficients_stable.py"


def project_root():
    """The nearest ancestor directory containing MARKER, or None."""
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / MARKER).is_file():
            return d
    return None


def ensure_on_path():
    """Prepend the repository root to sys.path.  Returns the root, or None.

    A no-op when the project code is already importable (installed, or the
    folder is already on the path), so it is safe to call repeatedly.
    """
    root = project_root()
    if root is None:
        return None
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
