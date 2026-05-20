"""Regression: ``S_near_abcd`` matches Peter's precomputed Mathematica data.

The data file ``data/RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m`` (214 MB,
Mathematica association keyed by 4-tuples of (n, l, m) modes) was generated
from Peter's ``SnearComponentNLM4`` in
``Mathematica/generate_4wave_coeffs.nb``. We standardize ``S_near_abcd`` on
the paper / Mathematica convention so it matches the data file directly to
machine precision. ``S_near_paper`` is now an alias of ``S_near_abcd``.

If the data file is missing the test skips with a pointer.
"""
import os
import re

import numpy as np
import pytest
from scipy.special import gamma

from rn_cwt import S_near_abcd, S_near_paper, gamma_abcd, iq

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m",
)


ENTRY_RE = re.compile(
    r"\{\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\}\}\s*->\s*"
    r"(\-?[\d\.]+)\`?[\d\.\-eE]*\s*([+\-])\s*"
    r"(\-?[\d\.]+)\`?[\d\.\-eE]*\*I",
    re.DOTALL,
)


def _parse_entries(text):
    for m in ENTRY_RE.finditer(text):
        groups = m.groups()
        modes = []
        for i in range(4):
            n, l, mm = int(groups[3 * i]), int(groups[3 * i + 1]), int(groups[3 * i + 2])
            modes.append({"n": n, "l": l, "m": mm})
        re_str, sign, im_str = groups[12], groups[13], groups[14]
        val = complex(float(re_str), float(im_str) * (1 if sign == "+" else -1))
        yield tuple(modes), val


@pytest.fixture(scope="module")
def peter_entries():
    if not os.path.exists(DATA_PATH):
        pytest.skip(
            f"Peter's data file not present at {DATA_PATH}; this is the "
            "high-precision Mathematica precomputed S-coefficients."
        )
    with open(DATA_PATH, "r") as f:
        text = f.read(120_000)
    return list(_parse_entries(text))[:60]


def test_S_near_abcd_matches_data_file(peter_entries):
    """S_near_abcd must match Peter's precomputed values to ~1e-12.

    Locks the paper-convention finite-part formula (Mma's ``SnearComponentNLM4``)
    against high-precision Mathematica values.
    """
    assert len(peter_entries) > 30, f"only {len(peter_entries)} entries parsed"
    worst = 0.0
    for modes, val_peter in peter_entries:
        a, b, c, d = modes
        val_us = S_near_abcd(a, b, c, d)
        rel = abs(val_us - val_peter) / max(abs(val_peter), 1e-30)
        worst = max(worst, rel)
    assert worst < 1e-12, f"worst rel error vs Peter's data: {worst:.3e}"


def test_S_near_paper_is_alias():
    """S_near_paper is exposed as an explicit alias for S_near_abcd."""
    assert S_near_paper is S_near_abcd
