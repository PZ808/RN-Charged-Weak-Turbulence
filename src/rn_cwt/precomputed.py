"""Load Peter's precomputed near-zone S-coefficient table.

The Mathematica notebook ``Mathematica/generate_4wave_coeffs.nb`` exports a
214 MB association ``precomputedS1b234`` to
``data/RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m`` containing high-precision
S-coefficients for all 4-mode combinations up to a certain (n, l, m) cutoff.

Generating the table from scratch in Python via ``S_near_abcd`` would take
on the order of 100 days for N=5 overtones × l_max=4 (per the paper §10.1
end-note). With Peter's table we can look up these values in O(1).

This module:
- parses the Mathematica association via regex on first call,
- caches the result to a Python pickle alongside the source for fast reload,
- exposes ``S_near_lookup`` (raises KeyError on miss) and
  ``S_near_cached`` (transparent fallback to the slow Python form).

Convention: ``S_near_abcd`` in this package matches Peter's
``SnearComponentNLM4`` directly (paper convention). See ``CLAUDE.md``
"S convention" section. The table values are paper-convention and slot into
``S_near_abcd``'s argument order ``(a, bar(b), c, d)`` — i.e., key
``((n1,l1,m1), (n2,l2,m2), (n3,l3,m3), (n4,l4,m4))`` corresponds to
``S_near_abcd(modes_1, modes_2, modes_3, modes_4)`` with the second slot
barred.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Dict, Tuple

ModeKey = Tuple[int, int, int]  # (n, l, m)
QuadKey = Tuple[ModeKey, ModeKey, ModeKey, ModeKey]


# Default path to Peter's data file, resolved relative to the repo root
# (i.e. two levels up from this module: src/rn_cwt/precomputed.py → ../../data/...).
_DEFAULT_DATA_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m",
    )
)


_ENTRY_RE = re.compile(
    r"\{\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\},\s*"
    r"\{(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\}\}\s*->\s*"
    r"(\-?[\d\.]+)\`?[\d\.\-eE]*\s*([+\-])\s*"
    r"(\-?[\d\.]+)\`?[\d\.\-eE]*\*I",
    re.DOTALL,
)


def parse_S_table_from_mathematica(path: str) -> Dict[QuadKey, complex]:
    """Parse Peter's Mathematica association file into a Python dict.

    Slow: O(file size). On the 214 MB production file this takes a few
    seconds. Use ``load_S_table`` for the cached version.
    """
    table: Dict[QuadKey, complex] = {}
    with open(path, "r") as f:
        text = f.read()
    for m in _ENTRY_RE.finditer(text):
        g = m.groups()
        key = tuple(
            (int(g[3 * i]), int(g[3 * i + 1]), int(g[3 * i + 2]))
            for i in range(4)
        )
        re_str, sign, im_str = g[12], g[13], g[14]
        val = complex(float(re_str), float(im_str) * (1 if sign == "+" else -1))
        table[key] = val
    return table


def load_S_table(
    path: str | None = None,
    cache_path: str | None = None,
) -> Dict[QuadKey, complex]:
    """Load Peter's S-table, using a pickle cache when available.

    Parameters
    ----------
    path : str or None
        Path to the Mathematica .m file. Defaults to the repo's
        ``data/RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m``.
    cache_path : str or None
        Path to the pickle cache. Defaults to ``<path>.pkl``.

    If the cache file exists and is at least as recent as the source, it is
    loaded. Otherwise the source is parsed and the cache is written.

    Raises ``FileNotFoundError`` if neither cache nor source is present.
    """
    if path is None:
        path = _DEFAULT_DATA_PATH
    if cache_path is None:
        cache_path = path + ".pkl"

    source_exists = os.path.exists(path)
    cache_exists = os.path.exists(cache_path)

    if cache_exists and (
        not source_exists or os.path.getmtime(cache_path) >= os.path.getmtime(path)
    ):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not source_exists:
        raise FileNotFoundError(
            f"S-table source not found at {path!r} and no cache at {cache_path!r}. "
            "Generate via Mathematica/generate_4wave_coeffs.nb."
        )

    table = parse_S_table_from_mathematica(path)
    with open(cache_path, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    return table


def _mode_to_key(mode: dict) -> ModeKey:
    return (int(mode["n"]), int(mode["l"]), int(mode.get("m", 0)))


def S_near_lookup(
    table: Dict[QuadKey, complex],
    a: dict,
    b: dict,
    c: dict,
    d: dict,
) -> complex:
    """Return ``table[(a, b, c, d)]`` for mode-dict arguments.

    Raises ``KeyError`` if the combination is not in the table.
    """
    key = (
        _mode_to_key(a),
        _mode_to_key(b),
        _mode_to_key(c),
        _mode_to_key(d),
    )
    return table[key]


def S_near_cached(
    table: Dict[QuadKey, complex],
    a: dict,
    b: dict,
    c: dict,
    d: dict,
) -> complex:
    """Look up in the table; fall back to ``S_near_abcd`` on a miss.

    Useful for assembling C-tensors over a mode set where some combinations
    are precomputed and others need to be evaluated on the fly.
    """
    from .radial import S_near_abcd

    key = (
        _mode_to_key(a),
        _mode_to_key(b),
        _mode_to_key(c),
        _mode_to_key(d),
    )
    val = table.get(key)
    if val is None:
        return S_near_abcd(a, b, c, d)
    return val
