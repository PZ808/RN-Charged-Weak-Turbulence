"""Regression tests for Peter's precomputed S-table loader.

Tests pull from ``data/RN_ZDMs_4modeCoeffs_Neq1_Leq4_coeffs_numeric.m`` (or its
pickle cache); skipped if neither is present.
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

from rn_cwt import (
    S_near_abcd,
    S_near_cached,
    S_near_lookup,
    load_S_table,
    parse_S_table_from_mathematica,
)
from rn_cwt.precomputed import _DEFAULT_DATA_PATH


@pytest.fixture(scope="module")
def table():
    if not (
        os.path.exists(_DEFAULT_DATA_PATH)
        or os.path.exists(_DEFAULT_DATA_PATH + ".pkl")
    ):
        pytest.skip(
            "Peter's S-table source/cache not present at "
            f"{_DEFAULT_DATA_PATH}; regenerate via Mathematica."
        )
    return load_S_table()


def test_table_loads_and_has_entries(table):
    assert isinstance(table, dict)
    assert len(table) > 100


def test_table_keys_are_4tuples_of_3tuples(table):
    key = next(iter(table))
    assert isinstance(key, tuple)
    assert len(key) == 4
    for mode in key:
        assert isinstance(mode, tuple)
        assert len(mode) == 3
        assert all(isinstance(x, int) for x in mode)


def test_lookup_matches_S_near_abcd_machine_precision(table):
    """Spot-check several entries — table values match Python S_near_abcd
    at machine precision (since we standardized on the paper convention)."""
    test_quads = [
        ({"n": 0, "l": 0, "m": 0},) * 4,
        (
            {"n": 0, "l": 0, "m": 0},
            {"n": 0, "l": 1, "m": 0},
            {"n": 0, "l": 0, "m": 0},
            {"n": 0, "l": 0, "m": 0},
        ),
        (
            {"n": 0, "l": 1, "m": -1},
            {"n": 0, "l": 1, "m": 0},
            {"n": 0, "l": 2, "m": 1},
            {"n": 0, "l": 0, "m": 0},
        ),
    ]
    worst = 0.0
    for a, b, c, d in test_quads:
        v_table = S_near_lookup(table, a, b, c, d)
        v_py = S_near_abcd(a, b, c, d)
        rel = abs(v_table - v_py) / max(abs(v_table), 1e-30)
        worst = max(worst, rel)
    assert worst < 1e-12, f"worst rel err vs S_near_abcd: {worst:.3e}"


def test_lookup_raises_on_missing_key(table):
    """Lookup of a mode combination not in the table should raise KeyError."""
    # Very high n / l unlikely to be in any precomputed table.
    huge = ({"n": 99, "l": 99, "m": 0},) * 4
    with pytest.raises(KeyError):
        S_near_lookup(table, *huge)


def test_cached_falls_back_to_python(table):
    """S_near_cached uses the table when available; falls back otherwise."""
    # Hit: present in table.
    a = {"n": 0, "l": 0, "m": 0}
    v_table = S_near_cached(table, a, a, a, a)
    v_py = S_near_abcd(a, a, a, a)
    assert np.isclose(v_table, v_py, rtol=1e-12)

    # Miss: very high n — falls back to Python S_near_abcd.
    high = {"n": 6, "l": 0, "m": 0}
    v_fallback = S_near_cached(table, high, high, high, high)
    v_direct = S_near_abcd(high, high, high, high)
    assert np.isclose(v_fallback, v_direct, rtol=1e-12)


def test_load_uses_cache(table):
    """Loading should be near-instant when the pickle cache exists."""
    import time

    t0 = time.time()
    _ = load_S_table()
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"cached load took {elapsed:.2f}s (expected < 1s)"


def test_parser_round_trip_via_pickle(tmp_path):
    """End-to-end: write a tiny Mathematica-format file, parse it, pickle it,
    reload via load_S_table — values preserved."""
    src = tmp_path / "tiny.m"
    src.write_text(
        "precomputedS1b234 = <|"
        "{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}} -> "
        "0.1234`13.0 + 0.5678`13.0*I, "
        "{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}} -> "
        "-0.5`13.0 - 1.25`13.0*I"
        "|>;"
    )
    parsed = parse_S_table_from_mathematica(str(src))
    assert len(parsed) == 2
    assert np.isclose(parsed[((0, 0, 0),) * 4], 0.1234 + 0.5678j)
    assert np.isclose(
        parsed[((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))], -0.5 - 1.25j
    )

    # Load via cache.
    cache = tmp_path / "tiny.m.pkl"
    table = load_S_table(str(src), str(cache))
    assert cache.exists()
    assert table == parsed
