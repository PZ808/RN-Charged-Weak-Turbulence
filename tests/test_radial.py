"""Near-zone radial coefficients — cross-checks between conventions.

The most load-bearing test is the canonical-vs-legacy convention bridge:
``C_abcd(i, a, b, c, d) == P4LMn_as_C_abcd(i, a, b, c, d)`` for any modes.
This is exactly the inline check in the notebook scratch cell.
"""
import numpy as np
import pytest

from rn_cwt import C_abcd, P4LMn_as_C_abcd, S_near_abcd, SLM, Anorm


L1_MODES = (
    {"n": 10, "l": 1},
    {"n": 30, "l": 1},
    {"n": 20, "l": 1},
    {"n": 1, "l": 1},
)


def test_C_abcd_matches_legacy_P4LMn():
    """Canonical C_abcd and legacy P4LMn must produce the same series coefficients.

    Notebook scratch cell line: ``print(x - y)`` for these modes is expected to be 0.
    """
    a, b, c, d = L1_MODES
    for i in [0, 1, 5, 10]:
        x = C_abcd(i, a, b, c, d)
        y = P4LMn_as_C_abcd(i, a, b, c, d)
        assert np.isclose(x, y), f"i={i}: canonical={x} legacy={y}"


def test_Anorm_finite_for_small_modes():
    for n in range(4):
        for l in range(3):
            val = Anorm(n, l)
            assert np.isfinite(val.real) and np.isfinite(val.imag)


def test_S_near_abcd_finite_l0_quartet():
    a = {"n": 2, "l": 0}
    b = {"n": 11, "l": 0}
    c = {"n": 7, "l": 0}
    d = {"n": 1, "l": 0}
    val = S_near_abcd(a, b, c, d)
    assert np.isfinite(val.real) and np.isfinite(val.imag)


def test_S_near_abcd_symmetric_in_unbarred_slots():
    """b_a(r) b_c(r) b_d(r) is symmetric under permutations of the unbarred
    slots a, c, d. So permuting them in S_near_abcd should leave the answer
    invariant. (The notebook lines 480-498 effectively check this for SLM.)
    """
    # Pick modes that are all distinct.
    a = {"n": 0, "l": 0}
    b = {"n": 1, "l": 0}  # barred
    c = {"n": 2, "l": 0}
    d = {"n": 3, "l": 0}

    val_acd = S_near_abcd(a, b, c, d)
    val_cad = S_near_abcd(c, b, a, d)  # swap a <-> c
    val_dca = S_near_abcd(d, b, c, a)  # swap a <-> d
    val_cda = S_near_abcd(c, b, d, a)  # swap c <-> d (after the d<->a)

    rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-30)
    assert rel(val_acd, val_cad) < 1e-9, f"a<->c: {val_acd} vs {val_cad}"
    assert rel(val_acd, val_dca) < 1e-9, f"a<->d: {val_acd} vs {val_dca}"
    assert rel(val_acd, val_cda) < 1e-9, f"c<->d: {val_acd} vs {val_cda}"


def test_SLM_legacy_returns_finite():
    """Legacy SLM still works for the same example modes."""
    val = SLM(
        {"n": 2, "l": 0},
        {"n": 11, "l": 0},
        {"n": 7, "l": 0},
        {"n": 1, "l": 0},
    )
    assert np.isfinite(val.real) and np.isfinite(val.imag)


# --------------------------------------------------------------------------
# Standalone near-zone QNM mode function b_near(n, ℓ, x̆)
# --------------------------------------------------------------------------

from rn_cwt import Anorm, b_near, iq, s


def test_b_near_n0_matches_closed_form():
    """For n=0 the polynomial part is 1 (PLM(0, ·, ·) = 1), and

        b_0(x̆) = (1 / Anorm(0, ℓ)) · (-x̆)^{s_0} (1 + x̆)^{s_0 + iq}.

    Check the implementation against this closed form at x̆ = 1.
    """
    s0 = s(0, 0)
    expected = np.exp(1j * np.pi * s0) * 2.0 ** (s0 + iq) / Anorm(0, 0)
    val = b_near(0, 0, 1.0)
    assert np.isclose(val, expected, atol=1e-14)


def test_b_near_large_x_falloff_matches_minus_h():
    """Asymptotically b_n ~ x̆^{-(h + n)} at large x̆.

    The full mode contains (-x̆)^{s_n} (1+x̆)^{s_n+iq}, both contributing
    x̆^{-h/2} at the leading order. The (1+x̆) factor approaches the
    asymptotic only slowly; comparing |b_0(10^4)|/|b_0(10^3)| gives < 1%
    residual from the exact 10^{-h} = (10000/1000)^{-h} prediction.
    """
    from rn_cwt import h as h_conf

    h_real = float(np.real(h_conf(0)))
    expected_ratio = 10.0 ** (-h_real)
    ratio = np.abs(b_near(0, 0, 1.0e4)) / np.abs(b_near(0, 0, 1.0e3))
    assert np.isclose(ratio, expected_ratio, rtol=5e-3), (
        f"|b_0(1e4)|/|b_0(1e3)| = {ratio:.6f}, expected {expected_ratio:.6f}"
    )


def test_b_near_array_input():
    """Accepts array x̆ and returns same-shape complex array."""
    x = np.linspace(0.1, 5.0, 7)
    vals = b_near(0, 0, x)
    assert vals.shape == x.shape
    assert np.iscomplexobj(vals)
    assert np.all(np.isfinite(vals))


def test_b_near_n_dependence_polynomial_order():
    """The polynomial sum has n+1 terms — for fixed (ℓ, x̆) the value differs
    nontrivially across n. Smoke check it's distinct."""
    x = 0.5
    v0, v1, v2 = b_near(0, 0, x), b_near(1, 0, x), b_near(2, 0, x)
    assert abs(v0 - v1) > 1e-3
    assert abs(v1 - v2) > 1e-3
