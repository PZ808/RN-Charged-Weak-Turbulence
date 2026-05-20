"""Background parameters and helper functions for near-extremal Reissner-Nordström.

Module-level constants encode the extremal RN background in units where r_+ = 1:

    rp = 1.0          horizon radius
    q  = 0.02         scalar charge (so q << 1, near-extremal)
    iq = 1j * q
    h0 = h(0)         conformal weight at l = 0
    Q  = 1.0          black-hole charge (unused; kept for parity with notebook)

Changing q or rp invalidates every lru_cache in angular.py and radial.py.
Restart any session after changing them.
"""
from cmath import sqrt as csqrt

import numpy as np
from scipy.special import gamma

rp = 1.0
q = 2.0 / 100.0
iq = 1j * q
Q = 1.0  # BH charge — unused; kept for parity with notebook

h0 = 0.5 + csqrt(0.25 - q**2)  # conformal weight at l = 0


def bar(f):
    """Complex conjugate."""
    return np.conj(f)


def h(l):
    """Conformal weight h_l = 1/2 + sqrt(1/4 + l(l+1) - q^2).

    Picks the +sqrt branch when the argument under the sqrt is non-negative,
    and the -sqrt branch otherwise (gives a complex conjugate pair of weights
    in the supplementary range; cf. Zimmerman 2017 eq. 1).
    """
    d2 = 0.25 + l * (l + 1) - q**2
    if d2 > 0:
        return 0.5 + csqrt(d2)
    else:
        return 0.5 - csqrt(d2)


def s(n, l):
    """QNM frequency in the variable s = -i omega:

        s_{n,l} = -(n + h(l) + i q) / 2.
    """
    return -(n + h(l) + iq) / 2.0


def w(n, l):
    """QNM frequency omega_{n,l} = -i (n + h(l) + i q) / 2."""
    return -1j * (n + h(l) + iq) / 2.0


def poch(a, n):
    """Pochhammer (rising-factorial) symbol (a)_n = a (a+1) ... (a+n-1).

    For integer n >= 0, uses the direct product form. The notebook's original
    formula ``(-1)^n * gamma(1-a)/gamma(1-a-n)`` is mathematically equivalent
    but relies on ``gamma(negative_integer) = +/-inf`` (so the quotient gives
    the correct zero when a factor crosses zero). Modern scipy (>= ~1.11)
    returns NaN at negative integer arguments, which propagates and breaks
    every downstream sum. The direct product avoids the issue entirely and
    agrees with the gamma form to ~1e-13 on non-degenerate inputs (verified
    in tests).

    For non-integer n, falls back to ``gamma(a + n) / gamma(a)``.
    """
    if type(n) == int and n >= 0:
        result = 1.0
        for k in range(n):
            result = result * (a + k)
        return result
    return gamma(a + n) / gamma(a)
