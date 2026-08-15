"""Near-zone radial coefficients.

Two parallel conventions live here. Both compute integrals of products of
four near-zone mode functions b_{n,l}(r):

- **Canonical** (used downstream by coupling.py): ``S_near_abcd(a, b, c, d)``
  with the **barred slot second**, schematically b_a bar(b_b) b_c b_d. Built
  from ``PLM_cached``, ``PLM``, ``C_abcd``, ``gamma_abcd``, ``Scomp_abcd``.

- **Legacy** (kept for cross-checks and batch table generation):
  ``SLM(k1, k2, k3, K)`` with the **barred slot last**, schematically
  b_{k1} b_{k2} b_{k3} bar(b_K). Built from ``P``, ``P2/P3/P4``,
  ``Scomp``/``ScompLM``. The ``P4LMn_as_C_abcd`` function bridges the two
  conventions and is exercised by ``tests/test_radial.py``.
"""
import itertools
from cmath import sqrt as csqrt
from functools import lru_cache
from math import fsum

import numpy as np
from scipy.special import gamma

from .background import bar, h, h0, iq, poch, s


def Anorm(n, l):
    """Mode normalization factor. Enters as ``1 / Anorm(n, l)`` per slot."""
    tmp = (
        1j
        * (-1.0) ** (h(l) + iq)
        * 2.0
        * gamma(n + 1.0)
        * gamma(2 * h(l) + n)
        * gamma(1 - h(l) - n - iq)
    ) / (gamma(h(l) - iq) * poch(h(l) + iq, n))
    return csqrt(tmp)


# --------------------------------------------------------------------------
# Stable polynomial machinery shared by both conventions.
# --------------------------------------------------------------------------

def stable_sum(z):
    """Accurately sum complex double-precision values."""
    z = [complex(v) for v in z]
    z.sort(key=abs)
    return complex(
        fsum(v.real for v in z),
        fsum(v.imag for v in z),
    )


def stableConvolve(a, b):
    """Coefficient-wise polynomial convolution with stable summation."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    na = len(a)
    nb = len(b)
    out = np.empty(na + nb - 1, dtype=np.complex128)

    for k in range(na + nb - 1):
        jmin = max(0, k - (nb - 1))
        jmax = min(k, na - 1)
        out[k] = stable_sum(
            a[j] * b[k - j]
            for j in range(jmin, jmax + 1)
        )

    return out


def stableMultiConvolve(arrays):
    """Balanced-tree convolution of several coefficient arrays."""
    arrays = [np.asarray(a, dtype=np.complex128) for a in arrays]

    if not arrays:
        return np.array([1.0 + 0.0j], dtype=np.complex128)

    while len(arrays) > 1:
        next_level = []
        for j in range(0, len(arrays) - 1, 2):
            next_level.append(stableConvolve(arrays[j], arrays[j + 1]))
        if len(arrays) % 2:
            next_level.append(arrays[-1])
        arrays = next_level

    return arrays[0]


def coeffAt(a, j):
    """Return coefficient j, with zero outside the polynomial range."""
    if 0 <= j < len(a):
        return a[j]
    return 0.0 + 0.0j


@lru_cache(None)
def _pTable_cached(n, ell):
    r"""Cached finite coefficient table ``(P_0,...,P_n)``.

    Uses the exact recurrence

        P_{j+1} / P_j =
            (-n+j) (1-2h-n+j)
            -----------------------------
            (j+1) (1-h-n-iq+j),

    rather than evaluating Pochhammer/Gamma ratios independently.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    hh = h(ell)
    out = [1.0 + 0.0j]

    for j in range(n):
        out.append(
            out[-1]
            * (-n + j)
            * (1 - 2 * hh - n + j)
            / ((j + 1) * (1 - hh - n - iq + j))
        )

    return tuple(out)


def pTable(n, ell):
    """Return ``[P_0,...,P_n]`` as a complex NumPy array."""
    return np.asarray(_pTable_cached(n, ell), dtype=np.complex128)


def pNLMTable(mode):
    """Return the finite P-coefficient table for a mode dict."""
    return pTable(mode["n"], mode["l"])


# --------------------------------------------------------------------------
# Canonical convention — barred slot SECOND (used by coupling.py).
# --------------------------------------------------------------------------

@lru_cache(None)
def PLM_cached(j, n, ell):
    """j-th series coefficient, read from the recurrence-generated table."""
    return coeffAt(_pTable_cached(n, ell), j)


def PLM(j, mode):
    """j-th series coefficient for ``mode = {'n': n, 'l': ell}``."""
    return PLM_cached(j, mode["n"], mode["l"])


def C_abcd_table(a, b, c, d):
    r"""All coefficients with barred slot ``b``.

    This is the coefficient array of

        P_a(x) conjugate(P_b(x)) P_c(x) P_d(x).
    """
    return stableMultiConvolve(
        [
            pNLMTable(a),
            np.conjugate(pNLMTable(b)),
            pNLMTable(c),
            pNLMTable(d),
        ]
    )


def C_abcd(i, a, b, c, d):
    """i-th coefficient of ``P_a conjugate(P_b) P_c P_d``."""
    return coeffAt(C_abcd_table(a, b, c, d), i)


def gamma_abcd(a, b, c, d):
    """gamma_abcd = s_a + bar(s_b) + s_c + s_d (barred slot is b)."""
    return (
        s(a["n"], a["l"])
        + bar(s(b["n"], b["l"]))
        + s(c["n"], c["l"])
        + s(d["n"], d["l"])
    )


def Scomp_abcd(j, a, b, c, d):
    r"""Canonical paper/Mathematica finite-part factor.

        (-1)^(j+gamma) Gamma(1+j+gamma)
        Gamma(-1-j-2gamma-2iq) / Gamma(-2iq-gamma)^2.

    For a whole j range prefer ``Scomp_abcd_table``; it evaluates this
    Gamma expression only at j=0 and then uses the exact recurrence.
    """
    gam = gamma_abcd(a, b, c, d)
    return (
        (-1) ** (j + gam)
        * gamma(1 + j + gam)
        * gamma(-1 - j - 2 * gam - 2 * iq)
        / gamma(-2 * iq - gam) ** 2
    )


def Scomp_abcd_table(nmax, a, b, c, d):
    r"""Return ``[(Scheck)_0,...,(Scheck)_nmax]`` by recurrence.

    The j-dependent Gamma product obeys

        S_{j+1}/S_j = (j+1+gamma)/(j+2+2gamma+2iq).
    """
    if nmax < 0:
        return np.empty(0, dtype=np.complex128)

    gam = gamma_abcd(a, b, c, d)
    out = np.empty(nmax + 1, dtype=np.complex128)
    out[0] = Scomp_abcd(0, a, b, c, d)

    for j in range(nmax):
        out[j + 1] = (
            out[j]
            * (j + 1 + gam)
            / (j + 2 + 2 * gam + 2 * iq)
        )

    return out


def S_near_abcd(a, b, c, d):
    """Near-zone radial coefficient with slot order (a, bar(b), c, d).

    The P coefficients are generated once by recurrence, the four-mode
    coefficient table is built by balanced convolutions, the finite-part
    factors are generated by their own recurrence, and the final sum uses
    ``stable_sum``.
    """
    coeffs = C_abcd_table(a, b, c, d)
    scomp = Scomp_abcd_table(len(coeffs) - 1, a, b, c, d)

    norm = (
        1 / Anorm(a["n"], a["l"])
        / bar(Anorm(b["n"], b["l"]))
        / Anorm(c["n"], c["l"])
        / Anorm(d["n"], d["l"])
    )

    return norm * stable_sum(coeffs * scomp)


# Alias: ``S_near_abcd`` already uses the paper convention.
S_near_paper = S_near_abcd


def b_near(n: int, ell: int, xbreve):
    """Near-zone supplementary QNM mode function on NHERN.

    Uses the same recurrence-generated P table as the coupling coefficients
    and evaluates the finite polynomial by Horner's rule.
    """
    x = np.asarray(xbreve, dtype=complex)
    s_n = s(n, ell)

    # (-xbreve)^s via the upper-half branch ln(-1)=+i*pi.
    factor_neg = np.exp(1j * np.pi * s_n) * x**s_n
    factor_one_plus = (1.0 + x) ** (s_n + iq)

    coeffs = pTable(n, ell)
    poly = np.zeros_like(x, dtype=complex)
    for cj in coeffs[::-1]:
        poly = poly * (-x) + cj

    return factor_neg * factor_one_plus * poly / Anorm(n, ell)


# --------------------------------------------------------------------------
# Legacy scalar/product interfaces.
# --------------------------------------------------------------------------

def P(j, n, ell):
    """Backwards-compatible scalar accessor for P_j(n,ell)."""
    return PLM_cached(j, n, ell)


def P2Table(n, m, ell):
    return stableConvolve(pTable(n, ell), pTable(m, ell))


def P3Table(n, m, p, ell):
    # P_p P_m conjugate(P_n), matching the original P3 definition.
    return stableMultiConvolve(
        [
            pTable(p, ell),
            pTable(m, ell),
            np.conjugate(pTable(n, ell)),
        ]
    )


def P4Table(n, m, p, q_arg, ell):
    # P_n P_m P_p conjugate(P_q), matching the original P4 definition.
    return stableMultiConvolve(
        [
            pTable(n, ell),
            pTable(m, ell),
            pTable(p, ell),
            np.conjugate(pTable(q_arg, ell)),
        ]
    )


def P2(k, n, m, ell):
    return coeffAt(P2Table(n, m, ell), k)


def P3(i, n, m, p, ell):
    return coeffAt(P3Table(n, m, p, ell), i)


def P4(i, n, m, p, q_arg, ell):
    return coeffAt(P4Table(n, m, p, q_arg, ell), i)


def P2LMTable(k1, k2):
    return stableConvolve(pNLMTable(k1), pNLMTable(k2))


def P3LMTable(k1, k2, k3):
    # P_k1 P_k2 conjugate(P_k3).
    return stableMultiConvolve(
        [
            pNLMTable(k1),
            pNLMTable(k2),
            np.conjugate(pNLMTable(k3)),
        ]
    )


def P4LMTable(k1, k2, k3, K):
    # Preserve the original P4LM algebra: P_k1 P_k2 conjugate(P_k3) P_K.
    return stableMultiConvolve(
        [
            pNLMTable(k1),
            pNLMTable(k2),
            np.conjugate(pNLMTable(k3)),
            pNLMTable(K),
        ]
    )


def P3LMnTable(k1, k2, k3):
    # P_k3 P_k2 conjugate(P_k1).
    return stableMultiConvolve(
        [
            np.conjugate(pNLMTable(k1)),
            pNLMTable(k2),
            pNLMTable(k3),
        ]
    )


def P4LMnTable(k1, k2, k3, K):
    # Preserve the original P4LMn algebra:
    # P_k1 P_K P_k3 conjugate(P_k2).
    return stableMultiConvolve(
        [
            pNLMTable(k1),
            pNLMTable(K),
            pNLMTable(k3),
            np.conjugate(pNLMTable(k2)),
        ]
    )


def P2LM(k, k1, k2):
    return coeffAt(P2LMTable(k1, k2), k)


# Same formula as P2LM (legacy parity with the notebook).
P2LMn = P2LM


def P3LM(i, k1, k2, k3):
    return coeffAt(P3LMTable(k1, k2, k3), i)


def P3LMn(i, k1, k2, k3):
    return coeffAt(P3LMnTable(k1, k2, k3), i)


def P4LM(i, k1, k2, k3, K):
    """Original P4LM algebra: k3 is the barred polynomial factor."""
    return coeffAt(P4LMTable(k1, k2, k3, K), i)


def P4LMn(i, k1, k2, k3, K):
    """Original P4LMn algebra: k2 is the barred polynomial factor."""
    return coeffAt(P4LMnTable(k1, k2, k3, K), i)


def P4LMn_as_C_abcd(i, a, b, c, d):
    """Legacy P4LMn re-expressed in C_abcd's slot order.

    C_abcd(i,a,b,c,d) == P4LMn(i,a,b,d,c).
    """
    return P4LMn(i, a, b, d, c)


def _bar_last_table(k1, k2, k3, K):
    """Coefficient table of P_k1 P_k2 P_k3 conjugate(P_K)."""
    return stableMultiConvolve(
        [
            pNLMTable(k1),
            pNLMTable(k2),
            pNLMTable(k3),
            np.conjugate(pNLMTable(K)),
        ]
    )


def Scomp(j, ntot):
    """Legacy ell=0 finite-part factor parameterized by total overtone count."""
    c = -0.5 * (4.0 * h0 + 2.0 * iq + ntot)
    return (
        (-1) ** (j + c + 1)
        * gamma(1 + j + c)
        * gamma(-1 - j - 2 * c - 2 * iq)
    ) / gamma(-2 * iq - c)


def Scomp_table(ntot):
    """Legacy ell=0 finite-part factors generated by recurrence."""
    c = -0.5 * (4.0 * h0 + 2.0 * iq + ntot)
    out = np.empty(ntot + 1, dtype=np.complex128)
    out[0] = Scomp(0, ntot)

    for j in range(ntot):
        out[j + 1] = out[j] * (j + 1 + c) / (j + 2 + 2 * c + 2 * iq)

    return out


def ScompLM(j, k1, k2, k3, K):
    """Legacy SLM finite-part factor with K barred."""
    c = (
        s(k1["n"], k1["l"])
        + s(k2["n"], k2["l"])
        + s(k3["n"], k3["l"])
        + bar(s(K["n"], K["l"]))
    )
    return (
        (-1) ** (j + c + 1)
        * gamma(1 + j + c)
        * gamma(-1 - j - 2.0 * c - 2.0 * iq)
    ) / gamma(-2.0 * iq - c)


def ScompLM_table(k1, k2, k3, K):
    """Legacy SLM finite-part factors generated by recurrence."""
    ntot = k1["n"] + k2["n"] + k3["n"] + K["n"]
    c = (
        s(k1["n"], k1["l"])
        + s(k2["n"], k2["l"])
        + s(k3["n"], k3["l"])
        + bar(s(K["n"], K["l"]))
    )

    out = np.empty(ntot + 1, dtype=np.complex128)
    out[0] = ScompLM(0, k1, k2, k3, K)

    for j in range(ntot):
        out[j + 1] = out[j] * (j + 1 + c) / (j + 2 + 2 * c + 2 * iq)

    return out


def SLM(k1, k2, k3, K):
    """Legacy near-zone coefficient with LAST argument K barred.

    The old implementation mixed two different barred slots in the j=0 and
    j>0 terms.  This version implements the documented convention uniformly:

        P_k1 P_k2 P_k3 conjugate(P_K).

    It preserves the legacy finite-part normalization, so it still differs
    from ``S_near_abcd(k1,K,k2,k3)`` by the stated j-independent convention
    factor.
    """
    coeffs = _bar_last_table(k1, k2, k3, K)
    scomp = ScompLM_table(k1, k2, k3, K)

    norms = (
        1 / Anorm(k1["n"], k1["l"])
        / Anorm(k2["n"], k2["l"])
        / Anorm(k3["n"], k3["l"])
        / bar(Anorm(K["n"], K["l"]))
    )

    return norms * stable_sum(coeffs * scomp)


def S0(n1, n2, n3, N):
    """Legacy SLM specialization with all ell=0 modes."""
    ell = 0
    k1 = {"n": n1, "l": ell, "m": 0}
    k2 = {"n": n2, "l": ell, "m": 0}
    k3 = {"n": n3, "l": ell, "m": 0}
    K = {"n": N, "l": ell, "m": 0}

    coeffs = _bar_last_table(k1, k2, k3, K)
    scomp = Scomp_table(n1 + n2 + n3 + N)

    norms = (
        1 / Anorm(n1, ell)
        / Anorm(n2, ell)
        / Anorm(n3, ell)
        / bar(Anorm(N, ell))
    )

    return norms * stable_sum(coeffs * scomp)


def create_SLM_table(K, n_range=range(3), l_range=range(5)):
    """Batch-compute SLM over canonically-ordered (k1, k2, k3) triplets, fixed K.

    Returns a list of dicts with keys ``n1, l1, n2, l2, n3, l3, n4b, l4b, SLM_value``.
    """
    table = []
    for k1_n, k2_n, k3_n in itertools.product(n_range, repeat=3):
        for k1_l, k2_l, k3_l in itertools.product(l_range, repeat=3):
            k1 = {"n": k1_n, "l": k1_l, "m": 0}
            k2 = {"n": k2_n, "l": k2_l, "m": 0}
            k3 = {"n": k3_n, "l": k3_l, "m": 0}
            k1c, k2c, k3c = sorted(
                [k1, k2, k3], key=lambda x: (x["n"], x["l"])
            )
            table.append({
                "n1": k1c["n"], "l1": k1c["l"],
                "n2": k2c["n"], "l2": k2c["l"],
                "n3": k3c["n"], "l3": k3c["l"],
                "n4b": K["n"], "l4b": K["l"],
                "SLM_value": SLM(k1c, k2c, k3c, K),
            })
    return table


def create_SLM_table_optimized(K, n_range=range(1), l_range=range(5)):
    """As ``create_SLM_table`` but memoizes SLM by canonical key to skip repeats."""
    computed = {}
    table = []
    for k1_n, k2_n, k3_n in itertools.product(n_range, repeat=3):
        for k1_l, k2_l, k3_l in itertools.product(l_range, repeat=3):
            k1 = {"n": k1_n, "l": k1_l, "m": 0}
            k2 = {"n": k2_n, "l": k2_l, "m": 0}
            k3 = {"n": k3_n, "l": k3_l, "m": 0}
            k1c, k2c, k3c = sorted(
                [k1, k2, k3], key=lambda x: (x["n"], x["l"])
            )
            key = (k1c["n"], k1c["l"], k2c["n"], k2c["l"], k3c["n"], k3c["l"])
            if key not in computed:
                computed[key] = SLM(k1c, k2c, k3c, K)
            table.append({
                "n1": k1c["n"], "l1": k1c["l"],
                "n2": k2c["n"], "l2": k2c["l"],
                "n3": k3c["n"], "l3": k3c["l"],
                "SLM_value": computed[key],
            })
    return table
