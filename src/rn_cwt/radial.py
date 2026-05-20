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
# Canonical convention — barred slot SECOND (used by coupling.py).
# --------------------------------------------------------------------------

@lru_cache(None)
def PLM_cached(j, n, ell):
    """j-th series coefficient for the (n, ell) mode (cached on (j, n, ell))."""
    return (
        poch(-n, j)
        * poch(1 - 2 * h(ell) - n, j)
        / (gamma(j + 1) * poch(1 - h(ell) - n - iq, j))
    )


def PLM(j, mode):
    """j-th series coefficient for ``mode = {'n': n, 'l': ell}``."""
    return PLM_cached(j, mode["n"], mode["l"])


def C_abcd(i, a, b, c, d):
    """i-th convolution coefficient with barred slot ``b``.

        C_i^{(a,b,c,d)} = sum_{j>=k>=ell>=0}
            PLM(i-j, a) * bar(PLM(j-k, b)) * PLM(k-ell, c) * PLM(ell, d).

    This is the i-th coefficient of the formal product of four (3 unbarred,
    1 barred) PLM series.
    """
    total = 0
    for j in range(i + 1):
        for k in range(j + 1):
            for ell in range(k + 1):
                total += (
                    PLM(i - j, a)
                    * bar(PLM(j - k, b))
                    * PLM(k - ell, c)
                    * PLM(ell, d)
                )
    return total


def gamma_abcd(a, b, c, d):
    """gamma_abcd = s_a + bar(s_b) + s_c + s_d (barred slot is b)."""
    return (
        s(a["n"], a["l"])
        + bar(s(b["n"], b["l"]))
        + s(c["n"], c["l"])
        + s(d["n"], d["l"])
    )


def Scomp_abcd(j, a, b, c, d):
    """Finite-part factor (S_abcd)_j with barred slot b.

    Uses the paper/Mathematica convention (`SnearComponentNLM4` in
    ``Mathematica/generate_4wave_coeffs.nb``):

        (-1)^{j+γ} Γ(1+j+γ) Γ(-1-j-2γ-2iq) / [Γ(-2iq-γ)]²

    The original Python notebook used a different finite-part regularization
    that differed by an overall j-independent factor ``-1 / Γ(-2iq-γ)`` —
    both are valid choices, and we standardize on the paper form here so
    that ``S_near_abcd`` matches Peter's precomputed data file directly.
    """
    gam = gamma_abcd(a, b, c, d)
    return (
        (-1) ** (j + gam)
        * gamma(1 + j + gam)
        * gamma(-1 - j - 2 * gam - 2 * iq)
        / gamma(-2 * iq - gam) ** 2
    )


def S_near_abcd(a, b, c, d):
    """Near-zone radial coefficient with slot order (a, bar(b), c, d).

    Sums j up to N_tot = sum of n's; pre-divides by Anorm products. This is
    the **canonical** convention used by ``coupling.build_C_notes``.
    """
    Ntot = a["n"] + b["n"] + c["n"] + d["n"]

    norm = (
        1 / Anorm(a["n"], a["l"])
        / bar(Anorm(b["n"], b["l"]))
        / Anorm(c["n"], c["l"])
        / Anorm(d["n"], d["l"])
    )

    total = 0
    for j in range(Ntot + 1):
        total += C_abcd(j, a, b, c, d) * Scomp_abcd(j, a, b, c, d)

    return norm * total


# Alias: ``S_near_abcd`` already uses the paper convention. Kept as an
# explicit name for callers that want to flag the convention they're using.
S_near_paper = S_near_abcd


def b_near(n: int, ell: int, xbreve):
    """Near-zone supplementary QNM mode function on NHERN.

    From the draft paper §10, eq. 2346-2347 (in r_+ = 1 units):

        b_n^{near}(x̆) = (1 / Anorm(n, ℓ)) · (-x̆)^{s̆_n} (1 + x̆)^{s̆_n + i q}
                        · sum_{j=0}^n PLM(j, n, ℓ) (-x̆)^j

    where s̆_n = s(n, ℓ) = -(h(ℓ) + n + i q)/2 is the supplementary-mode
    frequency. The polynomial coefficients PLM(j, n, ℓ) are exactly the
    notebook's series coefficients (paper's P_j^{(n)}), and the normalization
    1/Anorm matches the paper's N_n (eq. 2459) in r_+ = 1 units.

    Branch convention: (-x̆)^z = e^{i π z} · x̆^z, the upper-half branch
    ln(-1) = +iπ. For integer j the factor (-x̆)^j collapses to (-1)^j x̆^j
    and is branch-free; only the (-x̆)^{s̆_n} factor sees the branch.

    Parameters
    ----------
    n : int   overtone index (n = 0, 1, 2, ...).
    ell : int  angular index (ℓ = 0, 1, ...).
    xbreve : scalar or array of positive real x̆ values.

    Returns
    -------
    Complex scalar or ndarray of the mode amplitude at each x̆.

    Notes
    -----
    For ℓ small enough that h(ℓ) is in the supplementary range (h_+
    branch, q² < 1/4), the mode diverges at the horizon as x̆^{Re(s̆_n)}
    with Re(s̆_n) < 0, and decays at the throat boundary as x̆^{-(h+n)}.
    Numerical sampling should stay strictly away from x̆ = 0.
    """
    x = np.asarray(xbreve, dtype=complex)
    s_n = s(n, ell)

    # (-x̆)^{s̆_n} via upper branch.
    factor_neg = np.exp(1j * np.pi * s_n) * x**s_n
    # (1 + x̆)^{s̆_n + i q} — argument is real positive, no branch issue.
    factor_one_plus = (1.0 + x) ** (s_n + iq)

    # Polynomial sum: sum_j PLM(j, n, ℓ) (-x̆)^j = sum_j PLM(j, n, ℓ) (-1)^j x̆^j
    poly = np.zeros_like(x, dtype=complex)
    for j in range(n + 1):
        poly = poly + PLM_cached(j, n, ell) * ((-1) ** j) * x**j

    return factor_neg * factor_one_plus * poly / Anorm(n, ell)


# --------------------------------------------------------------------------
# Legacy convention — barred slot LAST (paper ordering).
# Kept for the P4LMn_as_C_abcd cross-check and batch table generation.
# --------------------------------------------------------------------------

def P(j, n, ell):
    """Uncached form of PLM_cached. Same formula, tuple args."""
    return (
        poch(-n, j)
        * poch(1 - 2 * h(ell) - n, j)
        / (gamma(j + 1) * poch(1 - h(ell) - n - iq, j))
    )


def P2(k, n, m, ell):
    tmp = 0
    for j in range(0, k + 1):
        tmp += P(k - j, n, ell) * P(j, m, ell)
    return tmp


def P3(i, n, m, p, ell):
    tmp = 0
    for j in range(0, i + 1):
        tmp += P2(j, p, m, ell) * bar(P(i - j, n, ell))
    return tmp


def P4(i, n, m, p, q_arg, ell):
    tmp = 0
    for j in range(0, i + 1):
        tmp += P3(j, q_arg, p, m, ell) * P(i - j, n, ell)
    return tmp


def P2LM(k, k1, k2):
    """Mode-dict P2 with both unbarred."""
    tmp = 0
    for j in range(0, k + 1):
        tmp += PLM(k - j, k1) * PLM(j, k2)
    return tmp


# P2LMn is the same formula as P2LM (alias for legacy parity with notebook).
P2LMn = P2LM


def P3LM(i, k1, k2, k3):
    """Mode-dict P3 with barred slot LAST (k3 barred)."""
    tmp = 0
    for j in range(0, i + 1):
        tmp += P2LM(j, k1, k2) * bar(PLM(i - j, k3))
    return tmp


def P3LMn(i, k1, k2, k3):
    """Mode-dict P3 with barred slot FIRST (k1 barred)."""
    tmp = 0
    for j in range(0, i + 1):
        tmp += P2LM(j, k3, k2) * bar(PLM(i - j, k1))
    return tmp


def P4LM(i, k1, k2, k3, K):
    """Mode-dict P4 with barred slot LAST (K barred)."""
    tmp = 0
    for j in range(0, i + 1):
        tmp += P3LM(j, k1, k2, k3) * PLM(i - j, K)
    return tmp


def P4LMn(i, k1, k2, k3, K):
    """Mode-dict P4 with barred slot in position 2 (K barred via P3LM(K,...))."""
    tmp = 0
    for j in range(0, i + 1):
        tmp += P3LM(j, K, k3, k2) * PLM(i - j, k1)
    return tmp


def P4LMn_as_C_abcd(i, a, b, c, d):
    """Cross-check: legacy P4LMn re-expressed in C_abcd's slot order.

    ``C_abcd(i, a, b, c, d) == P4LMn(i, a, b, d, c)`` must hold for any modes;
    this is verified in tests/test_radial.py.
    """
    return P4LMn(i, a, b, d, c)


def Scomp(j, ntot):
    """Legacy l=0 finite-part factor parameterized by total overtone count.

    Uses the module-level h0 (the l=0 conformal weight).
    """
    c = -0.5 * (4.0 * h0 + 2.0 * iq + ntot)
    return (
        (-1) ** (j + c + 1)
        * gamma(1 + j + c)
        * gamma(-1 - j - 2 * c - 2 * iq)
    ) / gamma(-2 * iq - c)


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


def SLM(k1, k2, k3, K):
    """Legacy near-zone coefficient with the LAST argument K barred.

    Equivalent to S_near_abcd(k1, K, k2, k3) up to convention. Used by the
    SLM table helpers below.
    """
    ntot = k1["n"] + k2["n"] + k3["n"] + K["n"]
    norms = (
        1
        / Anorm(k1["n"], k1["l"])
        / Anorm(k2["n"], k2["l"])
        / Anorm(k3["n"], k3["l"])
        / bar(Anorm(K["n"], K["l"]))
    )
    Slm = P4LMn(0, k1, K, k2, k3) * ScompLM(0, k1, k2, k3, K)
    for i in range(1, ntot + 1):
        Slm += P4LM(i, k1, k2, k3, K) * ScompLM(i, k1, k2, k3, K)
    return norms * Slm


def S0(n1, n2, n3, N):
    """Legacy SLM for all l=0 modes, parameterized by overtone numbers only."""
    ntot = n1 + n2 + n3 + N
    ell = 0
    norms = (
        1 / Anorm(n1, ell) / Anorm(n2, ell) / Anorm(n3, ell) / bar(Anorm(N, ell))
    )
    S_val = P4(0, n1, n2, n3, N, ell) * Scomp(0, ntot)
    for i in range(1, ntot + 1):
        S_val += P4(i, n1, n2, n3, N, ell) * Scomp(i, ntot)
    return norms * S_val


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
