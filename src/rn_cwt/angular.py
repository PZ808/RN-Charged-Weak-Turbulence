"""Angular overlap coefficient Q via Wigner-3j / Gaunt expansion.

The key object is

    Q^{ij}_{kp} = int_{S^2} Y_i Y_j bar(Y_k) bar(Y_p) dOmega.

Computed as a single sum over an internal multipole ell with a magnetic
selection rule m_i + m_j = m_k + m_p.
"""
from functools import lru_cache

import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j


@lru_cache(None)
def W3j(j1, j2, j3, m1, m2, m3):
    """Cached Wigner-3j symbol.

    Computed exactly with SymPy and returned as a float. The selection rule
    m1 + m2 + m3 == 0, magnetic bounds |m_i| <= j_i, and the triangle
    inequality are checked first to short-circuit zero contributions.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    return float(wigner_3j(
        S(j1), S(j2), S(j3),
        S(m1), S(m2), S(m3),
    ))


def gaunt(l1, l2, l3, m1, m2, m3):
    """Gaunt integral int Y_{l1 m1} Y_{l2 m2} Y_{l3 m3} dOmega."""
    return (
        np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
        * W3j(l1, l2, l3, 0, 0, 0)
        * W3j(l1, l2, l3, m1, m2, m3)
    )


def Q_element(li, mi, lj, mj, lk, mk, lp, mp):
    """Q^{ij}_{kp} = int Y_i Y_j bar(Y_k) bar(Y_p) dOmega.

    Expansion (see notebook header):

        Q^{ij}_{kp} = sum_ell (-1)^{m + mk + mp}
                      (2 ell + 1) / (4 pi)
                      * sqrt[(2 li+1)(2 lj+1)(2 lk+1)(2 lp+1)]
                      * (li lj ell; 0 0 0)(li lj ell; mi mj -m)
                      * (lk lp ell; 0 0 0)(lk lp ell; -mk -mp m)

    where m = mi + mj = mk + mp (magnetic selection rule).
    """
    if mi + mj != mk + mp:
        return 0.0

    m = mi + mj
    ell_min = max(abs(li - lj), abs(lk - lp), abs(m))
    ell_max = min(li + lj, lk + lp)

    if ell_min > ell_max:
        return 0.0

    pref = np.sqrt(
        (2 * li + 1) * (2 * lj + 1) * (2 * lk + 1) * (2 * lp + 1)
    ) / (4 * np.pi)
    phase = (-1) ** (m + mk + mp)

    total = 0.0
    for ell in range(ell_min, ell_max + 1):
        # (l1 l2 ell; 0 0 0) vanishes unless l1 + l2 + ell is even.
        if (li + lj + ell) % 2 != 0:
            continue
        if (lk + lp + ell) % 2 != 0:
            continue

        total += (
            phase
            * (2 * ell + 1)
            * pref
            * W3j(li, lj, ell, 0, 0, 0)
            * W3j(li, lj, ell, mi, mj, -m)
            * W3j(lk, lp, ell, 0, 0, 0)
            * W3j(lk, lp, ell, -mk, -mp, m)
        )

    return total
