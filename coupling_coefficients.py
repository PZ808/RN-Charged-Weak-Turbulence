"""Coupling-coefficient utilities extracted from ``coupling_coeffs.ipynb``.

Use the constructor when you want explicit physics parameters:

    from coupling_coefficients import CouplingCoefficients

    coeffs = CouplingCoefficients(q=0.02)
    C = coeffs.build_C_notes(modes, alpha=1.0)

For quick notebook use, the module-level functions call a default
``CouplingCoefficients()`` instance with the original notebook parameters.
"""

from __future__ import annotations

from cmath import sqrt as csqrt
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import gamma
from sympy import S
from sympy.physics.wigner import wigner_3j


@lru_cache(None)
def W3j(j1, j2, j3, m1, m2, m3):
    """
    Cached Wigner 3j symbol.

    Uses SymPy's exact implementation internally, then converts to float.
    """

    if m1 + m2 + m3 != 0:
        return 0.0

    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    return float(wigner_3j(S(j1), S(j2), S(j3), S(m1), S(m2), S(m3)))


@dataclass(frozen=True)
class CouplingCoefficients:
    """Calculator for angular, radial, and evolution-convention coefficients."""

    # default arguments
    q: float = 2/100.
    rp: float = 1.0
    Q: float = 1.0

    @property
    def iq(self):
        return 1j * self.q

    @property
    def h0(self):
        return 0.5 + csqrt(0.25 - self.q**2)

    @staticmethod
    def bar(f):
        """
        Calculate the complex conjugate of a function.
        """

        return np.conj(f)

    @staticmethod
    def poch(a, n):
        """
        Pochhammer symbol.
        """

        if type(n) == int:
            return (-1) ** n * gamma(1 - a) / gamma(1 - a - n)
        else:
            return gamma(a + n) / gamma(a)

    def h(self, l):
        d2 = 0.25 + l * (l + 1) - self.q**2
        if d2 > 0:
            return 0.5 + csqrt(d2)
        else:
            return 0.5 - csqrt(d2)

    def s(self, n, l):
        """
        qnm frequency in the variable s=-i omega.
        """

        return -(n + self.h(l) + self.iq) / 2.0

    def w(self, n, l):
        """
        qnm frequency in the variable s=-i omega.
        """

        return -1j * (n + self.h(l) + self.iq) / 2.0

    def Q_element(self, li, mi, lj, mj, lk, mk, lp, mp):
        """
        Compute

            Q^{ij}_{kp} = integral_{S^2} dOmega Y_i Y_j bar(Y)_k bar(Y)_p

        using the Wigner-3j formula from ``coupling_coeffs.ipynb``.
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
            if (li + lj + ell) % 2 != 0:
                continue
            if (lk + lp + ell) % 2 != 0:
                continue

            term = (
                phase
                * (2 * ell + 1)
                * pref
                * W3j(li, lj, ell, 0, 0, 0)
                * W3j(li, lj, ell, mi, mj, -m)
                * W3j(lk, lp, ell, 0, 0, 0)
                * W3j(lk, lp, ell, -mk, -mp, m)
            )

            total += term

        return total

    def Anorm(self, n, l):
        """Normalization factor for the near-zone radial functions."""

        tmp = (
            1j
            * (-1.0) ** (self.h(l) + self.iq)
            * 2.0
            * gamma(n + 1.0)
            * gamma(2 * self.h(l) + n)
            * gamma(1 - self.h(l) - n - self.iq)
        ) / (gamma(self.h(l) - self.iq) * self.poch(self.h(l) + self.iq, n))
        return csqrt(tmp)

    def P(self, j, n, ell):
        return self.poch(-n, j) * self.poch(1 - 2 * self.h(ell) - n, j) / (
            gamma(j + 1) * self.poch(1 - self.h(ell) - n - self.iq, j)
        )

    @lru_cache(None)
    def PLM_cached(self, j, n, ell):
        return (
            self.poch(-n, j)
            * self.poch(1 - 2 * self.h(ell) - n, j)
            / (gamma(j + 1) * self.poch(1 - self.h(ell) - n - self.iq, j))
        )

    def PLM(self, j, mode):
        return self.PLM_cached(j, mode["n"], mode["l"])

    def P2(self, k, n, m, ell):
        tmp = 0
        for j in range(0, k + 1):
            tmp += self.P(k - j, n, ell) * self.P(j, m, ell)
        return tmp

    def P3(self, i, n, m, p, ell):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P2(j, p, m, ell) * self.bar(self.P(i - j, n, ell))
        return tmp

    def P4(self, i, n, m, p, q_mode, ell):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P3(j, q_mode, p, m, ell) * self.P(i - j, n, ell)
        return tmp

    def P2LM(self, k, k1, k2):
        tmp = 0
        for j in range(0, k + 1):
            tmp += self.PLM(k - j, k1) * self.PLM(j, k2)
        return tmp

    def P3LM(self, i, k1, k2, k3):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P2LM(j, k1, k2) * self.bar(self.PLM(i - j, k3))
        return tmp

    def P4LM(self, i, k1, k2, k3, K):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P3LM(j, k1, k2, k3) * self.PLM(i - j, K)
        return tmp

    def P2LMn(self, k, k1, k2):
        tmp = 0
        for j in range(0, k + 1):
            tmp += self.PLM(k - j, k1) * self.PLM(j, k2)
        return tmp

    def P3LMn(self, i, k1, k2, k3):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P2LM(j, k3, k2) * self.bar(self.PLM(i - j, k1))
        return tmp

    def P4LMn(self, i, k1, k2, k3, K):
        tmp = 0
        for j in range(0, i + 1):
            tmp += self.P3LM(j, K, k3, k2) * self.PLM(i - j, k1)
        return tmp

    def C_abcd(self, i, a, b, c, d):
        """
        Coefficient C_i^{(a,b,c,d)} with the barred slot b.
        """

        total = 0

        for j in range(i + 1):
            for k in range(j + 1):
                for ell in range(k + 1):
                    total += (
                        self.PLM(i - j, a)
                        * self.bar(self.PLM(j - k, b))
                        * self.PLM(k - ell, c)
                        * self.PLM(ell, d)
                    )

        return total

    def P4LMn_as_C_abcd(self, i, a, b, c, d):
        return self.P4LMn(i, a, b, d, c)

    def gamma_abcd(self, a, b, c, d):
        """
        gamma_abcd = s_a + bar(s_b) + s_c + s_d.
        """

        return (
            self.s(a["n"], a["l"])
            + self.bar(self.s(b["n"], b["l"]))
            + self.s(c["n"], c["l"])
            + self.s(d["n"], d["l"])
        )

    def Scomp_abcd(self, j, a, b, c, d):
        """
        Finite-part coefficient (check S_abcd)_j.
        """

        gam = self.gamma_abcd(a, b, c, d)

        return (
            (-1) ** (j + gam + 1)
            * gamma(1 + j + gam)
            * gamma(-1 - j - 2 * gam - 2 * self.iq)
            / gamma(-2 * self.iq - gam)
        )

    def S_near_abcd(self, a, b, c, d):
        """
        Compute the near-zone coefficient with slots

            a, bar(b), c, d.

        This corresponds to the paper's C_i^{(a,b,c,d)} convention.
        """

        Ntot = a["n"] + b["n"] + c["n"] + d["n"]

        norm = (
            1
            / self.Anorm(a["n"], a["l"])
            / self.bar(self.Anorm(b["n"], b["l"]))
            / self.Anorm(c["n"], c["l"])
            / self.Anorm(d["n"], d["l"])
        )

        total = 0

        for j in range(Ntot + 1):
            total += self.C_abcd(j, a, b, c, d) * self.Scomp_abcd(
                j, a, b, c, d
            )

        return norm * total

    def Scomp(self, j, ntot):
        c = -0.5 * (4.0 * self.h0 + 2.0 * self.iq + ntot)
        denom = gamma(-2 * self.iq - c)

        return (
            (-1) ** (j + c + 1)
            * gamma(1 + j + c)
            * gamma(-1 - j - 2 * c - 2 * self.iq) ) / denom

    def ScompLM(self, j, k1, k2, k3, K):
        c = (
            self.s(k1["n"], k1["l"])
            + self.s(k2["n"], k2["l"])
            + self.s(k3["n"], k3["l"])
            + self.bar(self.s(K["n"], K["l"]))
        )
        denom = gamma(-2.0 * self.iq - c)

        return (
            (-1) ** (j + c + 1) 
            * gamma(1 + j + c)
            * gamma(-1 - j - 2.0 * c - 2.0 * self.iq)
        ) / denom

    def SLM(self, k1, k2, k3, K):
        ntot = k1["n"] + k2["n"] + k3["n"] + K["n"]

        norms = (
            1
            / self.Anorm(k1["n"], k1["l"])
            / self.Anorm(k2["n"], k2["l"])
            / self.Anorm(k3["n"], k3["l"])
            / self.bar(self.Anorm(K["n"], K["l"]))
        )

        Slm = self.P4LMn(0, k1, K, k2, k3) * self.ScompLM(0, k1, k2, k3, K)

        for i in range(1, ntot + 1):
            Slm += self.P4LM(i, k1, k2, k3, K) * self.ScompLM(
                i, k1, k2, k3, K
            )

        return norms * Slm

    def S0(self, n1, n2, n3, N):
        ntot = n1 + n2 + n3 + N

        ell = 0

        norms = (
            1
            / self.Anorm(n1, ell)
            / self.Anorm(n2, ell)
            / self.Anorm(n3, ell)
            / self.bar(self.Anorm(N, ell))
        )

        S = self.P4(0, n1, n2, n3, N, ell) * self.Scomp(0, ntot)

        for i in range(1, ntot + 1):
            S += self.P4(i, n1, n2, n3, N, ell) * self.Scomp(i, ntot)

        return norms * S

    def omega_mode(self, J):
        return self.w(J["n"], J["l"])

    def omega_array(self, modes):
        return np.array([self.omega_mode(J) for J in modes], dtype=complex)

    def Q_notes(self, i, j, k, lam, modes):
        """
        Q^{ij}_{k lam} = integral Y_i Y_j bar(Y_k) bar(Y_lam).
        """

        mi = modes[i]
        mj = modes[j]
        mk = modes[k]
        ml = modes[lam]

        return self.Q_element(
            mi["l"],
            mi.get("m", 0),
            mj["l"],
            mj.get("m", 0),
            mk["l"],
            mk.get("m", 0),
            ml["l"],
            ml.get("m", 0),
        )

    def S_notes(self, i, j, k, lam, modes):
        """
        Notes convention:

            S^{ij lam}_k ~ b_i b_j bar(b_k) b_lam

        S_near_abcd(a,b,c,d) has b barred, so

            S_notes(i,j,k,lam) = S_near_abcd(i,k,j,lam).
        """

        return self.S_near_abcd(modes[i], modes[k], modes[j], modes[lam])

    def coupling_entry(self, lam, i, j, k, modes, alpha=1.0):
        """
        C[lam,i,j,k] = c^{ij}_{lam,k}
                     = alpha Q^{ij}_{lam,k} S^{ij lam}_k.
        """

        ki = modes[i]
        kj = modes[j]
        kk = modes[k]
        klam = modes[lam]

        Qi = self.Q_element(
            ki["l"],
            ki.get("m", 0),
            kj["l"],
            kj.get("m", 0),
            klam["l"],
            klam.get("m", 0),
            kk["l"],
            kk.get("m", 0),
        )

        Si = self.S_near_abcd(ki, kk, kj, klam)

        return alpha * Qi * Si

    def build_coupling_tensor(self, modes, alpha=1.0):
        N = len(modes)
        C = np.zeros((N, N, N, N), dtype=complex)

        for lam in range(N):
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        C[lam, i, j, k] = self.coupling_entry(
                            lam, i, j, k, modes=modes, alpha=alpha
                        )

        return C

    def build_C_notes(self, modes, alpha=1.0):
        """
        Build C[lam,i,j,k] = c^{ij}_{lam,k}

        so that

            i A_lam' = sum_{i,j,k}
                       C[lam,i,j,k] A_i A_j bar(A_k)
                       exp(i Omega[lam,i,j,k] t).
        """

        N = len(modes)
        C = np.zeros((N, N, N, N), dtype=complex)

        for lam in range(N):
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        C[lam, i, j, k] = (
                            alpha
                            * self.Q_notes(i, j, k, lam, modes=modes)
                            * self.S_notes(i, j, k, lam, modes=modes)
                        )

        return C

    @staticmethod
    def build_Omega_notes(omega_arr, real_phase=False):
        """
        Omega[lam,i,j,k] = omega_lam + omega_k - omega_i - omega_j.
        """

        if real_phase:
            omega_arr = np.real(omega_arr)

        return (
            omega_arr[:, None, None, None]
            - omega_arr[None, :, None, None]
            - omega_arr[None, None, :, None]
            + omega_arr[None, None, None, :]
        )


_DEFAULT = CouplingCoefficients()


def bar(f):
    return CouplingCoefficients.bar(f)


def poch(a, n):
    return CouplingCoefficients.poch(a, n)


def h(l):
    return _DEFAULT.h(l)


def s(n, l):
    return _DEFAULT.s(n, l)


def w(n, l):
    return _DEFAULT.w(n, l)


def Q_element(li, mi, lj, mj, lk, mk, lp, mp):
    return _DEFAULT.Q_element(li, mi, lj, mj, lk, mk, lp, mp)


def Anorm(n, l):
    return _DEFAULT.Anorm(n, l)


def P(j, n, ell):
    return _DEFAULT.P(j, n, ell)


def PLM_cached(j, n, ell):
    return _DEFAULT.PLM_cached(j, n, ell)


def PLM(j, mode):
    return _DEFAULT.PLM(j, mode)


def P2(k, n, m, ell):
    return _DEFAULT.P2(k, n, m, ell)


def P3(i, n, m, p, ell):
    return _DEFAULT.P3(i, n, m, p, ell)


def P4(i, n, m, p, q_mode, ell):
    return _DEFAULT.P4(i, n, m, p, q_mode, ell)


def P2LM(k, k1, k2):
    return _DEFAULT.P2LM(k, k1, k2)


def P3LM(i, k1, k2, k3):
    return _DEFAULT.P3LM(i, k1, k2, k3)


def P4LM(i, k1, k2, k3, K):
    return _DEFAULT.P4LM(i, k1, k2, k3, K)


def P2LMn(k, k1, k2):
    return _DEFAULT.P2LMn(k, k1, k2)


def P3LMn(i, k1, k2, k3):
    return _DEFAULT.P3LMn(i, k1, k2, k3)


def P4LMn(i, k1, k2, k3, K):
    return _DEFAULT.P4LMn(i, k1, k2, k3, K)


def C_abcd(i, a, b, c, d):
    return _DEFAULT.C_abcd(i, a, b, c, d)


def P4LMn_as_C_abcd(i, a, b, c, d):
    return _DEFAULT.P4LMn_as_C_abcd(i, a, b, c, d)


def gamma_abcd(a, b, c, d):
    return _DEFAULT.gamma_abcd(a, b, c, d)


def Scomp_abcd(j, a, b, c, d):
    return _DEFAULT.Scomp_abcd(j, a, b, c, d)


def S_near_abcd(a, b, c, d):
    return _DEFAULT.S_near_abcd(a, b, c, d)


def Scomp(j, ntot):
    return _DEFAULT.Scomp(j, ntot)


def ScompLM(j, k1, k2, k3, K):
    return _DEFAULT.ScompLM(j, k1, k2, k3, K)


def SLM(k1, k2, k3, K):
    return _DEFAULT.SLM(k1, k2, k3, K)


def S0(n1, n2, n3, N):
    return _DEFAULT.S0(n1, n2, n3, N)


def omega_mode(J):
    return _DEFAULT.omega_mode(J)


def omega_array(modes):
    return _DEFAULT.omega_array(modes)


def Q_notes(i, j, k, lam, modes):
    return _DEFAULT.Q_notes(i, j, k, lam, modes)


def S_notes(i, j, k, lam, modes):
    return _DEFAULT.S_notes(i, j, k, lam, modes)


def coupling_entry(lam, i, j, k, modes, alpha=1.0):
    return _DEFAULT.coupling_entry(lam, i, j, k, modes, alpha=alpha)


def build_coupling_tensor(modes, alpha=1.0):
    return _DEFAULT.build_coupling_tensor(modes, alpha=alpha)


def build_C_notes(modes, alpha=1.0):
    return _DEFAULT.build_C_notes(modes, alpha=alpha)


def build_Omega_notes(omega_arr, real_phase=False):
    return CouplingCoefficients.build_Omega_notes(omega_arr, real_phase=real_phase)


__all__ = [
    "CouplingCoefficients",
    "W3j",
    "bar",
    "h",
    "s",
    "w",
    "poch",
    "Q_element",
    "Anorm",
    "P",
    "PLM_cached",
    "PLM",
    "P2",
    "P3",
    "P4",
    "P2LM",
    "P3LM",
    "P4LM",
    "P2LMn",
    "P3LMn",
    "P4LMn",
    "C_abcd",
    "P4LMn_as_C_abcd",
    "gamma_abcd",
    "Scomp_abcd",
    "S_near_abcd",
    "Scomp",
    "ScompLM",
    "SLM",
    "S0",
    "omega_mode",
    "omega_array",
    "Q_notes",
    "S_notes",
    "coupling_entry",
    "build_coupling_tensor",
    "build_C_notes",
    "build_Omega_notes",
]
