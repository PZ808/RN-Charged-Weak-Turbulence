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
from math import fsum
from functools import lru_cache

import numpy as np
from scipy.special import gamma
from sympy import S
from sympy.physics.wigner import wigner_3j

from rn_cwt.diagnostics import mode_label


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
class ModeUtilities:

    @staticmethod       
    def mode_label(k):
        return f"n={k['n']}, l={k['l']}, m={k.get('m', 0)}"

    @staticmethod
    def detuning_notes(modes, omega_arr, lam, i, j, k):

        return omega_arr[lam] + omega_arr[k] - omega_arr[i] - omega_arr[j]

    @staticmethod
    def analyze_mode_set(modes, omega_arr, C=None, detuning_tol=1e-8, coupling_tol=1e-14):
        N = len(modes)

        print("Modes:")
        for a, mode in enumerate(modes):
            print(a, mode_label(mode), "omega =", omega_arr[a])

        print("\nActive angular/resonant quartets:")

        count_by_mode = np.zeros(N, dtype=int)

        quartets = []

        for lam in range(N):
            for i in range(N):
                for j in range(N):
                    for k in range(N):

                        # magnetic selection
                        m_lam = modes[lam].get("m", 0)
                        m_i = modes[i].get("m", 0)
                        m_j = modes[j].get("m", 0)
                        m_k = modes[k].get("m", 0)

                        if m_i + m_j != m_k + m_lam:
                            continue

                        Om = detuning_notes(modes, omega_arr, lam, i, j, k)

                        if abs(np.real(Om)) > detuning_tol:
                            continue

                        if C is not None and abs(C[lam, i, j, k]) < coupling_tol:
                            continue

                        quartets.append((lam, i, j, k, Om))

                        count_by_mode[lam] += 1
                        count_by_mode[i] += 1
                        count_by_mode[j] += 1
                        count_by_mode[k] += 1

        print("number of selected quartets =", len(quartets))

        print("\nParticipation counts:")
        for a, c in enumerate(count_by_mode):
            print(a, mode_label(modes[a]), c)

        return quartets, count_by_mode


def hermitian_adjoint_notes(C):
    """
    Cdag[lam,i,j,k] = conjugate(C[i,lam,k,j])
    """
    return np.conjugate(np.transpose(C, (1, 0, 3, 2)))


def coupling_hermitian_defect(C, mask=None):
    if mask is None:
        mask = np.ones_like(C, dtype=bool)

    Cdag = hermitian_adjoint_notes(C)

    CH = 0.5 * (C + Cdag)
    CA = 0.5 * (C - Cdag)

    norm_H = np.linalg.norm(CH[mask])
    norm_A = np.linalg.norm(CA[mask])

    return norm_A / norm_H, norm_H, norm_A

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

    # ------------------------------------------------------------
    # Stable radial-polynomial coefficients and convolutions
    # ------------------------------------------------------------

    @staticmethod
    def stable_sum(z):
        """
        Accurate summation of complex double-precision numbers.

        Real and imaginary parts are accumulated independently with
        math.fsum.  Sorting by magnitude is a small additional guard
        against cancellation.
        """
        z = list(z)
        z.sort(key=abs)

        return complex(
            fsum(complex(x).real for x in z),
            fsum(complex(x).imag for x in z),
        )

    @staticmethod
    def stableConvolve(a, b):
        """
        Polynomial convolution with compensated summation.

        If
            A(x) = sum_i a[i] x^i
            B(x) = sum_j b[j] x^j,

        return the coefficient array of A(x) B(x).
        """
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)

        na = len(a)
        nb = len(b)
        out = np.empty(na + nb - 1, dtype=np.complex128)

        for k in range(na + nb - 1):
            jmin = max(0, k - (nb - 1))
            jmax = min(k, na - 1)

            terms = [
                a[j] * b[k - j]
                for j in range(jmin, jmax + 1)
            ]

            out[k] = CouplingCoefficients.stable_sum(terms)

        return out

    @staticmethod
    def stableMultiConvolve(arrays):
        """
        Balanced-tree convolution of an arbitrary number of coefficient
        arrays.
        """
        arrays = [np.asarray(a, dtype=np.complex128) for a in arrays]

        if len(arrays) == 0:
            return np.array([1.0 + 0.0j], dtype=np.complex128)

        while len(arrays) > 1:
            new_arrays = []

            for j in range(0, len(arrays) - 1, 2):
                new_arrays.append(
                    CouplingCoefficients.stableConvolve(
                        arrays[j], arrays[j + 1]
                    )
                )

            if len(arrays) % 2:
                new_arrays.append(arrays[-1])

            arrays = new_arrays

        return arrays[0]

    @staticmethod
    def coeffAt(a, j):
        """Return coefficient j, or zero outside the polynomial range."""
        if 0 <= j < len(a):
            return a[j]
        return 0.0 + 0.0j

    @lru_cache(None)
    def _pTable_cached(self, n, ell):
        """
        Cached tuple (P_0,...,P_n) generated from the exact first-order
        recurrence

            P_{j+1}/P_j =
              (-n+j) (1-2h-n+j)
              -----------------------------
              (j+1) (1-h-n-iq+j).

        This avoids separately evaluating large Pochhammer/Gamma ratios.
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        hh = self.h(ell)

        out = [1.0 + 0.0j]

        for j in range(n):
            out.append(
                out[-1]
                * (-n + j)
                * (1 - 2 * hh - n + j)
                / (
                    (j + 1)
                    * (1 - hh - n - self.iq + j)
                )
            )

        return tuple(out)

    def pTable(self, n, ell):
        """
        Return [P_0,...,P_n] as a complex NumPy array.
        """
        return np.asarray(self._pTable_cached(n, ell), dtype=np.complex128)

    def pNLMTable(self, mode):
        """
        Return [P_0,...,P_n] for a mode dictionary with keys 'n' and 'l'.
        """
        return self.pTable(mode["n"], mode["l"])

    # Backwards-compatible scalar coefficient accessors -----------------

    def P(self, j, n, ell):
        return self.coeffAt(self._pTable_cached(n, ell), j)

    @lru_cache(None)
    def PLM_cached(self, j, n, ell):
        return self.coeffAt(self._pTable_cached(n, ell), j)

    def PLM(self, j, mode):
        return self.PLM_cached(j, mode["n"], mode["l"])

    # Product coefficient tables ----------------------------------------

    def P2Table(self, n, m, ell):
        return self.stableConvolve(
            self.pTable(n, ell),
            self.pTable(m, ell),
        )

    def P3Table(self, n, m, p, ell):
        # P_p P_m \bar P_n, matching the original P3 convention.
        return self.stableMultiConvolve(
            [
                self.pTable(p, ell),
                self.pTable(m, ell),
                np.conjugate(self.pTable(n, ell)),
            ]
        )

    def P4Table(self, n, m, p, q_mode, ell):
        # P_n P_m P_p \bar P_q, matching the original P4 convention.
        return self.stableMultiConvolve(
            [
                self.pTable(n, ell),
                self.pTable(m, ell),
                self.pTable(p, ell),
                np.conjugate(self.pTable(q_mode, ell)),
            ]
        )

    def P2(self, k, n, m, ell):
        return self.coeffAt(self.P2Table(n, m, ell), k)

    def P3(self, i, n, m, p, ell):
        return self.coeffAt(self.P3Table(n, m, p, ell), i)

    def P4(self, i, n, m, p, q_mode, ell):
        return self.coeffAt(self.P4Table(n, m, p, q_mode, ell), i)

    def P2LMTable(self, k1, k2):
        return self.stableConvolve(
            self.pNLMTable(k1),
            self.pNLMTable(k2),
        )

    def P3LMTable(self, k1, k2, k3):
        # P_k1 P_k2 \bar P_k3.
        return self.stableMultiConvolve(
            [
                self.pNLMTable(k1),
                self.pNLMTable(k2),
                np.conjugate(self.pNLMTable(k3)),
            ]
        )

    def P4LMTable(self, k1, k2, k3, K):
        # P_k1 P_k2 \bar P_k3 P_K.
        return self.stableMultiConvolve(
            [
                self.pNLMTable(k1),
                self.pNLMTable(k2),
                np.conjugate(self.pNLMTable(k3)),
                self.pNLMTable(K),
            ]
        )

    def P2LM(self, k, k1, k2):
        return self.coeffAt(self.P2LMTable(k1, k2), k)

    def P3LM(self, i, k1, k2, k3):
        return self.coeffAt(self.P3LMTable(k1, k2, k3), i)

    def P4LM(self, i, k1, k2, k3, K):
        return self.coeffAt(self.P4LMTable(k1, k2, k3, K), i)

    def P2LMnTable(self, k1, k2):
        return self.P2LMTable(k1, k2)

    def P3LMnTable(self, k1, k2, k3):
        # P_k3 P_k2 \bar P_k1, matching the original P3LMn convention.
        return self.stableMultiConvolve(
            [
                np.conjugate(self.pNLMTable(k1)),
                self.pNLMTable(k2),
                self.pNLMTable(k3),
            ]
        )

    def P4LMnTable(self, k1, k2, k3, K):
        # Original P4LMn called P3LM(K,k3,k2), hence
        # P_k1 P_K P_k3 \bar P_k2.
        return self.stableMultiConvolve(
            [
                self.pNLMTable(k1),
                self.pNLMTable(K),
                self.pNLMTable(k3),
                np.conjugate(self.pNLMTable(k2)),
            ]
        )

    def P2LMn(self, k, k1, k2):
        return self.coeffAt(self.P2LMnTable(k1, k2), k)

    def P3LMn(self, i, k1, k2, k3):
        return self.coeffAt(self.P3LMnTable(k1, k2, k3), i)

    def P4LMn(self, i, k1, k2, k3, K):
        return self.coeffAt(self.P4LMnTable(k1, k2, k3, K), i)

    # Canonical four-mode polynomial ------------------------------------

    def C_abcd_table(self, a, b, c, d):
        """
        All coefficients C_j^{(a,b,c,d)} for

            P_a(x) conjugate(P_b(x)) P_c(x) P_d(x).

        The result has length
            n_a + n_b + n_c + n_d + 1.
        """
        return self.stableMultiConvolve(
            [
                self.pNLMTable(a),
                np.conjugate(self.pNLMTable(b)),
                self.pNLMTable(c),
                self.pNLMTable(d),
            ]
        )

    def C_abcd(self, i, a, b, c, d):
        return self.coeffAt(self.C_abcd_table(a, b, c, d), i)

    def P4LMn_as_C_abcd(self, i, a, b, c, d):
        return self.C_abcd(i, a, b, c, d)

    # S coefficient ------------------------------------------------------

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
        Direct finite-part coefficient (check S_abcd)_j.

        Scomp_abcd_table() should be preferred when many j values are
        needed, because it evaluates this Gamma expression only at j=0
        and obtains the rest by recurrence.
        """
        gam = self.gamma_abcd(a, b, c, d)

        return (
            (-1) ** (j + gam + 1)
            * gamma(1 + j + gam)
            * gamma(-1 - j - 2 * gam - 2 * self.iq)
            / gamma(-2 * self.iq - gam)
        )

    def Scomp_abcd_table(self, nmax, a, b, c, d):
        """
        Return [(check S)_0,...,(check S)_nmax] using

            S_{j+1}/S_j
              = (j+1+gamma)/(j+2+2 gamma+2 iq).

        This avoids repeatedly evaluating Gamma products at every j.
        """
        if nmax < 0:
            return np.empty(0, dtype=np.complex128)

        gam = self.gamma_abcd(a, b, c, d)

        out = np.empty(nmax + 1, dtype=np.complex128)
        out[0] = self.Scomp_abcd(0, a, b, c, d)

        for j in range(nmax):
            out[j + 1] = (
                out[j]
                * (j + 1 + gam)
                / (j + 2 + 2 * gam + 2 * self.iq)
            )

        return out

    def S_near_abcd(self, a, b, c, d):
        """
        Compute the near-zone coefficient for

            a, bar(b), c, d.

        Both ingredients are generated as full coefficient tables:
        C_j by stable balanced convolutions, and S_j by a first-order
        recurrence.  The final finite sum is accumulated with stable_sum.
        """
        coeffs = self.C_abcd_table(a, b, c, d)
        scomp = self.Scomp_abcd_table(len(coeffs) - 1, a, b, c, d)

        norm = (
            1
            / self.Anorm(a["n"], a["l"])
            / self.bar(self.Anorm(b["n"], b["l"]))
            / self.Anorm(c["n"], c["l"])
            / self.Anorm(d["n"], d["l"])
        )

        terms = coeffs * scomp
        return norm * self.stable_sum(terms)

    # Legacy interfaces now routed through the canonical stable S --------

    def Scomp(self, j, ntot):
        c = -0.5 * (4.0 * self.h0 + 2.0 * self.iq + ntot)
        denom = gamma(-2 * self.iq - c)

        return (
            (-1) ** (j + c + 1)
            * gamma(1 + j + c)
            * gamma(-1 - j - 2 * c - 2 * self.iq)
        ) / denom

    def ScompLM(self, j, k1, k2, k3, K):
        # Same gamma as S_near_abcd(k1, K, k2, k3).
        return self.Scomp_abcd(j, k1, K, k2, k3)

    def SLM(self, k1, k2, k3, K):
        """
        Stable LM coefficient with K in the barred slot:

            P_k1 P_k2 P_k3 bar(P_K).

        This is exactly S_near_abcd(k1, K, k2, k3).
        """
        return self.S_near_abcd(k1, K, k2, k3)

    def S0(self, n1, n2, n3, N):
        """
        ell=0 specialization of SLM, evaluated by the same stable machinery.
        """
        k1 = {"n": n1, "l": 0, "m": 0}
        k2 = {"n": n2, "l": 0, "m": 0}
        k3 = {"n": n3, "l": 0, "m": 0}
        K = {"n": N, "l": 0, "m": 0}

        return self.S_near_abcd(k1, K, k2, k3)

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
    
    @staticmethod
    def build_nondiag_mask_notes(N):
        """
            Remove diagonal frequency-shift terms:

            A_q A_lam bar(A_q)
            A_lam A_q bar(A_q)
        """

        mask = np.ones((N, N, N, N), dtype=bool)

        for lam in range(N):
            for q in range(N):
                mask[lam, q, lam, q] = False
                mask[lam, lam, q, q] = False

        return mask


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


def stable_sum(z):
    return CouplingCoefficients.stable_sum(z)


def stableConvolve(a, b):
    return CouplingCoefficients.stableConvolve(a, b)


def stableMultiConvolve(arrays):
    return CouplingCoefficients.stableMultiConvolve(arrays)


def pTable(n, ell):
    return _DEFAULT.pTable(n, ell)


def pNLMTable(mode):
    return _DEFAULT.pNLMTable(mode)


def C_abcd_table(a, b, c, d):
    return _DEFAULT.C_abcd_table(a, b, c, d)


def C_abcd(i, a, b, c, d):
    return _DEFAULT.C_abcd(i, a, b, c, d)


def P4LMn_as_C_abcd(i, a, b, c, d):
    return _DEFAULT.P4LMn_as_C_abcd(i, a, b, c, d)


def gamma_abcd(a, b, c, d):
    return _DEFAULT.gamma_abcd(a, b, c, d)


def Scomp_abcd(j, a, b, c, d):
    return _DEFAULT.Scomp_abcd(j, a, b, c, d)


def Scomp_abcd_table(nmax, a, b, c, d):
    return _DEFAULT.Scomp_abcd_table(nmax, a, b, c, d)


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

def build_nondiag_mask_notes(N):
    return CouplingCoefficients.build_nondiag_mask_notes(N)

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
    "stable_sum",
    "stableConvolve",
    "stableMultiConvolve",
    "pTable",
    "pNLMTable",
    "C_abcd_table",
    "C_abcd",
    "P4LMn_as_C_abcd",
    "gamma_abcd",
    "Scomp_abcd",
    "Scomp_abcd_table",
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
    "build_nondiag_mask_notes"
]
