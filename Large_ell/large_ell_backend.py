"""
Large-l back end for the RN charged-scalar QNM system.

Replaces the O(N^4) magnetic sum of the project's coupling code by

  * the closed form for the near-zone radial overlap at n = 0, evaluated in
    log space so that the cancelling factors 2^{+-H} never appear separately
    (see `large_ell_stirring.tex`, Sec. 4), and
  * the exact reduction of the magnetic sum to a single multipole sum
    (ibid., Sec. 5), which for the axisymmetric sector and for the
    m-averaged kinetic kernel both collapse to one matrix product.

This takes l_max from ~7 to ~60 for the full axisymmetric coupling tensor and
to a few hundred for the shell kernel.

Nothing in the project repository is imported at module level or modified;
`validate()` imports `coupling_coefficients_stable` only if it is available.

Conventions follow the project code exactly:

    C[lam,i,j,k] = alpha * Q^{ij}_{k lam} * S^{ij lam}_k
    Q^{ij}_{k lam} = int Y_i Y_j conj(Y_k) conj(Y_lam)
    S^{ij lam}_k   = S_near_abcd(i, k, j, lam)      (second slot conjugated)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, lgamma, log, pi

import numpy as np
from scipy.special import loggamma as _clgamma

__all__ = ["Backend", "validate"]


# ---------------------------------------------------------------------------
# Wigner 3j with all magnetic numbers zero
# ---------------------------------------------------------------------------


def _lf(n):
    return lgamma(n + 1.0)


def w3j000(j1: int, j2: int, j3: int) -> float:
    """Wigner 3j (j1 j2 j3; 0 0 0), via log-gammas.  Exact to ~1e-15."""
    J = j1 + j2 + j3
    if J % 2 or j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    g = J // 2
    lv = 0.5 * (_lf(J - 2 * j1) + _lf(J - 2 * j2) + _lf(J - 2 * j3) - _lf(J + 1))
    lv += _lf(g) - _lf(g - j1) - _lf(g - j2) - _lf(g - j3)
    return (-1) ** g * exp(lv)


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """Large-l evaluation of the RN near-zone coupling coefficients.

    Parameters
    ----------
    qhat : float
        q r_+ .
    rplus : float
        Horizon radius.
    lmax : int
        Largest multipole retained.
    """

    qhat: float = 0.6
    rplus: float = 1.0
    lmax: int = 40

    # ---------------- spectrum ----------------

    def h(self, ell):
        """Conformal weight h_l (supplementary branch)."""
        ell = np.asarray(ell, dtype=float)
        disc = 0.25 + ell * (ell + 1) - self.qhat**2
        if np.any(disc <= 0):
            bad = ell[np.asarray(disc) <= 0]
            raise ValueError(f"principal branch (complex h) at l = {bad}")
        return 0.5 + np.sqrt(disc)

    def nu(self, n, ell):
        """Near-horizon frequency nu = i s;  omega = q Phi_H + (sigma/r_+) nu."""
        return self.qhat / 2 - 0.5j * (self.h(ell) + n)

    def gamma_damp(self, n, ell):
        """Gamma_{n,l} in units of sigma / r_+  (i.e. Gamma = (sigma/r_+) * this)."""
        return 0.5 * (self.h(ell) + n)

    # ---------------- normalisation ----------------

    def log_N(self, ell, n=0):
        """log of the unit-norm constant N_{n,l}, principal branch.

        Matches 1 / CouplingCoefficients.Anorm(n, l) exactly (including sign),
        but never forms the overflowing intermediates.
        """
        h = np.asarray(self.h(ell), dtype=float)
        q = self.qhat
        iq = 1j * q

        # log of N^{-2}
        z = np.zeros_like(h, dtype=complex)
        z += 2 * log(self.rplus)
        z += 1j * pi / 2                      # log(i)
        z += 1j * pi * h - pi * q             # log (-1)^{h+iq} = i pi (h + i q)
        z += -_clgamma(h - iq)
        z += log(2.0) + lgamma(n + 1.0)
        z += _clgamma(2 * h + n)
        # Gamma(1 - h - n - iq) by reflection, with the sine argument reduced
        xr = np.mod(h + n, 2.0)
        z += log(pi) - np.log(np.sin(pi * (xr + iq))) - _clgamma(h + n + iq)
        # divide by the Pochhammer (h + iq)_n
        if n:
            z += _clgamma(h + iq) - _clgamma(h + iq + n)

        # principal branch, so that exp(-z/2) is the principal reciprocal sqrt
        z = z.real + 1j * (np.mod(z.imag + pi, 2 * pi) - pi)
        return -0.5 * z

    # ---------------- radial overlap ----------------

    def log_bare(self, H):
        """log of the n=0 'bare' integral, a function of H = sum_i h_i alone.

            e^{i pi (1 - H/2 - i q)} Gamma(1-H/2-iq) Gamma(H-1) / Gamma(H/2-iq)
        """
        H = np.asarray(H, dtype=float)
        q = self.qhat
        iq = 1j * q
        z = 1j * pi * (1 - H / 2 - iq)
        # Gamma(1 - H/2 - iq) by reflection
        xr = np.mod(H / 2, 2.0)
        z += log(pi) - np.log(np.sin(pi * (xr + iq))) - _clgamma(H / 2 + iq)
        z += _clgamma(H - 1)
        z += -_clgamma(H / 2 - iq)
        return z

    def S_slots(self, la, lb, lc, ld):
        """S_near_abcd(a, b, c, d) for n = 0, second slot conjugated.

        Accepts broadcastable arrays of multipole indices.
        """
        la, lb, lc, ld = (np.asarray(x) for x in (la, lb, lc, ld))
        H = self.h(la) + self.h(lb) + self.h(lc) + self.h(ld)
        lg = (self.log_N(la) + np.conj(self.log_N(lb))
              + self.log_N(lc) + self.log_N(ld) + self.log_bare(H))
        return np.exp(lg)

    # ---------------- angular tables ----------------

    @lru_cache(None)
    def _w3j_table(self):
        """W[a, b, L] = 3j(a b L; 0 0 0), a,b <= lmax, L <= 2 lmax."""
        lm, Lm = self.lmax, 2 * self.lmax
        W = np.zeros((lm + 1, lm + 1, Lm + 1))
        for a in range(lm + 1):
            for b in range(lm + 1):
                for L in range(abs(a - b), min(a + b, Lm) + 1):
                    W[a, b, L] = w3j000(a, b, L)
        return W

    @lru_cache(None)
    def _gaunt_tables(self):
        """Return (g, G2) with

            g[a,b,L]  = int Y_a0 Y_b0 Y_L0   (axisymmetric triple integral)
            G2[a,b,L] = Gaunt^2 = (2a+1)(2b+1)(2L+1)/(4 pi) * 3j(ab L;000)^2
        """
        W = self._w3j_table()
        lm, Lm = self.lmax, 2 * self.lmax
        a = np.arange(lm + 1)[:, None, None]
        b = np.arange(lm + 1)[None, :, None]
        L = np.arange(Lm + 1)[None, None, :]
        pref = np.sqrt((2 * a + 1) * (2 * b + 1) * (2 * L + 1) / (4 * pi))
        g = pref * W * W          # = Gaunt * 3j  (all m = 0)
        G2 = (pref * W) ** 2
        return g, G2

    # ---------------- assembled objects ----------------

    def Q_axisymmetric(self):
        """Q[i,j,k,lam] = int Y_i Y_j Y_k Y_lam for the m = 0 tower.

        Indices run over l = 0 .. lmax.  Uses
        Q = sum_L (int Y_i Y_j Y_L)(int Y_k Y_lam Y_L).
        """
        g, _ = self._gaunt_tables()
        return np.einsum("ijL,klL->ijkl", g, g, optimize=True)

    def C_axisymmetric(self, alpha=1.0, lmin=1):
        """Full m = 0 coupling tensor in the project's convention,

            C[lam,i,j,k] = alpha Q^{ij}_{k lam} S^{ij lam}_k ,

        for the n = 0 tower with l = lmin .. lmax.  Returns (C, ells).
        """
        ells = np.arange(lmin, self.lmax + 1)
        Qfull = self.Q_axisymmetric()
        Q = Qfull[np.ix_(ells, ells, ells, ells)]          # [i,j,k,lam]

        hv = self.h(ells)
        H = (hv[:, None, None, None] + hv[None, :, None, None]
             + hv[None, None, :, None] + hv[None, None, None, :])
        lN = self.log_N(ells)
        # S^{ij lam}_k = S_near_abcd(i, k, j, lam): slot b = k is conjugated
        lg = (lN[:, None, None, None] + np.conj(lN)[None, None, :, None]
              + lN[None, :, None, None] + lN[None, None, None, :]
              + self.log_bare(H))
        S = np.exp(lg)                                      # [i,j,k,lam]

        C = alpha * Q * S
        return np.einsum("ijkl->lijk", C), ells

    def W_shell(self, lmin=1):
        """m-averaged kinetic shell kernel

            W[lam,i,j,k] = |S^{ij lam}_k|^2 / (2 l_lam + 1)
                           * sum_L G^2_{ij L} G^2_{k lam L} / (2L+1)

        Returns (W, ells).
        """
        ells = np.arange(lmin, self.lmax + 1)
        _, G2 = self._gaunt_tables()
        Lm = 2 * self.lmax
        u = G2 / np.sqrt(2 * np.arange(Lm + 1) + 1.0)
        A = np.einsum("ijL,klL->ijkl", u, u, optimize=True)
        A = A[np.ix_(ells, ells, ells, ells)]               # [i,j,k,lam]

        hv = self.h(ells)
        H = (hv[:, None, None, None] + hv[None, :, None, None]
             + hv[None, None, :, None] + hv[None, None, None, :])
        lN = self.log_N(ells)
        lg = (lN[:, None, None, None] + np.conj(lN)[None, None, :, None]
              + lN[None, :, None, None] + lN[None, None, None, :]
              + self.log_bare(H))
        S2 = np.abs(np.exp(lg)) ** 2

        W = S2 * A / (2 * ells + 1.0)[None, None, None, :]
        return np.einsum("ijkl->lijk", W), ells


# ---------------------------------------------------------------------------
# validation against the project code
# ---------------------------------------------------------------------------


def validate(lmax_small=7, verbose=True):
    """Check the back end against coupling_coefficients_stable at small l."""
    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients

    be = Backend(qhat=0.6, rplus=1.0, lmax=lmax_small)
    Co = CouplingCoefficients(be.qhat, be.rplus)
    out = {}

    # --- 1. normalisation -------------------------------------------------
    err = 0.0
    for l in range(1, lmax_small + 1):
        ref = 1.0 / Co.Anorm(0, l)
        new = np.exp(be.log_N(l))
        err = max(err, abs(new / ref - 1))
    out["log_N"] = err

    # --- 2. radial overlap ------------------------------------------------
    err = 0.0
    for t in [(1, 1, 1, 1), (2, 1, 3, 2), (5, 3, 4, 4), (7, 2, 6, 5), (7, 7, 7, 7)]:
        ref = Co.S_near_abcd(*[{"n": 0, "l": l, "m": 0} for l in t])
        new = complex(be.S_slots(*t))
        err = max(err, abs(new / ref - 1))
    out["S_slots"] = err

    # --- 3. axisymmetric angular factor ----------------------------------
    Q = be.Q_axisymmetric()
    err = 0.0
    for t in [(1, 1, 1, 1), (2, 3, 2, 3), (4, 2, 3, 3), (5, 5, 4, 4), (6, 3, 5, 4)]:
        i, j, k, lam = t
        ref = Co.Q_element(i, 0, j, 0, k, 0, lam, 0)
        err = max(err, abs(Q[i, j, k, lam] - ref))
    out["Q_axisymmetric"] = err

    # --- 4. full axisymmetric coupling tensor ----------------------------
    modes = [{"n": 0, "l": l, "m": 0} for l in range(1, lmax_small + 1)]
    Cref = Co.build_C_notes(modes, alpha=1.0)
    Cnew, ells = be.C_axisymmetric(alpha=1.0, lmin=1)
    scale = np.abs(Cref).max()
    out["C_axisymmetric"] = float(np.abs(Cnew - Cref).max() / scale)

    # --- 5. shell kernel vs brute force ----------------------------------
    #   build the mode-resolved |C|^2 for all m at small lmax and average
    lm = min(4, lmax_small)
    modes_all = [{"n": 0, "l": l, "m": m}
                 for l in range(1, lm + 1) for m in range(-l, l + 1)]
    Call = Co.build_C_notes(modes_all, alpha=1.0)
    Wbrute = np.zeros((lm, lm, lm, lm))
    idx = {}
    for a, md in enumerate(modes_all):
        idx.setdefault(md["l"], []).append(a)
    for li in range(1, lm + 1):
        for lj in range(1, lm + 1):
            for lk in range(1, lm + 1):
                for ll in range(1, lm + 1):
                    blk = Call[np.ix_(idx[ll], idx[li], idx[lj], idx[lk])]
                    Wbrute[ll - 1, li - 1, lj - 1, lk - 1] = (
                        np.sum(np.abs(blk) ** 2) / (2 * ll + 1))
    be_small = Backend(qhat=0.6, rplus=1.0, lmax=lm)
    Wnew, _ = be_small.W_shell(lmin=1)
    out["W_shell"] = float(np.abs(Wnew - Wbrute).max() / np.abs(Wbrute).max())

    if verbose:
        print(f"validation against coupling_coefficients_stable (lmax = {lmax_small})")
        for k, v in out.items():
            flag = "ok " if v < 1e-10 else "!! "
            print(f"  {flag}{k:<20} max relative error {v:.3e}")
    return out


if __name__ == "__main__":
    validate()
