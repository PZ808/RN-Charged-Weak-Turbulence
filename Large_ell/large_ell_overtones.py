"""
Overtone extension of the large-l back end.

`large_ell_backend.Backend` covers the fundamental (n = 0) tower only.  This
module adds the full overtone-resolved coupling tensor, using the exact closed
form of `large_ell_stirring.tex`, Eq. (Sgen), rearranged so that the j-sum
costs no extra transcendental evaluations.

Starting point (note, Sec. 4.1):

    S^{acd}_b = (prod_i N_i) sum_{j=0}^{NS} P_j (-1)^{j+gamma+1}
                Gamma(1+j-G/2-i q) Gamma(G-j-1) / Gamma(G/2-i q)

with G = sum_i (h_{l_i} + n_i) and P the fourfold convolution of the
single-mode tables P^{(n)}_j.  Reflecting the first Gamma,

    Gamma(1+j-G/2-iq) = pi / [ (-1)^j sin(pi(G/2+iq)) Gamma(G/2-j+iq) ],

the sign (-1)^j cancels against (-1)^{j+gamma+1} and the j-dependence collapses
to a ratio of consecutive Gammas.  Dividing through by the j = 0 term,

    S^{acd}_b = (prod_i N_i) * bare(G) * sum_j P_j r_j(G),          (*)

    r_0 = 1,   r_j = r_{j-1} (G/2 - j + i q) / (G - j - 1),

where `bare` is exactly `Backend.log_bare` evaluated at G in place of
H = sum_i h_i.  So the whole overtone generalisation is a rational recursion on
top of the n = 0 machinery: no new log-gammas on the four-index grid, and the
r_j are O(2^{-j}) at large G, hence numerically benign.

The angular factor Q is overtone-blind, so the m = 0 Gaunt tables of the
fundamental back end are reused verbatim.

Nothing in the project repository is imported at module level or modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from large_ell_backend import Backend

__all__ = ["OvertoneBackend"]


@dataclass
class OvertoneBackend:
    """Overtone-resolved m = 0 coupling tensor.

    Modes are the pairs (n, l) with 0 <= n <= nmax and lmin <= l <= lmax,
    ordered l-major (so that the n = 0 tower is the slice n_of == 0).
    """

    qhat: float = 0.6
    rplus: float = 1.0
    lmax: int = 16
    nmax: int = 2
    lmin: int = 1

    base: Backend = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "base", Backend(qhat=self.qhat, rplus=self.rplus, lmax=self.lmax)
        )

    # ---------------- mode bookkeeping ----------------

    @property
    def modes(self):
        return [(n, l)
                for l in range(self.lmin, self.lmax + 1)
                for n in range(self.nmax + 1)]

    @property
    def n_of(self):
        return np.array([n for (n, _) in self.modes])

    @property
    def l_of(self):
        return np.array([l for (_, l) in self.modes])

    @property
    def nmodes(self):
        return len(self.modes)

    # ---------------- single-mode data ----------------

    def g_vec(self):
        """g_a = h_{l_a} + n_a."""
        return self.base.h(self.l_of) + self.n_of

    def logN_vec(self):
        """log N_{n,l} for every mode, principal branch."""
        out = np.zeros(self.nmodes, dtype=complex)
        for a, (n, l) in enumerate(self.modes):
            out[a] = complex(self.base.log_N(np.array([float(l)]), n=n)[0])
        return out

    def P_mat(self):
        """P[a, j] = P^{(n_a)}_j(l_a), zero-padded to j = 0 .. nmax.

            P^{(n)}_j = (-n)_j (1 - 2h - n)_j / [ j! (1 - h - n - i q)_j ]
        """
        q = self.qhat
        P = np.zeros((self.nmodes, self.nmax + 1), dtype=complex)
        for a, (n, l) in enumerate(self.modes):
            hl = float(self.base.h(np.array([float(l)]))[0])
            P[a, 0] = 1.0
            for j in range(n):
                P[a, j + 1] = (P[a, j] * (-n + j) * (1 - 2 * hl - n + j)
                               / ((j + 1) * (1 - hl - n - 1j * q + j)))
        return P

    # ---------------- the four-index overlap ----------------

    def S_tensor(self):
        """S[i, j, k, lam] = S^{ij lam}_k  (project convention, slot k conjugated).

        Implements (*) above.  Memory: a handful of nmodes**4 complex arrays.
        """
        q = self.qhat
        M = self.nmodes
        g = self.g_vec()
        P = self.P_mat()
        lN = self.logN_vec()

        # --- G and the j = 0 ("bare") factor ---------------------------------
        G = (g[:, None, None, None] + g[None, :, None, None]
             + g[None, None, :, None] + g[None, None, None, :])

        lg = (lN[:, None, None, None] + lN[None, :, None, None]
              + np.conj(lN)[None, None, :, None] + lN[None, None, None, :])
        lg = lg + self.base.log_bare(G)
        acc = np.zeros_like(G, dtype=complex)

        # --- pairwise convolutions ------------------------------------------
        # slots (a, c) = (i, j);  slots (b, d) = (k, lam), with b conjugated
        nj = self.nmax + 1
        U1 = np.zeros((M, M, 2 * self.nmax + 1), dtype=complex)
        U2 = np.zeros((M, M, 2 * self.nmax + 1), dtype=complex)
        Pc = np.conj(P)
        for ja in range(nj):
            for jc in range(nj):
                U1[:, :, ja + jc] += P[:, ja][:, None] * P[:, jc][None, :]
        for jb in range(nj):
            for jd in range(nj):
                U2[:, :, jb + jd] += Pc[:, jb][:, None] * P[:, jd][None, :]

        # --- j-sum with the rational recursion -------------------------------
        jmax = 4 * self.nmax
        r = np.ones_like(G, dtype=complex)
        for jtot in range(jmax + 1):
            if jtot:
                r = r * (G / 2 - jtot + 1j * q) / (G - jtot - 1)
            # coefficient of r_jtot: sum_{j1 + j2 = jtot} U1[i,j,j1] U2[k,lam,j2]
            coef = None
            for j1 in range(max(0, jtot - 2 * self.nmax),
                            min(2 * self.nmax, jtot) + 1):
                j2 = jtot - j1
                term = (U1[:, :, j1][:, :, None, None]
                        * U2[:, :, j2][None, None, :, :])
                coef = term if coef is None else coef + term
            if coef is None:
                continue
            if np.max(np.abs(coef)) == 0.0:
                continue
            acc += coef * r

        return np.exp(lg) * acc

    # ---------------- assembled objects ----------------

    def Q_tensor(self):
        """Q[i, j, k, lam] on the (n, l) mode list; overtone-blind."""
        Qfull = self.base.Q_axisymmetric()
        L = self.l_of
        return Qfull[np.ix_(L, L, L, L)]

    def C_tensor(self, alpha=1.0):
        """C[lam, i, j, k] = alpha Q^{ij}_{k lam} S^{ij lam}_k."""
        C = alpha * self.Q_tensor() * self.S_tensor()
        return np.einsum("ijkl->lijk", C)

    def W_shell(self):
        """m-averaged kinetic kernel on the (n, l) shells.

            W[lam,i,j,k] = |S|^2 / (2 l_lam + 1)
                           sum_L G^2_{l_i l_j L} G^2_{l_k l_lam L} / (2L+1)
        """
        _, G2 = self.base._gaunt_tables()
        Lm = 2 * self.lmax
        u = G2 / np.sqrt(2 * np.arange(Lm + 1) + 1.0)
        A = np.einsum("ijL,klL->ijkl", u, u, optimize=True)
        L = self.l_of
        A = A[np.ix_(L, L, L, L)]
        S2 = np.abs(self.S_tensor()) ** 2
        W = S2 * A / (2 * L + 1.0)[None, None, None, :]
        return np.einsum("ijkl->lijk", W)

    def gamma_damp(self):
        """Gamma_{n,l} in units of sigma / r_+ ."""
        return 0.5 * (self.base.h(self.l_of) + self.n_of)


# ---------------------------------------------------------------------------


def validate(lmax_small=5, nmax=3, verbose=True):
    """Check the overtone tensor against coupling_coefficients_stable."""
    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients

    ob = OvertoneBackend(qhat=0.6, rplus=1.0, lmax=lmax_small, nmax=nmax)
    Co = CouplingCoefficients(ob.qhat, ob.rplus)
    out = {}

    # --- normalisation at n > 0 ------------------------------------------
    err = 0.0
    for (n, l) in ob.modes:
        ref = 1.0 / Co.Anorm(n, l)
        new = complex(ob.base.log_N(np.array([float(l)]), n=n)[0])
        err = max(err, abs(np.exp(new) / ref - 1))
    out["log_N (n>0)"] = err

    # --- individual overlaps ---------------------------------------------
    S = ob.S_tensor()
    idx = {m: a for a, m in enumerate(ob.modes)}
    tests = [((0, 1), (0, 1), (0, 1), (0, 1)),
             ((1, 2), (0, 3), (2, 2), (1, 1)),
             ((2, 5), (1, 4), (0, 5), (3, 5)),
             ((3, 4), (3, 4), (3, 4), (3, 4)),
             ((1, 5), (2, 3), (3, 2), (0, 4))]
    err = 0.0
    tests = [t for t in tests
             if all(m in idx for m in t)]
    for (a, b, c, d) in tests:
        # S^{ij lam}_k = S_near_abcd(i, k, j, lam):  slot order (a, b, c, d)
        # here we feed (i, j, k, lam) = (a, c, b, d) to match
        i, j, k, lam = a, c, b, d
        ref = Co.S_near_abcd(*[{"n": n_, "l": l_, "m": 0}
                               for (n_, l_) in (a, b, c, d)])
        new = S[idx[i], idx[j], idx[k], idx[lam]]
        err = max(err, abs(new / ref - 1))
    out["S_tensor"] = err

    # --- full coupling tensor --------------------------------------------
    modes = [{"n": n, "l": l, "m": 0} for (n, l) in ob.modes]
    Cref = Co.build_C_notes(modes, alpha=1.0)
    Cnew = ob.C_tensor(alpha=1.0)
    out["C_tensor"] = float(np.abs(Cnew - Cref).max() / np.abs(Cref).max())

    if verbose:
        print(f"overtone validation  (lmax = {lmax_small}, nmax = {nmax}, "
              f"{ob.nmodes} modes)")
        for k, v in out.items():
            flag = "ok " if v < 1e-10 else "!! "
            print(f"  {flag}{k:<16} max relative error {v:.3e}")
    return out


if __name__ == "__main__":
    validate()
