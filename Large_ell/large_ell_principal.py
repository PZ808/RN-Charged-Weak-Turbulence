"""
Principal-branch (complex-h) extension of the large-l back end.

The conformal weight is

    h_l = 1/2 + sqrt(1/4 + l(l+1) - qhat^2)                   (supplementary)
    h_l = 1/2 - i sqrt(qhat^2 - l(l+1) - 1/4)                 (principal)

the sign in the second line fixed to match the project's `CouplingCoefficients.h`,
which returns `0.5 - csqrt(d2)` when the discriminant d2 is negative.  The principal
band is l(l+1) + 1/4 < qhat^2, i.e. roughly l < qhat: it is the near-horizon AdS_2
BF-bound-violating regime, and it is the structural analogue of near-extremal Kerr's
zero-damped tower.  Two things change there:

  * nu_{n,l} = qhat/2 - mu_l/2 - (i/2)(n + 1/2):  the damping Gamma ~ (n + 1/2) is
    INDEPENDENT of l, and the real frequency acquires an l-dependent ladder
    mu_l = sqrt(qhat^2 - l(l+1) - 1/4), so the exact degeneracy Delta == 0 of the
    supplementary branch is lifted;
  * a second independent invariant (the "energy" sum_l (2l+1) Re(nu_l) n_l) exists,
    where on the supplementary branch Re nu = qhat/2 for every mode and the energy is
    proportional to the action.

Everything else -- the closed forms for N, for the radial overlap, the magnetic-sum
reduction -- continues analytically.  The only code changes needed are in the two
places where a reflection formula reduces an argument mod 2: that reduction must act
on the real part alone.

Nothing in the project repository is imported at module level or modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np
from scipy.special import loggamma as _clgamma

from large_ell_backend import Backend

__all__ = ["PrincipalBackend", "band_edge", "validate"]


def band_edge(qhat):
    """Largest l on the principal branch:  l(l+1) + 1/4 < qhat^2."""
    l = 0
    while (l + 1) * (l + 2) + 0.25 < qhat ** 2:
        l += 1
    return l


def _modpi(z):
    """Reduce Re(z) mod 2, leaving Im(z) alone (sin(pi z) is 2-periodic)."""
    z = np.asarray(z, dtype=complex)
    return np.mod(z.real, 2.0) + 1j * z.imag


@dataclass(frozen=True)
class PrincipalBackend(Backend):
    """`Backend` with the conformal weight continued to the principal branch."""

    # ---------------- spectrum ----------------

    def h(self, ell):
        ell = np.asarray(ell, dtype=float)
        disc = 0.25 + ell * (ell + 1) - self.qhat ** 2
        out = np.where(disc > 0,
                       0.5 + np.sqrt(np.abs(disc)),
                       0.5 - 1j * np.sqrt(np.abs(disc)))
        return out.astype(complex)

    def mu(self, ell):
        """mu_l = -Im h_l >= 0 on the principal branch, 0 on the supplementary one."""
        return -np.imag(self.h(ell))

    def nu(self, n, ell):
        return self.qhat / 2 - 0.5j * (self.h(ell) + n)

    def gamma_damp(self, n, ell):
        """Gamma in units of sigma/r_+ : Re(h+n)/2.

        Principal branch: == (n + 1/2)/2, independent of l.
        """
        return 0.5 * np.real(self.h(ell) + n)

    def omega_real(self, n, ell):
        """Re nu.  Principal: qhat/2 - mu_l/2.  Supplementary: qhat/2 exactly."""
        return np.real(self.nu(n, ell))

    # ---------------- normalisation ----------------

    def log_N(self, ell, n=0):
        h = np.asarray(self.h(ell), dtype=complex)
        q = self.qhat
        iq = 1j * q

        z = np.zeros_like(h, dtype=complex)
        z += 2 * log(self.rplus)
        z += 1j * pi / 2
        z += 1j * pi * h - pi * q
        z += -_clgamma(h - iq)
        z += log(2.0) + lgamma(n + 1.0)
        z += _clgamma(2 * h + n)
        z += log(pi) - np.log(np.sin(pi * (_modpi(h + n) + iq))) - _clgamma(h + n + iq)
        if n:
            z += _clgamma(h + iq) - _clgamma(h + iq + n)

        z = z.real + 1j * (np.mod(z.imag + pi, 2 * pi) - pi)
        return -0.5 * z

    # ---------------- radial overlap ----------------

    def log_bare(self, H):
        H = np.asarray(H, dtype=complex)
        q = self.qhat
        iq = 1j * q
        z = 1j * pi * (1 - H / 2 - iq)
        z += log(pi) - np.log(np.sin(pi * (_modpi(H / 2) + iq))) - _clgamma(H / 2 + iq)
        z += _clgamma(H - 1)
        z += -_clgamma(H / 2 - iq)
        return z

    # ---------------- assembled objects ----------------
    # These must be overridden rather than inherited.  The project defines
    #   gamma_abcd = s_a + conj(s_b) + s_c + s_d,   s = -(n + h + i qhat)/2,
    # so the argument of the radial Gammas is
    #   G = h_a + CONJ(h_b) + h_c + h_d + sum n_i .
    # On the supplementary branch h is real and the conjugation is invisible;
    # on the principal branch conj(h) = 1 - h is the OTHER root, and getting
    # this wrong rescales the overlap by e^{O(mu)}.

    def _Hconj(self, la, lb, lc, ld):
        return (self.h(la) + np.conj(self.h(lb))
                + self.h(lc) + self.h(ld))

    def S_slots(self, la, lb, lc, ld):
        la, lb, lc, ld = (np.asarray(x) for x in (la, lb, lc, ld))
        H = self._Hconj(la, lb, lc, ld)
        lg = (self.log_N(la) + np.conj(self.log_N(lb))
              + self.log_N(lc) + self.log_N(ld) + self.log_bare(H))
        return np.exp(lg)

    def _four_index(self, ells):
        """(H, log prod N) on the grid [i, j, k, lam], slot b = k."""
        hv = self.h(ells)
        H = (hv[:, None, None, None] + hv[None, :, None, None]
             + np.conj(hv)[None, None, :, None] + hv[None, None, None, :])
        lN = self.log_N(ells)
        lg = (lN[:, None, None, None] + lN[None, :, None, None]
              + np.conj(lN)[None, None, :, None] + lN[None, None, None, :])
        return H, lg

    def C_axisymmetric(self, alpha=1.0, lmin=1):
        ells = np.arange(lmin, self.lmax + 1)
        Q = self.Q_axisymmetric()[np.ix_(ells, ells, ells, ells)]
        H, lg = self._four_index(ells)
        S = np.exp(lg + self.log_bare(H))
        return np.einsum("ijkl->lijk", alpha * Q * S), ells

    def W_shell(self, lmin=1):
        ells = np.arange(lmin, self.lmax + 1)
        _, G2 = self._gaunt_tables()
        Lm = 2 * self.lmax
        u = G2 / np.sqrt(2 * np.arange(Lm + 1) + 1.0)
        A = np.einsum("ijL,klL->ijkl", u, u, optimize=True)
        A = A[np.ix_(ells, ells, ells, ells)]
        H, lg = self._four_index(ells)
        S2 = np.abs(np.exp(lg + self.log_bare(H))) ** 2
        W = S2 * A / (2 * ells + 1.0)[None, None, None, :]
        return np.einsum("ijkl->lijk", W), ells


# ---------------------------------------------------------------------------


def validate(qhat=6.0, lmax_small=6, verbose=True):
    """Check the principal-branch back end against the project code.

    At qhat = 6 the band edge is l = 5, so l = 1..5 are principal and l = 6 is
    supplementary: one call exercises both branches and the crossover.
    """
    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients

    be = PrincipalBackend(qhat=qhat, rplus=1.0, lmax=lmax_small)
    Co = CouplingCoefficients(qhat, 1.0)
    out = {}

    err = 0.0
    for l in range(1, lmax_small + 1):
        ref = complex(Co.h(l))
        new = complex(be.h(np.array([float(l)]))[0])
        err = max(err, abs(new - ref))
    out["h"] = err

    err = 0.0
    for l in range(1, lmax_small + 1):
        for n in (0, 1, 2):
            ref = 1.0 / Co.Anorm(n, l)
            new = np.exp(complex(be.log_N(np.array([float(l)]), n=n)[0]))
            err = max(err, abs(new / ref - 1))
    out["log_N"] = err

    err = 0.0
    for t in [(1, 1, 1, 1), (2, 1, 3, 2), (5, 3, 4, 4), (6, 2, 5, 5),
              (1, 2, 5, 6), (6, 6, 6, 6)]:
        if max(t) > lmax_small:
            continue
        ref = Co.S_near_abcd(*[{"n": 0, "l": l, "m": 0} for l in t])
        new = complex(be.S_slots(*t))
        err = max(err, abs(new / ref - 1))
    out["S_slots"] = err

    modes = [{"n": 0, "l": l, "m": 0} for l in range(1, lmax_small + 1)]
    Cref = Co.build_C_notes(modes, alpha=1.0)
    Cnew, _ = be.C_axisymmetric(alpha=1.0, lmin=1)
    out["C_axisymmetric"] = float(np.abs(Cnew - Cref).max() / np.abs(Cref).max())

    if verbose:
        print(f"principal-branch validation  (qhat = {qhat}, band edge l = "
              f"{band_edge(qhat)}, lmax = {lmax_small})")
        for k, v in out.items():
            flag = "ok " if v < 1e-10 else "!! "
            print(f"  {flag}{k:<16} max error {v:.3e}")
    return out


if __name__ == "__main__":
    validate()
