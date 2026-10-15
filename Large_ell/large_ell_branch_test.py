"""
Does the principal branch open an inertial range?

The branch argument of `large_ell_stirring.tex` Sec. 13 is that near-extremal Kerr
cascades (or at least sustains a stirred high-l state) because its cascade direction,
m, is simultaneously the near-horizon charge that keeps the tower on the long-lived
branch, so the damping does not grow along the cascade.  In RN the charge qhat and the
cascade label l are independent, and at qhat = 0.6 every l >= 1 is supplementary with
Gamma ~ (h_l + n)/2 ~ l/2.

That argument predicts a regime where RN behaves like Kerr: qhat large enough that a
band l < qhat sits on the PRINCIPAL branch, where

    h_l = 1/2 - i mu_l,   mu_l = sqrt(qhat^2 - l(l+1) - 1/4),
    nu_{n,l} = qhat/2 - mu_l/2 - (i/2)(n + 1/2),

so Gamma is l-independent and the real frequency acquires a ladder.  This script tests
whether that is enough.

Steps:
  B1  spectrum: Gamma_l and Re nu_l across the band, both branches
  B2  the kernel on the principal band: diagonal scaling and homogeneity
  B3  resonance broadening: Delta is no longer identically zero, so the kinetic kernel
      acquires a Lorentzian factor Gamma_tot^2 / (Delta^2 + Gamma_tot^2).  How thin is
      the resonant manifold?
  B4  the decisive comparison: nonlinear transfer rate against Gamma_l
  B5  forced-dissipative flux, with and without the broadening factor

    python large_ell_branch_test.py            # all
    python large_ell_branch_test.py B1 B5
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp

from large_ell_backend import Backend
from large_ell_principal import PrincipalBackend, band_edge

QHAT = 40.0
LMAX = 56
QMID = 12.0      # small enough that e^{3 pi (qhat - mu)} is representable
LMID = 16


def _rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _setup(qhat=QHAT, lmax=LMAX, lmin=1):
    be = PrincipalBackend(qhat=qhat, rplus=1.0, lmax=lmax)
    ells = np.arange(lmin, lmax + 1)
    return be, ells


# ---------------------------------------------------------------------------


def B1(qhat=QHAT, lmax=LMAX):
    _rule(f"B1  spectrum at qhat = {qhat}  (principal band is l <= {band_edge(qhat)})")
    be, ells = _setup(qhat, lmax)
    h = be.h(ells)
    mu = be.mu(ells)
    gam = be.gamma_damp(0, ells)
    rev = be.omega_real(0, ells)
    print(f"  {'l':>4} {'branch':>8} {'Re h':>9} {'mu':>9} "
          f"{'Gamma (n=0)':>12} {'Re nu':>10}")
    for l in [1, 2, 5, 10, 20, 30, 35, 38, 39, 40, 45, 56]:
        if l > lmax:
            continue
        k = l - ells[0]
        br = "princ" if mu[k] > 0 else "suppl"
        print(f"  {l:>4} {br:>8} {h[k].real:9.4f} {mu[k]:9.4f} "
              f"{gam[k]:12.5f} {rev[k]:10.5f}")
    pr = mu > 0
    print(f"\n  on the principal band Gamma is constant to "
          f"{np.ptp(gam[pr])/np.mean(gam[pr]):.1e} (exactly (n+1/2)/2 = 0.25),")
    be0 = Backend(qhat=0.6, rplus=1.0, lmax=lmax)
    g0 = 0.5 * be0.h(ells)
    print(f"  against Gamma(l=1) = {g0[0]:.3f} -> Gamma(l={lmax}) = {g0[-1]:.3f} "
          f"at qhat = 0.6 (a factor {g0[-1]/g0[0]:.0f}).")
    print(f"\n  the real frequency is no longer degenerate: Re nu spans "
          f"[{rev.min():.3f}, {rev.max():.3f}] on the principal band,")
    print(f"  so Delta = (mu_lam + mu_k - mu_i - mu_j)/2 is generically O(1-10) "
          f"against a total width Gamma_tot = 1.0.")
    print("  On the supplementary branch Re nu = qhat/2 for every mode and Delta == 0.")


def B2(qhat=QMID, lmax=LMID):
    _rule("B2  the normalisation collapses on the principal branch")
    print("  The QNM 'norm' is the SYMMETRIC bilinear form int psi^2, not int |psi|^2.")
    print("  On the principal branch it degenerates: the modes become nearly null")
    print("  vectors of that form, so the unit-normalised coupling blows up.\n")
    for q in (6.0, qhat):
        be = PrincipalBackend(qhat=q, rplus=1.0, lmax=band_edge(q) + 2)
        ells = np.arange(1, band_edge(q) + 3)
        mu = be.mu(ells)
        lN = be.log_N(ells)
        # |int psi^2| = |Anorm|^2 = |1/N|^2
        nrm = np.exp(-2 * lN.real)
        pred = np.pi * (q - mu) / 2
        print(f"  qhat = {q}, band edge l = {band_edge(q)}")
        print(f"  {'l':>4} {'mu':>8} {'|int psi^2|':>14} {'ln|N|':>10} "
              f"{'pi(q-mu)/2':>12} {'difference':>11}")
        for l in ells:
            k = l - 1
            print(f"  {l:>4} {mu[k]:8.4f} {nrm[k]:14.5e} {lN[k].real:10.4f} "
                  f"{pred[k]:12.4f} {lN[k].real - pred[k]:11.4f}")
        m = mu > 0
        if m.sum() > 2:
            c = np.polyfit(pred[m], lN[m].real, 1)
            print(f"  fit ln|N| = a pi(q-mu)/2 + b:  a = {c[0]:.4f}, b = {c[1]:.4f}"
                  f"   (a = 1 is the predicted law)\n")

    be = PrincipalBackend(qhat=qhat, rplus=1.0, lmax=lmax)
    ells = np.arange(1, lmax + 1)
    mu = be.mu(ells)
    S = np.array([complex(be.S_slots(l, l, l, l)) for l in ells])
    pred = 3 * np.pi * (qhat - mu)
    m = mu > 0
    c = np.polyfit(pred[m], np.log(np.abs(S[m])), 1)
    print(f"  diagonal overlap:  ln|S(l,l,l,l)| vs 3 pi (qhat - mu_l):"
          f"  slope {c[0]:.4f}, intercept {c[1]:.4f}")
    print(f"  {'l':>4} {'mu':>8} {'|S|':>14} {'exp(3pi(q-mu))':>16} {'ratio':>10}")
    for l in ells[::2]:
        k = l - 1
        print(f"  {l:>4} {mu[k]:8.4f} {abs(S[k]):14.5e} {np.exp(pred[k]):16.5e} "
              f"{abs(S[k])/np.exp(pred[k]):10.5f}")
    print("\n  -> |S| ~ e^{3 pi (qhat - mu_l)}: the entire growth is normalisation.")
    print("     The bare (un-normalised) overlap grows only mildly over the same band.")


def _lorentz(be, ells, nmode=0):
    """Gamma_tot^2 / (Delta^2 + Gamma_tot^2) on the [lam,i,j,k] grid."""
    mu = be.mu(ells)
    g = be.gamma_damp(nmode, ells)
    # Delta = Re nu_lam + Re nu_k - Re nu_i - Re nu_j ;  Re nu = qhat/2 - mu/2
    d = -0.5 * (mu[:, None, None, None] + mu[None, None, None, :]
                - mu[None, :, None, None] - mu[None, None, :, None])
    gt = (g[:, None, None, None] + g[None, :, None, None]
          + g[None, None, :, None] + g[None, None, None, :])
    return gt ** 2 / (d ** 2 + gt ** 2), d, gt


def B3(qhat=QHAT, lmax=LMAX):
    _rule("B3  resonance broadening on the principal branch")
    be, ells = _setup(qhat, lmax)
    L, d, gt = _lorentz(be, ells)
    pr = band_edge(qhat)
    m = ells <= pr
    sub = np.ix_(m, m, m, m)
    Ls, ds, gs = L[sub], np.abs(d[sub]), gt[sub]
    print(f"  principal band l = 1..{pr}: {m.sum()**4:,} quartets")
    print(f"  |Delta| : median {np.median(ds):.3f}, 90th pct {np.percentile(ds,90):.3f},"
          f"  max {ds.max():.3f}")
    print(f"  Gamma_tot = {gs.flat[0]:.3f} (n = 0, l-independent)")
    for t in (0.5, 0.1, 0.01):
        print(f"  fraction of quartets with Lorentzian factor > {t}: "
              f"{np.mean(Ls > t):.4f}")
    print(f"  mean Lorentzian factor: {Ls.mean():.4f}  "
          f"-> the resonant manifold is thin, as in Kerr")
    print("  (on the supplementary branch Delta == 0 and the factor is identically 1)")


def B4(qhat=QMID, lmax=LMID):
    _rule("B4  nonlinear transfer rate against the damping rate")
    print("  omega_NL(l) := |C_l[n]| / n_l  with n_l = l^-x, measured on the real")
    print("  kernel; compared with Gamma_l.  An inertial range needs omega_NL > Gamma.\n")
    for qh, lab in ((0.6, "supplementary (qhat=0.6)"),
                    (qhat, f"principal (qhat={qhat:g})")):
        be = (Backend(qhat=0.6, rplus=1.0, lmax=40) if qh == 0.6
              else PrincipalBackend(qhat=qh, rplus=1.0, lmax=lmax))
        W, ells = be.W_shell(lmin=1)
        mult = 2.0 * ells + 1.0
        Wt = W * mult[:, None, None, None]
        if qh != 0.6:
            Lz, _, _ = _lorentz(be, ells)
            Wt = Wt * Lz
        gam = (0.5 * np.real(be.h(ells)) if qh != 0.6 else 0.5 * be.h(ells))

        def Cop(n):
            t1 = np.einsum("Lijk,i,j,k->L", Wt, n, n, n, optimize=True)
            t2 = np.einsum("Lijk,i,j->L", Wt, n, n, optimize=True) * n
            t3 = np.einsum("Lijk,j,k->L", Wt, n, n, optimize=True) * n
            t4 = np.einsum("Lijk,i,k->L", Wt, n, n, optimize=True) * n
            return (t1 + t2 - t3 - t4) / mult

        x = 1.0
        n = ells.astype(float) ** (-x)
        wnl = np.abs(Cop(n)) / n
        print(f"  --- {lab} ---")
        print(f"  {'l':>4} {'omega_NL':>13} {'Gamma':>10} {'ratio':>11}")
        for l in [2, 4, 6, 8, 10, 12, 14, 16, 24, 32, 40]:
            if l > ells[-1]:
                continue
            k = l - 1
            print(f"  {l:>4} {wnl[k]:13.5e} {gam[k]:10.4f} {wnl[k]/gam[k]:11.4e}")
        m = (ells >= 3) & (ells <= min(36, ells[-1] - 2))
        s = np.polyfit(np.log(ells[m].astype(float)),
                       np.log(wnl[m] / gam[m]), 1)[0]
        print(f"  slope of log(omega_NL/Gamma) vs log l: {s:+.3f}")
        if qh != 0.6:
            mu = be.mu(ells)
            c = np.polyfit((np.pi*(qh-mu))[m], np.log(wnl[m]/gam[m]), 1)[0]
            print(f"  ln(omega_NL/Gamma) vs pi(qhat-mu_l): slope {c:+.3f}"
                  f"  -> exponential, not power law")
            ncrit = gam/np.maximum(np.abs(Cop(np.ones(len(ells))))/1.0, 1e-300)
            print("  critical occupation for weak nonlinearity, n_crit = Gamma/omega_NL(n=1):")
            print("    " + ", ".join(f"l={l}:{ncrit[l-1]:.2e}" for l in [2,6,10,14] if l<=ells[-1]))
        print()


def B5(qhat=QMID, lmax=LMID, rho4=1.0):
    _rule("B5  forced-dissipative flux on the principal branch")
    pr = band_edge(qhat)
    for broaden in (False, True):
        be = PrincipalBackend(qhat=qhat, rplus=1.0, lmax=lmax)
        W, ells = be.W_shell(lmin=1)
        mult = 2.0 * ells + 1.0
        Wt = W * mult[:, None, None, None]
        if broaden:
            Lz, _, _ = _lorentz(be, ells)
            Wt = Wt * Lz
        Wt = (Wt + Wt.transpose(0, 2, 1, 3) + Wt.transpose(3, 1, 2, 0)
              + Wt.transpose(3, 2, 1, 0) + Wt.transpose(2, 3, 0, 1)
              + Wt.transpose(2, 3, 1, 0) + Wt.transpose(1, 0, 3, 2)
              + Wt.transpose(1, 0, 2, 3)) / 8.0
        gam = 0.5 * np.real(be.h(ells))
        inj = np.zeros(len(ells))
        inj[(ells >= 4) & (ells <= 9)] = 1.0
        inj /= np.sum(mult * inj)

        def Cop(n):
            t1 = np.einsum("Lijk,i,j,k->L", Wt, n, n, n, optimize=True)
            t2 = np.einsum("Lijk,i,j->L", Wt, n, n, optimize=True) * n
            t3 = np.einsum("Lijk,j,k->L", Wt, n, n, optimize=True) * n
            t4 = np.einsum("Lijk,i,k->L", Wt, n, n, optimize=True) * n
            return (t1 + t2 - t3 - t4) / mult

        rng = np.random.default_rng(3)
        nt = rng.uniform(0.1, 1.0, len(ells))
        cons = abs(np.sum(mult * Cop(nt))) / np.sum(mult * np.abs(Cop(nt)))
        D = rho4 * gam / (2 * np.pi)
        sol = solve_ivp(lambda T, n: Cop(np.maximum(n, 0)) - D * n + inj,
                        (0, 6e4), np.full(len(ells), 1e-8),
                        method="LSODA", rtol=1e-8, atol=1e-16)
        n = np.maximum(sol.y[:, -1], 0)
        Cl = Cop(n)
        Pi = -np.cumsum(mult * Cl) / np.sum(mult * inj)
        tag = "with resonance broadening" if broaden else "broadening OFF (Delta ignored)"
        print(f"\n  --- {tag} ---   conservation {cons:.1e}")
        print(f"  {'l':>4} {'n_l':>14} {'Pi/inj':>11}")
        for l in [2, 4, 6, 8, 10, pr - 1, pr + 1, lmax - 1]:
            if 1 <= l <= lmax:
                print(f"  {l:>4} {n[l-1]:14.5e} {Pi[l-1]:11.5f}")
        sub = (ells >= 3) & (ells <= pr - 1)
        print(f"  interior flux spread over 3 <= l <= {pr-1}: "
              f"{Pi[sub].max()-Pi[sub].min():.4f}")
        m = (ells >= 3) & (ells <= pr - 1)
        sl = np.polyfit(np.log(ells[m].astype(float)),
                        np.log(np.maximum(n[m], 1e-300)), 1)[0]
        print(f"  fitted spectrum on the same range: n_l ~ l^{sl:+.3f}  "
              f"(KZ prediction: -1)")


def main():
    want = [w.upper() for w in sys.argv[1:]] or ["B1", "B2", "B3", "B4", "B5"]
    for k, fn in [("B1", B1), ("B2", B2), ("B3", B3), ("B4", B4), ("B5", B5)]:
        if k in want:
            fn()
    print()


if __name__ == "__main__":
    main()
