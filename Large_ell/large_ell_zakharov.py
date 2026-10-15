"""
Zakharov treatment of the shell kinetic equation.

Replaces the power-counting locality estimate of `large_ell_stirring.tex`
Sec. 6.4 with the conformal (Zakharov) transformation, and tests every step
against the real kernel.

Setup.  In the continuum the m-averaged kinetic equation is

    m(l0) C(l0) = \\int dl1 dl2 dl3  Wt(l0,l1,l2,l3) F,      m(l) = 2l+1,
    F = n0 n1 n2 n3 [ 1/n0 + 1/n3 - 1/n1 - 1/n2 ],

so the quartet pairs as {0,3} against {1,2}, and Wt = m(l0) W is invariant under
the order-8 group preserving that partition (verified numerically on the n = 0
tower; it FAILS with overtones -- see Sec. 12 of the note).

Zakharov map (exchanging 0 with 1 and simultaneously 2 with 3, which is in the
symmetry group):

    l1 -> l0^2/l1,   l2 -> l0 l3/l1,   l3 -> l0 l2/l1 .

Then, writing gamma = deg Wt,

    Jacobian          l0^4 / l1^4
    kernel            (l0/l1)^gamma
    (l0 l1 l2 l3)^-x  (l0^4/l1^4)^-x
    bracket           -(l0/l1)^x

so the transformed integrand is the original times  -(l1/l0)^y  with

    y = 3x - gamma - 4 .

The three maps assemble to the Zakharov factor

    Z = 1 - (l1/l0)^y - (l2/l0)^y + (l3/l0)^y ,

which vanishes identically at y = 0.  With gamma = -1 this gives y = 3(x-1):

    KZ exponent  x = 1 ,

the same value the flux power counting gives (Pi ~ l^{3-3x}).  The second zero of
the collision integral is the bracket's own zero at x = 0, the equipartition of
action.  There is no third: Re nu = qhat/2 for every mode on the supplementary
branch, so the energy is proportional to the action and there is no independent
invariant and no dual cascade.

Steps:
  Z1  measure gamma = deg Wt, and the exponent y
  Z2  the dimensionless collision integral I(x) and its zeros
  Z3  locality: convergence of I(x) under l_max, direction by direction
  Z4  the marginal case: at x = 1 the Zakharov factor vanishes linearly while the
      IR integral diverges logarithmically, so the KZ constant is a finite 0 x inf.
      Test the Kraichnan log-corrected spectrum n ~ l^-1 (ln l)^-1/3.

    python large_ell_zakharov.py            # all
    python large_ell_zakharov.py Z2 Z4
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from large_ell_backend import Backend

QHAT = 0.6


def _rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_CACHE = {}


def kernel(lmax, qhat=QHAT):
    """(Wt, ells) with Wt = (2 l_lam + 1) W, symmetrised."""
    key = (lmax, qhat)
    if key not in _CACHE:
        be = Backend(qhat=qhat, rplus=1.0, lmax=lmax)
        W, ells = be.W_shell(lmin=1)
        mult = 2.0 * ells + 1.0
        Wt = W * mult[:, None, None, None]
        _CACHE[key] = (Wt, ells, mult)
    return _CACHE[key]


def collision(lmax, n, qhat=QHAT):
    Wt, ells, mult = kernel(lmax, qhat)
    t1 = np.einsum("Lijk,i,j,k->L", Wt, n, n, n, optimize=True)
    t2 = np.einsum("Lijk,i,j->L", Wt, n, n, optimize=True) * n
    t3 = np.einsum("Lijk,j,k->L", Wt, n, n, optimize=True) * n
    t4 = np.einsum("Lijk,i,k->L", Wt, n, n, optimize=True) * n
    return (t1 + t2 - t3 - t4) / mult


# ---------------------------------------------------------------------------


def Z1(lmax=40):
    _rule("Z1  homogeneity degree of the symmetric kernel")
    Wt, ells, _ = kernel(lmax)
    print("  Wt(a l, a l, a l, a l) / Wt(l,l,l,l) against a^gamma\n")
    print(f"  {'l':>4} {'a':>3} {'ratio':>14} {'gamma':>9}")
    gam = []
    for l in (4, 5, 6, 8):
        for a in (2, 3, 4, 5):
            if a * l <= lmax:
                r = (Wt[a*l-1, a*l-1, a*l-1, a*l-1]
                     / Wt[l-1, l-1, l-1, l-1])
                g = np.log(r) / np.log(a)
                gam.append(g)
                print(f"  {l:>4} {a:>3} {r:14.6e} {g:9.4f}")
    print(f"\n  mean gamma = {np.mean(gam):+.4f}  (asymptotically -1: Wt = (2l+1) W")
    print("  with W homogeneous of degree -2; the drift is the +1 in 2l+1 and the")
    print("  log in W_diag ~ ln l / l^2)")
    print("\n  off-diagonal check, Wt(l, 2l, l, 2l):")
    for l in (4, 6, 8):
        if 4 * l <= lmax:
            r = Wt[2*l-1, 4*l-1, 2*l-1, 4*l-1] / Wt[l-1, 2*l-1, l-1, 2*l-1]
            print(f"    l = {l}: ratio {r:.6e}  =>  gamma = {np.log(r)/np.log(2):+.4f}")
    print("\n  the drift is logarithmic: W_diag ~ c ln l / l^2 gives")
    print("    gamma(l,a) = -1 + ln(1 + ln a / ln l) / ln a")
    print(f"  {'l':>4} {'a':>3} {'measured':>10} {'predicted':>11}")
    for l in (4, 6, 8):
        for a in (2, 5):
            if a * l <= lmax:
                r = Wt[a*l-1, a*l-1, a*l-1, a*l-1] / Wt[l-1, l-1, l-1, l-1]
                g = np.log(r) / np.log(a)
                gp = -1 + np.log(1 + np.log(a)/np.log(l)) / np.log(a)
                print(f"  {l:>4} {a:>3} {g:10.4f} {gp:11.4f}")
    print("\n  => gamma -> -1 with a 1/ln l correction, so")
    print("     y = 3x - gamma - 4 -> 3(x - 1) and the Zakharov factor")
    print("     Z = 1 - (l1/l0)^y - (l2/l0)^y + (l3/l0)^y  vanishes at x = 1,")
    print("     the same exponent the flux power counting gives, x = (4+gamma)/3.")


def Z2(lmax=48, probe=10):
    _rule(f"Z2  the dimensionless collision integral I(x) (probe shell l = {probe})")
    Wt, ells, mult = kernel(lmax)
    print("  I(x) := l^(3x-1) C_l[n = l^-x]  -- the scale-free collision integral.")
    print("  Zeros of I are stationary power laws.\n")
    print(f"  {'x':>6} {'C_l':>15} {'I(x)':>15}")
    xs = np.linspace(0.0, 1.6, 33)
    vals = []
    for x in xs:
        n = ells.astype(float) ** (-x)
        C = collision(lmax, n)
        c = C[probe - 1]
        vals.append(c * probe ** (3 * x - 1))
    vals = np.array(vals)
    for k in range(0, len(xs), 2):
        n = ells.astype(float) ** (-xs[k])
        print(f"  {xs[k]:6.3f} {collision(lmax,n)[probe-1]:15.6e} {vals[k]:15.6e}")
    sg = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    print("\n  sign changes of I(x) at x ~ "
          + ", ".join(f"{0.5*(xs[i]+xs[i+1]):.3f}" for i in sg))
    print("  x = 0 is exact (the bracket vanishes for n = const: I(0) = "
          f"{vals[0]:.2e}).")

    print("\n  Is the second zero a universal exponent?  A genuine KZ state needs")
    print("  it to be independent of the probe shell and of the IR cutoff.\n")

    def zero_of(lmax_, probe_, lmin_):
        Wt_, ells_, mult_ = kernel(lmax_)
        sel = ells_ >= lmin_
        def I(x):
            n = np.where(sel, ells_.astype(float) ** (-x), 0.0)
            return collision(lmax_, n)[probe_ - 1]
        lo, hi = 0.5, 1.6
        if I(lo) * I(hi) > 0:
            return np.nan
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if I(lo) * I(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    print(f"  {'probe l':>8}" + "".join(f"{('lmin=%d' % L):>11}" for L in (1, 2, 3, 5))
          + f"{'lmax=56':>11}")
    for probe_ in (6, 10, 16, 24, 32):
        row = [zero_of(lmax, probe_, L) for L in (1, 2, 3, 5)]
        row.append(zero_of(56, probe_, 1))
        print(f"  {probe_:>8}" + "".join(f"{v:11.4f}" for v in row))
    gpred = -1.0
    print(f"\n  predicted KZ exponent x = (4 + gamma)/3 = {(4+gpred)/3:.3f} for gamma = -1,")
    print("  and 1.133 for the gamma ~ -0.6 actually measured at these l.  The")
    print("  numerical zero matches NEITHER and drifts with both the probe shell and")
    print("  the IR cutoff: the collision integral is not scale-free, which is the")
    print("  non-locality of Sec. 6.4 seen directly rather than by power counting.")


def Z3(probe=10, cutoffs=(24, 32, 40, 48, 56)):
    _rule("Z3  locality: does I(x) converge as the cutoff is removed?")
    print("  Power counting (note Sec. 6.4) predicts convergence only for")
    print("  3/4 < x < 1, with the UV edge at x = 3/4 and the IR edge at x = 1.\n")
    xs = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.20]
    print(f"  {'x':>6} " + "".join(f"{('lmax=%d' % L):>14}" for L in cutoffs)
          + f"{'behaviour':>14}")
    for x in xs:
        row = []
        for L in cutoffs:
            Wt, ells, mult = kernel(L)
            n = ells.astype(float) ** (-x)
            row.append(collision(L, n)[probe - 1] * probe ** (3 * x - 1))
        row = np.array(row)
        rel = abs(row[-1] - row[-2]) / max(abs(row[-1]), 1e-300)
        tag = "converged" if rel < 0.02 else ("drifting" if rel < 0.2 else "DIVERGENT")
        print(f"  {x:6.2f} " + "".join(f"{v:14.6e}" for v in row) + f"{tag:>14}")
    print("\n  The UV edge is a genuine cutoff dependence.  The IR edge at x = 1 is")
    print("  NOT a divergence of the physical problem: there is no l < 1, so the")
    print("  integral is cut off at the forcing scale.  What the IR non-locality")
    print("  means physically is that every high-l shell is driven DIRECTLY by the")
    print("  pump rather than by its neighbours -- a stirred state, not a cascade.")


def Z4(lmax=48, l0=3.0):
    _rule("Z4  the marginal case and the Kraichnan logarithm")
    print("  At x = 1 the Zakharov factor Z vanishes linearly in y = 3(x-1) while")
    print("  the IR integral diverges as 1/(1-x) = -3/y, so the KZ constant is a")
    print("  finite 0 x inf.  The standard resolution is a log-corrected spectrum")
    print("  n ~ l^-1 (ln(l/l0))^-s; constant flux fixes s = 1/3.\n")
    Wt, ells, mult = kernel(lmax)
    inj = np.sum(mult * np.where((ells >= 4) & (ells <= 9), 1.0, 0.0))

    def flux(n):
        C = collision(lmax, n)
        return -np.cumsum(mult * C)

    cases = {
        "n = l^-1              ": ells.astype(float) ** -1.0,
        "n = l^-1 (ln l/l0)^-1/3": (ells.astype(float) ** -1.0
                                    * np.maximum(np.log(ells / l0), 1e-3) ** (-1 / 3)),
        "n = l^-1 (ln l/l0)^-2/3": (ells.astype(float) ** -1.0
                                    * np.maximum(np.log(ells / l0), 1e-3) ** (-2 / 3)),
        "n = l^-0.9            ": ells.astype(float) ** -0.9,
        "n = const             ": np.ones(len(ells)),
    }
    print("  flux Pi(l), normalised to its value at l = 12, over the interior range")
    print(f"  {'spectrum':<24}" + "".join(f"{('l=%d' % l):>11}" for l in (12, 20, 28, 36, 44))
          + f"{'spread':>10}")
    for lab, n in cases.items():
        P = flux(n)
        ref = P[11]
        if abs(ref) < 1e-300:
            print(f"  {lab:<24}" + "  (identically zero flux: this is the equipartition)")
            continue
        r = [P[l - 1] / ref for l in (12, 20, 28, 36, 44)]
        sub = P[11:44] / ref
        print(f"  {lab:<24}" + "".join(f"{v:11.4f}" for v in r)
              + f"{sub.max()-sub.min():10.4f}")
    print("\n  The log correction makes the flux spread LARGER, i.e. less constant,")
    print("  not more: 0.937 for pure l^-1 against 0.972 and 0.979 for the log-")
    print("  corrected forms.  The marginal cascade is not realised -- which is what")
    print("  the empty window says: there is no exponent at which the integral")
    print("  converges, so there is nothing for the logarithm to correct.")

    print("\n  inertial-range condition: omega_NL / Gamma for n = A l^-1")
    be = Backend(qhat=QHAT, rplus=1.0, lmax=lmax)
    gam = 0.5 * be.h(ells)
    n = ells.astype(float) ** -1.0
    w = np.abs(collision(lmax, n)) / n
    print(f"  {'l':>4} {'omega_NL/Gamma':>16}")
    for l in (8, 12, 16, 24, 32, 40):
        if l <= lmax:
            print(f"  {l:>4} {w[l-1]/gam[l-1]:16.5e}")
    m = (ells >= 12) & (ells <= lmax - 8)
    s = np.polyfit(np.log(ells[m].astype(float)), np.log(w[m] / gam[m]), 1)[0]
    print(f"  slope {s:+.3f}: the window closes as a power of l, so an inertial")
    print("  range of any fixed width exists only below some l, never asymptotically.")


def main():
    want = [w.upper() for w in sys.argv[1:]] or ["Z1", "Z2", "Z3", "Z4"]
    for k, fn in [("Z1", Z1), ("Z2", Z2), ("Z3", Z3), ("Z4", Z4)]:
        if k in want:
            fn()
    print()


if __name__ == "__main__":
    main()
