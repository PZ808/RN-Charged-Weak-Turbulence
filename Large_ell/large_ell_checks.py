"""
Verification suite for `large_ell_stirring.tex`.

Reproduces every numerical claim in the note, section by section:

  A  spectrum           h_l expansion; exact degeneracy Re(nu)=qhat/2; Gamma_{n,l}
  B  normalisation      exact Pochhammer ratio; large-l law for |N_{n,l}|
  C  radial overlap     exact closed form at arbitrary overtone (vs the project code);
                        the n=0 specialisation; the large-l law and the 2^{+-G}
                        cancellation; the overtone staircase; the j-sum cancellation;
                        the phase of S
  D  angular            exact m-sum reduction (vs brute force over all magnetic
                        numbers); semiclassical 3j; the stretched corner; the
                        l^2 ln l fit
  E  shell kernel       homogeneity degree; diagonal scaling; direction-resolved
                        locality including the triangle-rule cutoffs

Nothing in the repository is modified.  Run from the directory containing
`coupling_coefficients_stable.py`:

    python large_ell_checks.py            # everything
    python large_ell_checks.py A C E      # selected sections

Requires numpy, scipy, sympy, mpmath.  Full run is a few minutes.
"""

from __future__ import annotations

import sys
import warnings
from math import exp, lgamma, log, pi, sqrt

warnings.filterwarnings("ignore")

import numpy as np
from mpmath import mp, mpc, mpf
from mpmath import arg as marg
from mpmath import cosh, exp as mexp, fabs, gamma as Gm, pi as mpi
from mpmath import rf, sinh, sqrt as msq

mp.dps = 50

QHAT = mpf("0.6")
RPLUS = mpf(1)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# A.  spectrum
# ===========================================================================


def h(ell):
    """Conformal weight (supplementary branch; real for l >= 1 at qhat = 0.6)."""
    ell = mpf(ell)
    disc = mpf(1) / 4 + ell * (ell + 1) - QHAT**2
    if disc <= 0:
        raise ValueError(f"principal branch at l={ell}")
    return mpf(1) / 2 + msq(disc)


def section_A():
    rule("A.  SPECTRUM")
    print("  h_l = l+1 - q^2/(2u) - q^4/(8u^3) - q^6/(16u^5) + ... ,   u = l + 1/2")
    print(f"  {'l':>5} {'h_l exact':>18} {'1 term':>12} {'2 terms':>12} {'3 terms':>12}")
    for l in [1, 4, 16, 64, 128]:
        u = mpf(l) + mpf(1) / 2
        e = h(l)
        a1 = l + 1 - QHAT**2 / (2 * u)
        a2 = a1 - QHAT**4 / (8 * u**3)
        a3 = a2 - QHAT**6 / (16 * u**5)
        print(f"  {l:>5} {float(e):18.12f} {float(abs(a1-e)):12.3e} "
              f"{float(abs(a2-e)):12.3e} {float(abs(a3-e)):12.3e}")
    print("  (columns are |truncation - exact|)")

    print(f"\n  Re nu_(n,l) = qhat/2 = {float(QHAT/2)} exactly for every (n,l)"
          "  =>  slow detuning Delta == 0")
    print(f"  {'(n,l)':>10} {'(h_l+n)/2':>20} {'(n+l+1)/2':>14} {'rel err':>10}")
    for (n, l) in [(0, 1), (0, 7), (1, 7), (0, 20), (0, 80), (3, 80)]:
        g = (h(l) + n) / 2
        ap = mpf(n + l + 1) / 2
        print(f"  {str((n,l)):>10} {float(g):20.12f} {float(ap):14.8f} "
              f"{float(abs(ap/g-1)):10.3e}")


# ===========================================================================
# B.  normalisation
# ===========================================================================


def N_inv_sq(n, ell):
    hl = h(ell)
    num = 1j * (-1) ** (hl + 1j * QHAT) * 2 * Gm(n + 1) * Gm(2 * hl + n) * Gm(1 - hl - n - 1j * QHAT)
    den = Gm(hl - 1j * QHAT) * rf(hl + 1j * QHAT, n)
    return RPLUS**2 * num / den


def N(n, ell):
    return 1 / msq(N_inv_sq(n, ell))


_A0 = msq(sinh(mpi * QHAT) * mexp(mpi * QHAT) / (msq(mpi) * RPLUS**2))


def N_asym(n, ell):
    hl = h(ell)
    return _A0 * 2 ** (-hl) * hl ** mpf("-0.25") * msq(hl**n / (Gm(n + 1) * 2**n))


def section_B():
    rule("B.  NORMALISATION  |N_{n,l}|")
    print("  exact overtone ratio:  N_n^-2 / N_0^-2 = n! (2h)_n / [(1-h-n-iq)_n (h+iq)_n]")
    print(f"  {'(n,l)':>10} {'exact':>18} {'Pochhammer form':>20}")
    for (n, l) in [(1, 20), (3, 20), (4, 80), (6, 160)]:
        hl = h(l)
        ex = abs(N(n, l) / N(0, l))
        po = abs(msq(rf(1 - hl - n - 1j * QHAT, n) * rf(hl + 1j * QHAT, n)
                     / (Gm(n + 1) * rf(2 * hl, n))))
        print(f"  {str((n,l)):>10} {float(ex):18.10e} {float(po):20.10e}")

    print("\n  |N| ~ [sinh(pi q) e^{pi q}/(sqrt(pi) r+^2)]^{1/2} 2^{-h} h^{-1/4} "
          "sqrt(h^n/(n! 2^n))")
    print(f"  {'(n,l)':>10} {'exact':>18} {'asymptotic':>18} {'ratio':>10}")
    for (n, l) in [(0, 20), (0, 80), (0, 160), (1, 80), (2, 80), (4, 160)]:
        e = abs(N(n, l))
        p = N_asym(n, l)
        print(f"  {str((n,l)):>10} {float(e):18.8e} {float(p):18.8e} {float(p/e):10.6f}")
    print("  (residual error is the O(n^2/l) of the Pochhammer limits)")


# ===========================================================================
# C.  radial overlap
# ===========================================================================


def P_table(n, ell):
    """Single-mode symbols P^(n)_j by the recurrence used in the project code."""
    hl = h(ell)
    out = [mpc(1)]
    for j in range(n):
        out.append(out[-1] * (-n + j) * (1 - 2 * hl - n + j)
                   / ((j + 1) * (1 - hl - n - 1j * QHAT + j)))
    return out


def _conv(a, b):
    out = [mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def S_terms(modes):
    """modes = [(n,l)] x4 in slot order (a, bbar, c, d).  Returns (terms, G)."""
    (na, la), (nb, lb), (nc, lc), (nd, ld) = modes
    gam = (-(h(la) + na + 1j * QHAT) / 2 - (h(lb) + nb - 1j * QHAT) / 2
           - (h(lc) + nc + 1j * QHAT) / 2 - (h(ld) + nd + 1j * QHAT) / 2)
    P = _conv(_conv(_conv(P_table(na, la),
                          [x.conjugate() for x in P_table(nb, lb)]),
                    P_table(nc, lc)), P_table(nd, ld))
    terms = []
    for j, pj in enumerate(P):
        Sj = ((-1) ** (j + gam + 1) * Gm(1 + j + gam)
              * Gm(-1 - j - 2 * gam - 2 * 1j * QHAT) / Gm(-2 * 1j * QHAT - gam))
        terms.append(pj * Sj)
    G = sum(h(l) + n for (n, l) in modes)
    return terms, G


def S_bare(modes):
    return sum(S_terms(modes)[0])


def S_full(modes):
    (na, la), (nb, lb), (nc, lc), (nd, ld) = modes
    pref = N(na, la) * N(nb, lb).conjugate() * N(nc, lc) * N(nd, ld)
    return pref * S_bare(modes)


def S_asym(ells):
    """Large-l law for the n=0 tower."""
    hs = [h(l) for l in ells]
    H = sum(hs)
    K = sinh(mpi * QHAT) ** 2 * mexp(3 * mpi * QHAT) / (4 * msq(mpi) * RPLUS**4)
    prod = mpf(1)
    for hl in hs:
        prod *= hl ** mpf("-0.25")
    from mpmath import sin as msin
    return K / abs(msin(mpi * (H / 2 + 1j * QHAT))) * (H / 2) ** mpf("-0.5") * prod


def section_C():
    rule("C.  NEAR-ZONE RADIAL OVERLAP")
    try:
        from _project_path import ensure_on_path
        ensure_on_path()
        from coupling_coefficients_stable import CouplingCoefficients
        Co = CouplingCoefficients(float(QHAT), float(RPLUS))
        print("  C1. closed form at arbitrary overtone, vs S_near_abcd")
        print(f"  {'modes (n,l) x4':>36} {'code':>14} {'closed':>14} {'ratio':>10}")
        for ms in [[(0, 1)] * 4,
                   [(1, 2), (0, 3), (2, 2), (1, 1)],
                   [(2, 5), (1, 4), (0, 6), (3, 5)],
                   [(1, 10)] * 4,
                   [(3, 12), (0, 8), (2, 9), (1, 11)],
                   [(2, 20)] * 4]:
            a, b, c, d = [{"n": n, "l": l, "m": 0} for (n, l) in ms]
            code = abs(Co.S_near_abcd(a, b, c, d))
            cl = float(abs(S_full(ms)))
            print(f"  {str(ms):>36} {code:14.6e} {cl:14.6e} {cl/code:10.6f}")
    except ImportError:
        print("  [!] coupling_coefficients_stable not importable -- skipping C1.")

    print("\n  C2. large-l law for the n=0 tower (dyadic factors cancel)")
    print(f"  {'(la,lb,lc,ld)':>22} {'exact':>15} {'asymptotic':>15} {'ratio':>9}")
    for ells in [(10, 10, 10, 10), (40, 40, 40, 40), (80, 80, 80, 80),
                 (40, 8, 24, 24), (80, 16, 48, 48), (100, 30, 70, 60), (50, 50, 2, 2)]:
        e = float(abs(S_full([(0, l) for l in ells])))
        p = float(S_asym(ells))
        print(f"  {str(ells):>22} {e:15.6e} {p:15.6e} {p/e:9.5f}")

    print("\n  C3. overtone staircase:  |S_bare| 2^{-G} ~ l^{-floor((N_S+1)/2)-1/2}")
    print(f"  {'pattern':>16} {'N_S':>4} " +
          " ".join(f"{('l=%d' % l):>13}" for l in [20, 40, 80]) + f" {'slope':>8}")
    for pat in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0),
                (1, 1, 1, 0), (1, 1, 1, 1), (2, 1, 1, 0), (2, 2, 0, 0)]:
        vals = []
        for l in [20, 40, 80]:
            t, G = S_terms([(pat[i], l) for i in range(4)])
            vals.append(float(fabs(sum(t)) * 2 ** (-G)))
        sl = log(vals[2] / vals[1]) / log(2.0)
        print(f"  {str(pat):>16} {sum(pat):>4} " +
              " ".join(f"{v:13.4e}" for v in vals) + f" {sl:8.4f}")

    print("\n  C4. cancellation in the j-sum:  |sum_j t_j| / max_j |t_j|")
    print(f"  {'pattern':>16} {'N_S':>4} " +
          " ".join(f"{('l=%d' % l):>13}" for l in [20, 40, 80, 160]))
    for pat in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0),
                (1, 1, 1, 1), (2, 2, 2, 2)]:
        row = []
        for l in [20, 40, 80, 160]:
            t, _ = S_terms([(pat[i], l) for i in range(4)])
            row.append(float(fabs(sum(t)) / max(fabs(x) for x in t)))
        print(f"  {str(pat):>16} {sum(pat):>4} " + " ".join(f"{v:13.4e}" for v in row))

    print("\n  C5. phase of S:  -> pi/2 when all legs are large, otherwise not")
    print(f"  {'config':>24} {'arg S':>12} {'arg S - pi/2':>14}")
    for c in [(10, 10, 10, 10), (40, 40, 40, 40), (160, 160, 160, 160),
              (20, 2, 20, 2), (80, 2, 80, 2), (320, 2, 320, 2)]:
        v = S_full([(0, l) for l in c])
        a = float(marg(v))
        print(f"  {str(c):>24} {a:12.7f} {a-float(mpi/2):14.3e}")
    print(f"\n  {'l':>6} {'|Re S|/|S| (diagonal)':>24}")
    for l in [10, 40, 160, 320]:
        v = S_full([(0, l)] * 4)
        print(f"  {l:>6} {float(fabs(v.real)/fabs(v)):24.6e}")
    print("  => high-high self-energy is asymptotically pure damping;")
    print("     mixed high-low keeps an O(1) real part (the pump-induced detuning).")


# ===========================================================================
# D.  angular sector
# ===========================================================================


def _lf(n):
    return lgamma(n + 1.0)


def w3j000(j1, j2, j3):
    J = j1 + j2 + j3
    if J % 2 or j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    g = J // 2
    lv = 0.5 * (_lf(J - 2 * j1) + _lf(J - 2 * j2) + _lf(J - 2 * j3) - _lf(J + 1))
    lv += _lf(g) - _lf(g - j1) - _lf(g - j2) - _lf(g - j3)
    return (-1) ** g * exp(lv)


def gaunt_sq(a, b, c):
    w = w3j000(a, b, c)
    return (2 * a + 1) * (2 * b + 1) * (2 * c + 1) / (4 * pi) * w * w


def angular_sum(l1, l2, l3, l4):
    lo = max(abs(l1 - l2), abs(l3 - l4))
    hi = min(l1 + l2, l3 + l4)
    if hi < lo:
        return 0.0
    return sum(gaunt_sq(l1, l2, L) * gaunt_sq(l3, l4, L) / (2 * L + 1)
               for L in range(lo, hi + 1))


def _area4(a, b, c):
    s = (a + b + c) / 2.0
    v = s * (s - a) * (s - b) * (s - c)
    return 4.0 * sqrt(v) if v > 0 else 0.0


def section_D():
    rule("D.  ANGULAR SECTOR")
    try:
        from _project_path import ensure_on_path
        ensure_on_path()
        from coupling_coefficients_stable import CouplingCoefficients
        Co = CouplingCoefficients(float(QHAT), float(RPLUS))

        def brute(l1, l2, l3, l4):
            tot = 0.0
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    for m3 in range(-l3, l3 + 1):
                        m4 = m1 + m2 - m3
                        if abs(m4) > l4:
                            continue
                        tot += abs(Co.Q_element(l1, m1, l2, m2, l3, m3, l4, m4)) ** 2
            return tot

        print("  D1. exact m-sum reduction vs brute force over all magnetic numbers")
        print(f"  {'(l1,l2,l3,l4)':>20} {'brute':>16} {'reduced':>16} {'ratio':>10}")
        for ells in [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 2, 2), (5, 3, 4, 4),
                     (6, 2, 4, 4), (6, 6, 6, 6)]:
            b = brute(*ells)
            r = angular_sum(*ells)
            print(f"  {str(ells):>20} {b:16.8e} {r:16.8e} {r/b:10.6f}")
    except ImportError:
        print("  [!] coupling_coefficients_stable not importable -- skipping D1.")

    print("\n  D2. semiclassical 3j:  3j(j1 j2 j3;000)^2 ~ 2/(pi Lambda)")
    print(f"  {'(j1,j2,j3)':>20} {'exact^2':>15} {'2/(pi Lambda)':>15} {'ratio':>9}")
    for t in [(40, 40, 40), (160, 160, 160), (80, 60, 40), (160, 120, 80), (200, 150, 100)]:
        w2 = w3j000(*t) ** 2
        L = _area4(t[0] + 0.5, t[1] + 0.5, t[2] + 0.5)
        pr = 2.0 / (pi * L)
        print(f"  {str(t):>20} {w2:15.6e} {pr:15.6e} {pr/w2:9.5f}")

    print("\n  D3. stretched corner:  3j(j,j,2j;000)^2 ~ j^{-3/2}/sqrt(8 pi)")

    def stretched(j1, j2):
        return exp(_lf(2 * j1) + _lf(2 * j2) - _lf(2 * j1 + 2 * j2 + 1)
                   + 2 * (_lf(j1 + j2) - _lf(j1) - _lf(j2)))

    print(f"  {'j':>6} {'recursion':>18} {'closed form':>18} {'x j^{3/2}':>12}")
    for j in [10, 40, 160]:
        print(f"  {j:>6} {w3j000(j,j,2*j)**2:18.8e} {stretched(j,j):18.8e} "
              f"{stretched(j,j)*j**1.5:12.6f}")
    print(f"  1/sqrt(8 pi) = {1/sqrt(8*pi):.6f}")

    print("\n  D4. equilateral angular sum:  A = a l^2 ln l + b l^2")
    ls = [32, 64, 128, 256, 512]
    A = [angular_sum(l, l, l, l) for l in ls]
    M = np.array([[l * l * log(l), l * l] for l in ls])
    (a, b), *_ = np.linalg.lstsq(M, np.array(A), rcond=None)
    print(f"  fit on l = {ls}:   a = {a:.6f},  b = {b:.6f}"
          f"   (analytic estimate a = 4/(3 pi^4) = {4/(3*pi**4):.6f})")
    print(f"  {'l':>6} {'A exact':>16} {'fit':>16} {'rel resid':>11}")
    for l, Av in zip(ls, A):
        f = a * l * l * log(l) + b * l * l
        print(f"  {l:>6} {Av:16.7e} {f:16.7e} {abs(f/Av-1):11.3e}")
    P = np.array([[log(l), 1.0] for l in ls])
    (c1, c0), *_ = np.linalg.lstsq(P, np.array([log(v) for v in A]), rcond=None)
    rs = [abs(exp(c0) * l**c1 / Av - 1) for l, Av in zip(ls, A)]
    print(f"  pure power fit A ~ {exp(c0):.5f} l^{c1:.5f}: residuals "
          + ", ".join(f"{r:.1e}" for r in rs) + "  (do not fall -> excluded)")


# ===========================================================================
# E.  shell kernel
# ===========================================================================


def shell_kernel(lam, l1, l2, l3, alpha=1.0):
    A = angular_sum(l1, l2, l3, lam)
    if A == 0.0:
        return 0.0
    s = float(abs(S_full([(0, l1), (0, l3), (0, l2), (0, lam)])))
    return alpha**2 * s * s * A / (2 * lam + 1)


def section_E():
    rule("E.  SHELL KERNEL")
    print("  E1. homogeneity: W(xi * shape) ~ xi^D,  expect D = -2 (with a log)")
    for sh in [(1, 1, 1, 1), (2, 1, 1.5, 1.5), (1, 2, 2, 1), (3, 2, 2, 3)]:
        print(f"  shape {sh}")
        prev = None
        for xi in [32, 64, 128, 256]:
            idx = tuple(int(round(xi * s)) for s in sh)
            v = shell_kernel(*idx)
            if v == 0:
                prev = None
                continue
            sl = log(v / prev[1]) / log(xi / prev[0]) if prev else float("nan")
            print(f"    xi={xi:>4} {str(idx):>24} {v:13.5e} slope {sl:8.4f}")
            prev = (xi, v)

    print("\n  E2. diagonal:  W(l,l,l,l) ~ c ln l / l^2")
    print(f"  {'l':>6} {'W':>16} {'slope':>9}")
    prev = None
    for l in [16, 32, 64, 128]:
        v = shell_kernel(l, l, l, l)
        sl = log(v / prev[1]) / log(l / prev[0]) if prev else float("nan")
        print(f"  {l:>6} {v:16.6e} {sl:9.4f}")
        prev = (l, v)

    l = 24
    base = shell_kernel(l, l, l, l)
    print(f"\n  E3. locality, base l = {l}, W0 = {base:.6e}")
    print("   (a) UV, two legs large: W(l; l, p, p) ~ p^-1")
    print(f"   {'p/l':>7} {'W/W0':>13} {'slope':>9}")
    prev = None
    for p in [l, 2 * l, 4 * l, 8 * l, 16 * l, 32 * l]:
        v = shell_kernel(l, l, p, p)
        sl = log(v / prev[1]) / log(p / prev[0]) if prev else float("nan")
        print(f"   {p/l:7.1f} {v/base:13.4e} {sl:9.4f}")
        prev = (p, v)
    print("   (b) UV, three legs large: W(l; p, p, p) ~ p^-3/2")
    print(f"   {'p/l':>7} {'W/W0':>13} {'slope':>9}")
    prev = None
    for p in [l, 2 * l, 4 * l, 8 * l, 16 * l, 32 * l]:
        v = shell_kernel(l, p, p, p)
        sl = log(v / prev[1]) / log(p / prev[0]) if prev else float("nan")
        print(f"   {p/l:7.1f} {v/base:13.4e} {sl:9.4f}")
        prev = (p, v)
    print("   (c) UV, one leg large: killed by the triangle rule")
    for p in [2 * l, 3 * l, 4 * l]:
        print(f"      p = {p:>4}   W = {shell_kernel(l, l, l, p):.4e}")
    print("   (d) IR, two legs small: W(l; l, p, p) -> const")
    print(f"   {'p/l':>7} {'W/W0':>13}")
    for p in [1, 2, 4, 8, 12, l]:
        print(f"   {p/l:7.3f} {shell_kernel(l, l, p, p)/base:13.4e}")
    print("   (e) IR, all legs small: killed by the triangle rule unless 3p >= l")
    for p in [4, 6, 8, l]:
        print(f"      p = {p:>4}   W = {shell_kernel(l, p, p, p):.4e}")
    print("\n   => locality window for n_l ~ l^-x :   3/4 < x < 1")


def main():
    want = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D", "E"]
    print("=" * 78)
    print(f"large-l verification suite   (qhat = {float(QHAT)}, r+ = {float(RPLUS)})")
    print("=" * 78)
    for key, fn in [("A", section_A), ("B", section_B), ("C", section_C),
                    ("D", section_D), ("E", section_E)]:
        if key in want:
            fn()
    print()


if __name__ == "__main__":
    main()
