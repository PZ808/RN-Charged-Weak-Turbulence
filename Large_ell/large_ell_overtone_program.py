"""
Overtone audit of the large-l programme.

`large_ell_program.py` runs the whole programme on the fundamental (n = 0)
tower.  This script asks whether that truncation is legitimate, in three steps:

  stepO1  the overtone law: how |S| depends on the overtone pattern, in
          unit-normalised modes, and how that differs from the bare staircase
  stepO2  the nine-mode truncation of the main paper: how much of the coupling
          weight is overtone-carrying
  stepO3  the forced high-l band and the kinetic flux, recomputed with the
          n <= nmax tower, against the n = 0 answers

    python large_ell_overtone_program.py            # all
    python large_ell_overtone_program.py O1 O3
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp

from large_ell_backend import Backend
from large_ell_overtones import OvertoneBackend

QHAT = 0.6
LSTAR = 3
NINE = [(0, 1, -1), (0, 2, 0), (1, 2, 2), (2, 1, 1), (1, 1, -1),
        (1, 1, 0), (1, 1, 1), (2, 1, 0), (0, 1, 1)]


def _rule(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


# ---------------------------------------------------------------------------


def stepO0():
    _rule("STEP O0  conditioning of the exact overtone j-sum in double precision")
    try:
        import mpmath as mp
        import large_ell_checks as LC
    except ImportError:
        print("  (mpmath unavailable; skipped)")
        return
    mp.mp.dps = 120
    print("  relative error of the double-precision j-sum against 120 digits.")
    print("  The same cancellation afflicts the project's S_near_abcd, which")
    print("  evaluates the same sum term by term.\n")
    lls = (2, 4, 8, 16, 32)
    print(f"  {'pattern':<16}{'NS':>3}" + "".join(f"{('l=%d' % l):>12}" for l in lls))
    for pat in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 1),
                (2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4)]:
        row = []
        for l in lls:
            ob = OvertoneBackend(qhat=QHAT, rplus=1.0, lmax=l,
                                 nmax=max(1, max(pat)), lmin=l)
            S = ob.S_tensor()
            ix = {m: a for a, m in enumerate(ob.modes)}
            dp = S[ix[(pat[0], l)], ix[(pat[2], l)], ix[(pat[1], l)], ix[(pat[3], l)]]
            ex = complex(LC.S_full([(n, l) for n in pat]))
            row.append(abs(dp / ex - 1))
        print(f"  {str(pat):<16}{sum(pat):>3}" + "".join(f"{u:12.2e}" for u in row))
    print("\n  -> double precision is safe to NS <= 4 at any l and to NS = 8 up to")
    print("     l ~ 30; beyond that the overtone overlap needs extended precision.")


def stepO1(lls=(2, 4, 8, 16, 32, 64, 128)):
    _rule("STEP O1  the overtone law in unit-normalised modes")

    def ratio(pat, l):
        ob = OvertoneBackend(qhat=QHAT, rplus=1.0, lmax=l,
                             nmax=max(pat), lmin=l)
        S = ob.S_tensor()
        ix = {m: a for a, m in enumerate(ob.modes)}
        num = abs(S[ix[(pat[0], l)], ix[(pat[2], l)],
                    ix[(pat[1], l)], ix[(pat[3], l)]])
        den = abs(S[ix[(0, l)], ix[(0, l)], ix[(0, l)], ix[(0, l)]])
        return num / den

    pats = [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1),
            (2, 0, 0, 0), (2, 2, 0, 0), (2, 2, 2, 2), (3, 2, 1, 0)]
    print("  |S(pattern)| / |S(0,0,0,0)| on the diagonal l_i = l\n")
    print(f"  {'pattern':<14}{'NS':>3}" + "".join(f"{('l=%d' % l):>10}" for l in lls)
          + f"{'slope':>9}{'pred':>7}")
    for p in pats:
        v = np.array([ratio(p, l) for l in lls])
        m = np.array(lls) >= 16
        s = np.polyfit(np.log(np.array(lls, float)[m]), np.log(v[m]), 1)[0]
        NS = sum(p)
        print(f"  {str(p):<14}{NS:>3}" + "".join(f"{u:10.4f}" for u in v)
              + f"{s:9.3f}{NS/2 - ((NS+1)//2):7.1f}")
    print("\n  predicted exponent  NS/2 - floor((NS+1)/2)  =  -1/2 (NS odd), 0 (NS even).")
    print("  The bare staircase l^-floor((NS+1)/2)-1/2 of the note is almost exactly")
    print("  cancelled by the overtone content of the normalisation, |N| ~ h^{n/2}:")
    print("  in unit-normalised modes the overtone channels are NOT suppressed.")

    print("\n  uniform tower, R_n = |S(n,n,n,n)|/|S(0,0,0,0)| at fixed l.")
    print("  NB  the j-sum cancels (that is the staircase), so for NS = 4n > 8 this")
    print("      ratio must be evaluated in extended precision -- see stepO0.")
    try:
        import mpmath as mp
        import large_ell_checks as LC
        mp.mp.dps = 150
        print(f"  {'n':>3}" + "".join(f"{('l=%d' % l):>10}" for l in (16, 32, 64, 128))
              + f"{'l->inf':>10}")
        for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            v = []
            for l in (16, 32, 64, 128):
                v.append(float(abs(LC.S_full([(n, l)] * 4))
                               / abs(LC.S_full([(0, l)] * 4))))
            x = np.array([1.0 / l for l in (16, 32, 64, 128)])
            ext = np.polyfit(x, v, 2)[-1]
            print(f"  {n:>3}" + "".join(f"{u:10.4f}" for u in v) + f"{ext:10.4f}")
    except ImportError:
        print("  (mpmath unavailable; skipped)")
    print("\n  decay in n is algebraic: the local slope of log R vs log n is -0.28")
    print("  on n in [1,8], -0.39 on [8,32], -0.40 on [16,32], i.e. R_n -> n^-1/2.")
    print("  So sum_n R_n^2 diverges logarithmically and the overtone sum is cut")
    print("  off only by the near-extremal bound n << r_+/sigma, giving a weight")
    print("  enhancement ~ ln(r_+/sigma) over the fundamental tower alone.")

    be = Backend(qhat=QHAT, rplus=1.0, lmax=max(lls))
    print("\n  damping gives no help at large l either:  Gamma_{n,l}/Gamma_{0,l} = 1 + n/h_l")
    print(f"  {'n':>3}" + "".join(f"{('l=%d' % l):>10}" for l in lls))
    h = be.h(np.array(lls, float))
    for n in (1, 2, 4):
        print(f"  {n:>3}" + "".join(f"{u:10.4f}" for u in 1 + n / h))


# ---------------------------------------------------------------------------


def stepO2():
    _rule("STEP O2  overtone content of the paper's nine-mode truncation")
    try:
        from _project_path import ensure_on_path
        ensure_on_path()
        from coupling_coefficients_stable import CouplingCoefficients
    except ImportError:
        print("  (project code unavailable; skipped)")
        return
    modes = [{"n": n, "l": l, "m": m} for (n, l, m) in NINE]
    Co = CouplingCoefficients(QHAT, 1.0)
    C = Co.build_C_notes(modes, alpha=1.0)
    nv = np.array([m["n"] for m in modes])
    NS = (nv[:, None, None, None] + nv[None, :, None, None]
          + nv[None, None, :, None] + nv[None, None, None, :])
    A = np.abs(C) ** 2
    tot = A.sum()
    nz = np.count_nonzero(np.abs(C) > 1e-14)
    print(f"  {len(modes)} modes, {nz} active quartets, sum|C|^2 = {tot:.5e}")
    print(f"\n  {'NS':>3} {'active':>7} {'sum|C|^2':>14} {'share':>8} {'max|C|':>11}")
    for ns in range(NS.max() + 1):
        msk = NS == ns
        k = np.count_nonzero(np.abs(C[msk]) > 1e-14)
        if not k:
            continue
        print(f"  {ns:>3} {k:>7} {A[msk].sum():14.5e} "
              f"{A[msk].sum()/tot:8.4f} {np.sqrt(A[msk].max()):11.4e}")
    m0 = NS == 0
    print(f"\n  n = 0 only:  {np.count_nonzero(np.abs(C[m0])>1e-14)}/{nz} quartets,"
          f"  {A[m0].sum()/tot:.1%} of sum|C|^2")
    print("  -> at l = 1, 2 the truncation to the fundamental tower keeps under a")
    print("     fifth of the coupling weight and 19 of 819 quartets.  The paper's")
    print("     nine-mode network is an overtone network.")


# ---------------------------------------------------------------------------


def _setup_ov(lmax, nmax, lstar=LSTAR, pump_n0=True):
    """Build the (n, l) coupling tensor and split into pump and response.

    With `pump_n0` the pump is held in the fundamental tower, so that it is
    literally the same object at every nmax and the only thing that changes
    between runs is the overtone content of the response.
    """
    ob = OvertoneBackend(qhat=QHAT, rplus=1.0, lmax=lmax, nmax=nmax)
    C = ob.C_tensor(alpha=1.0)
    L, N = ob.l_of, ob.n_of
    islow = (L <= lstar) & ((N == 0) if pump_n0 else True)
    low = np.where(islow)[0]
    high = np.where(~islow)[0]
    return ob, C, low, high


def _pump(nlow, seed=7):
    rng = np.random.default_rng(seed)
    c = np.exp(1j * rng.uniform(0, 2 * np.pi, nlow))
    return c / np.sqrt(np.sum(np.abs(c) ** 2))


def _pump_ops(C, low, high, cl):
    F = np.einsum("Labc,a,b,c->L", C[np.ix_(high, low, low, low)],
                  cl, cl, np.conj(cl))
    M = (np.einsum("LMab,a,b->LM", C[np.ix_(high, high, low, low)], cl, np.conj(cl))
         + np.einsum("LaMb,a,b->LM", C[np.ix_(high, low, high, low)], cl, np.conj(cl)))
    P = np.einsum("LabM,a,b->LM", C[np.ix_(high, low, low, high)], cl, cl)
    return F, M, P


_PERMS = None


def _sym_group():
    """The order-8 group generated by i<->j, k<->lam and (ij)<->(k lam),
    acting on an array indexed [lam, i, j, k]."""
    global _PERMS
    if _PERMS is None:
        gens = [(0, 2, 1, 3), (3, 1, 2, 0), (2, 3, 0, 1)]
        G = {(0, 1, 2, 3)}
        while True:
            new = {tuple(p[g[k]] for k in range(4)) for p in G for g in gens}
            if new <= G:
                break
            G |= new
        _PERMS = sorted(G)
    return _PERMS


def _symmetrise(T):
    P = _sym_group()
    return sum(T.transpose(p) for p in P) / len(P)


def _ham(T):
    si = lambda X: X.transpose(0, 2, 1, 3)
    sl = lambda X: X.transpose(3, 1, 2, 0)
    hp = lambda X: np.conj(X.transpose(1, 0, 3, 2))
    S = [T, si(T), sl(T), sl(si(T))]
    return (sum(S) + sum(hp(x) for x in S)) / 8.0


def stepO3(lmax=16, nmaxes=(0, 1, 2)):
    _rule(f"STEP O3  stability and flux with the overtone tower (l_max = {lmax})")

    print("  (a) forced high-l band: critical rho and the most unstable mode")
    print("      (the pump is held in the n = 0 tower, l <= 3, with the same")
    print("       phases at every nmax, so only the response changes)\n")
    print(f"  {'nmax':>5} {'seed':>5} {'modes':>6} {'max Re lam (rho=0)':>20} "
          f"{'C_Ham':>12} {'rho_c':>10} {'<l>':>7} {'<n>':>7}")
    for nmax in nmaxes:
      for seed in (7, 11, 23):
        ob, C, low, high = _setup_ov(lmax, nmax)
        NH = len(high)
        gam = ob.gamma_damp()[high]
        hl, nl = ob.l_of[high], ob.n_of[high]
        cl = _pump(len(low), seed)
        CH = _ham(C)

        def spec(T, rho):
            _, M, P = _pump_ops(T, low, high, cl)
            J = np.zeros((2 * NH, 2 * NH))
            for k in range(NH):
                for s, e0 in ((0, 1.0), (1, 1j)):
                    e = np.zeros(NH, dtype=complex)
                    e[k] = e0
                    v = -1j * (M @ e + P @ np.conj(e)) - rho * gam * e
                    J[:NH, k + s * NH] = v.real
                    J[NH:, k + s * NH] = v.imag
            return np.linalg.eig(J)

        a = spec(C, 0.0)[0].real.max()
        b = spec(CH, 0.0)[0].real.max()
        lo, hi = 0.0, 400.0
        while hi - lo > 1e-6 * max(1.0, hi):
            mid = 0.5 * (lo + hi)
            if spec(C, mid)[0].real.max() > 0:
                lo = mid
            else:
                hi = mid
        rc = 0.5 * (lo + hi)
        wv, V = spec(C, 0.9 * rc)
        j = int(np.argmax(wv.real))
        u = V[:NH, j] + 1j * V[NH:, j]
        p = np.abs(u) ** 2
        p /= p.sum()
        print(f"  {nmax:>5} {seed:>5} {len(ob.modes):>6} {a:20.6e} {b:12.3e} "
              f"{rc:10.4f} {np.sum(p*hl):7.2f} {np.sum(p*nl):7.3f}")
    print("\n  -> the instability survives the overtone tower at every seed.  Two")
    print("     things change: the Hamiltonian projection is no longer exactly")
    print("     neutral once n > 0 is present (P opens a parametric channel), and")
    print("     the most unstable mode acquires a small overtone content, but the")
    print("     growth is still dominated by the anti-Hermitian part of C.")

    print("\n  (b) m-averaged kinetic flux on the (n, l) shells\n")
    print("  The collision integral conserves the total action iff the kernel")
    print("  Wt = (2l_lam+1) W is invariant under i<->j, k<->lam and (ij)<->(k lam).")
    print("  On the n = 0 tower it is.  With overtones the k<->lam leg fails: the")
    print("  conjugated radial slot is not interchangeable once n_k /= n_lam.")
    print("  So we split Wt = Wt_S + Wt_A, run the conservative part, and measure")
    print("  the action production that the remainder puts in.\n")
    for nmax in nmaxes:
        ob, _, low, high = _setup_ov(lmax, nmax)
        W = ob.W_shell()
        L = ob.l_of
        mult = 2.0 * L + 1.0
        Wraw = W * mult[:, None, None, None]
        Wt = _symmetrise(Wraw)
        WA = Wraw - Wt
        asym = np.abs(WA).max() / np.abs(Wraw).max()
        gam = ob.gamma_damp()
        inj = np.zeros(len(L))
        inj[(L >= 4) & (L <= 9) & (ob.n_of == 0)] = 1.0
        inj /= np.sum(mult * inj)

        def _C(K, n):
            t1 = np.einsum("Lijk,i,j,k->L", K, n, n, n, optimize=True)
            t2 = np.einsum("Lijk,i,j->L", K, n, n, optimize=True) * n
            t3 = np.einsum("Lijk,j,k->L", K, n, n, optimize=True) * n
            t4 = np.einsum("Lijk,i,k->L", K, n, n, optimize=True) * n
            return (t1 + t2 - t3 - t4) / mult

        Cop = lambda n: _C(Wt, n)

        rng = np.random.default_rng(3)
        nt = rng.uniform(0.1, 1.0, len(L))
        cons = abs(np.sum(mult * Cop(nt))) / np.sum(mult * np.abs(Cop(nt)))
        craw = (abs(np.sum(mult * _C(Wraw, nt)))
                / np.sum(mult * np.abs(_C(Wraw, nt))))

        D = 1.0 * gam / (2 * np.pi)
        sol = solve_ivp(lambda T, n: Cop(np.maximum(n, 0)) - D * n + inj,
                        (0, 6e4), np.full(len(L), 1e-6),
                        method="LSODA", rtol=1e-8, atol=1e-14)
        n = np.maximum(sol.y[:, -1], 0)
        Cl = Cop(n)
        prod = np.sum(mult * _C(WA, n)) / np.sum(mult * inj)
        # shell-resolved flux: order the (n,l) modes by l, then n
        order = np.lexsort((ob.n_of, L))
        Pi = -np.cumsum((mult * Cl)[order]) / np.sum(mult * inj)
        Ls = L[order]
        occ = np.array([n[(L == l)].sum() for l in range(1, lmax + 1)])
        frac0 = (n[(ob.n_of == 0)].sum() / n.sum()) if n.sum() else np.nan
        sub = (Ls >= 10) & (Ls <= lmax - 2)
        print(f"  nmax = {nmax}:  |Wt_A|/|Wt| = {asym:.2e};  action balance "
              f"{craw:.1e} (raw) -> {cons:.1e} (symmetrised)")
        print(f"      action production by Wt_A in the steady state: "
              f"{prod:+.4f} x injection")
        print(f"      Pi/inj at l = 10, 14, {lmax-2}: "
              + ", ".join(f"{Pi[np.where(Ls==l)[0][-1]]:.4f}"
                          for l in (10, 14, lmax - 2)))
        print(f"      interior spread {Pi[sub].max()-Pi[sub].min():.4f},"
              f"  n=0 share of the occupation {frac0:.3f}")
        print(f"      occupation by l: "
              + ", ".join(f"{l}:{occ[l-1]:.2e}" for l in (4, 8, 12, lmax)))
    print("\n  -> no plateau in Pi at any nmax: the absence of a constant-flux")
    print("     cascade is not an artefact of the n = 0 truncation.")


# ---------------------------------------------------------------------------


def main():
    want = [w.upper() for w in sys.argv[1:]] or ["O0", "O1", "O2", "O3"]
    for k, fn in [("O0", stepO0), ("O1", stepO1), ("O2", stepO2), ("O3", stepO3)]:
        if k in want:
            fn()
    print()


if __name__ == "__main__":
    main()
