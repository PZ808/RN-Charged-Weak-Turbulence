"""
Steps 2-5 of the large-l programme of `large_ell_stirring.tex`, built on
`large_ell_backend.py`.

  step2  pump-induced linear operator M: its Hermitian / anti-Hermitian split,
         the scaling of Re M_ll and Im M_ll, and the induced slow detuning
  step3  stability of the forced high-l band: the real-linear operator
         A -> -i(M A + P conj(A)) - rho gamma A, its spectrum, the critical
         rho, and the shell content of the most unstable mode
  step4  stationary solution of the full forced nonlinear high-l system,
         undamped and damped, as a function of the cutoff l_diss
  step5  forced-dissipative m-averaged kinetic equation and the shell flux

Everything is axisymmetric (m = 0) and restricted to the fundamental n = 0
tower, following EBHW's equilibrium sector.  alpha = 1 and the pump is
normalised to |c_low|^2 = 1 throughout, so rho = sigma/eps^2 and
rho_4 = sigma/eps^4 are measured in units of alpha |c_low|^2.

    python large_ell_program.py            # all four steps
    python large_ell_program.py 2 3        # selected steps
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from large_ell_backend import Backend

LSTAR = 3          # pump occupies l <= LSTAR
QHAT = 0.6


# ---------------------------------------------------------------------------


def setup(lmax, lstar=LSTAR):
    be = Backend(qhat=QHAT, rplus=1.0, lmax=lmax)
    C, ells = be.C_axisymmetric(alpha=1.0, lmin=1)
    idx = {l: i for i, l in enumerate(ells)}
    low = np.array([idx[l] for l in ells if l <= lstar])
    high = np.array([idx[l] for l in ells if l > lstar])
    hl = np.array([ells[i] for i in high])
    return be, C, ells, low, high, hl


def pump(nlow, seed=7):
    """Unit-norm axisymmetric pump with random phases."""
    rng = np.random.default_rng(seed)
    c = np.exp(1j * rng.uniform(0, 2 * np.pi, nlow))
    return c / np.sqrt(np.sum(np.abs(c) ** 2))


def pump_operators(C, low, high, cl):
    """(F, M, P) of Eq. (forced) in the note."""
    F = np.einsum("Labc,a,b,c->L", C[np.ix_(high, low, low, low)], cl, cl, np.conj(cl))
    M = (np.einsum("LMab,a,b->LM", C[np.ix_(high, high, low, low)], cl, np.conj(cl))
         + np.einsum("LaMb,a,b->LM", C[np.ix_(high, low, high, low)], cl, np.conj(cl)))
    P = np.einsum("LabM,a,b->LM", C[np.ix_(high, low, low, high)], cl, cl)
    return F, M, P


def ham_project(T):
    si = lambda X: X.transpose(0, 2, 1, 3)
    sl = lambda X: X.transpose(3, 1, 2, 0)
    hp = lambda X: np.conj(X.transpose(1, 0, 3, 2))
    S = [T, si(T), sl(T), sl(si(T))]
    return (sum(S) + sum(hp(x) for x in S)) / 8.0


# ---------------------------------------------------------------------------


def step2(lmax=40):
    print("\n" + "=" * 76)
    print("STEP 2  pump-induced linear operator M")
    print("=" * 76)
    be, C, ells, low, high, hl = setup(lmax)

    nz = np.argwhere(np.abs(C) > 1e-14)[:200000]
    par = set(int((ells[a] + ells[b] + ells[c] + ells[d]) % 2) for a, b, c, d in nz)
    print(f"  parity of l_lam+l_i+l_j+l_k over nonzero entries: {par}"
          "   -> total l parity is conserved")

    cl = pump(len(low))
    F, M, P = pump_operators(C, low, high, cl)
    print(f"\n  ||F|| = {np.linalg.norm(F):.4e}   ||M|| = {np.linalg.norm(M):.4e}"
          f"   ||P|| = {np.linalg.norm(P):.4e}")
    print(f"  M Hermitian / anti-Hermitian norms:"
          f" {np.linalg.norm((M+M.conj().T)/2):.4e} / {np.linalg.norm((M-M.conj().T)/2):.4e}")
    print("  NB  P (and the quadratic U) do NOT vanish for an axisymmetric pump:")
    print("      the rule m_i+m_j = m_k+m_lam is satisfied trivially when all m = 0.")

    print(f"\n  source is band-limited to l <= 3 l* = {3*LSTAR}:")
    print(f"  {'l':>5} {'|F_l|':>14}")
    for k, l in enumerate(hl):
        if l <= 3 * LSTAR + 2:
            print(f"  {l:>5} {abs(F[k]):14.6e}")

    d = np.diag(M)
    print(f"\n  {'l':>5} {'Re M_ll':>15} {'Im M_ll':>15} {'|Re|/|M_ll|':>12}")
    for l in [4, 6, 8, 12, 16, 24, 32, 40]:
        if l <= lmax:
            k = int(np.where(hl == l)[0][0])
            print(f"  {l:>5} {d[k].real:15.6e} {d[k].imag:15.6e}"
                  f" {abs(d[k].real)/abs(d[k]):12.4f}")
    m = hl >= 8
    pr = np.polyfit(np.log(hl[m]), np.log(np.abs(d.real[m])), 1)[0]
    pi_ = np.polyfit(np.log(hl[m]), np.log(np.abs(d.imag[m])), 1)[0]
    print(f"  fits for l >= 8:   |Re M_ll| ~ l^{pr:.3f},   |Im M_ll| ~ l^{pi_:.3f}")

    print("\n  induced slow detuning  dDelta = Re(M_ll + M_33 - M_11 - M_22):")
    for q in [(10, 10, 10, 10), (12, 8, 10, 10), (20, 10, 16, 14), (40, 8, 24, 24)]:
        if max(q) <= lmax:
            ks = [int(np.where(hl == l)[0][0]) for l in q]
            dD = d[ks[0]].real + d[ks[3]].real - d[ks[1]].real - d[ks[2]].real
            print(f"      {str(q):>20}   dDelta = {dD:+.6e}")
    print("  (the linear spectrum has Delta == 0, so this is the only real detuning)")


def step3(lmax=40):
    print("\n" + "=" * 76)
    print("STEP 3  stability of the forced high-l band")
    print("=" * 76)
    be, C, ells, low, high, hl = setup(lmax)
    NH = len(high)
    gam = 0.5 * be.h(hl)
    cl = pump(len(low))
    C_H = ham_project(C)

    def spectrum(T, rho):
        _, M, P = pump_operators(T, low, high, cl)
        J = np.zeros((2 * NH, 2 * NH))
        for k in range(NH):
            for s, e0 in ((0, 1.0), (1, 1j)):
                e = np.zeros(NH, dtype=complex)
                e[k] = e0
                v = -1j * (M @ e + P @ np.conj(e)) - rho * gam * e
                J[:NH, k + s * NH] = v.real
                J[NH:, k + s * NH] = v.imag
        return np.linalg.eigvals(J), J

    print(f"  {'rho = sigma/eps^2':>18} {'max Re lam (full C)':>22} {'max Re lam (C_Ham)':>21}")
    for rho in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        a = spectrum(C, rho)[0].real.max()
        b = spectrum(C_H, rho)[0].real.max()
        print(f"  {rho:18.2f} {a:22.6e} {b:21.6e}")
    print("  -> the Hamiltonian projection is neutrally stable at rho = 0;")
    print("     the instability comes entirely from the anti-Hermitian part of C.")

    lo, hi = 0.0, 200.0
    while hi - lo > 1e-6 * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if spectrum(C, mid)[0].real.max() > 0:
            lo = mid
        else:
            hi = mid
    rc = 0.5 * (lo + hi)
    print(f"\n  critical rho_c = {rc:.5f}  ->  the band is unstable for"
          f"  sigma < {rc:.3f} alpha |c_low|^2 eps^2")

    w, J = spectrum(C, 0.9 * rc)
    wv, V = np.linalg.eig(J)
    j = int(np.argmax(wv.real))
    u = V[:NH, j] + 1j * V[NH:, j]
    p = np.abs(u) ** 2
    p /= p.sum()
    print(f"  most unstable mode at 0.9 rho_c (Re lam = {wv[j].real:.4e}):"
          f"  mean l = {np.sum(p*hl):.2f}")
    print("    shell weights (top 6): "
          + ", ".join(f"l={hl[k]}:{p[k]:.3f}" for k in np.argsort(-p)[:6]))


def _solve_stationary(lmax, rho, seed=7, nstep=24):
    be, C, ells, low, high, hl = setup(lmax)
    NH = len(high)
    gam = 0.5 * be.h(hl)
    c0 = pump(len(low), seed)
    Csub = C[high]

    def resid(x, s):
        A = x[:NH] + 1j * x[NH:]
        a = np.zeros(len(ells), dtype=complex)
        a[low] = s * c0
        a[high] = A
        G = (np.einsum("Lijk,i,j,k->L", Csub, a, a, np.conj(a), optimize=True)
             - 1j * rho * gam * A)
        return np.concatenate([G.real, G.imag])

    x = np.zeros(2 * NH)
    for s in np.linspace(1.0 / nstep, 1.0, nstep):
        x = root(resid, x, args=(s,), method="hybr", tol=1e-12).x
    return hl, x[:NH] + 1j * x[NH:], float(np.linalg.norm(resid(x, 1.0)))


def step4(cutoffs=(20, 30, 40)):
    print("\n" + "=" * 76)
    print("STEP 4  stationary solution of the forced nonlinear high-l system")
    print("=" * 76)
    for rho, tag in [(0.0, "undamped (EBHW-like)"), (8.0, "damped, above rho_c")]:
        print(f"\n  --- rho = {rho}   {tag} ---")
        store = {}
        for L in cutoffs:
            hl, A, res = _solve_stationary(L, rho)
            store[L] = (hl, A)
            m = hl >= 8
            p = np.polyfit(np.log(hl[m]), np.log(np.abs(A[m])), 1)[0]
            d = [np.log2(abs(A[k + 2] / A[k])) for k in range(len(hl) - 2) if hl[k] >= 10]
            print(f"   l_diss={L:>3}  residual {res:.1e}   power fit l^{p:+.3f}"
                  f"   mean log2|A_(l+2)/A_l| = {np.mean(d):+.4f}")
        print(f"   {'l':>5} " + " ".join(f"{('|A| @%d' % L):>16}" for L in cutoffs))
        for l in [4, 8, 12, 16, 20]:
            row = f"   {l:>5} "
            for L in cutoffs:
                hl, A = store[L]
                k = int(np.where(hl == l)[0][0])
                row += f" {abs(A[k]):16.6e}"
            print(row)


def _kinetic(lmax, rho4, Tmax=6e4, inj=None):
    be = Backend(qhat=QHAT, rplus=1.0, lmax=lmax)
    W, ells = be.W_shell(lmin=1)
    mult = 2 * ells + 1.0
    Wt = W * mult[:, None, None, None]          # fully symmetric kernel
    gam = 0.5 * be.h(ells)
    S = np.zeros(len(ells))
    if inj is None:                              # |F_l|^2 from step 2
        inj = {4: 1.512, 5: 1.508, 6: 0.693, 7: 0.581, 8: 0.201, 9: 0.080}
    for l, v in inj.items():
        if l <= lmax:
            S[l - 1] = v ** 2
    S /= np.sum(mult * S)

    def Cop(n):
        t1 = np.einsum("Lijk,i,j,k->L", Wt, n, n, n, optimize=True)
        t2 = np.einsum("Lijk,i,j->L", Wt, n, n, optimize=True) * n
        t3 = np.einsum("Lijk,j,k->L", Wt, n, n, optimize=True) * n
        t4 = np.einsum("Lijk,i,k->L", Wt, n, n, optimize=True) * n
        return (t1 + t2 - t3 - t4) / mult

    D = rho4 * gam / (2 * np.pi)
    sol = solve_ivp(lambda T, n: Cop(np.maximum(n, 0)) - D * n + S,
                    (0, Tmax), np.full(len(ells), 1e-6),
                    method="LSODA", rtol=1e-8, atol=1e-14)
    n = np.maximum(sol.y[:, -1], 0)
    return ells, mult, S, D, Cop, n


def step5(cutoffs=(20, 30, 40), rho4=1.0):
    print("\n" + "=" * 76)
    print("STEP 5  forced-dissipative kinetic equation and the shell flux")
    print("=" * 76)
    ells, mult, S, D, Cop, n = _kinetic(16, 0.0, Tmax=1.0)
    rng = np.random.default_rng(3)
    nt = rng.uniform(0.1, 1.0, len(ells))
    num = abs(np.sum(mult * Cop(nt)))
    den = np.sum(mult * np.abs(Cop(nt)))
    print(f"  conservation: |sum_l (2l+1) C_l| / sum_l (2l+1)|C_l| = {num/den:.2e}")
    print("  (the m-averaged kernel W*(2l+1) is fully index-symmetric, so the")
    print("   collision operator conserves the total action without any projection)")

    print(f"\n  Pi(L) = -sum_(l<=L) (2l+1) C_l[n],  normalised to the injection,"
          f"  rho_4 = {rho4}")
    for L in cutoffs:
        ells, mult, S, D, Cop, n = _kinetic(L, rho4)
        Cl = Cop(n)
        Pi = -np.cumsum(mult * Cl) / np.sum(mult * S)
        bal = np.max(np.abs(Cl - D * n + S)) / np.max(np.abs(D * n))
        print(f"\n   l_max = {L:>3}   steady-state residual {bal:.1e}")
        print(f"   {'l':>5} {'n_l':>14} {'Pi(l)':>12}")
        for l in [6, 10, 16, 20, 30, 40]:
            if l <= L:
                print(f"   {l:>5} {n[l-1]:14.5e} {Pi[l-1]:12.5f}")
        sub = (ells >= 10) & (ells <= L - 4)
        if sub.sum() > 3:
            print(f"   flux spread over 10 <= l <= {L-4}: "
                  f"{Pi[sub].max()-Pi[sub].min():.4f}")
    print("\n  -> no plateau, and the interior flux spread saturates rather than")
    print("     shrinking as the range widens: no constant-flux (KZ) cascade.")


def main():
    want = sys.argv[1:] or ["2", "3", "4", "5"]
    for k, fn in [("2", step2), ("3", step3), ("4", step4), ("5", step5)]:
        if k in want:
            fn()
    print()


if __name__ == "__main__":
    main()
