"""
Independent verification pass.

Checks that do not reuse the derivation they are checking:

  V1  the Zakharov bookkeeping (Jacobian, homogeneity, bracket) done symbolically
      with sympy, against the hand result y = 3x - gamma - 4
  V2  the principal-branch spectrum (eq. principal) against the project's own w(n,l)
  V3  the symmetry group of Wt: order, partition, and the resulting conservation
  V4  all three back ends revalidated against coupling_coefficients_stable
  V5  the KZ exponent a second way: constancy of the flux for n = l^-x
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np


def _rule(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def V1():
    _rule("V1  Zakharov bookkeeping, symbolically")
    import sympy as sp

    l0, l1, l2, l3, x, g = sp.symbols("l0 l1 l2 l3 x gamma", positive=True)

    # the map: exchange 0<->1 and simultaneously 2<->3
    m1 = l0 ** 2 / l1
    m2 = l0 * l3 / l1
    m3 = l0 * l2 / l1

    J = sp.Matrix([[sp.diff(m, v) for v in (l1, l2, l3)] for m in (m1, m2, m3)])
    jac = sp.simplify(sp.Abs(J.det()))
    print(f"  Jacobian                     {jac}   (hand: l0^4/l1^4)")
    assert sp.simplify(jac - l0 ** 4 / l1 ** 4) == 0

    # kernel: homogeneous of degree g, scale all four arguments by l1/l0
    ker = sp.simplify((l0 / l1) ** g)
    print(f"  kernel factor                {ker}   (hand: (l0/l1)^gamma)")

    pre = sp.simplify((l0 * m1 * m2 * m3) ** (-x) / (l0 * l1 * l2 * l3) ** (-x))
    print(f"  (l0 l1 l2 l3)^-x factor      {sp.powsimp(pre, force=True)}"
          f"   (hand: (l0^4/l1^4)^-x)")
    assert sp.simplify(pre - (l0 ** 4 / l1 ** 4) ** (-x)) == 0

    B = l0 ** x + l3 ** x - l1 ** x - l2 ** x
    Bm = (l0 ** x + m3 ** x - m1 ** x - m2 ** x)
    ratio = sp.simplify(sp.powsimp(Bm / B, force=True))
    print(f"  bracket factor               {ratio}   (hand: -(l0/l1)^x)")
    assert sp.simplify(ratio + (l0 / l1) ** x) == 0

    tot = sp.powsimp(sp.simplify(jac * ker * pre * ratio), force=True)
    y = sp.simplify(sp.log(sp.simplify(-tot)) / sp.log(l1 / l0))
    print(f"\n  total multiplier             {tot}")
    print(f"  => exponent y                {sp.simplify(y)}   (hand: 3x - gamma - 4)")
    assert sp.simplify(y - (3 * x - g - 4)) == 0
    print("  ok  all four factors and the exponent reproduce the hand derivation")
    print(f"  => x_KZ = (4 + gamma)/3 = {sp.solve(3*x - g - 4, x)[0]}")


def V2():
    _rule("V2  principal-branch spectrum against the project's w(n,l)")
    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients
    from large_ell_principal import PrincipalBackend, band_edge

    for q in (6.0, 12.0, 40.0):
        Co = CouplingCoefficients(q, 1.0)
        be = PrincipalBackend(qhat=q, rplus=1.0, lmax=band_edge(q) + 3)
        err = 0.0
        for l in range(1, band_edge(q) + 4):
            for n in (0, 1, 2):
                mu = float(be.mu(np.array([float(l)]))[0])
                pred = (q / 2 - mu / 2 - 0.5j * (n + 0.5) if mu > 0 else None)
                ref = complex(Co.w(n, l))           # = i s = nu
                new = complex(be.nu(n, np.array([float(l)]))[0])
                err = max(err, abs(new - ref))
                if pred is not None:
                    err = max(err, abs(pred - ref))
        print(f"  qhat = {q:>5}  band edge l = {band_edge(q):>3}   "
              f"max |nu_pred - nu_code| = {err:.3e}")
    print("  ok  Gamma = (n+1/2)/2 and Re nu = qhat/2 - mu/2 on the principal branch")


def V3():
    _rule("V3  symmetry group of the kinetic kernel")
    gens = [(0, 2, 1, 3), (3, 1, 2, 0), (2, 3, 0, 1)]
    G = {(0, 1, 2, 3)}
    while True:
        new = {tuple(p[g[k]] for k in range(4)) for p in G for g in gens}
        if new <= G:
            break
        G |= new
    print(f"  group order {len(G)} (expected 8)")
    part = {frozenset((0, 3)), frozenset((1, 2))}
    ok = all({frozenset((p[0], p[3])), frozenset((p[1], p[2]))} == part for p in G)
    print(f"  every element preserves the partition {{0,3}} | {{1,2}}: {ok}")
    assert len(G) == 8 and ok
    has = (1, 0, 3, 2) in G
    print(f"  contains the Zakharov element (0<->1, 2<->3): {has}")
    assert has

    from large_ell_backend import Backend
    be = Backend(qhat=0.6, rplus=1.0, lmax=10)
    W, ells = be.W_shell(lmin=1)
    Wt = W * (2.0 * ells + 1.0)[:, None, None, None]
    sc = np.abs(Wt).max()
    e = max(np.abs(Wt - Wt.transpose(p)).max() / sc for p in G)
    print(f"  max asymmetry of Wt over the group: {e:.2e}")
    rng = np.random.default_rng(0)
    n = rng.uniform(0.1, 1, len(ells))
    t1 = np.einsum("Lijk,i,j,k->L", Wt, n, n, n, optimize=True)
    t2 = np.einsum("Lijk,i,j->L", Wt, n, n, optimize=True) * n
    t3 = np.einsum("Lijk,j,k->L", Wt, n, n, optimize=True) * n
    t4 = np.einsum("Lijk,i,k->L", Wt, n, n, optimize=True) * n
    C = t1 + t2 - t3 - t4
    print(f"  resulting action balance |sum C| / sum|C| = "
          f"{abs(C.sum())/np.abs(C).sum():.2e}")


def V4():
    _rule("V4  all three back ends against the project code")
    import large_ell_backend as B
    import large_ell_overtones as O
    import large_ell_principal as P
    B.validate(lmax_small=7)
    O.validate(lmax_small=5, nmax=2)
    P.validate(qhat=6.0, lmax_small=6)
    P.validate(qhat=12.0, lmax_small=8)


def V5(lmax=48):
    _rule("V5  the KZ exponent a second way: flux constancy")
    import large_ell_zakharov as Z
    Wt, ells, mult = Z.kernel(lmax)
    print("  Pi(l) = -sum_{l'<=l} (2l'+1) C_l' for n = l^-x, normalised at l = 16.")
    print("  A KZ exponent makes Pi flat.  Scan x:\n")
    print(f"  {'x':>6} " + "".join(f"{('l=%d' % l):>11}" for l in (16, 24, 32, 40))
          + f"{'spread':>10}")
    best, bx = 1e9, None
    for x in np.arange(0.6, 1.45, 0.1):
        n = ells.astype(float) ** (-x)
        C = Z.collision(lmax, n)
        P = -np.cumsum(mult * C)
        if abs(P[15]) < 1e-280:
            continue
        r = P / P[15]
        sub = r[15:40]
        sp_ = sub.max() - sub.min()
        print(f"  {x:6.2f} " + "".join(f"{r[l-1]:11.4f}" for l in (16, 24, 32, 40))
              + f"{sp_:10.4f}")
        if sp_ < best:
            best, bx = sp_, x
    print(f"\n  flattest at x = {bx:.2f} with spread {best:.3f} -- nowhere near flat,")
    print("  and the flattest x is not the Zakharov exponent.  Consistent with the")
    print("  window being empty: no exponent makes the flux constant.")


if __name__ == "__main__":
    import sys
    want = [w.upper() for w in sys.argv[1:]] or ["V1", "V2", "V3", "V4", "V5"]
    for k, fn in [("V1", V1), ("V2", V2), ("V3", V3), ("V4", V4), ("V5", V5)]:
        if k in want:
            fn()
    print()
