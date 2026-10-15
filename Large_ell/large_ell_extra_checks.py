"""
Two supporting measurements quoted in `large_ell_stirring.tex`.

  X1  the UV tail exponent of the collision integral, measured as the increment
      per unit l_max (which IS the integrand at p = l_max), probe shell l0 = 8,
      l_max = 20 ... 60.  This is what supersedes the ray-based locality window.
  X2  the normalisation collapse on the principal branch, taken straight from the
      project's own Anorm and S_near_abcd so that it does not depend on any of my
      back ends: |int psi^2|, |N|, the normalised |S|, and the UN-normalised
      ("bare") quartic overlap over the same band.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np

PROBE = 8
LMAXES = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


def X1(probe=PROBE, cutoffs=tuple(LMAXES)):
    import large_ell_zakharov as Z
    print("=" * 78)
    print(f"X1  UV tail exponent.  probe shell l0 = {probe}, "
          f"l_max = {min(cutoffs)} ... {max(cutoffs)}")
    print("=" * 78)
    print("  dI/dl_max is the integrand at p = l_max; the ray estimate of")
    print("  note Sec. 6.4 predicts it to behave as p^(1/2 - 2x), so convergence")
    print("  would need 1/2 - 2x < -1, i.e. x > 3/4.\n")
    print(f"  {'x':>6} {'1/2-2x':>9} {'measured':>11} {'offset':>9} {'converges':>11}")
    rows = []
    for x in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4):
        I = []
        for L in cutoffs:
            Wt, ells, mult = Z.kernel(L)
            n = ells.astype(float) ** (-x)
            I.append(Z.collision(L, n)[probe - 1])
        I = np.array(I)
        d = np.diff(I)
        mid = 0.5 * (np.array(cutoffs[:-1]) + np.array(cutoffs[1:]))
        m = np.abs(d) > 0
        s = np.polyfit(np.log(mid[m]), np.log(np.abs(d[m])), 1)[0]
        pred = 0.5 - 2 * x
        rows.append((x, pred, s))
        print(f"  {x:6.2f} {pred:9.3f} {s:11.3f} {s-pred:9.3f} "
              f"{('yes' if s < -1 else 'no'):>11}")
    a = np.array(rows)
    print(f"\n  the offset is not constant: {a[0,2]-a[0,1]:.3f} at x = 0.6 falling to "
          f"{a[-1,2]-a[-1,1]:.3f} at x = 1.4 ({a[4,2]-a[4,1]:.3f} at x = 1).")
    j = np.argmax(a[:, 2] < -1)
    x0, x1 = a[j - 1, 0], a[j, 0]
    y0, y1 = a[j - 1, 2], a[j, 2]
    xc = x0 + (-1.0 - y0) * (x1 - x0) / (y1 - y0)
    print(f"  the measured exponent crosses -1 at x = {xc:.3f}: the UV edge, not 3/4.")


def X2(qhats=(6.0, 12.0)):
    from _project_path import ensure_on_path
    ensure_on_path()
    from coupling_coefficients_stable import CouplingCoefficients
    from large_ell_principal import PrincipalBackend, band_edge
    print("\n" + "=" * 78)
    print("X2  normalisation collapse, from the project code alone")
    print("=" * 78)
    for q in qhats:
        Co = CouplingCoefficients(q, 1.0)
        be = PrincipalBackend(qhat=q, rplus=1.0, lmax=band_edge(q) + 2)
        print(f"\n  qhat = {q}, principal band l <= {band_edge(q)}")
        print(f"  {'l':>3} {'mu':>8} {'|int psi^2|':>14} {'|N|':>12} "
              f"{'|S| normalised':>16} {'|S| bare':>14}")
        for l in range(1, band_edge(q) + 3):
            mu = float(be.mu(np.array([float(l)]))[0])
            A = Co.Anorm(0, l)
            N = abs(1.0 / A)
            md = [{"n": 0, "l": l, "m": 0}] * 4
            S = abs(Co.S_near_abcd(*md))
            print(f"  {l:>3} {mu:8.4f} {abs(A)**2:14.5e} {N:12.5e} "
                  f"{S:16.5e} {S/N**4:14.5e}")
        pr = band_edge(q)
        S1 = abs(Co.S_near_abcd(*[{"n": 0, "l": 1, "m": 0}] * 4))
        Sp = abs(Co.S_near_abcd(*[{"n": 0, "l": pr, "m": 0}] * 4))
        b1 = S1 / abs(1 / Co.Anorm(0, 1)) ** 4
        bp = Sp / abs(1 / Co.Anorm(0, pr)) ** 4
        print(f"  across l = 1 .. {pr}:  normalised |S| grows by {Sp/S1:.2e},"
              f"  bare |S| by {bp/b1:.2e}")


if __name__ == "__main__":
    import sys
    want = [w.upper() for w in sys.argv[1:]] or ["X1", "X2"]
    if "X1" in want:
        X1()
    if "X2" in want:
        X2()
    print()
