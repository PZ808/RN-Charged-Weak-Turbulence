"""Proper validation: PDE with shifted IC tracks the slow-amp prediction.

The slow-amp predicted field ψ_pred(t) = ε·exp((s_0 + Δs)t)·b_0 has
ψ̇_pred(0) = ε·(s_0 + Δs)·b_0  where  Δs = −i·ε²·C.

Running the PDE with the *linear* IC π(0) = ε·s_0·b_0 means we're starting
from a different initial condition than the slow-amp prediction's. The
mismatch ε·Δs·b_0 generates a boundary-layer transient with t² scaling at
small t.

If instead we initialize the PDE with the **shifted** velocity
π(0) = ε·(s_0 + Δs)·b_0, the PDE IC matches the slow-amp prediction's IC,
and the residual (PDE − ψ_pred) is just the next-order multi-scale
correction O(ε⁵).

Two columns × two rows:
  Row 1  Re of (PDE − linear)  vs  Re of (ψ_pred − linear)  — overlay,
         showing the slow-amp shift is captured by both.
  Row 2  Same for Im.

A small "residual" curve shows what's left after matching IC; it should be
much smaller than the predicted shift itself.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_cwt import b_near, build_C_notes_cached, load_S_table, s
from rn_cwt import q as q_default
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    eps = 0.3
    alpha = 1.0
    A0 = 1.0 + 0.0j
    s0 = s(0, 0)
    T = 2.0
    nout = 400

    table = load_S_table()
    C0000 = build_C_notes_cached(
        [{"n": 0, "l": 0, "m": 0}], table, alpha=alpha
    )[0, 0, 0, 0]
    Delta_s = -1j * eps**2 * C0000 * abs(A0)**2
    print(f"C = {C0000:.4e}")
    print(f"Δs = -i ε² C |A|² = {Delta_s:.4e}")
    print(f"|ε·Δs·b_0(mid)| sets the velocity IC mismatch ~ {eps*abs(Delta_s)*0.4:.3e}")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(0, 0, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    psi0 = eps * A0 * b0
    pi_linear = eps * A0 * s0 * b0
    pi_shifted = eps * A0 * (s0 + Delta_s) * b0

    # Two PDE runs
    t_arr, psi_pde_lin, _, sol_a = evolve_kg(
        grid, psi0, pi_linear, tmax=T, q=q_default, m_eff_sq=0.0,
        alpha=alpha, nout=nout,
    )
    _, psi_pde_shift, _, sol_b = evolve_kg(
        grid, psi0, pi_shifted, tmax=T, q=q_default, m_eff_sq=0.0,
        alpha=alpha, nout=nout,
    )
    assert sol_a.success and sol_b.success

    # Reference baselines
    psi_lin = eps * A0 * np.exp(s0 * t_arr) * b0[mid]                 # linear evolution
    psi_pred = eps * A0 * np.exp((s0 + Delta_s) * t_arr) * b0[mid]    # slow-amp prediction

    # Sample at mid
    psi_pde_lin_mid = psi_pde_lin[:, mid]
    psi_pde_shift_mid = psi_pde_shift[:, mid]

    # The "linear nonlinear correction" — what we've been plotting:
    dpsi_pde_lin = psi_pde_lin_mid - psi_lin
    # The "slow-amp predicted correction" — exact, linear in t at leading order:
    dpsi_pred = psi_pred - psi_lin
    # The PDE with shifted IC compared to the slow-amp prediction:
    dpsi_pde_shift_vs_pred = psi_pde_shift_mid - psi_pred
    # And the PDE with shifted IC compared to linear (for visual comparison):
    dpsi_pde_shift_vs_lin = psi_pde_shift_mid - psi_lin

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        rf"Validation with shifted IC — single-mode $b_0$, $\epsilon={eps}$, "
        rf"$\alpha={alpha}$, $r_* = {rstar[mid]:.2f}$"
        f"\n"
        rf"Slow-amp prediction: $\psi_{{\rm pred}} = \epsilon\,\exp((s_0 + \Delta s)t)\,b_0$, "
        rf"$\Delta s = -i\epsilon^2 C \approx {Delta_s.real:.2e} + {Delta_s.imag:.2e}\,i$",
        fontsize=10,
    )

    # Row 1: Re — three traces compared
    ax = axes[0, 0]
    ax.plot(t_arr, np.real(dpsi_pde_lin), "-", color="0.5", lw=1.2,
            label=r"PDE (linear IC) − linear")
    ax.plot(t_arr, np.real(dpsi_pred), "-", color="C3", lw=2.0,
            label=r"slow-amp pred − linear  (= $\epsilon(\exp(\Delta s\,t)-1)\,e^{s_0 t}\,b_0$)")
    ax.plot(t_arr, np.real(dpsi_pde_shift_vs_lin), "--", color="k", lw=1.4,
            label=r"PDE (shifted IC) − linear")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re $(\psi - \psi_{\rm lin})$")
    ax.set_title("Real part: shifted-IC PDE overlays the slow-amp prediction")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(t_arr, np.imag(dpsi_pde_lin), "-", color="0.5", lw=1.2,
            label=r"PDE (linear IC) − linear")
    ax.plot(t_arr, np.imag(dpsi_pred), "-", color="C3", lw=2.0,
            label=r"slow-amp pred − linear")
    ax.plot(t_arr, np.imag(dpsi_pde_shift_vs_lin), "--", color="k", lw=1.4,
            label=r"PDE (shifted IC) − linear")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im $(\psi - \psi_{\rm lin})$")
    ax.set_title("Imaginary part")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)

    # Row 2: residual PDE_shift − ψ_pred (the next-order correction we'd ignore)
    ax = axes[1, 0]
    ax.plot(t_arr, np.real(dpsi_pde_shift_vs_pred), "-", color="C2", lw=1.5,
            label=r"Re $(\psi_{\rm PDE,\,shifted} - \psi_{\rm pred})$  (residual)")
    ax.plot(t_arr, np.real(dpsi_pred), ":", color="C3", lw=1.0, alpha=0.6,
            label=r"Re $(\psi_{\rm pred} - \psi_{\rm lin})$ for scale ref")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re residual")
    ax.set_title("Re residual (= next-order multi-scale correction)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(t_arr, np.imag(dpsi_pde_shift_vs_pred), "-", color="C2", lw=1.5,
            label=r"Im $(\psi_{\rm PDE,\,shifted} - \psi_{\rm pred})$  (residual)")
    ax.plot(t_arr, np.imag(dpsi_pred), ":", color="C3", lw=1.0, alpha=0.6,
            label=r"Im $(\psi_{\rm pred} - \psi_{\rm lin})$ for scale ref")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im residual")
    ax.set_title("Im residual")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    out = "examples/13_shifted_ic_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")

    # Snapshot table
    print()
    print(f"{'t':>5}  {'|PDE_lin − lin|':>16}  {'|pred − lin|':>14}  "
          f"{'|PDE_shift − pred|':>20}  {'rel residual':>14}")
    for tt in [0.25, 0.5, 1.0, 1.5, 2.0]:
        i = np.argmin(np.abs(t_arr - tt))
        a = abs(dpsi_pde_lin[i])
        p = abs(dpsi_pred[i])
        r = abs(dpsi_pde_shift_vs_pred[i])
        rel = r / max(p, 1e-30)
        print(f"{t_arr[i]:5.2f}  {a:16.4e}  {p:14.4e}  {r:20.4e}  {rel:14.4f}")


if __name__ == "__main__":
    main()
