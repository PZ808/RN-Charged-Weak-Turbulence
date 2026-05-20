"""Single-mode α-scan: confirms the PDE/ODE temporal-scaling gap is universal.

Same setup as ``06_single_mode_analytic.py`` (single QNM b_0 IC, wide clean
grid, sample at r_*=-2.5) but with a sweep over α ∈ {0.1, 0.3, 1.0, 3.0}.

Two claims to verify:

1. **|Δψ_PDE|/α and |Δψ_ODE|/α both collapse onto a single curve.**
   For the PDE the nonlinear source is α|ψ|²ψ, leading order in α; for the
   ODE the analytic formula is |Δψ_ODE| = ε³|C_{0000}(α)|·t·e^{Re(s_0)t}|b_0|
   with C_{0000} ∝ α. So both signals are exactly linear in α and dividing
   by α makes the curves overlay (within the perturbative regime).

2. **The ratio |Δψ_PDE|/|Δψ_ODE| is independent of α.** Because both sides
   scale as α¹, the α cancels and the time-dependent ratio = const · t at
   early times — universal regime-of-validity feature of the slowly-varying-
   envelope ansatz, not a tuning artifact.

Saves examples/07_single_mode_alpha_scan.png.
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
    eps = 0.05
    A0 = 1.0 + 0.0j
    n, ell = 0, 0
    s0 = s(n, ell)
    T = 2.0
    nout = 300
    alpha_vals = [0.1, 0.3, 1.0, 3.0]

    table = load_S_table()
    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0_grid = b_near(n, ell, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    runs = {}
    for alpha in alpha_vals:
        modes = [{"n": n, "l": ell, "m": 0}]
        C0000 = build_C_notes_cached(modes, table, alpha=alpha)[0, 0, 0, 0]

        psi0 = eps * A0 * b0_grid
        pi0 = eps * A0 * s0 * b0_grid
        t_arr, psi_pde, _, sol = evolve_kg(
            grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
            alpha=alpha, nout=nout,
        )
        assert sol.success

        psi_lin = eps * A0 * np.exp(s0 * t_arr)[:, None] * b0_grid[None, :]

        # Analytic ODE
        A0_t = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0) ** 2 * t_arr)
        dpsi_ode_mid = eps * (A0_t - A0) * np.exp(s0 * t_arr) * b0_grid[mid]
        dpsi_pde_mid = psi_pde[:, mid] - psi_lin[:, mid]

        runs[alpha] = dict(t=t_arr,
                           dpsi_pde=np.abs(dpsi_pde_mid),
                           dpsi_ode=np.abs(dpsi_ode_mid),
                           C=C0000)

    # -- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        rf"Single-mode $b_0$ — $\alpha$-scan, $\epsilon={eps}$, sample $r_*={rstar[mid]:.2f}$",
        fontsize=12,
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alpha_vals)))

    # (0) |Δψ|/α vs t — collapses onto single curve for each method
    ax = axes[0]
    for color, alpha in zip(colors, alpha_vals):
        r = runs[alpha]
        ax.loglog(r["t"][1:], r["dpsi_pde"][1:] / alpha, "-", color=color, lw=1.6,
                  label=rf"$\alpha={alpha}$  PDE")
        ax.loglog(r["t"][1:], r["dpsi_ode"][1:] / alpha, "--", color=color, lw=1.0,
                  alpha=0.7)
    # Reference slopes anchored at the α=1 PDE at t=0.5
    r_ref = runs[1.0]
    anchor_t = 0.5
    anchor_val = np.interp(anchor_t, r_ref["t"], r_ref["dpsi_pde"])
    tref = np.array([5e-2, 2.0])
    ax.loglog(tref, anchor_val * (tref / anchor_t) ** 2, ":", color="k", lw=1.2,
              label=r"$\propto t^2$ (PDE forced transient)")
    anchor_ode = np.interp(anchor_t, r_ref["t"], r_ref["dpsi_ode"])
    ax.loglog(tref, anchor_ode * (tref / anchor_t), ":", color="0.4", lw=1.2,
              label=r"$\propto t$ (ODE slow envelope, before saturation)")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\Delta\psi|\,/\,\alpha$  at mid")
    ax.set_title(r"Magnitudes (solid: PDE, dashed: analytic ODE) — both linear in $\alpha$")
    ax.set_xlim(0.05, T)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(which="both", alpha=0.25)

    # (1) Ratio PDE/ODE vs t — α-independent
    ax = axes[1]
    for color, alpha in zip(colors, alpha_vals):
        r = runs[alpha]
        ratio = r["dpsi_pde"] / np.maximum(r["dpsi_ode"], 1e-30)
        ax.plot(r["t"], ratio, "-", color=color, lw=1.6, label=rf"$\alpha={alpha}$")
    # Linear-in-t fit from the α=1 early regime
    r_ref = runs[1.0]
    ratio_ref = r_ref["dpsi_pde"] / np.maximum(r_ref["dpsi_ode"], 1e-30)
    mask = (r_ref["t"] > 0.1) & (r_ref["t"] < 1.0)
    slope = np.mean(ratio_ref[mask] / r_ref["t"][mask])
    ax.plot(r_ref["t"], slope * r_ref["t"], "--", color="0.4", lw=1.0,
            label=rf"linear fit $\approx {slope:.2f}\,t$  (universal in $\alpha$)")
    ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\Delta\psi_{\rm PDE}|\,/\,|\Delta\psi_{\rm ODE}^{\rm analytic}|$")
    ax.set_title(r"Ratio collapses across $\alpha$ — confirms regime-of-validity, not tuning")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    out = "examples/07_single_mode_alpha_scan.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
