"""ε-scaling test: does the PDE/ODE gap shrink as the next-order correction?

If the gap between the full PDE and the leading-order amplitude ODE is the
ε²·ψ₂ next-order term in the multiple-scales expansion, then for single-mode
b_0 IC:

    |Δψ_PDE|       ~ ε³  (leading)
    |Δψ_ODE|       ~ ε³  (leading)
    |Δψ_PDE − Δψ_ODE|  ~ ε⁵  (next order)

so the relative gap |PDE − ODE|/|ODE| ~ ε², and the ratio |PDE|/|ODE| should
collapse to 1 as ε → 0 across the whole time window — including the short-
time t² regime that previously looked like a "discrepancy" between PDE and
ODE.

Scan ε ∈ {0.01, 0.02, 0.05, 0.1} on the wide clean NHERN grid at α=1, sample
at r_* = −2.5.

Two panels:
  (left)  |Δψ_PDE|/|Δψ_ODE_analytic| vs t for each ε. Curves should
          converge to 1 across the whole window as ε decreases.
  (right) At a fixed time slice, plot the relative gap
          |Δψ_PDE − Δψ_ODE|/|Δψ_ODE| vs ε on log-log.  Expect slope 2.

Saves examples/08_epsilon_scaling.png.
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
    alpha = 1.0
    A0 = 1.0 + 0.0j
    n, ell = 0, 0
    s0 = s(n, ell)
    T = 2.0
    nout = 200
    eps_vals = [0.01, 0.02, 0.05, 0.1]
    t_slice = 1.5  # fixed time for the right-panel log-log scaling

    table = load_S_table()
    C0000 = build_C_notes_cached(
        [{"n": n, "l": ell, "m": 0}], table, alpha=alpha
    )[0, 0, 0, 0]
    print(f"C[0,0,0,0] = {C0000:.4e}")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(n, ell, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    runs = {}
    for eps in eps_vals:
        psi0 = eps * A0 * b0
        pi0 = eps * A0 * s0 * b0
        t_arr, psi_pde, _, sol = evolve_kg(
            grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
            alpha=alpha, nout=nout,
        )
        assert sol.success

        # Linear analytic
        psi_lin = eps * A0 * np.exp(s0 * t_arr)[:, None] * b0[None, :]
        # Closed-form amplitude ODE: A(t) = A(0) exp(-i ε² C |A(0)|² t)
        A_t = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0) ** 2 * t_arr)
        dpsi_ode_mid = eps * (A_t - A0) * np.exp(s0 * t_arr) * b0[mid]
        dpsi_pde_mid = psi_pde[:, mid] - psi_lin[:, mid]

        runs[eps] = dict(
            t=t_arr,
            dpsi_pde=dpsi_pde_mid,
            dpsi_ode=dpsi_ode_mid,
        )
        # Print snapshot
        i = np.argmin(np.abs(t_arr - t_slice))
        gap = dpsi_pde_mid[i] - dpsi_ode_mid[i]
        print(f"  ε={eps:5.3f}  |Δψ_PDE|={abs(dpsi_pde_mid[i]):.3e}  "
              f"|Δψ_ODE|={abs(dpsi_ode_mid[i]):.3e}  "
              f"|gap|={abs(gap):.3e}  |gap|/|ODE|={abs(gap)/abs(dpsi_ode_mid[i]):.3e}")

    # -- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    fig.suptitle(
        rf"$\epsilon$-scaling: the PDE/ODE gap is at the SAME order as the leading signal"
        f"\n"
        rf"Single mode $b_0$, $\alpha={alpha}$, $q={q_default}$, sample $r_* = {rstar[mid]:.2f}$"
        f"\n"
        rf"Gap doesn't vanish as $\epsilon\to 0$  $\Rightarrow$  it's the non-QNM content the slow-amp ODE structurally misses",
        fontsize=10.5,
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(eps_vals)))

    # (0) Time-domain ratio for each ε
    ax = axes[0]
    for color, eps in zip(colors, eps_vals):
        r = runs[eps]
        ratio = np.abs(r["dpsi_pde"]) / np.maximum(np.abs(r["dpsi_ode"]), 1e-30)
        ax.plot(r["t"], ratio, "-", color=color, lw=1.6, label=rf"$\epsilon = {eps}$")
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\Delta\psi_{\rm PDE}| / |\Delta\psi_{\rm ODE}^{\rm analytic}|$")
    ax.set_title(r"All $\epsilon$ curves overlay — ratio is $\epsilon$-independent")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T)

    # (1) Relative gap |PDE − ODE| / |ODE| vs ε at fixed t — expect slope 2
    ax = axes[1]
    eps_arr = np.array(eps_vals)
    gap_rel = []
    for eps in eps_vals:
        r = runs[eps]
        i = np.argmin(np.abs(r["t"] - t_slice))
        gap = r["dpsi_pde"][i] - r["dpsi_ode"][i]
        gap_rel.append(abs(gap) / abs(r["dpsi_ode"][i]))
    gap_rel = np.array(gap_rel)
    ax.loglog(eps_arr, gap_rel, "o-", color="C0", ms=8, lw=1.6,
              label=rf"$|\Delta\psi_{{\rm PDE}} - \Delta\psi_{{\rm ODE}}|/|\Delta\psi_{{\rm ODE}}|$"
                    rf"  at $\check t = {t_slice}$")
    # slope-2 reference
    eps_ref = np.array([eps_arr.min(), eps_arr.max()])
    anchor = gap_rel[len(eps_vals)//2]
    eps_anchor = eps_arr[len(eps_vals)//2]
    ax.loglog(eps_ref, anchor * (eps_ref / eps_anchor) ** 2, "--", color="0.5", lw=1.0,
              label=r"$\propto \epsilon^2$  (next-order correction)")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"relative gap at $\check t = $" + f"{t_slice}")
    ax.set_title(r"Relative gap FLAT in $\epsilon$  $\Rightarrow$  gap is order $\epsilon^3$, same as the leading signal")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(which="both", alpha=0.25)

    out = "examples/08_epsilon_scaling.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
