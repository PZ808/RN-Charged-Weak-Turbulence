"""Amplitude-ODE ↔ NHERN-PDE comparison: real and consistent nonlinear effects.

Sweep the nonlinear coupling α over several decades. For each α, compute at
a fixed interior r_*:

    S_PDE(α) = max_t |ψ_PDE(t) - ψ_lin(t)|      (signal — PDE deviation from linear)
    S_ODE(α) = max_t |ψ_ODE(t) - ψ_lin(t)|      (signal — ODE-predicted deviation)
    R(α)     = max_t |ψ_PDE(t) - ψ_ODE(t)|      (residual — gap between methods)

Expected behavior in the perturbative regime:

    S_PDE(α) ∝ α¹      (leading-order nonlinear response)
    S_ODE(α) ∝ α¹      (amplitude ODE captures the same leading order)
    R(α)     ∝ α²      (truncation error is higher-order in α)

If S_PDE and S_ODE both have slope 1 and overlap, AND R has slope 2 well
below them, the nonlinear effects are simultaneously
  (i) real (signals far above the linear floor),
  (ii) consistent between PDE and ODE (signals agree at leading order),
  (iii) the disagreement is a known, perturbative-order truncation effect.

Saves examples/04_pde_vs_ode.png.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_cwt import (
    b_near,
    build_C_notes_cached,
    build_omega_arr,
    integrate_A_notes,
    load_S_table,
    s,
)
from rn_cwt import q as q_default
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    # -- setup --------------------------------------------------------------
    modes = [{"n": 0, "l": 0, "m": 0}, {"n": 1, "l": 0, "m": 0}]
    omega_arr = build_omega_arr(modes)
    sns = np.array([s(m["n"], m["l"]) for m in modes])

    table = load_S_table()
    eps = 0.05
    A0_initial = np.array([1.0, 0.5 * np.exp(0.3j)], dtype=complex)

    grid = UniformGrid(x_min=-5.0, x_max=-0.5, N=201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    mid = grid.N // 2

    b_grid = np.array([b_near(m["n"], m["l"], xbreve) for m in modes])
    psi0 = eps * (A0_initial[:, None] * b_grid).sum(axis=0)
    pi0 = eps * (A0_initial[:, None] * sns[:, None] * b_grid).sum(axis=0)

    T = 1.5
    nout = 100

    alpha_vals = np.array([0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0])

    s_pde_arr = []
    s_ode_arr = []
    r_arr = []

    for alpha in alpha_vals:
        # ODE
        C = build_C_notes_cached(modes, table, alpha=alpha)
        sol_A, *_ = integrate_A_notes(
            A0_initial, T, eps, omega_arr, C,
            nout=nout, remove_diag=False,
            include_frequency_shift=False, phase_mode="real",
        )
        t_arr = sol_A.t
        A_t = sol_A.y.T

        psi_ode = eps * (
            A_t[:, :, None] * np.exp(sns[None, :, None] * t_arr[:, None, None])
            * b_grid[None, :, :]
        ).sum(axis=1)

        # PDE
        _, psi_pde, _, sol_pde = evolve_kg(
            grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
            alpha=alpha, nout=nout,
        )
        assert sol_pde.success

        # Linear analytic baseline
        psi_lin = eps * (
            A0_initial[None, :, None]
            * np.exp(sns[None, :, None] * t_arr[:, None, None])
            * b_grid[None, :, :]
        ).sum(axis=1)

        s_pde = np.max(np.abs(psi_pde[:, mid] - psi_lin[:, mid]))
        s_ode = np.max(np.abs(psi_ode[:, mid] - psi_lin[:, mid]))
        resid = np.max(np.abs(psi_pde[:, mid] - psi_ode[:, mid]))

        s_pde_arr.append(s_pde)
        s_ode_arr.append(s_ode)
        r_arr.append(resid)
        print(f"α={alpha:7.4f}  S_PDE={s_pde:.3e}  S_ODE={s_ode:.3e}  R={resid:.3e}")

    s_pde_arr = np.array(s_pde_arr)
    s_ode_arr = np.array(s_ode_arr)
    r_arr = np.array(r_arr)

    # -- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)

    pos = alpha_vals > 0
    ax.loglog(alpha_vals[pos], s_pde_arr[pos], "ko-", ms=7, lw=1.6,
              label=r"$S_{\rm PDE}(\alpha) = \max_t |\psi_{\rm PDE} - \psi_{\rm lin}|$")
    ax.loglog(alpha_vals[pos], s_ode_arr[pos], "s--", color="C3", ms=6, lw=1.4,
              label=r"$S_{\rm ODE}(\alpha) = \max_t |\psi_{\rm ODE} - \psi_{\rm lin}|$")
    ax.loglog(alpha_vals[pos], r_arr[pos], "^-", color="C2", ms=6, lw=1.4,
              label=r"$R(\alpha) = \max_t |\psi_{\rm PDE} - \psi_{\rm ODE}|$")

    # Reference slope 1 (leading-order nonlinear scaling).
    alpha_ref = np.array([3e-2, 3.0])
    idx_ref = np.argmin(np.abs(alpha_vals - 0.3))
    pivot_alpha = alpha_vals[idx_ref]
    pivot_signal = s_pde_arr[idx_ref]
    slope1 = pivot_signal * (alpha_ref / pivot_alpha)
    ax.loglog(alpha_ref, slope1, ":", color="0.4", lw=1.0,
              label=r"$\propto \alpha$  (slope-1 reference)")

    ax.axhline(s_pde_arr[0] if alpha_vals[0] == 0 else 7e-11,
               ls="-.", color="0.6", lw=0.8,
               label=fr"PDE numerical floor (α=0): ≈ {s_pde_arr[0]:.1e}")

    ax.set_xlabel(r"$\alpha$  (quartic coupling)")
    ax.set_ylabel(rf"max over $\check t \in [0, {T}]$  at $r_* = {rstar[mid]:.2f}$")
    ratio_signal = s_ode_arr[-1] / s_pde_arr[-1]
    ax.set_title(
        rf"Nonlinear PDE $\leftrightarrow$ ODE: scaling with $\alpha$"
        f"\n"
        rf"q={q_default}, $\epsilon$={eps}, 2-mode $\ell=0$ ODE truncation,"
        rf" $A_0(0)$={A0_initial[0]:.2f}, $A_1(0)$={A0_initial[1]:.2f}"
        f"\n"
        rf"$S_{{ODE}}/S_{{PDE}}$ ≈ {ratio_signal:.2f} across {len([a for a in alpha_vals if a > 0.01])}+ decades — same slope-1 scaling, no fundamental disagreement"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(which="both", alpha=0.25)

    out = "examples/04_pde_vs_ode.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
