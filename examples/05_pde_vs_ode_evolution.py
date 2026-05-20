"""Time-domain evolutions: linear baseline vs PDE vs amplitude-ODE.

Drops a 2-mode QNM superposition initial data, evolves the full nonlinear
PDE and the amplitude ODE side-by-side, and plots:

Row 1: Re ψ at mid-point vs time  — linear, ODE, PDE all overlap visually
Row 2: Im ψ at mid-point vs time  — same overlap; the field is complex and
       the QNM rotates as it decays
Row 3: |ψ - ψ_lin| at mid-point vs time (log scale) — magnitudes of the
       nonlinear deviation; this avoids the Re/Im phase artifacts that
       made the previous Re-only deviation plot misleading.
Row 4: |ψ(T, r_*)| spatial snapshot at the final time.

Three α columns: 0 (linear floor), 0.3, 1.0.

Grid is N=401 to push the PDE numerical floor down to ~9e-12, well below
the α≥0.3 nonlinear signal.
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
    modes = [{"n": 0, "l": 0, "m": 0}, {"n": 1, "l": 0, "m": 0}]
    omega_arr = build_omega_arr(modes)
    sns = np.array([s(m["n"], m["l"]) for m in modes])
    table = load_S_table()
    eps = 0.05
    A0_init = np.array([1.0, 0.5 * np.exp(0.3j)], dtype=complex)

    # Higher resolution to push the PDE numerical floor to ~9e-12
    grid = UniformGrid(x_min=-5.0, x_max=-0.5, N=401)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    mid = grid.N // 2

    b_grid = np.array([b_near(m["n"], m["l"], xbreve) for m in modes])
    psi0 = eps * (A0_init[:, None] * b_grid).sum(axis=0)
    pi0 = eps * (A0_init[:, None] * sns[:, None] * b_grid).sum(axis=0)

    T = 2.0
    nout = 250
    alpha_cases = [0.0, 0.3, 1.0]

    runs = {}
    for alpha in alpha_cases:
        # ODE
        C = build_C_notes_cached(modes, table, alpha=alpha)
        sol_A, *_ = integrate_A_notes(
            A0_init, T, eps, omega_arr, C,
            nout=nout, remove_diag=False,
            include_frequency_shift=False, phase_mode="real",
        )
        t_arr = sol_A.t
        A_t = sol_A.y.T
        psi_ode = eps * (
            A_t[:, :, None]
            * np.exp(sns[None, :, None] * t_arr[:, None, None])
            * b_grid[None, :, :]
        ).sum(axis=1)

        # PDE
        _, psi_pde, _, sol_pde = evolve_kg(
            grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
            alpha=alpha, nout=nout,
        )
        assert sol_pde.success

        # Linear analytic
        psi_lin = eps * (
            A0_init[None, :, None]
            * np.exp(sns[None, :, None] * t_arr[:, None, None])
            * b_grid[None, :, :]
        ).sum(axis=1)

        runs[alpha] = dict(t=t_arr, psi_pde=psi_pde, psi_ode=psi_ode, psi_lin=psi_lin)

    # -- plot --------------------------------------------------------------
    fig, axes = plt.subplots(4, len(alpha_cases), figsize=(13, 12),
                             constrained_layout=True)
    fig.suptitle(
        rf"PDE vs amplitude-ODE on NHERN — time-domain evolutions"
        f"\n"
        rf"q={q_default}, $\epsilon=${eps}, 2-mode $\ell=0$: "
        rf"$A_0(0)={A0_init[0]:.2f}$, $A_1(0)={A0_init[1]:.2f}$, "
        rf"N={grid.N} grid"
        f"\n"
        rf"PDE numerical floor (α=0) ≈ 9e-12 at mid; α≥0.3 signal lives 3–4 decades above",
        fontsize=11,
    )

    for col, alpha in enumerate(alpha_cases):
        r = runs[alpha]

        # Row 1: Re ψ at mid
        ax = axes[0, col]
        ax.plot(r["t"], np.real(r["psi_lin"][:, mid]), "-", color="0.6",
                lw=1.6, label=r"Re $\psi_{\rm lin}$")
        ax.plot(r["t"], np.real(r["psi_ode"][:, mid]), "--", color="C3",
                lw=1.2, label=r"Re $\psi_{\rm ODE}$")
        ax.plot(r["t"], np.real(r["psi_pde"][:, mid]), ":", color="k",
                lw=1.4, label=r"Re $\psi_{\rm PDE}$")
        ax.set_title(rf"$\alpha = {alpha}$")
        if col == 0:
            ax.set_ylabel(rf"Re $\psi(\check t, r_*={rstar[mid]:.2f})$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

        # Row 2: Im ψ at mid  (NEW — the field is complex; this exposes the
        # imaginary component that the previous plot suppressed)
        ax = axes[1, col]
        ax.plot(r["t"], np.imag(r["psi_lin"][:, mid]), "-", color="0.6",
                lw=1.6, label=r"Im $\psi_{\rm lin}$")
        ax.plot(r["t"], np.imag(r["psi_ode"][:, mid]), "--", color="C3",
                lw=1.2, label=r"Im $\psi_{\rm ODE}$")
        ax.plot(r["t"], np.imag(r["psi_pde"][:, mid]), ":", color="k",
                lw=1.4, label=r"Im $\psi_{\rm PDE}$")
        if col == 0:
            ax.set_ylabel(rf"Im $\psi(\check t, r_*={rstar[mid]:.2f})$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

        # Row 3: |ψ - ψ_lin| at mid, log scale.  Magnitudes only, so the
        # complex-phase artifact (ODE deviation mostly Im, PDE has both Re
        # and Im, which made Re-only plots misleading) goes away.
        ax = axes[2, col]
        dev_pde = np.abs(r["psi_pde"][:, mid] - r["psi_lin"][:, mid])
        dev_ode = np.abs(r["psi_ode"][:, mid] - r["psi_lin"][:, mid])
        resid = np.abs(r["psi_pde"][:, mid] - r["psi_ode"][:, mid])
        ax.semilogy(r["t"], np.maximum(dev_pde, 1e-15), "-", color="k", lw=1.4,
                    label=r"$|\psi_{\rm PDE} - \psi_{\rm lin}|$")
        ax.semilogy(r["t"], np.maximum(dev_ode, 1e-15), "--", color="C3", lw=1.2,
                    label=r"$|\psi_{\rm ODE} - \psi_{\rm lin}|$")
        ax.semilogy(r["t"], np.maximum(resid, 1e-15), ":", color="C2", lw=1.2,
                    label=r"$|\psi_{\rm PDE} - \psi_{\rm ODE}|$")
        ax.axhline(9e-12, color="0.7", lw=0.6, ls="-.",
                   label="PDE floor (α=0)")
        ax.set_ylim(1e-13, 1e-5)
        if col == 0:
            ax.set_ylabel(r"$|\Delta\psi|$ at mid (log)")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25, which="both")

        # Row 4: spatial profile at t = T
        ax = axes[3, col]
        ax.plot(rstar, np.abs(r["psi_lin"][-1]), "-", color="0.6", lw=1.6,
                label=r"$|\psi_{\rm lin}(T)|$")
        ax.plot(rstar, np.abs(r["psi_ode"][-1]), "--", color="C3", lw=1.2,
                label=r"$|\psi_{\rm ODE}(T)|$")
        ax.plot(rstar, np.abs(r["psi_pde"][-1]), ":", color="k", lw=1.4,
                label=r"$|\psi_{\rm PDE}(T)|$")
        ax.set_xlabel(r"$r_*$")
        if col == 0:
            ax.set_ylabel(rf"$|\psi|$ at $\check t = T = {T}$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

    out = "examples/05_pde_vs_ode_evolution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
