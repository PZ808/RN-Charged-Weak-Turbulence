"""Overlay analytic and numerically-integrated ODE results on top of the
PDE data, for both linear-IC and shifted-IC cases, with Re/Im separated.

For single mode with diagonal Ω=0 the ODE has the closed-form A(t)=exp(−iε²Ct),
so the numerical solve_ivp and the analytic formula should agree to integrator
tolerance. We plot both as a sanity check.

Three curves per panel:
  - Grey solid: PDE numerical result (relative to linear baseline)
  - Red solid:  ODE analytic prediction ε(exp(Δs·t) − 1)·e^{s_0 t}·b_0
  - Red dashed: ODE numerical integration of i dA/dt = ε²·C·|A|²·A,
                reconstructed as ε(A(t)−1)·e^{s_0 t}·b_0
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

    print(f"C       = {C0000:.6e}")
    print(f"-i·ε²·C = {-1j*eps**2*C0000:.6e}")
    print(f"Δs      = {Delta_s:.6e}")
    print(f"Re(Δs) = {Delta_s.real:.4e}  (extra decay rate — negative means faster decay)")
    print(f"Im(Δs) = {Delta_s.imag:.4e}  (phase rotation)")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(0, 0, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    psi0 = eps * A0 * b0
    pi_linear  = eps * A0 * s0 * b0
    pi_shifted = eps * A0 * (s0 + Delta_s) * b0

    # PDE: linear-IC and shifted-IC
    t_arr, psi_pde_lin, _, _ = evolve_kg(
        grid, psi0, pi_linear, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )
    _, psi_pde_shift, _, _ = evolve_kg(
        grid, psi0, pi_shifted, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )

    # Linear baseline (analytic)
    psi_lin = eps * A0 * np.exp(s0 * t_arr) * b0[mid]

    # ODE analytic prediction
    A_analytic = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0)**2 * t_arr)
    psi_ode_analytic = eps * A_analytic * np.exp(s0 * t_arr) * b0[mid]

    # ODE numerical integration  i dA/dt = ε²·C·|A|²·A   → dA/dt = -i·ε²·C·|A|²·A
    def rhs(t, y):
        A = y[0] + 1j * y[1]
        dAdt = -1j * eps**2 * C0000 * abs(A)**2 * A
        return [dAdt.real, dAdt.imag]
    sol_ode = solve_ivp(
        rhs, (0.0, T), [A0.real, A0.imag],
        t_eval=t_arr, method="DOP853", rtol=1e-12, atol=1e-14,
    )
    A_numerical = sol_ode.y[0] + 1j * sol_ode.y[1]
    psi_ode_numerical = eps * A_numerical * np.exp(s0 * t_arr) * b0[mid]

    # Sanity: analytic and numerical ODE should agree
    ode_self_check = np.max(np.abs(A_analytic - A_numerical))
    print(f"\nAnalytic vs numerical ODE max disagreement: {ode_self_check:.3e}  (should be ~10⁻¹²)")

    # ----- plot -----
    psi_pde_lin_mid = psi_pde_lin[:, mid]
    psi_pde_shift_mid = psi_pde_shift[:, mid]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        rf"Numerical-ODE overlay — single-mode $b_0$, $\epsilon={eps}$, $\alpha={alpha}$,  "
        rf"$\Delta s = {Delta_s.real:.2e} + {Delta_s.imag:.2e}\,i$"
        f"\n"
        rf"Top row: PDE with linear IC.  Bottom row: PDE with shifted IC $\pi(0)=\epsilon(s_0+\Delta s)b_0$.",
        fontsize=10,
    )

    # Row 1: linear-IC PDE
    for col, (component, name) in enumerate([("real", "Re"), ("imag", "Im")]):
        ax = axes[0, col]
        op = np.real if component == "real" else np.imag
        ax.plot(t_arr, op(psi_pde_lin_mid - psi_lin), "-", color="0.4", lw=1.6,
                label=rf"{name} (PDE_lin − $\psi_{{\rm lin}}$)")
        ax.plot(t_arr, op(psi_ode_analytic - psi_lin), "-", color="C3", lw=1.8,
                label=rf"{name} ($\psi_{{\rm ODE,\,analytic}} - \psi_{{\rm lin}}$)")
        ax.plot(t_arr, op(psi_ode_numerical - psi_lin), "--", color="C1", lw=1.0,
                label=rf"{name} ($\psi_{{\rm ODE,\,numerical}} - \psi_{{\rm lin}}$)")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel(r"$\check t$")
        ax.set_ylabel(rf"{name} $(\psi - \psi_{{\rm lin}})$")
        ax.set_title(f"{name} part — PDE with **linear** IC vs ODE prediction")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    # Row 2: shifted-IC PDE
    for col, (component, name) in enumerate([("real", "Re"), ("imag", "Im")]):
        ax = axes[1, col]
        op = np.real if component == "real" else np.imag
        ax.plot(t_arr, op(psi_pde_shift_mid - psi_lin), "-", color="0.4", lw=1.6,
                label=rf"{name} (PDE_shift − $\psi_{{\rm lin}}$)")
        ax.plot(t_arr, op(psi_ode_analytic - psi_lin), "-", color="C3", lw=1.8,
                label=rf"{name} ($\psi_{{\rm ODE,\,analytic}} - \psi_{{\rm lin}}$)")
        ax.plot(t_arr, op(psi_ode_numerical - psi_lin), "--", color="C1", lw=1.0,
                label=rf"{name} ($\psi_{{\rm ODE,\,numerical}} - \psi_{{\rm lin}}$)")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel(r"$\check t$")
        ax.set_ylabel(rf"{name} $(\psi - \psi_{{\rm lin}})$")
        ax.set_title(f"{name} part — PDE with **shifted** IC vs ODE prediction")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    out = "examples/14_overlay_numerical_ode.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")

    # Snapshot
    print()
    print(f"At t=2.0:")
    i = -1
    pde_lin_val = psi_pde_lin_mid[i] - psi_lin[i]
    pde_shift_val = psi_pde_shift_mid[i] - psi_lin[i]
    ode_val = psi_ode_analytic[i] - psi_lin[i]
    print(f"  PDE_lin   − lin:  {pde_lin_val.real:+.3e}  +  {pde_lin_val.imag:+.3e} i")
    print(f"  ODE pred  − lin:  {ode_val.real:+.3e}  +  {ode_val.imag:+.3e} i")
    print(f"  PDE_shift − lin:  {pde_shift_val.real:+.3e}  +  {pde_shift_val.imag:+.3e} i")


if __name__ == "__main__":
    main()
