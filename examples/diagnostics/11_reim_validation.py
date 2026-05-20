"""Re/Im of Δψ_PDE vs Δψ_ODE — the actual validation plot.

Single-mode b_0 IC, ε=0.1 (a bit bigger than the usual 0.05 so the nonlinear
correction is 8× larger and easier to see), α=1, sample r_*=-2.5. The
nonlinear effect is the slight frequency shift Δs = -i·ε²·C ≈ -ε²·9.6e-4·i ⇒
mostly a small real-negative shift (extra decay rate ~2.4e-6 per unit t for
ε=0.05, ~1e-5 for ε=0.1) plus a tiny imaginary phase shift.

Three rows × two cols:
Row 1  Re ψ_PDE(t, mid)  vs  Re ψ_lin(t, mid)  — the actual field with the
       linear baseline. Visually almost identical for ε small (the nonlinear
       correction is ε³, the field is ε); the *difference* is in row 3.
Row 2  Im ψ_PDE  vs  Im ψ_lin  — same.
Row 3  Re/Im of Δψ_PDE and Δψ_ODE, overlaid — this is where you actually see
       the nonlinear effect and whether the ODE matches.
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
    eps = 0.1
    alpha = 1.0
    A0 = 1.0 + 0.0j
    s0 = s(0, 0)
    T = 2.0
    nout = 400

    table = load_S_table()
    C0000 = build_C_notes_cached(
        [{"n": 0, "l": 0, "m": 0}], table, alpha=alpha
    )[0, 0, 0, 0]
    print(f"C[0,0,0,0] = {C0000:.4e}")
    print(f"ε² C = {eps**2 * C0000:.4e}  ⇒  Δs = -i·ε²·C = {-1j*eps**2*C0000:.4e}")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(0, 0, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    psi0 = eps * A0 * b0
    pi0 = eps * A0 * s0 * b0
    t_arr, psi_pde, _, sol = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )
    assert sol.success

    psi_lin_mid = eps * A0 * np.exp(s0 * t_arr) * b0[mid]
    psi_pde_mid = psi_pde[:, mid]
    # Analytic ODE A(t)
    A_t = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0) ** 2 * t_arr)
    psi_ode_mid = eps * A_t * np.exp(s0 * t_arr) * b0[mid]

    dpsi_pde = psi_pde_mid - psi_lin_mid
    dpsi_ode = psi_ode_mid - psi_lin_mid

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(
        rf"Re/Im PDE vs ODE — single-mode $b_0$, $\epsilon={eps}$, $\alpha={alpha}$,"
        rf" $r_*={rstar[mid]:.2f}$"
        f"\n"
        rf"Field $|\psi_{{\rm lin}}|\sim\epsilon\cdot|b_0|\sim 4\times 10^{{-2}}$; "
        rf"nonlinear correction $|\Delta\psi|\sim\epsilon^3\sim 3\times 10^{{-7}}$",
        fontsize=11,
    )

    # Row 1: Re ψ — visually nearly identical because |Δψ| ≪ |ψ|
    ax = axes[0, 0]
    ax.plot(t_arr, np.real(psi_lin_mid), "-", color="0.5", lw=1.4,
            label=r"Re $\psi_{\rm lin}$")
    ax.plot(t_arr, np.real(psi_pde_mid), "--", color="k", lw=1.0,
            label=r"Re $\psi_{\rm PDE}$")
    ax.plot(t_arr, np.real(psi_ode_mid), ":", color="C3", lw=1.2,
            label=r"Re $\psi_{\rm ODE}$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re $\psi(\check t, r_*)$")
    ax.set_title("Re of the field — the nonlinear effect is invisible at this scale")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(t_arr, np.imag(psi_lin_mid), "-", color="0.5", lw=1.4,
            label=r"Im $\psi_{\rm lin}$")
    ax.plot(t_arr, np.imag(psi_pde_mid), "--", color="k", lw=1.0,
            label=r"Im $\psi_{\rm PDE}$")
    ax.plot(t_arr, np.imag(psi_ode_mid), ":", color="C3", lw=1.2,
            label=r"Im $\psi_{\rm ODE}$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im $\psi(\check t, r_*)$")
    ax.set_title("Im of the field")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Row 2: Re Δψ — PDE and ODE compared, plus residual
    ax = axes[1, 0]
    ax.plot(t_arr, np.real(dpsi_pde), "-", color="k", lw=1.5,
            label=r"Re $\Delta\psi_{\rm PDE}$")
    ax.plot(t_arr, np.real(dpsi_ode), "--", color="C3", lw=1.3,
            label=r"Re $\Delta\psi_{\rm ODE}$")
    ax.plot(t_arr, np.real(dpsi_pde - dpsi_ode), ":", color="C2", lw=1.2,
            label=r"Re $(\Delta\psi_{\rm PDE} - \Delta\psi_{\rm ODE})$  (residual)")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re $\Delta\psi$")
    ax.set_title("Real part of the nonlinear correction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(t_arr, np.imag(dpsi_pde), "-", color="k", lw=1.5,
            label=r"Im $\Delta\psi_{\rm PDE}$")
    ax.plot(t_arr, np.imag(dpsi_ode), "--", color="C3", lw=1.3,
            label=r"Im $\Delta\psi_{\rm ODE}$")
    ax.plot(t_arr, np.imag(dpsi_pde - dpsi_ode), ":", color="C2", lw=1.2,
            label=r"Im $(\Delta\psi_{\rm PDE} - \Delta\psi_{\rm ODE})$  (residual)")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im $\Delta\psi$")
    ax.set_title("Imaginary part of the nonlinear correction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Row 3: zoom on Re/Im Δψ with phase angle reading
    ax = axes[2, 0]
    # Show ε³ scale to make clear these are O(ε³)
    eps3 = eps**3
    ax.plot(t_arr, np.real(dpsi_pde)/eps3, "-", color="k", lw=1.5, label=r"Re $\Delta\psi_{\rm PDE}/\epsilon^3$")
    ax.plot(t_arr, np.real(dpsi_ode)/eps3, "--", color="C3", lw=1.3, label=r"Re $\Delta\psi_{\rm ODE}/\epsilon^3$")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(rf"Re $\Delta\psi/\epsilon^3$")
    ax.set_title(r"Re $\Delta\psi$ rescaled by $\epsilon^3$ — ODE captures the leading nonlinearity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[2, 1]
    ax.plot(t_arr, np.imag(dpsi_pde)/eps3, "-", color="k", lw=1.5, label=r"Im $\Delta\psi_{\rm PDE}/\epsilon^3$")
    ax.plot(t_arr, np.imag(dpsi_ode)/eps3, "--", color="C3", lw=1.3, label=r"Im $\Delta\psi_{\rm ODE}/\epsilon^3$")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(rf"Im $\Delta\psi/\epsilon^3$")
    ax.set_title(r"Im $\Delta\psi$ rescaled by $\epsilon^3$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    out = "examples/11_reim_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")

    # Print snapshot
    print()
    print(f"{'t':>5}  {'Re Δψ_PDE':>12}  {'Re Δψ_ODE':>12}  {'Im Δψ_PDE':>12}  {'Im Δψ_ODE':>12}")
    for tt in [0.25, 0.5, 1.0, 1.5, 2.0]:
        i = np.argmin(np.abs(t_arr - tt))
        print(f"{t_arr[i]:5.2f}  {dpsi_pde[i].real:12.3e}  {dpsi_ode[i].real:12.3e}  "
              f"{dpsi_pde[i].imag:12.3e}  {dpsi_ode[i].imag:12.3e}")


if __name__ == "__main__":
    main()
