"""Single-mode PDE vs analytic-ODE for the charged-scalar nonlinearity.

Single QNM initial data ψ(0) = ε b_0 on NHERN. With only one mode in the
truncation and all-ℓ=0 frequencies sharing the same real part q/2, the
detuning Ω vanishes identically and the amplitude ODE

    i Ȧ_0 = ε² C_{0000} |A_0|² A_0

is exactly solvable: A_0(t) = A_0(0) exp(−i ε² C_{0000} |A_0(0)|² t). For
our parameters |ε² C_{0000}| ≈ 2.4×10⁻⁶, so over t ∈ [0, 2] the linearised
form

    ΔA_0(t) ≈ −i ε² C_{0000} A_0(0) t

is accurate to ~10⁻¹⁰. The amplitude-ODE prediction for the field deviation
from the linear evolution is therefore a one-line formula

    |Δψ_ODE(t, r_*)| = ε³ |C_{0000}| · t · e^{Re(s_0) t} · |b_0(r_*)|

which we overlay on the full PDE deviation. The contrast is striking: the
PDE goes as t², the ODE as t·e^{Re(s_0) t}, so the ratio grows like t until
the QNM decay envelope kicks the ODE down at t ≈ 1/|Re(s_0)| = 2.

This is **not** a bug — it is the regime-of-validity of the slowly-varying-
envelope ansatz. The amplitude ODE is asymptotic for t·|Re(s_0)| ≫ 1; over
the integration window allowed by our finite domain, we are not yet in that
regime, so the early-time t² forced response in the PDE dominates the gap.

Saves examples/06_single_mode_analytic.png.
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
    alpha = 1.0
    A0 = 1.0 + 0.0j
    n, ell = 0, 0
    s0 = s(n, ell)

    # C_{0000} from Peter's S-table (paper / Mathematica convention)
    table = load_S_table()
    modes = [{"n": n, "l": ell, "m": 0}]
    C_arr = build_C_notes_cached(modes, table, alpha=alpha)
    C0000 = C_arr[0, 0, 0, 0]
    print(f"C[0,0,0,0] = {C0000:.6e}    |C| = {abs(C0000):.4e}")
    print(f"ε² C = {eps**2 * C0000:.4e}    Re(s_0) = {s0.real:.4f}")

    # Wide clean domain so the throat reflection arrives at t ≈ 2.45.
    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0_grid = b_near(n, ell, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    T = 2.0
    nout = 400

    # Single-mode QNM initial data
    psi0 = eps * A0 * b0_grid
    pi0 = eps * A0 * s0 * b0_grid
    t_pde, psi_pde, _, sol_pde = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )
    assert sol_pde.success

    # Linear analytic ψ(t, r_*) = ε A_0(0) e^{s_0 t} b_0(r_*)
    t_arr = t_pde
    psi_lin = eps * A0 * np.exp(s0 * t_arr)[:, None] * b0_grid[None, :]

    # Analytic ODE amplitude (RWA-exact: |A_0|=1, so equation has constant rate)
    A0_t = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0) ** 2 * t_arr)
    dA = A0_t - A0
    dpsi_ode = eps * dA[:, None] * np.exp(s0 * t_arr)[:, None] * b0_grid[None, :]

    # Diagnostics at mid
    dpsi_pde_mid = psi_pde[:, mid] - psi_lin[:, mid]
    dpsi_ode_mid = dpsi_ode[:, mid]

    # -- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        rf"Single mode $b_0$, analytic ODE vs PDE on NHERN"
        f"\n"
        rf"$q={q_default}$, $\epsilon={eps}$, $\alpha={alpha}$, $A_0(0)=1$, "
        rf"sample $r_* = {rstar[mid]:.2f}$"
        f"\n"
        rf"$C_{{0000}} = {C0000.real:.2e} + ({C0000.imag:.2e})i$,"
        rf" $|\epsilon^2 C| t \lesssim {abs(eps**2 * C0000 * T):.1e}$ over the run "
        rf"$\Rightarrow$ linear-in-$t$ ODE response exact to ~$10^{{-10}}$",
        fontsize=11,
    )

    # (0,0) log–log magnitude of deviation vs t at mid: PDE numerical, ODE analytic
    ax = axes[0, 0]
    ax.loglog(t_arr[1:], np.abs(dpsi_pde_mid[1:]), "-", color="k", lw=1.8,
              label=r"$|\Delta\psi_{\rm PDE}|$  (numerical)")
    ax.loglog(t_arr[1:], np.abs(dpsi_ode_mid[1:]), "--", color="C3", lw=1.6,
              label=r"$|\Delta\psi_{\rm ODE}^{\rm analytic}|$  $= \epsilon^3|C_{0000}|\,t\,e^{\mathrm{Re}\,s_0 t}|b_0|$")
    # slope references anchored at t=0.5
    tref = np.array([5e-2, 2.0])
    anchor_t, anchor_pde = 0.5, np.interp(0.5, t_arr, np.abs(dpsi_pde_mid))
    anchor_ode = np.interp(0.5, t_arr, np.abs(dpsi_ode_mid))
    ax.loglog(tref, anchor_pde * (tref / anchor_t) ** 2, ":", color="0.5", lw=1.0,
              label=r"$\propto t^2$ (forced-response transient)")
    ax.loglog(tref, anchor_ode * (tref / anchor_t), ":", color="C3", alpha=0.5, lw=1.0,
              label=r"$\propto t$ (slow-envelope drift)")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\Delta\psi(\check t, r_{*,{\rm mid}})|$")
    ax.set_title("Time-domain deviation at mid (log–log)")
    ax.set_xlim(t_arr[1], T * 1.05)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(which="both", alpha=0.25)

    # (0,1) Ratio PDE/ODE vs t (linear scale) — exposes the linear-in-t gap
    ax = axes[0, 1]
    ratio = np.abs(dpsi_pde_mid) / np.maximum(np.abs(dpsi_ode_mid), 1e-30)
    ax.plot(t_arr, ratio, "-", color="C0", lw=1.6,
            label=r"$|\Delta\psi_{\rm PDE}| / |\Delta\psi_{\rm ODE}^{\rm analytic}|$")
    # linear-in-t fit anchored at the early regime t < 1
    mask = (t_arr > 0.1) & (t_arr < 1.0)
    if mask.any():
        slope = np.mean(ratio[mask] / t_arr[mask])
        ax.plot(t_arr, slope * t_arr, "--", color="0.5", lw=1.0,
                label=rf"linear fit $\approx {slope:.2f}\,t$  (early-time regime)")
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"ratio")
    ax.set_title("PDE/ODE ratio  (= 1 at crossover)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    # Clean window: throat reflection at r_*=-0.05 reaches r_* = r_*,max + T
    # by t=T, so restrict the spatial plots to r_* < r_*,max - T.
    clean_mask = rstar < (rstar[-1] - T)
    rstar_clean = rstar[clean_mask]

    # (1,0) Spatial profile at t = T in the clean window:
    # PDE deviation vs ODE-analytic prediction.  The ODE prediction is by
    # construction proportional to |b_0(r_*)|; if the shapes match in the
    # clean window, the gap is purely a temporal-scaling effect (PDE has
    # a t² transient, ODE only the t·e^{Re(s)t} slow drift).
    ax = axes[1, 0]
    pde_T_clean = np.abs((psi_pde[-1] - psi_lin[-1])[clean_mask])
    ode_T_clean = np.abs(dpsi_ode[-1][clean_mask])
    ax.semilogy(rstar_clean, pde_T_clean, "-", color="k", lw=1.6,
                label=r"$|\Delta\psi_{\rm PDE}(T)|$")
    ax.semilogy(rstar_clean, ode_T_clean, "--", color="C3", lw=1.4,
                label=r"$|\Delta\psi_{\rm ODE}^{\rm analytic}(T)|\,\propto\,|b_0|$")
    ax.axvline(rstar[mid], color="0.7", lw=0.6, ls=":",
               label=rf"sample at $r_* = {rstar[mid]:.2f}$")
    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel(r"$|\Delta\psi(T, r_*)|$")
    ax.set_title(rf"Spatial profile at $\check t = T = {T}$  (BC-clean window)")
    ax.legend(fontsize=9)
    ax.grid(which="both", alpha=0.25)

    # (1,1) Non-modal residual at t = T in the clean window:
    # ψ_PDE − (linear + ODE-analytic Δψ).  This is the part of the PDE
    # response that the QNM-projected slow-envelope picture can't capture.
    # The ratio |residual|/|ψ_PDE − ψ_lin| tells us how much of the gap is
    # genuinely non-modal vs how much is just the t² envelope.
    ax = axes[1, 1]
    pde_dev_T = psi_pde[-1] - psi_lin[-1]
    ode_dev_T = dpsi_ode[-1]
    non_modal = pde_dev_T - ode_dev_T
    ax.semilogy(rstar_clean, np.abs(pde_dev_T[clean_mask]), "-", color="k", lw=1.6,
                label=r"$|\Delta\psi_{\rm PDE}|$")
    ax.semilogy(rstar_clean, np.abs(non_modal[clean_mask]), "-", color="C2", lw=1.4,
                label=r"$|\Delta\psi_{\rm PDE} - \Delta\psi_{\rm ODE}^{\rm analytic}|$  (non-modal)")
    ax.semilogy(rstar_clean, np.abs(ode_dev_T[clean_mask]), "--", color="C3", lw=1.2,
                label=r"$|\Delta\psi_{\rm ODE}^{\rm analytic}|$  (QNM-projected)")
    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel(r"$|\cdot|$ at $\check t = T$")
    ax.set_title("PDE deviation: QNM-projected + non-modal pieces")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(which="both", alpha=0.25)

    out = "examples/06_single_mode_analytic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
