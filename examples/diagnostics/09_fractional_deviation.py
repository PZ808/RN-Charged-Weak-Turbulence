"""Complex fractional deviation (Δψ_PDE − Δψ_ODE)/Δψ_ODE vs t.

Plus a sanity check: does switching the S_near convention from the paper/Mma
form to the original Python notebook form change the gap?

The two conventions differ by an overall j-independent factor
    K_conv = -1 / Γ(-2 i q - γ_abcd)
applied to S_near (per CLAUDE.md / radial.py docstring). For (0,0,0,0) with
ℓ=0, γ = 3 s_0 + bar(s_0) ≈ -2 − 0.02 i, so K_conv ≈ -1 − O(10⁻²) — close to
a sign flip. If the existing paper convention had been the wrong one, using
the other convention would make the gap larger (≈ ratio 1.85 instead of
0.84). If both conventions give similar gaps, the gap isn't a convention
artefact.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn

from rn_cwt import b_near, build_C_notes_cached, load_S_table, s
from rn_cwt import q as q_default
from rn_cwt.background import iq
from rn_cwt.radial import gamma_abcd
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    eps = 0.05
    alpha = 1.0
    A0 = 1.0 + 0.0j
    n_mode, ell = 0, 0
    s0 = s(n_mode, ell)
    T = 2.0
    nout = 400

    table = load_S_table()
    modes = [{"n": n_mode, "l": ell, "m": 0}]
    C_paper = build_C_notes_cached(modes, table, alpha=alpha)[0, 0, 0, 0]

    # Original-Python-notebook convention: multiply S (hence C) by K_conv
    gam = gamma_abcd(modes[0], modes[0], modes[0], modes[0])
    K_conv = -1.0 / gamma_fn(-2 * iq - gam)
    C_orig = C_paper * K_conv
    print(f"γ_abcd(0,0,0,0) = {gam:.4e}")
    print(f"K_conv = -1/Γ(-2iq-γ) = {K_conv:.4e}")
    print(f"C_paper = {C_paper:.4e}")
    print(f"C_orig  = {C_orig:.4e}")
    print(f"|C_paper|/|C_orig| = {abs(C_paper)/abs(C_orig):.4f}")
    print()

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0_grid = b_near(n_mode, ell, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    psi0 = eps * A0 * b0_grid
    pi0 = eps * A0 * s0 * b0_grid
    t_arr, psi_pde, _, sol_pde = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )
    assert sol_pde.success

    psi_lin = eps * A0 * np.exp(s0 * t_arr)[:, None] * b0_grid[None, :]
    dpsi_pde = psi_pde[:, mid] - psi_lin[:, mid]

    # Analytic ODE with each convention
    A_paper = A0 * np.exp(-1j * eps**2 * C_paper * abs(A0) ** 2 * t_arr)
    A_orig = A0 * np.exp(-1j * eps**2 * C_orig * abs(A0) ** 2 * t_arr)
    dpsi_ode_paper = eps * (A_paper - A0) * np.exp(s0 * t_arr) * b0_grid[mid]
    dpsi_ode_orig = eps * (A_orig - A0) * np.exp(s0 * t_arr) * b0_grid[mid]

    # Complex fractional deviation K(t) = (PDE − ODE)/ODE
    eps_floor = 1e-30
    K_paper = (dpsi_pde - dpsi_ode_paper) / (dpsi_ode_paper + eps_floor)
    K_orig = (dpsi_pde - dpsi_ode_orig) / (dpsi_ode_orig + eps_floor)

    # -- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        rf"Complex fractional deviation $K(t) = (\Delta\psi_{{\rm PDE}} - \Delta\psi_{{\rm ODE}})/\Delta\psi_{{\rm ODE}}$"
        f"\n"
        rf"Single mode $b_0$, $\epsilon={eps}$, $\alpha={alpha}$, sample $r_* = {rstar[mid]:.2f}$",
        fontsize=11,
    )

    # (0,0) Re K(t)
    ax = axes[0, 0]
    ax.plot(t_arr, np.real(K_paper), "-", color="k", lw=1.6, label="paper convention")
    ax.plot(t_arr, np.real(K_orig), "--", color="C3", lw=1.4,
            label="original Python notebook convention")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re $K(t)$")
    ax.set_title("Real part of fractional deviation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # (0,1) Im K(t)
    ax = axes[0, 1]
    ax.plot(t_arr, np.imag(K_paper), "-", color="k", lw=1.6, label="paper")
    ax.plot(t_arr, np.imag(K_orig), "--", color="C3", lw=1.4,
            label="original Python notebook")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im $K(t)$")
    ax.set_title("Imaginary part of fractional deviation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # (1,0) |K(t)| — magnitude. Try power-law fit on a log scale at early t.
    ax = axes[1, 0]
    mag_paper = np.abs(K_paper)
    mag_orig = np.abs(K_orig)
    ax.loglog(t_arr[1:], mag_paper[1:], "-", color="k", lw=1.6, label="paper")
    ax.loglog(t_arr[1:], mag_orig[1:], "--", color="C3", lw=1.4,
              label="original Python notebook")
    # Power-law reference (slope-1 line through small-t data)
    fit_mask = (t_arr > 0.05) & (t_arr < 0.5)
    if fit_mask.any():
        log_t = np.log(t_arr[fit_mask])
        log_mag = np.log(mag_paper[fit_mask])
        slope, intercept = np.polyfit(log_t, log_mag, 1)
        t_ref = np.array([0.05, T])
        ax.loglog(t_ref, np.exp(intercept) * t_ref ** slope, ":", color="0.4", lw=1.0,
                  label=rf"early-$t$ fit: $|K|\propto t^{{{slope:.2f}}}$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|K(t)|$")
    ax.set_title("Magnitude of fractional deviation (log-log)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(which="both", alpha=0.25)

    # (1,1) arg K(t) — phase
    ax = axes[1, 1]
    ax.plot(t_arr, np.angle(K_paper, deg=True), "-", color="k", lw=1.6, label="paper")
    ax.plot(t_arr, np.angle(K_orig, deg=True), "--", color="C3", lw=1.4,
            label="original Python notebook")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$\arg K(t)$  (deg)")
    ax.set_title("Phase of fractional deviation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    out = "examples/09_fractional_deviation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out}")

    # Print snapshot
    print("\nSnapshot of K(t) at selected times:")
    print(f"{'t':>5}  {'Re K_paper':>12}  {'Im K_paper':>12}  {'|K_paper|':>11}  {'|K_orig|':>11}")
    for tt in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        i = np.argmin(np.abs(t_arr - tt))
        print(f"{t_arr[i]:5.2f}  {K_paper[i].real:12.3e}  {K_paper[i].imag:12.3e}  "
              f"{abs(K_paper[i]):11.3e}  {abs(K_orig[i]):11.3e}")


if __name__ == "__main__":
    main()
