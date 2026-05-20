"""Test: does flipping the sign of C match the PDE concavity?

If Peter's S has a sign convention different from what we need for matching
the PDE on a Cauchy slice, then C_corrected = -C (with the existing |C|)
should produce an ODE prediction whose Re/Im have the SAME concavity as
the PDE. If yes, the long-standing concavity mismatch is a sign-convention
issue and we know where to look. If no, the issue is more subtle.

Single mode b_0 IC, ε=0.3, α=1.
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
    C_flip = -C0000  # the sign-flipped hypothesis

    print(f"C (original)        = {C0000:.4e}")
    print(f"C (sign-flipped)    = {C_flip:.4e}")
    print(f"Δs (original)       = -iε²C = {-1j*eps**2*C0000:.4e}")
    print(f"Δs (sign-flipped)   = +iε²C = {-1j*eps**2*C_flip:.4e}")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(0, 0, xbreve)
    mid = int(np.argmin(np.abs(rstar - (-2.5))))

    psi0 = eps * A0 * b0
    pi0 = eps * A0 * s0 * b0
    t_arr, psi_pde, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )

    # Linear baseline
    psi_lin = eps * A0 * np.exp(s0 * t_arr) * b0[mid]
    psi_pde_mid = psi_pde[:, mid]
    dpsi_pde = psi_pde_mid - psi_lin

    # ODE predictions: original C and sign-flipped C
    A_orig = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0)**2 * t_arr)
    A_flip = A0 * np.exp(-1j * eps**2 * C_flip * abs(A0)**2 * t_arr)
    dpsi_ode_orig = eps * (A_orig - A0) * np.exp(s0 * t_arr) * b0[mid]
    dpsi_ode_flip = eps * (A_flip - A0) * np.exp(s0 * t_arr) * b0[mid]

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        rf"Sign-flip test: single-mode $b_0$, $\epsilon={eps}$, $\alpha={alpha}$, $r_*={rstar[mid]:.2f}$"
        f"\n"
        rf"Does $C \to -C$ make the ODE prediction match the PDE concavity?",
        fontsize=11,
    )

    for col, (component, name) in enumerate([("real", "Re"), ("imag", "Im")]):
        ax = axes[col]
        op = np.real if component == "real" else np.imag
        ax.plot(t_arr, op(dpsi_pde), "-", color="k", lw=1.8,
                label=rf"{name} $\Delta\psi_{{\rm PDE}}$")
        ax.plot(t_arr, op(dpsi_ode_orig), "--", color="C3", lw=1.6,
                label=rf"{name} $\Delta\psi_{{\rm ODE}}$ with original $C$")
        ax.plot(t_arr, op(dpsi_ode_flip), "-.", color="C0", lw=1.6,
                label=rf"{name} $\Delta\psi_{{\rm ODE}}$ with $-C$")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel(r"$\check t$")
        ax.set_ylabel(rf"{name} $(\psi - \psi_{{\rm lin}})$")
        ax.set_title(f"{name} part")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    out = "examples/15_sign_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")

    # Snapshot at a few t values
    print()
    print(f"{'t':>5}  {'Re ΔΨ_PDE':>12}  {'Re ODE_orig':>12}  {'Re ODE_-C':>12}  "
          f"{'Im ΔΨ_PDE':>12}  {'Im ODE_orig':>12}  {'Im ODE_-C':>12}")
    for tt in [0.25, 0.5, 1.0, 1.5, 2.0]:
        i = np.argmin(np.abs(t_arr - tt))
        print(f"{t_arr[i]:5.2f}  "
              f"{dpsi_pde[i].real:12.3e}  {dpsi_ode_orig[i].real:12.3e}  {dpsi_ode_flip[i].real:12.3e}  "
              f"{dpsi_pde[i].imag:12.3e}  {dpsi_ode_orig[i].imag:12.3e}  {dpsi_ode_flip[i].imag:12.3e}")


if __name__ == "__main__":
    main()
