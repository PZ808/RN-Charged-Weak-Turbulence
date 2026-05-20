"""Amplitude validation — real-part of A_0(t) tracks ODE prediction.

Single-mode b_0 IC, ε=0.3 (perturbative: ε²|C|T ≈ 1.7×10⁻⁴), α=1.

The leading nonlinear effect is a frequency shift Δs = -i·ε²·C ≈ -8.6×10⁻⁵
+ 6.7×10⁻⁷·i. The dominant component is the *real* part (the slight extra
decay rate from Im(C) < 0). This is what we expect to see in the b_0
coefficient.

Plot Re(A_0(t) - 1) and Re(ΔA_ODE(t)) overlaid at r_* = -3 (where higher
modes don't dominate and the throat reflection hasn't arrived yet for
t < ~3.5). The agreement in shape and magnitude is the validation.

The imaginary part of ΔA is ~10⁻⁶, way smaller than the real part — visible
on a finer scale but at the level where non-modal extraction noise enters.
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
    ds = -1j * eps**2 * C0000
    print(f"C = {C0000:.4e}")
    print(f"ε²·|C|·T = {abs(eps**2 * C0000)*T:.2e}  (perturbative)")
    print(f"Δs = -iε²C = {ds:.4e}")
    print(f"  Re(Δs) = {ds.real:.4e}   (extra decay rate per t)")
    print(f"  Im(Δs) = {ds.imag:.4e}   (phase rotation rate per t)")

    grid = UniformGrid(x_min=-10.0, x_max=-0.05, N=1201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    b0 = b_near(0, 0, xbreve)

    psi0 = eps * A0 * b0
    pi0 = eps * A0 * s0 * b0
    t_arr, psi_pde, _, sol = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0, alpha=alpha, nout=nout,
    )
    assert sol.success

    # Analytic ODE A(t)
    A_ode = A0 * np.exp(-1j * eps**2 * C0000 * abs(A0) ** 2 * t_arr)
    dA_ode = A_ode - A0

    # Extract A_PDE(t) at r_* = -3 (clean window for t < 3.5)
    r_sample = -3.0
    i_r = int(np.argmin(np.abs(rstar - r_sample)))
    A_pde = psi_pde[:, i_r] / (eps * np.exp(s0 * t_arr) * b0[i_r])
    dA_pde = A_pde - A0

    # Two panels — Re(A − 1) vs t; Im(A − 1) vs t
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        rf"Amplitude validation: $A_0(t)$ — PDE vs ODE — single-mode $b_0$ IC,  "
        rf"$\epsilon={eps}$, $\alpha={alpha}$, sample $r_* = {rstar[i_r]:.1f}$"
        f"\n"
        rf"Predicted nonlinear frequency shift: $\Delta s = -i\epsilon^2 C \approx "
        rf"{ds.real:.1e} + {ds.imag:.1e}\,i$  (extra decay + small phase rotation)",
        fontsize=11,
    )

    # Re(ΔA)
    ax = axes[0]
    ax.plot(t_arr, np.real(dA_ode), "-", color="C3", lw=2.0,
            label=r"Re $(A_{\rm ODE} - 1)$  (analytic: $-i\epsilon^2 C t$)")
    ax.plot(t_arr, np.real(dA_pde), "--", color="k", lw=1.4,
            label=r"Re $(A_{\rm PDE} - 1)$  (extracted at $r_* = -3$)")
    ax.axhline(0, color="0.6", lw=0.5)
    # Reference line for the predicted slope
    ax.plot(t_arr, ds.real * t_arr, ":", color="0.5", lw=1.0,
            label=rf"slope-1 reference $\Delta s_{{\rm Re}} \cdot t = {ds.real:.2e}\cdot t$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Re $(A_0(t) - 1)$")
    ax.set_title(r"Real part of the b_0 coefficient shift")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)

    # Im(ΔA)
    ax = axes[1]
    ax.plot(t_arr, np.imag(dA_ode), "-", color="C3", lw=2.0,
            label=r"Im $(A_{\rm ODE} - 1)$  (analytic)")
    ax.plot(t_arr, np.imag(dA_pde), "--", color="k", lw=1.4,
            label=r"Im $(A_{\rm PDE} - 1)$  (extracted)")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"Im $(A_0(t) - 1)$")
    ax.set_title(r"Imaginary part — 100× smaller, near the extraction-floor")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    out = "examples/12_amplitude_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")

    # Print snapshot
    print()
    print(f"{'t':>5}  {'Re ΔA_ODE':>12}  {'Re ΔA_PDE':>12}  {'ratio':>7}  "
          f"{'Im ΔA_ODE':>12}  {'Im ΔA_PDE':>12}")
    for tt in [0.25, 0.5, 1.0, 1.5, 2.0]:
        i = np.argmin(np.abs(t_arr - tt))
        re_r = dA_pde[i].real / max(dA_ode[i].real, 1e-30, key=abs)
        print(f"{t_arr[i]:5.2f}  {dA_ode[i].real:12.3e}  {dA_pde[i].real:12.3e}  "
              f"{re_r:7.3f}  {dA_ode[i].imag:12.3e}  {dA_pde[i].imag:12.3e}")


if __name__ == "__main__":
    main()
