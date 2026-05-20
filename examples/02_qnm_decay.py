"""QNM-decay validation plot: PDE evolution of the fundamental b_{n=0,ℓ=0} mode.

Drop the standalone near-zone radial mode function ``rn_cwt.b_near(n=0, ℓ=0, x̆)``
as initial data into the linear NHERN-throat PDE, evolve, and verify the
result agrees with ψ(t, r_*) = b_0(r_*) · exp(s_0 · t) — i.e. that the PDE
solver reproduces the predicted complex QNM frequency

    s_0 = -(h(0) + i q)/2 ≈ -0.49981 - 0.01 i        (q = 0.02, r_+ = 1)

at the level set by spatial / temporal discretization. Saves
``examples/02_qnm_decay.png``.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_cwt import b_near, h
from rn_cwt import q as q_default
from rn_cwt import s
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    # Grid: bounded NHERN tortoise slab r_* ∈ [-5, -0.5] → x̆ ∈ [≈0.007, ≈1.54].
    grid = UniformGrid(x_min=-5.0, x_max=-0.5, N=201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)

    s_0 = s(0, 0)
    h_0 = h(0)

    # Initial data: single-mode QNM.
    psi0 = b_near(0, 0, xbreve)
    pi0 = s_0 * psi0

    # Evolve to T = 2.0 (decay factor e^{Re(s_0)·T} ≈ e^{-1.0} ≈ 0.37).
    # Boundary reflections off the Dirichlet ends propagate at speed 1 in r_*,
    # so the mid-point (r_* = -2.75) is causally isolated for ~ 2.25. T = 2.0
    # stays comfortably inside that window.
    T = 2.0
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0,
        tmax=T,
        q=q_default,
        m_eff_sq=0.0,
        nout=120,
    )
    assert sol.success

    # Theory: ψ_theory(t, r_*) = b_0(x̆) · e^{s_0 t}
    psi_theory = psi0[None, :] * np.exp(s_0 * t[:, None])

    # Mid-point probe (stays causally isolated for t < 2.25).
    mid = grid.N // 2
    psi_mid = psi[:, mid]
    psi_mid_theory = psi_theory[:, mid]

    # Diagnostics
    log_mag_pde = np.log(np.abs(psi_mid))
    slope_mag, _ = np.polyfit(t, log_mag_pde, 1)
    arg_unwrapped = np.unwrap(np.angle(psi_mid))
    slope_arg, _ = np.polyfit(t, arg_unwrapped, 1)

    print(f"q = {q_default}, h(0) = {complex(h_0)}")
    print(f"s_0       = {s_0}")
    print(f"slope log|ψ| at r_* = {rstar[mid]:.3f}: {slope_mag:.10f}")
    print(f"  Re(s_0) target:                {float(np.real(s_0)):.10f}")
    print(f"  relerr: {abs(slope_mag - np.real(s_0))/abs(np.real(s_0)):.3e}")
    print(f"slope arg ψ:                        {slope_arg:.10f}")
    print(f"  Im(s_0) target:                {float(np.imag(s_0)):.10f}")
    print(f"max relative residual ψ - ψ_theory: "
          f"{np.max(np.abs(psi_mid - psi_mid_theory))/abs(psi_mid_theory[0]):.3e}")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        rf"Linear PDE: fundamental near-zone QNM "
        rf"$b_{{n=0,\,\ell=0}}$ on NHERN throat "
        f"(q={q_default}, $r_+=1$)\n"
        rf"$\check s_0 = {s_0.real:.4f} {s_0.imag:+.4f}\,i$,  "
        rf"$h(0) = {float(np.real(h_0)):.4f}$",
        fontsize=13,
    )

    # Panel A — spatial snapshots
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    snap_idx = np.linspace(0, len(t) - 1, 6, dtype=int)
    for i, ci in zip(snap_idx, colors):
        ax.plot(rstar, np.abs(psi[i]), color=ci, lw=1.5,
                label=fr"$\check t={t[i]:.2f}$")
    ax.set_xlabel(r"$r_* = \ln[\check x / (\check x+1)]$")
    ax.set_ylabel(r"$|\psi(\check t,\,r_*)|$")
    ax.set_title(r"(A) Spatial profile $|\psi|$ at snapshots")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.25)

    # Panel B — log-magnitude decay vs theory
    ax = axes[0, 1]
    ax.plot(t, log_mag_pde, "o", ms=3.5, color="C0", label="PDE")
    ax.plot(t, np.log(np.abs(psi_mid_theory)), "-", color="C3", lw=1.5,
            label=fr"theory: $\log|b_0|$ + Re($\check s_0$)$\check t$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(rf"$\log|\psi(\check t,\,r_*={rstar[mid]:.2f})|$")
    relerr_mag = abs(slope_mag - np.real(s_0)) / abs(np.real(s_0))
    ax.set_title(
        rf"(B) Magnitude decay"
        "\n"
        rf"fitted = ${slope_mag:.8f}$,   "
        rf"Re($\check s_0$) = ${float(np.real(s_0)):.8f}$   "
        rf"(rel.\ err.\ {relerr_mag:.1e})"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel C — phase shift vs time
    ax = axes[1, 0]
    phase_pde = np.unwrap(np.angle(psi_mid)) - np.angle(psi_mid[0])
    phase_theory = np.imag(s_0) * t
    ax.plot(t, phase_pde, "o", ms=3.5, color="C0", label="PDE")
    ax.plot(t, phase_theory, "-", color="C3", lw=1.5,
            label=fr"theory: Im($\check s_0$)$\check t$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(rf"$\arg\psi(\check t) - \arg\psi(0)$  at $r_*={rstar[mid]:.2f}$")
    relerr_arg = abs(slope_arg - np.imag(s_0)) / abs(np.imag(s_0))
    ax.set_title(
        rf"(C) Phase rotation"
        "\n"
        rf"fitted = ${slope_arg:.8f}$,   "
        rf"Im($\check s_0$) = ${float(np.imag(s_0)):.8f}$   "
        rf"(rel.\ err.\ {relerr_arg:.1e})"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel D — relative residual vs theory at the interior point
    ax = axes[1, 1]
    rel_resid = np.abs(psi_mid - psi_mid_theory) / abs(psi_mid_theory[0])
    ax.semilogy(t, rel_resid, "o-", ms=3.5, lw=1.0, color="C2")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\psi_\mathrm{PDE} - \psi_\mathrm{theory}|"
                  r" / |\psi_\mathrm{theory}(0)|$")
    ax.set_title(r"(D) Relative residual at the interior point")
    ax.grid(which="both", alpha=0.25)

    out = "examples/02_qnm_decay.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
