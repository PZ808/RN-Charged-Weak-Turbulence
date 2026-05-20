"""Nonlinear charged-scalar KG on NHERN: linear vs nonlinear evolution.

Drop the fundamental QNM b_{n=0, ℓ=0} as initial data, evolve with α = 0
(linear) and α > 0 (nonlinear), and compare at a fixed interior point.

The linear evolution decays as ψ(t, r_*) = b_0(r_*) e^{s_0 t} exactly (per
``02_qnm_decay.py``). The nonlinear evolution adds the source

    α x̆(x̆+1) |ψ|² ψ

to the π̇ equation; for a pure-b_0 initial state this source generates
mode coupling into higher overtones (predominantly n=2 since 3 s_0 ≈ s_2
for ℓ=0). Whatever the projection details, the mid-point waveform should
visibly deviate from the linear prediction.

Saves examples/03_nonlinear_evolution.png.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_cwt import b_near, s
from rn_cwt import q as q_default
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    grid = UniformGrid(x_min=-5.0, x_max=-0.5, N=201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)

    s_0 = s(0, 0)
    psi0 = b_near(0, 0, xbreve)
    pi0 = s_0 * psi0

    T = 2.5
    nout = 250

    t_lin, psi_lin, _, sol_lin = evolve_kg(
        grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
        alpha=0.0, nout=nout,
    )

    alpha_vals = [0.1, 0.3, 0.7]
    runs = {}
    for a in alpha_vals:
        t_a, psi_a, _, sol_a = evolve_kg(
            grid, psi0, pi0, tmax=T, q=q_default, m_eff_sq=0.0,
            alpha=a, nout=nout,
        )
        runs[a] = (t_a, psi_a)
        print(f"α = {a}: solver success {sol_a.success}, steps {sol_a.nfev}")

    mid = grid.N // 2
    r_mid = rstar[mid]
    print(f"\nmid-point readout at r_* = {r_mid:.2f}")
    print(f"linear |ψ| range: [{np.min(np.abs(psi_lin[:, mid])):.4f}, "
          f"{np.max(np.abs(psi_lin[:, mid])):.4f}]")
    for a, (t_a, psi_a) in runs.items():
        diff = np.max(np.abs(psi_a[:, mid] - psi_lin[:, mid]))
        print(f"α = {a}: max |ψ_NL - ψ_lin| at mid = {diff:.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             constrained_layout=True)
    fig.suptitle(
        r"Nonlinear KG on NHERN: $\psi(\check t, r_*=$"
        f"{r_mid:.2f}"
        r"$)$,  $b_0$ initial data, $q$="
        f"{q_default}",
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(t_lin, np.abs(psi_lin[:, mid]), "k-", lw=1.8,
            label=r"$\alpha=0$ (linear)")
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_vals)))
    for (a, (t_a, psi_a)), ci in zip(runs.items(), colors):
        ax.plot(t_a, np.abs(psi_a[:, mid]), color=ci, lw=1.4,
                label=fr"$\alpha={a}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\psi(\check t, r_*\!=\!\mathrm{mid})|$")
    ax.set_title(r"(A) Mid-point waveform")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for (a, (t_a, psi_a)), ci in zip(runs.items(), colors):
        dev = np.abs(psi_a[:, mid] - psi_lin[:, mid])
        ax.semilogy(t_a, dev, color=ci, lw=1.4,
                    label=fr"$\alpha={a}$")
    ax.set_xlabel(r"$\check t$")
    ax.set_ylabel(r"$|\psi_{\rm NL} - \psi_{\rm lin}|$ at mid")
    ax.set_title(r"(B) Deviation from linear evolution")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, which="both")

    out = "examples/03_nonlinear_evolution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
