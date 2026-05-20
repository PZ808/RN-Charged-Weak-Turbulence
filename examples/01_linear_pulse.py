"""Phase 1 v2: linear charged-scalar KG on NHERN throat, tortoise r_*.

Run with ``uv run python examples/01_linear_pulse.py``. Saves a PNG and
prints a summary.

Setup
-----
- Coordinate: NHERN tortoise r_* = ln[x̆/(x̆+1)]. Grid r_* ∈ [-5, -0.5];
  x̆ ∈ [≈0.007, ≈1.54].
- Field equation (ℓ = m = 0 sector, in dimensionless near-zone time t̆):
      ∂²_t ψ - ∂²_{r_*} ψ = 2 i q x̆ ∂_t ψ + q² x̆² ψ.
- Initial data: real Gaussian centred at r_*,0 = -2.0, σ = 0.5; π = 0.
- BCs: Dirichlet at both endpoints (artifact; pulse will reflect — physical
  outgoing/normalizable BCs come with the compactification step).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_cwt import q as q_default
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def main():
    grid = UniformGrid(x_min=-5.0, x_max=-0.5, N=201)
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)

    psi0 = np.exp(-((rstar + 2.0) ** 2) / (2 * 0.5**2)).astype(complex)
    pi0 = np.zeros_like(psi0)

    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0,
        tmax=10.0,
        q=q_default,
        m_eff_sq=0.0,
        nout=200,
    )

    print(f"solver success: {sol.success}")
    print(f"steps: {sol.nfev}      (v0 raw-ρ took ~86k)")
    print(f"x̆ range on grid: [{xbreve.min():.4f}, {xbreve.max():.4f}]")
    print(f"|ψ| peak initial: {np.max(np.abs(psi[0])):.4f}")
    print(f"|ψ| peak final:   {np.max(np.abs(psi[-1])):.4f}")
    print(f"max |ψ| over t,r_*: {np.max(np.abs(psi)):.4f}")
    print(f"any NaN: {np.any(np.isnan(psi))}")

    fig, ax = plt.subplots(figsize=(7, 4))
    snapshot_idx = [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, len(t) - 1]
    for i in snapshot_idx:
        ax.plot(rstar, np.abs(psi[i]), label=f"t̆ = {t[i]:.2f}")
    ax.set_xlabel(r"$r_* = \ln[\check x / (\check x + 1)]$")
    ax.set_ylabel(r"$|\psi(\check t, r_*)|$")
    ax.set_title(r"Linear charged KG on NHERN throat (Phase 1 v2)")
    ax.legend()
    fig.tight_layout()
    out = "examples/01_linear_pulse.png"
    fig.savefig(out, dpi=120)
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
