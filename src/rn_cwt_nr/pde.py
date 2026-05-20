"""Charged-scalar Klein-Gordon on the NHERN throat, tortoise coordinate.

Background (NHERN near-zone in units r_+ = σ = 1; see §10 of the draft paper):

    ds^2 = -f d{t̆}^2 + d{x̆}^2 / f + dΩ^2,    f = x̆ (x̆ + 1)
    A_{t̆} = x̆

The matter Lagrangian (paper §2 eq. 2) is

    L_matter = - |D Φ|² - (α/2) |Φ|⁴,

so the field equation for the spherically reduced ψ = r Φ on the AdS_2
factor (in tortoise r_*) reads

    ∂²_t ψ - ∂²_{r_*} ψ  =  2 i q x̆ ∂_t ψ  +  V(r_*) ψ  +  α x̆(x̆+1) |ψ|² ψ

with V(r_*) = q² x̆² - m²_eff x̆(x̆+1)  and  m²_eff = ℓ(ℓ+1). The nonlinear
source α x̆(x̆+1) |ψ|² ψ comes from the (α/2)|Φ|⁴ self-interaction; the
prefactor is α (f/r²) which equals α x̆(x̆+1) in r_+ = 1, σ = 0 NHERN.

When α = 0 the equation is purely linear; the field reduces to the
charged-scalar KG and all existing linear tests are recovered unchanged.

For ℓ = 0 with small q the potential is tiny and positive (V ≈ q² x̆²); for
ℓ ≥ 1 it is strongly negative ("trapping") at large x̆. The friction term
2 i q x̆ ∂_t ψ adds gauge-induced phase rotation.

First-order form with conjugate momentum π = ∂_t ψ. For Phase 1 v2 the
spatial domain is a bounded r_* slab with Dirichlet BCs at both endpoints —
compactification of r_* and physical (outgoing-at-horizon, normalizable-at-
throat-boundary) BCs are the next iteration.
"""
from __future__ import annotations

import numpy as np

from .grid import UniformGrid, fd4_d1, fd4_d2


def build_state(psi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Pack (ψ, π) on the grid into the flat solve_ivp state vector."""
    return np.concatenate(
        [np.asarray(psi, dtype=complex), np.asarray(pi, dtype=complex)]
    )


def split_state(y: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the flat state into (ψ, π)."""
    return y[:N], y[N:]


def kg_rhs(
    t: float,
    y: np.ndarray,
    grid: UniformGrid,
    xbreve: np.ndarray,
    q: float,
    m_eff_sq: float,
    alpha: float = 0.0,
    horizon_bc: str = "outgoing",
    throat_bc: str = "dirichlet",
) -> np.ndarray:
    """RHS of (ψ, π) for the charged-scalar KG on NHERN in r_*.

    Linear by default; ``alpha != 0`` switches on the cubic
    ``-(α/(4π))·x̆(x̆+1)·|ψ|²·ψ`` self-interaction.

    Implements

        ∂_t ψ = π
        ∂_t π = ∂²_{r_*} ψ + 2 i q x̆ π + [q² x̆² - m²_eff x̆(x̆+1)] ψ

    in the bulk, on a uniform r_*-grid. State y = [ψ, π] of length 2N.

    Boundary conditions
    -------------------
    ``horizon_bc`` ∈ {"outgoing", "dirichlet"} sets the BC at i = 0
    (r_* → -∞, x̆ → 0).

    - ``"outgoing"`` (default): Sommerfeld condition ∂_t ψ = +∂_{r_*} ψ
      enforced by overriding the ψ̇ RHS at i=0 with a one-sided 4th-order
      FD evaluation of ∂_{r_*} ψ. **This is exact** at the horizon because
      the gauge friction (2iq x̆ ∂_t ψ) and the potential
      (q²x̆² - m²_eff x̆(x̆+1)) both vanish as x̆ → 0, leaving the free
      wave equation. The QNM asymptotic b_n ~ e^{s_n r_*} satisfies this
      automatically. Lets the QNM decay through the boundary cleanly,
      without the reflection artifact that Dirichlet introduces.

    - ``"dirichlet"``: ψ̇[0] = π̇[0] = 0, so ψ[0] stays at its initial value.
      Causes a reflection that contaminates the interior at t ≳ |r_*,min|;
      useful for tests that need a causally-isolated window.

    ``throat_bc`` ∈ {"dirichlet"} sets the BC at i = N-1 (r_* → 0⁻,
    x̆ → ∞). Currently only Dirichlet is supported there. The throat
    boundary has divergent potential and friction; a physical
    normalizable-falloff condition is left for a future iteration.
    """
    N = grid.N
    psi = y[:N]
    pi = y[N:]

    d2psi = fd4_d2(psi, grid.dx)

    V = q**2 * xbreve**2 - m_eff_sq * xbreve * (xbreve + 1.0)

    dpsi_dt = pi.copy()
    dpi_dt = d2psi + 2j * q * xbreve * pi + V * psi

    if alpha != 0.0:
        # Nonlinear source after spherical reduction ψ = B · Y_{00} (paper §5
        # eq. ≈1267): the source on ∂_t π = ∂²_t B is
        #     − (α / (4π)) · (f/r²) · |B|² B,
        # with f/r² = x̆(x̆+1) in NHERN and the 1/(4π) from |Y_{00}|².
        # The sign comes from moving the term across in paper eq. ≈629:
        # the RHS of the wave equation reads +α(f/r²)|ψ|²ψ in the form
        # −ψ̈ + ... = +α(f/r²)|ψ|²ψ, so ψ̈ = ... − α(f/r²)|ψ|²ψ.
        # The matching ODE side carries Q = 1/(4π) for the spherical sector
        # in Q_notes.
        dpi_dt = dpi_dt - (alpha / (4.0 * np.pi)) * xbreve * (xbreve + 1.0) * np.abs(psi) ** 2 * psi

    # Horizon side (i = 0)
    if horizon_bc == "outgoing":
        d1psi = fd4_d1(psi, grid.dx)
        dpsi_dt[0] = d1psi[0]  # ψ̇ = +∂_{r_*} ψ  (left-mover)
        # π̇[0] left as the bulk update — π[0] is now an auxiliary that does
        # not couple back into the physical ψ evolution.
    elif horizon_bc == "dirichlet":
        dpsi_dt[0] = 0.0
        dpi_dt[0] = 0.0
    else:
        raise ValueError(f"unknown horizon_bc={horizon_bc!r}")

    # Throat side (i = N-1)
    if throat_bc == "dirichlet":
        dpsi_dt[-1] = 0.0
        dpi_dt[-1] = 0.0
    else:
        raise ValueError(f"unknown throat_bc={throat_bc!r}")

    return np.concatenate([dpsi_dt, dpi_dt])


def kg_rhs_compact(
    t: float,
    state: np.ndarray,
    grid: UniformGrid,
    xbreve: np.ndarray,
    q: float,
    m_eff_sq: float,
    a_coeff: np.ndarray,
    b_coeff: np.ndarray,
) -> np.ndarray:
    """RHS of (ψ, π) on the compactified grid y = tanh(r_*/L), y ∈ (-1, 0).

    Same physical equation as ``kg_rhs`` but the spatial operator
    ∂²_{r_*} is realized via the Jacobian:

        ∂²_{r_*} ψ = a(y) ∂²_y ψ + b(y) ∂_y ψ,
        a(y) = (1 - y²)²/L²,    b(y) = -2 y (1 - y²)/L².

    Pre-compute ``a_coeff, b_coeff = coords.compact_spatial_coeffs(grid.x, L)``
    once outside the integrator.

    Parameters
    ----------
    grid : UniformGrid in y (both endpoints strictly inside (-1, 0)).
    xbreve : (N,) array of x̆ at the grid points, x̆ = x̆_of_rstar(rstar_of_y(y, L)).
    a_coeff, b_coeff : (N,) precomputed spatial-derivative coefficients.

    Boundary conditions: Dirichlet at both endpoints — RHS forced to zero
    at i = 0 (deep tortoise, near horizon) and i = N-1 (near throat boundary).
    """
    N = grid.N
    psi = state[:N]
    pi = state[N:]

    d1psi = fd4_d1(psi, grid.dx)
    d2psi = fd4_d2(psi, grid.dx)

    laplace_rstar = a_coeff * d2psi + b_coeff * d1psi
    V = q**2 * xbreve**2 - m_eff_sq * xbreve * (xbreve + 1.0)

    dpsi_dt = pi.copy()
    dpi_dt = laplace_rstar + 2j * q * xbreve * pi + V * psi

    dpsi_dt[0] = 0.0
    dpsi_dt[-1] = 0.0
    dpi_dt[0] = 0.0
    dpi_dt[-1] = 0.0

    return np.concatenate([dpsi_dt, dpi_dt])
