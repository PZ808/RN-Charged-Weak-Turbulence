"""Tests for the compactified linear KG evolver on NHERN.

Grid lives in y = tanh(r_*/L), y ∈ (-1, 0), with the spatial Laplacian
realized via Jacobian factors a(y), b(y). Same physical PDE as the bounded
version, just in a different chart that can reach the horizon (y → -1) and
the throat boundary (y → 0⁻).

The key correctness test cross-validates the compactified evolver against
the bounded one for the q=0, ℓ=0 free-wave reduction.
"""
import numpy as np
import pytest

from rn_cwt_nr import (
    UniformGrid,
    compact_spatial_coeffs,
    evolve_kg,
    evolve_kg_compact,
    rstar_of_y,
    xbreve_of_rstar,
    y_of_rstar,
)


@pytest.fixture
def y_grid():
    """Compactified slab y ∈ [-0.95, -0.05] with N=201 points."""
    return UniformGrid(x_min=-0.95, x_max=-0.05, N=201)


def test_y_grid_consistency(y_grid):
    """y → r_* → x̆ chain is positive and finite on the slab."""
    rstar = rstar_of_y(y_grid.x)
    xbreve = xbreve_of_rstar(rstar)
    assert np.all(rstar < 0)
    assert np.all(np.isfinite(rstar))
    assert np.all(xbreve > 0)
    assert np.all(np.isfinite(xbreve))


def test_compact_coeffs_vanish_at_horizon():
    """As y → -1 (horizon), a(y) ~ (1-y²)² → 0 quadratically and
    b(y) ~ -2y(1-y²) → 0 linearly. So a should shrink much faster than b."""
    a, b = compact_spatial_coeffs(np.array([-0.999, -0.99, -0.5, 0.0]), L=1.0)
    # Monotone increase in |a| from y = -1 toward y = 0.
    assert a[0] < a[1] < a[2] < a[3]
    # a vanishes quadratically — much smaller than the y = 0 reference.
    assert abs(a[0]) / abs(a[3]) < 1e-5
    # b vanishes linearly — smaller than reference but not as fast as a.
    assert abs(b[0]) / abs(b[2]) < 1e-2


def test_gaussian_pulse_does_not_blow_up_compact(y_grid):
    rstar = rstar_of_y(y_grid.x)
    psi0 = np.exp(-((rstar + 2.0) ** 2) / (2 * 0.5**2)).astype(complex)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg_compact(
        y_grid, psi0, pi0, tmax=2.0, q=0.02, m_eff_sq=0.0, L=1.0, nout=50,
    )
    assert sol.success
    assert not np.any(np.isnan(psi))
    init_peak = np.max(np.abs(psi0))
    final_peak = np.max(np.abs(psi[-1]))
    assert final_peak < 100 * init_peak


def test_constant_field_stationary_compact(y_grid):
    """ψ ≡ const, π = 0, q = 0, ℓ = 0 → stationary.

    All spatial derivatives of a constant are 0, so the Jacobian factors
    don't matter; same fixed-point as the bounded version.
    """
    psi0 = np.ones_like(y_grid.x, dtype=complex)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg_compact(
        y_grid, psi0, pi0, tmax=5.0, q=0.0, m_eff_sq=0.0, L=1.0, nout=20,
    )
    max_drift = np.max(np.abs(psi[-1, 5:-5] - 1.0))
    assert max_drift < 1e-8


def test_dirichlet_bcs_enforced_compact(y_grid):
    rstar = rstar_of_y(y_grid.x)
    psi0 = np.exp(-((rstar + 2.0) ** 2) / (2 * 0.5**2)).astype(complex)
    psi0[0] = 0.0
    psi0[-1] = 0.0
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg_compact(
        y_grid, psi0, pi0, tmax=1.0, q=0.02, m_eff_sq=0.0, L=1.0, nout=20,
    )
    assert np.allclose(psi[:, 0], 0.0, atol=1e-12)
    assert np.allclose(psi[:, -1], 0.0, atol=1e-12)


def test_q_zero_real_initial_data_stays_real_compact(y_grid):
    rstar = rstar_of_y(y_grid.x)
    psi0 = np.exp(-((rstar + 2.0) ** 2) / (2 * 0.5**2)).astype(complex)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg_compact(
        y_grid, psi0, pi0, tmax=1.0, q=0.0, m_eff_sq=0.0, L=1.0, nout=20,
    )
    assert np.max(np.abs(psi.imag)) < 1e-10


def test_compact_matches_bounded_in_interior_free_wave():
    """For q = 0, ℓ = 0, the PDE is the free wave ∂²_t ψ = ∂²_{r_*} ψ.

    The bounded and compactified evolvers should produce the same field
    at any common physical point r_*, modulo discretization error.
    """
    L = 1.0

    rstar_min, rstar_max = -4.0, -0.3

    # Bounded grid (uniform in r_*).
    bounded_grid = UniformGrid(x_min=rstar_min, x_max=rstar_max, N=401)
    rstar_bnd = bounded_grid.x
    r0, sigma = -2.0, 0.4
    psi0_bnd = np.exp(-((rstar_bnd - r0) ** 2) / (2 * sigma**2)).astype(complex)
    pi0_bnd = np.zeros_like(psi0_bnd)

    # Compactified grid (uniform in y).
    y_min, y_max = y_of_rstar(rstar_min, L=L), y_of_rstar(rstar_max, L=L)
    compact_grid = UniformGrid(x_min=float(y_min), x_max=float(y_max), N=401)
    rstar_cmp = rstar_of_y(compact_grid.x, L=L)
    psi0_cmp = np.exp(-((rstar_cmp - r0) ** 2) / (2 * sigma**2)).astype(complex)
    pi0_cmp = np.zeros_like(psi0_cmp)

    T = 1.0
    t_b, psi_b, _, sol_b = evolve_kg(
        bounded_grid, psi0_bnd, pi0_bnd, tmax=T, q=0.0, m_eff_sq=0.0, nout=2,
    )
    t_c, psi_c, _, sol_c = evolve_kg_compact(
        compact_grid, psi0_cmp, pi0_cmp, tmax=T, q=0.0, m_eff_sq=0.0, L=L, nout=2,
    )

    assert sol_b.success and sol_c.success

    # Compare at the common interior r_* range, away from boundaries.
    rstar_compare = np.linspace(-3.0, -0.8, 50)
    psi_b_interp = np.interp(rstar_compare, rstar_bnd, np.real(psi_b[-1])) + \
                   1j * np.interp(rstar_compare, rstar_bnd, np.imag(psi_b[-1]))
    # Compactified grid is uniform in y; interpolate via the monotone rstar_cmp.
    sort_idx = np.argsort(rstar_cmp)
    psi_c_interp = np.interp(
        rstar_compare, rstar_cmp[sort_idx], np.real(psi_c[-1])[sort_idx]
    ) + 1j * np.interp(
        rstar_compare, rstar_cmp[sort_idx], np.imag(psi_c[-1])[sort_idx]
    )

    diff = np.max(np.abs(psi_b_interp - psi_c_interp))
    # The compact grid is uniform in y so its dx_r* varies; near y → -1 the
    # effective r_* spacing is ~10× the bounded grid's. At N = 401 this gives
    # about 1-2% disagreement on the free-wave reduction, which is consistent
    # with the two schemes solving the same PDE on different grids.
    assert diff < 3e-2, f"free-wave bounded vs compact disagreement: {diff:.3e}"
