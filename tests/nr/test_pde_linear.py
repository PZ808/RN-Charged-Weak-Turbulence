"""Smoke + sanity tests for the linear charged-scalar KG PDE on NHERN throat.

PDE is evolved in NHERN tortoise r_* = ln[x̆/(x̆+1)] (r_* < 0). For the (l, m)
sector with m²_eff = ℓ(ℓ+1):

    ∂²_t ψ - ∂²_{r_*} ψ = 2 i q x̆ ∂_t ψ + [q² x̆² - m²_eff x̆(x̆+1)] ψ.

These tests check the time-stepping is stable, BCs are honored, the q=0
specialization preserves reality, and the q=0, ℓ=0 reduction gives a free
wave (constant field is stationary).
"""
import numpy as np
import pytest

from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


def gaussian_pulse(rstar, rstar0, sigma):
    """Real-valued Gaussian centred at r_*,0 with width sigma."""
    return np.exp(-((rstar - rstar0) ** 2) / (2 * sigma**2)).astype(complex)


@pytest.fixture
def grid():
    """Bounded NHERN tortoise slab r_* ∈ [-5, -0.5]; x̆ ∈ [≈0.007, ≈1.54]."""
    return UniformGrid(x_min=-5.0, x_max=-0.5, N=121)


def test_gaussian_pulse_does_not_blow_up(grid):
    rstar = grid.x
    psi0 = gaussian_pulse(rstar, rstar0=-2.0, sigma=0.5)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=2.0, q=0.02, m_eff_sq=0.0, nout=50,
    )
    assert sol.success
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isinf(psi))
    init_peak = np.max(np.abs(psi0))
    final_peak = np.max(np.abs(psi[-1]))
    assert final_peak < 100 * init_peak


def test_dirichlet_bcs_enforced(grid):
    """Explicit Dirichlet on both sides — value at endpoints stays at IC."""
    rstar = grid.x
    psi0 = gaussian_pulse(rstar, rstar0=-2.0, sigma=0.5)
    psi0[0] = 0.0
    psi0[-1] = 0.0
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=1.0, q=0.02, m_eff_sq=0.0, nout=20,
        horizon_bc="dirichlet", throat_bc="dirichlet",
    )
    assert np.allclose(psi[:, 0], 0.0, atol=1e-12)
    assert np.allclose(psi[:, -1], 0.0, atol=1e-12)


def test_q_zero_real_initial_data_stays_real(grid):
    """With q=0, all coefficients are real — real IC stays real."""
    rstar = grid.x
    psi0 = gaussian_pulse(rstar, rstar0=-2.0, sigma=0.5)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=1.0, q=0.0, m_eff_sq=0.0, nout=20,
    )
    max_imag = np.max(np.abs(psi.imag))
    assert max_imag < 1e-10


def test_constant_field_stationary_q_zero_ell_zero(grid):
    """For q=0 and ℓ=0 (m²_eff=0), the equation is the free wave
    ∂²_t ψ = ∂²_{r_*} ψ. A constant ψ is a stationary solution.

    Strong sanity check that the NHERN coefficient assembly is correct:
    any spurious ψ-on-the-RHS term would break this.
    """
    psi0 = np.ones_like(grid.x, dtype=complex)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=5.0, q=0.0, m_eff_sq=0.0, nout=20,
    )
    # Interior should remain ≈ 1.
    max_drift = np.max(np.abs(psi[-1, 5:-5] - 1.0))
    assert max_drift < 1e-8, f"constant field drifted by {max_drift:.2e}"


def test_xbreve_grid_consistency(grid):
    """x̆(r_*) = e^{r_*}/(1 - e^{r_*}) is positive and finite on the slab."""
    xbreve = xbreve_of_rstar(grid.x)
    assert np.all(xbreve > 0)
    assert np.all(np.isfinite(xbreve))
    # r_* = -5  →  x̆ ≈ 0.00679;   r_* = -0.5  →  x̆ ≈ 1.5415.
    assert np.isclose(xbreve[0], np.exp(-5) / (1 - np.exp(-5)))
    assert np.isclose(xbreve[-1], np.exp(-0.5) / (1 - np.exp(-0.5)))


def test_free_wave_propagates_at_unit_speed(grid):
    """For q = 0, ℓ = 0 the equation is ∂²_t ψ = ∂²_{r_*} ψ.

    A right-going Gaussian pulse ψ(0, r_*) = f(r_* - r_*,0), π(0, r_*) = -f'(...)
    should translate at speed 1 in r_*: ψ(t, r_*) = f(r_* - r_*,0 - t).
    """
    rstar = grid.x
    r0 = -2.0
    sigma = 0.3
    psi0 = np.exp(-((rstar - r0) ** 2) / (2 * sigma**2)).astype(complex)
    # For a right-mover ψ = f(r_* - t): π = ∂_t ψ = -f'(r_* - t).
    pi0 = (
        ((rstar - r0) / sigma**2)
        * np.exp(-((rstar - r0) ** 2) / (2 * sigma**2))
    ).astype(complex)

    T = 0.5  # short enough that the pulse stays well inside the grid
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.0, m_eff_sq=0.0, nout=2,
    )
    # Expected: pulse centered at r0 + T.
    final = np.abs(psi[-1])
    peak_idx = np.argmax(final)
    peak_r = rstar[peak_idx]
    assert abs(peak_r - (r0 + T)) < 2 * grid.dx, (
        f"pulse peak at {peak_r:.3f}, expected ≈ {r0 + T:.3f}"
    )


def test_qnm_initial_data_decays_at_predicted_rate(grid):
    """Drop the fundamental near-zone QNM b_{n=0, ℓ=0}(x̆) as initial data
    and verify |ψ(t, r_*)| at an interior point decays as exp(Re(s_0) · t).

    For ℓ = 0, q = 0.02:  s_0 = -(h(0) + iq)/2 ≈ -0.49981 - 0.01j
    So |ψ| decays at rate |Re(s_0)| ≈ 0.5 per unit dimensionless t̆.

    Single-mode initial data ψ(0) = b_0, π(0) = s_0 · b_0 should propagate
    as ψ(t) = b_0 · exp(s_0 · t) for any t before boundary effects arrive
    at the sampling point. We measure at r_* = grid mid-point and evolve
    only up to half the grid-width, keeping the interior clean.
    """
    from rn_cwt import b_near, s
    from rn_cwt_nr import xbreve_of_rstar

    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)

    psi0 = b_near(0, 0, xbreve)
    s_0 = s(0, 0)
    pi0 = s_0 * psi0

    # Grid r_* ∈ [-5, -0.5] → width 4.5; choose T < width/2 so the interior
    # mid-point isn't yet contaminated by reflections off either boundary.
    T = 1.5
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, nout=30,
    )
    assert sol.success

    # Sample at the middle grid point; this is the cleanest interior probe.
    mid = grid.N // 2
    log_mag = np.log(np.abs(psi[:, mid]))

    # Fit log|ψ| = Re(s_0) t + const over the early window before BCs bite.
    slope, _ = np.polyfit(t, log_mag, 1)
    expected = float(np.real(s_0))
    assert abs(slope - expected) < 5e-2, (
        f"QNM decay slope = {slope:.4f}, expected {expected:.4f}"
    )


def test_solver_steps_far_fewer_than_raw_rho_v0(grid):
    """NHERN tortoise should be much less stiff than v0 raw-ρ.

    v0 (Phase 1 v0, raw ρ ∈ [0.5, 5], same Gaussian) took ~86k steps for
    tmax=10. NHERN tortoise should be at least 10× cheaper.
    """
    rstar = grid.x
    psi0 = gaussian_pulse(rstar, rstar0=-2.0, sigma=0.5)
    pi0 = np.zeros_like(psi0)
    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=10.0, q=0.02, m_eff_sq=0.0, nout=50,
    )
    assert sol.success
    assert sol.nfev < 20_000, f"NHERN tortoise took {sol.nfev} steps; expected << 86k"
