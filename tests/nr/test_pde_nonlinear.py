"""Tests for the nonlinear charged-scalar KG on NHERN.

The PDE is the same as the linear case plus a source α x̆(x̆+1) |ψ|² ψ on
the π̇ equation. Default α = 0 should reduce to the linear theory bit-for-
bit, and α ≠ 0 should produce a measurable deviation that grows over time
and with the field amplitude.
"""
import numpy as np
import pytest

from rn_cwt import b_near, s
from rn_cwt_nr import UniformGrid, evolve_kg, xbreve_of_rstar


@pytest.fixture
def grid():
    return UniformGrid(x_min=-5.0, x_max=-0.5, N=201)


@pytest.fixture
def xbreve(grid):
    return xbreve_of_rstar(grid.x)


def _qnm_state(modes, amplitudes, xbreve):
    psi = np.zeros_like(xbreve, dtype=complex)
    pi = np.zeros_like(xbreve, dtype=complex)
    for (n, ell), A_n in zip(modes, amplitudes):
        b = b_near(n, ell, xbreve)
        psi = psi + A_n * b
        pi = pi + A_n * s(n, ell) * b
    return psi, pi


def test_alpha_zero_is_identical_to_linear(grid, xbreve):
    """With α = 0 the nonlinear-aware code path must produce exactly the same
    output as the explicit linear run — bit-for-bit, since the source term
    is short-circuited by ``if alpha != 0.0``.
    """
    psi0, pi0 = _qnm_state([(0, 0)], [1.0], xbreve)

    t_lin, psi_lin, pi_lin, _ = evolve_kg(
        grid, psi0, pi0, tmax=1.0, q=0.02, m_eff_sq=0.0, nout=20,
    )
    t_nl, psi_nl, pi_nl, _ = evolve_kg(
        grid, psi0, pi0, tmax=1.0, q=0.02, m_eff_sq=0.0, alpha=0.0, nout=20,
    )
    assert np.allclose(t_lin, t_nl)
    assert np.allclose(psi_lin, psi_nl)
    assert np.allclose(pi_lin, pi_nl)


def test_nonlinear_evolution_runs_clean(grid, xbreve):
    """Drop a moderate-amplitude QNM IC with α > 0; field stays finite."""
    psi0, pi0 = _qnm_state([(0, 0)], [1.0], xbreve)

    t, psi, pi, sol = evolve_kg(
        grid, psi0, pi0, tmax=1.0, q=0.02, m_eff_sq=0.0,
        alpha=0.3, nout=20,
    )
    assert sol.success
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isinf(psi))
    init_peak = np.max(np.abs(psi0))
    final_peak = np.max(np.abs(psi[-1]))
    assert final_peak < 10 * init_peak


def test_nonlinear_evolution_differs_from_linear(grid, xbreve):
    """At α ≠ 0 and substantial amplitude, the mid-point waveform must
    differ from the linear evolution by more than discretization noise.
    """
    psi0, pi0 = _qnm_state([(0, 0)], [1.0], xbreve)

    T = 0.8
    _, psi_lin, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, alpha=0.0, nout=20,
    )
    _, psi_nl, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, alpha=0.5, nout=20,
    )
    mid = grid.N // 2
    diff = np.max(np.abs(psi_lin[:, mid] - psi_nl[:, mid]))
    assert diff > 1e-5, (
        f"linear vs nonlinear (α=0.5) mid-point difference {diff:.3e} too small"
    )


def test_nonlinear_small_alpha_perturbative(grid, xbreve):
    """At small α the deviation from linear should scale linearly in α.

    For α₁, α₂ both small, |ψ_nl(α₂) - ψ_lin| / |ψ_nl(α₁) - ψ_lin| ≈ α₂/α₁.
    Probes that the nonlinear source enters at leading order as expected.
    """
    psi0, pi0 = _qnm_state([(0, 0)], [1.0], xbreve)
    T = 0.5

    _, psi_lin, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, alpha=0.0, nout=10,
    )
    _, psi_a1, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, alpha=0.01, nout=10,
    )
    _, psi_a2, _, _ = evolve_kg(
        grid, psi0, pi0, tmax=T, q=0.02, m_eff_sq=0.0, alpha=0.02, nout=10,
    )

    mid = grid.N // 2
    dev1 = np.abs(psi_a1[-1, mid] - psi_lin[-1, mid])
    dev2 = np.abs(psi_a2[-1, mid] - psi_lin[-1, mid])
    ratio = dev2 / dev1
    assert 1.7 < ratio < 2.3, (
        f"α₂/α₁ = 2; got |Δψ(α₂)| / |Δψ(α₁)| = {ratio:.3f} (expected ≈ 2)"
    )
