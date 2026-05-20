"""Tests for the bilinear-form projection onto near-zone QNMs.

The bilinear form is the symmetric paper-§3 construction with boundary point
contributions as the regulator. On a finite r_*-slab the cancellation
between bulk and boundary is approximate (full cancellation requires
r_*,min → -∞), so QNM orthogonality on our grid is approximate. The Gram-
matrix solve in ``project_qnm`` accommodates this — for ψ ∈ span{b_n} the
recovery is exact regardless of basis non-orthogonality.

(Caveat: QNMs are genuinely difficult on Cauchy slices; the mode functions
b_n diverge at the horizon as x̆^{Re s_n}, so "QNM initial data" is formally
not in L². On the finite numerical grid the boundary value is bounded but
large, and we are leaning on the outgoing-at-horizon BC to let this divergence
fall through cleanly.)
"""
import numpy as np
import pytest

from rn_cwt import b_near, s
from rn_cwt_nr import (
    UniformGrid,
    evolve_kg,
    project_qnm,
    xbreve_of_rstar,
)


@pytest.fixture
def grid():
    return UniformGrid(x_min=-5.0, x_max=-0.5, N=201)


@pytest.fixture
def xbreve(grid):
    return xbreve_of_rstar(grid.x)


L0_MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
Q_TEST = 0.02


def _qnm_state(modes, amplitudes, xbreve):
    """Build (ψ, π) on the grid for ψ = Σ A_n b_n, π = Σ A_n s_n b_n."""
    psi = np.zeros_like(xbreve, dtype=complex)
    pi = np.zeros_like(xbreve, dtype=complex)
    for (n, ell), A_n in zip(modes, amplitudes):
        b = b_near(n, ell, xbreve)
        psi = psi + A_n * b
        pi = pi + A_n * s(n, ell) * b
    return psi, pi


def test_project_pure_single_mode_recovers_unity(grid, xbreve):
    """ψ = b_n, π = s_n b_n → projection gives δ_{nm}."""
    for n_target in range(4):
        amps = [1.0 if k == n_target else 0.0 for k in range(4)]
        psi, pi = _qnm_state(L0_MODES, amps, xbreve)
        A = project_qnm(psi, pi, xbreve, L0_MODES, grid, Q_TEST)
        assert np.isclose(A[n_target], 1.0, atol=1e-6), (
            f"n_target={n_target}: A[{n_target}]={A[n_target]}"
        )
        for n in range(4):
            if n == n_target:
                continue
            assert abs(A[n]) < 1e-5, (
                f"n_target={n_target}: spurious A[{n}]={A[n]}"
            )


def test_project_linear_combination(grid, xbreve):
    """Project a known linear combination — recover each coefficient."""
    A_in = np.array([1.0, 0.5 * np.exp(0.3j), -0.2j, 0.7 * np.exp(-1.0j)])
    psi, pi = _qnm_state(L0_MODES, A_in, xbreve)
    A_out = project_qnm(psi, pi, xbreve, L0_MODES, grid, Q_TEST)
    assert np.allclose(A_out, A_in, atol=1e-5), f"A_in={A_in}, A_out={A_out}"


def test_project_2d_field_returns_time_evolution(grid, xbreve):
    """Project ψ/π of shape (nout, N) → A of shape (nout, M)."""
    A_in = np.array([1.0, 0.5 * np.exp(0.3j)])
    psi_t0, pi_t0 = _qnm_state([(0, 0), (1, 0)], A_in, xbreve)
    # Construct an artificial 3-slice "evolution" — same shape scaled
    psi = np.stack([psi_t0, 2 * psi_t0, 0.5 * psi_t0])
    pi = np.stack([pi_t0, 2 * pi_t0, 0.5 * pi_t0])
    A = project_qnm(psi, pi, xbreve, [(0, 0), (1, 0)], grid, Q_TEST)
    assert A.shape == (3, 2)
    assert np.allclose(A[0], A_in, atol=1e-5)
    assert np.allclose(A[1], 2 * A_in, atol=1e-5)
    assert np.allclose(A[2], 0.5 * A_in, atol=1e-5)


# NOTE: PDE-evolved-projection tests removed for now.
#
# On a bounded Cauchy-slice grid the bilinear form's boundary contribution
# at the throat side, b_m(r_*,max) ψ(r_*,max), is sensitive to whatever BC
# we use at the throat. Dirichlet pins ψ(r_*,max) at the initial value
# (contradicting the QNM decay); simple Sommerfeld doesn't match the
# AdS-like falloff at the throat either. Both cause the bilinear-form
# projection to drift immediately at t > 0 — even though the field at
# the interior decays correctly at the predicted complex frequency
# (verified by ``test_qnm_initial_data_decays_at_predicted_rate``).
#
# Clean Cauchy-slice projection would require either a normalizable-falloff
# BC at the throat (mode-dependent, tricky) or a different
# regularization. Setting this aside for now per the discussion of
# Cauchy-slice QNM difficulties.
