"""Evolution-side regression: detuning tensor + action conservation under C_H.

The action-conservation test is the cleanest sanity check that ties
everything together (coupling assembly + mask + Omega + integrator).
"""
import numpy as np

from rn_cwt import (
    build_C_notes,
    build_Omega_notes,
    build_omega_arr,
    hermitian_adjoint_notes,
    integrate_A_notes,
)


SMALL_MODES = [
    {"n": 1, "l": 0, "m": 0},
    {"n": 2, "l": 0, "m": 0},
    {"n": 3, "l": 0, "m": 0},
]


def test_Omega_shape():
    omega_arr = build_omega_arr(SMALL_MODES)
    Omega = build_Omega_notes(omega_arr)
    N = len(SMALL_MODES)
    assert Omega.shape == (N, N, N, N)


def test_Omega_symmetry_unbarred_pair():
    """Omega[lam,i,j,k] = omega_lam + omega_k - omega_i - omega_j is symmetric
    in (i, j) — the unbarred amplitude pair."""
    omega_arr = build_omega_arr(SMALL_MODES)
    Omega = build_Omega_notes(omega_arr)
    assert np.allclose(Omega, np.swapaxes(Omega, 1, 2))


def test_Omega_diagonal_zero():
    """Omega[lam, lam, lam, lam] = 0 (sum cancels)."""
    omega_arr = build_omega_arr(SMALL_MODES)
    Omega = build_Omega_notes(omega_arr)
    for lam in range(len(SMALL_MODES)):
        assert abs(Omega[lam, lam, lam, lam]) < 1e-14


def test_action_conservation_under_C_H():
    """The Hamiltonian projection C_H must conserve N = sum |A|^2 to solver tol."""
    C = build_C_notes(SMALL_MODES, alpha=1.0)
    Cdag = hermitian_adjoint_notes(C)
    CH = 0.5 * (C + Cdag)

    omega_arr = build_omega_arr(SMALL_MODES)
    A0 = np.array(
        [1.0, 0.2 * np.exp(0.3j), 0.1 * np.exp(1.1j)],
        dtype=complex,
    )

    sol, *_ = integrate_A_notes(
        A0,
        tmax=50.0,
        eps=0.05,
        omega_arr=omega_arr,
        C=CH,
        nout=200,
        remove_diag=True,
        include_frequency_shift=True,
        phase_mode="real",
    )

    P = np.abs(sol.y.T) ** 2
    Ntot = np.sum(P, axis=1)
    drift = (Ntot[-1] - Ntot[0]) / Ntot[0]
    assert abs(drift) < 1e-8, f"action drift under C_H too large: {drift}"
