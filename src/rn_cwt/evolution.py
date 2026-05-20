"""Interaction-picture amplitude evolution.

Solves

    i dot A_lam = eps^2 sum_{i,j,k} C[lam, i, j, k] A_i A_j bar(A_k)
                  * exp(i Omega[lam, i, j, k] t),

with three ``phase_mode`` options and optional removal of diagonal
frequency-shift terms. See README "Phase conventions" for the trade-offs.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .background import w


def omega_mode(J):
    """omega for a mode dict ``{'n': n, 'l': ell, ...}``."""
    return w(J["n"], J["l"])


def build_omega_arr(modes):
    """Convenience: assemble the omega array for a mode list."""
    return np.array([omega_mode(J) for J in modes], dtype=complex)


def build_Omega_notes(omega_arr, real_phase=False):
    """Omega[lam, i, j, k] = omega_lam + omega_k - omega_i - omega_j.

    With ``real_phase=True``, the real part of omega_arr is taken first.
    """
    if real_phase:
        omega_arr = np.real(omega_arr)
    return (
        omega_arr[:, None, None, None]
        - omega_arr[None, :, None, None]
        - omega_arr[None, None, :, None]
        + omega_arr[None, None, None, :]
    )


def rhs_A_notes(t, A, eps, C, Omega, mask=None, phase_mode="exact"):
    """RHS for solve_ivp of i dot A = eps^2 sum C A A bar(A) e^{i Omega t}."""
    if mask is None:
        mask = 1.0

    if phase_mode == "exact":
        phase = np.exp(1j * Omega * t)
    elif phase_mode == "real":
        phase = np.exp(1j * np.real(Omega) * t)
    elif phase_mode == "resonant":
        phase = 1.0
    else:
        raise ValueError("phase_mode must be 'exact', 'real', or 'resonant'.")

    Ceff = C * mask * phase
    nonlinear = np.einsum(
        "lijk,i,j,k->l",
        Ceff,
        A,
        A,
        np.conjugate(A),
        optimize=True,
    )
    return -1j * eps**2 * nonlinear


def build_nondiag_mask_notes(N):
    """Boolean mask removing diagonal self-frequency-shift terms:

        mask[lam, q, lam, q] = False   (A_q A_lam bar(A_q))
        mask[lam, lam, q, q] = False   (A_lam A_q bar(A_q)).
    """
    mask = np.ones((N, N, N, N), dtype=bool)
    for lam in range(N):
        for q in range(N):
            mask[lam, q, lam, q] = False
            mask[lam, lam, q, q] = False
    return mask


def omega_shift_notes(A0, C):
    """omega_shift[lam] = sum_q (C[lam, q, lam, q] + C[lam, lam, q, q]) |A_q|^2."""
    N = len(A0)
    shift = np.zeros(N, dtype=complex)
    for lam in range(N):
        for q in range(N):
            shift[lam] += (
                C[lam, q, lam, q] + C[lam, lam, q, q]
            ) * abs(A0[q]) ** 2
    return shift


def integrate_A_notes(
    A0,
    tmax,
    eps,
    omega_arr,
    C,
    nout=2000,
    remove_diag=True,
    include_frequency_shift=True,
    phase_mode="real",
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
):
    """Evolve interaction-picture amplitudes with ``scipy.integrate.solve_ivp``.

    Parameters mirror the notebook helper. Returns
    ``(sol, omega_tilde, Omega, mask)``: scipy OdeResult plus the renormalized
    frequency array, the precomputed detuning tensor, and the boolean mask used.
    """
    N = len(A0)

    if include_frequency_shift:
        omega_tilde = omega_arr + eps**2 * omega_shift_notes(A0, C)
    else:
        omega_tilde = omega_arr.copy()

    real_phase = phase_mode == "real"
    Omega = build_Omega_notes(omega_tilde, real_phase=real_phase)

    mask = np.ones((N, N, N, N), dtype=bool)
    if remove_diag:
        mask &= build_nondiag_mask_notes(N)

    t_eval = np.linspace(0.0, tmax, nout)
    sol = solve_ivp(
        rhs_A_notes,
        (0.0, tmax),
        np.asarray(A0, dtype=complex),
        t_eval=t_eval,
        args=(eps, C, Omega, mask, phase_mode),
        method=method,
        rtol=rtol,
        atol=atol,
    )

    return sol, omega_tilde, Omega, mask


def total_action(A):
    """N = sum_lam |A_lam|^2."""
    return np.sum(np.abs(A) ** 2)


def Ndot_notes(t, A, eps, C, Omega, mask=None, phase_mode="real"):
    """Instantaneous d/dt of total action |A|^2 sum, computed from the RHS."""
    if mask is None:
        mask = 1.0
    if phase_mode == "exact":
        phase = np.exp(1j * Omega * t)
    elif phase_mode == "real":
        phase = np.exp(1j * np.real(Omega) * t)
    elif phase_mode == "resonant":
        phase = 1.0
    else:
        raise ValueError("phase_mode must be 'exact', 'real', or 'resonant'.")

    F = np.einsum(
        "lijk,i,j,k->l",
        C * mask * phase,
        A,
        A,
        np.conjugate(A),
        optimize=True,
    )
    Z = np.vdot(A, F)  # sum_l conj(A_l) F_l
    return 2 * eps**2 * np.imag(Z)


def physical_from_A_solution(sol, omega_arr, eps=1.0, include_eps=False):
    """Reconstruct ``a_lam(t) = eps A_lam(t) exp(-i omega_lam t)`` from a solve_ivp result.

    With ``include_eps=False`` (default), returns ``A * exp(-i omega t)`` —
    matches the Mathematica genInterpsWithSoftener convention up to eps.
    """
    t = sol.t
    A = sol.y.T
    soft = np.exp(-1j * omega_arr[None, :] * t[:, None])
    if include_eps:
        return t, eps * A * soft
    return t, A * soft
