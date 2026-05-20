"""Kinetic-equation scaling diagnostics for spherical QNM truncations.

This module assumes the notebook has already built a coupling tensor

    C[lam, i, j, k] = C_{lam i j k}

in the notes convention

    i dA_lam/dt = eps**2 sum C[lam,i,j,k] A_i A_j conj(A_k)
                  exp(i Omega[lam,i,j,k] t).

The routines below do not construct the radial/angular coefficients.  They
take an existing finite tensor and test candidate spectra against the
generalized kinetic equation, including optional anti-Hermitian leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


KernelName = Literal["lorentzian", "finite_time", "gaussian"]


@dataclass(frozen=True)
class ScanResult:
    """One candidate scaling spectrum and its diagnostic scores."""

    p: float
    q: float
    residual: float
    flux_flatness: float | None = None
    flux_mean: float | None = None


def hermitian_adjoint(C: np.ndarray) -> np.ndarray:
    """Return Cdag[lam,i,j,k] = conjugate(C[i,lam,k,j])."""

    return np.conjugate(np.transpose(C, (1, 0, 3, 2)))


def split_hermitian(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a coupling tensor into Hermitian and anti-Hermitian pieces."""

    Cdag = hermitian_adjoint(C)
    C_H = 0.5 * (C + Cdag)
    C_A = 0.5 * (C - Cdag)
    return C_H, C_A


def build_nondiag_mask(N: int) -> np.ndarray:
    """Mask out diagonal nonlinear frequency-shift terms.

    Removes A_q A_lam conj(A_q) and A_lam A_q conj(A_q), matching the
    notebook convention.
    """

    mask = np.ones((N, N, N, N), dtype=bool)

    for lam in range(N):
        for q in range(N):
            mask[lam, q, lam, q] = False
            mask[lam, lam, q, q] = False

    return mask


def build_detuning(omega: np.ndarray) -> np.ndarray:
    """Build Omega[lam,i,j,k] = omega_lam + omega_k - omega_i - omega_j."""

    omega = np.asarray(omega, dtype=float)
    return (
        omega[:, None, None, None]
        - omega[None, :, None, None]
        - omega[None, None, :, None]
        + omega[None, None, None, :]
    )


def resonance_kernel(
    Omega: np.ndarray,
    *,
    width: float | None = None,
    T: float | None = None,
    kind: KernelName = "lorentzian",
) -> np.ndarray:
    """Approximate delta(Omega) for a finite truncation.

    ``lorentzian`` uses delta_eta(x) = (1/pi) eta/(x^2 + eta^2).
    ``gaussian`` uses a normalized Gaussian with standard deviation ``width``.
    ``finite_time`` uses |Delta_T|^2/(2*pi*T), which converges to delta(x).
    """

    Omega = np.asarray(Omega, dtype=float)

    if kind == "finite_time":
        if T is None or T <= 0:
            raise ValueError("finite_time kernel requires T > 0")
        half = 0.5 * Omega * T
        out = np.empty_like(Omega, dtype=float)
        small = np.abs(half) < 1e-8
        out[small] = T / (2.0 * np.pi)
        out[~small] = (
            np.sin(half[~small]) ** 2
            / (np.pi * T * (0.5 * Omega[~small]) ** 2)
        )
        return out

    if width is None or width <= 0:
        raise ValueError(f"{kind} kernel requires width > 0")

    if kind == "lorentzian":
        return (1.0 / np.pi) * width / (Omega**2 + width**2)

    if kind == "gaussian":
        return np.exp(-0.5 * (Omega / width) ** 2) / (
            np.sqrt(2.0 * np.pi) * width
        )

    raise ValueError("kind must be 'lorentzian', 'finite_time', or 'gaussian'")


def principal_value_kernel(
    Omega: np.ndarray,
    *,
    width: float,
) -> np.ndarray:
    """Regularized principal-value kernel for 1/Omega.

    Uses Omega/(Omega^2 + width^2), which suppresses the singular resonant
    core while approaching 1/Omega away from resonance.
    """

    if width <= 0:
        raise ValueError("width must be positive")

    Omega = np.asarray(Omega, dtype=float)
    return Omega / (Omega**2 + width**2)


def power_law_spectrum(
    modes: Iterable[dict],
    p: float,
    q: float = 0.0,
    *,
    amplitude: float = 1.0,
    n_shift: float = 1.0,
    ell_shift: float = 1.0,
    floor: float = 1e-300,
) -> np.ndarray:
    """Return n_lambda = amplitude*(n+n_shift)^(-p)*(ell+ell_shift)^(-q)."""

    vals = []
    for mode in modes:
        n_val = float(mode.get("n", 0)) + n_shift
        ell_val = float(mode.get("l", mode.get("ell", 0))) + ell_shift
        vals.append(amplitude * n_val ** (-p) * ell_val ** (-q))

    return np.maximum(np.asarray(vals, dtype=float), floor)


def kinetic_rhs_general(
    occupation: np.ndarray,
    C: np.ndarray,
    omega: np.ndarray,
    *,
    eps: float = 1.0,
    mask: np.ndarray | None = None,
    kernel: np.ndarray | None = None,
    kernel_width: float | None = None,
    kernel_T: float | None = None,
    kernel_kind: KernelName = "lorentzian",
    include_pv: bool = False,
    pv_width: float | None = None,
) -> np.ndarray:
    """Evaluate the generalized spherical-QNM kinetic RHS.

    The implemented expression is the unsymmetrized version appropriate for
    tensors that may contain an anti-Hermitian part:

        dot n_lam = resonant collision[C] + optional principal-value[C].

    If ``C`` is the Hamiltonian-projected tensor and has the usual quartic
    pair symmetries, this reduces to the familiar gain-loss collision form.
    """

    C = np.asarray(C, dtype=complex)
    occupation = np.asarray(occupation, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if C.ndim != 4 or len(set(C.shape)) != 1:
        raise ValueError("C must have shape (N,N,N,N)")

    N = C.shape[0]
    if occupation.shape != (N,):
        raise ValueError("occupation must have shape (N,)")
    if omega.shape != (N,):
        raise ValueError("omega must have shape (N,)")

    if mask is None:
        mask = np.ones_like(C, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    Omega = build_detuning(omega)

    if kernel is None:
        kernel = resonance_kernel(
            Omega,
            width=kernel_width,
            T=kernel_T,
            kind=kernel_kind,
        )
    else:
        kernel = np.asarray(kernel, dtype=float)

    ni = occupation[:, None, None]
    nj = occupation[None, :, None]
    nk = occupation[None, None, :]

    rhs = np.zeros(N, dtype=float)

    for lam in range(N):
        n_lam = occupation[lam]
        C_l = C[lam]
        M_l = mask[lam]

        # Shapes are arranged as [i,j,k] to match C_l.
        C_barred_pair = np.conjugate(np.transpose(C[:, :, :, lam], (1, 2, 0)))
        C_loss_i = np.transpose(C[:, lam, :, :], (0, 2, 1))
        C_loss_j = np.transpose(C[:, lam, :, :], (2, 0, 1))

        gain_1 = np.abs(C_l) ** 2 * ni * nj * nk
        gain_2_product = C_l * C_barred_pair
        loss_i_product = C_l * C_loss_i
        loss_j_product = C_l * C_loss_j

        bracket_complex = (
            gain_2_product * n_lam * ni * nj
            - loss_i_product * n_lam * nj * nk
            - loss_j_product * n_lam * ni * nk
        )

        resonant_bracket = (
            gain_1
            + np.real(gain_2_product) * n_lam * ni * nj
            - np.real(loss_i_product) * n_lam * nj * nk
            - np.real(loss_j_product) * n_lam * ni * nk
        )

        rhs[lam] = 4.0 * np.pi * eps**4 * np.sum(
            M_l * kernel[lam] * resonant_bracket
        )

        if include_pv:
            width = pv_width if pv_width is not None else kernel_width
            if width is None or width <= 0:
                raise ValueError("include_pv=True requires pv_width or kernel_width")
            pv = principal_value_kernel(Omega[lam], width=width)
            rhs[lam] -= 4.0 * eps**4 * np.sum(
                M_l * pv * np.imag(bracket_complex)
            )

    return rhs


def kinetic_rhs_hamiltonian_gain_loss(
    occupation: np.ndarray,
    C_H: np.ndarray,
    omega: np.ndarray,
    *,
    eps: float = 1.0,
    mask: np.ndarray | None = None,
    kernel: np.ndarray | None = None,
    kernel_width: float | None = None,
    kernel_T: float | None = None,
    kernel_kind: KernelName = "lorentzian",
) -> np.ndarray:
    """Evaluate the conservative gain-loss form for a Hamiltonian tensor."""

    C_H = np.asarray(C_H, dtype=complex)
    occupation = np.asarray(occupation, dtype=float)
    omega = np.asarray(omega, dtype=float)

    N = C_H.shape[0]
    if mask is None:
        mask = np.ones_like(C_H, dtype=bool)

    Omega = build_detuning(omega)
    if kernel is None:
        kernel = resonance_kernel(
            Omega,
            width=kernel_width,
            T=kernel_T,
            kind=kernel_kind,
        )

    rhs = np.zeros(N, dtype=float)
    n = occupation

    for lam in range(N):
        ni = n[:, None, None]
        nj = n[None, :, None]
        nk = n[None, None, :]
        n_lam = n[lam]

        gain_loss = (
            n_lam
            * ni
            * nj
            * nk
            * (1.0 / n_lam + 1.0 / nk - 1.0 / ni - 1.0 / nj)
        )

        rhs[lam] = 4.0 * np.pi * eps**4 * np.sum(
            mask[lam]
            * kernel[lam]
            * np.abs(C_H[lam]) ** 2
            * gain_loss
        )

    return rhs


def stationary_residual(
    rhs: np.ndarray,
    occupation: np.ndarray,
    *,
    relative: bool = True,
    floor: float = 1e-300,
) -> float:
    """Score how close a candidate spectrum is to stationary."""

    rhs = np.asarray(rhs, dtype=float)
    occupation = np.asarray(occupation, dtype=float)

    if relative:
        scaled = rhs / np.maximum(np.abs(occupation), floor)
    else:
        scaled = rhs

    return float(np.sqrt(np.mean(scaled**2)))


def shell_flux(
    rhs: np.ndarray,
    modes: Iterable[dict],
    *,
    shell: Literal["n", "ell", "n_plus_ell"] = "n",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative flux through integer shells.

    The returned flux is -sum_{s <= shell_value} dot n_s, i.e. positive flux
    means occupation leaves the low-shell region.
    """

    rhs = np.asarray(rhs, dtype=float)
    shell_values = []

    for mode in modes:
        n_val = int(mode.get("n", 0))
        ell_val = int(mode.get("l", mode.get("ell", 0)))
        if shell == "n":
            shell_values.append(n_val)
        elif shell == "ell":
            shell_values.append(ell_val)
        elif shell == "n_plus_ell":
            shell_values.append(n_val + ell_val)
        else:
            raise ValueError("shell must be 'n', 'ell', or 'n_plus_ell'")

    shell_values = np.asarray(shell_values, dtype=int)
    unique = np.unique(shell_values)
    flux = np.zeros_like(unique, dtype=float)

    for idx, value in enumerate(unique):
        flux[idx] = -float(np.sum(rhs[shell_values <= value]))

    return unique, flux


def flux_flatness_score(flux: np.ndarray, *, floor: float = 1e-300) -> tuple[float, float]:
    """Return coefficient of variation and mean absolute flux."""

    flux = np.asarray(flux, dtype=float)
    if flux.size == 0:
        return float("inf"), 0.0

    mean_abs = float(np.mean(np.abs(flux)))
    if mean_abs < floor:
        return float("inf"), mean_abs

    return float(np.std(flux) / mean_abs), mean_abs


def scan_power_laws(
    modes: list[dict],
    C: np.ndarray,
    omega: np.ndarray,
    p_values: Iterable[float],
    q_values: Iterable[float],
    *,
    eps: float = 1.0,
    eta: float | None = None,
    use_hamiltonian_gain_loss: bool = False,
    mask: np.ndarray | None = None,
    kernel_width: float | None = None,
    kernel_T: float | None = None,
    kernel_kind: KernelName = "lorentzian",
    include_pv: bool = False,
    pv_width: float | None = None,
    shell: Literal["n", "ell", "n_plus_ell"] | None = None,
    amplitude: float = 1.0,
    sort_by: Literal["residual", "flux_flatness"] = "residual",
) -> list[ScanResult]:
    """Scan power-law exponents and sort by stationary residual.

    If ``eta`` is provided, the tensor is replaced by ``C_H + eta*C_A``.
    If ``use_hamiltonian_gain_loss`` is true, the conservative gain-loss
    formula is used instead of the generalized unsymmetrized expression.
    """

    if eta is not None:
        C_H, C_A = split_hermitian(C)
        C_eff = C_H + eta * C_A
    else:
        C_eff = C

    N = C_eff.shape[0]
    if mask is None:
        mask = build_nondiag_mask(N)

    Omega = build_detuning(np.real(omega))
    kernel = resonance_kernel(
        Omega,
        width=kernel_width,
        T=kernel_T,
        kind=kernel_kind,
    )

    results: list[ScanResult] = []

    for p in p_values:
        for q in q_values:
            occupation = power_law_spectrum(modes, p, q, amplitude=amplitude)

            if use_hamiltonian_gain_loss:
                rhs = kinetic_rhs_hamiltonian_gain_loss(
                    occupation,
                    C_eff,
                    np.real(omega),
                    eps=eps,
                    mask=mask,
                    kernel=kernel,
                )
            else:
                rhs = kinetic_rhs_general(
                    occupation,
                    C_eff,
                    np.real(omega),
                    eps=eps,
                    mask=mask,
                    kernel=kernel,
                    include_pv=include_pv,
                    pv_width=pv_width if pv_width is not None else kernel_width,
                )

            residual = stationary_residual(rhs, occupation)

            flatness = None
            mean_flux = None
            if shell is not None:
                _, flux = shell_flux(rhs, modes, shell=shell)
                flatness, mean_flux = flux_flatness_score(flux)

            results.append(
                ScanResult(
                    p=float(p),
                    q=float(q),
                    residual=residual,
                    flux_flatness=flatness,
                    flux_mean=mean_flux,
                )
            )

    if sort_by == "residual":
        return sorted(results, key=lambda item: item.residual)

    if sort_by == "flux_flatness":
        return sorted(
            results,
            key=lambda item: (
                float("inf") if item.flux_flatness is None else item.flux_flatness
            ),
        )

    raise ValueError("sort_by must be 'residual' or 'flux_flatness'")
