"""Bilinear-form projection of (ψ, π) on a Cauchy slice onto near-zone QNMs.

Implements the paper's bilinear form (§3, eq. 89):

    ⟨⟨B, C⟩⟩ = i lim_{R→∞} { ∫_{-R}^R (B C̃ + C B̃) dr_* + B(-R)C(-R) + B(R)C(R) }

with C̃ = (∂_t - i q A_t) C the gauge-covariant time derivative. The form is
**symmetric** (not conjugate-symmetric), and the boundary point contributions
B(±R)C(±R) are the regulator that cancels the divergent bulk integral as
R → ∞ — modes diverge at the horizon side as b_n ~ x̆^{s_n} with
Re(s_n) < 0.

For a NHERN PDE on a bounded slab r_* ∈ [r_*,min, r_*,max] we evaluate the
same formula on the slab — no limit needed, just bulk integral + the two
boundary point contributions. The bulk/boundary cancellation is approximate
on a finite grid but the form is conserved on solutions of the linear PDE,
so the recovered A_n(t) should track A_n(0) e^{s_n t} regardless of where
r_*,min and r_*,max sit, as long as the modes are resolved.

Projection (paper §3, eq. ~95):

    A_n = ⟨⟨b_n, ψ⟩⟩ / ⟨⟨b_n, b_n⟩⟩

We solve the Gram-matrix system G A = c with G[m,n] = ⟨⟨b_m, b_n⟩⟩ and
c[m] = ⟨⟨b_m, ψ⟩⟩ instead of dividing by ‖b_n‖² alone, so the recovery is
correct even when the b_n are not exactly orthogonal on the finite grid.

In NHERN, A_t = x̆ (see paper §10 eq. 2281).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.integrate import trapezoid

from rn_cwt import b_near, s

from .grid import UniformGrid


def bilinear_form(
    B: np.ndarray,
    B_tilde: np.ndarray,
    C: np.ndarray,
    C_tilde: np.ndarray,
    grid: UniformGrid,
) -> complex:
    """Paper's symmetric bilinear form ⟨⟨B, C⟩⟩ on a finite r_*-grid.

    ⟨⟨B, C⟩⟩ = i { ∫_{r_*,min}^{r_*,max} (B C̃ + C B̃) dr_* + B(r_*,min)C(r_*,min) + B(r_*,max)C(r_*,max) }

    Bulk integrated by trapezoidal rule on the uniform r_*-grid. Boundary
    point contributions taken at the grid endpoints — these are the
    regulator that makes the form finite for QNM modes that diverge at the
    asymptotic boundaries.
    """
    integrand = B * C_tilde + C * B_tilde
    bulk = trapezoid(integrand, dx=grid.dx)
    boundary = B[0] * C[0] + B[-1] * C[-1]
    return 1j * (bulk + boundary)


def build_qnm_basis(
    modes: Sequence[tuple[int, int]],
    xbreve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the QNM basis arrays needed for bilinear-form projection.

    Returns
    -------
    bs : (M, N) complex      — b_n(x̆) for each mode
    sns : (M,) complex       — s_n for each mode (eigenvalue of ∂_t)
    b_tildes : (M, N) complex — b̃_n = (s_n - i q A_t) b_n with A_t = x̆
    """
    bs = np.array([b_near(n, ell, xbreve) for (n, ell) in modes], dtype=complex)
    sns = np.array([s(n, ell) for (n, ell) in modes], dtype=complex)
    return bs, sns


def project_qnm(
    psi: np.ndarray,
    pi: np.ndarray,
    xbreve: np.ndarray,
    modes: Sequence[tuple[int, int]],
    grid: UniformGrid,
    q: float,
) -> np.ndarray:
    """Bilinear-form projection of the Cauchy-slice data (ψ, π) onto QNMs.

    Per the paper's two-component formalism: the spatial part of the QNM
    expansion is ψ = Σ_n A_n b_n, and the time-derivative-like component is
    π = ∂_t ψ. The bilinear form picks out

        A_n = ⟨⟨b_n, ψ⟩⟩ / ⟨⟨b_n, b_n⟩⟩

    Implemented as a Gram-matrix solve G A = c so the recovery is exact for
    ψ ∈ span{b_n} regardless of basis non-orthogonality on the finite grid.

    Parameters
    ----------
    psi, pi : 1D (N,) or 2D (nout, N) complex arrays
        Cauchy-slice field and its time derivative on the grid.
    xbreve : (N,)
        x̆ values at the grid points (= xbreve_of_rstar(grid.x)).
    modes : sequence of (n, ℓ) tuples
    grid : UniformGrid in r_*
    q : scalar charge (in r_+ = 1 units)

    Returns
    -------
    A : (M,) complex (1D input) or (nout, M) complex (2D input)
        Mode amplitudes such that ψ ≈ Σ_n A_n b_n.
    """
    psi_arr = np.asarray(psi, dtype=complex)
    pi_arr = np.asarray(pi, dtype=complex)
    A_t = xbreve  # NHERN gauge: A_{t̆} = x̆

    bs, sns = build_qnm_basis(modes, xbreve)
    # b̃_n = (s_n - i q A_t) b_n
    b_tildes = (sns[:, None] - 1j * q * A_t[None, :]) * bs

    M = len(modes)

    # Gram matrix G[m, n] = ⟨⟨b_m, b_n⟩⟩
    G = np.zeros((M, M), dtype=complex)
    for m in range(M):
        for n in range(M):
            G[m, n] = bilinear_form(bs[m], b_tildes[m], bs[n], b_tildes[n], grid)

    def _project_slice(psi_s, pi_s):
        psi_tilde = pi_s - 1j * q * A_t * psi_s
        c = np.array([
            bilinear_form(bs[m], b_tildes[m], psi_s, psi_tilde, grid)
            for m in range(M)
        ])
        return np.linalg.solve(G, c)

    if psi_arr.ndim == 1:
        return _project_slice(psi_arr, pi_arr)
    if psi_arr.ndim == 2:
        nout = psi_arr.shape[0]
        A = np.zeros((nout, M), dtype=complex)
        for k in range(nout):
            A[k] = _project_slice(psi_arr[k], pi_arr[k])
        return A
    raise ValueError(f"ψ must be 1D or 2D, got shape {psi_arr.shape}")
