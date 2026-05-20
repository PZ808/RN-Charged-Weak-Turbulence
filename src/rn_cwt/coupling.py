"""Quartic mode-coupling tensor C and Hermitian-projection diagnostics.

The amplitude equation is

    i dot A_lam = eps^2 sum_{i,j,k} C[lam, i, j, k] A_i A_j bar(A_k)
                  * exp(i Omega[lam, i, j, k] t),

with the barred dynamical index k LAST (notes convention). Internally,
``S_near_abcd`` puts the barred slot SECOND — bridged by the swap

    S_notes(i, j, k, lam) = S_near_abcd(i, k, j, lam).
"""
import numpy as np

from .angular import Q_element
from .radial import S_near_abcd


def Q_notes(modes, i, j, k, lam):
    """Q^{ij}_{k lam} = int Y_i Y_j bar(Y_k) bar(Y_lam) dOmega — notes convention."""
    mi, mj, mk, ml = modes[i], modes[j], modes[k], modes[lam]
    return Q_element(
        mi["l"], mi.get("m", 0),
        mj["l"], mj.get("m", 0),
        mk["l"], mk.get("m", 0),
        ml["l"], ml.get("m", 0),
    )


def S_notes(modes, i, j, k, lam):
    """S^{ij lam}_k ~ b_i b_j bar(b_k) b_lam — notes convention.

    Internally calls S_near_abcd(modes[i], modes[k], modes[j], modes[lam]).
    """
    return S_near_abcd(modes[i], modes[k], modes[j], modes[lam])


def coupling_entry(modes, lam, i, j, k, alpha=1.0):
    """C[lam, i, j, k] = alpha * Q_notes(i,j,k,lam) * S_near_abcd(i,k,j,lam)."""
    return (
        alpha
        * Q_notes(modes, i, j, k, lam)
        * S_notes(modes, i, j, k, lam)
    )


def build_C_notes(modes, alpha=1.0):
    """Assemble the rank-4 quartic coupling tensor in the notes convention.

    Returns C[lam, i, j, k] of shape (N, N, N, N), complex.
    """
    N = len(modes)
    C = np.zeros((N, N, N, N), dtype=complex)
    for lam in range(N):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[lam, i, j, k] = coupling_entry(modes, lam, i, j, k, alpha=alpha)
    return C


# Alias for legacy notebook cells that imported `build_coupling_tensor`.
build_coupling_tensor = build_C_notes


def build_C_notes_cached(modes, table, alpha=1.0):
    """Same as ``build_C_notes`` but uses Peter's precomputed S-table.

    For radial coefficients found in ``table`` (a dict from
    ``rn_cwt.load_S_table``), uses the precomputed value; otherwise falls
    back to the slow Python ``S_near_abcd`` evaluation. For large mode sets
    this is orders of magnitude faster than the bare ``build_C_notes``.
    """
    from .precomputed import S_near_cached

    N = len(modes)
    C = np.zeros((N, N, N, N), dtype=complex)
    for lam in range(N):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    Q = Q_notes(modes, i, j, k, lam)
                    S = S_near_cached(
                        table,
                        modes[i],
                        modes[k],
                        modes[j],
                        modes[lam],
                    )
                    C[lam, i, j, k] = alpha * Q * S
    return C


def hermitian_adjoint_notes(C):
    """Cdag[lam, i, j, k] = conj(C[i, lam, k, j]).

    The notes-convention adjoint: swap first pair, swap last pair, conjugate.
    Implemented as np.conjugate(np.transpose(C, (1, 0, 3, 2))).
    """
    return np.conjugate(np.transpose(C, (1, 0, 3, 2)))


def coupling_hermitian_defect(C, mask=None):
    """Return ``(||C_A|| / ||C_H||, ||C_H||, ||C_A||)`` where

        C_H = 0.5 * (C + Cdag),    C_A = 0.5 * (C - Cdag).

    Restricted to ``mask`` if provided (boolean array of same shape as C).
    """
    if mask is None:
        mask = np.ones_like(C, dtype=bool)
    Cdag = hermitian_adjoint_notes(C)
    CH = 0.5 * (C + Cdag)
    CA = 0.5 * (C - Cdag)
    norm_H = np.linalg.norm(CH[mask])
    norm_A = np.linalg.norm(CA[mask])
    return norm_A / norm_H, norm_H, norm_A
