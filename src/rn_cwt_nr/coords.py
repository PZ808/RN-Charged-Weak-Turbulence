"""Coordinate utilities for the NHERN throat in near-zone variables.

Following §10 of the draft paper (the matched-asymptotic-expansion near-zone),
the metric on the NHERN throat in dimensionless near-zone coordinates is

    ds^2 = -f d{t̆}^2 + d{x̆}^2 / f + r_+^2 dΩ^2,       f = x̆ (x̆ + 1)
    A_{t̆} = x̆

where x̆ = (r - r_+)/(r_+ σ) is the near-zone radial coordinate and
{t̆} = σ t / r_+ the near-zone time. We work in units r_+ = 1, σ = 1.

The Poincaré horizon sits at x̆ = 0 and the matching surface to the far zone
("AdS conformal boundary" of the throat) at x̆ → ∞.

The associated tortoise coordinate is

    r_* = ln[ x̆ / (x̆ + 1) ],     dr_*/dx̆ = 1/f,

with r_* ∈ (-∞, 0): r_* → -∞ at the horizon, r_* → 0⁻ at the throat boundary.
This module exposes the bijection x̆ ↔ r_* and the Jacobian dx̆/dr_* = f.
"""
from __future__ import annotations

import numpy as np


def xbreve_of_rstar(rstar: np.ndarray | float) -> np.ndarray | float:
    """x̆(r_*) = e^{r_*} / (1 - e^{r_*}), valid for r_* < 0."""
    er = np.exp(rstar)
    return er / (1.0 - er)


def rstar_of_xbreve(xbreve: np.ndarray | float) -> np.ndarray | float:
    """r_*(x̆) = ln[ x̆ / (x̆ + 1) ], valid for x̆ > 0."""
    return np.log(xbreve / (xbreve + 1.0))


def f_of_xbreve(xbreve: np.ndarray | float) -> np.ndarray | float:
    """NHERN lapse-like function f = x̆ (x̆ + 1)."""
    return xbreve * (xbreve + 1.0)


def dxbreve_drstar(xbreve: np.ndarray | float) -> np.ndarray | float:
    """dx̆/dr_* = x̆ (x̆ + 1) = f."""
    return f_of_xbreve(xbreve)


# --------------------------------------------------------------------------
# Tanh compactification of r_* — y = tanh(r_*/L), y ∈ (-1, 0) for r_* ∈ (-∞, 0).
#
# Horizon (r_* → -∞) maps to y → -1; throat boundary (r_* → 0⁻) maps to y → 0⁻.
# The compactified spatial operator picks up Jacobian factors:
#     ∂_{r_*} = ((1 - y²)/L) ∂_y
#     ∂²_{r_*} = ((1 - y²)²/L²) ∂²_y - (2 y (1 - y²)/L²) ∂_y.
# --------------------------------------------------------------------------

def y_of_rstar(rstar: np.ndarray | float, L: float = 1.0) -> np.ndarray | float:
    """y = tanh(r_*/L), bounded compactification of r_* ∈ (-∞, 0) → y ∈ (-1, 0)."""
    return np.tanh(rstar / L)


def rstar_of_y(y: np.ndarray | float, L: float = 1.0) -> np.ndarray | float:
    """r_* = L · arctanh(y), valid for y ∈ (-1, 1)."""
    return L * np.arctanh(y)


def dy_drstar(y: np.ndarray | float, L: float = 1.0) -> np.ndarray | float:
    """dy/dr_* = (1 - y²)/L."""
    return (1.0 - y**2) / L


def compact_spatial_coeffs(y: np.ndarray, L: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (a(y), b(y)) such that ∂²_{r_*} ψ = a(y) ∂²_y ψ + b(y) ∂_y ψ.

        a(y) = (1 - y²)² / L²
        b(y) = -2 y (1 - y²) / L²
    """
    one_minus_y2 = 1.0 - y**2
    a = (one_minus_y2 / L) ** 2
    b = -2.0 * y * one_minus_y2 / L**2
    return a, b
