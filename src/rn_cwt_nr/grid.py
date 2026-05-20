"""1D uniform grid + 4th-order centered finite differences.

Interior uses standard 5-point centered stencils for d/dx and d²/dx²:

    u'_i  ≈ (-u[i+2] + 8 u[i+1] - 8 u[i-1] + u[i-2]) / (12 dx)
    u''_i ≈ (-u[i+2] + 16 u[i+1] - 30 u[i] + 16 u[i-1] - u[i-2]) / (12 dx²)

Boundaries use 5-point one-sided stencils. The first-derivative one-sided
stencils are 4th-order accurate; the second-derivative one-sided stencils are
only 3rd-order accurate (a 4th-order one-sided d² would need 6 points). This is
sufficient when the interior is the dominant error contribution; for stronger
boundary accuracy, switch to a 6-point one-sided d² stencil later.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformGrid:
    """Uniform grid x[0..N-1] with x[0] = x_min, x[N-1] = x_max."""
    x_min: float
    x_max: float
    N: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.N - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.N)


def fd4_d1(u: np.ndarray, dx: float) -> np.ndarray:
    """4th-order centered first derivative; biased 4th-order stencils at boundaries."""
    n = u.shape[-1]
    if n < 5:
        raise ValueError(f"Grid too small for 4th-order stencil (N={n})")

    du = np.empty_like(u)
    # Interior 5-point centered
    du[..., 2:-2] = (
        -u[..., 4:] + 8 * u[..., 3:-1] - 8 * u[..., 1:-3] + u[..., :-4]
    ) / (12 * dx)
    # Left boundary, 4th-order one-sided
    du[..., 0] = (
        -25 * u[..., 0] + 48 * u[..., 1] - 36 * u[..., 2]
        + 16 * u[..., 3] - 3 * u[..., 4]
    ) / (12 * dx)
    du[..., 1] = (
        -3 * u[..., 0] - 10 * u[..., 1] + 18 * u[..., 2]
        - 6 * u[..., 3] + u[..., 4]
    ) / (12 * dx)
    # Right boundary, 4th-order one-sided
    du[..., -2] = (
        3 * u[..., -1] + 10 * u[..., -2] - 18 * u[..., -3]
        + 6 * u[..., -4] - u[..., -5]
    ) / (12 * dx)
    du[..., -1] = (
        25 * u[..., -1] - 48 * u[..., -2] + 36 * u[..., -3]
        - 16 * u[..., -4] + 3 * u[..., -5]
    ) / (12 * dx)
    return du


def fd4_d2(u: np.ndarray, dx: float) -> np.ndarray:
    """4th-order centered second derivative; 3rd-order one-sided at boundaries.

    Note: the boundary stencils are 5-point and only 3rd-order accurate for d².
    """
    n = u.shape[-1]
    if n < 5:
        raise ValueError(f"Grid too small for 4th-order stencil (N={n})")

    d2u = np.empty_like(u)
    # Interior 5-point centered
    d2u[..., 2:-2] = (
        -u[..., 4:] + 16 * u[..., 3:-1] - 30 * u[..., 2:-2]
        + 16 * u[..., 1:-3] - u[..., :-4]
    ) / (12 * dx**2)
    # Boundaries — 5-point one-sided (3rd-order for d²)
    d2u[..., 0] = (
        35 * u[..., 0] - 104 * u[..., 1] + 114 * u[..., 2]
        - 56 * u[..., 3] + 11 * u[..., 4]
    ) / (12 * dx**2)
    d2u[..., 1] = (
        11 * u[..., 0] - 20 * u[..., 1] + 6 * u[..., 2]
        + 4 * u[..., 3] - u[..., 4]
    ) / (12 * dx**2)
    d2u[..., -2] = (
        -u[..., -5] + 4 * u[..., -4] + 6 * u[..., -3]
        - 20 * u[..., -2] + 11 * u[..., -1]
    ) / (12 * dx**2)
    d2u[..., -1] = (
        11 * u[..., -5] - 56 * u[..., -4] + 114 * u[..., -3]
        - 104 * u[..., -2] + 35 * u[..., -1]
    ) / (12 * dx**2)
    return d2u
