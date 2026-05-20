"""Finite-difference convergence on smooth test functions."""
import numpy as np
import pytest

from rn_cwt_nr import UniformGrid, fd4_d1, fd4_d2


@pytest.mark.parametrize("N", [21, 41, 81, 161, 321])
def test_fd4_d1_converges_4th_order(N):
    """4th-order centered FD on sin(2πx) should converge as N^{-4} in interior."""
    grid = UniformGrid(0.0, 1.0, N)
    x = grid.x
    u = np.sin(2 * np.pi * x)
    du_true = 2 * np.pi * np.cos(2 * np.pi * x)
    du = fd4_d1(u, grid.dx)
    # Use interior-only max norm to isolate the 4th-order behavior; boundary
    # stencils are also 4th-order for d/dx but with bigger constants.
    err = np.max(np.abs((du - du_true)[2:-2]))
    # For 4th-order err ≈ C * (1/N)^4. For sin(2πx) the leading-term constant
    # is (2π)^5 / 30 ≈ 326; allow ~3x headroom.
    assert err * (N - 1) ** 4 < 1000, f"N={N} err={err:.3e} err*(N-1)^4={err*(N-1)**4:.3e}"


def test_fd4_d1_order_is_4():
    """Check the empirical convergence rate is close to 4."""
    Ns = [41, 81, 161, 321]
    errs = []
    for N in Ns:
        grid = UniformGrid(0.0, 1.0, N)
        x = grid.x
        u = np.sin(2 * np.pi * x)
        du_true = 2 * np.pi * np.cos(2 * np.pi * x)
        du = fd4_d1(u, grid.dx)
        errs.append(np.max(np.abs((du - du_true)[2:-2])))
    # Halving dx should reduce error by 16 (= 2^4).
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    avg = np.mean(rates)
    assert 3.7 < avg < 4.3, f"convergence rate = {avg:.3f}, expected ~4"


def test_fd4_d2_converges_high_order_in_interior():
    """4th-order centered FD for d²/dx² on cos(2πx)."""
    Ns = [41, 81, 161, 321]
    errs = []
    for N in Ns:
        grid = UniformGrid(0.0, 1.0, N)
        x = grid.x
        u = np.cos(2 * np.pi * x)
        d2u_true = -(2 * np.pi) ** 2 * np.cos(2 * np.pi * x)
        d2u = fd4_d2(u, grid.dx)
        errs.append(np.max(np.abs((d2u - d2u_true)[2:-2])))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    avg = np.mean(rates)
    assert 3.7 < avg < 4.3, f"interior d² convergence rate = {avg:.3f}, expected ~4"


def test_grid_dx_and_x():
    grid = UniformGrid(0.5, 2.5, 11)
    assert np.isclose(grid.dx, 0.2)
    assert np.isclose(grid.x[0], 0.5) and np.isclose(grid.x[-1], 2.5)
    assert grid.x.shape == (11,)
