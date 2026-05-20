"""Method-of-lines wrapper for the charged-scalar KG (linear by default, optional cubic nonlinearity via ``alpha``) PDE on NHERN."""
import numpy as np
from scipy.integrate import solve_ivp

from .coords import compact_spatial_coeffs, rstar_of_y, xbreve_of_rstar
from .grid import UniformGrid
from .pde import build_state, kg_rhs, kg_rhs_compact


def evolve_kg(
    grid: UniformGrid,
    psi0: np.ndarray,
    pi0: np.ndarray,
    tmax: float,
    *,
    q: float,
    m_eff_sq: float = 0.0,
    alpha: float = 0.0,
    nout: int = 200,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    horizon_bc: str = "outgoing",
    throat_bc: str = "dirichlet",
):
    """Evolve the charged-scalar KG (linear by default, optional cubic nonlinearity via ``alpha``) on the NHERN throat.

    Grid is in NHERN tortoise r_* ∈ (r_*,min, r_*,max), both endpoints strictly
    negative. x̆ values are computed internally from r_*.

    Parameters
    ----------
    grid : UniformGrid in r_*.
    psi0, pi0 : (N,) complex arrays. Initial data for ψ and π = ∂_t ψ.
    tmax : final time (in dimensionless near-zone time t̆ = σ t / r_+).
    q : scalar charge in r_+ = 1 units.
    m_eff_sq : effective mass-squared on the AdS_2 throat, = ℓ(ℓ+1).
    nout : number of output time steps.
    method, rtol, atol : passed to ``scipy.integrate.solve_ivp``.
    horizon_bc : {"outgoing" (default), "dirichlet"}
        BC at the horizon side i=0. "outgoing" is the Sommerfeld condition
        ∂_t ψ = +∂_{r_*} ψ, exact in the x̆→0 limit and the physical choice
        for QNM evolution. "dirichlet" pins ψ at i=0 (causes reflection;
        useful only for tests that need a causal-isolation window).
    throat_bc : {"dirichlet" (default)}
        BC at the throat side i=N-1.

    Returns
    -------
    t : (nout,) array of t̆ values.
    psi : (nout, N) complex array of ψ(t̆, r_*).
    pi  : (nout, N) complex array of π(t̆, r_*) = ∂_{t̆} ψ.
    sol : raw scipy OdeResult.
    """
    rstar = grid.x
    xbreve = xbreve_of_rstar(rstar)
    N = grid.N
    y0 = build_state(psi0, pi0)

    t_eval = np.linspace(0.0, tmax, nout)
    sol = solve_ivp(
        kg_rhs,
        (0.0, tmax),
        y0,
        t_eval=t_eval,
        args=(grid, xbreve, q, m_eff_sq, alpha, horizon_bc, throat_bc),
        method=method,
        rtol=rtol,
        atol=atol,
    )

    Y = sol.y.T
    psi = Y[:, :N]
    pi = Y[:, N:]
    return sol.t, psi, pi, sol


def evolve_kg_compact(
    grid: UniformGrid,
    psi0: np.ndarray,
    pi0: np.ndarray,
    tmax: float,
    *,
    q: float,
    m_eff_sq: float = 0.0,
    L: float = 1.0,
    nout: int = 200,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Evolve the charged-scalar KG (linear by default, optional cubic nonlinearity via ``alpha``) on the NHERN throat in compactified
    coordinate y = tanh(r_*/L), y ∈ (-1, 0).

    Grid is uniform in y with both endpoints strictly in (-1, 0). x̆ and the
    Jacobian-modified spatial-derivative coefficients are computed internally
    from grid.x and L.

    Parameters
    ----------
    grid : UniformGrid in y.
    psi0, pi0 : (N,) complex arrays. Initial data on the y-grid.
    tmax : final time (in t̆).
    q, m_eff_sq : same physical parameters as ``evolve_kg``.
    L : compactification length scale. Default 1.0.
    nout, method, rtol, atol : passed to solve_ivp.

    Returns ``(t, psi, pi, sol)`` analogous to ``evolve_kg``.
    """
    y = grid.x
    rstar = rstar_of_y(y, L=L)
    xbreve = xbreve_of_rstar(rstar)
    a_coeff, b_coeff = compact_spatial_coeffs(y, L=L)
    N = grid.N
    y0 = build_state(psi0, pi0)

    t_eval = np.linspace(0.0, tmax, nout)
    sol = solve_ivp(
        kg_rhs_compact,
        (0.0, tmax),
        y0,
        t_eval=t_eval,
        args=(grid, xbreve, q, m_eff_sq, a_coeff, b_coeff),
        method=method,
        rtol=rtol,
        atol=atol,
    )

    Y = sol.y.T
    psi = Y[:, :N]
    pi = Y[:, N:]
    return sol.t, psi, pi, sol
