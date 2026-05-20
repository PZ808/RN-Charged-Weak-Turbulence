"""rn_cwt_nr — numerical-relativity validation for rn_cwt.

1+1D PDE solver for the charged scalar on the near-extremal Reissner-Nordström
near-horizon throat (NHERN = AdS_2 x S^2 with constant electric field), Klein-Gordon
symplectic projection on Cauchy slices, and PDE↔ODE comparison for the
amplitude system in rn_cwt.

**Phase 1 v2 (current)** — linear charged-scalar KG, l=m=0 sector, in NHERN
near-zone coordinates (x̆, t̆) and tortoise r_* = ln[x̆/(x̆+1)], bounded slab
r_* ∈ [r_*,min, r_*,max] with Dirichlet BCs. Method-of-lines with 4th-order
centered finite differences in r_* + DOP853 in time.

Conventions and equation follow §10 of the draft paper (matched-asymptotic
near-zone for a charged scalar on near-extremal RN).

Depends on ``rn_cwt`` for background parameters, mode frequencies, and
convention metadata. ``rn_cwt`` does not depend on ``rn_cwt_nr``.
"""
from .coords import (
    compact_spatial_coeffs,
    dxbreve_drstar,
    dy_drstar,
    f_of_xbreve,
    rstar_of_xbreve,
    rstar_of_y,
    xbreve_of_rstar,
    y_of_rstar,
)
from .grid import UniformGrid, fd4_d1, fd4_d2
from .pde import (
    build_state,
    kg_rhs,
    kg_rhs_compact,
    split_state,
)
from .projection import build_qnm_basis, project_qnm
from .solver import evolve_kg, evolve_kg_compact

__all__ = [
    "UniformGrid",
    "fd4_d1",
    "fd4_d2",
    # NHERN tortoise
    "xbreve_of_rstar",
    "rstar_of_xbreve",
    "f_of_xbreve",
    "dxbreve_drstar",
    # tanh compactification
    "y_of_rstar",
    "rstar_of_y",
    "dy_drstar",
    "compact_spatial_coeffs",
    # PDE RHS
    "kg_rhs",
    "kg_rhs_compact",
    "build_state",
    "split_state",
    # solvers
    "evolve_kg",
    "evolve_kg_compact",
    # projection
    "build_qnm_basis",
    "project_qnm",
]
