"""rn_cwt — RN Charged Weak Turbulence.

Quartic mode-coupling tensor and interaction-picture amplitude evolution
for a charged scalar field on near-extremal Reissner-Nordström.

**Conventions** (see README "Current model"):

- Notes convention: barred dynamical index LAST. C[lam, i, j, k] with bar on A_k. So

      i dot A_lam = eps^2 sum_{ijk} C[lam, i, j, k] A_i A_j bar(A_k) e^{i Omega t}.

- Internal ``S_near_abcd(a, b, c, d)`` has the barred slot SECOND.

- Bridged by ``S_notes(i, j, k, lam) = S_near_abcd(i, k, j, lam)``.

- Notes-convention Hermitian adjoint:
  ``Cdag[lam, i, j, k] = conj(C[i, lam, k, j])`` — transpose (1,0,3,2) + conjugate.
"""

from .background import (
    Q,
    bar,
    h,
    h0,
    iq,
    poch,
    q,
    rp,
    s,
    w,
)
from .angular import (
    Q_element,
    W3j,
    gaunt,
)
from .radial import (
    Anorm,
    b_near,
    C_abcd,
    P,
    P2,
    P2LM,
    P2LMn,
    P3,
    P3LM,
    P3LMn,
    P4,
    P4LM,
    P4LMn,
    P4LMn_as_C_abcd,
    PLM,
    PLM_cached,
    S0,
    SLM,
    S_near_abcd,
    S_near_paper,
    Scomp,
    ScompLM,
    Scomp_abcd,
    create_SLM_table,
    create_SLM_table_optimized,
    gamma_abcd,
)
from .coupling import (
    Q_notes,
    S_notes,
    build_C_notes,
    build_C_notes_cached,
    build_coupling_tensor,
    coupling_entry,
    coupling_hermitian_defect,
    hermitian_adjoint_notes,
)
from .evolution import (
    Ndot_notes,
    build_Omega_notes,
    build_nondiag_mask_notes,
    build_omega_arr,
    integrate_A_notes,
    omega_mode,
    omega_shift_notes,
    physical_from_A_solution,
    rhs_A_notes,
    total_action,
)
from .diagnostics import (
    inspect_solution,
    mode_label,
)
from .precomputed import (
    S_near_cached,
    S_near_lookup,
    load_S_table,
    parse_S_table_from_mathematica,
)

__all__ = [
    # background
    "Q", "bar", "h", "h0", "iq", "poch", "q", "rp", "s", "w",
    # angular
    "Q_element", "W3j", "gaunt",
    # radial canonical
    "Anorm", "PLM", "PLM_cached",
    "C_abcd", "gamma_abcd", "Scomp_abcd", "S_near_abcd", "S_near_paper",
    "b_near",
    # radial legacy
    "P", "P2", "P3", "P4",
    "P2LM", "P3LM", "P4LM",
    "P2LMn", "P3LMn", "P4LMn",
    "P4LMn_as_C_abcd",
    "Scomp", "ScompLM", "SLM", "S0",
    "create_SLM_table", "create_SLM_table_optimized",
    # coupling
    "Q_notes", "S_notes", "coupling_entry",
    "build_C_notes", "build_C_notes_cached", "build_coupling_tensor",
    "hermitian_adjoint_notes", "coupling_hermitian_defect",
    # evolution
    "omega_mode", "build_omega_arr",
    "build_Omega_notes", "rhs_A_notes",
    "build_nondiag_mask_notes", "omega_shift_notes",
    "integrate_A_notes",
    "total_action", "Ndot_notes", "physical_from_A_solution",
    # diagnostics
    "mode_label", "inspect_solution",
    # precomputed S-table
    "load_S_table", "parse_S_table_from_mathematica",
    "S_near_lookup", "S_near_cached",
]
