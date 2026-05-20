# `src/` — Python packages

Two sibling packages, installed together via the top-level `pyproject.toml`.
The split reflects a deliberate separation of concerns: `rn_cwt` is the
stable foundation (mode-coupling tensor, amplitude ODE); `rn_cwt_nr` is the
1+1D NR sandbox that uses `rn_cwt` for initial data and reference values but
is allowed to evolve independently.

## `rn_cwt/` — Quartic mode coupling and amplitude-ODE evolution

Refactor of the original `coupling_coeffs.ipynb` notebook into importable
modules. Provides:

- **`background`** — RN parameters (`rp`, `q`, `h0`, `iq`), conformal weight
  `h(l)`, QNM frequency `w(n, l)` and exponent `s(n, l)`, a robust `poch` that
  avoids the NaN failure mode of newer SciPy at negative-integer poles.
- **`angular`** — `Q_element(...)` for the four-mode angular Gaunt integral via
  cached Wigner-3j sums (`W3j`). Magnetic selection rule enforced.
- **`radial`** — Near-zone radial coefficients in **paper / Mathematica
  convention** (matches Peter's data file to ~10⁻¹² in `tests/test_s_convention.py`).
  Key entry points: `S_near_abcd(a, b, c, d)` (canonical name, barred slot is
  `b`); `b_near(n, ell, x̆)` for the radial QNM profile used to construct
  initial data.
- **`coupling`** — `build_C_notes(modes, alpha)` assembles the rank-4 tensor
  `C[lam, i, j, k]` (notes convention: barred dynamical index is `k`).
  `build_C_notes_cached(modes, table, alpha)` uses Peter's precomputed
  S-table when entries are available and falls back to `S_near_abcd` otherwise.
- **`evolution`** — Interaction-picture ODE
  ` i Ȧ_λ = ε² Σ C[λ,i,j,k] A_i A_j Ā_k exp(i Ω_λijk t) `
  via `integrate_A_notes(...)`. `phase_mode` ∈ {`real`, `exact`, `resonant`}
  (the README in the project root documents the trade-offs).
- **`precomputed`** — Parser and pickle cache for Peter's S-table
  (`data/RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m`).
- **`diagnostics`** — Mode labels and solution-printing helpers.

The S-convention is the paper / Mathematica form throughout; see CLAUDE.md for
the convention story (older Python-notebook convention differed by an overall
j-independent factor).

## `rn_cwt_nr/` — 1+1D charged-scalar KG on the NHERN throat

Method-of-lines PDE solver for the charged-scalar Klein-Gordon equation on
the near-horizon extremal RN throat in tortoise coordinate r_*, with
optional cubic self-interaction:

    ∂²_t ψ - ∂²_{r_*} ψ - 2iq x̆ ∂_t ψ - V(r_*) ψ
        = -(α/(4π)) x̆(x̆+1) |ψ|² ψ,
    V(r_*) = q² x̆² - m²_eff x̆(x̆+1).

- **`coords`** — Tortoise ↔ x̆ conversions; `tanh`-compactification helpers.
- **`grid`** — Uniform grid + 4th-order centered finite differences.
- **`pde`** — `kg_rhs` and `kg_rhs_compact` give the RHS in (ψ, π) form
  (both straight-r_* and compactified variants). Cubic nonlinearity is
  optional via the `alpha` keyword: `alpha=0` is the linear PDE, `alpha≠0`
  adds the `-(α/(4π))·x̆(x̆+1)·|ψ|²·ψ` source. Implements the outgoing-
  Sommerfeld BC at the horizon side (`horizon_bc="outgoing"`, default) —
  exact in the x̆→0 limit — and Dirichlet at the throat side (a known
  limitation; see open items in the main README).
- **`solver`** — `evolve_kg(...)` thin wrapper around
  `scipy.integrate.solve_ivp` (DOP853, rtol=1e-8, atol=1e-10).
- **`projection`** — Bilinear-form projection scaffolding (parked: clean QNM
  projection on a Cauchy slice requires Leaver-style analytic continuation;
  the cutoff-integral version diverges as the grid extends to the horizon).

## Dependency direction

`rn_cwt_nr` imports from `rn_cwt` (uses `b_near`, mode frequencies, `C`
coefficients for cross-checks); `rn_cwt` does not import from `rn_cwt_nr`.
Keep it that way when extending.

## Tests

```bash
uv run pytest                                   # all 71 tests, ~24 s
uv run pytest tests/test_s_convention.py        # the S-vs-Peter cross-check
uv run pytest tests/nr/                         # NR-side only
```

See the main `README.md` for usage examples and physics background.
