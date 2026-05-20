# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Exploratory research code for weakly-nonlinear, near-horizon QNM-like mode interactions on an extremal Reissner-Nordström background with a charged scalar field. The goal is to build a quartic mode-coupling tensor $C_{\lambda ijk}$ (angular Gaunt × near-zone radial finite-part integrals), evolve a finite-mode amplitude truncation in the interaction picture, and cross-check it against a 1+1D NR sim of the underlying PDE.

Python lives in three places:

1. **`coupling_coeffs.ipynb`** — the original exploratory notebook. Source of all conventions; **its stored output cells are stale** (see "Notebook stored outputs are stale" below).
2. **`src/rn_cwt/`** — importable modules extracted from the notebook: `background.py`, `angular.py`, `radial.py`, `coupling.py`, `evolution.py`, `precomputed.py`, `diagnostics.py`.
3. **`src/rn_cwt_nr/`** — 1+1D PDE solver for NR cross-validation: `coords.py`, `grid.py`, `pde_linear.py`, `solver.py`, `projection.py`. Imports from `rn_cwt`, never the other way. Keep that separation when extending.

Both packages are installed via the top-level `pyproject.toml`. See `src/README.md` for the module map.

Test suite: 71 tests under `tests/` and `tests/nr/`, all passing (~24 s).

A parallel Mathematica workflow in `Mathematica/` (`generate_4wave_coeffs.nb`, `evolve_amplitudes.nb`) is the cross-check for any new Python coefficient.

## Environment / commands

The user works with `uv`. Never use raw `python -m venv` / `pip` or activate-then-run; always go through uv.

```bash
uv venv                                          # one-time
uv pip install -e ".[dev]"                       # editable install + pytest, jupyter, pandas
uv run pytest                                    # 71 tests, ~24 s
uv run jupyter notebook coupling_coeffs.ipynb   # interactive
uv run python -c "from rn_cwt import build_C_notes, load_S_table"   # from modules
```

## Layered architecture

The mode-coupling pipeline has four conceptual layers:

1. **Background parameters** (module-level globals in `background.py`):
   `rp=1.`, `q=2/100.`, `h0`, `iq=1j*q`. These define the extremal RN geometry and charge. `h(l)` returns the conformal weight $h_\ell = \tfrac12 + \sqrt{1/4 + \ell(\ell+1) - q^2}$. QNM frequencies are `w(n,l) = -i(n + h(l) + iq)/2`; `s(n,l) = -i*w(n,l)` is the time-eigenvalue with Re<0. Changing `q` invalidates every cached coefficient.

2. **Angular block** (`angular.py`) — `Q_element(li,mi,lj,mj,lk,mk,lp,mp)` computes $Q^{ij}_{k\lambda} = \int_{S^2} Y_i Y_j \bar Y_k \bar Y_\lambda$ via a Wigner-3j sum. `W3j(...)` is `@lru_cache(None)` over SymPy's exact `wigner_3j`. The magnetic selection rule $m_i + m_j = m_k + m_p$ is enforced at the top — calls that violate it return 0 cheaply.

3. **Radial block** (`radial.py`) — `S_near_abcd(a,b,c,d)` returns the finite-part near-zone coefficient with the **barred slot in position b**: schematically $b_a\,\bar b_b\,b_c\,b_d$. It sums `C_abcd(j,a,b,c,d) * Scomp_abcd(j,a,b,c,d)` for `j = 0 ... Ntot` and divides by an `Anorm` product. `PLM_cached(j,n,ell)` is `@lru_cache(None)`.

   **The standalone radial mode function** is `b_near(n, ℓ, x̆)` — implements paper §10 eq. 2346–2347 directly, using the same `PLM_cached` series coefficients. Use this for QNM initial data in PDE evolutions.

   **`poch` was fixed during the refactor.** The notebook uses `poch(a,n) = (-1)^n Γ(1-a)/Γ(1-a-n)` for integer `n`. Modern scipy (≥ ~1.11) returns `NaN` at `Γ(negative_integer)` instead of `±inf`, so this formula NaNs out whenever the Pochhammer formally crosses zero (e.g. `poch(-n, j)` for `j > n+1`). The notebook stored outputs were generated under older scipy where `Γ(-k) = ±inf` and `finite/inf = 0` gave the correct result by accident. The module's `poch` uses the direct product `a (a+1) ... (a+n-1)` for non-negative integer `n` — mathematically identical, robust at the poles, and agrees with the gamma form to ~1e-12 on non-degenerate inputs.

4. **Coupling + evolution** (`coupling.py`, `evolution.py`) — `build_C_notes(modes, alpha)` assembles the rank-4 tensor `C[lam,i,j,k]` from `Q_notes` and `S_near_abcd`. `build_C_notes_cached(modes, table, alpha)` uses Peter's precomputed S-table where entries are available and falls back to direct computation otherwise. `integrate_A_notes(...)` evolves $i\dot A_\lambda = \epsilon^2 \sum C_{\lambda ijk} A_i A_j \bar A_k e^{i\Omega t}$ via `solve_ivp(method="DOP853", rtol=1e-10, atol=1e-12)`.

## S convention: paper / Mathematica

`S_near_abcd` (and via it, `build_C_notes`) uses the **paper / Mathematica convention** for the finite-part `Scomp` factor:

```
Scomp_j = (-1)^{j+γ} Γ(1+j+γ) Γ(-1-j-2γ-2iq) / [Γ(-2iq-γ)]²
```

i.e. matches Peter's `SnearComponentNLM4` in `Mathematica/generate_4wave_coeffs.nb` and the draft paper §10. Verified by `tests/test_s_convention.py` against Peter's precomputed data file `data/RN_ZDMs_4modeCoeffs_Neq1_coeffs_numeric.m` (~880 entries, worst-case relative error 1e-12).

`S_near_paper` is exposed as an explicit alias for `S_near_abcd`.

**Historical note (don't repeat the rabbit hole):** the *original* Python notebook used a different finite-part choice — $(-1)^{j+\gamma+1}$ and $\Gamma(-2iq-\gamma)$ to the first power — that differed by an overall j-independent factor $-1/\Gamma(-2iq-\gamma)$. Both are valid regularizations; we standardized on the paper form on 2026-05-12 so `rn_cwt` matches Peter's data file directly and so the standalone `b_near` (which is in paper convention) is consistent with `S_near_abcd`. This change shifts the absolute scale of `C` but leaves all structural properties (Hermitian-projection conservation, eps² scaling, anti-Hermitian defect ratios) invariant.

## Index conventions — the part that bites

The README's "Current model" section is load-bearing. The notes convention puts the **barred dynamical index last** (`k`):

```
C[lam, i, j, k] = c^{ij}_{lam,k}     <-- barred slot is k
S_notes(i,j,k,lam) ~ b_i b_j bar(b_k) b_lam
Q_notes(i,j,k,lam) = ∫ Y_i Y_j bar(Y_k) bar(Y_lam)
```

But the internal radial routine `S_near_abcd(a,b,c,d)` puts the **barred slot second**:

```
S_near_abcd(a,b,c,d) ~ b_a bar(b_b) b_c b_d
```

So the mapping is:

```python
S_notes(i, j, k, lam) = S_near_abcd(i, k, j, lam)        # swap k <-> j
C[lam, i, j, k]       = alpha * Q_notes(i,j,k,lam) * S_near_abcd(i, k, j, lam)
```

When working in this code, **always be explicit about which convention** a given expression is in. If you see `S_near_abcd(i, j, k, lam)` (no swap) being multiplied by `Q_notes`, that is almost certainly a bug. The Hermitian adjoint that matches the notes convention is `Cdag[lam,i,j,k] = conj(C[i,lam,k,j])`, implemented as `np.conjugate(np.transpose(C, (1,0,3,2)))` — note both pair-swaps.

## NR sim package (`rn_cwt_nr`)

1+1D charged-scalar Klein-Gordon on the NHERN throat, used to cross-check the amplitude ODE:

    ∂²_t ψ - ∂²_{r_*} ψ - 2iq x̆ ∂_t ψ - V(r_*) ψ = -(α/(4π)) x̆(x̆+1) |ψ|² ψ

with `V(r_*) = q² x̆² - m²_eff x̆(x̆+1)` and the 1/(4π) from the |Y_00|² of the spherical reduction.

Method of lines: 4th-order centered FD in r_* + DOP853 in time (`scipy.integrate.solve_ivp`, rtol=1e-8, atol=1e-10). Sommerfeld outgoing-at-horizon BC (`horizon_bc="outgoing"`, default) — exact in the x̆→0 limit where friction and potential vanish. Dirichlet at the throat side (`throat_bc="dirichlet"`); causes reflection that arrives at sample r_* at t = distance-to-throat, so usable integration window is bounded by that.

A compactified-y variant exists in `pde_linear.py::linear_kg_rhs_compact` (Dirichlet at both endpoints of the y∈(-1, 0) tanh-compact domain) for tests.

## PDE ↔ ODE validation status (as of 2026-05-20)

Solid:
- `α=0` PDE matches linear-QNM analytic to numerical floor (~10⁻¹² at N=1201 with rtol=1e-8).
- Nonlinear `Δψ_PDE` scales as `α·ε³` exactly across both `α∈[10⁻³, 3]` and `ε∈[10⁻², 10⁻¹]`.
- ODE Re/Im at fixed r_* have **the same signs** as PDE Re/Im (sign-flip test 2026-05-20 confirms original C is correct).
- ODE coupling C verified at amplitude level: simple pointwise extraction `A_PDE = ψ_PDE/(ε·e^{s_0 t}·b_0)` at multiple r_* agrees with `A_ODE = exp(-iε²Ct)` to ~10⁻⁶ in clean regions.

Open / structural:
- **Pointwise field-level concavity mismatch**: at fixed r_*, `Δψ_PDE ∝ t²·source(r)` (concave up/down from Duhamel of constant source), while `Δψ_ODE = ε·(A(t)-1)·e^{s_0 t}·b_0(r) ∝ t·e^{Re(s_0)t}·b_0(r)` (concave opposite). Same sign, opposite second derivative. **This is structural** — the slow-amp ansatz assumes ψ stays in the b_0 direction; the PDE source spatial profile is `|b_0|²·b_0`, which has projection AND orthogonal-to-b_0 components. Slow-amp captures only the projection; the orthogonal residual is what generates the t² behavior at fixed r. Not a sign error in C or a missing coupling channel (verified by N=1..16 truncation scans: high-n amplitudes drop fast, contribute < 0.5% at the sample r_*).
- **Clean amplitude-level validation** would require a regularized bilinear-form projection (Leaver-style analytic continuation) that filters out the non-modal forced response. Not implemented; the cutoff-integral version diverges as the grid extends to the horizon.
- **Long-time integration** past t ≈ 1/|Re(s_0)| is blocked by throat-side Dirichlet reflection. Non-reflecting throat BC would unlock this and let the field settle into the slow-amp prediction asymptotically.

Diagnostic scripts that walked through this analysis live in `examples/diagnostics/`.

## Diagnostics that matter

- **Total-action drift** $N = \sum_\lambda |A_\lambda|^2$ is the headline conservation check for the ODE. For the Hamiltonian-projected tensor `C_H = 0.5*(C + Cdag)` it should sit at solver roundoff. For the raw `C` it drifts at $O(\epsilon^2)$; the `eps_values = [0.025, 0.05, 0.1]` scan with constant `drift/eps**2` is the smoking gun that the leakage is physical (anti-Hermitian part of `C`) rather than integrator error.
- **`phase_mode`**: default to `"real"` (uses only $\mathrm{Re}\,\omega$ in the detunings). `"exact"` keeps complex QNM frequencies — for long integrations or modes with large $\Gamma$ imbalance this overflows because the interaction phase contains $e^{(\Gamma_\lambda + \Gamma_k - \Gamma_i - \Gamma_j)t}$. `"resonant"` sets phase to 1 and is for resonant truncations only.
- **Diagonal frequency shift**: `remove_diag=True` plus `include_frequency_shift=True` is the usual combination — `omega_shift_notes` absorbs the $A_q A_\lambda \bar A_q$ and $A_\lambda A_q \bar A_q$ self-shift terms into a renormalized `omega_tilde`, and the mask removes them from the RHS to avoid double-counting.

## Notebook stored outputs are stale

The cells in `coupling_coeffs.ipynb` that print `||C_H||`, `||C_A||`, `drift/eps²`, and the eps-scan all carry output values that **do not match what the current notebook source produces**. Verified by extracting the notebook code verbatim into a standalone script and running under scipy 1.10 (where the original `poch` formula works without NaN): get the same numbers the refactor gives, not the stored ones. The discrepancy is large where the comparison is well-posed (`||C_A||` differs by 45%) but invisible on the Hamiltonian-conservation diagnostic (matches at 1e-16, because *any* Hermitian C conserves N — that test never validated the absolute scale of C).

Best guess: the stored outputs predate the SLM-convention → S_near_abcd-convention refactor described in the README (the `S_notes(i,j,k,lam) = S_near_abcd(i,k,j,lam)` swap), and the eps-scan was never re-run.

Until that resolves, **treat the current notebook source as authoritative** and the stored output cells as historical artifacts. The tests in `tests/` are the live reference.

## Working on this code

- The notebook is messy by design (exploratory). It contains duplicated function names (`PLM` is redefined; `C` is used both as the coupling tensor and as a mode-dict variable in scratch cells). When asked to modify or extend, prefer adding new cells rather than rewriting old ones unless asked for cleanup.
- Caching matters for performance — `W3j`, `PLM_cached`, and the `load_S_table` pickle cache. Don't strip the `lru_cache` decorators. If you change `q` or `rp` mid-session, the caches become stale; restart the kernel.
- The Mathematica notebooks are the cross-reference for any new Python coefficient — when in doubt about a sign, factor of 2, or slot ordering, the convention used there is the source of truth.
- `Refs/` contains the two background papers (Gelles & Pretorius 2025; Baake & Rinne 2016). Cite them, don't try to re-derive their results from the code.
- The Overleaf project mirror lives in `6a01cb81233ef2c25264ea80/` (gitignored). It's a separate notes repo, not part of this project.
