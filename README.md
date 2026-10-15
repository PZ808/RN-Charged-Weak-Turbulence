# Turbulent Modes: Nonlinear QNM Mode Coupling

This repository contains numerical tools for studying weakly nonlinear interactions among near-horizon quasinormal-mode-like amplitudes. The current focus is a finite-mode truncation of a quartic interaction of the schematic form

```math
i\dot A_\lambda
=
\epsilon^2\sum_{i,j,k}
C_{\lambda i j k}\,
A_iA_j\bar A_k\,
\exp\left[i\Omega_{\lambda i j k}t\right],
```

with

```math
\Omega_{\lambda i j k}
=
\omega_\lambda+\omega_k-\omega_i-\omega_j,
```

together with its large-$\ell$ continuum limit.

The code is designed to separate the angular and radial pieces of the interaction coefficient,

```math
C_{\lambda i j k}
=
\alpha\,Q^{ij}_{k\lambda}\,S^{ij\lambda}_{k},
```

where

```math
Q^{ij}_{k\lambda}
=
\int_{\mathbb S^2}Y_iY_j\bar Y_k\bar Y_\lambda\,d\Omega,
```

and the near-zone radial coefficient is computed from finite-part contour integrals of products of radial mode functions.

The main goals are:

- construct reliable mode-coupling tensors from angular Gaunt/Wigner coefficients and near-zone radial coefficients;
- evolve weakly nonlinear finite-mode systems in an interaction picture;
- separate Hamiltonian and non-Hamiltonian pieces of the near-horizon dynamics;
- study resonant energy sharing, nonlinear normal modes, stability islands, and possible attractor-like behavior;
- determine whether the resulting weak-turbulence problem supports a constant-flux (Kolmogorov–Zakharov) cascade.

## Current model

The notes convention used in the Python evolution is

```math
i\dot A_\lambda
=
\epsilon^2\sum_{i,j,k}
c^{ij}_{\lambda,k}
A_iA_j\bar A_k
\exp\left[i(\omega_\lambda+\omega_k-\omega_i-\omega_j)t\right].
```

The barred dynamical index is the last index, `k`. Internally, the radial coefficient routine uses a barred-second convention,

```math
S_{\rm near}(a,b,c,d)
\sim b_a\,\bar b_b\,b_c\,b_d.
```

Therefore the notes convention is implemented by the map

```python
S_notes(i, j, k, lam) = S_near_abcd(i, k, j, lam)
```

and

```python
C[lam, i, j, k] = alpha * Q_notes(i, j, k, lam) * S_near_abcd(i, k, j, lam)
```

where

```math
Q_{\rm notes}(i,j,k,\lambda)
=
Q^{ij}_{k\lambda}
=
\int_{\mathbb S^2}Y_iY_j\bar Y_k\bar Y_\lambda\,d\Omega.
```

## Mathematical structure

The quartic interaction is motivated by an action of the form

```math
(\Phi^*\Phi)^2.
```

For a fully Hermitian Hamiltonian truncation, the coupling tensor satisfies the adjoint symmetry

```math
C_{\lambda i j k}
=
\overline{C_{i\lambda k j}}.
```

In near-horizon coordinates this symmetry is expected to hold only up to the order at which the approximation is Hamiltonian. The code therefore decomposes the raw coupling tensor into

```math
C=C_H+C_A,
```

with

```math
C^H_{\lambda i j k}
=
\frac12\left(C_{\lambda i j k}+\overline{C_{i\lambda k j}}\right),
```

and

```math
C^A_{\lambda i j k}
=
\frac12\left(C_{\lambda i j k}-\overline{C_{i\lambda k j}}\right).
```

The Hamiltonian-projected system conserves the total action

```math
N=\sum_\lambda |A_\lambda|^2
```

to machine precision in the current tests, while the raw near-zone tensor produces a small secular drift consistent with the non-Hermitian correction.

That drift is not only a numerical nuisance. In the stirred large-$\ell$ problem the anti-Hermitian part `C_A` turns out to drive a pump-sustained linear instability of the high-$\ell$ band, which the Hamiltonian projection removes entirely; see the next section.

## Large-$\ell$ asymptotics and stirring

[`Large_ell/`](Large_ell/) carries a self-contained programme that pushes the coupling tensor to large multipole and asks whether the system sustains a turbulent cascade. The analysis note is `large_ell_stirring.tex` / `.pdf`; the raw output of every run quoted in it is under [`paper/4_waves_RN/logs/`](paper/4_waves_RN/logs/).

The enabling step is an exact reduction of the magnetic sum to a single multipole sum,

```math
\sum_{\text{all }m}\left|Q^{12}_{34}\right|^{2}
=
\sum_L\frac{\mathcal G^{2}_{\ell_1\ell_2L}\,\mathcal G^{2}_{\ell_3\ell_4L}}{2L+1},
```

together with closed forms for the near-zone overlap evaluated entirely in log space, so that the cancelling factors $2^{\pm H}$ never appear separately. Between them these take `lmax` from about 7 to 40–60 for the full axisymmetric tensor, and to several hundred for the $m$-averaged shell kernel.

Main results:

- **The real frequency is exactly degenerate.** $\mathrm{Re}\,\nu=\hat q/2$ for every $(n,\ell)$, so the detuning vanishes identically and every quartet is exactly resonant. A corollary is that the energy is proportional to the action: there is one invariant, hence one equilibrium and one cascade candidate, and no dual cascade.
- **The overtone staircase does not survive normalisation.** The $\ell^{-\lfloor(N_S+1)/2\rfloor-1/2}$ suppression of the bare overlap is cancelled almost exactly by the overtone content of the normalisation, leaving $\ell^{-1/2}$ for odd total overtone number $N_S$ and no suppression at all for even. The decay in $n$ at fixed $\ell$ is only algebraic, $R_n\to n^{-1/2}$. Consequence for finite-mode work: of the 819 active quartets in the nine-mode truncation used in the paper draft, only 19 have all $n=0$, and they carry 17.9% of $\sum|C|^2$.
- **The anti-Hermitian part drives an instability.** Linearising about a band-limited pump, every eigenvalue of the response operator has positive real part at zero damping; the Hamiltonian projection is neutral on the $n=0$ tower. The instability is cut off by QNM damping at $\sigma<\rho_c\,\alpha|c_{\rm low}|^2\epsilon^2$ with $\rho_c=3.19\pm1.40$ over pump phases (24 of 24 random pumps unstable). Its kinetic counterpart is a loss of the $k\leftrightarrow\lambda$ symmetry of the collision kernel once overtones are present, which produces action at a few per cent of the injection rate.
- **There is no Kolmogorov–Zakharov cascade.** The shell flux never plateaus and the interior spread saturates under cutoff refinement. The Zakharov transform gives $y=3x-\gamma-4$ and $x_{\rm KZ}=(4+\gamma)/3\to1$, but the collision integral does not converge there: the UV edge, measured directly rather than power-counted, is at $x=1.096$, while infrared insensitivity requires $x<1$. The locality window is empty. The dominant configurations are $(p,p,O(1))$ — two legs at the UV cutoff and one at the infrared cutoff — so high-$\ell$ shells couple directly to the pump rather than to their neighbours.
- **The principal branch is not a way out.** At large $\hat q$ the band $\ell(\ell+1)+\tfrac14<\hat q^2$ has $h=\tfrac12-i\mu_\ell$, $\ell$-independent damping $\Gamma=(n+\tfrac12)/2$, a real frequency ladder and a thin resonant manifold — structurally the near-extremal Kerr tower. But it is also the band on which the symmetric bilinear form $\int\psi^2$ that makes these modes orthogonal collapses (14.6 to $2.6\times10^{-11}$ across $\ell=1\ldots11$ at $\hat q=12$), so the normalised couplings grow like $e^{3\pi(\hat q-\mu_\ell)}$ and weak nonlinearity fails before any inertial range can be used.

### Modules

| module | purpose |
| --- | --- |
| `_project_path.py` | locates the repository root so the folder can live anywhere |
| `large_ell_backend.py` | vectorised `n = 0` coupling tensor and `m`-averaged shell kernel |
| `large_ell_overtones.py` | overtone-resolved tensor, via a rational recursion on top of the `n = 0` closed form |
| `large_ell_principal.py` | continuation to complex `h` (principal branch) |
| `large_ell_checks.py` | asymptotics suite A–E, with an mpmath path for high precision |
| `large_ell_program.py` | pump operator, band stability, stationary solution, forced–dissipative kinetics |
| `large_ell_overtone_program.py` | overtone audit (conditioning, the overtone law, the nine-mode set, the programme rerun) |
| `large_ell_zakharov.py` | Zakharov transform, homogeneity degree, locality |
| `large_ell_branch_test.py` | principal-branch spectrum, normalisation collapse, resonance broadening |
| `large_ell_extra_checks.py` | UV tail exponent and the collapse measured from the project code alone |
| `large_ell_verify.py` | independent verification suite V1–V5 |

Each back end is validated against `coupling_coefficients_stable` at small `l` — relative errors between `1e-11` and `1e-16` on every assembled object — and none of them import or modify it outside their `validate()` functions.

## Features

- **Angular coupling coefficients** using Wigner 3j/Gaunt formulas:
  - computes `Q_element(li, mi, lj, mj, lk, mk, lp, mp)`;
  - supports block structure from the magnetic selection rule;
  - checks symmetry and Hermiticity properties.

- **Near-zone radial coefficients**:
  - computes finite-part coefficients from truncated hypergeometric/Pochhammer series;
  - uses a canonical barred-second convention `S_near_abcd(a,b,c,d)`;
  - provides wrappers for the notes/evolution convention.

- **Mode evolution**:
  - integrates the interaction-picture amplitude system with `solve_ivp`;
  - supports exact complex phases, real-frequency phases, and resonant truncations;
  - optionally removes diagonal self-frequency-shift terms.

- **Hamiltonian diagnostics**:
  - computes total-action drift;
  - projects raw couplings to the Hamiltonian part;
  - tests drift scaling with `epsilon`;
  - measures anti-Hermitian defects.

- **Nonlinear dynamics diagnostics**:
  - modal power exchange;
  - normalized power simplex dynamics;
  - Poincare sections;
  - relative equilibrium / nonlinear normal mode searches;
  - Lyapunov exponent estimates.

- **Large-$\ell$ / weak-turbulence tools** (see above):
  - log-space closed forms and the exact magnetic-sum reduction;
  - overtone-resolved and principal-branch continuations;
  - shell kernel, forced–dissipative kinetics, and shell-flux diagnostics;
  - Zakharov transform and direct locality measurement.

## Repository layout

The original `coupling_coeffs.ipynb` notebook is now refactored into two
sibling Python packages under `src/`:

- **`rn_cwt`** — mode-coupling tensor and interaction-picture amplitude
  ODE (the "stable foundation").
- **`rn_cwt_nr`** — 1+1D charged-scalar KG PDE solver on the NHERN throat,
  for cross-checking the amplitude ODE against direct numerical evolution.

See [`src/README.md`](src/README.md) for the package-level docs (module map,
S-convention, dependency direction, test commands). Diagnostic example
scripts from PDE↔ODE validation work live under
[`examples/diagnostics/`](examples/diagnostics/); the foundational examples
that drive the workflow are in [`examples/`](examples/) directly.

The large-$\ell$ programme is deliberately kept outside the packages, in
[`Large_ell/`](Large_ell/), together with the analysis note it supports. It
depends on `coupling_coefficients_stable` only for validation, and finds it by
walking up from its own location (`_project_path.py`), so the folder can be
moved without editing any imports.

## Installation

Recommended (uses [`uv`](https://docs.astral.sh/uv/) for environment + lockfile):

```bash
uv venv                           # one-time
uv pip install -e ".[dev]"        # editable install with pytest, jupyter
uv run pytest                     # 71 tests, ~24 s
uv run jupyter notebook coupling_coeffs.ipynb
```

Note: **run `uv run pytest` after a fresh clone** — the first time anything
touches the S-table loader, the 214 MB Mathematica association
`data/RN_ZDMs_4modeCoeffs_Neq1_Leq4_coeffs_numeric.m` is regex-parsed (~30 s) and
cached as a small pickle alongside (`*.m.pkl`, ignored by git). All
subsequent test/example runs read the pickle in ~0.1 s. Running the test
suite is the easiest way to build that cache.

Alternatively, with a plain virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then register the environment as a Jupyter kernel (optional):

```bash
python -m ipykernel install --user --name turbulent-modes --display-name "Python (turbulent-modes)"
```

## Quick start

### 1. Define a small mode set

For example, a four-mode overtone sector with `l=m=0`:

```python
modes = [
    {"n": 1,  "l": 0, "m": 0},
    {"n": 2,  "l": 0, "m": 0},
    {"n": 7,  "l": 0, "m": 0},
    {"n": 11, "l": 0, "m": 0},
]
```

### 2. Build the coupling tensor

```python
C = build_C_notes(modes, alpha=1.0)
```

This builds

```math
C_{\lambda i j k}=c^{ij}_{\lambda,k}.
```

### 3. Split Hamiltonian and non-Hamiltonian pieces

```python
Cdag = np.conjugate(np.transpose(C, (1, 0, 3, 2)))
C_H = 0.5 * (C + Cdag)
C_A = 0.5 * (C - Cdag)
```

The Hamiltonian-projected tensor `C_H` should conserve total action in the interaction-picture dynamics.

### 4. Evolve the system

```python
eps = 0.05
T = 200.0

A0 = np.array([
    1.0,
    0.2*np.exp(0.3j),
    0.1*np.exp(1.1j),
    0.05*np.exp(-0.7j),
], dtype=complex)

sol, omega_tilde, Omega, mask = integrate_A_notes(
    A0,
    T,
    eps,
    omega_arr,
    C_H,
    nout=1000,
    remove_diag=True,
    include_frequency_shift=True,
    phase_mode="real",
)
```

### 5. Check total-action conservation

```python
P = np.abs(sol.y.T)**2
N = np.sum(P, axis=1)

print("relative drift =", (N[-1] - N[0]) / N[0])
```

For `C_H`, the drift should be at the level of numerical roundoff. For the raw tensor `C`, the drift scales approximately like `eps**2` in the current tests.

### 6. Large-$\ell$ tensors

```python
from large_ell_backend import Backend

be = Backend(qhat=0.6, rplus=1.0, lmax=40)
C, ells = be.C_axisymmetric(alpha=1.0, lmin=1)   # 1.0 s, 41 MB
W, _    = be.W_shell(lmin=1)                     # m-averaged shell kernel
```

Run `python large_ell_verify.py` from `Large_ell/` to revalidate all three
back ends against `coupling_coefficients_stable`.

## Diagnostics

### Modal power exchange

```python
P = np.abs(sol.y.T)**2

for j, mode in enumerate(modes):
    plt.plot(sol.t, P[:, j], label=f"n={mode['n']}, l={mode['l']}, m={mode['m']}")

plt.xlabel("t")
plt.ylabel(r"$|A_\lambda(t)|^2$")
plt.legend()
plt.show()
```

### Normalized power simplex

```python
p = P / np.sum(P, axis=1)[:, None]

for j, mode in enumerate(modes):
    plt.plot(sol.t, p[:, j], label=f"n={mode['n']}")

plt.xlabel("t")
plt.ylabel(r"$p_\lambda=|A_\lambda|^2/\sum_\mu |A_\mu|^2$")
plt.legend()
plt.show()
```

### Epsilon scaling of non-Hamiltonian drift

```python
eps_values = [0.025, 0.05, 0.1]

for ep in eps_values:
    sol_ep, _, _, _ = integrate_A_notes(
        A0, T, ep, omega_arr, C,
        nout=1000,
        remove_diag=True,
        include_frequency_shift=True,
        phase_mode="real",
    )

    P_ep = np.abs(sol_ep.y.T)**2
    N_ep = np.sum(P_ep, axis=1)
    drift = (N_ep[-1] - N_ep[0]) / N_ep[0]

    print(ep, drift, drift / ep**2)
```

A roughly constant `drift / eps**2` indicates that the leakage is coming from the weak nonlinear RHS rather than from solver error.

## Phase conventions

Three phase modes are useful:

- `phase_mode="real"`: uses only real detunings. This is the safest setting for weak-turbulence style tests.
- `phase_mode="exact"`: uses the full complex QNM detuning. This can produce exponential growth or decay in the interaction-picture phase.
- `phase_mode="resonant"`: keeps only resonant or selected terms and sets the phase to unity.

For QNM frequencies

```math
\omega_\lambda=\Omega_\lambda-i\Gamma_\lambda,
```

an exact complex interaction phase contains factors of the form

```math
\exp\left[(\Gamma_\lambda+\Gamma_k-\Gamma_i-\Gamma_j)t\right].
```

This may overflow for long integrations if nonresonant quartets have positive growth exponent. For exploratory Hamiltonian dynamics, use `phase_mode="real"` or a resonant truncation.

Note that on the supplementary branch the *real* detuning vanishes identically,
$\mathrm{Re}\,\nu=\hat q/2$ for every mode, so `phase_mode="real"` makes every
quartet resonant. This is a property of the spectrum, not a bug.

## Physical versus interaction amplitudes

The evolution is usually performed for interaction-picture amplitudes `A`. To reconstruct the softened/physical amplitude, use

```math
a_\lambda(t)=\epsilon A_\lambda(t)e^{-i\omega_\lambda t}.
```

In code:

```python
a_phys = eps * sol.y.T * np.exp(-1j * omega_arr[None, :] * sol.t[:, None])
```

Plotting `|A|` shows nonlinear energy sharing in the interaction picture. Plotting `|a_phys|` shows the same dynamics on top of the QNM/near-horizon exponential damping.

## Nonlinear dynamics program

The next stage is to analyze the finite-mode truncation as a nonlinear dynamical system.

Suggested studies:

1. **Hamiltonian core**: set `C = C_H` and look for invariant tori, stability islands, nonlinear normal modes, and chaotic layers.
2. **Controlled leakage**: study `C = C_H + eta*C_A` for `0 <= eta <= 1`.
3. **Relative equilibria**: solve

   ```math
   F_\lambda(B)=\nu B_\lambda
   ```

   for nonlinear normal modes.

4. **Poincare sections**: use phase combinations such as

   ```math
   \phi=\arg A_1+\arg A_2-\arg A_3-\arg A_0
   ```

   and plot crossings in the normalized power simplex.

5. **Lyapunov scans**: estimate chaos as a function of initial powers, phases, and the leakage parameter `eta`.

The expected qualitative picture is:

```text
eta = 0:     Hamiltonian resonant exchange, tori, islands, separatrices, chaos.
0 < eta << 1: slow drift across Hamiltonian structures.
eta ~ 1:     near-horizon non-Hermitian dynamics with possible attractor-like behavior.
```

## Notes on numerical precision

The radial coefficient is a finite alternating sum over `j` of Pochhammer and
gamma products. The cancellation in that sum is *structural*, not incidental:
it is the same cancellation that produces the overtone staircase, so it gets
worse exactly where the physics gets interesting. Measured against 120-digit
arithmetic, the relative error of the double-precision sum is, with
`N_S` the total overtone number of the quartet:

| `N_S` | `l = 2` | `l = 8` | `l = 32` |
| --- | --- | --- | --- |
| 0 | 3.7e-15 | 1.1e-14 | 7.3e-14 |
| 4 | 2.8e-14 | 2.3e-13 | 3.2e-12 |
| 8 | 4.8e-12 | 1.2e-09 | 2.2e-08 |
| 12 | 1.3e-09 | 3.9e-07 | 5.7e-04 |
| 16 | 6.5e-08 | 1.1e-04 | 5.9e+00 |

Double precision is therefore safe to `N_S <= 4` at any `l`, and to `N_S = 8`
up to `l ~ 30`; beyond that the overlap needs extended precision. This affects
`S_near_abcd` exactly as it affects the new back ends — the two agree with each
other to the digits they both have, and lose the same digits together.
`large_ell_checks.py` provides an mpmath path for the regimes where that
matters.

Other recommended checks:

- compare direct nested sums with convolution-based sums;
- test symmetry under permutation of unbarred radial slots;
- cache Wigner 3j symbols and radial coefficients;
- validate small-mode results against direct integration where possible;
- on the principal branch, remember that the conjugated slot carries
  $\bar h=1-h$, not $h$ — the distinction is invisible for real `h` and
  rescales the overlap by $e^{O(\mu)}$ when `h` is complex.

## Roadmap

- [ ] Split notebook utilities into importable modules.
- [ ] Add unit tests for `Q_element`, `S_near_abcd`, and coupling-index conventions.
- [ ] Add regression tests comparing raw and Hamiltonian-projected action drift.
- [ ] Implement Poincare section and Lyapunov diagnostic scripts.
- [ ] Add examples for four-mode and eight-mode overtone truncations.
- [ ] Add optional high-precision radial coefficient backend.
- [ ] Compare Python evolution against the Mathematica notebook workflow.

Large-$\ell$ programme:

- [ ] Analytic account of the mixed UV–infrared corner `(p, p, O(1))` that
      dominates the collision integral, to turn the measured empty locality
      window into a proof.
- [ ] Overtones at large `l` *and* large `n` together: the overtone sum is cut
      off by the near-extremal bound `n << r_+/sigma`, not by the coupling.
- [ ] A positive-definite norm for the principal-branch modes, to make the
      collapse a statement about physical amplitudes rather than about the
      normalisation convention.
- [ ] Non-axisymmetric coherent sector; the shell kernel already handles all
      `m` kinetically.
- [ ] Multi-charge model, in which the cascade label *is* the protecting
      charge — the structural feature RN lacks and Kerr has.

## Citation / acknowledgement

This is an exploratory research code for weakly nonlinear near-horizon QNM dynamics and weak-turbulence-style amplitude equations. If you use or adapt this code, please cite the relevant accompanying notes or paper draft once available.
