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
\omega_\lambda+\omega_k-\omega_i-\omega_j.
```

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
- study resonant energy sharing, nonlinear normal modes, stability islands, and possible attractor-like behavior.

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


## Installation

Create and activate a Python environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy sympy matplotlib jupyter
```

Optional tools for development:

```bash
pip install ipykernel pytest black ruff
```

Then register the environment as a Jupyter kernel:

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

Some radial coefficients involve large alternating sums of Pochhammer and gamma functions. For large overtone numbers, the result can be sensitive to summation order and floating-point cancellation. Recommended checks:

- compare direct nested sums with convolution-based sums;
- test symmetry under permutation of unbarred radial slots;
- use higher precision for large `n` sectors if necessary;
- cache Wigner 3j symbols and radial coefficients;
- validate small-mode results against direct integration where possible.

## Roadmap

- [ ] Split notebook utilities into importable modules.
- [ ] Add unit tests for `Q_element`, `S_near_abcd`, and coupling-index conventions.
- [ ] Add regression tests comparing raw and Hamiltonian-projected action drift.
- [ ] Implement Poincare section and Lyapunov diagnostic scripts.
- [ ] Add examples for four-mode and eight-mode overtone truncations.
- [ ] Add optional high-precision radial coefficient backend.
- [ ] Compare Python evolution against the Mathematica notebook workflow.

## Citation / acknowledgement

This is an exploratory research code for weakly nonlinear near-horizon QNM dynamics and weak-turbulence-style amplitude equations. If you use or adapt this code, please cite the relevant accompanying notes or paper draft once available.

