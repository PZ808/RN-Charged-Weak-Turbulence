# Large-ell regime experiments

This directory adds experiments and diagnostics only. It imports the existing
`Large_ell/large_ell_backend.py` and the setup, projection, and pump helpers in
`large_ell_program.py`. No coefficient, normalization, or base evolution routine
is changed. There is no commit or alteration of pre-existing user changes.

## Run

From the repository root, with the usual project environment:

```sh
uv run --extra dev pytest experiments/large_ell_regimes/test_regimes.py -q
uv run python experiments/large_ell_regimes/regimes.py --lmax 16 --out experiments/large_ell_regimes/results_L16
```

An environment independent of the project can instead be used:

```sh
uv run --no-project --with numpy --with scipy --with matplotlib --with pytest python experiments/large_ell_regimes/regimes.py --lmax 16
```

Outputs: `kinetic.png/pdf`, `coherent.png/pdf`, numerical arrays in two `.npz`
files (no pickle needed), and `report.json`. The report records parameters,
source hashes, all budget residuals, event stops, and tolerance refinement.
Nonpositive occupations, failed solves, or failed numerical acceptance checks
raise errors. Occupations are never silently clipped.

`--samples`, `--seed`, `--kinetic-end`, and `--coherent-end` control sampling,
pump realization, and duration. Output directories should be distinct for
different runs; running again in the same directory replaces its results.

## Figure 1: closed conservative kinetics

The driver uses the same four-wave gain/loss contraction and shell kernel as
the existing `_kinetic`, without source or damping. Initial action is concentrated
in ell=2..4, on a positive background; the mean occupation is normalized to one.
Total action is N=sum (2ell+1)n_ell, so RJ occupation is N/sum(2ell+1).
The plotted distance is sqrt(sum g (n/n_RJ-1)^2 / sum g).

Flux is Pi(L)=-sum_{ell<=L}g_ell C_ell. It is an instantaneous conservative
transfer diagnostic, not evidence by itself of a stationary cascade. Positive
flux denotes transfer out of the lower shells. The terminal flux must vanish.
The finite network is intentionally modest: this figure addresses transient
redistribution followed by thermalization, not a continuum locality proof.

Normalization matters: C[a n]=a^3 C[n], so n_a(T)=a n(a^2 T).
The selected mean occupation therefore fixes the numerical kinetic clock.
Different lmax values here have different total initial actions; use common
initial occupations/action in a future cutoff-convergence study. These runs
are solver checks, not claims of cutoff independence.

## Figure 2: coherent response and nonlinear action budgets

C_eta=H+eta R, H=ham_project(C), R=C-H. R is the residual under the existing
full Hamiltonian projection, not an independently defined matrix adjoint.
All runs use the fundamental, axisymmetric tower with qhat=.6, rplus=1,
alpha=1, and a frozen unit-action pump on ell<=3.

The 9x17 response map uses the real Jacobian of
A'=-i(M A+P A*)-rho gamma A, with gamma=h/2. Since F is nonzero, A=0 is not a
solution of the full forced equations: these eigenvalues are homogeneous
response rates at zero high-band amplitude, not a stationary-state stability
classification. Stationary-state stability requires a Jacobian at that state.

The 5x5 budget maps integrate the full cubic equations from the same small
high-band seed for each (eta,rho). They show finite-time mean signed rates,
not steady-state rates. The decomposition is:

* Hamiltonian pump exchange: high-band H contribution minus internal-high H
  contribution. The latter sums to zero and is checked independently.
* Residual production/loss: 2 Re <A,-i eta R[a,a,a*]> over high modes.
* QNM loss: -2 rho sum gamma |A|^2.

The pump is maintained externally. Hamiltonian high-band action can therefore
grow even at eta=0. Residual production includes interactions with the pump
and within the high band; it need not have a definite sign.

Budget integrals are auxiliary ODE variables. The independent check compares
their sum with the action change computed from evolved amplitudes. Tighter
tolerances re-evolve all 25 grid trajectories. A high-band action of 100 stops
an integration, flags the report, and prevents a numerical blow-up from being
presented as saturation. A stopped run's mean covers its actual elapsed time;
inspect flags before comparing maps or lengthening the integration.
Crosses on the maps and trajectories identify these stopped runs. The exact
event point is retained, even if it falls between requested output times.
Large high-band action also calls for reassessing the frozen-pump and weakly
nonlinear approximations; the cap is not a prediction of physical saturation.

## Analytic checks and units

These are dimensionless reduced equations. Coherent time is tau=eps^2 t;
rho=sigma/eps^2 in the existing driver's rplus=1 convention. Amplitude loss
-rho gamma A implies occupation loss -2rho gamma n. The kinetic operator
keeps the existing driver's overall normalization; restoring an absolute
physical kinetic clock requires retaining its chosen 2pi/prefactor convention.
At the scaling level T~eps^4 t and rho4=sigma/eps^4=rho/eps^2. The two figures
are separate controlled experiments, not trajectories for one fixed physical
eps and sigma. No SI units or common physical decay time are inferred.

Tests cover:

* Literal quartet sum versus optimized collision contraction.
* Weighted action conservation, RJ stationarity, and cubic homogeneity.
* Entropy derivative versus the positive quartet-square formula, including 1/4.
* Shell-flux discrete divergence identity.
* Hamiltonian projection idempotence and action conservation.
* Exact high-band action balance and damping sign/factor two.
* Pump scaling F~c^3, M/P~c^2.
* Real-linear Jacobian versus complex action and a matrix-exponential solution.
* One-mode nonlinear gain sign and exact exponential linear decay.

The production run additionally checks central finite differences of the full
cubic RHS against the pump Jacobian, integrated budgets, entropy monotonicity,
and trajectory convergence after tightening both solver tolerances tenfold.
The JSON report is the numerical evidence, including when RJ has not yet been
reached. Longer runs, several pump seeds, larger cutoffs, overtones, physical
distributed damping, and boundary-only sinks remain separate studies.
Once the RJ distance reaches roughly 1e-12 or below, its subsequent small
oscillations reflect numerical roundoff and solver error, not resolved physics.

## Controlled kinetic damping sweep

`damping_sweep.py` adds the driven comparison without changing this directory's
original driver or any base tool:

```sh
uv run --extra dev pytest experiments/large_ell_regimes/test_damping_sweep.py experiments/large_ell_regimes/test_regimes.py -q
uv run python experiments/large_ell_regimes/damping_sweep.py
```

Defaults: cutoffs 16 and 24, rho4=0.01,0.1,1,10, total action injection I=1
into ell=2..4, initial total action N=1, qhat=.6. The physical occupation loss
is D_ell=rho4 (h_ell/2)/(2pi), exactly the existing `_kinetic` convention.
The boundary sink is constant on ell>=ceil(.75 lmax) and zero elsewhere.
Its amplitude is chosen to match sum(g D) of the physical profile: the two
profiles have equal total loss for a uniform occupation. This does not match
their actual stationary occupations, action, or residence times. Forcing and
initial total action are held fixed across cutoffs. The boundary moves with
the cutoff; these are controlled finite-network comparisons, not a continuum
limit at fixed physical boundary location.

Each of 16 cases integrates in successively longer intervals until
sum(g |n'|)/I < 1e-7 or `--max-time` (default 1e6) is reached. Every case is
repeated at tenfold tighter solver tolerances and a 1e-8 stationarity criterion.
No case is labeled stationary just because its allotted time ended. The
integrated loss is an auxiliary ODE variable checking N-N0=I T-integrated_loss.
The steady check is Pi(L)=sum_{ell<=L} g(S-Dn); the nonstationary identity
additionally includes -sum g n'. Occupations are not clipped.

Outputs under `damping_results/`: spectra and flux comparison, sweep
diagnostics, full time histories and profiles in individual NPZ files, and
`report.json` with checks, source hashes, and software versions. The script
raises an error if any case fails its stationarity, balance, or refinement
criteria. Use `--out` to preserve an earlier run when changing parameters.

The flux gap is defined by 4<ell<ceil(.75 lmax). A boundary-only sink enforces
a flat steady flux there by conservation, even if quartet transfer is nonlocal.
Consequently a plateau in the dashed curves is evidence of sustained action
throughput across shell cuts, not by itself a local KZ cascade. A nearly RJ
shape can carry finite throughput: the forcing requires small deviations from
detailed balance. In this forced problem distance from RJ is a shape diagnostic,
not a guaranteed monotonically decreasing entropy measure. It uses the
instantaneous mean occupation, since total action is no longer conserved.
