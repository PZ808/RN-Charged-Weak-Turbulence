# Controlled damping sweep: results

The implemented comparison uses the existing fundamental shell kernel, without
modifying base tools or the original experiment driver. Sixteen steady problems
were solved: cutoffs 16 and 24, rho4=0.01,0.1,1,10, and two sink profiles. Every
case was independently repeated at tenfold tighter solver tolerances and a
tenfold tighter stationarity threshold. All cases converged and passed.

Both sink profiles use the same action injection I=1 on ell=2..4 and the same
initial total action N=1. Distributed QNM damping is D=rho4 h/(4pi). The control
sink is nonzero only on the upper quarter of shells and is normalized to have
the same sum(g D) as the QNM profile. This matches loss for uniform occupations,
not stationary action or the actual loss in individual shells. The latter
adjusts self-consistently and totals I at stationarity in both cases.

For Lmax=24, the gap between forcing and boundary sink is ell=5..17:

| rho4 | Distributed gap flux / I, min..max | Distributed RJ distance | Boundary RJ distance |
|---:|---:|---:|---:|
| 0.01 | 0.623..0.985 | 0.000689 | 0.000878 |
| 0.1 | 0.543..0.967 | 0.343 | 0.289 |
| 1 | 0.0280..0.578 | 2.89 | 0.947 |
| 10 | 2.89e-8..0.00554 | 5.43 | 1.07 |

The boundary-only gap flux is 1 to numerical accuracy in all four cases.
The RJ distance is a weighted RMS shape difference from the constant spectrum
with the same instantaneous total action. A small value does not mean zero flux:
at weak damping, large occupation permits small relative departures from
detailed balance to carry the imposed unit throughput. This follows also from
the cubic scaling of the collision operator. The near-RJ forced states differ
from the closed, exactly zero-flux equilibrium in the original figure.

Distributed damping absorbs action throughout the band, so flux decreases
with shell number. At rho4=1 about 62% of injection is absorbed in gap shells
5..17 (including loss in shell 5); flux across successive gap boundaries drops
from 0.578 I to 0.028 I. At rho4=10 the flux already crossing ell=5 is below
0.006 I: almost all input is removed near the forcing region. A small flux
*spread* in this case reflects nearly zero throughput, not a recovered cascade.

The boundary-only plateau is an exact steady balance consequence wherever
S=D=0. It demonstrates sustained transport across shell cuts, but does not
prove scale-local interactions, a universal KZ spectrum, or cutoff independence.
Nonlocal quartets can also carry that flux. Changing cutoff changes the sink's
location; these are two finite systems, not a proven continuum limit.

## Numerical evidence

* Maximum stationarity residual sum(g |n'|)/I: 8.9e-10.
* Maximum steady injection/loss mismatch: 8.9e-10.
* Maximum flux versus cumulative source-minus-loss residual / I: 8.9e-10.
* Maximum integrated action-balance residual: 5.4e-12 relative.
* Maximum steady occupation refinement difference: 1.1e-9 relative to its maximum.

`damping_results/report.json` contains all per-case results and source hashes.
The figures are `damping_comparison.png/pdf` and `damping_diagnostics.png/pdf`;
individual NPZ files include profiles, stationary spectra, fluxes, and sampled
time histories. These runs retain the conservative, fundamental kinetic closure;
they do not include the coherent non-Hamiltonian gain or overtone kernel defects.

Reproduce with:

```sh
uv run python experiments/large_ell_regimes/damping_sweep.py
uv run --extra dev pytest experiments/large_ell_regimes/test_damping_sweep.py experiments/large_ell_regimes/test_regimes.py -q
```
