# Checked numerical examples

Both saved runs use seed 7, kinetic duration 30, coherent duration 0.5,
301 requested output times, and the fundamental tower. All numerical acceptance
criteria passed at lmax=12 and 16. Every experiment was repeated at tenfold
tighter relative and absolute solver tolerances. This is numerical solver
verification at two cutoffs, not a continuum convergence claim.

For lmax=16:

| Measure | Result |
|---|---:|
| Relative total-action drift, closed kinetics | 9.4e-13 |
| Final relative weighted RJ distance | 6.4e-13 |
| Relative kinetic trajectory change on tolerance refinement | 3.8e-10 |
| Full cubic Jacobian vs central finite difference | 4.1e-12 relative |
| Largest integrated coherent budget residual | 4.6e-10 relative |
| Largest selected coherent amplitude refinement difference | 5.3e-9 relative |
| Action-budget map refinement difference | 2.2e-10 relative to map maximum |

Entropy decreases only at roundoff level after equilibration (largest negative
sample increment approximately 1.4e-11 absolute). A distance near 1e-12 is a
numerical floor, not a measurement of physical accuracy to twelve digits.

An independent eigenvalue root gives rho_c=3.9044002529391664 at eta=1,
matching the notes' rounded 3.904. At eta=1,rho=6 the zero-high-band
homogeneous response rate is -6.364273708. Nevertheless the full forced
nonlinear evolution reaches the N_high=100 action limit before tau=0.5.
The initial-state response threshold does not establish global boundedness.
The figure identifies stopped runs with crosses; none is labeled saturation.
There are 10 stopped grid trajectories at lmax=16 and 9 at lmax=12.

The apparent growing branches should not be extrapolated as physical runaway
or a proven finite-time singularity. The maintained pump and truncated weakly
nonlinear equations need reassessment once high-band action becomes large.
Establishing physical saturation requires additional work.

The kinetic example shows outward transient redistribution and eventual
equipartition. It does not establish a KZ spectrum or an inertial interval.
The non-Hamiltonian residual supplies positive action in the saved coherent
examples, while QNM damping removes action. Their budgets are recorded
separately, so action production cannot be mistaken for conservative transport.

`results_L12/report.json` and `results_L16/report.json` contain detailed
per-run evidence and source hashes. Source hashes were checked again after
execution. All code changes are confined to this experiment directory; the
existing numerical tools and pre-existing local changes were left untouched.
