"""Isolated large-ell experiments; imports the existing backends unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import platform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Large_ell'))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/qnm-regimes-mpl')
import numpy as np
from scipy.integrate import solve_ivp
from large_ell_backend import Backend
from large_ell_program import setup, pump, pump_operators, ham_project


def kinetic_operator(W, n):
    """Same gain/loss contraction as the existing _kinetic; no clipping."""
    return (np.einsum('lijk,i,j,k->l', W, n, n, n, optimize=True)
            + n * np.einsum('lijk,i,j->l', W, n, n, optimize=True)
            - n * np.einsum('lijk,j,k->l', W, n, n, optimize=True)
            - n * np.einsum('lijk,i,k->l', W, n, n, optimize=True))


def nonlinear(C, a):
    return np.einsum('lijk,i,j,k->l', C, a, a, a.conj(), optimize=True)


def real_matrix(M, P, damping):
    """Jacobian of -i(M A + P A*) - damping A in (Re A, Im A)."""
    B, E = -1j * M - np.diag(damping), -1j * P
    return np.block([[(B + E).real, -(B - E).imag],
                     [(B + E).imag, (B - E).real]])


def coherent_parts(H, R, low, high, cl, gam, eta, rho, A):
    a = np.zeros(H.shape[0], complex)
    a[low], a[high] = cl, A
    internal = a.copy()
    internal[low] = 0
    # H internal transfer sums to zero over high modes. The difference is
    # exchange with the externally maintained reservoir, including all orders.
    fH = -1j * nonlinear(H[high], a)
    fI = -1j * nonlinear(H[high], internal)
    fR = -1j * eta * nonlinear(R[high], a)
    fD = -rho * gam * A
    rates = np.array([2 * np.vdot(A, fH - fI).real,
                      2 * np.vdot(A, fR).real,
                      2 * np.vdot(A, fD).real])
    return fH + fR + fD, rates, 2 * np.vdot(A, fI).real


def integrate_checked(fun, span, y0, times, rtol, atol, **kwargs):
    method = kwargs.pop('method', 'DOP853')
    sol = solve_ivp(fun, span, y0, t_eval=times, rtol=rtol, atol=atol,
                    method=method, **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def kinetic_run(lmax, end, samples, rtol=2e-9, atol=2e-11):
    be = Backend(qhat=0.6, rplus=1., lmax=lmax)
    W, ell = be.W_shell(lmin=1)
    g = 2 * ell + 1.
    # Positive background makes entropy well-defined. Mean occupation = 1;
    # absolute kinetic time therefore refers to this explicitly chosen scale.
    n0 = np.full(len(ell), 1e-3)
    n0[(ell >= 2) & (ell <= 4)] = 1
    n0 *= g.sum() / (g @ n0)
    times = np.r_[0., np.geomspace(end * 1e-7, end, samples - 1)]
    sol = integrate_checked(lambda t, n: kinetic_operator(W, n), (0, end),
                            n0, times, rtol, atol, method='LSODA')
    n = sol.y.T
    if n.min() <= 0:
        raise RuntimeError('Nonpositive occupation; reduce tolerances, do not clip.')
    C = np.array([kinetic_operator(W, v) for v in n])
    N = n @ g
    eq = N[0] / g.sum()
    distance = np.sqrt(((n / eq - 1)**2 @ g) / g.sum())
    entropy = np.log(n) @ g
    flux = -np.cumsum(C * g, axis=1)
    rng = np.random.default_rng(17)
    probe = rng.uniform(.2, 1.8, len(ell))
    cp = kinetic_operator(W, probe)
    conservation = abs(g @ cp) / max(g @ abs(cp), 1e-30)
    eq_res = np.max(abs(kinetic_operator(W, np.full(len(ell), eq))))
    # C[a n] = a^3 C[n] => n_a(T)=a n(a^2 T).
    cubic = np.linalg.norm(kinetic_operator(W, 1.7 * probe) - 1.7**3 * cp)
    cubic /= max(np.linalg.norm(1.7**3 * cp), 1e-30)
    # Exact identity S_eq-S = sum g log(n_eq/n) at fixed total action.
    entropy_gap = g.sum() * np.log(eq) - entropy
    return dict(ell=ell, g=g, t=times, n=n, flux=flux, distance=distance,
                entropy=entropy, entropy_gap=entropy_gap, N=N, equilibrium=eq), dict(
        random_state_action_residual=float(conservation),
        RJ_collision_residual=float(eq_res), cubic_scaling_residual=float(cubic),
        relative_action_drift=float(np.max(abs(N/N[0]-1))),
        minimum_occupation=float(n.min()),
        minimum_entropy_increment=float(np.diff(entropy).min()),
        minimum_entropy_gap=float(entropy_gap.min()),
        final_relative_RJ_distance=float(distance[-1]),
        terminal_flux_relative=float(np.max(abs(flux[:,-1])) /
                                     max(np.max(abs(flux)), 1e-30)))


def coherent_run(lmax, seed, end, samples, rtol=2e-9, atol=2e-11):
    be, C, ell, low, high, hl = setup(lmax)
    H = ham_project(C)
    R = C - H
    cl = pump(len(low), seed)
    gam = .5 * be.h(hl)
    etas, rhos = np.linspace(0, 1, 9), np.linspace(0, 8, 17)
    growth = np.empty((len(etas), len(rhos)))
    for i, eta in enumerate(etas):
        _, M, P = pump_operators(H + eta * R, low, high, cl)
        for j, rho in enumerate(rhos):
            growth[i,j] = np.linalg.eigvals(real_matrix(M, P, rho*gam)).real.max()
    # Local linearization is at A_high=0. F is usually nonzero, so this is
    # a homogeneous response rate, not stability of a stationary solution.
    _, M, P = pump_operators(H + .6 * R, low, high, cl)
    J = real_matrix(M, P, 2. * gam)
    rng = np.random.default_rng(5)
    v = rng.normal(size=2*len(high))
    z = v[:len(high)] + 1j*v[len(high):]
    delta = 1e-6
    def rhs(A):
        return coherent_parts(H,R,low,high,cl,gam,.6,2.,A)[0]
    fd = (rhs(delta*z)-rhs(-delta*z))/(2*delta)
    fd = np.r_[fd.real, fd.imag]
    jac_error = np.linalg.norm(fd-J@v)/np.linalg.norm(J@v)
    records, summaries = [], []
    budget_etas, budget_rhos = np.linspace(0,1,5), np.linspace(0,8,5)
    budget_grid = np.empty((5,5,3))
    stop_grid = np.zeros((5,5),bool)
    # Finite-time averages, including below/above the expected threshold.
    selected = [(0.,0.), (0.,4.), (.5,2.), (1.,2.), (1.,6.)]
    for eta in budget_etas:
      for rho in budget_rhos:
        A0 = np.full(len(high), 1e-5, complex)
        times = np.linspace(0,end,samples)
        def fun(t,y):
            f,b,_ = coherent_parts(H,R,low,high,cl,gam,eta,rho,y[:len(high)])
            return np.r_[f, b.astype(complex)]
        def stop(t,y):
            return 100. - np.sum(abs(y[:len(high)])**2)
        stop.terminal, stop.direction = True, -1
        sol = integrate_checked(fun,(0,end),np.r_[A0,np.zeros(3,complex)],
                                times,rtol,atol,events=stop,max_step=.02)
        if len(sol.t_events[0]) and sol.t_events[0][0] > sol.t[-1]:
            # Include the actual stopping point rather than the last output
            # sample, which can substantially underestimate rapidly growing N.
            sol.t = np.r_[sol.t,sol.t_events[0][0]]
            sol.y = np.column_stack([sol.y,sol.y_events[0][0]])
        A = sol.y[:len(high)].T
        cumulative = sol.y[len(high):].real.T
        N = np.sum(abs(A)**2,axis=1)
        parts = [coherent_parts(H,R,low,high,cl,gam,eta,rho,a) for a in A]
        rates = np.array([x[1] for x in parts])
        internal_error = max(abs(x[2]) for x in parts)
        residual = N-N[0]-cumulative.sum(axis=1)
        scale = max(np.max(N),np.max(np.sum(abs(cumulative),axis=1)),1e-20)
        summary = dict(eta=eta,rho=rho, reached_time=float(sol.t[-1]),
                       stopped_at_action_limit=bool(len(sol.t_events[0])),
                       relative_integrated_budget_error=float(max(abs(residual))/scale),
                       internal_H_action_rate_error=float(internal_error),
                       final_action=float(N[-1]),
                       mean_rates=(cumulative[-1]/sol.t[-1]).tolist())
        summaries.append(summary)
        budget_grid[int(round(eta*4)),int(round(rho/2))] = cumulative[-1]/sol.t[-1]
        stop_grid[int(round(eta*4)),int(round(rho/2))] = bool(len(sol.t_events[0]))
        if (eta,rho) in selected:
            records.append(dict(eta=eta,rho=rho,t=sol.t,N=N,rates=rates,
                                cumulative=cumulative,budget_residual=residual,A=A,
                                stopped=bool(len(sol.t_events[0]))))
    return dict(etas=etas,rhos=rhos,growth=growth,ell=hl,records=records,
                budget_etas=budget_etas,budget_rhos=budget_rhos,budget_grid=budget_grid,
                stop_grid=stop_grid), dict(
        finite_difference_J_relative_error=float(jac_error),runs=summaries,
        H_projection_idempotence=float(np.linalg.norm(ham_project(H)-H)/np.linalg.norm(H)))


def plot_results(k, c, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ix = np.unique(np.linspace(0,len(k['t'])-1,6).astype(int))
    for j in ix:
        label=f"T={k['t'][j]:.3g}"
        ax[0].loglog(k['ell'],k['n'][j],label=label)
        ax[1].plot(k['ell'],k['flux'][j],label=label)
    ax[0].axhline(k['equilibrium'],color='k',ls='--',label='finite-network RJ')
    ax[0].set(xlabel=r'$\ell$',ylabel=r'$n_\ell$ (per mode)',title='Closed conservative kinetics')
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Shell boundary L',ylabel=r'$\Pi(L,T)$',title='Instantaneous action flux')
    ax[1].axhline(0,color='k',lw=.5)
    ax[2].loglog(k['t'][1:],np.maximum(k['distance'][1:],1e-16))
    ax[2].set(xlabel='Kinetic time T',ylabel='Relative weighted RMS distance',title='Distance from RJ')
    for suffix in ('png','pdf'): fig.savefig(out/f'kinetic.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    im=ax[0,0].pcolormesh(c['rhos'],c['etas'],c['growth'],shading='nearest',cmap='coolwarm')
    ax[0,0].contour(c['rhos'],c['etas'],c['growth'],levels=[0],colors='k')
    ax[0,0].set(xlabel=r'$\rho$',ylabel=r'$\eta$',title='Homogeneous response rate at zero high-band amplitude')
    fig.colorbar(im,ax=ax[0,0],label=r'$\max\,\Re\lambda$')
    for r in c['records']:
        label=f"eta={r['eta']:g}, rho={r['rho']:g}"
        ax[0,1].semilogy(r['t'],r['N'],label=label)
        if r['stopped']: ax[0,1].plot(r['t'][-1],r['N'][-1],'kx',ms=7)
    ax[0,1].set(xlabel=r'Slow time $\tau$',ylabel=r'$N_{high}$',title='Full cubic evolution; frozen unit-action pump')
    ax[0,1].legend(fontsize=7)
    for j,label in enumerate(('Hamiltonian pump exchange','Residual interaction','QNM loss (signed)')):
        im=ax[1,j].pcolormesh(c['budget_rhos'],c['budget_etas'],c['budget_grid'][:,:,j],
                              shading='nearest',cmap='coolwarm')
        ax[1,j].set(xlabel=r'$\rho$',ylabel=r'$\eta$',title=label)
        fig.colorbar(im,ax=ax[1,j],label='Integrated action contribution / elapsed time')
        ii,jj=np.where(c['stop_grid'])
        ax[1,j].scatter(c['budget_rhos'][jj],c['budget_etas'][ii],marker='x',c='k',s=20)
    fig.suptitle('Fundamental axisymmetric tower; crosses: action limit reached (not saturation)',fontsize=10)
    r=c['records'][-1]
    ax[0,2].plot(r['t'],r['N']-r['N'][0],label='Numerical action change')
    ax[0,2].plot(r['t'],r['cumulative'].sum(axis=1),'--',label='Integrated analytic balance')
    ax[0,2].set(xlabel=r'$\tau$',ylabel=r'$\Delta N_{high}$',title=f"Budget closure: eta={r['eta']:g}, rho={r['rho']:g}")
    ax[0,2].legend(fontsize=8)
    for suffix in ('png','pdf'): fig.savefig(out/f'coherent.{suffix}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lmax',type=int,default=16)
    p.add_argument('--kinetic-end',type=float,default=30.)
    p.add_argument('--coherent-end',type=float,default=.5)
    p.add_argument('--samples',type=int,default=301)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args=p.parse_args()
    if args.lmax<6 or args.samples<10 or min(args.kinetic_end,args.coherent_end)<=0:
        p.error('Need lmax >= 6, samples >= 10, and positive durations.')
    args.out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'Large_ell/large_ell_backend.py',ROOT/'Large_ell/large_ell_program.py']
    source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    print('Running closed kinetic experiment...',flush=True)
    k,kchecks=kinetic_run(args.lmax,args.kinetic_end,args.samples)
    print('Running coherent growth and 25 action-budget trajectories...',flush=True)
    c,cchecks=coherent_run(args.lmax,args.seed,args.coherent_end,args.samples)
    # Independent tighter solve tests trajectories, not just algebraic identities.
    print('Repeating both experiments at tenfold tighter tolerances...',flush=True)
    kt,_=kinetic_run(args.lmax,args.kinetic_end,args.samples,rtol=2e-10,atol=2e-12)
    ct,_=coherent_run(args.lmax,args.seed,args.coherent_end,args.samples,rtol=2e-10,atol=2e-12)
    kchecks['tolerance_refinement_relative_n_error']=float(np.max(abs(k['n']-kt['n']))/np.max(abs(kt['n'])))
    convergence=[]
    for r,s in zip(c['records'],ct['records']):
        count=min(len(r['t'])-int(r['stopped']),len(s['t'])-int(s['stopped']))
        convergence.append(float(np.max(abs(r['A'][:count]-s['A'][:count]))/max(np.max(abs(s['A'][:count])),1e-20)))
    cchecks['tolerance_refinement_relative_A_errors']=convergence
    cchecks['tolerance_refinement_relative_budget_grid_error']=float(
        np.max(abs(c['budget_grid']-ct['budget_grid'])) /
        max(np.max(abs(ct['budget_grid'])),1e-20))
    import scipy
    if source_hashes != {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:
        raise RuntimeError('Imported experiment sources changed during execution; rerun.')
    report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_sha256=source_hashes,
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                kinetic=kchecks,coherent=cchecks,
                conventions=dict(kinetic='T scales as eps^4 t; overall collision normalization as existing driver; mean initial occupation=1',
                                 coherent='tau=eps^2 t; alpha=1; unit-action frozen pump; rho=sigma/eps^2',
                                 comparison='rho4=rho/eps^2 at common eps; figures are not one shared physical trajectory'))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.out/'kinetic.npz',**k)
    arrays={key:c[key] for key in ('etas','rhos','growth','ell','budget_etas','budget_rhos','budget_grid','stop_grid')}
    for i,r in enumerate(c['records']):
        arrays.update({f'run{i}_{key}':val for key,val in r.items()})
    np.savez_compressed(args.out/'coherent.npz',**arrays)
    plot_results(k,c,args.out)
    checks=[kchecks['relative_action_drift']<1e-7,
            kchecks['random_state_action_residual']<1e-11,
            kchecks['RJ_collision_residual']<1e-10,
            kchecks['minimum_entropy_increment']> -1e-7,
            kchecks['tolerance_refinement_relative_n_error']<1e-6,
            cchecks['finite_difference_J_relative_error']<1e-7,
            max(convergence)<1e-6,
            cchecks['tolerance_refinement_relative_budget_grid_error']<1e-6,
            all(r['relative_integrated_budget_error']<1e-6 for r in cchecks['runs'])]
    report['acceptance_checks_passed']=bool(all(checks))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(acceptance_checks_passed=report['acceptance_checks_passed'],
                         kinetic=kchecks,coherent_max_budget_error=max(r['relative_integrated_budget_error'] for r in cchecks['runs']),
                         stopped_runs=int(c['stop_grid'].sum()),output=str(args.out)),indent=2))
    if not all(checks): raise RuntimeError('Consistency criteria failed; inspect report.json')


if __name__=='__main__': main()
