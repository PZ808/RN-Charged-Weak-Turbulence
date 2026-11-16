"""Forced kinetic damping comparison. No base tools are modified."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from regimes import Backend, kinetic_operator, ROOT


def profiles(ell, h, rho4, boundary_fraction=.75, injection=1.):
    """Equal injection and equal uniform-state weighted damping, not equal N."""
    g=2*ell+1.
    S=np.where((ell>=2)&(ell<=4),1.,0.)
    S*=injection/(g@S)
    physical=rho4*(.5*h)/(2*np.pi)  # exactly existing _kinetic convention
    mask=ell>=int(np.ceil(boundary_fraction*ell[-1]))
    if ell[mask][0]<=4:
        raise ValueError('Need a gap between forcing and boundary sink.')
    boundary=mask.astype(float)*(g@physical)/(g@mask)
    return S,physical,boundary,mask


def diagnostics(W,g,n,S,D):
    C=kinetic_operator(W,n)
    rhs=C+S-D*n
    I=float(g@S)
    flux=-np.cumsum(g*C)
    # Valid away from stationarity as well: Pi=cumsum(g(S-Dn-n')).
    budget_flux=np.cumsum(g*(S-D*n-rhs))
    eq=(g@n)/g.sum()
    return dict(C=C,rhs=rhs,flux=flux,budget_flux=budget_flux,
                injection=I,loss=float(g@(D*n)),N=float(g@n),
                relative_RJ_distance=float(np.sqrt(g@((n/eq-1)**2)/g.sum())),
                stationarity=float(g@abs(rhs)/I))


def evolve(W,ell,S,D,mask,max_time=1e6,steady_tol=1e-7,rtol=2e-8,atol=2e-12):
    g=2*ell+1.
    # Same initial total action for every cutoff, profile and damping rate.
    n0=np.full(len(ell),1/g.sum())
    N0=float(g@n0)
    y=np.r_[n0,0.]
    history_t=[0.]; history_n=[n0.copy()]
    time=0.; endpoint=1.
    def rhs(t,y):
        n=y[:-1]
        return np.r_[kinetic_operator(W,n)+S-D*n,g@(D*n)]
    while time<max_time:
        target=min(endpoint,max_time)
        ts=np.linspace(time,target,17)[1:]
        sol=solve_ivp(rhs,(time,target),y,t_eval=ts,method='LSODA',
                      rtol=rtol,atol=atol)
        if not sol.success: raise RuntimeError(sol.message)
        if sol.y[:-1].min() < -10*atol:
            raise RuntimeError('Resolved negative occupation; refine solver, do not clip.')
        y=sol.y[:,-1]; time=float(sol.t[-1])
        history_t.extend(sol.t.tolist()); history_n.extend(sol.y[:-1].T.copy())
        diag=diagnostics(W,g,y[:-1],S,D)
        if diag['stationarity']<steady_tol: break
        endpoint*=4
    n=y[:-1]
    if np.min(n)<=0: raise RuntimeError('Final occupation is nonpositive.')
    diag=diagnostics(W,g,n,S,D)
    I=diag['injection']
    gap=(ell>4)&(ell<ell[mask][0])
    predicted=np.cumsum(g*(S-D*n))
    scale=max(N0+I*time,abs(y[-1]),diag['N'])
    report=dict(converged=bool(diag['stationarity']<steady_tol),
        reached_time=time,stationarity=diag['stationarity'],
        injection=I,loss=diag['loss'],N=diag['N'],
        relative_RJ_distance=diag['relative_RJ_distance'],
        relative_injection_loss_error=float(abs(I-diag['loss'])/I),
        relative_integrated_action_error=float(abs(diag['N']-N0-I*time+y[-1])/scale),
        steady_flux_balance_error=float(np.max(abs(diag['flux']-predicted))/I),
        terminal_flux_error=float(abs(diag['flux'][-1])/I),
        gap_flux_min=float(np.min(diag['flux'][gap])/I),
        gap_flux_max=float(np.max(diag['flux'][gap])/I),
        gap_flux_spread=float(np.ptp(diag['flux'][gap])/I),
        gap_loss_fraction=float(g[gap]@(D*n)[gap]/I),
        minimum_occupation=float(n.min()))
    data=dict(ell=ell,g=g,S=S,D=D,n=n,t=np.array(history_t),
              history_n=np.array(history_n),flux=diag['flux'],
              steady_budget_flux=predicted,cumulative_loss=np.array(y[-1]))
    return data,report


def make_plots(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cutoffs=sorted(set(x['lmax'] for x in records))
    rhos=sorted(set(x['rho4'] for x in records))
    fig,ax=plt.subplots(len(cutoffs),3,figsize=(14,4*len(cutoffs)),squeeze=False,layout='constrained')
    colors=plt.cm.viridis(np.linspace(.05,.9,len(rhos)))
    for row,L in enumerate(cutoffs):
        for rec in [r for r in records if r['lmax']==L]:
            d=rec['data']; color=colors[rhos.index(rec['rho4'])]
            style='-' if rec['kind']=='physical' else '--'
            label=f"{rec['kind']}, rho4={rec['rho4']:g}"
            eq=rec['report']['N']/d['g'].sum()
            ax[row,0].loglog(d['ell'],d['n']/eq,style,color=color,label=label)
            ax[row,1].plot(d['ell'],d['flux']/rec['report']['injection'],style,color=color)
            loss=d['g']*d['D']*d['n']/rec['report']['injection']
            ax[row,2].plot(d['ell'],np.cumsum(loss),style,color=color)
        onset=int(np.ceil(.75*L))
        for a in ax[row]:
            a.axvspan(2,4,color='grey',alpha=.12)
            a.axvline(onset,color='grey',ls=':',lw=1)
            a.set_xlabel('Angular shell ell')
        ax[row,0].axhline(1,color='k',lw=.5)
        ax[row,0].set(title=f'Lmax={L}: normalized steady spectrum',ylabel='n / instantaneous RJ reference')
        ax[row,0].legend(fontsize=6,ncol=2)
        ax[row,1].axhline(1,color='k',lw=.5)
        ax[row,1].set(title='Action flux / injection',ylabel='Pi(L) / I')
        ax[row,2].set(title='Cumulative absorption / injection',ylabel='sum(g D n) / I')
    fig.suptitle('Matched forcing; matched damping for uniform occupation. Solid: distributed QNM loss; dashed: boundary sink.')
    for ext in ('png','pdf'): fig.savefig(out/f'damping_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for L in cutoffs:
        for kind in ('physical','boundary'):
            rows=sorted([r for r in records if r['lmax']==L and r['kind']==kind],key=lambda r:r['rho4'])
            x=[r['rho4'] for r in rows]
            label=f'{kind}, Lmax={L}'
            style='o-' if kind=='physical' else 's--'
            ax[0].loglog(x,[r['report']['relative_RJ_distance'] for r in rows],style,label=label)
            ax[1].semilogx(x,[r['report']['gap_flux_spread'] for r in rows],style,label=label)
            ax[2].loglog(x,[r['report']['N'] for r in rows],style,label=label)
    for a in ax: a.set_xlabel('rho4'); a.legend(fontsize=7)
    ax[0].set(title='Departure from RJ shape',ylabel='Weighted relative RMS distance')
    ax[1].set(title='Flux variation between source and sink',ylabel='(max Pi - min Pi) / I')
    ax[2].set(title='Steady total action',ylabel='N')
    for ext in ('png','pdf'): fig.savefig(out/f'damping_diagnostics.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoffs',nargs='+',type=int,default=[16,24])
    p.add_argument('--rho4',nargs='+',type=float,default=[.01,.1,1.,10.])
    p.add_argument('--max-time',type=float,default=1e6)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'damping_results')
    args=p.parse_args()
    if min(args.cutoffs)<12 or min(args.rho4)<=0 or args.max_time<=0:
        p.error('cutoffs >=12, rho4>0 and max-time>0 required')
    args.out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path(__file__).with_name('regimes.py'),
           ROOT/'Large_ell/large_ell_backend.py',ROOT/'Large_ell/large_ell_program.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    records=[]; reports=[]
    for L in args.cutoffs:
        be=Backend(qhat=.6,rplus=1.,lmax=L)
        W,ell=be.W_shell(lmin=1)
        for rho4 in args.rho4:
            S,physical,boundary,mask=profiles(ell,be.h(ell),rho4)
            for kind,D in [('physical',physical),('boundary',boundary)]:
                print(f'Lmax={L} rho4={rho4:g} {kind}',flush=True)
                data,report=evolve(W,ell,S,D,mask,max_time=args.max_time)
                refined,check=evolve(W,ell,S,D,mask,max_time=args.max_time,
                                     steady_tol=1e-8,rtol=2e-9,atol=2e-13)
                report['refinement_relative_n_error']=float(np.max(abs(data['n']-refined['n']))/np.max(refined['n']))
                report['refinement_flux_error_over_injection']=float(np.max(abs(data['flux']-refined['flux']))/report['injection'])
                report['refined_converged']=check['converged']
                report.update(lmax=L,rho4=rho4,kind=kind)
                report['passed']=bool(report['converged'] and check['converged']
                    and report['relative_integrated_action_error']<1e-6
                    and report['steady_flux_balance_error']<1e-6
                    and report['relative_injection_loss_error']<1e-6
                    and report['refinement_relative_n_error']<1e-5
                    and report['refinement_flux_error_over_injection']<1e-5)
                name=f'L{L}_{kind}_rho{rho4:g}'
                np.savez_compressed(args.out/f'{name}.npz',**data)
                records.append(dict(lmax=L,rho4=rho4,kind=kind,data=data,report=report))
                reports.append(report)
                print(f"  steady={report['converged']} t={report['reached_time']:.3g} residual={report['stationarity']:.2g}",flush=True)
    if hashes!={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}:
        raise RuntimeError('Sources changed during execution; rerun.')
    report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_sha256=hashes,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                matching='Identical g-weighted injection and sum g D; actual steady N is not matched.',
                runs=reports,passed=all(r['passed'] for r in reports))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    make_plots(records,args.out)
    if not report['passed']: raise RuntimeError('Some cases failed convergence/checks; inspect report.json.')
    print('All damping sweep checks passed.',flush=True)


if __name__=='__main__': main()
