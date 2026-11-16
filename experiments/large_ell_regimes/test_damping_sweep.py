import numpy as np
from numpy.testing import assert_allclose
from damping_sweep import profiles,diagnostics,evolve
from regimes import Backend


def test_profiles_match_uniform_loss_and_fixed_injection():
    for L in (12,16,24):
        ell=np.arange(1,L+1); g=2*ell+1.
        S,D,B,mask=profiles(ell,ell+1.,.1)
        assert_allclose(g@S,1.)
        assert_allclose(g@D,g@B)
        assert np.all(B[~mask]==0)
        assert np.all(S[ell>4]==0)
        assert np.all(D>0)


def test_nonstationary_flux_and_action_identity():
    be=Backend(qhat=.6,rplus=1.,lmax=12)
    W,ell=be.W_shell(lmin=1); g=2*ell+1.
    S,D,_,_=profiles(ell,be.h(ell),1.)
    n=np.random.default_rng(4).uniform(.01,.1,len(ell))
    d=diagnostics(W,g,n,S,D)
    assert_allclose(d['flux'],d['budget_flux'],atol=1e-12)
    assert_allclose(g@d['rhs'],g@S-g@(D*n),atol=1e-12)


def test_zero_collision_forced_decay_has_exact_solution():
    # An independent solvable balance, not another implementation of C.
    ell=np.arange(1,13); g=2*ell+1.
    W=np.zeros((12,)*4)
    S=np.full(12,1/g.sum()); D=np.full(12,2.)
    mask=ell>=9
    data,report=evolve(W,ell,S,D,mask,max_time=100)
    assert report['converged']
    n0=1/g.sum()
    expected=S/D+(n0-S/D)*np.exp(-2*data['t'][-1])
    assert_allclose(data['n'],expected,atol=1e-10)
    assert report['relative_integrated_action_error']<1e-8


def test_boundary_sink_stationary_throughput():
    be=Backend(qhat=.6,rplus=1.,lmax=12)
    W,ell=be.W_shell(lmin=1)
    S,_,D,mask=profiles(ell,be.h(ell),1.)
    data,report=evolve(W,ell,S,D,mask,max_time=1e4)
    assert report['converged']
    gap=(ell>4)&(ell<ell[mask][0])
    assert_allclose(data['flux'][gap],report['injection'],rtol=2e-7)
    assert report['relative_injection_loss_error']<1e-7
