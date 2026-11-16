"""Analytic identities and independent contractions for experiment diagnostics."""
import numpy as np
from numpy.testing import assert_allclose
from regimes import (Backend, kinetic_operator, nonlinear, real_matrix,
                     coherent_parts, setup, pump, pump_operators, ham_project)


def test_kinetic_against_literal_quartet_sum():
    W, ell = Backend(qhat=.6,rplus=1.,lmax=6).W_shell(lmin=1)
    n=np.random.default_rng(3).uniform(.2,2,len(ell))
    literal=np.zeros(len(n))
    for l,i,j,k in np.ndindex(W.shape):
        literal[l]+=W[l,i,j,k]*(n[i]*n[j]*n[k]+n[l]*n[i]*n[j]
                                     -n[l]*n[j]*n[k]-n[l]*n[i]*n[k])
    assert_allclose(kinetic_operator(W,n),literal,atol=1e-12,rtol=1e-12)
    g=2*ell+1
    C=kinetic_operator(W,n)
    assert abs(g@C)<1e-12*np.sum(g*abs(C))
    assert (g/n)@C>=-1e-12  # H theorem
    # Independent positive quartet expression; fixes the symmetry factor.
    entropy_production=0.
    for l,i,j,k in np.ndindex(W.shape):
        bracket=1/n[l]+1/n[k]-1/n[i]-1/n[j]
        entropy_production+=g[l]*W[l,i,j,k]*n[l]*n[i]*n[j]*n[k]*bracket**2/4
    assert_allclose((g/n)@C,entropy_production,rtol=1e-12)
    flux=-np.cumsum(g*C)
    assert_allclose(np.diff(np.r_[0.,flux]),-g*C,atol=1e-12)
    assert_allclose(kinetic_operator(W,np.ones(len(n))),0,atol=1e-12)
    assert_allclose(kinetic_operator(W,2*n),8*C,atol=1e-12)


def test_coherent_budget_and_projection():
    be,C,ell,low,high,hl=setup(8)
    H=ham_project(C); R=C-H
    assert_allclose(ham_project(H),H,atol=1e-13)
    assert_allclose(H+R,C,atol=1e-13)
    a=np.random.default_rng(8).normal(size=len(ell))+1j*np.arange(len(ell))*.01
    assert abs(2*np.vdot(a,-1j*nonlinear(H,a)).real)<1e-10
    A=a[high]; cl=pump(len(low)); gam=.5*be.h(hl)
    f,b,internal=coherent_parts(H,R,low,high,cl,gam,.4,2.,A)
    assert abs(internal)<1e-10
    assert_allclose(2*np.vdot(A,f).real,b.sum(),atol=1e-10)
    assert_allclose(b[2],-4*np.sum(gam*abs(A)**2),atol=1e-12)


def test_real_jacobian_and_pump_scaling():
    be,C,ell,low,high,hl=setup(8)
    cl=pump(len(low)); F,M,P=pump_operators(C,low,high,cl)
    F2,M2,P2=pump_operators(C,low,high,2*cl)
    assert_allclose(F2,8*F)
    assert_allclose(M2,4*M)
    assert_allclose(P2,4*P)
    z=np.arange(len(high))+.3j
    damping=.5*be.h(hl)
    expected=-1j*(M@z+P@z.conj())-damping*z
    assert_allclose(real_matrix(M,P,damping)@np.r_[z.real,z.imag],
                    np.r_[expected.real,expected.imag],atol=1e-12)


def test_one_mode_sign_units_and_exact_decay():
    # i A'=(u+iv)|A|^2 A-id A gives n'=2v n^2-2d n.
    from regimes import integrate_checked
    A=.7+.2j; u,v,d=.3,.4,.8
    f=-1j*(u+1j*v)*abs(A)**2*A-d*A
    assert_allclose(2*(A.conjugate()*f).real,2*v*abs(A)**4-2*d*abs(A)**2)
    t=np.linspace(0,2,51)
    s=integrate_checked(lambda t,a:-d*a,(0,2),np.array([A]),t,1e-11,1e-13)
    assert_allclose(abs(s.y[0])**2,abs(A)**2*np.exp(-2*d*t),rtol=2e-10)
    # Time conversion: eps^2/eps^4 times; amplitude vs occupation loss factor 2.
    eps,sigma,gamma=.1,1e-5,3.
    rho,rho4=sigma/eps**2,sigma/eps**4
    assert_allclose(rho4,rho/eps**2)
    physical_t=12.
    assert_allclose(np.exp(-rho*gamma*(eps**2*physical_t)),
                    np.exp(-sigma*gamma*physical_t))


def test_linear_response_against_matrix_exponential():
    from scipy.linalg import expm
    from regimes import integrate_checked
    _,C,_,low,high,hl=setup(8)
    _,M,P=pump_operators(C,low,high,pump(len(low)))
    J=real_matrix(M,P,np.zeros(len(high)))
    x0=np.random.default_rng(3).normal(size=2*len(high))
    t=np.linspace(0,.1,21)
    sol=integrate_checked(lambda t,x:J@x,(0,.1),x0,t,1e-11,1e-13)
    assert_allclose(sol.y[:,-1],expm(.1*J)@x0,rtol=1e-9,atol=1e-11)


def test_seed7_threshold_against_notes_reference():
    # A regression against the notes, separate from the analytic identity tests.
    from scipy.optimize import brentq
    be,C,_,low,high,hl=setup(16)
    _,M,P=pump_operators(C,low,high,pump(len(low),7))
    def growth(rho):
        return np.linalg.eigvals(real_matrix(M,P,rho*.5*be.h(hl))).real.max()
    assert_allclose(brentq(growth,0,8),3.904,atol=5e-4,rtol=0)
