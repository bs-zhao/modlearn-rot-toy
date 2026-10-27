"""Deterministic mathematical checks: python tests/test_toys.py (no pytest)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from rottoy.models import MODELS,WRAPPERS,evaluate,plane_rotation
from rottoy.sampling import paired_grid
from rottoy.operators import derivative_bank,rotated_derivatives
from rottoy.metrics import local_metrics,sweep_metrics

def bank(name,wrapper="identity",scale="raw"):
    info=MODELS[name]
    x,t,_,_=paired_grid(info.k,info.dx,16,24,123)
    return derivative_bank(name,wrapper,x,t,16,24,scale)

def tests():
    for name,m in MODELS.items():
        X=np.full((5,m.dx),.3);theta=np.full((5,m.k),.4)
        for w in WRAPPERS:
            y=evaluate(name,w,torch.tensor(X,dtype=torch.float64),
                torch.tensor(theta,dtype=torch.float64))
            assert y.shape==(5,) and torch.isfinite(y).all(),(name,w)
    b=bank("T2")
    for angle in (0,23,45,88):
        _,H,_=rotated_derivatives(b,plane_rotation(3,0,1,angle))
        assert np.max(abs(H))<1e-12
    b=bank("T3")
    m0=local_metrics(b,plane_rotation(2,0,1,0))
    m45=local_metrics(b,plane_rotation(2,0,1,45))
    assert abs(m0["uu_offdiag"]-1)<1e-12
    assert m45["uu_offdiag"]<1e-12
    assert m0["uu_diagonal"]<1e-12 and m45["uu_diagonal"]>.999
    X=np.column_stack((np.linspace(.01,.99,35),np.linspace(.99,.01,35)))
    s0=sweep_metrics("T3","identity",plane_rotation(2,0,1,0),X,31,.4,.2)
    s45=sweep_metrics("T3","identity",plane_rotation(2,0,1,45),X,31,.4,.2)
    assert s0["sweep_affine"]<1e-22 and s45["sweep_affine"]>1e-6
    b=bank("T1","identity","log")
    c0=local_metrics(b,plane_rotation(2,0,1,0))
    c45=local_metrics(b,plane_rotation(2,0,1,45))
    assert c0["grad_xcov_offdiag"]<1e-12
    assert abs(c45["grad_xcov_offdiag"]-.5)<1e-9
    b=bank("T3","square")
    a=local_metrics(b,plane_rotation(2,0,1,0))
    c=local_metrics(b,plane_rotation(2,0,1,37))
    for n in ("u_grad_energy","xu_total_energy","x_gradient_energy","x_curvature_energy"):
        assert np.isclose(a[n],c[n],atol=1e-10,rtol=1e-10),(n,a[n],c[n])
    assert np.linalg.matrix_rank(bank("T4").gtheta[0],tol=1e-9)==3
    assert np.linalg.matrix_rank(bank("T4ID").gtheta[0],tol=1e-9)==4
    r=plane_rotation(2,0,1,32)
    x0=torch.tensor([.36,.61],dtype=torch.float64)
    theta0=torch.tensor([.47,.58],dtype=torch.float64)
    rt=torch.tensor(r,dtype=torch.float64)
    u0=(rt@theta0).detach().requires_grad_(True)
    grad=torch.autograd.functional.jacobian(
        lambda u:evaluate("T3","sigmoid",x0,rt.T@u),u0)
    tn=theta0.detach().requires_grad_(True)
    gtheta=torch.autograd.functional.jacobian(
        lambda t:evaluate("T3","sigmoid",x0,t),tn)
    assert np.allclose(grad.detach().numpy(),gtheta.detach().numpy()@r.T,atol=1e-12)
    print("ALL TESTS PASS: analytic derivatives, T3 counterexample, T4 rank, invariance")
if __name__=="__main__":tests()
