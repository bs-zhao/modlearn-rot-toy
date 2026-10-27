"""T11 reproduces the CR3/CR2-4P logit exactly, under unit-cube mapping.

Source: modlearn-rot-2/experiment/choice/research_based/s0_cr2_design_recovery.py
weighting(p,gamma)=expit(gamma*logit(clip(p,1e-12,1-1e-12)))
V = wg*(gain/100)**alpha - lambda*wl*(loss/100)**alpha.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from scipy.special import expit,logit
from rottoy.models import MODELS,get_variant,evaluate
from rottoy.backends import AnalyticBackend
from rottoy.sampling import paired_grid
from rottoy.operators import derivative_bank
from rottoy.toy_models.t11 import physical_parameters,ETA_ABS_BOUND

def cr3_numpy_reference(x,theta):
    """Independent literal transcription of the original CR3's numpy code."""
    a=.02+(.999-.02)*theta[:,0]
    lam=.3+(5-.3)*theta[:,1]
    gamma=.02+(.999-.02)*theta[:,2]
    tau=.5+(4-.5)*theta[:,3]
    raw=x.copy()
    raw[:,[0,1,3,4]] *= 200.
    def w(p):return expit(gamma*logit(np.clip(p,1e-12,1-1e-12)))
    def value(g,l,p):
        return w(p)*np.power(g/100.,a)-lam*w(1-p)*np.power(l/100.,a)
    return tau*(value(raw[:,0],raw[:,1],raw[:,2])-
                value(raw[:,3],raw[:,4],raw[:,5]))

def test():
    assert "T11" in MODELS
    assert MODELS["T11"].k==4 and MODELS["T11"].dx==6
    assert np.allclose(MODELS["T11"].eta_bounds,[-ETA_ABS_BOUND,ETA_ABS_BOUND])
    X=np.array([
        [.75,.10,.35,.25,.50,.65],
        [.50,.00,.75,.00,.95,.25],
        [.01,.90,.10,.65,.20,.80],
        [.32,.73,.50,.64,.21,.50],
    ],dtype=np.float64)
    theta=np.array([
        [.75,.52,.40,.23],
        [.01,.91,.83,.96],
        [.95,.20,.17,.34],
        [.42,.62,.80,.88],
    ],dtype=np.float64)
    expect=cr3_numpy_reference(X,theta)
    got=evaluate("T11-id",torch.tensor(X),torch.tensor(theta)).detach().numpy()
    assert np.allclose(got,expect,atol=1e-12,rtol=1e-12),(got,expect)
    sig=evaluate("T11-sig",torch.tensor(X),torch.tensor(theta)).detach().numpy()
    assert np.allclose(sig,expit(expect),atol=1e-12,rtol=1e-12)
    inverse=evaluate("T11-sig",torch.tensor(X),torch.tensor(theta),"inverse").detach().numpy()
    assert np.allclose(inverse,expect,atol=1e-12,rtol=1e-12)
    assert (got>=-ETA_ABS_BOUND).all() and (got<=ETA_ABS_BOUND).all()

    # Structural zero 1: no loss outcomes -> lambda does not enter eta.
    ga=torch.tensor(X.copy(),dtype=torch.float64)
    ga[:,1]=0;ga[:,4]=0
    ut=torch.tensor(theta,requires_grad=True)
    e=evaluate("T11-id",ga,ut)
    grad=torch.autograd.grad(e.sum(),ut)[0]
    assert float(grad[:,1].abs().max()) < 1e-12

    # Structural zero 2: pA=pB=.5 -> gamma has no effect.
    gn=torch.tensor(X.copy(),dtype=torch.float64)
    gn[:,2]=gn[:,5]=.5
    ut=torch.tensor(theta,requires_grad=True)
    e=evaluate("T11-id",gn,ut)
    grad=torch.autograd.grad(e.sum(),ut)[0]
    assert float(grad[:,2].abs().max()) < 1e-12

    # Full generic X/U derivatives and ALL six transform versions.
    xx,tt,_,_=paired_grid(4,6,4,8,4009)
    bank=derivative_bank(AnalyticBackend("T11-sig"),xx,tt,4,8)
    for name in ("gtheta","htheta","hxtheta","gx","y","hxx"):
        assert np.isfinite(getattr(bank,name)).all(),name
    for suffix in ("id","sig","sq","tanh","sp","exp"):
        v=get_variant(f"T11-{suffix}")
        val=evaluate(v,torch.tensor(xx),torch.tensor(tt))
        assert torch.isfinite(val).all(),suffix
        lo,hi=v.y_bounds
        assert np.all(val.detach().numpy()>=lo-1e-10) and np.all(val.detach().numpy()<=hi+1e-10)
    print("PASS T11: CR3 numeric formula, sigmoid, bounds, null-condition derivatives, generic derivative bank")

if __name__=="__main__":
    test()
