"""Deterministic registry, mathematical and continuous-output regression checks."""
from pathlib import Path
import sys,math,re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from rottoy.models import MODELS,get_variant,variant_names,evaluate,plane_rotation
from rottoy.transforms import TRANSFORMS,apply_transform,inverse_transform
from rottoy.metrics import METRICS
from rottoy.backends import AnalyticBackend
from rottoy.sampling import paired_grid
from rottoy.operators import derivative_bank,rotated_derivatives,operator_bank,operator_metrics

def test():
    assert list(MODELS)==[f"T{i}" for i in range(1,10)]
    assert list(TRANSFORMS)==["id","sig","sq","tanh","sp","exp"]
    assert len(variant_names())==9*len(TRANSFORMS)
    assert list(METRICS)==[f"M{i:03d}" for i in range(1,25)]
    assert METRICS["M017"].key=="operator_v1_jd"
    assert METRICS["M018"].key=="operator_v2_jd"
    assert get_variant("T9-sig").id=="T9-sig"
    assert get_variant("T9").id=="T9-id"
    assert len({get_variant(v).id for v in variant_names()})==len(variant_names())
    for name,model in MODELS.items():
        x=torch.full((7,model.dx),.37,dtype=torch.float64)
        theta=torch.full((7,model.k),.63,dtype=torch.float64)
        for suffix in TRANSFORMS:
            variant=name+"-"+suffix
            v=evaluate(variant,x,theta)
            assert v.shape==(7,) and torch.isfinite(v).all(),variant
            assert v.is_floating_point(),variant
            if suffix=="sig":assert ((v>0)&(v<1)).all()
            if suffix=="sq":assert (v>=0).all()
            if suffix=="sp":assert (v>0).all()
            if TRANSFORMS[suffix].invertible:
                original=evaluate(variant,x,theta,"inverse")
                direct=evaluate(name+"-id",x,theta)
                assert torch.allclose(original,direct,atol=5e-8,rtol=5e-8),(variant,original,direct)
    try:evaluate("T3-sq",torch.ones((2,2)),torch.ones((2,2)),"inverse")
    except ValueError:pass
    else:raise AssertionError("Square inverse incorrectly allowed")

    x=torch.tensor([[.8,.2,.6,.3],[.1,.7,.9,.4]],dtype=torch.float64)
    b=torch.tensor([[.2,.3,.4,.5],[.7,.6,.5,.4]],dtype=torch.float64)
    y=b[:,0]+math.sqrt(3)*b[:,1]*(x[:,0]-x[:,1])+(
      math.sqrt(5)/2)*b[:,2]*(3*x[:,0]**2-3*x[:,1]**2)+math.sqrt(3)*b[:,3]*(x[:,2]-x[:,3])
    assert torch.allclose(evaluate("T9-id",x,b),y,atol=1e-13)
    assert torch.allclose(evaluate("T9-sig",x,b),torch.sigmoid(y),atol=1e-13)

    x,t,_,_=paired_grid(2,2,4,11,2026)
    bank=derivative_bank(AnalyticBackend("T3-id"),x,t,4,11)
    a=rotated_derivatives(bank,plane_rotation(2,0,1,0))
    b45=rotated_derivatives(bank,plane_rotation(2,0,1,45))
    Ha,Hb=a[1],b45[1]
    assert np.mean(Ha[...,0,1]**2)>1e-5
    assert np.max(abs(Hb[...,0,1]))<1e-10
    assert np.max(abs(Ha[...,0,0]))<1e-12
    assert np.mean(Hb[...,0,0]**2)>1e-5
    # T2 has zero UU Hessian in identity-link scale for every rotation.
    x,t,_,_=paired_grid(3,2,4,8,22)
    b2=derivative_bank(AnalyticBackend("T2-id"),x,t,4,8)
    assert np.max(abs(b2.htheta))<1e-12
    x,t,_,_=paired_grid(4,4,6,24,234)
    fb=operator_bank(AnalyticBackend("T9-sig"),x,t,6,24,scale="inverse")
    assert len(fb.names_all)==1+4+4+6
    for phi in (0,25,45):
        v=operator_metrics(fb,plane_rotation(4,1,2,phi))
        assert len(v)==8 and np.isfinite(list(v.values())).all()
    # T9 inverse-link has only one quadratic X dimension family.
    for name,matrix in zip(fb.names_all,fb.raw_matrices):
        if name in ("X1:second","X2:second"):
            assert matrix[2,2]>1e-8,(name,matrix)
            rest=matrix.copy();rest[2,2]=0
            assert np.max(abs(rest))<1e-8,(name,matrix)
    # Documentation must be complete; no second model-specific code path.
    root=Path(__file__).resolve().parents[1]
    tm=(root/"catalog"/"README_TOY_MODELS.md").read_text(encoding="utf-8")
    mm=(root/"catalog"/"README_METRICS.md").read_text(encoding="utf-8")
    toys=re.findall(r"(?m)^\| \*\*(T\d+)\*\* \|",tm)
    mids=re.findall(r"(?m)^\| \*\*(M\d{3})\*\* \|",mm)
    assert toys==list(MODELS),(toys,list(MODELS))
    # Range integrity: all model bounds must contain values on the physical cube.
    from rottoy.transforms import transformed_bounds
    for name,model in MODELS.items():
        assert len(model.eta_bounds)==2
        for code in TRANSFORMS:
            from rottoy.models import get_variant
            bounds=get_variant(name+"-"+code).y_bounds
            assert bounds==transformed_bounds(model.eta_bounds,code)
            xv=torch.rand((128,model.dx),dtype=torch.float64)
            tv=torch.rand((128,model.k),dtype=torch.float64)
            yv=evaluate(name+"-"+code,xv,tv).detach().numpy()
            assert np.min(yv)>=bounds[0]-1e-9,(name,code,bounds,np.min(yv))
            assert np.max(yv)<=bounds[1]+1e-9,(name,code,bounds,np.max(yv))
    assert mids==list(METRICS),(mids,list(METRICS))
    print("PASS: T1-T9, six continuous transforms, M001-M024, gradients and X/U integral operators")

if __name__=="__main__":test()
