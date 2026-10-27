"""Original C2 and historical choice v1/v2 metric regression tests.

Run: python tests/test_historical_choice.py
"""
import sys, math, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from rottoy.models import MODELS,evaluate,plane_rotation
from rottoy.sampling import paired_grid
from rottoy.backends import AnalyticBackend,NeuralBackend,MLP
from rottoy.historical_choice import (build_historical_banks,
    historical_metrics,save_historical_banks)

def tests():
    assert "C2" in MODELS
    assert (MODELS["C2"].k,MODELS["C2"].dx)==(4,4)
    x=torch.tensor([[.8,.2,.6,.3],[.1,.7,.9,.4]],dtype=torch.float64)
    t=torch.tensor([[.2,.3,.4,.5],[.7,.6,.5,.4]],dtype=torch.float64)
    eta=(t[:,0]+math.sqrt(3)*t[:,1]*(x[:,0]-x[:,1])
         +(math.sqrt(5)/2)*t[:,2]*((3*x[:,0]**2-1)-(3*x[:,1]**2-1))
         +math.sqrt(3)*t[:,3]*(x[:,2]-x[:,3]))
    assert torch.allclose(evaluate("C2","identity",x,t),eta,atol=1e-13)
    assert torch.allclose(evaluate("C2","sigmoid",x,t),torch.sigmoid(eta),atol=1e-13)
    assert torch.allclose(evaluate("C2","sigmoid",x,t,"inverse-link"),eta,atol=1e-12)

    xx,tt,_,xp=paired_grid(4,4,6,24,2026)
    analytic=AnalyticBackend("C2","sigmoid")
    bank=build_historical_banks(analytic,"sigmoid",xx,tt,6,24)
    assert bank.fisher.shape==(4,4) and min(bank.fisher_eigenvalues)>-1e-10
    assert len(bank.names_all)==1+4+4+6
    assert len(bank.operators_v1)<=5
    assert len(bank.operators_v2)<=15
    names=list(bank.names_all)
    b3=2
    for row in ("d2X12_logodds_1","d2X22_logodds_1"):
        mat=bank.raw_matrices[names.index(row)]
        assert mat[b3,b3]>1e-4
        assert max(abs(mat[k,l]) for k in range(4) for l in range(4)
                   if (k,l)!=(b3,b3))<1e-9, (row,mat)
    assert np.allclose(bank.fisher,bank.fisher.T,atol=1e-12)
    for phi in (0,30,45,90):
        R=plane_rotation(4,1,2,phi)
        m=historical_metrics(bank,R)
        assert set(m)==set(("historical_v1_jd","historical_v2_jd",
            "historical_v1_level","historical_v1_first",
            "historical_v2_second_diagonal","historical_v2_second_mixed",
            "historical_fisher_offdiag","historical_fisher_trace"))
        assert min(m.values())>=-1e-10
        assert np.isclose(m["historical_fisher_trace"],np.trace(bank.fisher))
    with tempfile.TemporaryDirectory() as folder:
        name=Path(folder)/"operators.npz"
        save_historical_banks(name,bank)
        with np.load(name) as a:
            assert len(a["all_names"])==15
            assert a["v2_matrices"].shape[1:]==(4,4)
    # Both analytic and differentiable neural backends use the same code.
    network=MLP(8,width=10,depth=2)
    backend=NeuralBackend("C2","sigmoid",network)
    xb,tb,_,_=paired_grid(4,4,3,6,44)
    nbank=build_historical_banks(backend,"sigmoid",xb,tb,3,6,batch_size=16)
    assert np.isfinite(nbank.fisher).all()
    assert np.isfinite(list(historical_metrics(nbank,np.eye(4)).values())).all()
    print("PASS: exact C2 Bernoulli p, v1/v2 operator inventory, second-X signature, neural backend")

if __name__=="__main__":tests()
