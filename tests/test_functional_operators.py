"""Original C2 and historical choice v1/v2 metric regression tests.

Run: python tests/test_functional_operators.py
"""
import sys, math, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from rottoy.models import MODELS,evaluate,plane_rotation
from rottoy.sampling import paired_grid
from rottoy.backends import AnalyticBackend,NeuralBackend,MLP
from rottoy.operators import (build_operator_bank,
    operator_family_metrics,save_operator_bank)

def tests():
    assert "T9" in MODELS and "C2" not in MODELS and "T4ID" not in MODELS
    assert (MODELS["T9"].k,MODELS["T9"].dx)==(4,4)
    x=torch.tensor([[.8,.2,.6,.3],[.1,.7,.9,.4]],dtype=torch.float64)
    t=torch.tensor([[.2,.3,.4,.5],[.7,.6,.5,.4]],dtype=torch.float64)
    eta=(t[:,0]+math.sqrt(3)*t[:,1]*(x[:,0]-x[:,1])
         +(math.sqrt(5)/2)*t[:,2]*((3*x[:,0]**2-1)-(3*x[:,1]**2-1))
         +math.sqrt(3)*t[:,3]*(x[:,2]-x[:,3]))
    assert torch.allclose(evaluate("T9","identity",x,t),eta,atol=1e-13)
    assert torch.allclose(evaluate("T9","sigmoid",x,t),torch.sigmoid(eta),atol=1e-13)
    assert torch.allclose(evaluate("T9","sigmoid",x,t,"inverse-link"),eta,atol=1e-12)

    xx,tt,_,xp=paired_grid(4,4,6,24,2026)
    analytic=AnalyticBackend("T9","sigmoid")
    bank=build_operator_bank(analytic,"sigmoid",xx,tt,6,24)
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
        m=operator_family_metrics(bank,R)
        assert set(m)==set(("operator_v1_jd","operator_v2_jd",
            "operator_v1_level","operator_v1_first",
            "operator_v2_second_diagonal","operator_v2_second_mixed",
            "operator_fisher_offdiag","operator_fisher_trace"))
        assert min(m.values())>=-1e-10
        assert np.isclose(m["operator_fisher_trace"],np.trace(bank.fisher))
    with tempfile.TemporaryDirectory() as folder:
        name=Path(folder)/"operators.npz"
        save_operator_bank(name,bank)
        with np.load(name) as a:
            assert len(a["all_names"])==15
            assert a["v2_matrices"].shape[1:]==(4,4)
    # Both analytic and differentiable neural backends use the same code.
    network=MLP(8,width=10,depth=2)
    backend=NeuralBackend("T9","sigmoid",network)
    xb,tb,_,_=paired_grid(4,4,3,6,44)
    nbank=build_operator_bank(backend,"sigmoid",xb,tb,3,6,batch_size=16)
    assert np.isfinite(nbank.fisher).all()
    assert np.isfinite(list(operator_family_metrics(nbank,np.eye(4)).values())).all()
    print("PASS: exact C2 Bernoulli p, v1/v2 operator inventory, second-X signature, neural backend")

if __name__=="__main__":tests()
