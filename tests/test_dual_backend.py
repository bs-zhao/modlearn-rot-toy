"""Run python tests/test_dual_backend.py. No pytest or editable install."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from rottoy.models import MODELS,evaluate,plane_rotation
from rottoy.sampling import paired_grid
from rottoy.backends import AnalyticBackend,train_neural,load_neural,checkpoint_path
from rottoy.operators import derivative_bank
from rottoy.metrics import local_metrics,sweep_metrics
from rottoy.__main__ import resolve_models
from rottoy.metrics import metric_id

def main():
    assert "T7" in MODELS
    assert resolve_models(["T7","T3"])==["T7","T3"]
    assert resolve_models(["all"])==list(MODELS)
    try:resolve_models(["MISSING"])
    except ValueError:pass
    else:raise AssertionError("Unknown model silently accepted")
    x,t,_,xp=paired_grid(2,1,4,8,123)
    a=AnalyticBackend("T7","sigmoid")
    assert torch.allclose(a.predict(torch.tensor(x),torch.tensor(t)),
        evaluate("T7","sigmoid",torch.tensor(x),torch.tensor(t)))
    with tempfile.TemporaryDirectory() as root:
        f,metadata=train_neural(root,"T7","sigmoid",ntrain=128,nval=64,
                               epochs=5,width=8,depth=2,batch_size=32,seed=12)
        loaded=load_neural(checkpoint_path(root,"T7","sigmoid"),"T7","sigmoid")
        assert torch.allclose(f.predict(torch.tensor(x),torch.tensor(t)),
            loaded.predict(torch.tensor(x),torch.tensor(t)))
        banks=[derivative_bank("T7","sigmoid",x,t,4,8,backend=b) for b in (a,f)]
        for bank in banks:
            assert bank.hxtheta.shape==(4,8,1,2)
            assert np.isfinite(bank.gtheta).all()
            assert metric_id("xu_group_l1") in local_metrics(bank,plane_rotation(2,0,1,45))
        for backend in (a,f):
            m=sweep_metrics("T7","sigmoid",np.eye(2),xp,7,.35,.2,backend=backend)
            assert np.isfinite(list(m.values())).all()
    print("DUAL BACKEND PASS: plugin, training, saved network, paired operators/sweeps")

if __name__=="__main__":main()
