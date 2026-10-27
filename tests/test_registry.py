"""Check that adding model or metric does not introduce a separate pipeline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rottoy.models import MODELS
from rottoy.metrics import METRICS,metric_id,register_metric
from rottoy.__main__ import resolve_models
model_ids=[int(name[1:]) for name in MODELS]
assert model_ids==list(range(1,max(model_ids)+1)),model_ids
assert "C2" not in MODELS and "T4ID" not in MODELS
n_metrics=len(METRICS)
assert n_metrics>=24
assert list(METRICS)==[f"M{i:03}" for i in range(1,n_metrics+1)]
assert metric_id("operator_v1_jd")=="M017"
assert metric_id("operator_v2_jd")=="M018"
assert resolve_models(["T9","T1"])==["T9","T1"]
next_id=f"M{n_metrics+1:03d}"
assert register_metric("temporary_formula_probe","testing registry extension")==next_id
assert next_id in METRICS
print(f"UNIFIED REGISTRY PASS: T1..T{max(model_ids)}; M001..M{n_metrics+1:03d} automatic extension")
