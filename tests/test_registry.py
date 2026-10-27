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

# Documentation must cover every registered toy/metric with a unique table row.
import re
project_root=Path(__file__).resolve().parents[1]
models_doc=(project_root/"catalog"/"README_TOY_MODELS.md").read_text(encoding="utf-8")
metrics_doc=(project_root/"catalog"/"README_METRICS.md").read_text(encoding="utf-8")
listed_models=re.findall(r"(?m)^\| \*\*(T\d+)\*\* \|",models_doc)
listed_metrics=re.findall(r"(?m)^\| \*\*(M\d{3})\*\* \|",metrics_doc)
assert listed_models==list(MODELS),f"Toy catalog out of sync: {listed_models} != {list(MODELS)}"
assert listed_metrics==list(METRICS),f"Metric catalog out of sync: {listed_metrics} != {list(METRICS)}"

next_id=f"M{n_metrics+1:03d}"
assert register_metric("temporary_formula_probe","testing registry extension")==next_id
assert next_id in METRICS
print(f"UNIFIED REGISTRY PASS: T1..T{max(model_ids)}; M001..M{n_metrics+1:03d} automatic extension")
