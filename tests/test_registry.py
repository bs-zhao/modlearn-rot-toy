"""Check that adding model or metric does not introduce a separate pipeline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rottoy.models import MODELS
from rottoy.metrics import METRICS,metric_id,register_metric
from rottoy.__main__ import resolve_models
assert list(MODELS)==[f"T{i}" for i in range(1,10)], list(MODELS)
assert "C2" not in MODELS and "T4ID" not in MODELS
assert list(METRICS)==[f"M{i:03}" for i in range(1,25)]
assert metric_id("operator_v1_jd")=="M017"
assert metric_id("operator_v2_jd")=="M018"
assert resolve_models(["T9","T1"])==["T9","T1"]
assert register_metric("temporary_formula_probe","testing registry extension")=="M025"
assert "M025" in METRICS
print("UNIFIED REGISTRY PASS: T1..T9; M001..M025 automatic extension")
