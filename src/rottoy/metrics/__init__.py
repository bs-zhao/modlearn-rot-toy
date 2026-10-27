"""Universal metrics; every metric applies identically to eta and surrogate F."""
from .specs import METRICS,BY_KEY,MetricSpec,register_metric,metric_id
from .pointwise import local_metrics
from .sweeps import sweep_metrics
from ..operators import operator_metrics

LOCAL_KEYS={key for key in list(BY_KEY)[:11]}
SWEEP_KEYS={key for key in list(BY_KEY)[11:16]}
OPERATOR_KEYS={key for key in list(BY_KEY)[16:]}

def compute_metrics(R,selected,*,derivatives=None,operator_bank=None,
                    backend=None,x_nodes=None,sweep_steps=101,
                    sweep_radius=.45,rectangle_step=.2,analysis_scale="raw"):
    """Calculate only requested metric families, with one API for all toys/backends."""
    keys={METRICS[m].key for m in selected}
    measured={}
    if keys & LOCAL_KEYS:
        if derivatives is None:raise ValueError("Derivative bank required")
        measured.update(local_metrics(derivatives,R))
    if keys & SWEEP_KEYS:
        if backend is None or x_nodes is None:raise ValueError("Sweep requires backend/X nodes")
        measured.update(sweep_metrics(backend.variant,R,x_nodes,sweep_steps,
                                      sweep_radius,rectangle_step,analysis_scale,backend))
    if keys & OPERATOR_KEYS:
        if operator_bank is None:raise ValueError("Functional operator bank required")
        measured.update(operator_metrics(operator_bank,R))
    return {mid:float(measured[METRICS[mid].key]) for mid in selected}
