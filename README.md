# modlearn-rot-toy

**One registry of toy functions (`T1, T2, ...`), one registry of functional metrics (`M001, M002, ...`), one benchmark pipeline.**

The original C2 choice-probability formula, models with parameter interaction, and the formerly separate Fisher / first- and second-X-derivative function operators are **ordinary entries** in these registries. There are no C2-specific run commands, backend classes, or metric scripts. Both the exact analytic function `eta'` and a differentiable approximate network `F'` go through the same operators, numerical X/U quadrature, rotation sweeps, metric IDs and plots.

## Model registry (append new T numbers indefinitely)

| ID | Formula for the base `eta(X,theta)` | Source / role |
|---|---|---|
| T1 | `a*x**b` | Amplitude and exponent |
| T2 | `a*x1 + b*x2 + c` | Linear |
| T3 | `a*x1 + a*b*x2` | Parameter multiplication / 45-degree counterexample |
| T4 | `a*x1**b + c*d*x2**b` | Nonidentifiable 4-parameter negative control |
| T5 | `a*x1 + b*x1**2 + c*x2 + d` | Shared-X polynomial |
| T6 | `a*x1 + b*x2 + c*x1*x2 + d` | X interaction |
| T7 | `a*exp(b*x)` | Exponential plugin example |
| T8 | `a*x1**b + c*d*x2**b + c*x3` | Identifiable extension of T4, previously labeled `T4ID` |
| T9 | `b1+sqrt(3)*b2*(X1A-X1B)+sqrt(5)/2*b3*(3*X1A**2-3*X1B**2)+sqrt(3)*b4*(X2A-X2B)` | Formula from the original C2 experiment, **now a normal toy** |

All physical `theta` and individual `X` values lie in **[0,1]**, with strictly interior integration nodes for power functions. The rotated coordinates `u=R theta` need not individually lie in [0,1]. The default rotation is an orthogonal plane rotation; at 0 degrees, axes are aligned.

Every toy supports `identity`, `sigmoid`, `square` output wrappers, with optional `raw`, `log`, and (for sigmoid) `inverse-link` analysis scales. For T9, `--wrappers sigmoid` gives its original probability `p=sigmoid(eta)`; the default raw scale analyzes the probability, while `inverse-link` analyzes its logit. The original C2 sampling range was not necessarily [0,1]; this repo deliberately applies its formula to the unified toy range rather than attempting to reproduce old datasets.

### Adding a toy

Create `src/rottoy/toy_models/t10.py`:

```python
from ..models import register_model

@register_model("T10", k=2, dx=1, description="A new toy function")
def t10(x, theta):
    return theta[..., 0] * x[..., 0] + theta[..., 1]
```

Nothing changes in the CLI, numerical operators, network backends, metric registry, or plotting. All registered toy names must be of the form `T` followed by a positive integer; run `python run.py --list-models`.

## Metrics: numbered append-only registry

**24 metrics, IDs M001–M024**, with stable meanings. New metrics must be added using `register_metric(name,description)` in the **shared** `src/rottoy/metrics.py`; this assigns the next M number automatically. Implement the metric within the same common metric library and return it through `evaluate_metrics`. No model-specific metric file is created.

| ID | Indicator |
|---|---|
| M001 | XU RMS group L1 sparsity |
| M002 | XU group energy concentration |
| M003 | Local normalized XU L1 (known counterexample for T5/T9) |
| M004 | U-gradient variation across X: covariance offdiagonal |
| M005 | U-gradient Gram offdiagonal |
| M006 | UU offdiagonal Hessian fraction |
| M007 | UU diagonal Hessian fraction |
| M008 | U gradient total energy (orthogonal invariant) |
| M009 | XU total energy (orthogonal invariant) |
| M010 | X gradient energy (rotation invariant) |
| M011 | XX curvature energy (rotation invariant) |
| M012 | Global U-sweep affine fit residual |
| M013 | Minimum normalized response-shape variation |
| M014 | Minimum rank-one U-sweep matrix approximation error |
| M015 | U-sweep second finite difference curvature |
| M016 | UU finite-rectangle mixed effect |
| M017 | Function-operator v1 joint diagonalization: level + first X |
| M018 | Function-operator v2 joint diagonalization: level + first and second X |
| M019 | Operator v1 level component |
| M020 | Operator v1 first-X component |
| M021 | Operator v2 diagonal second-X component |
| M022 | Operator v2 mixed second-X component |
| M023 | Logistic-weighted gradient information offdiagonal |
| M024 | Logistic-weighted gradient information trace (orthogonal invariant) |

Metrics M017–M024 retain their **mathematical lineage** from [decoder_fisher_jointdiag v1/v2](https://github.com/Cognition-Decision-Modeling-Lab/modlearn-rot-2/tree/main/experiment/rotation/choice/decoder_fisher_jointdiag), but there is no historical/choice-specific execution path. The generic operator bank computes:
```text
g(X,u) = grad_u h(X,u)
psi_0(u) = integral_X g(X,u) dQ_X
psi_j(u) = integral_X sd(X_j) * derivative_Xj g(X,u) dQ_X
psi_jl(u) = integral_X sd(X_j)*sd(X_l)*derivative_XjXl g(X,u) dQ_X
K_alpha = integral_U psi_alpha(u) psi_alpha(u)^T dQ_U
```
X and U integrations use numerical quadrature nodes, **not participants and trials**. Matrices are filtered by relative energy, normalized by Frobenius norm, and optimized by joint offdiagonal energy. For `sigmoid` response, these function operators use the logit scale. M023–M024 use the corresponding sigmoid/logistic Fisher weighting as a generic comparison function, not a task-specific empirical likelihood. The predictive Fisher active-rank estimate is a saved diagnostic; it is not used to truncate the toy coordinates.

## How to run (no editable installation or pytest)

```bash
cd /fs/scratch/PDE0114/projects/modlearn-rot-toy
git pull --ff-only
source .venv/bin/activate
python run.py --list-models
python run.py --list-metrics
python tests/test_toys.py
python tests/test_dual_backend.py
python tests/test_functional_operators.py
python run.py --models T9 --wrappers sigmoid --pairs 2,3 --u-points 24 --x-points 100 --sweep-steps 101 --angles 37
```

Several models and selected metrics, using the same interface:

```bash
python run.py --models T1 T3 T9 --wrappers sigmoid --metrics M004 M006 M017 M018 --pairs all --u-points 32 --x-points 100
```

Exact `eta'` and trained approximation `F'`:

```bash
python run.py --models T3 T9 --wrappers sigmoid --backend both --train-neural --nn-epochs 150 --metrics all
```

For fast numerical tests, use `--u-points 8 --x-points 24 --sweep-steps 21 --angles 19`. By default `--u-points` and `--x-points` are respectively **24 and 100 quadrature nodes**, not simulated individuals or trials. `--x-mode axis_sweeps` scans each input dimension separately, with other X coordinates at 0.5. `--operator-bank none` disables the M017–M024 family for faster simpler runs.

The historical finite difference defaults, exposed as generic quadrature controls, are `--x-first-step-fraction .02` and `--x-second-step-fraction .10` times the numerical stimulus SD. Relative operator energy filters are `--operator-v1-min-relative-energy 1e-4`, `--operator-v2-min-relative-energy .01`.

## Shared analytic / neural interface

Both backends implement `predict(x, theta, scale)`. The first differentiable surrogate is a small MLP trained using sampled `(theta,X,Y)`; frozen network parameters do not change during rotations. The same differentiable bank, U-sweep and numerical operator quadrature are used for `eta'` and `F'`. A future MP/NP approximator can provide the same backend interface.

All outputs are saved under the Git-ignored `outputs/toy_rotation/`:

```text
outputs/toy_rotation/T9/sigmoid/raw/u2_u3/
  metrics_by_angle.csv          # columns angle_deg, M001, ..., M024
  metric_optima.csv             # M IDs, minimum angle and true-axis value
  metric_heatmap.png            # angles on X; metric IDs on Y
  metric_curves.png
  xu_effect_map.png
  operator_v1_v2_curves.png   # retained math provenance, not separate pipeline
  settings.json
# When --backend both:
  analytic/metrics_by_angle.csv
  neural/metrics_by_angle.csv
  paired_metrics_by_angle.csv
  paired_overlay_curves.png
  paired_error_heatmap.png
# Generic operator matrices, saved independently of the rotated angles:
.../analytic/functional_operator_matrices.npz
```

As a mathematical test bed, good agreement of function values does **not** guarantee matched derivatives or identical best rotation angles. A metric may provably favor a wrong coordinate system: for instance, M006 prefers a 45-degree mixture in T3. Such negative results are intentional.
