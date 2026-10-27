# modlearn-rot-toy — analytic functional rotation benchmark

A **standalone** Python test bed for assessing post-hoc coordinate simplicity on exact generating functions before designing any unsupervised rotation loss. True parameter values and each X dimension are sampled in **[0,1]**. The initial benchmark sets ideal latent z = theta and rotates its coordinates. It **does not** train a neural network; the analytic PyTorch callable is the exact differentiable decoder.

## Quick start (Windows CMD)

```bat
cd /d D:\Projects\modlearn-rot-toy
python -m pip install -r requirements.txt
python tests/test_toys.py
python run.py --models T1 T2 T3 T4 --wrappers all
```

Faster preview:
```bat
python run.py --models T1 T3 --wrappers identity --theta-samples 16 --x-samples 32 --sweep-steps 41 --angles 19
```

All plane pairs and toy functions (larger sweep/plot workload):
```bat
python run.py --models all --wrappers all --pairs all --theta-samples 24 --x-samples 100 --sweep-steps 101 --angles 37
```

The default rotation scan is **-90 to +90 degrees** with 37 points, including the real parameter-aligned reference at 0 degrees. The `--pairs` option uses `first` (the u1/u2 plane), `all` (every coordinate pair) or a particular 1-based pair such as `1,3`.


## Automatic toy-model discovery and paired eta/F diagnostics

Add a Python module under `src/rottoy/toy_models/` with a function decorated by `@register_model("T100",k=2,dx=1,description="a*x**b")`; it will appear under `python run.py --list-models` and run via `--models T100 T1` without edits elsewhere. The separate T7 plugin is a working demonstration.

The same metric bank supports `--backend analytic` (default), `--backend neural`, and `--backend both`. The neural backend currently uses a smooth MLP trained on simulated `(theta,X,Y)` samples. It approximates the exact wrapped response `Y`, and exposes `predict(x,theta,scale)` to the shared derivative/sweep code. Future MP or Neural Process decoders can implement that interface. Neural checkpoints are trained once per model/wrapper on physical theta/X in [0,1] and **not retrained at each rotation**.

```bat
python run.py --list-models
python run.py --models T7 T3 --wrappers identity --theta-samples 8 --x-samples 32 --sweep-steps 21 --angles 19
python run.py --models T1 T3 --wrappers identity --backend both --train-neural --nn-epochs 150
python run.py --models T1 T3 --wrappers identity --backend both
python tests/test_dual_backend.py
```

Paired outputs are saved below `outputs/toy_rotation/<MODEL>/<WRAPPER>/<SCALE>/<PAIR>/` with separate `analytic/` and `neural/` metric curves and heatmaps; `paired_overlay_curves.png`, `paired_error_heatmap.png`, and `paired_metrics_by_angle.csv` compare exact analytic and approximate neural indicators on **the same** rotations and probes. `settings.json` reports prediction, X/U gradient, and XX/XU/UU Hessian errors. Model/checkpoint output stays under ignored `outputs/`.

**Important:** fitting function values well does not guarantee derivative or rotation-minimum agreement. MLP is a first surrogate adapter; it is not yet an MP/NP implementation.


## Original C2 binary-choice testbed and historical v1/v2 metrics

The original [C2 generator](https://github.com/Cognition-Decision-Modeling-Lab/modlearn-rot-2/blob/main/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py) is now registered as **`C2`**. Native X feature order is `(X1_A, X1_B, X2_A, X2_B)`; native parameters are `(b1, b2, b3, b4)`. The exact choice logit is:

```text
eta = b1 + sqrt(3)*b2*(X1_A-X1_B)
         + sqrt(5)/2*b3*((3*X1_A**2-1)-(3*X1_B**2-1))
         + sqrt(3)*b4*(X2_A-X2_B)
P(Y=1 | X, theta) = sigmoid(eta), 1 = option A
```

The toy plugin returns the **logit** `eta`; select `--wrappers sigmoid` to evaluate the actual choice probability `p`. Use `--analysis-scale inverse-link` to inspect the recovered choice logit. (The model's original X design was uniform **[-1,1]** and theta normal; this **toy** project deliberately uses [0,1] for all raw X and theta, as requested. The formula is identical but historical sampling distributions and numerical recovery results are not.)

Run a compact benchmark comparing exact analytic `eta'` and a trained approximate neural `F'`:

```bash
python tests/test_historical_choice.py
python run.py --list-models
python run.py --models C2 --wrappers sigmoid --pairs 2,3 --backend analytic --theta-samples 8 --x-samples 24 --sweep-steps 21 --angles 19
python run.py --models C2 --wrappers sigmoid --pairs all --backend both --train-neural --nn-epochs 150
```

The same CLI supports `--models C2 T1 T3`. No neural training is needed for analytic-only runs.

**Historical [choice decoder_fisher_jointdiag v1](https://github.com/Cognition-Decision-Modeling-Lab/modlearn-rot-2/tree/main/experiment/rotation/choice/decoder_fisher_jointdiag/v1_distributional) and [v2](https://github.com/Cognition-Decision-Modeling-Lab/modlearn-rot-2/tree/main/experiment/rotation/choice/decoder_fisher_jointdiag/v2_higher_order) functional indicators** are on by default and use the **same X/theta samples and both function backends**:

- **Binary Fisher:** `E[p(1-p)gg^T]`, `g=grad_theta log(p/(1-p))`, plus Fisher eigenvalues, offdiagonal energy and invariant trace. For the `sigmoid` wrapper, use the inverse logit to compute the gradient. For other wrappers, treat the wrapped response as a choice logit for this historical diagnostic.
- **v1:** for each subject/parameter point, compute the mean log-odds gradient over its observed X samples and its mean first-X derivative `sd(X_j) d_j g`; form **outer product AFTER within-subject mean**. Normalize each operator by its Frobenius norm, filter raw operator energy below `1e-4` times maximum, then average rotated offdiagonal energy (`historical_v1_jd`).
- **v2:** same level and first-X operators, plus `sd(X_j)**2 d_jj g` and `sd(X_j)*sd(X_l) d_jl g`. Filter below `0.01` of maximum, normalize each retained operator, and compute `historical_v2_jd`.
- **Other plotted scalar rows:** v1 level/first and v2 diagonal-second/mixed-second group contributions; `historical_fisher_offdiag` and `historical_fisher_trace`. These are separate measurements, not automatically identifiable objectives.

Historical finite differences match original recipes: first derivative step **0.02 × observed SD(X)**; second derivative step **0.10 × observed SD(X)**, with one-sided stencils within the legal [0,1] toy X box. Thresholds and steps are configurable using `--historical-first-step-fraction`, `--historical-second-step-fraction`, `--historical-v1-min-relative-energy`, and `--historical-v2-min-relative-energy`. Pass `--no-historical` for faster runs.

**Important difference from the original pipeline:** in this analytic toy, all theta dimensions are retained, while the original v1/v2 first determined a predictive active subspace by eigenanalysis of Fisher in whitened NP latent `q`. The full Fisher eigenvalues are exported for comparison. These are faithful reproductions of the **orientation signatures, operator normalization and JD objective**, not a replication of NP active-space selection or empirical subject sampling.

Per model/backend, the output includes `historical_v1_v2_operators.npz` (raw and retained operator matrices, names, energies, Fisher matrix/eigenvalues, finite-difference steps). The angle-based `metrics_by_angle.csv` and `metric_heatmap.png` now include the historical rows, with a dedicated `historical_v1_v2_curves.png` for v1 vs v2. Paired eta/F comparison files also include them.

## Test bed

| Model | Analytic generator | Motivation |
|---|---|---|
| T1 | `a*x**b` | Amplitude vs exponent |
| T2 | `a*x1+b*x2+c` | Linear baseline |
| T3 | `a*x1+a*b*x2` | Real UU interaction; diagonal-Hessian trap |
| T4 | `a*x1**b+c*d*x2**b` | Nonidentifiable c,d |
| T4ID | T4 + `c*x3` | Identifiability control |
| T5 | `a*x1+b*x1**2+c*x2+d` | C2-like shared-X polynomial |
| T6 | `a*x1+b*x2+c*x1*x2+d` | C3-like X interaction |

Each generator supports **identity**, **sigmoid(eta)**, and **eta²** wrappers. The default metric scale `--analysis-scale raw` analyzes the wrapped output itself. Use `--analysis-scale log` for positive responses, or `--analysis-scale inverse-link --wrappers sigmoid` to explicitly analyze sigmoid logits.

All physical theta and X coordinates lie in [0,1]. Quasi-Monte Carlo integration uses strictly interior nodes because e.g. d(x**b)/dx can diverge at x=0. Rotated u coordinates may lie outside [0,1]; the inverse-mapped physical theta always stays in [0,1]. The U-axis sweep uses theta = (.5,...,.5) + s*R[k,:], s in [-.45,.45]; rectangle probes use the same legal support.

## X sampling and sweep resolution

Native X dimensions and true parameter dimensions always lie in [0,1]. The default X integration uses joint Sobol sampling. You can instead sweep **each X dimension separately from 0 to 1** using midpoint nodes, holding other X coordinates at 0.5:

```bat
python run.py --models T3 --wrappers identity --x-mode axis_sweeps --x-samples 100 --sweep-steps 101 --output outputs/x_axis_sweeps
```

Here `--x-samples 100` means 100 points **per X dimension**, giving `100 * D_X` stimulus nodes. The derivative matrices and U-sweep output are then evaluated at these nodes. Use different output paths for different X sampling schemes to prevent overwriting experiment results.

## Representative functional metrics

The benchmark calculates **16 original metrics plus 8 historical choice Fisher/v1/v2 metrics** by default (24 total), not all proposed library metrics:

| Family | Implemented indicators |
|---|---|
| XU | `xu_group_l1`, `xu_energy_concentration`, `local_xu_l1` |
| U-gradient covariance | `grad_xcov_offdiag`, `grad_gram_offdiag` |
| UU curvature | `uu_offdiag`, `uu_diagonal` |
| Invariant controls | `u_grad_energy`, `xu_total_energy`, `x_gradient_energy`, `x_curvature_energy` |
| Global U sweep | `sweep_affine`, `sweep_shape_min`, `sweep_rank1_min`, `sweep_curvature` |
| Finite UU rectangle | `rectangle_uu` |

**Important:** many metrics intentionally prefer incorrect axes in some models. For example, T3 has a nonzero UU cross derivative at true axes and a zero UU cross derivative at 45 degrees, so `uu_offdiag` must fail. Some invariants are mathematically incapable of selecting a rotation. Raw, log and sigmoid probability scales can have very different rankings.

The derivative bank is computed once from the exact generator for sampled (X,theta). For each orthogonal R, U gradients, Hessians and XU derivatives are transferred by the exact chain rule; this equals differentiation of F'(X,u)=F(X,R.T@u) and avoids redundant autodiff at every angle. Global U sweeps and finite differences are separately evaluated by direct forward passes.

## Outputs

Output files are under ignored `outputs/toy_rotation/`, arranged as:
```
T3/identity/raw/u1_u2/
  metrics_by_angle.csv     # unnormalized absolute values
  metric_optima.csv        # metric minimum and true-axis value
  metric_heatmap.png       # angle on x, metric name on y, row-normalized color
  metric_curves.png        # six metrics in raw units
  xu_effect_map.png        # X_j versus U_k RMS mixed-gradient matrix at 0°
  settings.json
all_metric_optima.csv
```

The heatmap is normalized **independently by row**. Constant metrics are shown in neutral middle color. Read the CSV for absolute values. A minimum away from a true-axis permutation/sign-equivalence is a counterexample to the metric as a standalone structural loss.

### Mathematical unit checks

`python tests/test_toys.py` verifies output wrappers, T2 zero UU Hessian, T3's incorrect 45° Hessian diagonalization, T3 affine U sweeps at true axes, T1 log-stimulus gradient covariance, orthogonal invariants, T4/T4ID fixed-theta Jacobian ranks, and independent autograd vs rotated chain rule.

A GitHub Actions job runs these checks plus `python tests/test_historical_choice.py` and uploads preview plots as an artifact after each push.

## Scope

This is **phase 1: analytic test functions and representative indicators**. Do not infer that the numerical minimum of a metric automatically identifies generating parameters. Neural surrogate training, frozen neural-process adapters, a full 72-metric registry, or optimizers are future phases. All code is independent from the original `modlearn-rot-2` project.
