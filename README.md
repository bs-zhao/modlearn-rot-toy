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

The benchmark currently calculates **16** metrics, not all proposed library metrics:

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

A GitHub Actions job runs these checks and uploads preview plots as an artifact after each push.

## Scope

This is **phase 1: analytic test functions and representative indicators**. Do not infer that the numerical minimum of a metric automatically identifies generating parameters. Neural surrogate training, frozen neural-process adapters, a full 72-metric registry, or optimizers are future phases. All code is independent from the original `modlearn-rot-2` project.
