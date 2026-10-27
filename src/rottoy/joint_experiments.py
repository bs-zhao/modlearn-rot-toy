"""Multi-angle rotation exploration and independent metric-wise SO(K) searches.

Search over K(K-1)/2 Givens angles, rather than one pair at a time.
It is a multistart local numerical search, NOT a proof of global optimality.
"""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import METRICS,compute_metrics
from .rotations import joint_rotation,planes,signed_permutation_distance,orthogonality_error

def _write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def _evaluate(R,selected,backend,bank,operator,x_nodes,cfg):
    return compute_metrics(R,selected,derivatives=bank,operator_bank=operator,
        backend=backend,x_nodes=x_nodes,sweep_steps=cfg.sweep_steps,
        sweep_radius=cfg.sweep_radius,rectangle_step=cfg.rectangle_step,
        analysis_scale=cfg.analysis_scale)

def _candidate_heatmap(scores,selected,path,title):
    """Rows are M metrics, columns are independent full SO(K) candidates.

    Candidate 0 is identity, others are independently sampled six-angle
    rotations for T8; colors are normalized separately within each metric.
    """
    matrix=np.asarray([[row[mid] for row in scores] for mid in selected],dtype=float)
    low=np.min(matrix,axis=1,keepdims=True)
    span=np.ptp(matrix,axis=1,keepdims=True)
    normalized=np.divide(matrix-low,span,out=np.full_like(matrix,.5),
                         where=span>1e-12)
    fig,ax=plt.subplots(figsize=(max(8,len(scores)*.13+3),
                                max(3.5,len(selected)*.38+2)),layout="constrained")
    im=ax.imshow(normalized,origin="upper",aspect="auto",cmap="viridis",
                 vmin=0,vmax=1)
    ax.set_yticks(range(len(selected)),selected)
    ax.set_xlabel("Full-dimensional rotation candidate index (0 = identity)")
    ax.set_title(title+" | each candidate mixes all K latent coordinates")
    fig.colorbar(im,ax=ax,label="Row-normalized metric (constant row = 0.5)",shrink=.78)
    fig.savefig(path,dpi=145)
    plt.close(fig)

def _rotation_heatmap(R,metric,path):
    fig,ax=plt.subplots(figsize=(5,4.4),layout="constrained")
    im=ax.imshow(np.abs(R),vmin=0,vmax=1,cmap="viridis")
    for i in range(len(R)):
        for j in range(len(R)):
            ax.text(j,i,f"{R[i,j]:+.2f}",ha="center",va="center",
                    color="white" if abs(R[i,j])<.65 else "black",fontsize=8)
    ax.set_xticks(range(len(R)),[f"theta{j+1}" for j in range(len(R))])
    ax.set_yticks(range(len(R)),[f"u{i+1}" for i in range(len(R))])
    ax.set_xlabel("Physical parameter axes")
    ax.set_ylabel("Rotated axes")
    ax.set_title(f"{metric} | best found joint rotation R")
    fig.colorbar(im,ax=ax,label="|R_ij|",shrink=.85)
    fig.savefig(path,dpi=145)
    plt.close(fig)

def _summary_chart(rows,path,title):
    labels=[r["metric"] for r in rows]
    gains=[r["relative_improvement"] for r in rows]
    fig,ax=plt.subplots(figsize=(max(8,len(rows)*.52+3),5),layout="constrained")
    ax.bar(range(len(rows)),gains)
    ax.set_xticks(range(len(rows)),labels,rotation=90)
    ax.set_ylabel("(M(identity)-M(best)) / max(|M(identity)|, 1e-12)")
    ax.set_title(title+" | best discovered full-dimensional gain")
    ax.axhline(0,color="black",lw=.8)
    ax.grid(axis="y",alpha=.2)
    fig.savefig(path,dpi=145)
    plt.close(fig)

def search_joint(cfg,variant,selected,backends,derivatives,operators,x_nodes,root):
    """Evaluate identical random rotations then optimize EACH selected M independently."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    k=variant.k
    pairs=planes(k);d=len(pairs)
    keys=[f"phi_{i+1}_{j+1}_deg" for i,j in pairs]
    rng=np.random.default_rng(cfg.seed+97)
    angles=np.concatenate([
        np.zeros((1,d)),
        rng.uniform(-180,180,size=(cfg.joint_samples,d)),
    ],axis=0)
    rotations=[joint_rotation(k,v) for v in angles]
    if any(orthogonality_error(R)>1e-10 for R in rotations):
        raise AssertionError("Givens rotations must remain orthogonal")
    all_summaries={}
    for label,backend in backends.items():
        destination=root if cfg.backend=="analytic" else root/label
        destination.mkdir(parents=True,exist_ok=True)
        scores=[_evaluate(R,selected,backend,derivatives.get(label),
                          operators.get(label),x_nodes,cfg) for R in rotations]
        candidates=[]
        for n,(angle,R,score) in enumerate(zip(angles,rotations,scores)):
            candidates.append(dict(candidate_id=n,**dict(zip(keys,angle.tolist())),
                signed_permutation_distance=signed_permutation_distance(R),**score))
        _write_csv(destination/"joint_candidates.csv",candidates)
        _candidate_heatmap(scores,selected,destination/"joint_candidates_heatmap.png",
                           f"{variant.id} ({label})")
        best_rows=[];best_angles=[];best_matrices=[]
        for metric in selected:
            baseline=float(scores[0][metric])
            observed=np.asarray([row[metric] for row in scores],dtype=float)
            order=np.argsort(observed)
            candidates_start=[np.zeros(d)]
            for ix in order:
                if len(candidates_start)>=1+cfg.joint_restarts:break
                candidate=np.asarray(angles[ix])
                if not any(np.allclose(candidate,z) for z in candidates_start):
                    candidates_start.append(candidate)
            best_value=float(np.min(observed))
            best_angle=np.asarray(angles[int(np.argmin(observed))],dtype=float)
            n_evaluations=len(scores)
            opt_status="sampled"
            flat=np.allclose(observed,baseline,rtol=1e-7,atol=1e-10)
            if not flat and METRICS[metric].behavior!="invariant":
                def objective(v):
                    R=joint_rotation(k,v)
                    return float(_evaluate(R,[metric],backend,derivatives.get(label),
                        operators.get(label),x_nodes,cfg)[metric])
                for start in candidates_start:
                    result=minimize(objective,np.asarray(start,dtype=float),
                        method="L-BFGS-B",bounds=[(-180.,180.)]*d,
                        options={"maxiter":cfg.joint_maxiter,"ftol":1e-12})
                    n_evaluations+=int(result.nfev)
                    if np.isfinite(result.fun) and result.fun<best_value:
                        best_value=float(result.fun)
                        best_angle=result.x
                        opt_status="optimized" if result.success else "optimized_partial"
            R_best=joint_rotation(k,best_angle)
            distance=signed_permutation_distance(R_best)
            improvement=baseline-best_value
            relative=improvement/max(abs(baseline),1e-12)
            row=dict(metric=metric,name=METRICS[metric].name,
                at_identity=baseline,best_found=best_value,
                absolute_improvement=improvement,relative_improvement=relative,
                signed_permutation_distance=distance,
                best_found_aligned=bool(distance<.05),
                flat=bool(flat),status="flat_invariant" if flat else opt_status,
                evaluations=n_evaluations,**dict(zip(keys,best_angle.tolist())))
            best_rows.append(row)
            best_angles.append(best_angle)
            best_matrices.append(R_best)
            _rotation_heatmap(R_best,metric,destination/f"best_rotation_{metric}.png")
            print(f"  {variant.id}/{label} {metric}: identity={baseline:.6g}, "
                  f"best={best_value:.6g}, distance_to_axis_permutation={distance:.4f}",
                  flush=True)
        _write_csv(destination/"joint_search_summary.csv",best_rows)
        np.savez_compressed(destination/"best_rotations.npz",
            metric_ids=np.asarray(selected,dtype=str),
            angle_pairs=np.asarray(pairs,dtype=int),
            angles_deg=np.asarray(best_angles),
            rotations=np.asarray(best_matrices),
            baseline=np.asarray([r["at_identity"] for r in best_rows]),
            optimum=np.asarray([r["best_found"] for r in best_rows]))
        _summary_chart(best_rows,destination/"joint_search_improvements.png",
                       f"{variant.id} ({label})")
        settings=dict(variant=variant.id,backend=label,k=k,angle_pairs=pairs,
            search_method="multistart L-BFGS-B finite-difference gradient",
            random_candidates=cfg.joint_samples,restarts=cfg.joint_restarts,
            max_iterations=cfg.joint_maxiter,seed=cfg.seed,
            definition="independent optimization of each M over K(K-1)/2 Givens angles",
            caveat="local numerical best, not proven global; sign/permutation equivalence")
        (destination/"settings.json").write_text(json.dumps(settings,indent=2),
                                                  encoding="utf-8")
        all_summaries[label]=best_rows
    if cfg.backend=="both":
        neural={r["metric"]:r for r in all_summaries["neural"]}
        compared=[]
        for row in all_summaries["analytic"]:
            other=neural[row["metric"]]
            compared.append(dict(metric=row["metric"],
                eta_identity=row["at_identity"],F_identity=other["at_identity"],
                eta_best=row["best_found"],F_best=other["best_found"],
                eta_distance=row["signed_permutation_distance"],
                F_distance=other["signed_permutation_distance"]))
        _write_csv(root/"paired_joint_search_summary.csv",compared)
    return all_summaries
