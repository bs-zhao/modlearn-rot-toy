"""One CLI and one evaluation pipeline for every model variant and metric."""
from pathlib import Path
import argparse,csv,itertools,json
import numpy as np
from .models import MODELS,get_variant,variant_names,plane_rotation
from .transforms import TRANSFORMS
from .sampling import paired_grid
from .backends import AnalyticBackend,train_neural,load_neural,checkpoint_path
from .operators import derivative_bank,operator_bank,save_operator_bank
from .metrics import METRICS,LOCAL_KEYS,SWEEP_KEYS,OPERATOR_KEYS,compute_metrics
from .plotting import write_figures,plot_comparison,plot_xu
from .rotations import joint_path,planes,path_weights
from .joint_experiments import search_joint

def _csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def _pairs(k,selection):
    if selection=="all":return list(itertools.combinations(range(k),2))
    if selection=="first":return [(0,1)]
    try:i,j=(int(z)-1 for z in selection.split(","))
    except Exception as ex:raise ValueError("--pairs must be first, all, or i,j") from ex
    if not 0<=i<j<k:raise ValueError("--pairs requires 1 <= i < j <= K")
    return [(i,j)]

def _selected_variants(names):
    choices=variant_names() if names==["all"] else names
    variants=[get_variant(identifier) for identifier in choices]
    return list({v.id:v for v in variants}.values())

def _metadata_errors(a,b):
    return {name+"_mse":float(np.mean((getattr(a,name)-getattr(b,name))**2))
            for name in ("y","gtheta","htheta","hxtheta","gx","hxx")}

def run(config):
    variants=_selected_variants(config.models)
    selected=list(METRICS) if config.metrics==["all"] else list(dict.fromkeys(config.metrics))
    unknown=set(selected)-set(METRICS)
    if unknown:raise ValueError(f"Unknown M metric IDs: {sorted(unknown)}")
    if not selected:raise ValueError("Select at least one metric")
    need_derivatives=any(METRICS[mid].key in LOCAL_KEYS for mid in selected)
    need_operator=any(METRICS[mid].key in OPERATOR_KEYS for mid in selected)
    if config.operator_scale=="inverse":
        for v in variants:
            if not TRANSFORMS[v.transform].invertible:
                raise ValueError(f"{v.id}: inverse operator scale undefined for square")
    if config.analysis_scale=="inverse":
        for v in variants:
            if not TRANSFORMS[v.transform].invertible:
                raise ValueError(f"{v.id}: inverse scale undefined for square")
    out=Path(config.output)
    angles=np.linspace(config.angle_min,config.angle_max,config.angles)
    if min(abs(angles))>1e-9:raise ValueError("Angle grid must include 0°")
    summary=[]
    for v in variants:
        xx,tt,_,x_nodes=paired_grid(v.k,v.dx,config.u_points,config.x_points,
                                    config.seed,config.x_mode)
        backends={}
        training=None
        if config.backend in ("analytic","both"):
            backends["analytic"]=AnalyticBackend(v)
        if config.backend in ("neural","both"):
            if config.train_neural:
                net,training=train_neural(out,v,n_train=config.nn_train_samples,
                    n_val=config.nn_val_samples,epochs=config.nn_epochs,
                    width=config.nn_width,depth=config.nn_depth,lr=config.nn_lr,
                    batch_size=config.nn_batch_size,seed=config.seed)
            else:net=load_neural(out,v)
            backends["neural"]=net
        banks={};ops={}
        for backend_name,backend in backends.items():
            if need_derivatives or config.xu_map:
                banks[backend_name]=derivative_bank(backend,xx,tt,config.u_points,len(x_nodes),
                                                    config.analysis_scale)
            if need_operator:
                op=operator_bank(backend,xx,tt,config.u_points,len(x_nodes),
                    scale=config.operator_scale,first_step=config.x_first_step,
                    second_step=config.x_second_step,v1_filter=config.v1_filter,
                    v2_filter=config.v2_filter)
                ops[backend_name]=op
                bankfolder=out/v.model.name/v.transform/config.analysis_scale/backend_name
                bankfolder.mkdir(parents=True,exist_ok=True)
                save_operator_bank(bankfolder/"functional_operators.npz",op)
        jobs=[]
        if config.rotation_mode in ("pair","all"):
            for i,j in _pairs(v.k,config.pairs):
                jobs.append((f"u{i+1}_u{j+1}",
                    lambda phi,i=i,j=j:plane_rotation(v.k,i,j,phi)))
        if config.rotation_mode in ("joint-path","all"):
            jobs.append((f"joint_{config.joint_path}",
                lambda phi:joint_path(v.k,phi,config.joint_path,config.seed)))
        for rotation_label,rotate in jobs:
            folder=out/v.model.name/v.transform/config.analysis_scale/rotation_label
            folder.mkdir(parents=True,exist_ok=True)
            by_backend={}
            for backend_name,backend in backends.items():
                destination=folder if config.backend=="analytic" else folder/backend_name
                destination.mkdir(parents=True,exist_ok=True)
                rows=[]
                for phi in angles:
                    R=rotate(float(phi))
                    scores=compute_metrics(R,selected,derivatives=banks.get(backend_name),
                        operator_bank=ops.get(backend_name),backend=backend,x_nodes=x_nodes,
                        sweep_steps=config.sweep_steps,sweep_radius=config.sweep_radius,
                        rectangle_step=config.rectangle_step,analysis_scale=config.analysis_scale)
                    rows.append(dict(angle_deg=round(float(phi),8),**scores))
                _csv(destination/"metrics_by_angle.csv",rows)
                np.savez_compressed(destination/"rotation_matrices.npz",
                    angle_deg=np.asarray(angles,dtype=float),
                    rotations=np.stack([rotate(float(a)) for a in angles]),
                    angle_pairs=np.asarray(planes(v.k),dtype=int))
                optimum=write_figures(rows,destination,f"{v.id}/{backend_name}/{rotation_label}")
                _csv(destination/"metric_optima.csv",optimum)
                if config.xu_map and backend_name in banks:
                    plot_xu(banks[backend_name],destination,f"{v.id} XU")
                summary.extend(dict(variant=v.id,pair=rotation_label,
                                     backend=backend_name,**item) for item in optimum)
                by_backend[backend_name]=rows
            settings=dict(variant=v.id,k=v.k,dx=v.dx,
                          rotation_label=rotation_label,
                          givens_order=planes(v.k),
                          joint_path_coefficients=(path_weights(v.k,config.joint_path,config.seed).tolist()
                              if rotation_label.startswith("joint_") else None),
                          selected_metrics=selected,
                          metrics={m:METRICS[m].name for m in selected},
                          training=training,options=vars(config))
            if config.backend=="both":
                lhs=by_backend["analytic"];rhs=by_backend["neural"]
                settings["metric_error"]=plot_comparison(lhs,rhs,folder,v.id)
                if need_derivatives:
                    settings["derivative_error"]=_metadata_errors(banks["analytic"],banks["neural"])
                paired=[]
                for ar,nr in zip(lhs,rhs):
                    row={"angle_deg":ar["angle_deg"]}
                    for mid in selected:
                        row[mid+"__eta"]=ar[mid];row[mid+"__F"]=nr[mid]
                        row[mid+"__difference"]=nr[mid]-ar[mid]
                    paired.append(row)
                _csv(folder/"paired_metrics_by_angle.csv",paired)
            (folder/"settings.json").write_text(json.dumps(settings,indent=2,ensure_ascii=False),
                                                encoding="utf-8")
            print(f"[{v.id}] {config.backend}: {folder}",flush=True)
        if config.rotation_mode in ("joint-search","all"):
            folder=out/v.model.name/v.transform/config.analysis_scale/"joint_search"
            search_joint(config,v,selected,backends,banks,ops,x_nodes,folder)
    out.mkdir(parents=True,exist_ok=True)
    _csv(out/"all_metric_optima.csv",summary)
    return summary

def parser():
    p=argparse.ArgumentParser(description="Continuous-output toy functional rotation benchmark")
    p.add_argument("--list-models",action="store_true")
    p.add_argument("--list-metrics",action="store_true")
    p.add_argument("--describe-metric",default=None)
    p.add_argument("--models",nargs="+",default=["T1-id","T2-id"],
        help="T3-id T9-sig etc; bare T3 means T3-id; all = every variant")
    p.add_argument("--metrics",nargs="+",default=["all"],help="M001 M018 ... or all")
    p.add_argument("--backend",choices=("analytic","neural","both"),default="analytic")
    p.add_argument("--train-neural",action="store_true")
    p.add_argument("--nn-train-samples",type=int,default=4096)
    p.add_argument("--nn-val-samples",type=int,default=1024)
    p.add_argument("--nn-epochs",type=int,default=120)
    p.add_argument("--nn-width",type=int,default=64)
    p.add_argument("--nn-depth",type=int,default=3)
    p.add_argument("--nn-lr",type=float,default=.002)
    p.add_argument("--nn-batch-size",type=int,default=512)
    p.add_argument("--analysis-scale",choices=("raw","log","inverse"),default="raw")
    p.add_argument("--operator-scale",choices=("raw","inverse"),default="raw",
        help="Output scale for M017-M024; inverse uses g^-1 where defined")
    p.add_argument("--u-points",type=int,default=24)
    p.add_argument("--x-points",type=int,default=100)
    p.add_argument("--x-mode",choices=("sobol","axis_sweeps"),default="sobol")
    p.add_argument("--angles",type=int,default=37)
    p.add_argument("--angle-min",type=float,default=-90)
    p.add_argument("--angle-max",type=float,default=90)
    p.add_argument("--pairs",default="first",
        help="Pair mode: first, all, or i,j; used only when mode=pair/all")
    p.add_argument("--rotation-mode",choices=("pair","joint-path","joint-search","all"),
        default="pair",
        help="Pair plane curves; a simultaneous K-axis curve; full K(K-1)/2-dimensional search; or all")
    p.add_argument("--joint-path",choices=("balanced","alternating","ramped","random"),
        default="ramped",
        help="How each Givens plane angle changes with the joint-path x-axis angle")
    p.add_argument("--joint-samples",type=int,default=48,
        help="Extra seeded random full-dimensional candidate rotations")
    p.add_argument("--joint-restarts",type=int,default=2,
        help="Extra local minimization starts per metric, plus identity")
    p.add_argument("--joint-maxiter",type=int,default=65,
        help="Maximum iterations per full-dimensional optimizer restart")
    p.add_argument("--sweep-steps",type=int,default=101)
    p.add_argument("--sweep-radius",type=float,default=.45)
    p.add_argument("--rectangle-step",type=float,default=.2)
    p.add_argument("--x-first-step",type=float,default=.02)
    p.add_argument("--x-second-step",type=float,default=.10)
    p.add_argument("--v1-filter",type=float,default=1e-4)
    p.add_argument("--v2-filter",type=float,default=.01)
    p.add_argument("--xu-map",action="store_true",help="Optional full XU map (extra derivatives)")
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--output",default="outputs/toy_rotation")
    return p

def cli():
    p=parser()
    a=p.parse_args()
    if a.list_models:
        for name,m in MODELS.items():
            print(f"{name:>4}  K={m.k} D_X={m.dx} eta in {m.eta_bounds} | {m.description}")
            print("      "+", ".join(f"{name}-{suffix}: Y in {get_variant(name+chr(45)+suffix).y_bounds}" for suffix in TRANSFORMS))
        return
    if a.list_metrics:
        for mid,s in METRICS.items():print(f"{mid} {s.name} [{s.behavior}]")
        return
    if a.describe_metric:
        if a.describe_metric not in METRICS:p.error("Unknown metric")
        s=METRICS[a.describe_metric]
        print(f"{s.id}  {s.name}\nFormula: {s.formula}\nNumerical: {s.numerical_method}"
              f"\nMeaning: {s.meaning}\nBehavior: {s.behavior}")
        return
    if a.u_points<2 or a.x_points<3 or a.angles<3 or a.sweep_steps<5:
        p.error("Require --u-points>=2, --x-points>=3, --angles>=3, --sweep-steps>=5")
    if not 0<a.sweep_radius<=.49 or not 0<a.rectangle_step<=.25:
        p.error("Invalid sweep radius or rectangle step")
    if a.joint_samples<2 or a.joint_restarts<0 or a.joint_maxiter<1:
        p.error("Require --joint-samples>=2, --joint-restarts>=0, --joint-maxiter>=1")
    if a.nn_train_samples<16 or a.nn_val_samples<16 or a.nn_epochs<1:
        p.error("Invalid network sample count or epoch count")
    if a.x_first_step<=0 or a.x_second_step<=0 or not 0<=a.v1_filter<=1 or not 0<=a.v2_filter<=1:
        p.error("Invalid operator steps/filters")
    run(a)

if __name__=="__main__":cli()
