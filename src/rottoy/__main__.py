"""Run one or multiple named toys; exact eta, differentiable F, or both."""
from pathlib import Path
import argparse,csv,json,itertools
import numpy as np
from .models import MODELS,WRAPPERS,plane_rotation
from .sampling import paired_grid
from .operators import derivative_bank
from .metrics import local_metrics,sweep_metrics
from .plots import make_figures,plot_xu_effect_map,plot_backend_comparison
from .backends import AnalyticBackend,train_neural,load_neural,checkpoint_path
from .historical_choice import (build_historical_banks,historical_metrics,save_historical_banks,original_predictive_rank)

def writerows(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def resolve_models(names):
    if "all" in names and len(names)>1:raise ValueError("Do not combine all with model names")
    selected=list(MODELS) if names==["all"] else names
    unknown=[n for n in selected if n not in MODELS]
    if unknown:raise ValueError(f"Unknown models {unknown}; run --list-models")
    return list(dict.fromkeys(selected))

def pair_indices(k,choice):
    if choice=="all":return list(itertools.combinations(range(k),2))
    if choice=="first":return [(0,1)]
    try:pair=tuple(int(n)-1 for n in choice.split(","))
    except ValueError:raise ValueError("--pairs must be first, all, or i,j")
    if len(pair)!=2 or min(pair)<0 or max(pair)>=k or pair[0]>=pair[1]:
        raise ValueError("--pairs must have 1<=i<j<=K")
    return [pair]

def bank_for(a,backend,model,wrapper,x,theta,x_count):
    b=derivative_bank(model,wrapper,x,theta,a.theta_samples,x_count,a.analysis_scale,backend=backend)
    for key in ("y","gtheta","gx","htheta","hxtheta","hxx"):
        if not np.isfinite(getattr(b,key)).all():
            raise RuntimeError(f"Nonfinite {key}: {model}/{wrapper}/{backend.kind}")
    return b

def bank_errors(a,b):
    errors={}
    for key in ("y","gtheta","gx","htheta","hxtheta","hxx"):
        p,q=getattr(a,key),getattr(b,key)
        errors[key+"_mse"]=float(np.mean((q-p)**2))
        errors[key+"_relative_rmse"]=float(np.sqrt(np.mean((q-p)**2)/max(np.mean(p*p),1e-15)))
    return errors

def run(a):
    output=Path(a.output)
    models=resolve_models(a.models)
    wrappers=list(WRAPPERS) if "all" in a.wrappers else a.wrappers
    if a.analysis_scale=="inverse-link" and any(w!="sigmoid" for w in wrappers):
        raise ValueError("inverse-link requires sigmoid only")
    angles=np.linspace(a.angle_min,a.angle_max,a.angles)
    if min(abs(angles))>1e-9:raise ValueError("Angle grid must include zero")
    summaries=[]
    for model in models:
        info=MODELS[model]
        x,theta,_,xpoints=paired_grid(info.k,info.dx,a.theta_samples,a.x_samples,a.seed,a.x_mode)
        for link in wrappers:
            available={}
            train_meta=None
            if a.backend in ("analytic","both"):
                available["analytic"]=AnalyticBackend(model,link)
            if a.backend in ("neural","both"):
                path=checkpoint_path(output,model,link)
                if a.train_neural:
                    nn,train_meta=train_neural(output,model,link,ntrain=a.nn_train_samples,
                        nval=a.nn_val_samples,epochs=a.nn_epochs,width=a.nn_width,
                        depth=a.nn_depth,lr=a.nn_lr,batch_size=a.nn_batch_size,seed=a.seed)
                elif path.is_file():nn=load_neural(path,model,link)
                else:raise FileNotFoundError(f"No checkpoint at {path}; use --train-neural")
                available["neural"]=nn
            banks={name:bank_for(a,b,model,link,x,theta,len(xpoints)) for name,b in available.items()}
            historical={}
            if not a.no_historical:
                for kind,backend in available.items():
                    print(f"[{model}/{link}/{kind}] building legacy v1/v2 functional operator banks...",flush=True)
                    hb=build_historical_banks(backend,link,x,theta,a.theta_samples,len(xpoints),
                        first_step_fraction=a.historical_first_step_fraction,
                        second_step_fraction=a.historical_second_step_fraction,
                        v1_min_relative_energy=a.historical_v1_min_relative_energy,
                        v2_min_relative_energy=a.historical_v2_min_relative_energy)
                    historical[kind]=hb
                    dest=output/model/link/a.analysis_scale/("analytic" if a.backend=="both" else kind)
                    dest.mkdir(parents=True,exist_ok=True)
                    save_historical_banks(dest/"historical_v1_v2_operators.npz",hb)
            for i,j in pair_indices(info.k,a.pairs):
                pname=f"u{i+1}_u{j+1}"
                folder=output/model/link/a.analysis_scale/pname
                folder.mkdir(parents=True,exist_ok=True)
                both={}
                for kind,backend in available.items():
                    target=folder if a.backend=="analytic" else folder/kind
                    target.mkdir(parents=True,exist_ok=True)
                    rows=[]
                    for phi in angles:
                        R=plane_rotation(info.k,i,j,float(phi))
                        vals=local_metrics(banks[kind],R)
                        if kind in historical: vals.update(historical_metrics(historical[kind],R))
                        vals.update(sweep_metrics(model,link,R,xpoints,a.sweep_steps,
                            a.sweep_radius,a.rectangle_step,a.analysis_scale,backend=backend))
                        rows.append(dict(angle_deg=round(float(phi),8),**vals))
                    writerows(target/"metrics_by_angle.csv",rows)
                    summary=make_figures(rows,target,f"{model}/{link}/{kind} - {pname}")
                    writerows(target/"metric_optima.csv",summary)
                    plot_xu_effect_map(banks[kind],np.eye(info.k),target,
                        f"{model}/{link}/{kind}: true coordinate XU map")
                    both[kind]=rows
                    summaries.extend(dict(model=model,wrapper=link,pair=pname,backend=kind,
                        scale=a.analysis_scale,**s) for s in summary)
                settings=dict(model=model,wrapper=link,pair=pname,k=info.k,dx=info.dx,
                    identifiable=info.identifiable,training=train_meta,**vars(a))
                if historical:
                    settings["historical_v1_v2"]={
                        kind:dict(fisher_eigenvalues=hb.fisher_eigenvalues.tolist(),
                            fisher_predictive_active_rank=original_predictive_rank(
                                hb.fisher_eigenvalues,
                                relative_threshold=a.historical_active_relative_threshold,
                                cumulative_energy=a.historical_active_cumulative_energy),
                            active_rank_is_diagnostic_only=True,
                            v1_operator_names=list(hb.names_v1),
                            v2_operator_names=list(hb.names_v2),
                            v1_operator_raw_energies=hb.energies_v1.tolist(),
                            v2_operator_raw_energies=hb.energies_v2.tolist(),
                            first_derivative_steps=hb.steps_first.tolist(),
                            second_derivative_steps=hb.steps_second.tolist(),
                            fisher_coordinate_space="full theta (no active subspace truncation)",
                            logit_scale="sigmoid link inverse, otherwise raw generator as logit")
                        for kind,hb in historical.items()}
                if a.backend=="both":
                    settings["metric_mse"]=plot_backend_comparison(both["analytic"],
                        both["neural"],folder,f"{model}/{link}/{pname}")
                    settings["operator_errors"]=bank_errors(banks["analytic"],banks["neural"])
                    paired=[]
                    for ea,ef in zip(both["analytic"],both["neural"]):
                        row=dict(angle_deg=ea["angle_deg"])
                        for metric in ea:
                            if metric=="angle_deg":continue
                            row[metric+"__eta"]=ea[metric]
                            row[metric+"__F"]=ef[metric]
                            row[metric+"__difference"]=ef[metric]-ea[metric]
                        paired.append(row)
                    writerows(folder/"paired_metrics_by_angle.csv",paired)
                (folder/"settings.json").write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding="utf-8")
                print(f"[{model}/{link}/{pname}] {a.backend}: {folder}",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    writerows(output/"all_metric_optima.csv",summaries)
    print(f"Completed {len(summaries)} metric rows in {output.resolve()}")

def cli():
    p=argparse.ArgumentParser(description="Extensible toy functional-rotation benchmark")
    p.add_argument("--list-models",action="store_true",help="Print all registered toy models")
    p.add_argument("--models",nargs="+",default=["T1","T2","T3","T4"],help="One, many, or all")
    p.add_argument("--wrappers",nargs="+",choices=("all",*WRAPPERS),default=["all"])
    p.add_argument("--backend",choices=("analytic","neural","both"),default="analytic")
    p.add_argument("--train-neural",action="store_true",help="Fit surrogate theta,X -> Y")
    p.add_argument("--nn-train-samples",type=int,default=4096)
    p.add_argument("--nn-val-samples",type=int,default=1024)
    p.add_argument("--nn-epochs",type=int,default=150)
    p.add_argument("--nn-width",type=int,default=64)
    p.add_argument("--nn-depth",type=int,default=3)
    p.add_argument("--nn-lr",type=float,default=.002)
    p.add_argument("--nn-batch-size",type=int,default=512)
    p.add_argument("--analysis-scale",choices=("raw","log","inverse-link"),default="raw")
    p.add_argument("--no-historical",action="store_true",help="Skip v1/v2 Fisher functional operators (faster)")
    p.add_argument("--historical-first-step-fraction",type=float,default=.02)
    p.add_argument("--historical-second-step-fraction",type=float,default=.10)
    p.add_argument("--historical-v1-min-relative-energy",type=float,default=1e-4)
    p.add_argument("--historical-active-relative-threshold",type=float,default=.001)
    p.add_argument("--historical-active-cumulative-energy",type=float,default=.995)
    p.add_argument("--historical-v2-min-relative-energy",type=float,default=.01)
    p.add_argument("--pairs",default="first")
    p.add_argument("--theta-samples",type=int,default=24)
    p.add_argument("--x-samples",type=int,default=100)
    p.add_argument("--x-mode",choices=("sobol","axis_sweeps"),default="sobol")
    p.add_argument("--sweep-steps",type=int,default=101)
    p.add_argument("--angle-min",type=float,default=-90)
    p.add_argument("--angle-max",type=float,default=90)
    p.add_argument("--angles",type=int,default=37)
    p.add_argument("--sweep-radius",type=float,default=.45)
    p.add_argument("--rectangle-step",type=float,default=.2)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--output",default="outputs/toy_rotation")
    a=p.parse_args()
    if a.list_models:
        for n,m in MODELS.items():print(f"{n:<8} theta={m.k} X={m.dx} | {m.description}")
        return
    if a.theta_samples<2 or a.x_samples<3 or a.sweep_steps<5 or a.angles<3:
        p.error("Need theta>=2, x>=3, sweep>=5, angles>=3")
    if not 0<a.sweep_radius<=.49 or not 0<a.rectangle_step<=.25:
        p.error("Invalid sweep radius or rectangle step")
    if a.nn_train_samples<16 or a.nn_val_samples<16 or a.nn_epochs<1 or a.nn_width<2 or a.nn_depth<1:
        p.error("Invalid neural training settings")
    run(a)

if __name__=="__main__":cli()
