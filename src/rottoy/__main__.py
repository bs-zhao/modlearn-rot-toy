"""Command line rotation diagnostic benchmark."""
from pathlib import Path
import argparse,csv,json,itertools
import numpy as np
from .models import MODELS,WRAPPERS,plane_rotation
from .sampling import paired_grid
from .operators import derivative_bank
from .metrics import local_metrics,sweep_metrics
from .plots import make_figures,plot_xu_effect_map

def writerows(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def run(args):
    output=Path(args.output)
    models=list(MODELS) if "all" in args.models else args.models
    wrappers=list(WRAPPERS) if "all" in args.wrappers else args.wrappers
    if args.analysis_scale=="inverse-link" and any(w!="sigmoid" for w in wrappers):
        raise ValueError("--analysis-scale inverse-link requires --wrappers sigmoid")
    angles=np.linspace(args.angle_min,args.angle_max,args.angles)
    if min(abs(angles))>1e-9:raise ValueError("Angle grid must include 0 degrees")
    summaries=[]
    for model in models:
        info=MODELS[model]
        x,theta,_,xpoints=paired_grid(info.k,info.dx,args.theta_samples,args.x_samples,args.seed)
        for link in wrappers:
            print(f"[{model}/{link}] analytic derivative bank...",flush=True)
            bank=derivative_bank(model,link,x,theta,args.theta_samples,args.x_samples,args.analysis_scale)
            if not all(np.isfinite(getattr(bank,key)).all() for key in ("gtheta","htheta","hxtheta","gx","hxx","y")):
                raise RuntimeError(f"Nonfinite derivatives: {model}/{link}")
            if args.pairs=="all":pairs=list(itertools.combinations(range(info.k),2))
            elif args.pairs=="first":pairs=[(0,1)]
            else:
                pair=tuple(int(q)-1 for q in args.pairs.split(","))
                if len(pair)!=2 or min(pair)<0 or max(pair)>=info.k or pair[0]>=pair[1]:
                    raise ValueError("--pairs should be first, all or 1-based pair i,j")
                pairs=[pair]
            for i,j in pairs:
                pname=f"u{i+1}_u{j+1}"
                folder=output/model/link/args.analysis_scale/pname
                folder.mkdir(parents=True,exist_ok=True)
                rows=[]
                for angle in angles:
                    r=plane_rotation(info.k,i,j,float(angle))
                    metrics=local_metrics(bank,r)
                    metrics.update(sweep_metrics(model,link,r,xpoints,args.sweep_steps,
                        args.sweep_radius,args.rectangle_step,args.analysis_scale))
                    rows.append(dict(angle_deg=round(float(angle),8),**metrics))
                writerows(folder/"metrics_by_angle.csv",rows)
                summary=make_figures(rows,folder,
                    f"{model} ({link}, {args.analysis_scale}), rotating {pname}")
                plot_xu_effect_map(bank,plane_rotation(info.k,i,j,0),folder,
                    f"{model} {link}, true axes at 0 deg")
                writerows(folder/"metric_optima.csv",summary)
                settings=dict(model=model,wrapper=link,pair=pname,k=info.k,dx=info.dx,
                    identifiable=info.identifiable,**vars(args),summary=summary)
                (folder/"settings.json").write_text(json.dumps(settings,indent=2,ensure_ascii=False),
                                                      encoding="utf-8")
                summaries.extend([dict(model=model,wrapper=link,pair=pname,
                    scale=args.analysis_scale,**entry) for entry in summary])
                report=[v for v in summary if v["metric"] in
                    ("xu_group_l1","grad_xcov_offdiag","uu_offdiag","sweep_affine")]
                print(" ",pname,[(v["metric"],v["argmin_angle_deg"]) for v in report],flush=True)
    output.mkdir(parents=True,exist_ok=True)
    writerows(output/"all_metric_optima.csv",summaries)
    print(f"Wrote {len(summaries)} metric rows under {output.resolve()}")

def cli():
    p=argparse.ArgumentParser(description="Analytic toy models: rotation vs functional simplicity")
    p.add_argument("--models",nargs="+",choices=("all",*MODELS),default=["T1","T2","T3","T4"])
    p.add_argument("--wrappers",nargs="+",choices=("all",*WRAPPERS),default=["all"])
    p.add_argument("--analysis-scale",choices=("raw","log","inverse-link"),default="raw")
    p.add_argument("--pairs",default="first",help="first, all, or 1-based 1,2")
    p.add_argument("--theta-samples",type=int,default=24)
    p.add_argument("--x-samples",type=int,default=100)
    p.add_argument("--sweep-steps",type=int,default=101)
    p.add_argument("--angle-min",type=float,default=-90)
    p.add_argument("--angle-max",type=float,default=90)
    p.add_argument("--angles",type=int,default=37)
    p.add_argument("--sweep-radius",type=float,default=.45)
    p.add_argument("--rectangle-step",type=float,default=.2)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--output",default="outputs/toy_rotation")
    a=p.parse_args()
    if a.theta_samples<2 or a.x_samples<3 or a.sweep_steps<5 or a.angles<3 or not 0<a.sweep_radius<=.49 or not 0<a.rectangle_step<=.25:
        p.error("theta>=2, x>=3, sweep>=5, angles>=3, sweep-radius (0,.49], rectangle-step (0,.25]")
    run(a)

if __name__=="__main__":cli()
