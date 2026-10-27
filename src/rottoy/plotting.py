"""Generic figures for any Tn-suffix and any subset of M metrics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import METRICS
from .operators import rotated_derivatives

def write_figures(rows,folder,title):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=[mid for mid in METRICS if mid in rows[0]]
    angles=np.array([float(r["angle_deg"]) for r in rows])
    Y=np.array([[float(r[mid]) for r in rows] for mid in names])
    if len(angles)<2:raise ValueError("Need >=2 angles for plots")
    vmin=np.min(Y,axis=1,keepdims=True);rng=np.ptp(Y,axis=1,keepdims=True)
    color=np.divide(Y-vmin,rng,out=np.full_like(Y,.5),where=rng>1e-12)
    fig,ax=plt.subplots(figsize=(11,max(3.2,len(names)*.36+1.9)),layout="constrained")
    im=ax.imshow(color,aspect="auto",origin="upper",cmap="viridis",
        extent=[angles[0]-.5*(angles[1]-angles[0]),angles[-1]+.5*(angles[-1]-angles[-2]),len(names)-.5,-.5])
    ax.set_yticks(range(len(names)),[f"{mid}  {METRICS[mid].name}" for mid in names],fontsize=8)
    ax.axvline(0,c="white",ls="--",lw=1.1)
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_title(title+" | row-normalized metric heatmap")
    fig.colorbar(im,ax=ax,label="Within-row normalized value (flat = .5)",shrink=.75)
    fig.savefig(folder/"metric_heatmap.png",dpi=150);plt.close(fig)

    ncols=min(3,len(names));nrows=int(np.ceil(len(names)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,2.9*nrows),layout="constrained")
    for i,ax in enumerate(np.atleast_1d(axes).flat):
        if i>=len(names):ax.axis("off");continue
        mid=names[i]
        ax.plot(angles,Y[i],lw=1.7)
        ax.axvline(0,c="gray",ls="--",lw=.8)
        ax.grid(alpha=.2)
        ax.set_title(f"{mid}: {METRICS[mid].name}",fontsize=9)
        ax.set_xlabel("Angle (°)");ax.set_ylabel("Raw metric")
    fig.suptitle(title+" | all selected metrics",fontsize=12)
    fig.savefig(folder/"metric_curves.png",dpi=150);plt.close(fig)

    zero=int(np.argmin(abs(angles)))
    minima=[]
    for i,mid in enumerate(names):
        idx=int(np.argmin(Y[i]));flat=bool(np.ptp(Y[i])<=1e-10*max(1,np.max(abs(Y[i]))))
        minima.append(dict(metric=mid,name=METRICS[mid].name,
            at_zero=float(Y[i,zero]),min_value=float(Y[i,idx]),
            argmin_angle_deg=float(angles[idx]),flat=flat,
            min_includes_zero=bool(np.isclose(Y[i,zero],Y[i,idx],rtol=1e-6,atol=1e-10))))
    return minima

def plot_xu(bank,folder,title):
    _,_,xu=rotated_derivatives(bank,np.eye(bank.gtheta.shape[-1]))
    matrix=np.sqrt(np.mean(xu*xu,axis=(0,1)))
    fig,ax=plt.subplots(figsize=(max(5,matrix.shape[1]+2),max(3,matrix.shape[0]*.7+2)),layout="constrained")
    im=ax.imshow(matrix,cmap="magma",aspect="auto")
    ax.set_xticks(range(matrix.shape[1]),[f"u{k+1}" for k in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]),[f"x{j+1}" for j in range(matrix.shape[0])])
    fig.colorbar(im,ax=ax,shrink=.85,label="RMS ∂²h/∂X∂u")
    ax.set_title(title)
    fig.savefig(Path(folder)/"xu_effect_map.png",dpi=150);plt.close(fig)

def plot_comparison(analytic,neural,folder,title):
    ids=[mid for mid in METRICS if mid in analytic[0]]
    angles=np.array([r["angle_deg"] for r in analytic])
    if not np.allclose(angles,[r["angle_deg"] for r in neural]):
        raise ValueError("Backends must share identical angle grid")
    A=np.array([[r[m] for r in analytic] for m in ids])
    F=np.array([[r[m] for r in neural] for m in ids])
    ncols=min(3,len(ids));nrows=int(np.ceil(len(ids)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,3*nrows),layout="constrained")
    for idx,ax in enumerate(np.atleast_1d(axes).flat):
        if idx>=len(ids):ax.axis("off");continue
        ax.plot(angles,A[idx],label="analytic eta",lw=1.8)
        ax.plot(angles,F[idx],label="neural F",lw=1.8,ls="--")
        ax.axvline(0,c="gray",ls=":",lw=.9)
        ax.grid(alpha=.2);ax.set_title(ids[idx]+" "+METRICS[ids[idx]].name,fontsize=8)
    np.atleast_1d(axes).flat[0].legend(fontsize=8)
    fig.suptitle(title+" | paired metrics")
    fig.savefig(Path(folder)/"paired_overlay_curves.png",dpi=150);plt.close(fig)
    denom=np.maximum(np.ptp(A,axis=1,keepdims=True),1e-10)
    values=abs(F-A)/denom
    fig,ax=plt.subplots(figsize=(11,max(3.3,len(ids)*.35+1.8)),layout="constrained")
    im=ax.imshow(values,origin="upper",aspect="auto",cmap="magma",
        extent=[angles[0]-(angles[1]-angles[0])/2,angles[-1]+(angles[-1]-angles[-2])/2,len(ids)-.5,-.5])
    ax.set_yticks(range(len(ids)),ids)
    ax.axvline(0,c="white",ls="--")
    ax.set_xlabel("Rotation angle (degrees)");ax.set_title(title+" | normalized F−eta discrepancy")
    fig.colorbar(im,ax=ax,label="Absolute metric error / analytic metric range",shrink=.75)
    fig.savefig(Path(folder)/"paired_error_heatmap.png",dpi=150);plt.close(fig)
    return {mid:float(np.mean((F[i]-A[i])**2)) for i,mid in enumerate(ids)}
