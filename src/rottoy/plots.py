"""Per-model indicator heatmaps, rotation curves and XU functional effect matrices."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .operators import rotated_derivatives

METRIC_ORDER=[
"xu_group_l1","xu_energy_concentration","local_xu_l1",
"grad_xcov_offdiag","grad_gram_offdiag","uu_offdiag","uu_diagonal",
"sweep_affine","sweep_shape_min","sweep_rank1_min","sweep_curvature",
"rectangle_uu","u_grad_energy","xu_total_energy",
"x_gradient_energy","x_curvature_energy"
]

def make_figures(rows,directory:Path,title:str):
    directory.mkdir(parents=True,exist_ok=True)
    rows=sorted(rows,key=lambda r:float(r["angle_deg"]))
    xs=np.array([float(r["angle_deg"]) for r in rows])
    names=[n for n in METRIC_ORDER if n in rows[0]]
    data=np.array([[float(r[n]) for r in rows] for n in names])
    lo=data.min(axis=1,keepdims=True);hi=data.max(axis=1,keepdims=True)
    normalized=np.divide(data-lo,hi-lo,out=np.full_like(data,.5),where=(hi-lo)>1e-12)
    fig,ax=plt.subplots(figsize=(11,6.6),layout="constrained")
    im=ax.imshow(normalized,aspect="auto",cmap="viridis",origin="upper",vmin=0,vmax=1,
        extent=[xs[0]-(xs[1]-xs[0])/2,xs[-1]+(xs[1]-xs[0])/2,len(names)-.5,-.5])
    ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(names,fontsize=9)
    ax.set_xticks(np.arange(np.ceil(xs[0]/15)*15,xs[-1]+1,15))
    ax.set_xlabel("Rotation angle (degrees)")
    ax.axvline(0,color="white",linewidth=1.2,linestyle="--",alpha=.85)
    ax.set_title(title+" | metric heatmap (rowwise min-max)",fontsize=12,fontweight="bold")
    fig.colorbar(im,ax=ax,label="Normalized per metric (constant row = 0.5)",shrink=.77)
    fig.savefig(directory/"metric_heatmap.png",dpi=145)
    plt.close(fig)
    show=["xu_group_l1","grad_xcov_offdiag","uu_offdiag","uu_diagonal",
          "sweep_affine","rectangle_uu"]
    fig,axes=plt.subplots(2,3,figsize=(12.5,7),layout="constrained")
    for ax,name in zip(axes.flat,show):
        if name in names:
            ax.plot(xs,data[names.index(name)],linewidth=2)
            ax.scatter([0],[float(rows[np.argmin(abs(xs))][name])],s=22,zorder=3)
        ax.axvline(0,color="gray",linewidth=.9,linestyle="--")
        ax.grid(alpha=.2);ax.set_title(name,fontsize=10)
        ax.set_xlabel("Rotation angle (degrees)");ax.set_ylabel("Raw metric")
    fig.suptitle(title+" | selected metric curves",fontweight="bold")
    fig.savefig(directory/"metric_curves.png",dpi=145)
    plt.close(fig)
    summary=[]
    for idx,name in enumerate(names):
        best=int(np.argmin(data[idx]));zero=int(np.argmin(abs(xs)))
        summary.append(dict(metric=name,true_axis_value=float(data[idx,zero]),
            min_value=float(data[idx,best]),argmin_angle_deg=float(xs[best]),
            range=float(np.ptp(data[idx])),
            best_is_true_axis=bool(abs(xs[best])<1e-9 or
                np.isclose(data[idx,zero],data[idx,best],rtol=1e-5,atol=1e-10))))
    return summary

def plot_xu_effect_map(bank,r,folder:Path,label:str):
    _,_,xu=rotated_derivatives(bank,r)
    mat=np.sqrt(np.mean(xu*xu,axis=(0,1)))
    fig,ax=plt.subplots(figsize=(max(5,.8*mat.shape[1]+2),max(3,.5*mat.shape[0]+2)),layout="constrained")
    im=ax.imshow(mat,aspect="auto",cmap="magma",vmin=0)
    ax.set_xticks(range(mat.shape[1]),[f"u{i+1}" for i in range(mat.shape[1])])
    ax.set_yticks(range(mat.shape[0]),[f"x{i+1}" for i in range(mat.shape[0])])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,f"{mat[i,j]:.2g}",ha="center",va="center",fontsize=8,
                    color="white" if mat[i,j]<.6*mat.max() else "black")
    ax.set_title(label+" | RMS mixed XU gradient")
    fig.colorbar(im,ax=ax,label="RMS |d2 h/dX dU|",shrink=.8)
    fig.savefig(folder/"xu_effect_map.png",dpi=145)
    plt.close(fig)

def plot_backend_comparison(analytic_rows,neural_rows,folder,title):
    """Compare F and eta on matching angle and metric grid."""
    folder.mkdir(parents=True,exist_ok=True)
    names=[m for m in METRIC_ORDER if m in analytic_rows[0]]
    xs=np.array([r["angle_deg"] for r in analytic_rows],float)
    if not np.allclose(xs,[r["angle_deg"] for r in neural_rows]):raise ValueError("Angle mismatch")
    A=np.array([[r[m] for r in analytic_rows] for m in names],float)
    F=np.array([[r[m] for r in neural_rows] for m in names],float)
    scale=np.maximum(np.ptp(A,axis=1,keepdims=True),1e-10)
    fig,ax=plt.subplots(figsize=(11,6.6),layout="constrained")
    im=ax.imshow(np.abs(F-A)/scale,aspect="auto",cmap="magma",origin="upper",
        extent=[xs[0]-(xs[1]-xs[0])/2,xs[-1]+(xs[1]-xs[0])/2,len(names)-.5,-.5])
    ax.set_yticks(range(len(names)),names,fontsize=9)
    ax.axvline(0,color="white",ls="--")
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_title(title+" | normalized absolute F - eta metric difference",fontsize=11)
    fig.colorbar(im,ax=ax,label="Absolute gap / analytic angle range",shrink=.8)
    fig.savefig(folder/"paired_error_heatmap.png",dpi=145);plt.close(fig)
    show=["xu_group_l1","grad_xcov_offdiag","uu_offdiag","uu_diagonal","sweep_affine","rectangle_uu"]
    fig,axes=plt.subplots(2,3,figsize=(12.5,7),layout="constrained")
    for ax,name in zip(axes.flat,show):
        if name in names:
            idx=names.index(name)
            ax.plot(xs,A[idx],label="analytic eta",lw=2)
            ax.plot(xs,F[idx],label="neural F",ls="--",lw=2)
        ax.axvline(0,color="gray",ls=":",lw=.8)
        ax.set_title(name,fontsize=10);ax.set_xlabel("Rotation angle (degrees)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title+" | corresponding metrics")
    fig.savefig(folder/"paired_overlay_curves.png",dpi=145);plt.close(fig)
    return {m:float(np.mean((F[i]-A[i])**2)) for i,m in enumerate(names)}
