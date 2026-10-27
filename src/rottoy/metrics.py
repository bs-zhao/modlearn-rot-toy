"""Representative functional indicators. Some are deliberate negative controls."""
import numpy as np
import torch
from .models import evaluate
from .operators import rotated_derivatives

METRICS={
"xu_group_l1":"RMS group XU sparsity",
"xu_energy_concentration":"Quadratic energy concentration",
"local_xu_l1":"Per-location normalized row L1, C2 negative control",
"grad_xcov_offdiag":"U-gradient stimulus covariance offdiagonal",
"grad_gram_offdiag":"U gradient Gram offdiagonal",
"uu_offdiag":"UU mixed-curvature fraction",
"uu_diagonal":"UU self-curvature fraction",
"u_grad_energy":"Orthogonally invariant U gradient energy",
"xu_total_energy":"Orthogonally invariant XU squared energy",
"x_gradient_energy":"X gradient energy invariant control",
"x_curvature_energy":"XX Hessian energy invariant control",
"sweep_affine":"Best affine U-sweep fit residual",
"sweep_shape_min":"Minimum normalized response-shape variation",
"sweep_rank1_min":"Minimum rank-one sweep matrix approximation error",
"sweep_curvature":"Second finite difference sweep curvature",
"rectangle_uu":"Finite rectangle UU interaction squared"
}
INVARIANTS={"u_grad_energy","xu_total_energy","x_gradient_energy","x_curvature_energy"}

def _offdiag_sq(a):
    return np.sum(a*a)-np.sum(np.diag(a)**2)

def local_metrics(bank,r,eps=1e-12):
    gu,hu,hxu=rotated_derivatives(bank,r)
    groups=np.sqrt(np.mean(np.sum(hxu**2,axis=2),axis=(0,1)))
    gn=np.linalg.norm(groups)
    group_l1=float(groups.sum()/(np.sqrt(len(groups))*gn)) if gn>eps else 0.
    e=groups**2
    concentrate=float(1-np.sum((e/e.sum())**2)) if e.sum()>eps else 0.
    localnorm=np.linalg.norm(hxu,axis=-1)
    loc_l1=np.divide(np.sum(np.abs(hxu),axis=-1),localnorm,
        out=np.zeros_like(localnorm),where=localnorm>eps)
    loc_l1=float(np.sum(loc_l1)/max(np.count_nonzero(localnorm>eps),1))
    centered=gu-gu.mean(axis=1,keepdims=True)
    cov=np.mean(np.einsum("nxk,nxl->nkl",centered,centered,optimize=True)/gu.shape[1],axis=0)
    gram=np.mean(np.einsum("nxk,nxl->nkl",gu,gu,optimize=True)/gu.shape[1],axis=0)
    cn=np.sum(cov*cov);gn2=np.sum(gram*gram)
    xc=float(max(0,_offdiag_sq(cov))/cn) if cn>eps else 0.
    gg=float(max(0,_offdiag_sq(gram))/gn2) if gn2>eps else 0.
    HS=float(np.mean(np.sum(hu*hu,axis=(-1,-2))))
    HD=float(np.mean(np.sum(np.diagonal(hu,axis1=-2,axis2=-1)**2,axis=-1)))
    return {
        "xu_group_l1":group_l1,
        "xu_energy_concentration":concentrate,
        "local_xu_l1":loc_l1,
        "grad_xcov_offdiag":xc,
        "grad_gram_offdiag":gg,
        "uu_offdiag":float(max(0.,HS-HD)/HS) if HS>eps else 0.,
        "uu_diagonal":float(HD/HS) if HS>eps else 0.,
        "u_grad_energy":float(np.mean(np.sum(gu*gu,axis=-1))),
        "xu_total_energy":float(np.mean(np.sum(hxu*hxu,axis=(-1,-2)))),
        "x_gradient_energy":float(np.mean(np.sum(bank.gx**2,axis=-1))),
        "x_curvature_energy":float(np.mean(np.sum(bank.hxx**2,axis=(-1,-2))))
    }

def _eval(model,link,x,theta,scale="raw",backend=None):
    with torch.no_grad():
        xt=torch.as_tensor(np.array(x,copy=True),dtype=torch.float64)
        tt=torch.as_tensor(np.array(theta,copy=True),dtype=torch.float64)
        y=(backend.predict(xt,tt,scale) if backend is not None
           else evaluate(model,link,xt,tt,scale))
    return y.detach().cpu().numpy()

def sweep_metrics(model,link,r,x_bank,sweep_steps,sweep_radius,rect_step,analysis_scale="raw",backend=None):
    k=len(r);offsets=np.linspace(-sweep_radius,sweep_radius,sweep_steps)
    center=np.full(k,.5)
    if np.any(center[None,:]+offsets[:,None]*r[0][None,:]<0):
        raise ValueError("U sweep outside native theta cube")
    xx=np.tile(x_bank,(sweep_steps,1))
    D=np.column_stack((np.ones_like(offsets),offsets))
    inv=np.linalg.pinv(D)
    affine=[];shape=[];rankerr=[];curv=[]
    for axis in range(k):
        theta=center[None,:]+offsets[:,None]*r[axis][None,:]
        if theta.min()<-1e-10 or theta.max()>1+1e-10:raise ValueError("Theta outside [0,1]")
        expanded=np.repeat(theta,len(x_bank),axis=0)
        y=_eval(model,link,xx,expanded,analysis_scale,backend).reshape(sweep_steps,len(x_bank))
        fitted=D@(inv@y)
        scale=max(float(np.var(y)),1e-12)
        affine.append(float(np.mean((y-fitted)**2)/scale))
        magn=np.linalg.norm(y,axis=1,keepdims=True)
        normalized=y/np.maximum(magn,1e-12)
        shape.append(float(np.mean((normalized-normalized.mean(axis=0))**2)))
        gram=y@y.T if y.shape[0]<=y.shape[1] else y.T@y
        largest=max(float(np.linalg.eigvalsh(gram)[-1]),0.)
        rankerr.append(float(max(0.,1-largest/max(float(np.sum(y*y)),1e-12))))
        curv.append(float(np.mean(np.diff(y,n=2,axis=0)**2)/(offsets[1]-offsets[0])**4))
    rect=[]
    for i in range(k):
        for j in range(i+1,k):
            out=[]
            for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1)):
                theta=center+rect_step*(si*r[i]+sj*r[j])
                if theta.min()<0 or theta.max()>1:raise ValueError("UU rectangle outside [0,1]")
                out.append(_eval(model,link,x_bank,np.broadcast_to(theta,(len(x_bank),k)),analysis_scale,backend))
            diff=(out[0]-out[1]-out[2]+out[3])/(4*rect_step**2)
            rect.append(float(np.mean(diff**2)))
    return {
        "sweep_affine":float(np.mean(affine)),
        "sweep_shape_min":float(min(shape)),
        "sweep_rank1_min":float(min(rankerr)),
        "sweep_curvature":float(np.mean(curv)),
        "rectangle_uu":float(np.mean(rect)),
    }
