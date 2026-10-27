"""M012–M016: nonlocal U sweeps of continuous response functions."""
import numpy as np
import torch
from ..models import evaluate

def _eval(variant,x,theta,scale="raw",backend=None):
    with torch.no_grad():
        xt=torch.as_tensor(np.array(x,copy=True),dtype=torch.float64)
        tt=torch.as_tensor(np.array(theta,copy=True),dtype=torch.float64)
        y=(backend.predict(xt,tt,scale) if backend is not None
           else evaluate(variant,xt,tt,scale))
    return y.detach().cpu().numpy()

def sweep_metrics(variant,r,x_bank,sweep_steps,sweep_radius,rect_step,scale="raw",backend=None):
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
        y=_eval(variant,xx,expanded,scale,backend).reshape(sweep_steps,len(x_bank))
        fitted=D@(inv@y)
        variance_scale=max(float(np.var(y)),1e-12)
        affine.append(float(np.mean((y-fitted)**2)/variance_scale))
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
                out.append(_eval(variant,x_bank,np.broadcast_to(theta,(len(x_bank),k)),scale,backend))
            diff=(out[0]-out[1]-out[2]+out[3])/(4*rect_step**2)
            rect.append(float(np.mean(diff**2)))
    return {
        "sweep_affine":float(np.mean(affine)),
        "sweep_shape_min":float(min(shape)),
        "sweep_rank1_min":float(min(rankerr)),
        "sweep_curvature":float(np.mean(curv)),
        "rectangle_uu":float(np.mean(rect)),
    }


