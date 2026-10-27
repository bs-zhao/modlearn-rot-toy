"""Exact-function autodiff and chain-rule rotations; one derivative bank per model."""
from dataclasses import dataclass
import numpy as np
import torch
from .models import evaluate

@dataclass
class DerivativeBank:
    gtheta: np.ndarray
    htheta: np.ndarray
    hxtheta: np.ndarray
    gx: np.ndarray
    y: np.ndarray
    hxx: np.ndarray

def derivative_bank(model,link,x,theta,n_theta,n_x,analysis_scale="raw",backend=None):
    xx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    tt=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    yy=(backend.predict(xx,tt,analysis_scale) if backend is not None
        else evaluate(model,link,xx,tt,analysis_scale))
    gt,gx=torch.autograd.grad(yy.sum(),(tt,xx),create_graph=True)
    K,D=tt.shape[1],xx.shape[1]
    ht=[]
    for k in range(K):
        if gt[:,k].requires_grad:
            g=torch.autograd.grad(gt[:,k].sum(),tt,retain_graph=True,allow_unused=True)[0]
            if g is None:g=torch.zeros_like(tt)
        else:g=torch.zeros_like(tt)
        ht.append(g)
    hx=[];hxx=[]
    for j in range(D):
        if gx[:,j].requires_grad:
            g=torch.autograd.grad(gx[:,j].sum(),tt,retain_graph=True,allow_unused=True)[0]
            h=torch.autograd.grad(gx[:,j].sum(),xx,retain_graph=True,allow_unused=True)[0]
            if g is None:g=torch.zeros_like(tt)
            if h is None:h=torch.zeros_like(xx)
        else:g=torch.zeros_like(tt);h=torch.zeros_like(xx)
        hx.append(g);hxx.append(h)
    def conv(v,tail):return v.detach().cpu().numpy().reshape(n_theta,n_x,*tail)
    return DerivativeBank(conv(gt,(K,)),conv(torch.stack(ht,1),(K,K)),
                          conv(torch.stack(hx,1),(D,K)),conv(gx,(D,)),
                          conv(yy,()),conv(torch.stack(hxx,1),(D,D)))

def rotated_derivatives(bank,r):
    gu=np.einsum("nxk,ik->nxi",bank.gtheta,r,optimize=True)
    hu=np.einsum("ik,nxkl,jl->nxij",r,bank.htheta,r,optimize=True)
    hxu=np.einsum("nxdj,kj->nxdk",bank.hxtheta,r,optimize=True)
    return gu,hu,hxu
