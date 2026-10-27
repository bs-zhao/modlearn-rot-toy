"""Shared analytic/neural X/U derivative operators, independent of toy identity."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class DerivativeBank:
    # Array shapes: (Nu,Nx,K), (Nu,Nx,K,K), (Nu,Nx,D,K), etc.
    gtheta: np.ndarray
    htheta: np.ndarray
    hxtheta: np.ndarray
    gx: np.ndarray
    y: np.ndarray
    hxx: np.ndarray

def derivative_bank(backend,x,theta,n_u_points,n_x_points,scale="raw"):
    xt=torch.as_tensor(np.array(x,copy=True),dtype=torch.float64).requires_grad_(True)
    tt=torch.as_tensor(np.array(theta,copy=True),dtype=torch.float64).requires_grad_(True)
    y=backend.predict(xt,tt,scale)
    gt,gx=torch.autograd.grad(y.sum(),(tt,xt),create_graph=True)
    k,d=tt.shape[1],xt.shape[1]
    ht,hxu,hxx=[],[],[]
    for j in range(k):
        if gt[:,j].requires_grad:
            temp=torch.autograd.grad(gt[:,j].sum(),tt,retain_graph=True,allow_unused=True)[0]
            ht.append(torch.zeros_like(tt) if temp is None else temp)
        else:ht.append(torch.zeros_like(tt))
    for j in range(d):
        if gx[:,j].requires_grad:
            a=torch.autograd.grad(gx[:,j].sum(),tt,retain_graph=True,allow_unused=True)[0]
            b=torch.autograd.grad(gx[:,j].sum(),xt,retain_graph=True,allow_unused=True)[0]
            hxu.append(torch.zeros_like(tt) if a is None else a)
            hxx.append(torch.zeros_like(xt) if b is None else b)
        else:
            hxu.append(torch.zeros_like(tt));hxx.append(torch.zeros_like(xt))
    def arr(v,*shape):return v.detach().cpu().numpy().reshape(n_u_points,n_x_points,*shape)
    return DerivativeBank(arr(gt,k),arr(torch.stack(ht,dim=1),k,k),
                          arr(torch.stack(hxu,dim=1),d,k),arr(gx,d),
                          arr(y),arr(torch.stack(hxx,dim=1),d,d))

def rotated_derivatives(bank,R):
    # R maps theta to u, with theta=R.T @ u.
    gu=np.einsum("nxk,ik->nxi",bank.gtheta,R,optimize=True)
    hu=np.einsum("ik,nxkl,jl->nxij",R,bank.htheta,R,optimize=True)
    hxu=np.einsum("nxdj,kj->nxdk",bank.hxtheta,R,optimize=True)
    return gu,hu,hxu
