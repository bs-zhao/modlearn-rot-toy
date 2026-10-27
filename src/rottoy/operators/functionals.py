"""X/U numerical function operators (v1/v2 families), all toy-agnostic.

A function is continuous-valued. Even for sigmoid variants there is no
Bernoulli sampling. The optional inverse analysis scale recovers base eta.
"""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class OperatorBank:
    info_matrix: np.ndarray
    info_eigenvalues: np.ndarray
    names_v1: tuple
    operators_v1: np.ndarray
    energies_v1: np.ndarray
    names_v2: tuple
    operators_v2: np.ndarray
    energies_v2: np.ndarray
    families_v2: tuple
    names_all: tuple
    raw_matrices: np.ndarray
    raw_energies: np.ndarray
    steps_first: np.ndarray
    steps_second: np.ndarray

def _gradient(backend,x,theta,scale="raw",batch_size=256):
    if len(x)!=len(theta):raise ValueError("X and theta must have paired rows")
    gradients=[];outputs=[]
    for i in range(0,len(x),batch_size):
        xt=torch.as_tensor(np.array(x[i:i+batch_size],copy=True),dtype=torch.float64)
        tt=torch.as_tensor(np.array(theta[i:i+batch_size],copy=True),
                           dtype=torch.float64).requires_grad_(True)
        with torch.enable_grad():
            y=backend.predict(xt,tt,scale)
            g=torch.autograd.grad(y.sum(),tt)[0]
        gradients.append(g.detach().numpy());outputs.append(y.detach().numpy())
    g=np.concatenate(gradients);h=np.concatenate(outputs)
    if not np.isfinite(g).all() or not np.isfinite(h).all():
        raise ValueError("Non-finite operator values/derivatives")
    return g,h

def _first_stencil(x,j,h):
    xp=x.copy();xm=x.copy()
    low=x[:,j]<h;high=x[:,j]>1-h
    xp[:,j]+=np.where(high,0.,h)
    xm[:,j]+=np.where(low,0.,-h)
    return xp,xm,xp[:,j]-xm[:,j]

def _second_stencil(x,j,h):
    xp=x.copy();xm=x.copy()
    low=x[:,j]<h;high=x[:,j]>1-h
    xp[:,j]+=np.where(high,-h,h)
    xm[:,j]+=np.where(low,2*h,np.where(high,-2*h,-h))
    return xp,xm,np.where(low,1,np.where(high,-1,0))

def _second_diff(base,plus,minus,flags,h):
    return np.where((flags==1)[:,None],(minus-2*plus+base)/h**2,
        np.where((flags==-1)[:,None],(base-2*plus+minus)/h**2,
                 (plus-2*base+minus)/h**2))

def _filter_normalize(names,families,mats,threshold):
    mats=np.asarray(mats,dtype=np.float64)
    energy=np.linalg.norm(mats,axis=(-2,-1))
    bound=max(float(np.max(energy)),1e-15)
    mask=(energy>=threshold*bound)&(energy>1e-14)
    k=mats.shape[-1]
    return (tuple(np.array(names)[mask]),mats[mask]/energy[mask,None,None]
            if mask.any() else np.zeros((0,k,k)),
            energy[mask],tuple(np.array(families)[mask]))

def operator_bank(backend,x,theta,n_u,n_x,*,scale="raw",first_step=.02,
                  second_step=.10,v1_filter=1e-4,v2_filter=.01,batch_size=256):
    x=np.asarray(x,dtype=float);theta=np.asarray(theta,dtype=float)
    if len(x)!=len(theta) or len(x)!=n_u*n_x:raise ValueError("Expected Nu x Nx grid")
    if np.any(x<0) or np.any(x>1):raise ValueError("X must be in [0,1]")
    k=theta.shape[-1];d=x.shape[-1]
    sd=np.std(x,axis=0)
    if np.any(sd<1e-9):raise ValueError("Each X dimension needs nonzero variance")
    h1=np.maximum(first_step*sd,1e-5)
    h2=np.maximum(second_step*sd,1e-4)
    if np.max(h2)>=.49 or np.max(h1)>=.49:raise ValueError("Finite-difference step too large")
    g,h=_gradient(backend,x,theta,scale,batch_size)
    # This is a sigmoid-weighted functional information operator, not a
    # model-specific Fisher likelihood for continuous Y.
    p=1/(1+np.exp(-np.clip(h,-50,50)))
    info=np.einsum("n,ni,nj->ij",p*(1-p),g,g,optimize=True)/len(g)
    info=.5*(info+info.T)
    names=["level"];families=["level"];signatures=[g.reshape(n_u,n_x,k).mean(1)]
    def include(name,family,values):
        signatures.append(values.reshape(n_u,n_x,k).mean(axis=1))
        names.append(name);families.append(family)
    for j in range(d):
        xp,xm,den=_first_stencil(x,j,h1[j])
        gp,_=_gradient(backend,xp,theta,scale,batch_size)
        gm,_=_gradient(backend,xm,theta,scale,batch_size)
        include(f"X{j+1}:first","first",(gp-gm)*sd[j]/den[:,None])
    for j in range(d):
        xp,xm,flags=_second_stencil(x,j,h2[j])
        gp,_=_gradient(backend,xp,theta,scale,batch_size)
        gm,_=_gradient(backend,xm,theta,scale,batch_size)
        sec=_second_diff(g,gp,gm,flags,h2[j])*sd[j]**2
        include(f"X{j+1}:second","second_diagonal",sec)
    for j in range(d):
        for l in range(j+1,d):
            jp,jm,denj=_first_stencil(x,j,h2[j])
            lp,lm,denl=_first_stencil(x,l,h2[l])
            corners=[]
            for a,b in ((jp,lp),(jp,lm),(jm,lp),(jm,lm)):
                xx=x.copy();xx[:,j]=a[:,j];xx[:,l]=b[:,l]
                corner,_=_gradient(backend,xx,theta,scale,batch_size)
                corners.append(corner)
            diff=(corners[0]-corners[1]-corners[2]+corners[3])*sd[j]*sd[l]/(denj*denl)[:,None]
            include(f"X{j+1}X{l+1}:mixed","second_mixed",diff)
    signatures=np.stack(signatures,axis=1)
    matrices=np.einsum("nok,nol->okl",signatures,signatures,optimize=True)/n_u
    raw=np.linalg.norm(matrices,axis=(1,2))
    size_v1=1+d
    n1,m1,e1,f1=_filter_normalize(names[:size_v1],families[:size_v1],
                                   matrices[:size_v1],v1_filter)
    n2,m2,e2,f2=_filter_normalize(names,families,matrices,v2_filter)
    return OperatorBank(info,np.linalg.eigvalsh(info)[::-1],n1,m1,e1,n2,m2,e2,f2,
                        tuple(names),matrices,raw,h1,h2)

def _jd(matrices,R):
    if not len(matrices):return 0.
    rotated=np.einsum("ik,mkl,jl->mij",R,matrices,R,optimize=True)
    energy=np.sum(rotated**2,axis=(-2,-1))-np.sum(np.diagonal(rotated,axis1=-2,axis2=-1)**2,axis=-1)
    return float(np.mean(np.maximum(0,energy)))

def operator_metrics(bank,R):
    M=R@bank.info_matrix@R.T
    result={
       "operator_v1_jd":_jd(bank.operators_v1,R),
       "operator_v2_jd":_jd(bank.operators_v2,R),
       "operator_fisher_offdiag":float(max(0,np.sum(M*M)-np.sum(np.diag(M)**2))),
       "operator_fisher_trace":float(np.trace(M)),
    }
    for metric,operators,families in (
        ("operator_v1_level",bank.operators_v1,
            ["level" if n=="level" else "other" for n in bank.names_v1]),
        ("operator_v1_first",bank.operators_v1,
            ["first" if n.endswith(":first") else "other" for n in bank.names_v1]),
        ("operator_v2_second_diagonal",bank.operators_v2,bank.families_v2),
        ("operator_v2_second_mixed",bank.operators_v2,bank.families_v2),
    ):
        family={"operator_v1_level":"level","operator_v1_first":"first",
                "operator_v2_second_diagonal":"second_diagonal",
                "operator_v2_second_mixed":"second_mixed"}[metric]
        mask=np.array([v==family for v in families])
        result[metric]=_jd(operators[mask],R)
    return result

def save_operator_bank(filename,bank):
    np.savez_compressed(filename,info=bank.info_matrix,eigenvalues=bank.info_eigenvalues,
        v1_names=np.array(bank.names_v1,str),v1_ops=bank.operators_v1,
        v1_energies=bank.energies_v1,v2_names=np.array(bank.names_v2,str),
        v2_ops=bank.operators_v2,v2_energies=bank.energies_v2,
        v2_families=np.array(bank.families_v2,str),
        all_names=np.array(bank.names_all,str),all_ops=bank.raw_matrices,
        all_energies=bank.raw_energies,first_steps=bank.steps_first,
        second_steps=bank.steps_second)
