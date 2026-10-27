"""Historical choice v1_distributional / v2_higher_order operator metrics.

Faithfully reconstructs the *orientation operator construction* of
Cognition-Decision-Modeling-Lab/modlearn-rot-2:
  - binary log-odds theta gradients
  - per-person means over X (BEFORE outer products)
  - SD(X) and SD(X)^2 normalization of X derivatives
  - v1 level + first X derivatives
  - v2 level + first + diagonal XX + mixed XX derivatives
  - raw energy filter, per-operator Frobenius normalization, joint offdiagonal

Toy theta is already the full  K-dimensional coordinate; historical Fisher
active-space rank selection is reported as diagnostics but NOT imposed here.
X is sampled on [0,1] per toy configuration, unlike the original C2's [-1,1].
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class HistoricalBanks:
    fisher: np.ndarray
    fisher_eigenvalues: np.ndarray
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

def _logit_gradient(backend, x, theta, wrapper, batch_size=256):
    if len(x)!=len(theta):raise ValueError("Paired X and theta required")
    grad=[];values=[]
    # For categorical Fisher and historical signatures work on LOG ODDS.
    # identity/square responses are interpreted as binary choice logits;
    # sigmoid responses are probabilities and mapped back via logit.
    scale="inverse-link" if wrapper=="sigmoid" else "raw"
    for start in range(0,len(x),batch_size):
        end=min(start+batch_size,len(x))
        xt=torch.as_tensor(np.array(x[start:end],copy=True),dtype=torch.float64)
        tt=torch.tensor(np.array(theta[start:end],copy=True),dtype=torch.float64,requires_grad=True)
        with torch.enable_grad():
            y=backend.predict(xt,tt,scale)
            g=torch.autograd.grad(y.sum(),tt)[0]
        values.append(y.detach().cpu().numpy())
        grad.append(g.detach().cpu().numpy())
    g=np.concatenate(grad);v=np.concatenate(values)
    if not(np.isfinite(g).all() and np.isfinite(v).all()):
        raise ValueError("Nonfinite log-odds gradient or output; check sigmoid saturation")
    return g,v

def _first_pair(x,j,h):
    xp=x.copy();xm=x.copy()
    lower=x[:,j]<h;upper=x[:,j]>1-h
    if np.any(lower&upper):raise ValueError("Step exceeds coordinate domain")
    xp[:,j]+=np.where(upper,0.,h)
    xm[:,j]-=np.where(lower,0.,h)
    return xp,xm,xp[:,j]-xm[:,j]

def _second_triplet(x,j,h):
    xp=x.copy();xm=x.copy()
    lower=x[:,j]<h;upper=x[:,j]>1-h
    if np.any(lower&upper):raise ValueError("Step exceeds coordinate domain")
    xp[:,j]+=np.where(upper,-h,h)
    xm[:,j]+=np.where(lower,2*h,np.where(upper,-2*h,-h))
    flag=np.where(lower,1,np.where(upper,-1,0))
    return xp,xm,flag

def _second_combine(base,g1,g2,flag,h):
    return np.where((flag==1)[:,None],(g2-2*g1+base)/h**2,
        np.where((flag==-1)[:,None],(base-2*g1+g2)/h**2,
            (g1-2*base+g2)/h**2))

def _normalize(names, families, matrices, relative):
    mats=np.asarray(matrices,dtype=np.float64)
    energies=np.linalg.norm(mats,axis=(1,2))
    maxenergy=max(float(np.max(energies)),1e-15)
    keep=(energies>=relative*maxenergy) & (energies>1e-14)
    if not keep.any():
        k=mats.shape[-1]
        return (),np.empty((0,k,k),float),np.empty(0),()
    return (tuple(np.asarray(names)[keep]),mats[keep]/energies[keep,None,None],
            energies[keep],tuple(np.asarray(families)[keep]))

def build_historical_banks(backend,wrapper,x,theta,n_subjects,n_x,
                           *,first_step_fraction=.02,second_step_fraction=.10,
                           v1_min_relative_energy=1e-4,
                           v2_min_relative_energy=.01,batch_size=256):
    x=np.asarray(x,np.float64);theta=np.asarray(theta,np.float64)
    n,d=x.shape;k=theta.shape[1]
    if len(x)!=n_subjects*n_x or n_x<2 or n_subjects<2:
        raise ValueError("Expected theta subjects crossed with observed X nodes")
    if not np.all((x>=0)&(x<=1)) or np.any(~np.isfinite(x)):
        raise ValueError("Toy X must lie in [0,1]")
    sd=np.std(x,axis=0)
    if (sd<=1e-8).any():raise ValueError("Constant X dimension")
    h1=np.maximum(first_step_fraction*sd,1e-5)
    h2=np.maximum(second_step_fraction*sd,1e-4)
    if any(h>0.49 for h in (*h1,*h2)):raise ValueError("FD steps too large")

    g,eta=_logit_gradient(backend,x,theta,wrapper,batch_size)
    p=torch.sigmoid(torch.from_numpy(eta)).numpy()
    fisher=np.einsum("n,ni,nj->ij",p*(1-p),g,g,optimize=True)/n
    fisher=(fisher+fisher.T)/2
    eigen=np.linalg.eigvalsh(fisher)[::-1]
    base=g.reshape(n_subjects,n_x,k)
    names=["level_logodds_1"];families=["level"];sigs=[base.mean(axis=1)]
    def add(name,fam,pertrial):
        z=pertrial.reshape(n_subjects,n_x,k).mean(axis=1)
        sigs.append(z);names.append(name);families.append(fam)

    for j in range(d):
        xp,xm,den=_first_pair(x,j,h1[j])
        gp,_=_logit_gradient(backend,xp,theta,wrapper,batch_size)
        gm,_=_logit_gradient(backend,xm,theta,wrapper,batch_size)
        first=(gp-gm)*sd[j]/den[:,None]
        add(f"dX{j+1}_logodds_1","first",first)

    for j in range(d):
        xp,xm,flag=_second_triplet(x,j,h2[j])
        gp,_=_logit_gradient(backend,xp,theta,wrapper,batch_size)
        gm,_=_logit_gradient(backend,xm,theta,wrapper,batch_size)
        sec=_second_combine(g,gp,gm,flag,h2[j])*sd[j]**2
        add(f"d2X{j+1}2_logodds_1","second_diagonal",sec)
    for j in range(d):
        for l in range(j+1,d):
            jp,jm,dj=_first_pair(x,j,h2[j]);kp,km,dk=_first_pair(x,l,h2[l])
            gg=[]
            for a,b in ((jp,kp),(jp,km),(jm,kp),(jm,km)):
                xx=x.copy();xx[:,j]=a[:,j];xx[:,l]=b[:,l]
                gl,_=_logit_gradient(backend,xx,theta,wrapper,batch_size)
                gg.append(gl)
            sec=(gg[0]-gg[1]-gg[2]+gg[3])*sd[j]*sd[l]/(dj*dk)[:,None]
            add(f"d2X{j+1}X{l+1}_logodds_1","second_mixed",sec)

    sig=np.stack(sigs,axis=1)
    matrices=np.einsum("sok,sol->okl",sig,sig,optimize=True)/n_subjects
    energies=np.linalg.norm(matrices,axis=(1,2))
    v1_size=1+d
    n1,m1,e1,f1=_normalize(names[:v1_size],families[:v1_size],
                            matrices[:v1_size],v1_min_relative_energy)
    n2,m2,e2,f2=_normalize(names,families,matrices,v2_min_relative_energy)
    return HistoricalBanks(fisher,eigen,n1,m1,e1,n2,m2,e2,f2,
                           tuple(names),matrices,energies,h1,h2)

def _jd(mats,R):
    if not len(mats):return 0.
    rotated=np.einsum("ik,mkl,jl->mij",R,mats,R,optimize=True)
    numerator=np.sum(rotated**2,axis=(1,2))-np.sum(np.diagonal(rotated,axis1=1,axis2=2)**2,axis=1)
    return float(np.mean(np.maximum(numerator,0.)))

LEGACY_METRIC_NAMES=(
    "historical_v1_jd","historical_v2_jd","historical_v1_level",
    "historical_v1_first","historical_v2_second_diagonal",
    "historical_v2_second_mixed","historical_fisher_offdiag",
    "historical_fisher_trace",
)

def historical_metrics(banks,R):
    # The last two are diagnostics for the unrotated full-space Fisher.
    full=R@banks.fisher@R.T
    out={
      "historical_v1_jd":_jd(banks.operators_v1,R),
      "historical_v2_jd":_jd(banks.operators_v2,R),
      "historical_fisher_offdiag":float(np.sum(full**2)-np.sum(np.diag(full)**2)),
      "historical_fisher_trace":float(np.trace(full)),
    }
    for label,ms,fs in (
        ("historical_v1_level",banks.operators_v1,["level" if n.startswith("level") else "other" for n in banks.names_v1]),
        ("historical_v1_first",banks.operators_v1,["first" if n.startswith("dX") else "other" for n in banks.names_v1]),
        ("historical_v2_second_diagonal",banks.operators_v2,banks.families_v2),
        ("historical_v2_second_mixed",banks.operators_v2,banks.families_v2)):
        family={"historical_v1_level":"level",
                "historical_v1_first":"first",
                "historical_v2_second_diagonal":"second_diagonal",
                "historical_v2_second_mixed":"second_mixed"}[label]
        mask=np.array([f==family for f in fs],dtype=bool)
        out[label]=_jd(ms[mask],R)
    return out

def save_historical_banks(path,b):
    np.savez_compressed(path,
        fisher=b.fisher,fisher_eigenvalues=b.fisher_eigenvalues,
        v1_names=np.asarray(b.names_v1,str),v1_matrices=b.operators_v1,
        v1_raw_energies=b.energies_v1,
        v2_names=np.asarray(b.names_v2,str),v2_matrices=b.operators_v2,
        v2_raw_energies=b.energies_v2,v2_families=np.asarray(b.families_v2,str),
        all_names=np.asarray(b.names_all,str),all_raw_matrices=b.raw_matrices,
        all_raw_energies=b.raw_energies,
        first_steps=b.steps_first,second_steps=b.steps_second)
