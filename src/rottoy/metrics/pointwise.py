"""M001–M011: pointwise and integrated derivative metrics."""
import numpy as np
from ..operators import rotated_derivatives

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

