"""Sampling physical X and theta on [0,1]^D without singular endpoints."""
import numpy as np
from scipy.stats import qmc

def sobol_interior(dim,n,seed,endpoint_eps=1e-4):
    if n<2:raise ValueError("n >= 2 required")
    arr=qmc.Sobol(dim,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    return endpoint_eps+(1-2*endpoint_eps)*arr

def unit_grid(n):
    if n<3:raise ValueError("n>=3")
    return (np.arange(n,dtype=float)+.5)/n

def paired_grid(k,dx,n_theta,n_x,seed):
    theta=sobol_interior(k,n_theta,seed)
    x=sobol_interior(dx,n_x,seed+10007)
    xx=np.broadcast_to(x[None,:,:],(n_theta,n_x,dx)).reshape(-1,dx)
    tt=np.broadcast_to(theta[:,None,:],(n_theta,n_x,k)).reshape(-1,k)
    return xx,tt,theta,x
