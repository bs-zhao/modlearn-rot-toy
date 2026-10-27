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

def sample_x(dx,n,seed,mode="sobol"):
    if mode=="sobol":
        return sobol_interior(dx,n,seed)
    if mode=="axis_sweeps":
        g=unit_grid(n)
        x=np.full((dx,n,dx),.5,dtype=float)
        for j in range(dx):x[j,:,j]=g
        return x.reshape(dx*n,dx)
    raise ValueError(f"Unknown X sampling mode {mode}")

def paired_grid(k,dx,n_theta,n_x,seed,x_mode="sobol"):
    theta=sobol_interior(k,n_theta,seed)
    x=sample_x(dx,n_x,seed+10007,mode=x_mode)
    n_x=len(x)
    xx=np.broadcast_to(x[None,:,:],(n_theta,n_x,dx)).reshape(-1,dx)
    tt=np.broadcast_to(theta[:,None,:],(n_theta,n_x,k)).reshape(-1,k)
    return xx,tt,theta,x
