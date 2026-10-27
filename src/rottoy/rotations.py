"""Rotation families for any K-dimensional continuous toy.

SO(K) has K*(K-1)/2 independent Givens plane angles. Pair scans are
coordinate slices, whereas joint scans compose ALL planes.

The full search uses a fixed ordered Givens product: matrices always remain
orthogonal (determinant +1), so optimizer variables are angles, not entries.
"""
from itertools import combinations
import numpy as np

def planes(k):
    if k<2:raise ValueError("At least two latent coordinates required")
    return tuple(combinations(range(k),2))

def givens(k,i,j,degrees):
    if not 0<=i<j<k:raise ValueError("Invalid Givens plane")
    a=np.deg2rad(float(degrees));c=np.cos(a);s=np.sin(a)
    R=np.eye(k)
    R[i,i]=R[j,j]=c
    R[i,j]=-s;R[j,i]=s
    return R

def joint_rotation(k,angles_degrees):
    """R=G_last ... G_second G_first, in lexicographic plane order."""
    p=planes(k)
    angles=np.asarray(angles_degrees,dtype=float)
    if angles.shape!=(len(p),):
        raise ValueError(f"K={k} requires {len(p)} angles, got {angles.shape}")
    if not np.isfinite(angles).all():raise ValueError("Angles must be finite")
    R=np.eye(k)
    for (i,j),angle in zip(p,angles):
        R=givens(k,i,j,angle)@R
    return R

def path_weights(k,pattern="balanced",seed=20261027):
    """Nonzero coefficients in each plane, so one phi varies all planes.

    'balanced': each Givens angle is phi.
    'alternating': successive Givens angles change sign.
    'ramped': coefficients 1/num_planes,...,1, avoiding coincident symmetries.
    'random': fixed reproducible unit-magnitude coefficients (seeded).
    """
    m=len(planes(k))
    if pattern=="balanced":w=np.ones(m)
    elif pattern=="alternating":w=(-1.)**np.arange(m)
    elif pattern=="ramped":w=np.arange(1,m+1,dtype=float)/m
    elif pattern=="random":
        rng=np.random.default_rng(seed)
        w=rng.uniform(.25,1.,size=m)*rng.choice([-1.,1.],size=m)
    else:raise ValueError("Unknown joint path: "+pattern)
    return w

def joint_path(k,phi_degrees,pattern="balanced",seed=20261027):
    weights=path_weights(k,pattern,seed)
    return joint_rotation(k,weights*float(phi_degrees))

def signed_permutation_distance(R):
    """Frobenius distance |R| to closest permutation (zero at true-axis equivalents)."""
    from scipy.optimize import linear_sum_assignment
    R=np.asarray(R,dtype=float)
    if R.ndim!=2 or R.shape[0]!=R.shape[1]:raise ValueError("Square rotation required")
    row,col=linear_sum_assignment(-np.abs(R))
    P=np.zeros_like(R);P[row,col]=1.
    return float(np.linalg.norm(np.abs(R)-P))

def orthogonality_error(R):
    return float(np.linalg.norm(R.T@R-np.eye(len(R))))
