"""Independent test-bed generators; native X and theta in [0,1]."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class ToyModel:
    name: str
    k: int
    dx: int
    description: str
    identifiable: bool

MODELS = {
    "T1": ToyModel("T1", 2, 1, "a*x**b (amplitude / exponent)", True),
    "T2": ToyModel("T2", 3, 2, "a*x1+b*x2+c (linear)", True),
    "T3": ToyModel("T3", 2, 2, "a*x1+a*b*x2 (parameter interaction)", True),
    "T4": ToyModel("T4", 4, 2, "a*x1**b+c*d*x2**b (nonidentifiable)", False),
    "T4ID": ToyModel("T4ID", 4, 3, "T4+c*x3 (identifiable control)", True),
    "T5": ToyModel("T5", 4, 2, "a*x1+b*x1**2+c*x2+d (C2-like)", True),
    "T6": ToyModel("T6", 4, 2, "a*x1+b*x2+c*x1*x2+d (C3-like)", True),
}
WRAPPERS = ("identity", "sigmoid", "square")

def generator(name, x, theta):
    if name not in MODELS: raise ValueError(f"Unknown model {name}")
    a,b = theta[...,0],theta[...,1]
    x1=x[...,0]
    if name=="T1": return a*x1.pow(b)
    x2=x[...,1]
    if name=="T2": return a*x1+b*x2+theta[...,2]
    if name=="T3": return a*x1+a*b*x2
    c,d=theta[...,2],theta[...,3]
    if name in ("T4","T4ID"):
        y=a*x1.pow(b)+c*d*x2.pow(b)
        return y+c*x[...,2] if name=="T4ID" else y
    if name=="T5": return a*x1+b*x1.square()+c*x2+d
    if name=="T6": return a*x1+b*x2+c*x1*x2+d
    raise AssertionError(name)

def wrapper(y,name):
    if name=="identity": return y
    if name=="sigmoid": return torch.sigmoid(y)
    if name=="square": return y.square()
    raise ValueError(name)

def evaluate(name,link,x,theta,analysis_scale="raw"):
    y=wrapper(generator(name,x,theta),link)
    if analysis_scale=="raw": return y
    if analysis_scale=="log": return torch.log(y.clamp_min(1e-20))
    if analysis_scale=="inverse-link":
        if link!="sigmoid": raise ValueError("Only sigmoid admits inverse-link")
        return torch.logit(y.clamp(min=1e-12,max=1-1e-12))
    raise ValueError(analysis_scale)

def plane_rotation(k,i,j,angle_deg):
    if not 0<=i<j<k: raise ValueError("0 <= i < j < K required")
    r=np.eye(k)
    c,s=np.cos(np.deg2rad(angle_deg)),np.sin(np.deg2rad(angle_deg))
    r[i,i]=r[j,j]=c;r[i,j]=-s;r[j,i]=s
    return r
