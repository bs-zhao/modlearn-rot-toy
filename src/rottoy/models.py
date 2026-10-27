"""Dynamic plugin-based toy registry; every theta and X coordinate is in [0,1]."""
from dataclasses import dataclass
import importlib
import pkgutil
import numpy as np
import torch

@dataclass(frozen=True)
class ToyModel:
    name: str
    k: int
    dx: int
    description: str
    identifiable: bool

MODELS = {}
_GENERATORS = {}
WRAPPERS = ("identity", "sigmoid", "square")

def register_model(name,k,dx,description,identifiable=True):
    def deco(fn):
        if not name.isidentifier() or name in MODELS or k<2 or dx<1:
            raise ValueError(f"Invalid or duplicate model name/dimensions: {name}")
        MODELS[name]=ToyModel(name,k,dx,description,identifiable)
        _GENERATORS[name]=fn
        return fn
    return deco

def discover_models():
    from . import toy_models
    for item in pkgutil.iter_modules(toy_models.__path__,toy_models.__name__+"."):
        if not item.name.rsplit(".",1)[-1].startswith("_"):
            importlib.import_module(item.name)

def generator(name,x,theta):
    if name not in MODELS:raise ValueError(f"Unknown model {name}. Available: {list(MODELS)}")
    if x.shape[-1]!=MODELS[name].dx or theta.shape[-1]!=MODELS[name].k:
        raise ValueError("Incorrect X or theta dimensionality")
    return _GENERATORS[name](x,theta)

def wrapper(y,name):
    if name=="identity": return y
    if name=="sigmoid": return torch.sigmoid(y)
    if name=="square": return y.square()
    raise ValueError(name)

def transform_output(y,link,analysis_scale="raw"):
    if analysis_scale=="raw":return y
    if analysis_scale=="log":return torch.log(y.clamp_min(1e-20))
    if analysis_scale=="inverse-link":
        if link!="sigmoid":raise ValueError("Only sigmoid admits inverse-link")
        return torch.logit(y.clamp(min=1e-12,max=1-1e-12))
    raise ValueError(analysis_scale)

def evaluate(name,link,x,theta,analysis_scale="raw"):
    return transform_output(wrapper(generator(name,x,theta),link),link,analysis_scale)

def plane_rotation(k,i,j,angle_deg):
    if not 0<=i<j<k:raise ValueError("0 <= i < j < K required")
    r=np.eye(k)
    c,s=np.cos(np.deg2rad(angle_deg)),np.sin(np.deg2rad(angle_deg))
    r[i,i]=r[j,j]=c
    r[i,j]=-s
    r[j,i]=s
    return r

discover_models()
