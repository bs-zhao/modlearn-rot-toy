"""Only base toy generators T1,T2,... are registered here.

A variant T9-sig combines a base generator and one continuous output transform.
This module is also the sole parser of model variant identifiers.
"""
from dataclasses import dataclass
import importlib,pkgutil,re
import numpy as np
from .transforms import TRANSFORMS,apply_transform,analysis_transform

@dataclass(frozen=True)
class ToyModel:
    name: str
    k: int
    dx: int
    description: str
    identifiable: bool
    eta_bounds: tuple

@dataclass(frozen=True)
class ToyVariant:
    model: ToyModel
    transform: str
    @property
    def id(self):return f"{self.model.name}-{self.transform}"
    @property
    def k(self):return self.model.k
    @property
    def dx(self):return self.model.dx
    @property
    def y_bounds(self):
        from .transforms import transformed_bounds
        return transformed_bounds(self.model.eta_bounds,self.transform)

_GENERATORS={}
MODELS={}

def register_model(name,k,dx,description,identifiable=True,eta_bounds=None):
    def decorator(fn):
        if not re.fullmatch(r"T[1-9]\d*",name):raise ValueError("Toy name must be T1,T2,...")
        if name in MODELS:raise ValueError(f"Duplicate toy {name}")
        if k<2 or dx<1:raise ValueError("Expected >=2 parameters and >=1 X dimension")
        if eta_bounds is None or len(eta_bounds)!=2:
            raise ValueError(f"{name} must specify analytic eta_bounds=(lower,upper)")
        low,high=map(float,eta_bounds)
        if not np.isfinite(low) or not np.isfinite(high) or low>high:
            raise ValueError(f"Invalid bounds for {name}: {eta_bounds}")
        MODELS[name]=ToyModel(name,k,dx,description,identifiable,(low,high))
        _GENERATORS[name]=fn
        return fn
    return decorator

def discover_models():
    from . import toy_models
    for module in pkgutil.iter_modules(toy_models.__path__,toy_models.__name__+"."):
        if not module.name.rsplit(".",1)[-1].startswith("_"):
            importlib.import_module(module.name)

def get_variant(identifier):
    # Bare base names are an explicit alias for identity, for easy migration.
    parts=identifier.split("-")
    if len(parts)==1:parts.append("id")
    if len(parts)!=2:raise ValueError("Expected Tn-suffix, e.g. T9-sig")
    name,suffix=parts
    if name not in MODELS:raise ValueError(f"Unknown toy: {name}. Use --list-models")
    if suffix not in TRANSFORMS:raise ValueError(f"Unknown suffix: {suffix}; valid: {list(TRANSFORMS)}")
    return ToyVariant(MODELS[name],suffix)

def variant_names():
    return [f"{model}-{suffix}" for model in MODELS for suffix in TRANSFORMS]

def generator(name,x,theta):
    m=MODELS[name]
    if x.shape[-1]!=m.dx or theta.shape[-1]!=m.k:
        raise ValueError(f"{name} requires X dim {m.dx} and theta dim {m.k}")
    return _GENERATORS[name](x,theta)

def evaluate(variant,x,theta,scale="raw"):
    v=get_variant(variant) if isinstance(variant,str) else variant
    y=apply_transform(generator(v.model.name,x,theta),v.transform)
    return analysis_transform(y,v.transform,scale)

def plane_rotation(k,i,j,angle_deg):
    if not 0<=i<j<k:raise ValueError("Need 0 <= i < j < K")
    r=np.eye(k)
    c,s=np.cos(np.deg2rad(angle_deg)),np.sin(np.deg2rad(angle_deg))
    r[i,i]=r[j,j]=c
    r[i,j]=-s;r[j,i]=s
    return r

discover_models()
MODELS=dict(sorted(MODELS.items(),key=lambda item:int(item[0][1:])))
