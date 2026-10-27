"""Continuous-valued output transforms shared by every T model.

The output is always a real-valued deterministic Y. A sigmoid variant is NOT
a Bernoulli random outcome. Inverses exist only for one-to-one transforms.
"""
from dataclasses import dataclass
from collections import OrderedDict
import torch
import torch.nn.functional as nnf

@dataclass(frozen=True)
class Transform:
    code: str
    name: str
    math: str
    invertible: bool
    note: str

TRANSFORMS=OrderedDict((
    ("id",Transform("id","identity",r"g(t)=t",True,"Untransformed continuous response")),
    ("sig",Transform("sig","sigmoid",r"g(t)=1/(1+e^{-t})",True,"Bounded (0,1), not Bernoulli")),
    ("sq",Transform("sq","square",r"g(t)=t^2",False,"Global sign is lost")),
    ("tanh",Transform("tanh","tanh",r"g(t)=tanh(t)",True,"Bounded (-1,1)")),
    ("sp",Transform("sp","softplus",r"g(t)=log(1+e^t)",True,"Positive smooth response")),
    ("exp",Transform("exp","exponential",r"g(t)=e^t",True,"Positive; can amplify range")),
))

def apply_transform(t,code):
    if code=="id":return t
    if code=="sig":return torch.sigmoid(t)
    if code=="sq":return t.square()
    if code=="tanh":return torch.tanh(t)
    if code=="sp":return nnf.softplus(t)
    if code=="exp":return torch.exp(t)
    raise KeyError(f"Unknown transform: {code}")

def inverse_transform(y,code):
    """Inverse when mathematical mapping is injective; stabilized numerically."""
    if code=="id":return y
    if code=="sig":return torch.logit(y.clamp(1e-12,1-1e-12))
    if code=="tanh":
        z=y.clamp(-1+1e-12,1-1e-12)
        return torch.atanh(z)
    if code=="sp":
        z=y.clamp_min(1e-12)
        return z+torch.log(-torch.expm1(-z))
    if code=="exp":return torch.log(y.clamp_min(1e-20))
    raise ValueError("Square transform is not globally invertible")

def analysis_transform(y,code,scale="raw"):
    if scale=="raw":return y
    if scale=="log":
        return torch.log(y.clamp_min(1e-20))
    if scale=="inverse":
        return inverse_transform(y,code)
    raise ValueError(f"Unknown analysis scale: {scale}")

def output_domain_transform(v,code):
    """Network output head; enforces the response codomain without using eta."""
    if code=="id":return v
    if code=="sig":return torch.sigmoid(v)
    if code=="sq":return nnf.softplus(v)
    if code=="tanh":return torch.tanh(v)
    if code=="sp":return nnf.softplus(v)
    if code=="exp":return torch.exp(v.clamp(max=20))
    raise KeyError(code)

def transformed_bounds(bounds,code):
    """Exact continuous-output interval from the registered eta interval."""
    from math import exp,log,tanh,log1p
    low,high=map(float,bounds)
    if low>high:raise ValueError("Bad eta bounds")
    def sigmoid(x):
        return 1/(1+exp(-x)) if x>=0 else exp(x)/(1+exp(x))
    def softplus(x):return max(x,0)+log1p(exp(-abs(x)))
    if code=="id":return low,high
    if code=="sig":return sigmoid(low),sigmoid(high)
    if code=="tanh":return tanh(low),tanh(high)
    if code=="sp":return softplus(low),softplus(high)
    if code=="exp":return exp(low),exp(high)
    if code=="sq":
        return (0 if low<=0<=high else min(low**2,high**2),
                max(low**2,high**2))
    raise KeyError(code)
