"""Example: new toys require a single plugin module, without editing CLI."""
import torch
from ..models import register_model

@register_model("T7",2,1,"a*exp(b*x), dynamic plugin example")
def t7(x,theta):return theta[...,0]*torch.exp(theta[...,1]*x[...,0])
