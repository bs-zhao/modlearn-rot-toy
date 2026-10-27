"""Example: new toys require a single plugin module, without editing CLI."""
import torch
from ..models import register_model

@register_model("T7",2,1,"a*exp(b*x), dynamic plugin example",eta_bounds=(0,2.718281828459045))
def t7(x,theta):return theta[...,0]*torch.exp(theta[...,1]*x[...,0])
