"""Toy T9: original C2 generator from modlearn-rot-2.

Original Bernoulli choice probability: p = sigmoid(eta). For the toy
wrapper interface, this plugin returns eta and --wrappers sigmoid yields p.
Raw stimulus order: (X1_A, X1_B, X2_A, X2_B).
Generic toy model; no special choice-specific backend or metrics.
"""
import math

_A=2*math.sqrt(3)+1.5*math.sqrt(5)
from ..models import register_model

@register_model("T9",4,4,"C2 equation: b1 + sqrt(3)b2*DeltaX1 + (sqrt(5)/2)b3*DeltaP2(X1) + sqrt(3)b4*DeltaX2",eta_bounds=(-_A,1+_A))
def t9(x,theta):
    x1a,x1b,x2a,x2b=(x[...,j] for j in range(4))
    b1,b2,b3,b4=(theta[...,j] for j in range(4))
    return (b1 + math.sqrt(3)*b2*(x1a-x1b)
            + (math.sqrt(5)/2)*b3*((3*x1a.square()-1)-(3*x1b.square()-1))
            + math.sqrt(3)*b4*(x2a-x2b))
