"""Exact original C2 generator from modlearn-rot-2.

Original Bernoulli choice probability: p = sigmoid(eta). For the toy
wrapper interface, this plugin returns eta and --wrappers sigmoid yields p.
Raw stimulus order: (X1_A, X1_B, X2_A, X2_B).
Do not replace raw X with the hand-crafted design matrix.
"""
import math
from ..models import register_model

@register_model("C2",4,4,"Original C2 binary choice: p=sigmoid(b1+linear X1+quadratic X1+linear X2)")
def c2_logit(x,theta):
    x1a,x1b,x2a,x2b=(x[...,j] for j in range(4))
    b1,b2,b3,b4=(theta[...,j] for j in range(4))
    return (b1 + math.sqrt(3)*b2*(x1a-x1b)
            + (math.sqrt(5)/2)*b3*((3*x1a.square()-1)-(3*x1b.square()-1))
            + math.sqrt(3)*b4*(x2a-x2b))
