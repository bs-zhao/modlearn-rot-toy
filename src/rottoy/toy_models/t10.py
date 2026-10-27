"""T10: amplitude, exponent, amplitude ratio, and additive offset.

eta(x,theta) = a*x1**b + a*c*x2**b + d
theta = (a,b,c,d); X = (x1,x2), each coordinate in [0,1].
"""
from ..models import register_model

@register_model(
    "T10", k=4, dx=2,
    description="a*x1**b + a*c*x2**b + d",
    eta_bounds=(0, 3),
)
def t10(x, theta):
    a, b, c, d = (theta[..., j] for j in range(4))
    x1, x2 = x[..., 0], x[..., 1]
    return a * x1.pow(b) + a * c * x2.pow(b) + d
