"""Original testbed generators in the extensible plugin registry."""
from ..models import register_model

@register_model("T1",2,1,"a*x**b (amplitude / exponent)")
def t1(x,t):return t[...,0]*x[...,0].pow(t[...,1])

@register_model("T2",3,2,"a*x1+b*x2+c (linear)")
def t2(x,t):return t[...,0]*x[...,0]+t[...,1]*x[...,1]+t[...,2]

@register_model("T3",2,2,"a*x1+a*b*x2 (parameter interaction)")
def t3(x,t):return t[...,0]*x[...,0]+t[...,0]*t[...,1]*x[...,1]

@register_model("T4",4,2,"a*x1**b+c*d*x2**b (nonidentifiable)",identifiable=False)
def t4(x,t):return t[...,0]*x[...,0].pow(t[...,1])+t[...,2]*t[...,3]*x[...,1].pow(t[...,1])

@register_model("T4ID",4,3,"T4+c*x3 (identifiable control)")
def t4id(x,t):return t4(x,t)+t[...,2]*x[...,2]

@register_model("T5",4,2,"a*x1+b*x1**2+c*x2+d (C2-like)")
def t5(x,t):return t[...,0]*x[...,0]+t[...,1]*x[...,0].square()+t[...,2]*x[...,1]+t[...,3]

@register_model("T6",4,2,"a*x1+b*x2+c*x1*x2+d (C3-like)")
def t6(x,t):return t[...,0]*x[...,0]+t[...,1]*x[...,1]+t[...,2]*x[...,0]*x[...,1]+t[...,3]
