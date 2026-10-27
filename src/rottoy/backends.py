"""Interchangeable analytic eta and differentiable approximate F backends.

The neural surrogate learns continuous (theta, X) -> Y using MSE.
It never uses generating parameter identities inside the metric optimizer.
A future MP/NP can implement exactly predict(x,theta,scale).
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .models import evaluate,get_variant
from .transforms import output_domain_transform,analysis_transform
from .sampling import sobol_interior

class AnalyticBackend:
    kind="eta"
    def __init__(self,variant):
        self.variant=get_variant(variant) if isinstance(variant,str) else variant
    def predict(self,x,theta,scale="raw"):
        return evaluate(self.variant,x,theta,scale)

class MLP(nn.Module):
    def __init__(self,n_inputs,width=64,depth=3):
        super().__init__()
        layers=[];dim=n_inputs
        for _ in range(depth):
            layers.extend([nn.Linear(dim,width),nn.Tanh()])
            dim=width
        layers.append(nn.Linear(dim,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x,theta):
        return self.net(torch.cat((x,theta),dim=-1)).squeeze(-1)

class NeuralBackend:
    kind="F"
    def __init__(self,variant,network):
        self.variant=get_variant(variant) if isinstance(variant,str) else variant
        self.network=network.eval().to(dtype=torch.float64)
        for p in self.network.parameters():p.requires_grad_(False)
    def predict(self,x,theta,scale="raw"):
        y=output_domain_transform(self.network(x,theta),self.variant.transform)
        return analysis_transform(y,self.variant.transform,scale)

def checkpoint_path(root,variant):
    v=get_variant(variant) if isinstance(variant,str) else variant
    return Path(root)/"checkpoints"/v.model.name/v.transform/"mlp.pt"

def load_neural(root,variant):
    v=get_variant(variant) if isinstance(variant,str) else variant
    path=checkpoint_path(root,v)
    obj=torch.load(path,map_location="cpu",weights_only=True)
    if (obj["variant"]!=v.id or obj["k"]!=v.k or obj["dx"]!=v.dx):
        raise ValueError("Checkpoint incompatible with requested toy variant")
    net=MLP(v.k+v.dx,obj["width"],obj["depth"])
    net.load_state_dict(obj["network"])
    return NeuralBackend(v,net)

def train_neural(root,variant,*,n_train=4096,n_val=1024,epochs=120,
                 width=64,depth=3,lr=.002,batch_size=512,seed=20261026):
    v=get_variant(variant) if isinstance(variant,str) else variant
    torch.manual_seed(seed)
    def sample(n,number):
        arr=sobol_interior(v.dx+v.k,n,number)
        x=torch.tensor(arr[:,:v.dx],dtype=torch.float32)
        theta=torch.tensor(arr[:,v.dx:],dtype=torch.float32)
        with torch.no_grad():y=evaluate(v,x,theta,"raw")
        return x,theta,y
    x,t,y=sample(n_train,seed);vx,vt,vy=sample(n_val,seed+217)
    net=MLP(v.dx+v.k,width,depth)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    best=float("inf");best_epoch=0;state=None
    for epoch in range(1,epochs+1):
        net.train()
        for idx in torch.randperm(len(x)).split(batch_size):
            pred=output_domain_transform(net(x[idx],t[idx]),v.transform)
            loss=(pred-y[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            prediction=output_domain_transform(net(vx,vt),v.transform)
            val=float((prediction-vy).square().mean())
        if val<best:
            best=val;best_epoch=epoch
            state={k:a.detach().clone() for k,a in net.state_dict().items()}
    path=checkpoint_path(root,v)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(variant=v.id,k=v.k,dx=v.dx,width=width,depth=depth,network=state),path)
    stats=dict(variant=v.id,architecture="continuous-MLP",objective="MSE(X,theta -> Y)",
               n_train=n_train,n_val=n_val,epochs=epochs,best_epoch=best_epoch,
               best_validation_mse=best,seed=seed,checkpoint=str(path))
    path.with_suffix(".json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    return load_neural(root,v),stats
