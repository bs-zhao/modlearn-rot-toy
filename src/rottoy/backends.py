"""Function backends share predict(x,theta,scale) and all differentiable metrics.

This first surrogate is a smooth MLP. Future MP/NP implementations can provide
the same interface without modifying derivative, sweep or plotting libraries.
"""
from pathlib import Path
import json
import torch
from torch import nn
from .models import evaluate,MODELS,transform_output
from .sampling import sobol_interior

class AnalyticBackend:
    kind="eta"
    def __init__(self,model,wrapper):
        self.model,self.wrapper=model,wrapper
    def predict(self,x,theta,scale="raw"):
        return evaluate(self.model,self.wrapper,x,theta,scale)

class MLP(nn.Module):
    def __init__(self,n_in,width=64,depth=3):
        super().__init__()
        modules=[];dim=n_in
        for _ in range(depth):
            modules.extend([nn.Linear(dim,width),nn.Tanh()])
            dim=width
        modules.append(nn.Linear(dim,1))
        self.net=nn.Sequential(*modules)
    def forward(self,x,theta):
        return self.net(torch.cat((x,theta),dim=-1)).squeeze(-1)

class NeuralBackend:
    kind="F"
    def __init__(self,model,wrapper,network):
        self.model,self.wrapper=model,wrapper
        self.network=network.eval().to(dtype=torch.float64)
        for p in self.network.parameters():p.requires_grad_(False)
    def predict(self,x,theta,scale="raw"):
        return transform_output(self.network(x,theta),self.wrapper,scale)

def training_samples(model,wrapper,n,seed):
    m=MODELS[model]
    arr=sobol_interior(m.k+m.dx,n,seed)
    x=torch.tensor(arr[:,:m.dx],dtype=torch.float32)
    theta=torch.tensor(arr[:,m.dx:],dtype=torch.float32)
    with torch.no_grad():y=evaluate(model,wrapper,x,theta,"raw")
    return x,theta,y

def checkpoint_path(root,model,wrapper):
    return Path(root)/"checkpoints"/model/f"{wrapper}_mlp.pt"

def load_neural(path,model,wrapper):
    obj=torch.load(path,map_location="cpu",weights_only=True)
    info=MODELS[model]
    if obj["model"]!=model or obj["wrapper"]!=wrapper or obj["k"]!=info.k or obj["dx"]!=info.dx:
        raise ValueError("MLP checkpoint model, wrapper or dimensions mismatch")
    network=MLP(info.k+info.dx,obj["width"],obj["depth"])
    network.load_state_dict(obj["network"])
    return NeuralBackend(model,wrapper,network)

def train_neural(root,model,wrapper,*,ntrain=4096,nval=1024,epochs=150,
                 width=64,depth=3,lr=.002,batch_size=512,seed=20261026):
    torch.manual_seed(seed)
    x,t,y=training_samples(model,wrapper,ntrain,seed)
    xv,tv,yv=training_samples(model,wrapper,nval,seed+10917)
    info=MODELS[model]
    net=MLP(info.k+info.dx,width,depth)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    best=float("inf");epoch_best=0;state=None
    for epoch in range(epochs):
        net.train()
        for idx in torch.randperm(len(x)).split(batch_size):
            pred=net(x[idx],t[idx])
            loss=(pred-y[idx]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        net.eval()
        with torch.no_grad():value=float((net(xv,tv)-yv).square().mean())
        if value<best:
            best=value;epoch_best=epoch+1
            state={k:v.detach().clone() for k,v in net.state_dict().items()}
    path=checkpoint_path(root,model,wrapper)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model,wrapper=wrapper,k=info.k,dx=info.dx,width=width,depth=depth,network=state),path)
    metadata=dict(model=model,wrapper=wrapper,network="MLP",train_samples=ntrain,
        val_samples=nval,epochs=epochs,width=width,depth=depth,lr=lr,
        seed=seed,best_validation_mse=best,best_epoch=epoch_best,
        target="(theta,X) -> wrapped Y, raw output",checkpoint=str(path))
    path.with_suffix(".json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return load_neural(path,model,wrapper),metadata
