"""CUDA-graph forward/backward only; original AdamW updates stay outside graphs."""
import torch
from torch import nn
class WeightedLoss(nn.Module):
 def __init__(self,net,kind):super().__init__();self.net=net;self.kind=kind
 def forward(self,x,y,w,channels):
  out=self.net(x);pred=(out*channels).sum(1) if self.kind=='head' else out[:,0];num=(w*(pred-y).square()).sum();den=w.sum();return num/den,num.detach(),den.detach()
def make_graphs(net,kind,x,y,w,channels,batch_size):
 graphs={}
 for B in sorted({min(batch_size,len(y)),len(y)%batch_size or batch_size}):
  wrapper=WeightedLoss(net,kind).train();ch=torch.nn.functional.one_hot(channels[:B],num_classes=4).float() if kind=='head' else channels[:B].clone();sample=(x[:B].clone(),y[:B].clone(),w[:B].clone(),ch)
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):graphed=torch.cuda.make_graphed_callables(wrapper,sample)
  def call(x,y,w,c,g=graphed):return g(x,y,w,torch.nn.functional.one_hot(c,num_classes=4).float() if kind=='head' else c)
  graphs[B]=call
 return graphs
