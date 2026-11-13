"""Train head or hybrid on a frozen final outer fold; requires CUDA.

Training-only weight selection is not repeated: the immutable original folds
provide each variant's own selected weight and train/test IDs.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from sam2.modeling.sam2_utils import MLP
from graph_step import make_graphs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--dataset',choices=['GVLM_v1_0','GVLM_v1_1','CAS','jz','bijie'],required=True)
    ap.add_argument('--model',choices=['head','hybrid'],required=True)
    ap.add_argument('--fold-index',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required by the frozen training protocol')
    if a.output.exists():raise FileExistsError(a.output)
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folds=json.loads((a.data/'splits'/f'{a.dataset}.json').read_text())[a.model]
    fold=folds[a.fold_index]
    source='GVLM_v1_0' if a.dataset=='GVLM_v1_1' else a.dataset
    with np.load(a.data/'features'/source/'cache.npz') as z:
        tr=np.arange(len(z['unit'])) if a.dataset=='GVLM_v1_1' else np.array(fold['train_ids'],np.int64)
        X=z['X'][tr].reshape(-1,z['X'].shape[-1]);y=z['Y'][tr].reshape(-1)
        candidates=z['Y'].shape[2];F=X.shape[-1]
        w=np.repeat(z['eqarea'][tr],2*candidates)
        weight=fold['weight']
        w=np.ones_like(w) if weight=='unweighted' else np.sqrt(w) if weight=='sqrt' else w
    tok=np.load(a.data/'frozen'/source/'tokens.npy',mmap_mode='r')
    T=np.asarray(tok[tr]).reshape(-1,256)
    if a.model=='head':
        model=MLP(256,256,4,3,sigmoid_output=True)
        model.load_state_dict(torch.load(a.data/'models/head_initial.pt',map_location='cpu',weights_only=True))
        inputs=T;pre={}
    else:
        X=np.where(np.isfinite(X),X,np.nan);med=np.zeros(F)
        for j in range(F):
            v=X[:,j];med[j]=float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.
        X=np.where(np.isfinite(X),X,med);mu=X.mean(0);sd=X.std(0);sd[sd==0]=1.
        inputs=np.concatenate([T,(X-mu)/sd],1).astype(np.float32)
        pre={'median':med,'mean':mu,'std':sd}
        model=nn.Sequential(nn.Linear(256+F,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,1))
    model=model.cuda().float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    xt=torch.tensor(inputs,dtype=torch.float32,device='cuda')
    yt=torch.tensor(y,dtype=torch.float32,device='cuda');wt=torch.tensor(w,dtype=torch.float32,device='cuda')
    channels=torch.tensor(np.tile(np.arange(candidates)%3+1,len(tr)*2),device='cuda')
    rng=torch.Generator(device='cpu').manual_seed(0)
    graphs=make_graphs(model,a.model,xt,yt,wt,channels,256)
    losses=[]
    for epoch in range(1,31):
        model.train();perm=torch.randperm(len(y),generator=rng).to('cuda')
        numerator=torch.zeros((),dtype=torch.float64,device='cuda');denominator=numerator.clone()
        for offset in range(0,len(y),256):
            ids=perm[offset:offset+256];optimizer.zero_grad(set_to_none=True)
            loss,num,den=graphs[len(ids)](xt[ids],yt[ids],wt[ids],channels[ids])
            loss.backward();optimizer.step();numerator+=num.detach().double();denominator+=den.detach().double()
        losses.append(dict(epoch=epoch,weighted_mse=float(numerator)/float(denominator)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
        preprocessing={k:v.tolist() for k,v in pre.items()},kind=a.model),a.output)
    a.output.with_suffix('.json').write_text(json.dumps(dict(dataset=a.dataset,source=source,model=a.model,
        seed=0,weight=weight,training_ids=tr.tolist(),epochs=losses,selection='original training-only selected weight; final epoch'),indent=2)+'\n')

if __name__=='__main__':main()
