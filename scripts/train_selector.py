"""Refit one final GBT using frozen training IDs and its training-only chosen weight."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import joblib
from landslide_sam.selector import train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dataset',required=True,choices=['GVLM_v1_0','GVLM_v1_1','CAS','jz','bijie'])
    p.add_argument('--fold-index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    fold=json.loads((a.data/'splits'/f'{a.dataset}.json').read_text())['gbt'][a.fold_index]
    source='GVLM_v1_0' if a.dataset=='GVLM_v1_1' else a.dataset
    with np.load(a.data/'features'/source/'cache.npz') as z:
        tr=np.arange(len(z['unit'])) if a.dataset=='GVLM_v1_1' else np.array(fold['train_ids'])
        model=train(z['X'][tr],z['Y'][tr],z['eqarea'][tr],fold['weight'])
    a.output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,a.output)

if __name__=='__main__':main()
