"""Check packaged GBT parameters and candidate choices against frozen predictions."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from landslide_sam.selector import parameters

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    records=[]
    index=list(csv.DictReader((a.data/'models/MODELS.csv').open()))
    for r in index:
        if r['model']!='gbt':continue
        d=r['dataset'];i=int(r['fold_index0'])
        fold=json.loads((a.data/'splits'/f'{d}.json').read_text())['gbt'][i]
        with np.load(a.data/'features'/d/'cache.npz') as z:
            ids=np.arange(len(z['unit'])) if d=='GVLM_v1_1' else np.array(fold['test_ids'])
            X=z['X'][ids];shape=X.shape[:-1]
        model=joblib.load(a.data/r['model_file'])
        if model.get_params(deep=False)!=parameters():raise RuntimeError('STOP: model parameters mismatch '+r['model_file'])
        with threadpool_limits(4):got=model.predict(X.reshape(-1,X.shape[-1])).reshape(shape)
        with np.load(a.data/'frozen'/d/'predictions.npz') as z:expected=z['gbt'][ids]
        mismatches=int(np.count_nonzero(got.argmax(2)!=expected.argmax(2)))
        records.append(dict(dataset=d,fold_index0=i,test_objects=len(ids),
            candidate_choice_mismatches=mismatches,max_absolute_score_difference=float(np.max(np.abs(got-expected))),
            source_model=r['model_file'],source_prediction=f'frozen/{d}/predictions.npz:gbt; outer test IDs',
            status='PASS' if mismatches==0 else 'FAIL'))
        if mismatches:raise RuntimeError('STOP: candidate choices mismatch '+r['model_file'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(f'PASS: {len(records)} GBT models; zero frozen candidate-choice mismatches')

if __name__=='__main__':main()
