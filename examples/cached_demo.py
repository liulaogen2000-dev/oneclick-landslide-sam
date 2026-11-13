"""CPU example for a packaged frozen click; recomputes features from source RGB.

Candidate masks/stability scores are frozen SAM outputs. This is not live SAM
inference and accepts only the packaged click. Use delineate.py for new clicks.
"""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from landslide_sam.candidates import Candidate
from landslide_sam.features import extract_features
from landslide_sam.selector import load,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--dataset',choices=['GVLM_v1_0','CAS'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rec=next(r for r in json.loads((a.data/'examples/INDEX.json').read_text()) if r['dataset']==a.dataset)
    rgb=np.asarray(Image.open(a.datasets/rec['image_relative_path']).convert('RGB'))
    gsd=(rec['gsd_x_m'],rec['gsd_y_m'])
    c=[]
    with np.load(a.data/rec['candidate_file']) as q:
        expected=q['features'];point=q['point_yx'];shape=tuple(q['image_shape'])
        expected_scores=q['frozen_gbt_scores'].copy()
        if rgb.shape[:2]!=shape:raise RuntimeError('STOP: source image geometry mismatch')
        for i in range(len(expected)):
            win=tuple(map(int,q['windows'][i//3]));h,w=win[2:]
            mask=np.unpackbits(q[f'mask_{i:02d}'])[:h*w].reshape(h,w).astype(bool)
            nominal=[512,1024,2048,4096][i//3] if a.dataset=='GVLM_v1_0' else 512*gsd[0]/.59
            c.append(Candidate(mask,win,expected[i,0],expected[i,1],np.empty((0,0)),np.empty(0),nominal))
    features=extract_features(rgb,point,c,a.dataset,gsd)
    if not np.array_equal(np.isnan(features),np.isnan(expected)):
        raise RuntimeError('STOP: feature missingness mismatch')
    feature_difference=float(np.nanmax(np.abs(features-expected)))
    if not np.array_equal(features,expected,equal_nan=True):
        raise RuntimeError(f'STOP: frozen features differ; max absolute difference={feature_difference}')
    selected,confidence,scores=predict(load(a.data/rec['model_file']),features)
    if selected!=int(expected_scores.argmax()):raise RuntimeError('STOP: frozen candidate selection mismatch')
    chosen=c[selected];y,x,h,w=chosen.window
    full=np.zeros(shape,bool);full[y:y+h,x:x+w]=chosen.mask
    a.output.mkdir(parents=True)
    Image.fromarray(full.astype('uint8')*255).save(a.output/'mask.png')
    result=dict(dataset=a.dataset,click_xy=[int(point[1]),int(point[0])],selected_candidate=selected,
                confidence=confidence,candidate_count=len(c),feature_count=features.shape[1],
                max_absolute_feature_difference=feature_difference,status='PASS',
                SAM_forward='NOT_RUN: frozen candidates used',image=rec['image_relative_path'])
    (a.output/'confidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
