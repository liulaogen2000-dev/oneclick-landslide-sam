"""Read public source rasters and check reference counts against frozen object IDs.

No query list is reselected and no source raster is written or redistributed.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
Image.MAX_IMAGE_PIXELS=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    index=list(csv.DictReader((a.data/'DATASET_SOURCES.csv').open()))
    records=[]
    for d in ['GVLM_v1_0','GVLM_v1_1','CAS','jz','bijie']:
        with np.load(a.data/'frozen'/d/'evaluation_arrays.npz') as e:
            units=e['units'];counts={str(u):(int(g),int(n)) for u,g,n in zip(units,e['GG'],e['OC'])
                                  if not (d=='GVLM_v1_1' and str(u)=='Saint_Sebastian_Brazil')}
        with np.load(a.data/'features'/d/'cache.npz') as z:
            objects={u:[(int(oid),int(area)) for unit,oid,area in zip(z['unit'],z['obj'],z['garea']) if unit==u]
                     for u in counts}
        for r in index:
            if r['dataset']!=d or r['unit'] not in counts:continue
            image_path=a.datasets/r['image_relative_path'];label_path=a.datasets/r['label_relative_path']
            with Image.open(image_path) as im:imshape=(im.height,im.width)
            with Image.open(label_path) as im:
                if label_path.suffix.lower() in ['.tif','.tiff']:
                    raw=np.asarray(im)
                    if raw.ndim==3:raw=raw[...,0]
                else:raw=np.asarray(im.convert('L'))
                reference=raw>0
            if reference.shape!=imshape:raise RuntimeError('STOP: source image/label shape mismatch '+d+'/'+r['unit'])
            labels,_=ndimage.label(reference,np.ones((3,3),bool))
            sizes=np.bincount(labels.ravel())
            gg,n=counts[r['unit']]
            matched=int(reference.sum())==gg and len(objects[r['unit']])==n
            matched=matched and all(oid<len(sizes) and int(sizes[oid])==area for oid,area in objects[r['unit']])
            records.append(dict(dataset=d,unit=r['unit'],reference_foreground_pixels=int(reference.sum()),
                frozen_foreground_pixels=gg,frozen_query_objects=n,status='PASS' if matched else 'FAIL',
                source_label=r['label_relative_path'],source_metadata_line=r['source_metadata_line']))
            if not matched:
                raise RuntimeError('STOP: public source does not match frozen counts '+d+'/'+r['unit'])
        print(d+': source labels checked',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(f'PASS: {len(records)} evaluated source images/labels; frozen query components and foreground counts exact')

if __name__=='__main__':main()
