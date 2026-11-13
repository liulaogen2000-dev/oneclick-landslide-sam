"""Input: RGB image and clicks in global [x,y] coordinates. No labels are read."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from landslide_sam.candidates import FrozenSAM
from landslide_sam.features import extract_features
from landslide_sam.selector import load, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--clicks', type=Path, required=True, help='JSON list of [x,y] clicks')
    p.add_argument('--dataset', choices=['GVLM_v1_0','GVLM_v1_1','CAS','jz','bijie'], required=True)
    p.add_argument('--gsd', nargs=2, type=float, metavar=('X_METRES','Y_METRES'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.image.suffix.lower() in ['.tif','.tiff']:
        import rasterio
        with rasterio.open(a.image) as image:
            rgb=np.moveaxis(image.read()[:3],0,-1)
    else:
        rgb = np.asarray(Image.open(a.image).convert('RGB'))
    clicks = json.loads(a.clicks.read_text())
    sam, selector = FrozenSAM(a.checkpoint), load(a.model)
    a.output.mkdir(parents=True, exist_ok=False)
    union = np.zeros(rgb.shape[:2], bool)
    records = []
    for i, (x, y) in enumerate(clicks):
        if int(x) != x or int(y) != y:
            raise ValueError('The frozen simulated-click interface expects integer pixel centres')
        c = sam.generate(rgb, (y,x), a.dataset, a.gsd)
        X = extract_features(rgb, (y,x), c, a.dataset, a.gsd)
        index, confidence, scores = predict(selector, X)
        chosen = c[index]; yy, xx, h, w = chosen.window
        full = np.zeros(rgb.shape[:2], bool)
        full[yy:yy+h, xx:xx+w] = chosen.mask
        union |= full
        Image.fromarray(full.astype('uint8')*255).save(a.output / f'object_{i:04d}.png')
        records.append(dict(click_xy=[x,y], selected_candidate=index, confidence=confidence,
                            candidate_scores=scores.tolist()))
    Image.fromarray(union.astype('uint8')*255).save(a.output/'union.png')
    (a.output/'confidence.json').write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__':
    main()
