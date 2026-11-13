"""Frozen hand-crafted features; arithmetic is preserved in _frozen_features.py."""
import numpy as np
from . import _frozen_features as frozen

FEATURES = ('sam', 'stab', 'aratio', 'comp', 'dcent', 'br', 'sat', 'sob', 'grad',
            'logarea', 'arank', 'logscale', 'ar', 'xscale_iou')


def extract_features(rgb, point_yx, candidates, dataset, gsd=None):
    scene = dataset in ('GVLM_v1_0', 'GVLM_v1_1')
    expected = 12 if scene else 3
    if len(candidates) != expected:
        raise ValueError(f'{dataset} requires {expected} candidates')
    result = []
    for s in range(expected // 3):
        group = candidates[s*3:s*3+3]
        areas = [int(c.mask.sum()) for c in group]
        ranks = np.argsort(np.argsort(-np.array(areas), kind='stable'), kind='stable') + 1
        y, x, h, w = group[0].window
        crop = rgb[y:y+h, x:x+w]
        planes = frozen.planes(crop)
        local_point = (int(point_yx[0])-y, int(point_yx[1])-x)
        for k, c in enumerate(group):
            if dataset in ('jz', 'bijie'):
                frozen._DS = dataset
                f = frozen.cand_feats(c.mask, c.score, c.logits, *planes, None,
                                      local_point, h*w, areas, int(ranks[k]))
                f['logscale'] = float(np.log2(c.nominal_side))
            else:
                f = frozen.feature(c.mask, c.score, c.stability, crop, local_point,
                                   areas, ranks[k], c.nominal_side, planes)
                if dataset != 'GVLM_v1_0':
                    if gsd is None:
                        raise ValueError('Native GSD is required for physical-area features')
                    f['logarea'] = float(np.log10(areas[k]*gsd[0]*gsd[1])) if areas[k] else None
            result.append(f)
    if scene:
        cross = np.zeros(12)
        for j in range(12):
            for k in range(j+1, 12):
                if j//3 == k//3:
                    continue
                a, b = candidates[j], candidates[k]
                v = frozen.mask_iou(a.mask, a.window, b.mask, b.window)
                cross[j], cross[k] = max(cross[j], v), max(cross[k], v)
        for i, f in enumerate(result):
            f['xscale_iou'] = float(cross[i])
    names = FEATURES if scene else FEATURES[:-1]
    return np.array([[np.nan if f[name] is None else f[name] for name in names] for f in result], float)
