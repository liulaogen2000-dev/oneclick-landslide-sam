"""Reference data enter here only after predictions have been frozen."""
import numpy as np


def mask_counts(prediction, reference):
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    return int((prediction & reference).sum()), int(prediction.sum()), int(reference.sum())


def scores(tp, pp, gg):
    return {'iou': tp / max(pp + gg - tp, 1),
            'precision': tp / max(pp, 1), 'recall': tp / max(gg, 1)}


def object_iou(prediction, full_reference_object):
    return scores(*mask_counts(prediction, full_reference_object))['iou']


def union_map(masks, shape):
    result = np.zeros(shape, bool)
    for mask, (y, x, h, w) in masks:
        result[y:y+h, x:x+w] |= mask
    return result


def paired_statistics(T, P, G, OS, OC, clusters, repeats=10000, seed=0):
    """Shared cluster draws for every method; ratio statistics recomputed per draw."""
    groups = [np.flatnonzero(clusters == name) for name in sorted(set(clusters))]
    def aggregate(a):
        return np.asarray([a[g].sum(0) for g in groups])
    et, ep, eg, eo, ec = [aggregate(v) for v in (T, P, G, OS, OC)]
    cluster_iou = et / np.maximum(ep + eg[:, None] - et, 1)
    point = {'iou': T.sum(0)/np.maximum(P.sum(0)+G.sum()-T.sum(0), 1),
             'equal_iou': cluster_iou.mean(0), 'object_iou': OS.sum(0)/max(OC.sum(), 1)}
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(repeats, len(groups)))
    w = np.asarray([np.bincount(v, minlength=len(groups)) for v in draws], float)
    bt, bp, bg, bo, bc = [w @ v for v in (et, ep, eg, eo, ec)]
    boot = {'iou': bt/np.maximum(bp+bg[:, None]-bt, 1),
            'equal_iou': w @ cluster_iou / len(groups),
            'object_iou': bo/np.maximum(bc[:, None], 1)}
    return point, boot


def percentile_interval(samples):
    return np.nanpercentile(samples, [2.5, 97.5], axis=0)
