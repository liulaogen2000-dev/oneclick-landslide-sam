"""Deployable ranking uses predicted confidence or area only."""
import numpy as np


def boxed_indices(confidence, predicted_area, pools, k, order='confidence'):
    if k not in range(0,101,10) or order not in ('confidence', 'predicted_area'):
        raise ValueError('Invalid boxing allocation')
    pools = np.asarray(pools)
    chosen = []
    for name in sorted(set(pools)):
        ids = np.flatnonzero(pools == name)
        key = np.asarray(confidence)[ids] if order == 'confidence' else -np.asarray(predicted_area)[ids]
        ranking = np.argsort(key, kind='stable')
        chosen.extend(ids[ranking[:len(ids)*int(k)//100]])
    return np.asarray(chosen, int)


def k90(k_grid, whole_image_iou, all_box_iou):
    return next((int(k) for k, value in zip(k_grid, whole_image_iou) if value >= .9*all_box_iou), None)
