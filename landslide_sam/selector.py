"""GBT fitting and deployment, with the frozen parameter set."""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def parameters():
    return json.loads((Path(__file__).resolve().parent.parent / 'configs/frozen_model.json').read_text())['gbt_params']


def train(X, y, equivalent_reference_area, weight='unweighted'):
    if weight not in ('unweighted', 'sqrt', 'area'):
        raise ValueError('Unknown frozen weighting')
    n, points, candidates, features = X.shape
    w = np.repeat(equivalent_reference_area, points*candidates)
    if weight == 'sqrt':
        w = np.sqrt(w)
    model = HistGradientBoostingRegressor(**parameters())
    with threadpool_limits(4):
        model.fit(X.reshape(-1, features), y.reshape(-1),
                  **({} if weight == 'unweighted' else {'sample_weight': w}))
    return model


def predict(model, features):
    with threadpool_limits(4):
        scores = model.predict(features)
    selected = int(np.argmax(scores))
    return selected, float(scores[selected]), scores


def load(path):
    return joblib.load(path)
