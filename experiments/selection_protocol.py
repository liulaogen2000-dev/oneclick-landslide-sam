"""Training-only nested selection, shared across all three selectors.

The validation callback must return summed TP, PP and GG for all held-out
validation images and BOTH P1/P2. It must not receive outer-test labels.
This callback interface keeps mask-union evaluation separate from fitting.
"""
import numpy as np

WEIGHTS=('unweighted','sqrt','area')


def nested_weight(outer_training_groups, fit_predict, validation_counts):
    scores=[]
    for weight in WEIGHTS:
        tp=pp=gg=0
        for held in outer_training_groups:
            train=[g for g in outer_training_groups if g!=held]
            model=fit_predict(train,weight)
            t,p,g=validation_counts(model,held)
            tp+=t;pp+=p;gg+=g
        scores.append(tp/max(pp+gg-tp,1))
    return WEIGHTS[int(np.argmax(scores))],dict(zip(WEIGHTS,scores))


def tile_weight(fit_on_train, validation_counts):
    scores=[]
    for weight in WEIGHTS:
        t,p,g=validation_counts(fit_on_train(weight))
        scores.append(t/max(p+g-t,1))
    return WEIGHTS[int(np.argmax(scores))],dict(zip(WEIGHTS,scores))
