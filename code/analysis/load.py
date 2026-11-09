"""Shared loader: every forecast (trained, naive, training-free, pretrained) for a
run, aligned to the same origins and ground truth."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, PRED_DIR, RESULTS, TARGETS, TRAINED_MODELS, kind_of  # noqa: E402

TF_DIR = RESULTS / "training_free_predictions"
NAIVE = ["Naive-Persistence", "Naive-MA15", "Naive-MA60", "Naive-EWMA60"]
CLOSED_FORM = ["AR-Global", "AR-Local", "Ridge-Univariate", "Ridge-Sensors"]
PRETRAINED = ["ChronosBolt-Small", "Chronos2", "Chronos2-Sensors"]
ENSEMBLE = "Ensemble-Mean-13"          # unweighted mean of the 13 trained models (no selection)
FAMILY = {**{m: "trained" for m in TRAINED_MODELS}, **{m: "naive" for m in NAIVE},
          **{m: "closed-form" for m in CLOSED_FORM}, **{m: "pretrained" for m in PRETRAINED},
          ENSEMBLE: "ensemble"}


def runs():
    for r in (1, 2, 3):
        for i in (1, 2, 3):
            for t in TARGETS:
                yield f"R{r}", i, t, f"R{r}_I{i}__{t}"


def load_run(run):
    """Returns truth[fold] (n,15), origins[fold], preds[method][fold] (n,15)."""
    z = np.load(PRED_DIR / f"{run}.npz")
    truth = {f: z[f"ytrue|{f}"].astype(np.float64) for f in FOLDS}
    origins = {f: z[f"origins|{f}"].astype(int) for f in FOLDS}
    preds = {m: {f: z[f"{m}|{f}|0.0"].astype(np.float64) for f in FOLDS} for m in TRAINED_MODELS}
    preds[ENSEMBLE] = {f: np.mean([preds[m][f] for m in TRAINED_MODELS], axis=0) for f in FOLDS}
    tf = np.load(TF_DIR / f"{run}.npz")
    for key in tf.files:
        m, f = key.split("|")
        if m.endswith("_alpha"):
            continue
        preds.setdefault(m, {})[f] = tf[key]
    c2s = TF_DIR / f"{run}__c2s.npz"
    if c2s.exists():
        zz = np.load(c2s)
        for key in zz.files:
            m, f = key.split("|")
            preds.setdefault(m, {})[f] = zz[key]
    preds = {m: v for m, v in preds.items() if len(v) == len(FOLDS)}
    return truth, origins, preds


__all__ = ["runs", "load_run", "FAMILY", "NAIVE", "CLOSED_FORM", "PRETRAINED", "kind_of"]
