"""B3 - post-hoc recalibration of point forecasts, CAUSAL protocol.

Correction of the earlier study: there, the rescaling constant for a test fold
was fitted on all OTHER folds, including later ones (future information), the
shrink factor was clipped to [0.02, 0.98], and the learned variant (SCAS) was
replaced by B1 whenever B1 scored better on the TEST fold. None of that is done
here.

Protocol: folds are time-ordered. For test fold k (k = 2..5) the calibration is
fitted only on (forecast, truth) pairs of folds 1..k-1, i.e. data that precede
fold k. Fold 1 has no past and is excluded from ALL methods' scores here, so
raw and calibrated numbers are always compared on identical folds 2..5.
Parameters are per horizon step h (1..15).

Variants
  raw        unchanged forecast
  shrink     y' = m + b (y_hat - m), m = mean past forecast, b = OLS slope (unclipped)
  affine     y' = a + b y_hat (OLS on past folds)
  anchored   y' = y_last + b (y_hat - y_last), y_last = last observed value at the
             origin; b = through-origin OLS. Anchoring at the current level removes
             the level drift that defeats the other two on non-stationary channels.
Outputs: results/tables/b03_recal_run_level.csv, b03_recal_summary.csv, b03_recal_tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, TABLES  # noqa: E402
from load import FAMILY, kind_of, load_run, runs  # noqa: E402

OP_H = {"vibration": 3, "temperature": 15}
VARIANTS = ["raw", "shrink", "affine", "anchored"]


def fit_apply(variant, P, Y, L, Pt, Lt):
    """P, Y, L: past forecasts / truth / last value, shape (n, 15); Pt, Lt: test."""
    if variant == "raw":
        return Pt
    out = np.empty_like(Pt)
    for h in range(Pt.shape[1]):
        p, y = P[:, h], Y[:, h]
        if variant == "shrink":
            m = p.mean()
            b = np.dot(p - m, y - y.mean()) / max(np.dot(p - m, p - m), 1e-12)
            out[:, h] = m + b * (Pt[:, h] - m)
        elif variant == "affine":
            b = np.dot(p - p.mean(), y - y.mean()) / max(np.dot(p - p.mean(), p - p.mean()), 1e-12)
            a = y.mean() - b * p.mean()
            out[:, h] = a + b * Pt[:, h]
        elif variant == "anchored":
            dp, dy = p - L[:, 0], y - L[:, 0]
            b = np.dot(dp, dy) / max(np.dot(dp, dp), 1e-12)
            out[:, h] = Lt[:, 0] + b * (Pt[:, h] - Lt[:, 0])
    return out


def main():
    rows = []
    for reg, it, tgt, run in runs():
        kind = kind_of(tgt)
        H = OP_H[kind]
        truth, _, preds = load_run(run)
        last = {f: preds["Naive-Persistence"][f][:, :1] for f in FOLDS}
        for m, pf in preds.items():
            if FAMILY.get(m) not in ("trained", "ensemble", "closed-form", "pretrained"):
                continue
            for v in VARIANTS:
                mses = []
                for k in range(1, len(FOLDS)):
                    past = FOLDS[:k]
                    P = np.vstack([pf[f] for f in past])
                    Y = np.vstack([truth[f] for f in past])
                    L = np.vstack([last[f] for f in past])
                    f = FOLDS[k]
                    yc = fit_apply(v, P, Y, L, pf[f], last[f])
                    mses.append(float(((yc[:, :H] - truth[f][:, :H]) ** 2).mean()))
                rows.append(dict(run=run, region=reg, iteration=it, target=tgt, kind=kind, method=m,
                                 family=FAMILY[m], variant=v, h=H, mse_folds2to5=float(np.mean(mses))))
    d = pd.DataFrame(rows)
    raw = d[d.variant == "raw"].set_index(["run", "method"]).mse_folds2to5
    d["ratio_vs_raw"] = d.mse_folds2to5 / d.set_index(["run", "method"]).index.map(raw).values
    d.to_csv(TABLES / "b03_recal_run_level.csv", index=False)

    summ = d[d.variant != "raw"].groupby(["kind", "family", "variant"]).agg(
        cells=("ratio_vs_raw", "size"), median_ratio=("ratio_vs_raw", "median"),
        share_improved=("ratio_vs_raw", lambda x: (x < 1).mean())).reset_index()
    summ.to_csv(TABLES / "b03_recal_summary.csv", index=False)

    # tests at run level: per run, mean log-ratio across the 13 TRAINED models (one unit per run)
    tests = []
    tr = d[(d.family == "trained") & (d.variant != "raw")].copy()
    tr["lr"] = np.log(tr.ratio_vs_raw)
    per_run = tr.groupby(["kind", "variant", "run"]).lr.mean().reset_index()
    for (kind, v), g in per_run.groupby(["kind", "variant"]):
        tests.append(dict(kind=kind, variant=v, units="runs (mean over 13 trained models)",
                          n=len(g), median_ratio=float(np.exp(g.lr.median())),
                          runs_improved=int((g.lr < 0).sum()), p_wilcoxon=stats.wilcoxon(g.lr).pvalue))
    # does recalibration change which trained model is best?
    best_change = []
    for (kind, v), g in d[d.family == "trained"].groupby(["kind", "variant"]):
        if v == "raw":
            continue
        n_same = 0
        for run, gg in g.groupby("run"):
            raw_best = d[(d.run == run) & (d.family == "trained") & (d.variant == "raw")] \
                .sort_values("mse_folds2to5").method.iloc[0]
            n_same += int(gg.sort_values("mse_folds2to5").method.iloc[0] == raw_best)
        best_change.append(dict(kind=kind, variant=v, runs=g.run.nunique(), best_trained_model_unchanged=n_same))
    pd.DataFrame(tests).to_csv(TABLES / "b03_recal_tests.csv", index=False)
    pd.DataFrame(best_change).to_csv(TABLES / "b03_recal_best_model_stability.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print(pd.DataFrame(best_change).to_string(index=False))


if __name__ == "__main__":
    main()
