"""B5 - can a training-free predictability score, computed on TRAINING rows only,
anticipate how the trained models will behave?

Scores per run (all from the first 45% of rows = F1's training block, so no
test information):
  lambda      signal fraction: exp(intercept) of a line fitted to log ACF(1..6)
              (the earlier study's estimator; capped at 1 there, UNcapped here)
  acf_h       autocorrelation at the operating horizon
  ridge_val   validation skill of Ridge-Univariate vs persistence (F1 val block)
Outcomes (from B2/B4, operating horizon):
  share of the 13 trained models that beat the best naive baseline
  median overconfidence (sd_ratio / r) of the trained models
  median gain of anchored recalibration for the trained models (B3)
Spearman correlations, overall (n=45) and within each channel kind, are
reported in full, whatever their sign or significance.
Output: results/tables/b05_predictability.csv, b05_correlations.csv
(the latter with a Holm-adjusted p over the 4 outcomes of each scope x score)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES, TRAINED_MODELS  # noqa: E402
from load import kind_of, runs  # noqa: E402

OP_H = {"vibration": 3, "temperature": 15}


def _holm(p):
    p = np.asarray(p, float)
    order, adj, run_max = np.argsort(p), np.empty(len(p)), 0.0
    for k, idx in enumerate(order):
        run_max = max(run_max, (len(p) - k) * p[idx])
        adj[idx] = min(1.0, run_max)
    return adj


def lam(y):
    s = pd.Series(y)
    k = np.arange(1, 7)
    a = np.array([s.autocorr(i) for i in k])
    m = np.isfinite(a) & (a > 0)
    return float(np.exp(np.polyfit(k[m], np.log(a[m]), 1)[1])) if m.sum() >= 3 else np.nan


def main():
    rl = pd.read_csv(TABLES / "b02_run_level.csv")
    dec = pd.read_csv(TABLES / "b04_decomposition.csv")
    rec = pd.read_csv(TABLES / "b03_recal_run_level.csv")
    rows = []
    for reg, it, tgt, run in runs():
        kind = kind_of(tgt)
        H = OP_H[kind]
        df = pd.read_csv(DATA / f"{reg}_I{it}.csv")
        col = "y" if tgt == "self_temperature_one" else tgt
        y = df[col].to_numpy(float)[:int(0.45 * len(df))]
        op = rl[(rl.run == run) & (rl.h == H)]
        tr = op[op.method.isin(TRAINED_MODELS)]
        dd = dec[(dec.run == run) & dec.method.isin(TRAINED_MODELS)]
        rr = rec[(rec.run == run) & rec.method.isin(TRAINED_MODELS) & (rec.variant == "anchored")]
        rows.append(dict(run=run, kind=kind, lambda_uncapped=lam(y), acf_h=float(pd.Series(y).autocorr(H)),
                         share_trained_beat_naive=float(tr.beats_best_naive.mean()),
                         median_overconf=float(dd.overconf.median()),
                         median_anchor_ratio=float(rr.ratio_vs_raw.median()),
                         best_skill=float(op.skill_vs_best_naive.max())))
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "b05_predictability.csv", index=False)
    cor = []
    for scope, g in [("all", d)] + list(d.groupby("kind")):
        for x in ("lambda_uncapped", "acf_h"):
            for y in ("share_trained_beat_naive", "median_overconf", "median_anchor_ratio", "best_skill"):
                gg = g[[x, y]].dropna()
                if gg[x].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(gg[x], gg[y])
                cor.append(dict(scope=scope, score=x, outcome=y, n=len(gg), spearman=rho, p=p))
    cor = pd.DataFrame(cor)
    # Holm correction over the 4 outcomes tested for each (scope, score)
    cor["p_holm_4_outcomes"] = cor.groupby(["scope", "score"]).p.transform(lambda s: _holm(s.values))
    cor.to_csv(TABLES / "b05_correlations.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby("kind")[["lambda_uncapped", "acf_h"]].describe().round(3).T.to_string())
    print(cor.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
