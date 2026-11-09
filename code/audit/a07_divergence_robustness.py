"""A7 - diverged trained-model forecasts, and whether they drive any headline result.

Found after the first audit, while checking the tuning re-runs: in a few
(model, run, fold) cells the stored trained-model forecast diverged, e.g. NHITS
on R3_I2 / self_temperature_one / F2 predicts up to 92 degC when the truth stays
below 37 degC (MSE 526x persistence). These forecasts are exactly what the
models produced; they are kept in every primary result. This script

  1. lists every cell whose operating-horizon MSE exceeds T x persistence MSE
     on the same fold, for T = 10, 20, 50 (the threshold is not tuned: all three
     are reported and the strictest, 10, is used for the robustness check);
  2. recomputes the headline comparisons with the flagged cells removed
     (model AND reference scored on the same remaining folds);
  3. adds Ensemble-Median-13 (per-step median of the 13 trained forecasts), which
     a single diverged member cannot dominate, beside Ensemble-Mean-13.

Outputs: results/audit/a07_divergent_cells.csv, results/audit/a07_robustness.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import FOLDS, RESULTS, TABLES, TRAINED_MODELS  # noqa: E402
from load import NAIVE, kind_of, load_run, runs  # noqa: E402

OP_H = {"vibration": 3, "temperature": 15}
AUD = RESULTS / "audit"
THRESHOLDS = (10, 20, 50)
T_ROBUST = 10
FREE = ["Ridge-Univariate", "Ridge-Sensors", "AR-Global", "AR-Local", "ChronosBolt-Small", "Chronos2"]


def wil(lr):
    lr = np.asarray(lr, float)
    return stats.wilcoxon(lr).pvalue if len(lr) >= 5 and (lr != 0).any() else np.nan


def main():
    long = pd.read_csv(TABLES / "b02_mse_long.csv")
    op = long[long.h == long.kind.map(OP_H)].copy()

    # Ensemble-Median-13 at the operating horizon, fold level
    med_rows = []
    for reg, it, tgt, run in runs():
        truth, _, preds = load_run(run)
        H = OP_H[kind_of(tgt)]
        for f in FOLDS:
            p = np.median(np.stack([preds[m][f] for m in TRAINED_MODELS]), axis=0)
            med_rows.append(dict(run=run, kind=kind_of(tgt), fold=f, method="Ensemble-Median-13",
                                 family="ensemble", mse=float(((p[:, :H] - truth[f][:, :H]) ** 2).mean())))
    op = pd.concat([op, pd.DataFrame(med_rows)], ignore_index=True)

    pers = op[op.method == "Naive-Persistence"].set_index(["run", "fold"]).mse
    tr = op[op.method.isin(TRAINED_MODELS)].copy()
    tr["x_persistence"] = tr.mse.values / pers.loc[list(zip(tr.run, tr.fold))].values
    cells = tr[tr.x_persistence > min(THRESHOLDS)][["run", "kind", "fold", "method", "mse", "x_persistence"]]
    for t in THRESHOLDS:
        cells[f"over_{t}x"] = cells.x_persistence > t
    cells.sort_values("x_persistence", ascending=False).to_csv(AUD / "a07_divergent_cells.csv", index=False)
    bad = set(map(tuple, cells[cells.x_persistence > T_ROBUST][["run", "fold", "method"]].values))

    rows = []

    def add(kind, claim, variant, value, better=None, n=None, p=None):
        rows.append(dict(kind=kind, claim=claim, variant=variant, value=round(float(value), 4),
                         runs_better=better, n_runs=n, p_wilcoxon=p))

    for kind in ("vibration", "temperature"):
        g = op[op.kind == kind]
        for variant in ("as reported", f"diverged cells (>{T_ROBUST}x persistence) removed"):
            if variant == "as reported":
                keep = g
            else:   # drop flagged (run, fold) for the flagged model only; references keep all folds
                keep = g[[(r, f, m) not in bad for r, f, m in zip(g.run, g.fold, g.method)]]
            # per (run, method): mean over folds; a model compared to a reference uses common folds
            fm = keep.pivot_table(index=["run", "fold"], columns="method", values="mse")

            def run_mse(m, ref=None):
                d = fm[[m] + ([ref] if ref else [])].dropna()
                return d.groupby(level="run").mean()

            naive_best = fm[NAIVE].groupby(level="run").mean().min(axis=1)
            n = fm.index.get_level_values("run").nunique()
            beats = {}
            for m in TRAINED_MODELS:
                d = fm[[m] + NAIVE].dropna().groupby(level="run").mean()
                beats[m] = int((d[m] < d[NAIVE].min(axis=1)).sum())
            add(kind, "trained models beating best naive in a majority of runs", variant,
                sum(v > n / 2 for v in beats.values()))
            tr_run = fm[TRAINED_MODELS].groupby(level="run").mean()      # mean over remaining folds
            med_tr = tr_run.median(axis=1)
            for m in FREE + ["Ensemble-Mean-13", "Ensemble-Median-13"]:
                if m not in fm:
                    continue
                for ref_name in ("TiDE", "median trained model"):
                    if ref_name == "TiDE":
                        d = run_mse(m, "TiDE")
                        lr = np.log(d[m] / d["TiDE"])
                    else:
                        a = fm[m].groupby(level="run").mean()
                        lr = np.log(a / med_tr).dropna()
                    add(kind, f"{m} vs {ref_name}: median MSE ratio", variant, np.exp(lr.median()),
                        int((lr < 0).sum()), len(lr), wil(lr))
            add(kind, "best naive (reference)", variant, naive_best.median())

        # anchored recalibration with flagged model x run cells removed (folds 2-5 used by b03)
        rec = pd.read_csv(TABLES / "b03_recal_run_level.csv")
        rec = rec[(rec.kind == kind) & rec.method.isin(TRAINED_MODELS) & (rec.variant != "raw")]
        flagged_mr = {(r, m) for r, f, m in bad if f != FOLDS[0]}
        for variant, d in (("as reported", rec),
                           (f"diverged cells (>{T_ROBUST}x persistence) removed",
                            rec[[(r, m) not in flagged_mr for r, m in zip(rec.run, rec.method)]])):
            for v in ("shrink", "affine", "anchored"):
                per_run = np.log(d[d.variant == v].ratio_vs_raw).groupby(d[d.variant == v].run).mean()
                add(kind, f"recalibration '{v}': median MSE ratio vs raw", variant, np.exp(per_run.median()),
                    int((per_run < 0).sum()), len(per_run), wil(per_run))

    out = pd.DataFrame(rows)
    out.to_csv(AUD / "a07_robustness.csv", index=False)
    print(f"flagged cells: " + ", ".join(f">{t}x: {int((cells.x_persistence > t).sum())}" for t in THRESHOLDS)
          + f" of {len(tr)}")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
