"""B2 - accuracy of every method, recomputed from stored forecasts.

Metric: MSE over steps 1..h (cumulative, the convention of the original study)
for h = 1..15, per run, fold and method. Primary operating horizons are the
ones fixed in the original study BEFORE this analysis (vibration h=3,
temperature h=15); all horizons are reported so the choice can be inspected.

Unit of statistical analysis = run (45: 9 datasets x 5 targets), never a
horizon or a fold, because those are not independent. A dataset-level
robustness check (9 units per kind group) is also produced.

Outputs (results/tables/):
  b02_mse_long.csv          run x fold x method x h, MSE
  b02_run_level.csv         run x method x h, MSE averaged over 5 folds, skill vs
                            strongest naive baseline (chosen per run, i.e. the
                            hardest bar for every model)
  b02_summary.csv           per kind x method at the operating horizon
  b02_tests.csv             paired Wilcoxon tests, Holm-corrected
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, TABLES  # noqa: E402
from load import FAMILY, NAIVE, kind_of, load_run, runs  # noqa: E402

HMAX = 15
OP_H = {"vibration": 3, "temperature": 15}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run_max = 0
    for k, idx in enumerate(order):
        run_max = max(run_max, (len(p) - k) * p[idx])
        adj[idx] = min(1.0, run_max)
    return adj


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows, r2rows = [], []
    for reg, it, tgt, run in runs():
        truth, _, preds = load_run(run)
        H = OP_H[kind_of(tgt)]
        for m, pf in preds.items():
            r2 = [1 - ((truth[f][:, :H] - pf[f][:, :H]) ** 2).sum()
                  / ((truth[f][:, :H] - truth[f][:, :H].mean()) ** 2).sum() for f in FOLDS]
            r2rows.append((run, kind_of(tgt), m, FAMILY.get(m, "other"), H, float(np.mean(r2))))
            for f in FOLDS:
                se = (pf[f] - truth[f]) ** 2
                cum = np.cumsum(se.mean(0)) / np.arange(1, HMAX + 1)
                for h in range(1, HMAX + 1):
                    rows.append((run, reg, it, tgt, kind_of(tgt), f, m, FAMILY.get(m, "other"), h, cum[h - 1]))
    long = pd.DataFrame(rows, columns=["run", "region", "iteration", "target", "kind", "fold",
                                       "method", "family", "h", "mse"])
    long.to_csv(TABLES / "b02_mse_long.csv", index=False)
    pd.DataFrame(r2rows, columns=["run", "kind", "method", "family", "h", "r2_fold_mean"]) \
        .to_csv(TABLES / "b02_r2_operating_horizon.csv", index=False)

    rl = long.groupby(["run", "region", "iteration", "target", "kind", "method", "family", "h"],
                      as_index=False).mse.mean()
    best_naive = rl[rl.method.isin(NAIVE)].groupby(["run", "h"]).mse.min().rename("naive_best")
    rl = rl.join(best_naive, on=["run", "h"])
    rl["skill_vs_best_naive"] = 1 - rl.mse / rl.naive_best
    rl["beats_best_naive"] = rl.mse < rl.naive_best
    rl.to_csv(TABLES / "b02_run_level.csv", index=False)

    op = rl[rl.h == rl.kind.map(OP_H)]
    summ = op.groupby(["kind", "family", "method"]).agg(
        n_runs=("run", "size"), median_skill=("skill_vs_best_naive", "median"),
        mean_skill=("skill_vs_best_naive", "mean"), runs_beating_best_naive=("beats_best_naive", "sum")
    ).reset_index()
    ranks = op.assign(rank=op.groupby("run").mse.rank()).groupby(["kind", "method"])["rank"].mean()
    summ = summ.join(ranks.rename("mean_rank"), on=["kind", "method"]).sort_values(["kind", "mean_rank"])
    summ.to_csv(TABLES / "b02_summary.csv", index=False)

    # paired tests at the operating horizon: each training-free / pretrained method vs
    # (a) TiDE (best trained model in the original study), (b) the run's median trained model
    tests = []
    for kind, g in op.groupby("kind"):
        piv = g.pivot(index="run", columns="method", values="mse")
        trained = [m for m in piv.columns if FAMILY.get(m) == "trained"]
        med_tr = piv[trained].median(axis=1)
        for m in piv.columns:
            if FAMILY.get(m) not in ("closed-form", "pretrained", "naive", "ensemble"):
                continue
            for ref_name, ref in (("TiDE", piv["TiDE"]), ("median trained model", med_tr)):
                lr = np.log(piv[m] / ref).dropna()
                p = stats.wilcoxon(lr).pvalue if (lr != 0).any() else 1.0
                tests.append(dict(kind=kind, method=m, reference=ref_name, n_runs=len(lr),
                                  median_mse_ratio=float(np.exp(lr.median())),
                                  runs_method_better=int((lr < 0).sum()), p_wilcoxon=p))
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_wilcoxon)
    tests.to_csv(TABLES / "b02_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
