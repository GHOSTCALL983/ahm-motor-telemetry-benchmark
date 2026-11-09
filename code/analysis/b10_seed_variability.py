"""B10 - training-seed variability (seeds 42, 101, 202) on the 14 iteration-1 runs.

Written and fixed BEFORE the extra-seed results were looked at (they were
found during the final audit; the earlier audit had wrongly stated that only
seed 42 existed).

Data: seed 42 = the stored runs used everywhere else; seeds 101 and 202 =
results/trained_model_predictions_extra_seeds/ (import_seed_runs.py). Same
code, data, folds, origins and settings; only the seed differs. 14 runs:
iteration 1, 3 regions x 5 targets, minus Region 1 / self_x_ac (not repeated).
Vibration: 8 runs; temperature: 6 runs.

Questions and fixed analyses:
  1. Integrity: origins and ground truth identical to seed 42.
  2. How much does a model's error move with the seed? Per run x model:
     operating-horizon MSE (mean over 5 folds) at each seed; coefficient of
     variation over the 3 seeds and max/min ratio; median over runs.
  3. Was seed 42 a favourable draw? Share of run x model cells in which seed 42
     has the lowest MSE of the three (1/3 expected if seeds are exchangeable)
     and mean rank (2 expected).
  4. Do the headline conclusions hold at every seed? For each seed separately,
     and for the per-model mean forecast over the 3 seeds ("seed-mean"):
     trained models beating the strongest naive baseline in a majority of runs,
     Ridge-Univariate vs TiDE, Ridge-Univariate vs the run's median trained
     model, how often TiDE is the best trained model, and the causal anchored
     recalibration (same function and protocol as b03).
Training-free and naive forecasts do not depend on the seed; they are the same
in every column.

Outputs (results/tables/): b10_seed_integrity.csv, b10_seed_spread.csv,
b10_seed_conclusions.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, RESULTS, TABLES, TRAINED_MODELS  # noqa: E402
from load import NAIVE, kind_of, load_run, runs  # noqa: E402
from b03_recalibration_causal import fit_apply  # noqa: E402

SEED_DIR = RESULTS / "trained_model_predictions_extra_seeds"
OP_H = {"vibration": 3, "temperature": 15}
SEEDS = [42, 101, 202]


def mse(p, y, H):
    return float(((p[:, :H] - y[:, :H]) ** 2).mean())


def anchored_ratio(pf, truth, last, H):
    raw, cal = [], []
    for k in range(1, len(FOLDS)):
        past, f = FOLDS[:k], FOLDS[k]
        yc = fit_apply("anchored", np.vstack([pf[g] for g in past]), np.vstack([truth[g] for g in past]),
                       np.vstack([last[g] for g in past]), pf[f], last[f])
        raw.append(mse(pf[f], truth[f], H))
        cal.append(mse(yc, truth[f], H))
    return np.mean(cal) / np.mean(raw)


def main():
    integ, cells, concl_rows = [], [], []
    per_run = {}
    for reg, it, tgt, run in runs():
        if it != 1 or not (SEED_DIR / f"{run}__seed101.npz").exists():
            continue
        kind, H = kind_of(tgt), OP_H[kind_of(tgt)]
        truth, origins, preds = load_run(run)
        tr = {42: {m: preds[m] for m in TRAINED_MODELS}}
        for s in SEEDS[1:]:
            z = np.load(SEED_DIR / f"{run}__seed{s}.npz")
            integ.append(dict(run=run, seed=s,
                              origins_identical=all(np.array_equal(z[f"origins|{f}"], origins[f]) for f in FOLDS),
                              truth_max_abs_diff=max(float(np.abs(z[f"ytrue|{f}"] - truth[f]).max()) for f in FOLDS),
                              nan_cells=int(sum(np.isnan(z[f"{m}|{f}|0.0"]).sum() for m in TRAINED_MODELS for f in FOLDS))))
            tr[s] = {m: {f: z[f"{m}|{f}|0.0"].astype(np.float64) for f in FOLDS} for m in TRAINED_MODELS}
        tr["seed-mean"] = {m: {f: np.mean([tr[s][m][f] for s in SEEDS], axis=0) for f in FOLDS} for m in TRAINED_MODELS}
        last = {f: preds["Naive-Persistence"][f][:, :1] for f in FOLDS}
        other = {m: np.mean([mse(preds[m][f], truth[f], H) for f in FOLDS]) for m in NAIVE + ["Ridge-Univariate"]}
        per_run[run] = dict(kind=kind, other=other, trained={}, anchored={})
        for s, d in tr.items():
            per_run[run]["trained"][s] = {m: np.mean([mse(d[m][f], truth[f], H) for f in FOLDS]) for m in TRAINED_MODELS}
            per_run[run]["anchored"][s] = {m: anchored_ratio(d[m], truth, last, H) for m in TRAINED_MODELS}
        for m in TRAINED_MODELS:
            v = np.array([per_run[run]["trained"][s][m] for s in SEEDS])
            cells.append(dict(run=run, kind=kind, model=m, mse_seed42=v[0], mse_seed101=v[1], mse_seed202=v[2],
                              cv=v.std(ddof=1) / v.mean(), max_over_min=v.max() / v.min(),
                              seed42_rank=int(stats.rankdata(v)[0]), seed42_best=bool(v[0] == v.min())))
    integ, cells = pd.DataFrame(integ), pd.DataFrame(cells)
    TABLES.mkdir(parents=True, exist_ok=True)
    integ.to_csv(TABLES / "b10_seed_integrity.csv", index=False)

    spread = cells.groupby(["kind", "model"]).agg(
        n_runs=("run", "size"), median_cv=("cv", "median"), median_max_over_min=("max_over_min", "median"),
        worst_max_over_min=("max_over_min", "max"), share_seed42_best=("seed42_best", "mean"),
        mean_seed42_rank=("seed42_rank", "mean")).reset_index()
    allrow = cells.groupby("kind").agg(n_runs=("run", "nunique"), median_cv=("cv", "median"),
                                       median_max_over_min=("max_over_min", "median"),
                                       worst_max_over_min=("max_over_min", "max"),
                                       share_seed42_best=("seed42_best", "mean"),
                                       mean_seed42_rank=("seed42_rank", "mean")).reset_index().assign(model="ALL 13")
    spread = pd.concat([spread, allrow], ignore_index=True)
    spread.to_csv(TABLES / "b10_seed_spread.csv", index=False)

    for kind in ("vibration", "temperature"):
        rr = {r: v for r, v in per_run.items() if v["kind"] == kind}
        n = len(rr)
        for s in SEEDS + ["seed-mean"]:
            beats = {m: sum(v["trained"][s][m] < min(v["other"][x] for x in NAIVE) for v in rr.values())
                     for m in TRAINED_MODELS}
            lr_t = np.array([np.log(v["other"]["Ridge-Univariate"] / v["trained"][s]["TiDE"]) for v in rr.values()])
            lr_m = np.array([np.log(v["other"]["Ridge-Univariate"] / np.median(list(v["trained"][s].values())))
                             for v in rr.values()])
            tide_best = sum(min(v["trained"][s], key=v["trained"][s].get) == "TiDE" for v in rr.values())
            anc = np.array([np.mean(np.log(list(v["anchored"][s].values()))) for v in rr.values()])
            concl_rows.append(dict(
                kind=kind, seed=str(s), n_runs=n,
                trained_beating_best_naive_in_majority=int(sum(b > n / 2 for b in beats.values())),
                models_beating=", ".join(f"{m} {b}/{n}" for m, b in sorted(beats.items(), key=lambda x: -x[1]) if b),
                ridge_vs_tide_median_ratio=float(np.exp(np.median(lr_t))), ridge_better_than_tide=int((lr_t < 0).sum()),
                ridge_vs_median_trained_ratio=float(np.exp(np.median(lr_m))),
                ridge_better_than_median_trained=int((lr_m < 0).sum()),
                tide_best_trained_model_runs=int(tide_best),
                anchored_median_ratio=float(np.exp(np.median(anc))), anchored_runs_improved=int((anc < 0).sum())))
    concl = pd.DataFrame(concl_rows)
    concl.to_csv(TABLES / "b10_seed_conclusions.csv", index=False)
    pd.set_option("display.width", 250)
    print(integ.agg(dict(origins_identical="all", truth_max_abs_diff="max", nan_cells="sum")).to_string())
    print(spread.round(3).to_string(index=False))
    print(concl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
