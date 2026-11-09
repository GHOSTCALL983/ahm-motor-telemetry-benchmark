"""B9 - does hyperparameter tuning change the conclusions?

Written and fixed BEFORE the tuning results were seen.

Data: results/tuning/ (import with code/training/import_tuning_runs.py).
18 runs = 9 datasets x {self_x_ac (vibration), self_temperature_one (temperature)}.
Models TiDE, TimeXer, NHITS; 8 configurations each; configuration 0 is the
original setting; configurations 1-7 were drawn from a fixed grid with a fixed
seed (2026) before any run. Every configuration was trained on each of the 5
folds exactly as in the original study and forecast (a) the fold's validation
block and (b) the fold's test origins.

Selection rule (the only one used for claims):
  for each fold and model, pick the configuration with the lowest MSE on that
  fold's VALIDATION block at the operating horizon (vibration h=3,
  temperature h=15). The validation block precedes the test block. Test data
  are never used to select anything.
  "Tuned-Any" additionally picks, per fold, the best of all 24 model x
  configuration pairs on validation.
Reference only (never used for claims, labelled as such in every table):
  "Oracle" picks the configuration with the lowest TEST MSE. It is an upper
  bound on what tuning could achieve with perfect hindsight.

Checks before any comparison:
  * test origins and ground truth identical to the stored trained runs;
  * every validation target precedes every test target;
  * configuration 0 re-run vs the stored original forecasts (GPU training is
    not bit-reproducible, so the difference is reported, not required to be 0).

Addition made after the first two runs returned, BEFORE any tuned result was
looked at: the configuration-0 re-run reproduced the stored TiDE forecasts
(test MSE ratio 0.998-1.001) but not NHITS (0.44-1.24) or TimeXer (0.78-1.38),
although the settings are identical (GPU training of these architectures is
not bit-reproducible under a fixed seed). Tuned is therefore also compared
with the configuration-0 re-run from the same session, which isolates the
effect of the hyperparameters from run-to-run training noise; the run-to-run
spread itself is reported (b09_tuning_selection.csv).

Unit of analysis: run (9 per kind). Paired Wilcoxon, Holm within kind. Win
counts are reported next to every p-value.

Outputs (results/tables/):
  b09_tuning_integrity.csv   per run x fold checks
  b09_tuning_selection.csv   per run x model x fold: chosen configuration
  b09_tuning_run_level.csv   per run x method: MSE at the operating horizon
  b09_tuning_summary.csv     per kind x method: median ratio to original TiDE, wins
  b09_tuning_tests.csv       paired tests, Holm-corrected
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, RESULTS, TABLES  # noqa: E402
from load import NAIVE, kind_of, load_run  # noqa: E402

TUNE_DIR = Path(os.environ.get("B09_TUNE_DIR", RESULTS / "tuning"))
OUT = Path(os.environ.get("B09_OUT", TABLES))   # overrides exist only for the self-test
TUNED = ["TiDE", "TimeXer", "NHITS"]
OP_H = {"vibration": 3, "temperature": 15}
REFERENCE = ["Ridge-Univariate", "Ridge-Sensors", "AR-Local", "Chronos2", "ChronosBolt-Small"]


def mse(p, y, H):
    return float(((p[:, :H] - y[:, :H]) ** 2).mean())


def holm(p):
    p = np.asarray(p, float)
    order, adj, run_max = np.argsort(p), np.empty(len(p)), 0.0
    for k, idx in enumerate(order):
        run_max = max(run_max, (len(p) - k) * p[idx])
        adj[idx] = min(1.0, run_max)
    return adj


def main():
    files = sorted(TUNE_DIR.glob("*.npz"))
    if not files:
        raise SystemExit("no tuning runs in results/tuning (run import_tuning_runs.py first)")
    integ, sel, rl = [], [], []
    for fz in files:
        run = fz.stem
        tgt = run.split("__")[1]
        kind, H = kind_of(tgt), OP_H[kind_of(tgt)]
        z = np.load(fz)
        n_cfg = len(json.loads(fz.with_suffix(".config.json").read_text())["configs"]["TiDE"])
        truth, origins, preds = load_run(run)
        fold_mse = {}                                   # method -> list over folds

        def add(m, v):
            fold_mse.setdefault(m, []).append(v)

        for f in FOLDS:
            y_te, y_va = z[f"ytrue_test|{f}"].astype(np.float64), z[f"ytrue_val|{f}"].astype(np.float64)
            o_te, o_va = z[f"origins_test|{f}"].astype(int), z[f"origins_val|{f}"].astype(int)
            integ.append(dict(run=run, fold=f,
                              origins_identical=bool(np.array_equal(o_te, origins[f])),
                              truth_max_abs_diff=float(np.abs(y_te - truth[f]).max()),
                              val_before_test=bool(o_va.max() + y_va.shape[1] <= o_te.min()),
                              n_val=len(o_va), n_test=len(o_te)))
            any_best = (np.inf, None, None)
            for m in TUNED:
                cands = {k: (z[f"{m}|cfg{k}|{f}|val"].astype(np.float64), z[f"{m}|cfg{k}|{f}|test"].astype(np.float64))
                         for k in range(n_cfg) if f"{m}|cfg{k}|{f}|test" in z.files and f"{m}|cfg{k}|{f}|val" in z.files}
                vm = {k: mse(v, y_va, H) for k, (v, _) in cands.items()}
                tm = {k: mse(t, truth[f], H) for k, (_, t) in cands.items()}
                k_sel, k_orc = min(vm, key=vm.get), min(tm, key=tm.get)
                orig = mse(preds[m][f], truth[f], H)
                sel.append(dict(run=run, kind=kind, model=m, fold=f, n_configs_ok=len(cands),
                                cfg_selected=k_sel, val_mse_selected=vm[k_sel], test_mse_selected=tm[k_sel],
                                cfg_oracle_REFERENCE_ONLY=k_orc, test_mse_oracle=tm[k_orc],
                                test_mse_cfg0_rerun=tm.get(0, np.nan), test_mse_original_stored=orig))
                add(f"{m}-Original", orig)
                add(f"{m}-cfg0-rerun", tm.get(0, np.nan))
                add(f"{m}-Tuned", tm[k_sel])
                add(f"{m}-Oracle(test-selected, reference only)", tm[k_orc])
                if vm[k_sel] < any_best[0]:
                    any_best = (vm[k_sel], m, tm[k_sel])
            add("Tuned-Any", any_best[2])
            for m in REFERENCE + NAIVE:
                if m in preds:
                    add(m, mse(preds[m][f], truth[f], H))
        for m, v in fold_mse.items():
            rl.append(dict(run=run, kind=kind, h=H, method=m, mse=float(np.mean(v))))
    integ, sel, rl = pd.DataFrame(integ), pd.DataFrame(sel), pd.DataFrame(rl)
    OUT.mkdir(parents=True, exist_ok=True)
    integ.to_csv(OUT / "b09_tuning_integrity.csv", index=False)
    sel.to_csv(OUT / "b09_tuning_selection.csv", index=False)

    naive_best = rl[rl.method.isin(NAIVE)].groupby("run").mse.min().rename("naive_best")
    base = rl[rl.method == "TiDE-Original"].set_index("run").mse.rename("tide_original")
    rl = rl[~rl.method.isin(NAIVE)].join(naive_best, on="run").join(base, on="run")
    rl = pd.concat([rl, naive_best.reset_index().rename(columns={"naive_best": "mse"}).assign(
        method="Best-Naive", kind=lambda d: d.run.map(lambda r: kind_of(r.split("__")[1])),
        h=lambda d: d.kind.map(OP_H)).join(naive_best, on="run").join(base, on="run")], ignore_index=True)
    rl["ratio_to_tide_original"] = rl.mse / rl.tide_original
    rl["beats_best_naive"] = rl.mse < rl.naive_best
    rl.to_csv(OUT / "b09_tuning_run_level.csv", index=False)

    summ = rl.groupby(["kind", "method"]).agg(
        n_runs=("run", "size"), median_mse=("mse", "median"),
        median_ratio_to_tide_original=("ratio_to_tide_original", "median"),
        runs_beating_best_naive=("beats_best_naive", "sum")).reset_index()
    summ.to_csv(OUT / "b09_tuning_summary.csv", index=False)

    pairs = [(f"{m}-Tuned", f"{m}-Original") for m in TUNED] + \
            [(f"{m}-Tuned", f"{m}-cfg0-rerun") for m in TUNED] + \
            [("Tuned-Any", "TiDE-Original"), ("Tuned-Any", "Best-Naive")] + \
            [("Tuned-Any", r) for r in ["Ridge-Univariate", "Ridge-Sensors", "Chronos2"]]
    tests = []
    for kind in ("vibration", "temperature"):
        piv = rl[rl.kind == kind].pivot(index="run", columns="method", values="mse")
        block = []
        for a, b in pairs:
            if a not in piv or b not in piv:
                continue
            d = piv[[a, b]].dropna()
            p = stats.wilcoxon(d[a], d[b]).pvalue if len(d) >= 5 and (d[a] != d[b]).any() else np.nan
            block.append(dict(kind=kind, a=a, b=b, n_runs=len(d), median_ratio_a_over_b=float((d[a] / d[b]).median()),
                              runs_a_better=int((d[a] < d[b]).sum()), p_wilcoxon=p))
        ps = [x["p_wilcoxon"] for x in block]
        ok = [k for k, p in enumerate(ps) if np.isfinite(p)]
        adj = holm([ps[k] for k in ok]) if ok else []
        for k, x in enumerate(block):
            x["p_holm"] = adj[ok.index(k)] if k in ok else np.nan
        tests += block
    pd.DataFrame(tests).to_csv(OUT / "b09_tuning_tests.csv", index=False)

    print(f"[integrity] origins identical {integ.origins_identical.all()}, "
          f"truth max diff {integ.truth_max_abs_diff.max():.2e}, val before test {integ.val_before_test.all()}")
    print(summ.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
