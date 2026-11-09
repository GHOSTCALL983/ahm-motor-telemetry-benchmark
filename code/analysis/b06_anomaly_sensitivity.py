"""B6 - anomaly flags and a sensitivity analysis that excludes them.

The authors identify the motor stop and the spikes as anomalies. They are NOT
removed from the released data or from the primary analysis (removing data
after seeing that it hurts results would bias every comparison). Instead:

1. Flags are defined by fixed rules on the reference sensor's signals only,
   never on model errors:
     stopped    centred 10-min mean of self_vib_ac < 0.1 m/s^2
     vib_spike  self_vib_ac > median + 6 x robust SD (1.4826 x MAD), median and
                MAD taken over the dataset's running (not stopped) minutes
     temp_jump  |1-min change of bearing temperature| > 2 degC within a segment
   Flagged intervals are written to data/anomaly_intervals.csv.
2. Sensitivity: every forecast origin whose input or forecast window
   [o-128, o+15) touches a flagged minute is dropped, for EVERY method alike,
   and the main comparisons are recomputed on the remaining origins.
Outputs: data/anomaly_intervals.csv, results/tables/b06_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, FOLDS, INPUT_SIZE, TABLES, TRAINED_MODELS  # noqa: E402
from load import FAMILY, NAIVE, kind_of, load_run, runs  # noqa: E402
from b03_recalibration_causal import fit_apply  # noqa: E402

H = 15
OP_H = {"vibration": 3, "temperature": 15}


def flags_for(df):
    v = df.self_vib_ac
    stopped = v.rolling(10, center=True, min_periods=1).mean() < 0.1
    run_v = v[~stopped]
    med = run_v.median()
    rsd = 1.4826 * (run_v - med).abs().median()
    spike = v > med + 6 * rsd
    jump = df.groupby("segment_id").y.diff().abs() > 2
    return pd.DataFrame({"stopped": stopped, "vib_spike": spike, "temp_jump": jump.fillna(False)})


def intervals(df, fl, name):
    out = []
    ds = pd.to_datetime(df.ds)
    for col in fl.columns:
        x = fl[col].to_numpy()
        if not x.any():
            continue
        idx = np.flatnonzero(x)
        brk = np.flatnonzero(np.diff(idx) != 1)
        for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]):
            out.append(dict(dataset=name, type=col, start=str(ds.iloc[idx[a]]), end=str(ds.iloc[idx[b]]),
                            minutes=int(idx[b] - idx[a] + 1)))
    return out


def main():
    ints, masks = [], {}
    for r in (1, 2, 3):
        for i in (1, 2, 3):
            name = f"R{r}_I{i}"
            df = pd.read_csv(DATA / f"{name}.csv", usecols=["ds", "segment_id", "y", "self_vib_ac"])
            fl = flags_for(df)
            ints += intervals(df, fl, name)
            bad = fl.any(axis=1).to_numpy()
            cb = np.r_[0, np.cumsum(bad)]                 # prefix sums for fast window checks
            masks[name] = lambda o, cb=cb: (cb[o + H] - cb[o - INPUT_SIZE]) == 0
    pd.DataFrame(ints).to_csv(DATA / "anomaly_intervals.csv", index=False)

    rows, kept = [], []
    for reg, it, tgt, run in runs():
        kind = kind_of(tgt)
        h = OP_H[kind]
        truth, origins, preds = load_run(run)
        keep = {f: masks[f"{reg}_I{it}"](origins[f]) for f in FOLDS}
        kept.append(dict(run=run, origins=sum(len(origins[f]) for f in FOLDS),
                         kept=int(sum(keep[f].sum() for f in FOLDS))))
        last = {f: preds["Naive-Persistence"][f][:, :1] for f in FOLDS}
        for m, pf in preds.items():
            mse = [((pf[f][keep[f], :h] - truth[f][keep[f], :h]) ** 2).mean() for f in FOLDS if keep[f].any()]
            rows.append(dict(run=run, kind=kind, method=m, family=FAMILY.get(m, "other"), variant="raw-all-folds",
                             mse=float(np.mean(mse))))
            if FAMILY.get(m) in ("trained", "ensemble"):
                for v in ("raw", "anchored"):
                    ms = []
                    for k in range(1, len(FOLDS)):
                        past = FOLDS[:k]
                        P = np.vstack([pf[f][keep[f]] for f in past])
                        Y = np.vstack([truth[f][keep[f]] for f in past])
                        L = np.vstack([last[f][keep[f]] for f in past])
                        f = FOLDS[k]
                        if not keep[f].any() or len(P) == 0:
                            continue
                        yc = fit_apply(v, P, Y, L, pf[f][keep[f]], last[f][keep[f]])
                        ms.append(((yc[:, :h] - truth[f][keep[f], :h]) ** 2).mean())
                    rows.append(dict(run=run, kind=kind, method=m, family=FAMILY[m], variant=f"{v}-folds2to5",
                                     mse=float(np.mean(ms))))
    d = pd.DataFrame(rows)
    pd.DataFrame(kept).to_csv(TABLES / "b06_origins_kept.csv", index=False)
    raw = d[d.variant == "raw-all-folds"].copy()
    nb = raw[raw.method.isin(NAIVE)].groupby("run").mse.min()
    raw["skill_vs_best_naive"] = 1 - raw.mse / raw.run.map(nb)
    raw.to_csv(TABLES / "b06_run_level_excl_anomalies.csv", index=False)

    out = []
    for kind, g in raw.groupby("kind"):
        piv = g.pivot(index="run", columns="method", values="mse")
        med_tr = piv[TRAINED_MODELS].median(axis=1)
        beat = (piv[TRAINED_MODELS].lt(piv.index.map(nb), axis=0)).sum()
        out.append(dict(kind=kind, comparison="trained models beating best naive in majority of runs",
                        value=int((beat > len(piv) / 2).sum()), n_runs=len(piv), p=np.nan))
        for m in ("Ensemble-Mean-13", "Ridge-Univariate", "Ridge-Sensors", "ChronosBolt-Small", "Chronos2",
                  "Chronos2-Sensors"):
            if m not in piv:
                continue
            for ref_name, ref in (("TiDE", piv["TiDE"]), ("median trained model", med_tr)):
                lr = np.log(piv[m] / ref)
                out.append(dict(kind=kind, comparison=f"{m} vs {ref_name}: median MSE ratio",
                                value=float(np.exp(lr.median())), n_runs=len(lr),
                                runs_method_better=int((lr < 0).sum()), p=stats.wilcoxon(lr).pvalue))
        rc = d[(d.kind == kind) & (d.family == "trained") & d.variant.str.endswith("folds2to5")]
        rp = rc.pivot_table(index=["run", "method"], columns="variant", values="mse")
        lr = np.log(rp["anchored-folds2to5"] / rp["raw-folds2to5"]).groupby("run").mean()
        out.append(dict(kind=kind, comparison="anchored recalibration of trained models: median MSE ratio",
                        value=float(np.exp(lr.median())), n_runs=len(lr), runs_method_better=int((lr < 0).sum()),
                        p=stats.wilcoxon(lr).pvalue))
    out = pd.DataFrame(out)
    out.to_csv(TABLES / "b06_sensitivity_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(ints).groupby(["dataset", "type"]).minutes.agg(["count", "sum"]).to_string())
    k = pd.DataFrame(kept)
    print(f"\norigins kept: {k.kept.sum()} of {k.origins.sum()}; per run min {k.kept.min()}")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
