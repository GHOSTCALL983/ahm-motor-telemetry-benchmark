"""B4 - Murphy (1988) decomposition of MSE for every method, all 45 runs.

For truth t and forecast p (steps 1..H pooled within a fold):
    MSE = (mean_p - mean_t)^2 + (sd_p - r sd_t)^2 + sd_t^2 (1 - r^2)
          bias^2                conditional bias      unexplained
The identity is asserted on every cell. sd_ratio/r = 1 is the condition for a
zero conditional-bias term (Murphy 1988; Taylor 2001). Reported per fold and
averaged over folds; overconfidence is summarised per method as the median
across runs of the fold-averaged sd_ratio/r.

Also decomposes the anchored-recalibrated trained forecasts (folds 2..5, from
B3's protocol) to show which term the anchoring removes.
Outputs: results/tables/b04_decomposition.csv, b04_overconfidence_by_model.csv,
         b04_anchor_effect.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, TABLES  # noqa: E402
from load import FAMILY, kind_of, load_run, runs  # noqa: E402
from b03_recalibration_causal import fit_apply  # noqa: E402

OP_H = {"vibration": 3, "temperature": 15}


def decompose(t, p):
    t, p = np.asarray(t, float).ravel(), np.asarray(p, float).ravel()
    mt, mp, st, sp = t.mean(), p.mean(), t.std(), p.std()
    r = float(np.corrcoef(t, p)[0, 1]) if sp > 1e-12 and st > 1e-12 else 0.0
    b2, amp, ph = (mp - mt) ** 2, (sp - r * st) ** 2, st ** 2 * (1 - r ** 2)
    mse = float(((p - t) ** 2).mean())
    assert abs(b2 + amp + ph - mse) <= 1e-9 * max(b2, amp, ph, mse, 1.0), "identity broke"
    return dict(mse=mse, bias2=b2, cond_bias=amp, unexplained=ph, r=r,
                sd_ratio=sp / st if st > 0 else np.nan)


def main():
    rows, anc = [], []
    for reg, it, tgt, run in runs():
        kind = kind_of(tgt)
        H = OP_H[kind]
        truth, _, preds = load_run(run)
        last = {f: preds["Naive-Persistence"][f][:, :1] for f in FOLDS}
        for m, pf in preds.items():
            acc = [decompose(truth[f][:, :H], pf[f][:, :H]) for f in FOLDS]
            d = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
            d["overconf"] = float(np.mean([a["sd_ratio"] / a["r"] if a["r"] > 0.05 else np.nan
                                           for a in acc]))
            rows.append(dict(run=run, kind=kind, method=m, family=FAMILY.get(m, "other"), h=H, **d))
            if FAMILY.get(m) == "trained":
                for v in ("raw", "anchored"):
                    acc = []
                    for k in range(1, len(FOLDS)):
                        P = np.vstack([pf[f] for f in FOLDS[:k]])
                        Y = np.vstack([truth[f] for f in FOLDS[:k]])
                        L = np.vstack([last[f] for f in FOLDS[:k]])
                        f = FOLDS[k]
                        yc = fit_apply(v, P, Y, L, pf[f], last[f])
                        acc.append(decompose(truth[f][:, :H], yc[:, :H]))
                    anc.append(dict(run=run, kind=kind, method=m, variant=v,
                                    **{k: float(np.mean([a[k] for a in acc])) for k in acc[0]}))
    d = pd.DataFrame(rows)
    for c in ("bias2", "cond_bias", "unexplained"):
        d[c + "_pct"] = 100 * d[c] / d.mse
    d.to_csv(TABLES / "b04_decomposition.csv", index=False)
    oc = d.groupby(["kind", "family", "method"]).agg(
        median_overconf=("overconf", "median"), median_r=("r", "median"),
        median_bias_pct=("bias2_pct", "median"), median_cond_bias_pct=("cond_bias_pct", "median"),
        median_unexplained_pct=("unexplained_pct", "median")).reset_index().sort_values(["kind", "median_overconf"])
    oc.to_csv(TABLES / "b04_overconfidence_by_model.csv", index=False)
    a = pd.DataFrame(anc)
    a.to_csv(TABLES / "b04_anchor_effect.csv", index=False)
    s = a.groupby(["kind", "variant"])[["mse", "bias2", "cond_bias", "unexplained", "r"]].median()
    pd.set_option("display.width", 220)
    print(oc.round(3).to_string(index=False))
    print("\nmedian over (run x trained model), folds 2-5:\n", s.to_string())


if __name__ == "__main__":
    main()
