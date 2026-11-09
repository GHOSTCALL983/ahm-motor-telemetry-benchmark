"""Audit 4 - floor-aware training, cross-domain transfer, weight merging.

Floor-aware training (iteration 1 only, 26 targets, device-named original
study): MSE recomputed from stored predictions for the plain and floor-aware
variants of TiDE and TSMixerx. Unit of the test = target (26), operating
horizon (vibration 3, temperature 15). Device ids are replaced by AHM labels.
Known protocol issue (disclosed): the variance floor used lambda computed on
the FULL record, i.e. including test blocks.

Transfer matrix (TiDE, TimeXer; fold F5; one model per source domain): copied
with neutral domain names; marginal means by shift type recomputed; per-target
ratio of each off-diagonal cell to that target's in-domain cell; and a split
into forward-in-time vs backward-in-time temporal transfer.

Weight merging: weights were chosen by grid search on the TEST predictions
(pooled over the three test domains). Only the three un-merged checkpoints'
MSE (M1, M2, M3) are free of that selection; merged rows are flagged.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env_path, RESULTS  # noqa: E402

ORIG = env_path("ORIGINAL_ROOT", "original project, for comparison")
AUD, TR = RESULTS / "audit", RESULTS / "transfer"
DEV = {"s001": "AHM-01", "s005": "AHM-02", "s010": "AHM-03", "s003": "AHM-04", "s004": "AHM-05",
       "s006": "AHM-06", "s007": "AHM-07", "s008": "AHM-08"}
OP_H = {"vibration": 3, "temperature": 15}
neutral = lambda s: re.sub(r"^[A-Z]+_(R\d_I\d)$", r"\1", str(s))  # noqa: E731


def floor():
    rows = []
    st = ORIG / "study" / "Iteration_1"
    for reg_dir in sorted(st.glob("Region_*")):
        for tdir in sorted(reg_dir.iterdir()):
            base = tdir / "results" / "predictions.npz"
            if not base.exists():
                continue
            dev, ch = tdir.name.split("_", 1)
            kind = "temperature" if "temperature" in ch else "vibration"
            H = OP_H[kind]
            zb = np.load(base)
            for fm, folder in (("TSMixerx-Floor", "results_floor"), ("TiDE-Floor", "results_floor"),
                               ("TiDE-Floor", "results_floor_tide_ext")):
                p = tdir / folder / "predictions.npz"
                if not p.exists():
                    continue
                zf = np.load(p)
                folds = sorted({k.split("|")[1] for k in zf.files if k.startswith(fm + "|")})
                if not folds:
                    continue
                bm = fm.replace("-Floor", "")
                mb, mf = [], []
                for f in folds:
                    yt = zb[f"ytrue|{f}"][:, :H].astype(float)
                    assert np.allclose(yt, zf[f"ytrue|{f}"][:, :H]), "ground truth differs"
                    mb.append(((zb[f"{bm}|{f}|0.0"][:, :H] - yt) ** 2).mean())
                    mf.append(((zf[f"{fm}|{f}|0.0"][:, :H] - yt) ** 2).mean())
                rows.append(dict(region=reg_dir.name, device=DEV.get(dev, dev), channel=ch, kind=kind,
                                 model=bm, h=H, n_folds=len(folds), mse_plain=np.mean(mb),
                                 mse_floor=np.mean(mf)))
    d = pd.DataFrame(rows).drop_duplicates(["region", "device", "channel", "model"])
    d["delta_pct"] = 100 * (d.mse_floor / d.mse_plain - 1)
    d.to_csv(AUD / "a04_floor_recomputed.csv", index=False)
    out = []
    for m, g in d.groupby("model"):
        out.append(dict(model=m, n_targets=len(g), median_delta_pct=g.delta_pct.median(),
                        mean_delta_pct=g.delta_pct.mean(), targets_improved=int((g.delta_pct < 0).sum()),
                        p_wilcoxon=stats.wilcoxon(np.log(g.mse_floor / g.mse_plain)).pvalue))
    return pd.DataFrame(out)


def transfer():
    TR.mkdir(parents=True, exist_ok=True)
    res = []
    for fn, ch in (("TRANSFER_MATRIX.csv", "self_x_ac"), ("TRANSFER_MATRIX_TEMPERATURE.csv", "self_temperature_one")):
        t = pd.read_csv(ORIG / "study_p4" / fn)
        t["source"], t["target"] = t.source.map(neutral), t.target.map(neutral)
        t.to_csv(TR / f"transfer_matrix_{ch}.csv", index=False)
        it = lambda s: int(s[-1])  # noqa: E731
        diag = t[t["shift"] == "diagonal"].set_index(["model", "target"]).MSE
        t["ratio_vs_in_domain"] = t.MSE / t.set_index(["model", "target"]).index.map(diag).values
        t["direction"] = np.where(t.source.map(it) < t.target.map(it), "forward",
                                  np.where(t.source.map(it) > t.target.map(it), "backward", "same-period"))
        for excl in (False, True):
            g = t[t.target != "R2_I3"] if excl else t
            for (m, sh), gg in g.groupby(["model", "shift"]):
                res.append(dict(channel=ch, model=m, shift=sh, excl_R2_I3_target=excl, cells=len(gg),
                                mean_mse=gg.MSE.mean(), median_ratio_vs_in_domain=gg.ratio_vs_in_domain.median()))
        for (m, dr), gg in t[t["shift"].isin(["temporal-only", "both"])].groupby(["model", "direction"]):
            res.append(dict(channel=ch, model=m, shift=f"temporal ({dr})", excl_R2_I3_target=False,
                            cells=len(gg), mean_mse=gg.MSE.mean(),
                            median_ratio_vs_in_domain=gg.ratio_vs_in_domain.median()))
    r = pd.DataFrame(res)
    r.to_csv(AUD / "a04_transfer_summary.csv", index=False)
    return r


def merge():
    m = pd.read_csv(ORIG / "study_p4" / "MERGE_RESULTS.csv")
    m["selection_on_test"] = ~m.state.isin(["M1", "M2", "M3"])
    m.to_csv(TR / "merge_results_FLAGGED.csv", index=False)
    return m[["state", "weights", "MSE", "selection_on_test"]]


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    f = floor()
    f.to_csv(AUD / "a04_floor_summary.csv", index=False)
    print(f.round(4).to_string(index=False))
    print(transfer().round(5).to_string(index=False))
    print(merge().to_string(index=False))
