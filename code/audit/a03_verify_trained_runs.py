"""Audit 3 - integrity of the 45 trained-model runs against the released data.

For every run and fold:
  * stored ground truth == target column of the released file at the stored origins
  * every origin lies in its fold's test block and its full window
    [o - 128, o + 15) inside one gap-free segment (no window spans a gap)
  * test blocks of different folds do not overlap
  * which models are present, NaN predictions, number of origins
  * MSE recomputed from stored predictions == MSE in the run's own metrics file
Writes results/audit/a03_trained_runs.csv (one row per run).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (env_path, DATA, FOLD_FRAC, FOLDS, INPUT_SIZE, PRED_DIR, RESULTS,  # noqa: E402
                    TARGETS, TRAINED_MODELS)

H = 15
ORIG = env_path("ORIGINAL_ROOT", "original project, for comparison") / "study_canonical"


def main():
    out = []
    for r in (1, 2, 3):
        for i in (1, 2, 3):
            df = pd.read_csv(DATA / f"R{r}_I{i}.csv")
            N = len(df)
            seg = df.segment_id.to_numpy()
            for t in TARGETS:
                col = "y" if t == "self_temperature_one" else t
                yv = df[col].to_numpy(float)
                z = np.load(PRED_DIR / f"R{r}_I{i}__{t}.npz")
                models = sorted({k.split("|")[0] for k in z.files if k.count("|") == 2})
                truth_ok, window_ok, in_block, n_or, nan_cells = True, True, True, [], 0
                blocks = []
                mse_new = {}
                for f in FOLDS:
                    o = z[f"origins|{f}"].astype(int)
                    yt = z[f"ytrue|{f}"].astype(float)
                    n_or.append(len(o))
                    truth = np.stack([yv[k:k + H] for k in o])
                    truth_ok &= bool(np.allclose(truth, yt, atol=1e-3))
                    _, ve, te = (int(N * x) for x in FOLD_FRAC[f])
                    in_block &= bool((o >= ve).all() and (o + H <= te).all())
                    window_ok &= bool(((seg[o - INPUT_SIZE] == seg[o + H - 1]) &
                                       (seg[o] == seg[o - INPUT_SIZE])).all())
                    blocks.append((ve, te))
                    for m in models:
                        p = z[f"{m}|{f}|0.0"].astype(float)
                        nan_cells += int(np.isnan(p).sum())
                        mse_new.setdefault(m, []).append(float(np.nanmean((p - yt) ** 2)))
                blocks.sort()
                disjoint = all(blocks[k][1] <= blocks[k + 1][0] for k in range(len(blocks) - 1))
                mm = pd.read_csv(ORIG / f"Iteration_{i}" / f"Region_{r}" / t / "results" / "metrics_main.csv")
                mm = mm[mm.noise_sigma == 0.0].groupby("model").MSE.apply(list)
                mse_dev = max(abs(a - b) / max(abs(b), 1e-9)
                              for m in models if m in mm.index
                              for a, b in zip(mse_new[m], mm[m]))
                out.append(dict(run=f"R{r}_I{i}__{t}", n_models=len(models),
                                all_13_trained=set(TRAINED_MODELS) <= set(models),
                                baselines=",".join(m for m in models if m.startswith("Naive")),
                                origins_per_fold=",".join(map(str, n_or)), truth_matches_data=truth_ok,
                                origins_in_test_block=in_block, windows_within_segment=window_ok,
                                test_blocks_disjoint=disjoint, nan_prediction_cells=nan_cells,
                                max_rel_mse_dev_vs_logged=round(mse_dev, 6)))
    d = pd.DataFrame(out)
    d.to_csv(RESULTS / "audit" / "a03_trained_runs.csv", index=False)
    print(d.to_string(index=False))
    print("\nALL CHECKS:", {c: bool(d[c].all()) for c in
                            ["all_13_trained", "truth_matches_data", "origins_in_test_block",
                             "windows_within_segment", "test_blocks_disjoint"]},
          "| total NaN cells:", int(d.nan_prediction_cells.sum()),
          "| max MSE deviation vs logged:", d.max_rel_mse_dev_vs_logged.max())


if __name__ == "__main__":
    main()
