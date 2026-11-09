"""A8 - the pilot that removed LSTM, GRU and FEDformer from the roster.

Before the main study, a pilot trained 10 models on bearing temperature (y) of
the three iteration-1 datasets with the EARLIER split scheme (S1-S5: train /
validation / test fractions 60/20/20, 70/15/15, 80/10/10, 75/15/10, 90/5/5, so
the test blocks overlap), seed 42, 80 origins per split. LSTM, GRU and FEDformer
were dropped after it; TabNet was kept.

This script recomputes the pilot ratio for each model from the stored pilot
forecasts: mean MSE over the 5 splits (15-step horizon, no input noise) divided
by the same for persistence, per region, and the mean over the 3 regions.
Source (not in the release): ORIGINAL_ROOT/study/_archive/kaggle_raw_results/
Output: results/audit/a08_pilot_roster.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, env_path  # noqa: E402

SRC = env_path("ORIGINAL_ROOT", "original project holding the pilot results") / "study" / "_archive" / "kaggle_raw_results"
SPLITS = ["S1_60_20_20", "S2_70_15_15", "S3_80_10_10", "S4_75_15_10", "S5_90_05_05"]


def main():
    rows = []
    for r in (1, 2, 3):
        src = next(SRC.glob(f"*R{r}_I1")) / "predictions.npz"
        z = np.load(src)
        models = sorted({k.split("|")[0] for k in z.files} - {"origins", "ytrue"})

        def mse(m, s):
            return float(np.mean((z[f"{m}|{s}|0.0"].astype(float) - z[f"ytrue|{s}"].astype(float)) ** 2))
        pers = np.mean([mse("Naive-Persistence", s) for s in SPLITS])
        for m in models:
            v = np.mean([mse(m, s) for s in SPLITS])
            rows.append(dict(region=f"R{r}", model=m, mse_over_persistence=v / pers if np.isfinite(v) else np.nan))
    d = pd.DataFrame(rows)
    wide = d.pivot(index="model", columns="region", values="mse_over_persistence")
    wide["mean_3_regions"] = wide[["R1", "R2", "R3"]].mean(axis=1, skipna=False)
    wide = wide.sort_values("mean_3_regions").reset_index()
    wide.to_csv(RESULTS / "audit" / "a08_pilot_roster.csv", index=False)
    print(wide.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
