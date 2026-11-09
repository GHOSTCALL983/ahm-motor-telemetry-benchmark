"""Audit 5 - checks that need the site-confidential raw index (stage-1 private cache).

Region 3 configuration vs usable minutes (minutes inside runs of >= 466
consecutive minutes in which every listed device reports), whole record.
Writes results/audit/a05_region3_configurations.csv (anonymised labels only).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env_path, NEED_RUN, RESULTS  # noqa: E402

CACHE = env_path("PRIVATE_CACHE", "stage-1 cache of the raw export")


def usable(devs):
    a = np.load(CACHE / f"{devs[0]}_minutes.npy")
    for d in devs[1:]:
        a = np.intersect1d(a, np.load(CACHE / f"{d}_minutes.npy"), assume_unique=True)
    brk = np.flatnonzero(np.diff(a) != 1)
    ln = np.r_[brk + 1, len(a)] - np.r_[0, brk + 1]
    return int(ln[ln >= NEED_RUN].sum())


if __name__ == "__main__":
    cfgs = [["AHM-08", "AHM-07", "AHM-09"], ["AHM-08", "AHM-07"], ["AHM-07", "AHM-09"], ["AHM-08", "AHM-09"]]
    d = pd.DataFrame([dict(configuration="+".join(c), usable_minutes=usable(c)) for c in cfgs])
    full = d.loc[d.configuration == "AHM-08+AHM-07", "usable_minutes"].iloc[0]
    d["loss_vs_retained_pair_pct"] = (100 * (1 - d.usable_minutes / full)).round(1)
    d.to_csv(RESULTS / "audit" / "a05_region3_configurations.csv", index=False)
    print(d.to_string(index=False))
