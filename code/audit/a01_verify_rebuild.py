"""Audit 1 - does the clean rebuild reproduce the datasets the models were trained on?

Compares every released file in data/ with the original training files (path
given by ORIGINAL_DATA_DIR; not part of the release). Checks identical shape,
column names/order, timestamps, segment ids, and values (max abs difference).
Also checks the iteration-boundary overlap and segment contiguity.
Writes results/audit/a01_rebuild_check.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env_path, DATA, RESULTS  # noqa: E402

ORIG = env_path("ORIGINAL_ROOT", "original project, for comparison") / "data_canonical"


def main():
    rows = []
    last_ds = {}
    for r in ("R1", "R2", "R3"):
        for i in (1, 2, 3):
            new = pd.read_csv(DATA / f"{r}_I{i}.csv")
            old_p = next(ORIG.glob(f"Iteration_{i}/Region_{r[-1]}/*_R{r[-1]}_I{i}.csv"))
            old = pd.read_csv(old_p)
            same_cols = list(new.columns) == list(old.columns)
            same_shape = new.shape == old.shape
            num = [c for c in new.columns if c not in ("unique_id", "ds")]
            maxdiff = float(np.nanmax(np.abs(new[num].to_numpy(float) - old[num].to_numpy(float)))) \
                if same_shape and same_cols else np.nan
            same_ds = bool((new.ds.values == old.ds.values).all()) if same_shape else False
            ds = pd.to_datetime(new.ds)
            gaps = int(sum((g.diff().dropna() != pd.Timedelta("1min")).sum()
                           for _, g in ds.groupby(new.segment_id)))
            rows.append(dict(dataset=f"{r}_I{i}", rows=len(new), cols=new.shape[1],
                             same_shape=same_shape, same_columns=same_cols, same_timestamps=same_ds,
                             max_abs_value_diff=maxdiff, within_segment_gaps=gaps,
                             first=str(ds.min()), last=str(ds.max())))
            last_ds[(r, i)] = (ds.min(), ds.max())
    out = pd.DataFrame(rows)
    # boundary overlap between consecutive iterations of the same region
    ov = []
    for r in ("R1", "R2", "R3"):
        for i in (1, 2):
            ov.append(f"{r} I{i}->I{i+1}: last={last_ds[(r, i)][1]} first_next={last_ds[(r, i + 1)][0]} "
                      f"overlap={last_ds[(r, i)][1] >= last_ds[(r, i + 1)][0]}")
    out.to_csv(RESULTS / "audit" / "a01_rebuild_check.csv", index=False)
    print(out.to_string(index=False))
    print("\n".join(ov))
    print("\nTOTAL rows:", int(out.rows.sum()))


if __name__ == "__main__":
    main()
