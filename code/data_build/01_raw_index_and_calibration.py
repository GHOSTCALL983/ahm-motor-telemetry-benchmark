"""Stage 1 - read every raw device export once and derive device-level facts.

Outputs (all device-level, anonymised):
  results/audit/device_raw_summary.csv   rows, span, coverage, temperature extremes
  results/audit/gravity_calibration.csv  per-device accelerometer scale k_d
A private cache of per-device frames is written OUTSIDE the release folder
(PRIVATE_CACHE), because raw timestamps are only needed to rebuild the data.

Gravity calibration: the accelerometer is DC-coupled, so the static resultant
sqrt(x^2+y^2+z^2) is dominated by gravity. k_d = median resultant over the
device's full record (raw units); samples are rescaled by G0 / k_d.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env_path, RESULTS, raw_config  # noqa: E402

TSFMT = "%a %b %d %Y %H:%M:%S GMT%z"
AX = ["vibration_x", "vibration_y", "vibration_z"]
PRIVATE_CACHE = env_path("PRIVATE_CACHE", "where to write the stage-1 cache")
PRIVATE_CACHE.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def read_raw(path):
    parts = []
    for ck in pd.read_csv(path, usecols=["temperature_one", "temperature_two"] + AX + ["timestamp"],
                          chunksize=1_000_000):
        ck["ds"] = pd.to_datetime(ck.timestamp.str.slice(0, 33).str.strip(),
                                  format=TSFMT, errors="coerce", utc=True)
        parts.append(ck.drop(columns="timestamp"))
    return pd.concat(parts, ignore_index=True)


def main():
    raw_dir, files = raw_config()
    rows, cal = [], []
    for lab, fn in files.items():
        d = read_raw(raw_dir / fn)
        n_all = len(d)
        d.to_parquet(PRIVATE_CACHE / f"{lab}.parquet")
        ok = d.dropna(subset=["ds"])
        mins = np.unique(ok.ds.values.astype("datetime64[m]").astype(np.int64))
        span_min = int(mins.max() - mins.min() + 1)
        res = np.sqrt((d[AX].astype(float) ** 2).sum(axis=1))
        res = res[np.isfinite(res)]
        q = np.percentile(res, [5, 25, 50, 75, 95])
        cal.append(dict(device=lab, n=int(res.size), med=round(float(q[2]), 4),
                        p5=round(float(q[0]), 4), p25=round(float(q[1]), 4),
                        p75=round(float(q[3]), 4), p95=round(float(q[4]), 4),
                        iqr_pct=round(float(100 * (q[3] - q[1]) / q[2]), 2)))
        # repeated-sample share per axis (consecutive identical readings)
        rep = {a: round(float(100 * (ok[a].diff() == 0).mean()), 2) for a in AX}
        rows.append(dict(device=lab, raw_rows=n_all, parsed_rows=len(ok),
                         first=str(ok.ds.min()), last=str(ok.ds.max()),
                         distinct_minutes=int(mins.size), span_minutes=span_min,
                         coverage_of_own_span_pct=round(100 * mins.size / span_min, 1),
                         t1_raw_max=float(ok.temperature_one.max()),
                         t2_raw_max=float(ok.temperature_two.max()),
                         **{f"repeat_pct_{a[-1]}": v for a, v in rep.items()}))
        np.save(PRIVATE_CACHE / f"{lab}_minutes.npy", mins)
        print(f"{lab}: {n_all:>9,} rows, {mins.size:>7,} distinct minutes, k_d={q[2]:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "device_raw_summary.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "gravity_calibration.csv", index=False)
    print("[ok] stage 1 written")


if __name__ == "__main__":
    main()
