"""Stage 2 - choose the three time windows and build the nine released datasets.

Faithful re-implementation of the original pipeline, with anonymised labels.

1. Windows. A forecast window needs NEED_RUN consecutive minutes in which every
   sensor of a region reports. Three non-overlapping windows of L days, common
   to all regions, are chosen to maximise the WORST region's usable minutes
   (ties broken by total). Criterion = sensor coverage only; no model output is
   consulted.
2. Per-minute features. PT100 readings are corrected for the firmware nominal-
   resistance error (97 -> 100 ohm, exact Callendar-Van Dusen re-solve), out-of-
   envelope temperatures are removed, accelerations are scaled to m/s^2 with
   the per-device gravity constant from stage 1. Vibration AC amplitude per axis
   = trailing 3-min standard deviation of raw samples. All rolling features are
   trailing (causal) and computed within gap-free segments.
3. Role-canonical columns (self / nb1 / nb2) so all regions share one schema.
4. One collinearity-pruning decision (|r| > 0.98) on the pooled first 45% of
   rows of all nine datasets; raw target-candidate channels are never pruned.

Outputs: data/R{r}_I{i}.csv, data/schema.json, data/manifest.csv,
         results/audit/iteration_windows.csv
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (env_path, DATA, G0, MIN_SEG, NEED_RUN, NOM_FIRMWARE, NOM_TRUE, PRUNE_BASIS,  # noqa: E402
                    PRUNE_THR, REGIONS, RESULTS, RTD_A, RTD_B, TLIM, INPUT_SIZE)

PRIVATE_CACHE = env_path("PRIVATE_CACHE", "stage-1 cache of the raw export")
AX = ["vibration_x", "vibration_y", "vibration_z"]
_Z1, _Z2, _Z4 = -RTD_A, RTD_A ** 2 - 4 * RTD_B, 2 * RTD_B
HORIZON_ORIGINS = 18          # origin count convention of the original manifest
LENGTHS = range(8, 28)


def pt100_fix(t):
    """Temperature reported with nominal 97 ohm -> temperature with nominal 100 ohm.
    Exact for t > 0 degC (the library uses another branch below 0; such samples are
    treated as faults and removed)."""
    t = np.asarray(t, dtype=float)
    rt = ((t * _Z4 - _Z1) ** 2 - _Z2) / (4 * RTD_B / NOM_FIRMWARE)
    out = (_Z1 + np.sqrt(_Z2 + (4 * RTD_B / NOM_TRUE) * rt)) / _Z4
    return np.where(t > 0, out, np.nan)


# ------------------------------------------------------------------ windows
def region_runs(devs):
    a = np.load(PRIVATE_CACHE / f"{devs[0]}_minutes.npy")
    for d in devs[1:]:
        a = np.intersect1d(a, np.load(PRIVATE_CACHE / f"{d}_minutes.npy"), assume_unique=True)
    brk = np.flatnonzero(np.diff(a) != 1)
    st = np.concatenate(([0], brk + 1))
    en = np.concatenate((brk + 1, [len(a)]))
    return np.stack([a[st], a[en - 1] + 1], 1)


def choose_windows():
    devs = {r: [c["self"], c["nb1"]] + ([c["nb2"]] if c["nb2"] else []) for r, c in REGIONS.items()}
    runs = {r: region_runs(v) for r, v in devs.items()}

    def val(r, w0, w1):
        R = runs[r]
        ov = np.minimum(R[:, 1], w1) - np.maximum(R[:, 0], w0)
        return int(ov[ov >= NEED_RUN].sum())

    lo = max(runs[r][0, 0] for r in REGIONS)
    hi = min(runs[r][-1, 1] for r in REGIONS)
    d0 = lo // 1440
    nd = int(hi // 1440 - d0) + 1
    best = None
    for L in LENGTHS:
        if nd < 3 * L:
            continue
        v = {r: np.array([val(r, int((d0 + s) * 1440), int((d0 + s + L) * 1440))
                          for s in range(nd - L + 1)]) for r in REGIONS}
        mn = np.minimum.reduce([v[r] for r in REGIONS])
        a, b, sel = 0, int(mn.max()), None
        while a <= b:                       # largest T such that 3 windows reach T
            T = (a + b) // 2
            s2, i = [], 0
            while i < len(mn) and len(s2) < 3:
                if mn[i] >= T:
                    s2.append(i)
                    i += L
                else:
                    i += 1
            if len(s2) == 3:
                sel, a = (T, s2), T + 1
            else:
                b = T - 1
        if sel is None:
            continue
        T, s2 = sel
        tot = sum(int(v[r][s]) for r in REGIONS for s in s2)
        if best is None or (T, tot) > best[0]:
            best = ((T, tot), L, s2)
    (_, _), L, s2 = best
    rows = []
    for i, s in enumerate(s2, 1):
        w0, w1 = int((d0 + s) * 1440), int((d0 + s + L) * 1440)
        for r in REGIONS:
            u = val(r, w0, w1)
            rows.append(dict(region=r, iteration=i, devices=",".join(devs[r]),
                             start=pd.Timestamp(w0, unit="m").strftime("%Y-%m-%dT%H:%M:%SZ"),
                             end=pd.Timestamp(w1, unit="m").strftime("%Y-%m-%dT%H:%M:%SZ"),
                             days=L, usable_min=u, usable_pct=round(100 * u / (L * 1440), 1)))
    return pd.DataFrame(rows).sort_values(["iteration", "region"]).reset_index(drop=True)


# ------------------------------------------------------------------ features
_RAW = {}


def load(dev, k):
    if dev not in _RAW:
        d = pd.read_parquet(PRIVATE_CACHE / f"{dev}.parquet").dropna(subset=["ds"])
        d = d.drop_duplicates(subset=["ds"]).sort_values("ds").set_index("ds")
        for c in ("temperature_one", "temperature_two"):
            d[c] = pt100_fix(d[c])
            d.loc[(d[c] < TLIM[0]) | (d[c] > TLIM[1]), c] = np.nan
        d[AX] = d[AX] * (G0 / k)
        _RAW[dev] = d
    return _RAW[dev]


def per_minute(dev, k, t0, t1, role):
    d = load(dev, k).loc[t0:t1]
    g = d.resample("1min")
    o = pd.DataFrame(index=g["temperature_one"].mean().index)
    for c in ("temperature_one", "temperature_two"):
        o[f"{role}__{c}"] = g[c].mean()
    r = d[AX].rolling("3min", min_periods=3).std(ddof=1).resample("1min").last()
    for a in AX:
        o[f"{role}__{a[-1]}_ac"] = r[a]
    o[f"{role}__vib_ac"] = np.sqrt((r ** 2).sum(axis=1))
    o[f"{role}__vib_dc"] = np.sqrt((g[AX].mean() ** 2).sum(axis=1))
    return o


def build_raw(region, it, t0, t1, kd):
    c = REGIONS[region]
    roles = [("self", c["self"]), ("nb1", c["nb1"])] + ([("nb2", c["nb2"])] if c["nb2"] else [])
    parts = [per_minute(dev, kd[dev], t0, t1, role) for role, dev in roles]
    f = parts[0].join(parts[1:], how="inner").dropna()
    brk = np.flatnonzero(np.diff(f.index.values).astype("timedelta64[m]").astype(int) != 1)
    seg = np.zeros(len(f), int)
    seg[brk + 1] = 1
    f["segment_id"] = seg.cumsum()
    sz = f.groupby("segment_id").size()
    f = f[f.segment_id.isin(sz[sz >= MIN_SEG].index)].copy()
    f["segment_id"] = f.groupby("segment_id").ngroup()
    f["y"] = f["self__temperature_one"]
    G = f.groupby("segment_id", group_keys=False)

    def w(col, fn):
        return G[col].transform(fn)

    for n in (15, 60, 240):
        f[f"y_roll_mean_{n}"] = w("y", lambda s: s.rolling(n, min_periods=n).mean())
        f[f"y_roll_std_{n}"] = w("y", lambda s: s.rolling(n, min_periods=n).std())
    for k2 in (1, 5, 15, 60):
        f[f"y_diff_{k2}"] = w("y", lambda s: s.diff(k2))
    f["y_roll_iqr_60"] = w("y", lambda s: (s.rolling(60, min_periods=60).quantile(.75)
                                           - s.rolling(60, min_periods=60).quantile(.25)))
    for sp in (15, 60):
        f[f"y_ewm_{sp}"] = w("y", lambda s: s.ewm(span=sp, adjust=False).mean())
    for role, _ in roles:
        p = f"{role}__"
        f[f"{role}_temperature_one"] = f[p + "temperature_one"]
        f[f"{role}_temperature_two"] = f[p + "temperature_two"]
        f[f"{role}_t_grad"] = f[p + "temperature_one"] - f[p + "temperature_two"]
        f[f"{role}_t1_diff_15"] = w(p + "temperature_one", lambda x: x.diff(15))
        for a in ("x", "y", "z"):
            src = f"{p}{a}_ac"
            f[f"{role}_{a}_ac"] = f[src]
            f[f"{role}_{a}_ac_mean_60"] = w(src, lambda x: x.rolling(60, min_periods=60).mean())
            f[f"{role}_{a}_ac_std_60"] = w(src, lambda x: x.rolling(60, min_periods=60).std())
        f[f"{role}_vib_ac"] = f[p + "vib_ac"]
        f[f"{role}_vib_dc"] = f[p + "vib_dc"]
        f[f"{role}_vib_ac_ewm_60"] = w(p + "vib_ac", lambda x: x.ewm(span=60, adjust=False).mean())
    if not c["nb2"]:                                 # single-neighbour region: zero-fill nb2
        nb2 = ["nb2_temperature_one", "nb2_temperature_two", "nb2_t_grad", "nb2_t1_diff_15",
               "nb2_vib_ac", "nb2_vib_dc", "nb2_vib_ac_ewm_60"]
        nb2 += [f"nb2_{a}_ac{s}" for a in ("x", "y", "z") for s in ("", "_mean_60", "_std_60")]
        for col in nb2:
            f[col] = 0.0
        f["nb2_present"] = 0
        f["d_self_nb2"] = 0.0
    else:
        f["nb2_present"] = 1
        f["d_self_nb2"] = f["self_temperature_one"] - f["nb2_temperature_one"]
    f["d_self_nb1"] = f["self_temperature_one"] - f["nb1_temperature_one"]
    idx = f.index
    f["tod_sin"] = np.sin(2 * np.pi * (idx.hour * 60 + idx.minute) / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * (idx.hour * 60 + idx.minute) / 1440)
    f = f.dropna()
    sz = f.groupby("segment_id").size()
    f = f[f.segment_id.isin(sz[sz >= MIN_SEG].index)].copy()
    f["segment_id"] = f.groupby("segment_id").ngroup()
    f = f.reset_index().rename(columns={"index": "ds"})
    f["unique_id"] = f"{region}_I{it}"
    exog = [col for col in f.columns if col not in ("unique_id", "ds", "y", "segment_id")
            and "__" not in col]
    return f[["unique_id", "ds", "segment_id", "y"] + exog]


def main():
    out_audit = RESULTS / "audit"
    win = choose_windows()
    win.to_csv(out_audit / "iteration_windows.csv", index=False)
    print(win.to_string(index=False), flush=True)
    kd = pd.read_csv(out_audit / "gravity_calibration.csv").set_index("device")["med"].to_dict()

    frames = {}
    for _, r in win.iterrows():
        frames[(r.region, int(r.iteration))] = build_raw(r.region, int(r.iteration),
                                                         pd.Timestamp(r.start), pd.Timestamp(r.end), kd)
    protected = {f"{ro}_{ch}" for ro in ("self", "nb1", "nb2")
                 for ch in ("temperature_one", "temperature_two", "x_ac", "y_ac", "z_ac")}
    exog_cols = [c for c in next(iter(frames.values())).columns
                 if c not in ("unique_id", "ds", "segment_id", "y")]
    pooled = pd.concat([f[exog_cols].iloc[:int(len(f) * PRUNE_BASIS)] for f in frames.values()],
                       ignore_index=True)
    cm = pooled.corr().abs()
    kept, drop = [], []
    for c in exog_cols:
        if c in protected:
            kept.append(c)
            continue
        (drop if any(cm.loc[c, k] > PRUNE_THR for k in kept) else kept).append(c)
    DATA.mkdir(parents=True, exist_ok=True)
    recs = []
    for (reg, it), f in frames.items():
        final = f[["unique_id", "ds", "segment_id", "y"] + kept]
        final.to_csv(DATA / f"{reg}_I{it}.csv", index=False)
        sz = final.groupby("segment_id").size()
        recs.append(dict(dataset=f"{reg}_I{it}", region=reg, zone=REGIONS[reg]["zone"], iteration=it,
                         rows=len(final), segments=int(final.segment_id.nunique()),
                         seg_min=int(sz.min()), seg_max=int(sz.max()), features=len(kept),
                         origins_capacity=int((sz - INPUT_SIZE - HORIZON_ORIGINS + 1).clip(lower=0).sum()),
                         start=str(final.ds.min()), end=str(final.ds.max()),
                         y_mean=round(float(final.y.mean()), 4), y_std=round(float(final.y.std()), 4),
                         y_min=round(float(final.y.min()), 4), y_max=round(float(final.y.max()), 4)))
    pd.DataFrame(recs).to_csv(DATA / "manifest.csv", index=False)
    (DATA / "schema.json").write_text(json.dumps(dict(
        kept_columns=kept, pruned_columns=drop, prune_basis=PRUNE_BASIS, prune_threshold=PRUNE_THR,
        protected_columns=sorted(protected), target_column_in_files="y = self_temperature_one"), indent=2))
    print(f"[ok] kept {len(kept)}, pruned {len(drop)}: {drop}")


if __name__ == "__main__":
    main()
