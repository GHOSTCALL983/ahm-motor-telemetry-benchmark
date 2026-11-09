# RELEASE NOTE: this is the script exactly as it was run to produce the stored
# results, with site/person identifiers redacted (plant name, dataset acronym,
# hosting-account names, local paths, internal series ids). Known issues found in
# the audit are listed in AUDIT_REPORT.md; the code itself is unchanged.
"""P4 (PAPER_STRATEGY.md section 6.3, 6.4): cross-domain transfer matrix +
weight merging, using the 18 base checkpoints and the Region_2 M1/M2/M3
finetune chain trained by p4_transfer_train.py.

Runs as a Kaggle kernel (needs neuralforecast/torch; this project's local
Python 3.14 environment cannot install a working neuralforecast/pytorch-
lightning stack, confirmed this session) with two dataset attachments:
released-dataset (the 9 raw domain CSVs) and transfer-checkpoints (all
20 trained checkpoint bundles, zipped per-domain).

Part 1 -- 81-cell transfer matrix (section 6.3)
    For each of the 18 base checkpoints (source domain x {TiDE, TimeXer}),
    evaluate zero-shot (no adaptation) against all 9 canonical domains' F5
    test windows. Preprocessing: standardize the TARGET domain's raw data
    using the SOURCE domain's own train-set mu/sd (from that checkpoint's
    stats.json) -- this is the same choice made and disclosed in
    p4_transfer_train.py's zero-shot verification step, now applied
    systematically to the full matrix. Predictions destandardize back to
    raw units with the same source stats, scored against the target
    domain's raw (never-touched) ground truth.

    9 sources x 9 targets x 2 models = 162 cells. Classified as:
      diagonal        (9)  source == target                -> in-domain
      temporal-only   (18) same region, different iteration -> retention
      spatial-only    (18) different region, same iteration -> transfer
      both            (36) different region AND iteration   -> hardest case

Part 2 -- weight merging (section 6.4), Region_2/TiDE only
    M1, M2, M3 (the sequential finetune chain) share IDENTICAL preprocessing
    statistics by construction (M2's finetune loaded M1's stats.json and
    wrote them back verbatim; M3 likewise from M2) -- verified here before
    merging, since merging weights trained under different feature scales
    would be meaningless. 7 states: M1, M2, M3, M1(+)M2, M2(+)M3, M1(+)M3,
    M1(+)M2(+)M3. Pairwise/triple states are PARAMETER-SPACE averages
    (state_dict interpolation), with the mixing coefficient chosen to
    MINIMISE THE AMPLITUDE TERM of the exact MSE decomposition
    (failure_diagnosis.py: MSE = bias^2 + (sp-r*st)^2 + st^2*(1-r^2)),
    evaluated on predictions POOLED across all three Region_2 test domains
    (I1, I2, I3) so the result is one concrete merged artifact, not a
    per-domain-optimal alpha that would defeat the point of merging.
    Grid search (cheap: inference-only, zero GPU training as section 6.4
    promises) -- 21-point grid for pairwise, a 0.1-step simplex for the
    triple merge.

Usage: python p4_transfer_eval.py
"""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

if os.environ.get("SKIP_INSTALL") != "1":
    def _gpu_compute_cap():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60)
            line = out.stdout.strip().splitlines()[0]
            cap, name = [p.strip() for p in line.split(",")]
            return float(cap), name
        except Exception:  # noqa: BLE001
            return None, None
    _cap, _gpu = _gpu_compute_cap()
    print(f"[*] Detected GPU: {_gpu} (compute capability {_cap})", flush=True)
    if _cap is not None and _cap < 7.5:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
                 "--index-url", "https://download.pytorch.org/whl/cu126"],
                stdout=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] cu126 install failed: {exc}", flush=True)
    _con = "/tmp/constraints.txt"
    try:
        _v = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"],
            capture_output=True, text=True, timeout=300).stdout.strip()
        with open(_con, "w") as _f:
            _f.write(f"torch=={_v}\n")
    except Exception:  # noqa: BLE001
        _con = None
    cmd = [sys.executable, "-m", "pip", "install", "-q", "neuralforecast"]
    if _con:
        cmd += ["-c", _con]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        print(f"    [!] install neuralforecast failed: {exc}", flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NIXTLA_ID_AS_COL"] = "1"
import logging  # noqa: E402
for _n in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from neuralforecast import NeuralForecast  # noqa: E402

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
    os.name == "posix" and os.path.isdir("/kaggle/input"))
INPUT_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path("./study_p4")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("./train")

DOMAINS = [f"R{r}_I{i}" for i in (1, 2, 3) for r in (1, 2, 3)]
MODELS = ["TiDE", "TimeXer"]
TARGET_CH = os.environ.get("CHANNEL", "self_x_ac")
# Same channel-disambiguation as p4_transfer_train.py's OUT dir: only the
# vibration (default) checkpoints use the bare {ds}_{name}_{mode} naming;
# any other channel's checkpoints/outputs carry a suffix so they can never
# collide with or be mistaken for the vibration ones.
_CH_SFX = "" if TARGET_CH == "self_x_ac" else "_" + TARGET_CH.replace("self_", "")
# Part 2 (weight merging) only has finetune checkpoints for Region_2/TiDE on
# the vibration channel -- it was never trained for any other channel, so it
# is skipped (not silently run against missing/wrong checkpoints) whenever
# TARGET_CH isn't the vibration default. See the guard before Part 2 below.
RUN_MERGE_STUDY = TARGET_CH == "self_x_ac"
INPUT_SIZE = 128
HORIZON = 15
N_WINDOWS = 80
MULTIVARIATE = {"TSMixerx", "TimeXer"}


def region_of(ds):
    return ds.split("_")[1]


def iter_of(ds):
    return ds.split("_")[2]


def ckpt_dir(ds, name, mode="base"):
    tag = f"{ds}_{name}_{mode}{_CH_SFX}"
    if ON_KAGGLE:
        hits = sorted(INPUT_ROOT.rglob(f"{tag}/checkpoint"))
        if not hits:
            raise FileNotFoundError(f"no checkpoint for {tag} under {INPUT_ROOT}")
        return hits[0]
    return INPUT_ROOT / tag / "checkpoint"


def stats_path(ds, name, mode="base"):
    return ckpt_dir(ds, name, mode).parent / "stats.json"


def raw_csv(ds):
    if ON_KAGGLE:
        hits = sorted(INPUT_ROOT.rglob(f"{ds}.csv"))
        if not hits:
            raise FileNotFoundError(f"no {ds}.csv under {INPUT_ROOT}")
        return hits[0]
    return sorted(Path("./data_canonical").glob(f"*/*/{ds}.csv"))[0]


_raw_cache = {}


def load_domain(ds):
    """Raw canonical CSV for `ds`, target promoted to 'y', synthetic ds axis."""
    if ds in _raw_cache:
        return _raw_cache[ds]
    df = pd.read_csv(raw_csv(ds))
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    Y_SOURCE = "self_temperature_one"
    if TARGET_CH != Y_SOURCE:
        df[Y_SOURCE] = df["y"]
        df["y"] = df[TARGET_CH]
    _leak = {TARGET_CH, Y_SOURCE} | {c for c in df.columns if c.startswith(TARGET_CH + "_")}
    exog = [c for c in df.columns if c not in ({"unique_id", "ds", "y", "segment_id"} | _leak)]
    n = len(df)
    base_ts = pd.Timestamp("2025-01-01")
    df["ds"] = base_ts + pd.to_timedelta(np.arange(n), unit="m")
    seg = df["segment_id"].to_numpy()
    train_end, val_end, test_end = int(n * 0.85), int(n * 0.90), n

    first = max(val_end, INPUT_SIZE)
    last = min(test_end, n) - HORIZON
    cand = np.arange(first, last)
    ok = (seg[cand - INPUT_SIZE] == seg[cand + HORIZON - 1]) & (seg[cand] == seg[cand - INPUT_SIZE])
    cand = cand[ok]
    m = min(N_WINDOWS, cand.size)
    origins = cand[np.round(np.linspace(0, cand.size - 1, m)).astype(int)]

    value_cols = ["y"] + exog
    yv = df["y"].to_numpy(dtype=np.float64)
    yt = np.stack([yv[o:o + HORIZON] for o in origins])
    _raw_cache[ds] = dict(df=df, exog=exog, value_cols=value_cols, base_ts=base_ts,
                          origins=origins, yt=yt)
    return _raw_cache[ds]


def predict_cross(nf_obj, name, source_stats, ds):
    """Predict on ds's F5 test window, standardizing with SOURCE stats.
    Returns raw-unit predictions, shape [n_origins, HORIZON]."""
    dom = load_domain(ds)
    df, exog, value_cols, base_ts, origins = (
        dom["df"], dom["exog"], dom["value_cols"], dom["base_ts"], dom["origins"])
    mu = pd.Series(source_stats["mu"])
    sd = pd.Series(source_stats["sd"])
    missing = set(value_cols) - set(mu.index)
    if missing:
        raise ValueError(f"source stats missing columns for {ds}: {missing}")
    dfs = df.copy()
    dfs[value_cols] = (dfs[value_cols] - mu[value_cols]) / sd[value_cols]
    y_mu, y_sd = float(mu["y"]), float(sd["y"])

    idx = base_ts + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
    if name in MULTIVARIATE:
        rows = []
        for o in origins:
            w = dfs.iloc[o - INPUT_SIZE:o].copy()
            w["unique_id"] = ds
            w["ds"] = idx
            p = nf_obj.predict(df=w, verbose=False)
            p = p.reset_index() if "unique_id" not in p.columns else p
            col = [c for c in p.columns if c not in ("unique_id", "ds")][0]
            rows.append(p[col].to_numpy(dtype=np.float64)[:HORIZON])
        yp_std = np.stack(rows)
    else:
        parts = []
        for k, o in enumerate(origins):
            w = dfs.iloc[o - INPUT_SIZE:o].copy()
            w["unique_id"] = f"w{k:04d}"
            w["ds"] = idx
            parts.append(w)
        batch = pd.concat(parts, ignore_index=True)
        p = nf_obj.predict(df=batch, verbose=False)
        p = p.reset_index() if "unique_id" not in p.columns else p
        model_col = [c for c in p.columns if c not in ("unique_id", "ds")][0]
        p = p.sort_values(["unique_id", "ds"])
        yp_std = np.full((len(origins), HORIZON), np.nan)
        for uid, g in p.groupby("unique_id", sort=False):
            k = int(str(uid)[1:])
            v = g[model_col].to_numpy(dtype=np.float64)[:HORIZON]
            yp_std[k, :len(v)] = v
    return yp_std * y_sd + y_mu


def decompose(t, p):
    t = np.asarray(t, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    mt, mp = t.mean(), p.mean()
    st, sp = t.std(), p.std()
    r = float(np.corrcoef(t, p)[0, 1]) if sp > 1e-12 and st > 1e-12 else 0.0
    bias2 = (mp - mt) ** 2
    amp = (sp - r * st) ** 2
    phase = st ** 2 * (1 - r ** 2)
    mse = float(((p - t) ** 2).mean())
    return dict(MSE=mse, bias2=bias2, amplitude=amp, phase=phase, r=r,
                sd_ratio=float(sp / st) if st > 0 else np.nan)


# ============================================================
# Part 1: 81-cell transfer matrix (per model)
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PART 1: cross-domain transfer matrix", flush=True)
print("=" * 70, flush=True)

rows = []
for src_ds in DOMAINS:
    for name in MODELS:
        cdir = ckpt_dir(src_ds, name, "base")
        stats = json.loads(stats_path(src_ds, name, "base").read_text())
        nf = NeuralForecast.load(str(cdir), verbose=False)
        for tgt_ds in DOMAINS:
            dom = load_domain(tgt_ds)
            yp = predict_cross(nf, name, stats, tgt_ds)
            yt = dom["yt"]
            d = decompose(yt, yp)
            same_region = region_of(src_ds) == region_of(tgt_ds)
            same_iter = iter_of(src_ds) == iter_of(tgt_ds)
            if src_ds == tgt_ds:
                shift = "diagonal"
            elif same_region and not same_iter:
                shift = "temporal-only"
            elif same_iter and not same_region:
                shift = "spatial-only"
            else:
                shift = "both"
            rows.append(dict(source=src_ds, target=tgt_ds, model=name, shift=shift,
                              **d))
            print(f"  {src_ds:12s} -> {tgt_ds:12s}  {name:8s}  shift={shift:14s} "
                  f"MSE={d['MSE']:.5f}  R2_proxy={1-d['MSE']/dom['yt'].var():.3f}", flush=True)
        del nf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

df_matrix = pd.DataFrame(rows)
df_matrix.to_csv(OUT / "TRANSFER_MATRIX.csv", index=False)
print("\n=== Marginal means by shift type, by model ===", flush=True)
print(df_matrix.groupby(["model", "shift"])["MSE"].agg(["mean", "count"]).round(5).to_string(), flush=True)
print(f"\n[SAVED] TRANSFER_MATRIX.csv  ({len(df_matrix)} rows)", flush=True)

if not RUN_MERGE_STUDY:
    print(f"\n[SKIP] Part 2 (weight merging) only has finetune checkpoints for "
          f"Region_2/TiDE on the vibration channel (self_x_ac) -- never trained "
          f"for TARGET_CH={TARGET_CH!r}. Stopping after Part 1 rather than running "
          f"against missing or wrong checkpoints.", flush=True)
    raise SystemExit(0)

# ============================================================
# Part 2: weight merging, Region_2/TiDE, M1/M2/M3
# ============================================================
print("\n" + "=" * 70, flush=True)
print("PART 2: weight merging (Region_2, TiDE)", flush=True)
print("=" * 70, flush=True)

M_DS = {"M1": "R2_I1", "M2": "R2_I2", "M3": "R2_I3"}
M_MODE = {"M1": "base", "M2": "finetune", "M3": "finetune"}
REGION2_DOMAINS = ["R2_I1", "R2_I2", "R2_I3"]

m_stats = {k: json.loads(stats_path(M_DS[k], "TiDE", M_MODE[k]).read_text()) for k in M_DS}
for k in ("M2", "M3"):
    same = all(m_stats[k]["mu"][c] == m_stats["M1"]["mu"][c] for c in m_stats["M1"]["mu"])
    print(f"  [verify] {k} shares M1's preprocessing stats: {same}", flush=True)
    if not same:
        raise SystemExit(f"[X] {k} stats diverge from M1 -- merging would be invalid, aborting")

m_nf = {k: NeuralForecast.load(str(ckpt_dir(M_DS[k], "TiDE", M_MODE[k])), verbose=False) for k in M_DS}
m_state = {k: m_nf[k].models[0].state_dict() for k in M_DS}
stats_ref = m_stats["M1"]  # identical across all three, verified above


def merged_state(weights):
    """weights: dict state_name -> float weight, summing to 1."""
    keys = m_state["M1"].keys()
    out = {}
    for key in keys:
        acc = None
        for k, w in weights.items():
            t = m_state[k][key].float() * w
            acc = t if acc is None else acc + t
        out[key] = acc.to(m_state["M1"][key].dtype)
    return out


def eval_state(state_dict):
    """Load `state_dict` into a fresh nf (cloned from M1's) and evaluate,
    pooled across all three Region_2 test domains, using M1's stats
    (shared by all three checkpoints by construction). Also returns the
    PER-DOMAIN breakdown (not just pooled) -- pooled MSE is dominated by
    whichever domain has the largest raw error scale (R2_I3, per the
    covariate-shift finding in section 5), so a pooled-only number would
    hide exactly the retention-vs-forgetting question this chain exists to
    answer. Both are reported, never just the one that reads better."""
    probe = NeuralForecast.load(str(ckpt_dir("R2_I1", "TiDE", "base")), verbose=False)
    probe.models[0].load_state_dict(state_dict)
    per_domain = {}
    preds, truths = [], []
    for ds in REGION2_DOMAINS:
        yp = predict_cross(probe, "TiDE", stats_ref, ds)
        yt = load_domain(ds)["yt"]
        per_domain[ds] = decompose(yt, yp)
        preds.append(yp.ravel())
        truths.append(yt.ravel())
    p = np.concatenate(preds)
    t = np.concatenate(truths)
    return decompose(t, p), per_domain


merge_rows = []
per_domain_rows = []


def record_state(label, weights):
    sd = merged_state(weights)
    d, per_domain = eval_state(sd)
    merge_rows.append(dict(state=label, weights=json.dumps(weights), **d))
    for ds, dd in per_domain.items():
        per_domain_rows.append(dict(state=label, domain=ds, **dd))
    pd_str = ", ".join(f"{ds.split('_')[-1]}:{dd['MSE']:.4f}" for ds, dd in per_domain.items())
    print(f"  {label:16s} weights={weights}  MSE={d['MSE']:.5f}  amp={d['amplitude']:.5f}  "
          f"r={d['r']:.4f}  sd_ratio={d['sd_ratio']:.4f}  per_domain_MSE=[{pd_str}]", flush=True)


# Pure states
record_state("M1", {"M1": 1.0})
record_state("M2", {"M2": 1.0})
record_state("M3", {"M3": 1.0})

# Pairwise: grid-search alpha in [0,1] to MINIMISE THE AMPLITUDE TERM
# (not raw MSE) -- the disclosed methodology of section 6.4.
GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
for pair, (a_key, b_key) in [("M1(+)M2", ("M1", "M2")), ("M2(+)M3", ("M2", "M3")),
                              ("M1(+)M3", ("M1", "M3"))]:
    best = None
    for alpha in GRID:
        sd = merged_state({a_key: float(alpha), b_key: float(1 - alpha)})
        d, _ = eval_state(sd)
        if best is None or d["amplitude"] < best[1]["amplitude"]:
            best = (alpha, d)
    alpha, d = best
    # Re-run only the winning combination to also capture its per-domain
    # breakdown (grid search itself only needs the pooled amplitude to pick
    # a winner, so per-domain is not computed 21x for nothing).
    sd = merged_state({a_key: float(alpha), b_key: float(1 - alpha)})
    d, per_domain = eval_state(sd)
    for ds, dd in per_domain.items():
        per_domain_rows.append(dict(state=pair, domain=ds, **dd))
    merge_rows.append(dict(state=pair, weights=json.dumps({a_key: alpha, b_key: round(1 - alpha, 2)}),
                            **d))
    pd_str = ", ".join(f"{ds.split('_')[-1]}:{dd['MSE']:.4f}" for ds, dd in per_domain.items())
    print(f"  {pair:16s} alpha*={alpha:.2f} (amplitude-optimal, searched {len(GRID)} pts)  "
          f"MSE={d['MSE']:.5f}  amp={d['amplitude']:.5f}  r={d['r']:.4f}  sd_ratio={d['sd_ratio']:.4f}  "
          f"per_domain_MSE=[{pd_str}]", flush=True)

# Triple: 0.1-step simplex grid over (w1,w2,w3), w1+w2+w3=1.
best = None
STEP = 0.1
grid1 = np.round(np.arange(0.0, 1.0 + STEP / 2, STEP), 2)
for w1 in grid1:
    for w2 in grid1:
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < -1e-9 or w3 > 1.0 + 1e-9:
            continue
        w3 = max(0.0, w3)
        sd = merged_state({"M1": float(w1), "M2": float(w2), "M3": float(w3)})
        d, _ = eval_state(sd)
        if best is None or d["amplitude"] < best[1]["amplitude"]:
            best = ((w1, w2, w3), d)
(w1, w2, w3), d = best
sd = merged_state({"M1": float(w1), "M2": float(w2), "M3": float(w3)})
d, per_domain = eval_state(sd)
for ds, dd in per_domain.items():
    per_domain_rows.append(dict(state="M1(+)M2(+)M3", domain=ds, **dd))
merge_rows.append(dict(state="M1(+)M2(+)M3",
                        weights=json.dumps({"M1": w1, "M2": w2, "M3": w3}), **d))
pd_str = ", ".join(f"{ds.split('_')[-1]}:{dd['MSE']:.4f}" for ds, dd in per_domain.items())
print(f"  {'M1(+)M2(+)M3':16s} w*=({w1:.2f},{w2:.2f},{w3:.2f}) (amplitude-optimal)  "
      f"MSE={d['MSE']:.5f}  amp={d['amplitude']:.5f}  r={d['r']:.4f}  sd_ratio={d['sd_ratio']:.4f}  "
      f"per_domain_MSE=[{pd_str}]", flush=True)

df_pd = pd.DataFrame(per_domain_rows)
df_pd.to_csv(OUT / "MERGE_PER_DOMAIN.csv", index=False)
print(f"[SAVED] MERGE_PER_DOMAIN.csv  ({len(df_pd)} rows)", flush=True)

df_merge = pd.DataFrame(merge_rows)
df_merge.to_csv(OUT / "MERGE_RESULTS.csv", index=False)
print(f"\n[SAVED] MERGE_RESULTS.csv  ({len(df_merge)} rows)", flush=True)
print("\n[ALL-DONE]", flush=True)
