# RELEASE NOTE: this is the script exactly as it was run to produce the stored
# results, with site/person identifiers redacted (plant name, dataset acronym,
# hosting-account names, local paths, internal series ids). Known issues found in
# the audit are listed in AUDIT_REPORT.md; the code itself is unchanged.
"""P4 (PAPER_STRATEGY.md section 6, 7): base/fine-tune training for the
81-cell cross-domain transfer matrix and the M1/M2/M3 weight-merge chain.

Unlike kaggle_train_all.py (5-fold CV, full 13-model roster, per-target),
this script trains and PERSISTS actual model weights -- the transfer matrix
needs to run a model trained on domain A directly against domain B's data,
which is impossible from predictions.npz alone (no prior script in this
project saves checkpoints; only aggregated metrics/predictions against the
model's OWN training domain).

Design decisions (disclosed here, not just in PAPER_STRATEGY.md, so the
run itself is self-documenting):

- **Channel**: self_x_ac only (vibration). Chosen because it is the channel
  carrying the one real, disclosed temporal-shift event in this dataset
  (R2_I3 covariate shift, section 5) -- a transfer/retention study is only
  informative where there is a real shift to measure retention against.
- **Model roster**: TiDE + TimeXer (section 6.3's "TiDE, SCAS-TiDE, and one
  runner-up" -- TimeXer is the consistent #2 by calibration across every
  diagnostic this session, FAILURE_ATTRIBUTION.csv). SCAS itself is a
  POST-HOC head fit on frozen predictions (scas.py) and is applied to
  TiDE's transfer-matrix predictions afterward, not trained here.
- **Fold**: F5_test90_100 only (train=[0,85%), val=[85%,90%), test=[90%,
  100%)) -- the fold trained on the most data, i.e. the one closest to how
  a single deployed model would actually be trained. The 81-cell matrix
  therefore reports ONE model per domain, not a 5-fold average like the
  headline numbers -- a disclosed reduction in statistical power relative
  to sections 1-4, made necessary by the cost of persisting+moving weights
  for every fold x domain x model combination.
- **Cross-domain preprocessing**: standardize the TARGET domain's raw data
  using the SOURCE domain's own train-set mu/sd (computed once at save time
  and persisted in stats.json), not the target domain's own statistics.
  This reproduces literally the same preprocessing pipeline the model was
  calibrated with -- "run domain A's exact trained model on domain B" means
  the model's input scale contract does not change, only the underlying
  physical values do. Predictions are destandardized back with the SAME
  source-domain y_mu/y_sd, then scored against the target domain's raw
  (never-standardized) ground truth. This script only trains + saves; the
  cross-domain scoring itself lives in p4_transfer_eval.py, run only after
  the base/finetune checkpoints below are confirmed to round-trip correctly.
- **Continued-fit semantics for M2=finetune(M1), M3=finetune(M2)**: verified
  against the real NeuralForecast source (nixtla/neuralforecast core.py,
  2026-07-31) before spending any Kaggle quota on it -- `fit()`'s
  `use_init_models` defaults to False, meaning a second `fit()` call on an
  already-fitted (or reloaded) NeuralForecast object CONTINUES training from
  existing weights rather than reinitializing. This script never passes
  use_init_models=True, so MODE=finetune genuinely continues training rather
  than silently retraining from scratch under a misleading name.

Modes (env vars):
  MODE=base       -- train NAME from scratch on DS's F5 fold, save checkpoint.
  MODE=finetune   -- load the checkpoint at PREV_CKPT (a prior base/finetune
                     run's output, attached as a Kaggle input), continue
                     fitting on DS's F5 fold, save a new checkpoint.
  DS              -- one of the 9 canonical dataset names (R{1,2,3}_I{1,2,3}).
  NAME            -- "TiDE" or "TimeXer".
  SMOKE=1         -- tiny max_steps, for a fast save/load/continue-fit
                     round-trip check before committing real quota.

Usage: python p4_transfer_train.py   (env vars set by the launcher, as with
       every other script in this project)
"""
import glob
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                           stdout=subprocess.DEVNULL)


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


if os.environ.get("SKIP_INSTALL") != "1":
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
    for _pkg in ("neuralforecast",):
        cmd = [sys.executable, "-m", "pip", "install", "-q", _pkg]
        if _con:
            cmd += ["-c", _con]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] install {_pkg} failed: {exc}", flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NIXTLA_ID_AS_COL"] = "1"
import logging  # noqa: E402
for _n in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_n).setLevel(logging.ERROR)

DS = os.environ.get("DS", "R1_I1")
NAME = os.environ.get("NAME", "TiDE")
MODE = os.environ.get("MODE", "base")
PREV_CKPT = os.environ.get("PREV_CKPT", "")
SMOKE = os.environ.get("SMOKE") == "1"
TARGET_CH = os.environ.get("CHANNEL", "self_x_ac")
# Local (non-Kaggle) output dir must be channel-disambiguated: the vibration
# run's OUT below is {DS}_{NAME}_{MODE} with no channel in it (18 existing
# checkpoints already live under that naming), so a same-named temperature
# run would silently overwrite them. Vibration (the default) keeps the
# original bare naming for backward compatibility; any other channel gets a
# suffix.
_CH_SFX = "" if TARGET_CH == "self_x_ac" else "_" + TARGET_CH.replace("self_", "")

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
    os.name == "posix" and os.path.isdir("/kaggle/input"))
if ON_KAGGLE:
    OUT = Path("/kaggle/working")
    _hits = sorted(Path("/kaggle/input").rglob(DS + ".csv"))
    if not _hits:
        print("[X] " + DS + ".csv not found under /kaggle/input:", flush=True)
        for _p in sorted(Path("/kaggle/input").rglob("*"))[:100]:
            print("   ", _p, flush=True)
        raise SystemExit("Attach released-dataset to this kernel.")
    DATA_PATH = str(_hits[0])
    if MODE == "finetune":
        # PREV_DIR is the producing run's OUT dir (stats.json lives directly
        # in it; checkpoint/ is a sibling subdirectory of it) -- NOT the
        # checkpoint/ dir itself. NeuralForecast.load() needs PREV_DIR +
        # "/checkpoint" specifically; that indirection is applied at the
        # load() call site below, not here, so this one variable stays
        # unambiguous for both uses (stats.json read + NF checkpoint load).
        _ck_hits = sorted(Path("/kaggle/input").rglob("stats.json"))
        _ck_hits = [p for p in _ck_hits if PREV_CKPT in str(p.parent)] or _ck_hits
        if not _ck_hits:
            raise SystemExit(f"[X] MODE=finetune but no checkpoint found "
                             f"matching PREV_CKPT={PREV_CKPT!r} under /kaggle/input")
        PREV_DIR = str(_ck_hits[0].parent)
    print(f"[*] Found data at: {DATA_PATH}", flush=True)
else:
    _pat = f"./data_canonical/*/*/{DS}.csv"
    DATA_PATH = glob.glob(_pat)[0]
    OUT = Path(rf".\study_p4\{DS}_{NAME}_{MODE}{_CH_SFX}")
    PREV_DIR = PREV_CKPT
OUT.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 128
HORIZON = 15
MAX_STEPS = 20 if SMOKE else 3000
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
WINDOWS_BATCH_SIZE = 256
VAL_CHECK_STEPS = 10 if SMOKE else 50
EARLY_STOP_PATIENCE = 2 if SMOKE else 15
RANDOM_SEED = 42
TARGET = "y"
N_WINDOWS = 8 if SMOKE else 80

ACCELERATOR = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        try:
            _a = torch.randn(64, 64, device="cuda")
            _ = (_a @ _a).sum().item()
            ACCELERATOR = "gpu"
            print(f"[*] GPU OK: {torch.cuda.get_device_name(0)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] GPU unusable: {exc} -- falling back to CPU", flush=True)
    else:
        print("[*] No GPU visible, using CPU", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[!] torch check failed: {exc}", flush=True)

# ---- data ----
print("\n[*] Loading data...", flush=True)
df = pd.read_csv(DATA_PATH)
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

Y_SOURCE = "self_temperature_one"
if TARGET_CH != Y_SOURCE:
    df[Y_SOURCE] = df["y"]
    df["y"] = df[TARGET_CH]
_leak = {TARGET_CH, Y_SOURCE} | {c for c in df.columns if c.startswith(TARGET_CH + "_")}
EXOG = [c for c in df.columns if c not in ({"unique_id", "ds", TARGET, "segment_id"} | _leak)]
print(f"    target={TARGET_CH}  exog={len(EXOG)}  rows={len(df):,}", flush=True)

N = len(df)
BASE_TS = pd.Timestamp("2025-01-01")
df["ds"] = BASE_TS + pd.to_timedelta(np.arange(N), unit="m")
VALUE_COLS = [TARGET] + EXOG

# F5 fold only: train=[0,85%), val=[85%,90%), test=[90%,100%) -- same
# definition as kaggle_train_all.py's F5_test90_100, reused for consistency.
train_end, val_end, test_end = int(N * 0.85), int(N * 0.90), N
SEG = df["segment_id"].to_numpy()


def make_origins():
    first = max(val_end, INPUT_SIZE)
    last = min(test_end, N) - HORIZON
    if last <= first:
        return np.array([], dtype=int)
    cand = np.arange(first, last)
    ok = (SEG[cand - INPUT_SIZE] == SEG[cand + HORIZON - 1]) & (SEG[cand] == SEG[cand - INPUT_SIZE])
    cand = cand[ok]
    if cand.size == 0:
        return cand
    n = min(N_WINDOWS, cand.size)
    return cand[np.round(np.linspace(0, cand.size - 1, n)).astype(int)]


origins = make_origins()
if origins.size == 0:
    raise SystemExit(f"[X] {DS}: no valid F5 test origins")

if MODE == "base":
    mu = df.iloc[:train_end][VALUE_COLS].mean()
    sd = df.iloc[:train_end][VALUE_COLS].std().replace(0.0, 1.0)
else:
    stats = json.loads((Path(PREV_DIR) / "stats.json").read_text())
    mu = pd.Series(stats["mu"])
    sd = pd.Series(stats["sd"])
    missing = set(VALUE_COLS) - set(mu.index)
    if missing:
        raise SystemExit(f"[X] finetune stats.json missing columns: {missing}")

dfs = df.copy()
dfs[VALUE_COLS] = (dfs[VALUE_COLS] - mu) / sd
y_mu, y_sd = float(mu[TARGET]), float(sd[TARGET])
train_val = dfs.iloc[:val_end]
val_size = val_end - train_end


def truth_matrix():
    yv = df[TARGET].to_numpy(dtype=np.float64)
    return np.stack([yv[o:o + HORIZON] for o in origins])


def history_batch():
    parts = []
    idx = BASE_TS + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
    for k, o in enumerate(origins):
        w = dfs.iloc[o - INPUT_SIZE:o].copy()
        w["unique_id"] = f"w{k:04d}"
        w["ds"] = idx
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def collect(pred_df, model_col):
    p = pred_df.sort_values(["unique_id", "ds"])
    out = np.full((len(origins), HORIZON), np.nan)
    for uid, g in p.groupby("unique_id", sort=False):
        k = int(str(uid)[1:])
        v = g[model_col].to_numpy(dtype=np.float64)[:HORIZON]
        out[k, :len(v)] = v
    return out


# TimeXer (and TSMixerx, not used in this roster but sharing the same
# constructor pattern) is instantiated with n_series=1, so a single predict()
# call across multiple unique_ids/windows breaks inside its internal
# concat -- caught for real on the first non-smoke TimeXer push (RuntimeError:
# "Sizes of tensors must match... Expected size 32 but got size 1"). Same fix
# kaggle_train_all.py already uses: predict one window at a time.
MULTIVARIATE = {"TSMixerx", "TimeXer"}


def run_predict(nf_obj, name):
    """Predict on `origins`, standardized-scale, shape [n_origins, HORIZON]."""
    if name in MULTIVARIATE:
        idx = BASE_TS + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
        rows = []
        for o in origins:
            w = dfs.iloc[o - INPUT_SIZE:o].copy()
            w["unique_id"] = DS
            w["ds"] = idx
            p = nf_obj.predict(df=w, verbose=False)
            p = p.reset_index() if "unique_id" not in p.columns else p
            col = [c for c in p.columns if c not in ("unique_id", "ds")][0]
            rows.append(p[col].to_numpy(dtype=np.float64)[:HORIZON])
        return np.stack(rows)
    batch = history_batch()
    pred = nf_obj.predict(df=batch, verbose=False)
    model_col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
    return collect(pred, model_col)


def build_model(name):
    from neuralforecast.models import TiDE, TimeXer
    base = dict(h=HORIZON, input_size=INPUT_SIZE, max_steps=MAX_STEPS,
                learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                val_check_steps=VAL_CHECK_STEPS,
                early_stop_patience_steps=EARLY_STOP_PATIENCE,
                scaler_type="robust", random_seed=RANDOM_SEED,
                enable_progress_bar=False, logger=False,
                enable_model_summary=False, accelerator=ACCELERATOR,
                devices=1, gradient_clip_val=1.0)
    win = dict(base, windows_batch_size=WINDOWS_BATCH_SIZE)
    fuse = dict(hist_exog_list=EXOG)
    if name == "TiDE":
        return TiDE(**win, **fuse, hidden_size=256, alias="TiDE")
    if name == "TimeXer":
        return TimeXer(**win, **fuse, n_series=1, hidden_size=128,
                       n_heads=4, e_layers=2, patch_len=16, alias="TimeXer")
    raise ValueError(name)


from neuralforecast import NeuralForecast  # noqa: E402

t0 = time.time()
if MODE == "base":
    print(f"\n[*] MODE=base  DS={DS}  NAME={NAME}  (F5 fold, from scratch)", flush=True)
    nf = NeuralForecast(models=[build_model(NAME)], freq="1min")
    nf.fit(df=train_val, val_size=val_size, verbose=False)
else:
    print(f"\n[*] MODE=finetune  DS={DS}  NAME={NAME}  PREV_CKPT={PREV_CKPT}", flush=True)
    nf = NeuralForecast.load(str(Path(PREV_DIR) / "checkpoint"), verbose=False)

    # Verification, not just convenience: predict on THIS domain's own test
    # window using the JUST-LOADED weights, before any further fit() call.
    # mu/sd at this point are still the SOURCE domain's (loaded from
    # PREV_DIR/stats.json), so this is exactly the source-domain-stats
    # cross-domain preprocessing the transfer matrix uses -- if load()
    # silently failed to restore real weights (e.g. returned a freshly
    # re-initialized model), this would score at or near random-init level;
    # a non-trivial R2 here is direct evidence the checkpoint round-trip
    # actually preserved trained weights, not just that no exception fired.
    _zs_yp = run_predict(nf, NAME) * y_sd + y_mu
    _zs_yt = truth_matrix()
    _zs_err = _zs_yt - _zs_yp
    _zs_mse = float(np.nanmean(_zs_err ** 2))
    _zs_r2 = 1.0 - float(np.nansum(_zs_err ** 2)) / float(np.nansum((_zs_yt - np.nanmean(_zs_yt)) ** 2))
    print(f"    [verify] zero-shot (loaded weights, pre-finetune) on {DS}: "
          f"MSE={_zs_mse:.5f}  R2={_zs_r2:.4f}", flush=True)

    # use_init_models left at its default (False) deliberately -- continues
    # training from the loaded weights instead of reinitializing. Verified
    # against neuralforecast/core.py before this was wired in (see docstring).
    nf.fit(df=train_val, val_size=val_size, verbose=False)
train_time = time.time() - t0
print(f"    trained in {train_time:.1f}s", flush=True)

# ---- in-domain (diagonal) evaluation, clean data only ----
yp = run_predict(nf, NAME) * y_sd + y_mu
yt = truth_matrix()
err = yt - yp
mse = float(np.nanmean(err ** 2))
r2 = 1.0 - float(np.nansum(err ** 2)) / float(np.nansum((yt - np.nanmean(yt)) ** 2))
print(f"    in-domain (own F5 test): MSE={mse:.5f}  R2={r2:.4f}", flush=True)

# ---- persist checkpoint + stats + in-domain reference ----
ckpt_dir = OUT / "checkpoint"
nf.save(path=str(ckpt_dir), save_dataset=False, overwrite=True)
(OUT / "stats.json").write_text(json.dumps(dict(
    mu={k: float(v) for k, v in mu.items()}, sd={k: float(v) for k, v in sd.items()},
    y_mu=y_mu, y_sd=y_sd, ds=DS, name=NAME, mode=MODE, prev_ckpt=PREV_CKPT,
    target_channel=TARGET_CH, exog=EXOG, train_end=int(train_end),
    val_end=int(val_end), test_end=int(test_end), train_time_s=round(train_time, 1),
    in_domain_MSE=round(mse, 6), in_domain_R2=round(r2, 4),
    **(dict(pre_finetune_zeroshot_MSE=round(_zs_mse, 6),
            pre_finetune_zeroshot_R2=round(_zs_r2, 4)) if MODE == "finetune" else {})),
    indent=2))
np.savez_compressed(OUT / "in_domain_predictions.npz",
                     ytrue=yt.astype(np.float32), ypred=yp.astype(np.float32),
                     origins=origins.astype(np.int32))
print(f"\n[DONE] checkpoint -> {ckpt_dir}", flush=True)
print(f"[DONE] stats.json + in_domain_predictions.npz -> {OUT}", flush=True)
