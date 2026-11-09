# RELEASE NOTE: this is the script exactly as it was run to produce the stored
# results, with site/person identifiers redacted (plant name, dataset acronym,
# hosting-account names, local paths, internal series ids). Known issues found in
# the audit are listed in AUDIT_REPORT.md; the code itself is unchanged.
"""
================================================================================
PER-TARGET FORECASTING & ABLATION STUDY
================================================================================
a steel plant (anonymised) - one sensor channel per run.

Region and target are chosen by the DS and TARGET_CH environment variables and
recorded in run_config.json; nothing about either is hard-coded here.

10 Models x 5 Splits x Gaussian-noise levels (4 under PROTOCOL=full, 1 under
PROTOCOL=reduced), evaluated with rolling-origin windows over the held-out test
span.

Design notes
------------
* SENSOR FUSION: the target is TARGET_CHANNEL. Every other channel from the
  region's fused sensors (temp_one, temp_two, vib_x/y/z per sensor) is supplied
  as a *historical* exogenous input. The target's own column and anything
  derived from it are dropped from the exogenous list -- see
  run_config.json["dropped_as_leakage"] for the per-run list.

* MODEL CAPABILITY: NeuralForecast raises if a model with EXOGENOUS_HIST=False
  receives hist_exog_list (see neuralforecast/common/_base_model.py). Verified
  against v3.x source: FEDformer, PatchTST, TimesNet, iTransformer, Informer and
  Autoformer are all EXOGENOUS_HIST=False. The roster below therefore uses
  fusion-capable models everywhere except FEDformer, which is required by the
  study and is run univariate. That contrast is itself reported as an ablation.

* EVALUATION: every model is scored on the SAME set of rolling forecast origins
  per split (N_WINDOWS origins x HORIZON steps), so numbers are directly
  comparable across models. Metrics are always computed against CLEAN ground
  truth.

* NOISE ABLATION: models are trained once on clean data; Gaussian noise is then
  injected into the *inference inputs* (target history + exogenous history),
  scaled by each channel's training-set standard deviation. This measures
  genuine input robustness. Noise is never added to the ground truth.

Run on Kaggle GPU (Tesla T4/P100).
================================================================================
"""

import subprocess
import sys


def install(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
    )


import os as _os


def _gpu_compute_cap():
    """GPU compute capability via nvidia-smi, without importing torch."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        line = out.stdout.strip().splitlines()[0]
        cap, name = [p.strip() for p in line.split(",")]
        return float(cap), name
    except Exception:  # noqa: BLE001
        return None, None


if _os.environ.get("SKIP_INSTALL") != "1":
    # Kaggle currently ships torch 2.10.0+cu128. The cu128 wheels are built
    # WITHOUT Pascal kernels (sm_60), so on a P100 every CUDA op dies with
    #   "CUDA error: no kernel image is available for execution on the device"
    # even though torch.cuda.is_available() reports True. The Kaggle API cannot
    # request a specific GPU (only enable_gpu), so when we land on a pre-Turing
    # card the fix is to install a cu126 build, which still ships sm_60.
    _cap, _gpu = _gpu_compute_cap()
    print(f"[*] Detected GPU: {_gpu} (compute capability {_cap})", flush=True)

    if _cap is not None and _cap < 7.5:
        print(f"    sm_{int(_cap * 10)} is not covered by the preinstalled "
              f"cu128 build -> installing torch cu126 (this takes a few "
              f"minutes)...", flush=True)
        # torchvision/torchaudio MUST be downgraded in lockstep: they carry C
        # extensions compiled against a specific torch ABI. Leaving the stock
        # 0.25.0 next to torch 2.7.1 breaks `import torchvision`, which
        # pytorch_lightning reaches through torchmetrics, killing every model.
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
                 "--index-url", "https://download.pytorch.org/whl/cu126"],
                stdout=subprocess.DEVNULL)
            print("    torch/vision/audio cu126 installed.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] cu126 install failed: {exc}", flush=True)

    # Pin whatever torch is now present so neuralforecast cannot replace it.
    _con = "/tmp/constraints.txt"
    try:
        _v = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.__version__.split('+')[0])"],
            capture_output=True, text=True, timeout=300).stdout.strip()
        with open(_con, "w") as _f:
            _f.write(f"torch=={_v}\n")
        print(f"    pinning torch=={_v}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    [!] could not pin torch: {exc}", flush=True)
        _con = None

    print("[*] Installing neuralforecast + pytorch-tabnet...", flush=True)
    for _pkg in ("neuralforecast", "pytorch-tabnet"):
        cmd = [sys.executable, "-m", "pip", "install", "-q", _pkg]
        if _con:
            cmd += ["-c", _con]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] install {_pkg} failed: {exc}", flush=True)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NIXTLA_ID_AS_COL"] = "1"

import logging  # noqa: E402

for _n in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# ============================================================
# CONFIGURATION
# ============================================================

# NOTE: a bare "/kaggle/working" check is drive-relative on Windows and can
# produce false positives, so require a Kaggle-only env var or a real POSIX path.
DS_NAME = os.environ.get("DS", "R1_I1")   # which of the 9 masters

# CANONICAL=1 switches to the role-canonical rebuild (PAPER_STRATEGY.md 6.2):
# self_*/nb1_*/nb2_* column names, identical across all three regions, in
# place of device-id names (s001_*, s006_*, s008_*). This is what makes
# cross-region transfer (section 6/P4) possible at all -- an R1-trained
# model's input slot and an R3-trained model's input slot now mean the same
# physical role, not the same device id (which does not exist in both).
CANONICAL = os.environ.get("CANONICAL", "") == "1"

# Which sensor channel is being forecast. The published files ship the target
# sensor's temperature_one already promoted to column "y"; any other channel is
# promoted here at load time. Recorded verbatim in run_config.json so a result
# folder can never be ambiguous about what it predicted.
if CANONICAL:
    # Same channel name in every region by construction -- the whole point of
    # the rebuild.
    Y_SOURCE = "self_temperature_one"
else:
    Y_SOURCE = {"R1": "s001_temperature_one",
                "R2": "s006_temperature_one",
                "R3": "s008_temperature_one"}[DS_NAME.split("_")[1]]
TARGET_CHANNEL = os.environ.get("TARGET_CH", "") or Y_SOURCE
REGION_DIR = "Region_" + DS_NAME.split("_")[1][1]
# full = 5 splits x 4 noise levels (headline targets)
# reduced = 5 splits, sigma=0 only (extended channel sweep)
PROTOCOL = os.environ.get("PROTOCOL", "full")

# Floor-aware training: replaces the roster with TiDE/TSMixerx variants whose
# loss hard-floors predicted variance at a known, precomputed noise level
# (derived from the target's signal_fraction, not fit from data) instead of
# training against plain MSE. FLOOR_VAR must be supplied by the launcher --
# it is target-specific and already computed in every completed run's
# postprocess_summary.json. Interface verified against the real
# neuralforecast source via train/smoke_floor_loss.py before this was wired
# in; see that file's history for why (the naive nn.Module attempt crashed
# on a missing is_distribution_output attribute, and the fix is to subclass
# DistributionLoss and override only get_distribution).
FLOOR_MODE = os.environ.get("FLOOR_MODE", "") == "1"
FLOOR_VAR = float(os.environ.get("FLOOR_VAR", "0") or "0")
if FLOOR_MODE and FLOOR_VAR < 0:
    raise SystemExit("FLOOR_VAR must be >= 0 ((1-signal_fraction)*Var(y)); "
                     "0 is valid (e.g. temperature, signal_fraction==1) and is "
                     "the control case where the floor adds no constraint.")


def _local_ds():
    import glob
    _tree = "data_canonical" if CANONICAL else "data"
    pat = f"./{_tree}/*/*/" + DS_NAME + ".csv"
    return glob.glob(pat)[0]


ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
    os.name == "posix" and os.path.isdir("/kaggle/input")
)
if ON_KAGGLE:
    OUT = Path("/kaggle/working")
    # Do not hard-code the mount slug: locate the CSV wherever it is attached and
    # print the tree if it is missing, so a mount problem is diagnosable from the log.
    _hits = sorted(Path("/kaggle/input").rglob(DS_NAME + ".csv"))
    if not _hits:
        print("[X] " + DS_NAME + ".csv not found. Contents of /kaggle/input:", flush=True)
        for _p in sorted(Path("/kaggle/input").rglob("*"))[:100]:
            print("   ", _p, flush=True)
        raise SystemExit("Attach the released dataset to this kernel.")
    DATA_PATH = str(_hits[0])
    print(f"[*] Found data at: {DATA_PATH}", flush=True)
else:
    DATA_PATH = _local_ds()
    _study_root = ("study_canonical" if CANONICAL else "study")
    _iter_dir = "Iteration_" + DS_NAME.split("_I")[-1]
    OUT = (Path(rf".\{_study_root}") / _iter_dir
           / REGION_DIR / TARGET_CHANNEL / ("results_floor" if FLOOR_MODE else "results"))
OUT.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("SMOKE") == "1"   # fast end-to-end validation run

INPUT_SIZE = 128          # 512 was tested and scored WORSE (TiDE R2 0.5609
                          # vs 0.5977 at 128), so the shorter lookback stands.
HORIZON = 15              # forecast horizon (min) - the operational lead time.
                          # 96 was inherited from benchmarks whose step is an
                          # HOUR; on minute data that is 1.6 h, far beyond what
                          # an operator can act on. Training FOR 20 lets the
                          # models spend capacity on the near horizon instead
                          # of a long tail nobody uses.
# Budget follows the long-horizon forecasting convention. The Time-Series-
# Library benchmark (Informer / Autoformer / FEDformer / PatchTST / TimesNet /
# iTransformer) trains 10 epochs, batch 32, early-stop patience 3; Nixtla's own
# NeuralForecast benchmarks use max_steps=1000. On this data one epoch is
# ~23,279 train windows / 256 = ~91 steps, so 1000 steps ~= 11 epochs and the
# two conventions coincide. Early stopping means most models halt before that.
MAX_STEPS = 20 if SMOKE else 3000         # ~33 epochs; a ceiling, not a target
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
WINDOWS_BATCH_SIZE = 256
VAL_CHECK_STEPS = 10 if SMOKE else 50
# First full run stopped after ~400-700 steps and left GRU/LSTM/KAN with
# NEGATIVE R2 (worse than the mean), i.e. badly undertrained. Patience is
# raised so training only halts after a genuine plateau.
EARLY_STOP_PATIENCE = 2 if SMOKE else 15  # 15 x 50 = 750 steps of no gain

# Kaggle kills a session at 12 h. Past this point we stop starting new model
# runs and finalise what is already done, so a long run degrades to "fewer
# models fully evaluated" instead of "everything lost to a timeout".
TIME_BUDGET_S = 10.5 * 3600
N_WINDOWS = 8 if SMOKE else 80            # rolling forecast origins per split
RANDOM_SEED = int(os.environ.get("SEED", "42"))
TARGET = "y"

# Expanding-window rolling-origin cross-validation with DISJOINT test blocks.
#
# The previous scheme used five nested suffixes (60/20/20, 70/15/15, 80/10/10,
# 75/15/10, 90/05/05). Every test span ended at the same final timestamp, so
# S5 was a subset of S3 which was a subset of S2 which was a subset of S1 --
# and S3/S4 shared an identical test span outright, differing only in the
# train/val boundary. "Validated across five splits" therefore overstated the
# evidence: the folds were not independent observations.
#
# Each fold below tests on its own 10% block of the record. The blocks tile the
# final 50% without overlap, and training expands to everything preceding the
# fold's validation window, which is how the model would actually be deployed.
# Values are (train_end, val_end, test_end) as fractions of the row count.
_ALL_SPLITS = {
    "F1_test50_60": (0.45, 0.50, 0.60),
    "F2_test60_70": (0.55, 0.60, 0.70),
    "F3_test70_80": (0.65, 0.70, 0.80),
    "F4_test80_90": (0.75, 0.80, 0.90),
    "F5_test90_100": (0.85, 0.90, 1.00),
}
SPLITS = {"F1_test50_60": _ALL_SPLITS["F1_test50_60"]} if SMOKE else _ALL_SPLITS

SUB_HORIZONS = [1, 3, 4, 5, 6, 9, 12, 15]
NOISE_LEVELS = [0.0] if PROTOCOL == "reduced" else [0.0, 0.01, 0.05, 0.1]

# Models that produce multivariate batches -> must be predicted one window at a
# time because they are instantiated with n_series=1.
MULTIVARIATE = {"TSMixerx", "TimeXer"}

print(f"[*] Environment : {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"[*] Output dir  : {OUT}", flush=True)

ACCELERATOR = "cpu"
try:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        # A real kernel launch: is_available() alone stays True even when the
        # installed wheel has no kernels compiled for this card's architecture.
        try:
            _a = torch.randn(64, 64, device="cuda")
            _ = (_a @ _a).sum().item()
            ACCELERATOR = "gpu"
            print(f"[*] GPU         : {name} sm_{cap[0]}{cap[1]} - kernels OK "
                  f"(torch {torch.__version__})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] GPU         : {name} sm_{cap[0]}{cap[1]} present but "
                  f"UNUSABLE with torch {torch.__version__}: {exc}", flush=True)
            print("[!] Falling back to CPU - expect a much longer run.", flush=True)
    else:
        print(f"[*] GPU         : none (torch {torch.__version__})", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[!] torch check failed: {exc}", flush=True)

# ============================================================
# DATA
# ============================================================

print("\n[*] Loading fused data...", flush=True)
df = pd.read_csv(DATA_PATH)
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# Promote the requested channel to the target column. For the headline runs
# TARGET_CHANNEL == Y_SOURCE and column "y" is already correct, so this is a
# no-op; for the extended sweep the chosen channel replaces it.
if TARGET_CHANNEL != Y_SOURCE:
    if TARGET_CHANNEL not in df.columns:
        raise SystemExit("[X] TARGET_CH %s absent from %s" % (TARGET_CHANNEL, DS_NAME))
    # The outgoing target is not discarded — it becomes an ordinary exogenous
    # channel under its real name, so the y_roll_*/y_diff_* features stay valid
    # and the target sensor's temperature is still available as an input.
    df[Y_SOURCE] = df["y"]
    df["y"] = df[TARGET_CHANNEL]

# Drop every column that duplicates the *incoming* target: the channel itself
# and anything derived from it. Leaving any of these in would leak the target
# into its own inputs.
_leak = {TARGET_CHANNEL} | {
    c for c in df.columns if c.startswith(TARGET_CHANNEL + "_")}
if TARGET_CHANNEL == Y_SOURCE:
    _leak.add(Y_SOURCE)                      # headline run: original behaviour
EXOG = [c for c in df.columns
        if c not in ({"unique_id", "ds", TARGET, "segment_id"} | _leak)]
print("    target channel: %s  (dropped %d leaking cols: %s)"
      % (TARGET_CHANNEL, len(_leak & set(df.columns)),
         sorted(_leak & set(df.columns))), flush=True)
SEG = df["segment_id"].to_numpy()

N = len(df)
print(f"    rows={N:,}  exog={len(EXOG)}  span={df['ds'].min()} -> {df['ds'].max()}")
print(f"    exog list: {EXOG}", flush=True)

# NeuralForecast needs a regular grid; the source has 24 gaps. Windows are built
# positionally, so a synthetic contiguous timestamp axis is used for model I/O
# while the real timestamps are retained for reporting.
REAL_DS = df["ds"].to_numpy()
BASE_TS = pd.Timestamp("2025-01-01")
df["ds"] = BASE_TS + pd.to_timedelta(np.arange(N), unit="m")

VALUE_COLS = [TARGET] + EXOG


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """MSE / RMSE / R2 / MAPE / MAE / sMAPE, rounded to 4 dp."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return {k: float("nan") for k in
                ("MSE", "RMSE", "R2", "MAPE", "MAE", "sMAPE")}

    err = yt - yp
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    denom = np.maximum(np.abs(yt), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    sdenom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, 1e-8)
    smape = float(np.mean(np.abs(err) / sdenom) * 100.0)

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "MAE": round(mae, 4),
        "sMAPE": round(smape, 4),
    }


# ============================================================
# SPLITS / WINDOWS
# ============================================================

def split_bounds(train_frac, val_frac, test_frac):
    """Row indices for one fold. Fractions are cumulative END points."""
    return int(N * train_frac), int(N * val_frac), int(N * test_frac)


def make_origins(val_end, test_end):
    """Rolling forecast origins that lie WHOLLY inside one contiguous segment.

    The data has 21 gap-free segments separated by breaks of up to 74 hours.
    Choosing origins by row position alone let a window straddle a break, so a
    "96 minutes ahead" forecast was sometimes days ahead. That is what produced
    the inflated errors and the spurious negative R2 values in earlier runs:
    19.5% of windows were affected. A candidate origin o is valid only when
    rows [o - INPUT_SIZE, o + HORIZON) share a single segment_id.
    """
    first = max(val_end, INPUT_SIZE)
    last = min(test_end, N) - HORIZON
    if last <= first:
        return np.array([], dtype=int)
    cand = np.arange(first, last)
    ok = (SEG[cand - INPUT_SIZE] == SEG[cand + HORIZON - 1]) & \
         (SEG[cand] == SEG[cand - INPUT_SIZE])
    cand = cand[ok]
    if cand.size == 0:
        return cand
    n = min(N_WINDOWS, cand.size)
    return cand[np.round(np.linspace(0, cand.size - 1, n)).astype(int)]


def truth_matrix(origins):
    """Clean ground truth, shape [n_origins, HORIZON]."""
    yv = df[TARGET].to_numpy(dtype=np.float64)
    return np.stack([yv[o:o + HORIZON] for o in origins])


def standardize(train_end):
    """Z-score every channel using TRAIN-SET statistics only.

    NeuralForecast's scaler_type fits ONE scaler PER SERIES. Because each rolling
    origin is submitted as its own unique_id at predict time, that scaler would be
    fitted on a single 128-row window while training used the whole series - a
    train/inference distribution mismatch. Run 2 showed exactly that signature:
    models with internal normalisation (TiDE, TimeXer/use_norm, TSMixerx/revin)
    scored positive R2 while models relying on the NF scaler (LSTM, GRU, KAN) went
    negative. Standardising once, globally, with train statistics removes the
    mismatch, so scaler_type is left off entirely.
    """
    mu = df.iloc[:train_end][VALUE_COLS].mean()
    sd = df.iloc[:train_end][VALUE_COLS].std().replace(0.0, 1.0)
    out = df.copy()
    out[VALUE_COLS] = (out[VALUE_COLS] - mu) / sd
    return out, float(mu[TARGET]), float(sd[TARGET])


def noisy_frame(sigma, source, rng):
    """Gaussian noise on the inference inputs, in standardised units.

    After standardisation each training channel has unit variance, so drawing
    N(0, sigma) is exactly 'sigma x training std' in the original units.
    """
    if sigma == 0:
        return source
    out = source.copy()
    noise = rng.normal(0.0, sigma, size=(N, len(VALUE_COLS)))
    out[VALUE_COLS] = out[VALUE_COLS].to_numpy(dtype=np.float64) + noise
    return out


def history_batch(source, origins):
    """Stack the INPUT_SIZE-row history preceding each origin.

    Each origin becomes its own unique_id so a single predict() call covers all
    windows. Timestamps are synthetic and contiguous per window.
    """
    parts = []
    idx = BASE_TS + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
    for k, o in enumerate(origins):
        w = source.iloc[o - INPUT_SIZE:o].copy()
        w["unique_id"] = f"w{k:04d}"
        w["ds"] = idx
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def collect(pred_df, model_col, n_origins):
    """Reshape NF predictions into [n_origins, HORIZON] ordered by window id."""
    p = pred_df.sort_values(["unique_id", "ds"])
    out = np.full((n_origins, HORIZON), np.nan)
    for uid, g in p.groupby("unique_id", sort=False):
        k = int(str(uid)[1:])
        v = g[model_col].to_numpy(dtype=np.float64)[:HORIZON]
        out[k, :len(v)] = v
    return out


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(name):
    """Fresh instance of one model. hist_exog only where NF supports it."""
    from neuralforecast.models import (BiTCN, GRU, LSTM, NHITS, TFT, TiDE,
                                       TimeXer, TSMixerx, FEDformer)

    base = dict(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        val_check_steps=VAL_CHECK_STEPS,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        scaler_type="robust",   # TemporalNorm, applied per window (verified)
        random_seed=RANDOM_SEED,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        accelerator=ACCELERATOR,
        devices=1,   # Lightning rejects the CPUAccelerator default of -1
        gradient_clip_val=1.0,  # FEDformer diverged to all-NaN without this
    )
    win = dict(base, windows_batch_size=WINDOWS_BATCH_SIZE)
    fuse = dict(hist_exog_list=EXOG)

    if name in ("TiDE-Floor", "TSMixerx-Floor"):
        import torch
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.losses.pytorch import MAE as NFMAE

        class FloorNormalLoss(DistributionLoss):
            """Gaussian NLL with variance hard-floored at FLOOR_VAR. Verified
            against the real interface via smoke_floor_loss.py: get_distribution
            is the single hook _base_model.py calls for both training and every
            evaluation/prediction path, so overriding only it correctly threads
            the floor through everywhere without touching scale_decouple,
            domain_map, or __call__."""

            def __init__(self, floor_var, **kw):
                super().__init__(distribution="Normal", **kw)
                self.floor_var = float(floor_var)

            def get_distribution(self, distr_args, **distribution_kwargs):
                loc, scale = distr_args
                scale_floored = torch.sqrt(scale ** 2 + self.floor_var)
                return super().get_distribution((loc, scale_floored), **distribution_kwargs)

        floor_loss = FloorNormalLoss(FLOOR_VAR)
        if name == "TiDE-Floor":
            return TiDE(**win, **fuse, hidden_size=256,
                       loss=floor_loss, valid_loss=NFMAE(), alias="TiDE-Floor")
        return TSMixerx(**win, **fuse, n_series=1, n_block=2, ff_dim=64,
                        loss=floor_loss, valid_loss=NFMAE(), alias="TSMixerx-Floor")
    if name == "LSTM":
        return LSTM(**base, **fuse, encoder_hidden_size=128,
                    encoder_n_layers=2, decoder_hidden_size=128, alias="LSTM")
    if name == "GRU":
        return GRU(**base, **fuse, encoder_hidden_size=128,
                   encoder_n_layers=2, decoder_hidden_size=128, alias="GRU")
    if name == "FEDformer":
        # EXOGENOUS_HIST=False in NeuralForecast -> univariate by necessity.
        # n_head is hard-asserted to 8 in fedformer.py; any other value raises.
        # lr 1e-3 made FEDformer diverge to 100%% NaN predictions; the
        # Time-Series-Library convention for these transformers is 1e-4.
        return FEDformer(**{**win, "learning_rate": 1e-5}, hidden_size=64,
                         n_head=8, modes=32, encoder_layers=2,
                         decoder_layers=1, alias="FEDformer")
    if name == "NHITS":
        return NHITS(**win, **fuse, alias="NHITS")
    if name == "TiDE":
        return TiDE(**win, **fuse, hidden_size=256, alias="TiDE")
    if name == "BiTCN":
        return BiTCN(**win, **fuse, hidden_size=32, alias="BiTCN")
    if name == "TFT":
        # TFT runs a variable-selection network over every exogenous channel, so
        # memory scales with 52 features x 128 lookback x windows_batch_size and
        # it OOM'd at 3.38 GB on the P100. Smaller batches and hidden width keep
        # it inside 16 GB; the architecture and its inputs are unchanged.
        return TFT(**{**win, "windows_batch_size": 64,
                      "inference_windows_batch_size": 64},
                   **fuse, hidden_size=64, n_head=4,
                   dropout=0.1, alias="TFT")
    if name == "TSMixerx":
        return TSMixerx(**win, **fuse, n_series=1, n_block=2,
                        ff_dim=64, alias="TSMixerx")
    if name == "TimeXer":
        return TimeXer(**win, **fuse, n_series=1, hidden_size=128,
                       n_heads=4, e_layers=2, patch_len=16, alias="TimeXer")

    # ---- second wave -------------------------------------------------------
    # Added because the first roster produced a single winner: only TiDE beat
    # naive persistence on the temperature targets, so the whole result rested
    # on one architecture. These are all EXOGENOUS_HIST=True in NeuralForecast,
    # i.e. they can actually use the fused sensor channels. Imported lazily so
    # that a class missing from the installed version skips that model rather
    # than aborting the run.
    if name == "NBEATSx":
        from neuralforecast.models import NBEATSx
        return NBEATSx(**win, **fuse, alias="NBEATSx")
    if name == "MLP":
        from neuralforecast.models import MLP
        return MLP(**win, **fuse, hidden_size=256, num_layers=3, alias="MLP")
    if name == "TCN":
        from neuralforecast.models import TCN
        return TCN(**base, **fuse, encoder_hidden_size=128,
                   decoder_hidden_size=128, alias="TCN")
    if name == "DilatedRNN":
        from neuralforecast.models import DilatedRNN
        return DilatedRNN(**base, **fuse, encoder_hidden_size=128,
                          alias="DilatedRNN")
    if name == "KAN":
        from neuralforecast.models import KAN
        return KAN(**win, **fuse, alias="KAN")
    if name == "DeepNPTS":
        from neuralforecast.models import DeepNPTS
        return DeepNPTS(**win, **fuse, hidden_size=128, alias="DeepNPTS")
    raise ValueError(name)


# Roster after a pilot on the temperature targets. Dropped, with reasons:
#   FEDformer  all-NaN in 3/3 regions -- diverged under gradient clipping,
#              lr 1e-5 and reduced spectral modes alike; also
#              EXOGENOUS_HIST=False, so it cannot use the fused channels.
#   LSTM 6.06x / GRU 6.00x / TabNet 6.07x mean MSE relative to naive
#              persistence, in every region. A model six times worse than
#              "assume no change" carries no information about the data.
# This is roster selection and must be reported as such: the pilot and its
# numbers are disclosed, not the survivors alone.
NF_MODELS = ["NHITS", "TiDE", "BiTCN", "TFT", "TSMixerx", "TimeXer",
             "NBEATSx", "MLP", "TCN", "DilatedRNN", "KAN", "DeepNPTS"]
# Reference baselines - no training. Without them a model's R2 cannot be
# judged: on a strongly autocorrelated series persistence is already strong.
BASELINES = ["Naive-Persistence", "Naive-MA60"]
# TabNet is RETAINED for the vibration study. It was cut from the temperature
# roster at 6.07x persistence, but on vibration X it lands mid-pack (skill 0.22,
# ahead of BiTCN, NHITS, TSMixerx, NBEATSx and KAN). A model's failure on one
# target is not grounds for excluding it from a different one.
ALL_MODELS = NF_MODELS + ["TabNet"]

# ROSTER=<comma-list> restricts NF_MODELS to a named subset, keeping
# BASELINES always on (a model's R2 is meaningless without them). Used by P2
# (iterations 2/3 from-scratch training) to fit within Kaggle's free-tier
# weekly quota: the full 13-model roster over 30 targets was estimated at
# ~35-45 GPU-hours/account (more than the 30h/week allowance), so P2 trains
# only TiDE + TSMixerx -- the same restriction section 6.3 already applies to
# the transfer matrix itself, so P2's output matches what P4 actually needs
# rather than training 11 extra models nothing downstream uses.
_roster = os.environ.get("ROSTER", "")
if _roster:
    NF_MODELS = [m.strip() for m in _roster.split(",") if m.strip()]
    ALL_MODELS = NF_MODELS + (["TabNet"] if "TabNet" in _roster else [])

if FLOOR_MODE:
    # No TabNet, no naive-baseline retrain -- the baseline run already has
    # those, and this roster is compared against that run's TiDE/TSMixerx
    # rows at matching folds/horizons, not re-derived from scratch.
    # FLOOR_MODEL optionally restricts to a single variant -- used for the
    # 15-channel extension, where the 11-target pilot already showed
    # TiDE-Floor's effect is null (many folds at exactly 0.0% delta), so
    # re-running it everywhere would spend free-tier quota reproducing a
    # non-effect instead of covering more of the real TSMixerx-Floor signal.
    _floor_model = os.environ.get("FLOOR_MODEL", "")
    NF_MODELS = [_floor_model] if _floor_model else ["TiDE-Floor", "TSMixerx-Floor"]
    ALL_MODELS = list(NF_MODELS)

MODEL_META = {
    "Naive-Persistence": dict(year=0, family="Baseline (last value)", fusion=False),
    "Naive-MA60":        dict(year=0, family="Baseline (60-min mean)", fusion=False),
    "LSTM":      dict(year=1997, family="Recurrent",      fusion=True),
    "GRU":       dict(year=2014, family="Recurrent",      fusion=True),
    "FEDformer": dict(year=2022, family="Freq-Transformer", fusion=False),
    "NHITS":     dict(year=2023, family="MLP-Hierarchical", fusion=True),
    "TiDE":      dict(year=2023, family="MLP-Encoder",    fusion=True),
    "BiTCN":     dict(year=2023, family="Temporal-Conv",  fusion=True),
    "TFT":       dict(year=2021, family="Fusion-Transformer", fusion=True),
    "TSMixerx":  dict(year=2023, family="MLP-Mixer",      fusion=True),
    "TimeXer":   dict(year=2024, family="Exog-Transformer", fusion=True),
    "TabNet":    dict(year=2021, family="Tabular-Attentive", fusion=True),
    "NBEATSx":   dict(year=2022, family="MLP-Basis",       fusion=True),
    "MLP":       dict(year=2017, family="MLP-Plain",       fusion=True),
    "TCN":       dict(year=2018, family="Temporal-Conv",   fusion=True),
    "DilatedRNN": dict(year=2017, family="Recurrent-Dilated", fusion=True),
    "KAN":       dict(year=2024, family="Kolmogorov-Arnold", fusion=True),
    "DeepNPTS":  dict(year=2022, family="Non-Parametric",  fusion=True),
    "TiDE-Floor":     dict(year=2023, family="MLP-Encoder (floor-aware NLL)", fusion=True),
    "TSMixerx-Floor": dict(year=2023, family="MLP-Mixer (floor-aware NLL)",   fusion=True),
}


# ============================================================
# TABNET (sliding-window tabular regression)
# ============================================================

def tabnet_windows(source, starts):
    """Flatten [INPUT_SIZE x n_channels] histories into tabular rows."""
    vals = source[VALUE_COLS].to_numpy(dtype=np.float32)
    return np.stack([vals[s - INPUT_SIZE:s].ravel() for s in starts])


def run_tabnet(train_end, val_end, origins, y_true, rng, split_name, results,
               dfs, y_mu, y_sd):
    from pytorch_tabnet.tab_model import TabNetRegressor

    yv = dfs[TARGET].to_numpy(dtype=np.float32)
    stride = 4  # subsample training origins to keep the design matrix tractable

    tr = np.arange(INPUT_SIZE, train_end - HORIZON, stride)
    va = np.arange(train_end + INPUT_SIZE, val_end - HORIZON, stride)
    if len(tr) < 32 or len(va) < 8:
        print("    [!] TabNet: not enough windows, skipping", flush=True)
        return None

    Xtr = tabnet_windows(dfs, tr)
    Ytr = np.stack([yv[s:s + HORIZON] for s in tr])
    Xva = tabnet_windows(dfs, va)
    Yva = np.stack([yv[s:s + HORIZON] for s in va])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysd = Ytr.mean(), Ytr.std() + 1e-8

    model = TabNetRegressor(n_d=32, n_a=32, n_steps=4, gamma=1.5,
                            n_independent=2, n_shared=2, seed=RANDOM_SEED,
                            verbose=0,
                            device_name="cuda" if ACCELERATOR == "gpu" else "cpu")
    t0 = time.time()
    model.fit(((Xtr - mu) / sd), ((Ytr - ymu) / ysd),
              eval_set=[(((Xva - mu) / sd), ((Yva - ymu) / ysd))],
              max_epochs=3 if SMOKE else 100,
              patience=3 if SMOKE else 15, batch_size=512,
              virtual_batch_size=128, drop_last=False)
    train_time = time.time() - t0
    print(f"    TabNet trained in {train_time:.1f}s", flush=True)

    preds = {}
    for sigma in NOISE_LEVELS:
        src = noisy_frame(sigma, dfs, rng)
        Xte = tabnet_windows(src, origins)
        t1 = time.time()
        yp = (model.predict((Xte - mu) / sd) * ysd + ymu) * y_sd + y_mu
        infer = time.time() - t1
        preds[sigma] = yp.astype(np.float32)
        record(results, "TabNet", split_name, sigma, y_true, yp,
               train_time, infer)
    return preds


# ============================================================
# RESULT RECORDING
# ============================================================

def record(results, model, split, sigma, y_true, y_pred, train_time, infer_time):
    """Append overall, per-sub-horizon and per-step rows for one run."""
    m = compute_metrics(y_true, y_pred)
    m.update(model=model, split=split, noise_sigma=sigma,
             n_windows=int(y_true.shape[0]), horizon=HORIZON,
             train_time_s=round(float(train_time), 4),
             infer_time_s=round(float(infer_time), 4),
             fusion=MODEL_META[model]["fusion"],
             family=MODEL_META[model]["family"])
    results["main"].append(m)

    for h in SUB_HORIZONS:
        hm = compute_metrics(y_true[:, :h], y_pred[:, :h])
        hm.update(model=model, split=split, noise_sigma=sigma, horizon=h)
        results["horizon"].append(hm)

    ae = np.abs(y_true - y_pred)
    se = (y_true - y_pred) ** 2
    for step in range(HORIZON):
        results["step"].append(dict(
            model=model, split=split, noise_sigma=sigma, step=step + 1,
            MAE=round(float(np.nanmean(ae[:, step])), 4),
            RMSE=round(float(np.sqrt(np.nanmean(se[:, step]))), 4),
        ))

    print(f"      sigma={sigma:<5} MSE={m['MSE']:>10.4f} RMSE={m['RMSE']:>9.4f} "
          f"R2={m['R2']:>8.4f} MAPE={m['MAPE']:>8.4f}", flush=True)


def flush_results(results, preds_store):
    pd.DataFrame(results["main"]).to_csv(OUT / "metrics_main.csv", index=False)
    pd.DataFrame(results["horizon"]).to_csv(OUT / "metrics_horizon.csv", index=False)
    pd.DataFrame(results["step"]).to_csv(OUT / "metrics_per_step.csv", index=False)
    np.savez_compressed(OUT / "predictions.npz", **preds_store)
    with open(OUT / "run_config.json", "w") as f:
        json.dump(dict(input_size=INPUT_SIZE, horizon=HORIZON,
                       max_steps=MAX_STEPS, n_windows=N_WINDOWS,
                       learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                       splits=SPLITS,
                       split_scheme="expanding-window rolling-origin CV; "
                                    "5 disjoint 10% test blocks tiling the "
                                    "final 50%; fractions are cumulative "
                                    "(train_end, val_end, test_end)",
                       noise_levels=NOISE_LEVELS,
                       sub_horizons=SUB_HORIZONS, exog=EXOG,
                       target=TARGET, target_channel=TARGET_CHANNEL,
                       y_source_in_published_csv=Y_SOURCE,
                       target_kind=("temperature" if "temperature" in TARGET_CHANNEL
                                    else "vibration"),
                       dataset=DS_NAME, region=REGION_DIR, iteration=1,
                       protocol=PROTOCOL, dropped_as_leakage=sorted(_leak),
                       models=ALL_MODELS,
                       model_meta=MODEL_META, seed=RANDOM_SEED,
                       n_rows=int(N)), f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    from neuralforecast import NeuralForecast

    results = {"main": [], "horizon": [], "step": []}
    preds_store = {}
    t_start = time.time()

    for split_name, (tr_pct, va_pct, te_pct) in SPLITS.items():
        train_end, val_end, test_end = split_bounds(tr_pct, va_pct, te_pct)
        origins = make_origins(val_end, test_end)
        if origins.size == 0:
            print(f"[!] {split_name}: test span too small, skipped", flush=True)
            continue

        y_true = truth_matrix(origins)
        rng = np.random.default_rng(RANDOM_SEED)

        print(f"\n{'=' * 74}")
        print(f"  FOLD {split_name}  train=[0,{tr_pct:.0%}) "
              f"val=[{tr_pct:.0%},{va_pct:.0%}) test=[{va_pct:.0%},{te_pct:.0%})")
        print(f"  rows: train={train_end:,} val={val_end - train_end:,} "
              f"test={test_end - val_end:,} | origins={len(origins)}")
        print(f"{'=' * 74}", flush=True)

        preds_store[f"ytrue|{split_name}"] = y_true.astype(np.float32)
        preds_store[f"origins|{split_name}"] = origins.astype(np.int32)

        dfs, y_mu, y_sd = standardize(train_end)
        train_val = dfs.iloc[:val_end]
        val_size = val_end - train_end

        for name in NF_MODELS:
            if time.time() - t_start > TIME_BUDGET_S:
                print(f"\n  [!] time budget reached - skipping {name} onward",
                      flush=True)
                break
            print(f"\n  --- {name} [{split_name}] ---", flush=True)
            try:
                nf = NeuralForecast(models=[build_model(name)], freq="1min")
                t0 = time.time()
                nf.fit(df=train_val, val_size=val_size, verbose=False)
                train_time = time.time() - t0
                print(f"    trained in {train_time:.1f}s", flush=True)

                for sigma in NOISE_LEVELS:
                    src = noisy_frame(sigma, dfs, rng)
                    t1 = time.time()
                    if name in MULTIVARIATE or name.replace("-Floor", "") in MULTIVARIATE:
                        rows = []
                        for k, o in enumerate(origins):
                            w = src.iloc[o - INPUT_SIZE:o].copy()
                            w["unique_id"] = DS_NAME
                            w["ds"] = BASE_TS + pd.to_timedelta(
                                np.arange(INPUT_SIZE), unit="m")
                            p = nf.predict(df=w, verbose=False)
                            p = p.reset_index() if "unique_id" not in p.columns else p
                            rows.append(p[name].to_numpy(dtype=np.float64)[:HORIZON])
                        y_pred = np.stack(rows) * y_sd + y_mu
                    else:
                        batch = history_batch(src, origins)
                        p = nf.predict(df=batch, verbose=False)
                        p = p.reset_index() if "unique_id" not in p.columns else p
                        y_pred = collect(p, name, len(origins)) * y_sd + y_mu
                    infer = time.time() - t1

                    record(results, name, split_name, sigma, y_true, y_pred,
                           train_time, infer)
                    preds_store[f"{name}|{split_name}|{sigma}"] = y_pred.astype(np.float32)

                del nf
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

            except Exception as exc:  # noqa: BLE001
                print(f"    [X] {name} FAILED: {exc}", flush=True)
                traceback.print_exc()

            flush_results(results, preds_store)

        # ---- naive baselines ----
        for sigma in NOISE_LEVELS:
            src = noisy_frame(sigma, dfs, rng)
            sv = src[TARGET].to_numpy(dtype=np.float64)
            last = np.stack([np.full(HORIZON, sv[o - 1]) for o in origins])
            ma = np.stack([np.full(HORIZON, sv[o - 60:o].mean()) for o in origins])
            for nm, arr in (("Naive-Persistence", last), ("Naive-MA60", ma)):
                yp = arr * y_sd + y_mu
                record(results, nm, split_name, sigma, y_true, yp, 0.0, 0.0)
                preds_store[f"{nm}|{split_name}|{sigma}"] = yp.astype(np.float32)

        # ---- TabNet ----
        # Guarded by membership in ALL_MODELS. NOTE: an earlier version of this
        # guard only wrapped the time-budget check, leaving the call below it
        # unconditional, so TabNet ran even when excluded. The whole block is
        # now behind the membership test.
        if "TabNet" not in ALL_MODELS:
            continue
        if time.time() - t_start > TIME_BUDGET_S:
            print("\n  [!] time budget reached - skipping TabNet", flush=True)
            continue
        print(f"\n  --- TabNet [{split_name}] ---", flush=True)
        try:
            tp = run_tabnet(train_end, val_end, origins, y_true, rng,
                            split_name, results, dfs, y_mu, y_sd)
            if tp:
                for sigma, arr in tp.items():
                    preds_store[f"TabNet|{split_name}|{sigma}"] = arr
        except Exception as exc:  # noqa: BLE001
            print(f"    [X] TabNet FAILED: {exc}", flush=True)
            traceback.print_exc()

        flush_results(results, preds_store)
        print(f"\n  [checkpoint] {split_name} done, "
              f"elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)

    # ---- final summaries ----
    flush_results(results, preds_store)
    main_df = pd.DataFrame(results["main"])
    if not main_df.empty:
        clean = main_df[main_df.noise_sigma == 0.0]
        for metric in ("MSE", "RMSE", "R2", "MAPE", "MAE", "sMAPE"):
            clean.pivot_table(index="model", columns="split", values=metric) \
                 .round(4).to_csv(OUT / f"summary_{metric}.csv")
        print("\n" + "=" * 74)
        print("  CLEAN (sigma=0) RESULTS - RMSE by split")
        print("=" * 74)
        print(clean.pivot_table(index="model", columns="split",
                                values="RMSE").round(4).to_string())

    print(f"\n[DONE] total {(time.time() - t_start) / 60:.1f} min, "
          f"{len(results['main'])} metric rows", flush=True)


if __name__ == "__main__":
    main()
