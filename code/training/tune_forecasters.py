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
# HYPERPARAMETER CHECK (appended; everything above is verbatim
# from train_forecasters.py so folds, origins and preprocessing
# are identical to the stored runs)
# ============================================================
# For each model, config 0 is the ORIGINAL setting; configs 1..7 are drawn
# without replacement from a fixed grid with a fixed seed (decided before any
# run). Every config is trained on each fold exactly as in the original study,
# then predicts (a) the fold's VALIDATION block and (b) the fold's TEST origins.
# Selection happens OFFLINE and uses ONLY the validation predictions.
import itertools  # noqa: E402
import random  # noqa: E402

if os.environ.get("TUNE_OUT"):                     # keep tuning output away from the stored runs
    OUT = Path(os.environ["TUNE_OUT"])
    OUT.mkdir(parents=True, exist_ok=True)
TUNE_MODELS = [m.strip() for m in os.environ.get("TUNE_MODELS", "TiDE,TimeXer,NHITS").split(",") if m.strip()]
N_CONFIGS = 2 if SMOKE else int(os.environ.get("N_CONFIGS", "8"))

ORIGINAL = {
    "TiDE": dict(hidden_size=256, dropout=0.3, learning_rate=1e-3, num_encoder_layers=1),
    "NHITS": dict(width=512, dropout_prob_theta=0.0, learning_rate=1e-3),
    "TimeXer": dict(hidden_size=128, dropout=0.1, learning_rate=1e-3),
}
GRID = {
    "TiDE": dict(hidden_size=[128, 256, 512], dropout=[0.1, 0.3], learning_rate=[3e-4, 1e-3, 3e-3],
                 num_encoder_layers=[1, 2]),
    "NHITS": dict(width=[256, 512, 1024], dropout_prob_theta=[0.0, 0.2], learning_rate=[3e-4, 1e-3, 3e-3]),
    "TimeXer": dict(hidden_size=[64, 128, 256], dropout=[0.1, 0.3], learning_rate=[3e-4, 1e-3, 3e-3]),
}


def configs_for(model):
    keys = list(GRID[model])
    allc = [dict(zip(keys, v)) for v in itertools.product(*[GRID[model][k] for k in keys])]
    allc = [c for c in allc if c != ORIGINAL[model]]
    random.Random(2026).shuffle(allc)
    return [ORIGINAL[model]] + allc[:N_CONFIGS - 1]


def build_tuned(model, cfg):
    from neuralforecast.models import NHITS, TiDE, TimeXer
    base = dict(h=HORIZON, input_size=INPUT_SIZE, max_steps=MAX_STEPS, learning_rate=cfg["learning_rate"],
                batch_size=BATCH_SIZE, val_check_steps=VAL_CHECK_STEPS,
                early_stop_patience_steps=EARLY_STOP_PATIENCE, scaler_type="robust", random_seed=RANDOM_SEED,
                enable_progress_bar=False, logger=False, enable_model_summary=False,
                accelerator=ACCELERATOR, devices=1, gradient_clip_val=1.0,
                windows_batch_size=WINDOWS_BATCH_SIZE, hist_exog_list=EXOG)
    if model == "TiDE":
        return TiDE(**base, hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
                    num_encoder_layers=cfg["num_encoder_layers"], alias="TiDE")
    if model == "NHITS":
        w = cfg["width"]
        return NHITS(**base, mlp_units=[[w, w]] * 3, dropout_prob_theta=cfg["dropout_prob_theta"], alias="NHITS")
    if model == "TimeXer":
        return TimeXer(**base, n_series=1, hidden_size=cfg["hidden_size"], n_heads=4, e_layers=2,
                       patch_len=16, dropout=cfg["dropout"], alias="TimeXer")
    raise ValueError(model)


def predict(nf, model, src, origins, y_mu, y_sd):
    if model in MULTIVARIATE:
        rows = []
        for o in origins:
            w = src.iloc[o - INPUT_SIZE:o].copy()
            w["unique_id"] = DS_NAME
            w["ds"] = BASE_TS + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
            p = nf.predict(df=w, verbose=False)
            p = p.reset_index() if "unique_id" not in p.columns else p
            rows.append(p[model].to_numpy(dtype=np.float64)[:HORIZON])
        return np.stack(rows) * y_sd + y_mu
    p = nf.predict(df=history_batch(src, origins), verbose=False)
    p = p.reset_index() if "unique_id" not in p.columns else p
    return collect(p, model, len(origins)) * y_sd + y_mu


def tune_main():
    from neuralforecast import NeuralForecast
    store, log = {}, []
    t_start = time.time()
    cfgs = {m: configs_for(m) for m in TUNE_MODELS}
    for split_name, (tr_pct, va_pct, te_pct) in SPLITS.items():
        train_end, val_end, test_end = split_bounds(tr_pct, va_pct, te_pct)
        o_te = make_origins(val_end, test_end)
        o_va = make_origins(train_end, val_end)       # same rule, applied to the validation block
        store[f"origins_test|{split_name}"] = o_te.astype(np.int32)
        store[f"origins_val|{split_name}"] = o_va.astype(np.int32)
        store[f"ytrue_test|{split_name}"] = truth_matrix(o_te).astype(np.float32)
        store[f"ytrue_val|{split_name}"] = truth_matrix(o_va).astype(np.float32)
        dfs, y_mu, y_sd = standardize(train_end)
        train_val = dfs.iloc[:val_end]
        for model in TUNE_MODELS:
            for k, cfg in enumerate(cfgs[model]):
                if time.time() - t_start > TIME_BUDGET_S:
                    print("[!] time budget reached", flush=True)
                    break
                t0 = time.time()
                try:
                    nf = NeuralForecast(models=[build_tuned(model, cfg)], freq="1min")
                    nf.fit(df=train_val, val_size=val_end - train_end, verbose=False)
                    store[f"{model}|cfg{k}|{split_name}|val"] = predict(nf, model, dfs, o_va, y_mu, y_sd).astype(np.float32)
                    store[f"{model}|cfg{k}|{split_name}|test"] = predict(nf, model, dfs, o_te, y_mu, y_sd).astype(np.float32)
                    status = "ok"
                except Exception as exc:  # noqa: BLE001
                    status = f"FAILED: {exc}"
                    traceback.print_exc()
                log.append(dict(model=model, cfg=k, fold=split_name, seconds=round(time.time() - t0, 1),
                                status=status, **cfg))
                print(f"{split_name} {model} cfg{k} {cfg} {status} {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT / "tuning_predictions.npz", **store)
                pd.DataFrame(log).to_csv(OUT / "tuning_log.csv", index=False)
    with open(OUT / "tuning_configs.json", "w") as f:
        json.dump(dict(dataset=DS_NAME, target_channel=TARGET_CHANNEL, configs=cfgs, seed=RANDOM_SEED,
                       input_size=INPUT_SIZE, horizon=HORIZON, max_steps=MAX_STEPS, n_windows=N_WINDOWS,
                       splits=SPLITS, exog=EXOG, dropped_as_leakage=sorted(_leak)), f, indent=2)
    print(f"[DONE] {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    tune_main()
