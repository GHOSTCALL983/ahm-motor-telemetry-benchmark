"""Shared constants and paths for the whole project.

Device labels (AHM-01 ... AHM-11) are anonymised. The mapping from labels to
the raw export files is site-confidential and is read from a private JSON file
that is NOT part of this release (set RAW_CONFIG to its path). Every script
that only uses the released, processed data works without it.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the release folder
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
PRED_DIR = RESULTS / "trained_model_predictions"

# ---------------------------------------------------------------- physics
G0 = 9.80665                       # standard gravity, m/s^2 (ISO 80000-3)
RTD_A, RTD_B = 3.9083e-3, -5.775e-7  # Callendar-Van Dusen coefficients (IEC 60751)
NOM_FIRMWARE, NOM_TRUE = 97.0, 100.0  # PT100 nominal resistance: as configured / correct
TLIM = (-20.0, 250.0)              # plausible operating envelope, degC

# ---------------------------------------------------------------- dataset design
INPUT_SIZE, HORIZON_BUILD = 128, 18   # used only for minimum segment length
MIN_SEG = INPUT_SIZE + HORIZON_BUILD + 80
NEED_RUN = 466                         # minutes a run must last to be usable for windows
PRUNE_BASIS, PRUNE_THR = 0.45, 0.98    # collinearity pruning: first 45% of rows, |r| > 0.98

REGIONS = {                            # reference (target) sensor first, then neighbours by role
    "R1": {"zone": "Hot",  "self": "AHM-01", "nb1": "AHM-02", "nb2": "AHM-03"},
    "R2": {"zone": "Mid",  "self": "AHM-06", "nb1": "AHM-04", "nb2": "AHM-05"},
    "R3": {"zone": "Cool", "self": "AHM-08", "nb1": "AHM-07", "nb2": None},
}
RETAINED = ["AHM-01", "AHM-02", "AHM-03", "AHM-04", "AHM-05",
            "AHM-06", "AHM-07", "AHM-08"]
EXCLUDED = {"AHM-09": "fusion viability: uptime misaligned with its region (not a fault)",
            "AHM-10": "measured fault: low coverage, temperature spike, stuck axis",
            "AHM-11": "measured fault: very low coverage, temperature spike"}
TARGETS = ["self_temperature_one", "self_temperature_two",
           "self_x_ac", "self_y_ac", "self_z_ac"]
FOLDS = ["F1_test50_60", "F2_test60_70", "F3_test70_80", "F4_test80_90", "F5_test90_100"]
FOLD_FRAC = {"F1_test50_60": (0.45, 0.50, 0.60), "F2_test60_70": (0.55, 0.60, 0.70),
             "F3_test70_80": (0.65, 0.70, 0.80), "F4_test80_90": (0.75, 0.80, 0.90),
             "F5_test90_100": (0.85, 0.90, 1.00)}
TRAINED_MODELS = ["BiTCN", "DeepNPTS", "DilatedRNN", "KAN", "MLP", "NBEATSx", "NHITS",
                  "TCN", "TFT", "TSMixerx", "TabNet", "TiDE", "TimeXer"]


def kind_of(target):
    return "temperature" if "temperature" in target else "vibration"


def dataset_path(region, iteration):
    return DATA / f"{region}_I{iteration}.csv"


def env_path(name, why):
    """Location of private material, given only through an environment variable:
    PRIVATE_CACHE  stage-1 cache built from the raw export (site-confidential)
    ORIGINAL_ROOT  the original working project, needed only by the audit
                   scripts that compare the release against it."""
    p = os.environ.get(name)
    if not p:
        raise SystemExit(f"{name} is not set ({why}).")
    return Path(p)


def raw_config():
    p = os.environ.get("RAW_CONFIG")
    if not p or not Path(p).exists():
        raise SystemExit("RAW_CONFIG not set: the raw export is site-confidential and "
                         "only needed to rebuild the dataset from scratch.")
    c = json.loads(Path(p).read_text())
    return Path(c["raw_dir"]), c["files"]
