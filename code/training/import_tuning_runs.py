"""Copy the hyperparameter-check runs into the release with neutral names and
sanitised configurations.

Source: tune_forecasters.py (train_forecasters.py + tune_tail.py), run on Kaggle
(GPU Tesla T4, torch 2.11.0+cu130), one run per (dataset, target) for 9 datasets x 2 targets
(self_x_ac, self_temperature_one). Each run trains TiDE, TimeXer and NHITS with
8 configurations each (configuration 0 = the original setting) on all 5 folds
and stores validation-block and test-origin forecasts. The original location is
ORIGINAL_ROOT/study_tuning (not part of the release).

Output per run: results/tuning/{R}_{I}__{target}.npz, .config.json, .log.csv
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, env_path  # noqa: E402

SRC = env_path("ORIGINAL_ROOT", "original project holding the tuning runs") / "study_tuning"
DST = RESULTS / "tuning"
CHANNELS = ["self_x_ac", "self_temperature_one"]
KEEP = ["target_channel", "configs", "seed", "input_size", "horizon", "max_steps", "n_windows",
        "splits", "exog", "dropped_as_leakage"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    n, missing = 0, []
    for i in (1, 2, 3):
        for r in (1, 2, 3):
            for ch in CHANNELS:
                hits = sorted(SRC.glob(f"*R{r}_I{i}"))     # source folders carry a site prefix
                src = (hits[0] if hits else SRC / f"R{r}_I{i}") / ch
                stem = f"R{r}_I{i}__{ch}"
                if not (src / "tuning_configs.json").exists():
                    missing.append(stem)
                    continue
                shutil.copyfile(src / "tuning_predictions.npz", DST / f"{stem}.npz")
                shutil.copyfile(src / "tuning_log.csv", DST / f"{stem}.log.csv")
                c = json.loads((src / "tuning_configs.json").read_text())
                clean = {k: c[k] for k in KEEP if k in c}
                clean.update(dataset=re.sub(r"^.*_(R\d_I\d)$", r"\1", c["dataset"]),
                             region=f"R{r}", iteration=i)
                (DST / f"{stem}.config.json").write_text(json.dumps(clean, indent=2))
                n += 1
    print(f"[ok] imported {n} tuning runs; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
