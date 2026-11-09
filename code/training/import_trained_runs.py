"""Copy the 45 trained-model runs (3 iterations x 3 regions x 5 targets) into the
release with neutral names and sanitised run configurations.

Source runs were produced by train_forecasters.py on Kaggle (GPU P100), one run
per (dataset, target): 13 neural models + 2 naive baselines, 5 folds, 80
forecast origins per fold, horizon 15 min, seed 42. The original location is
given by ORIGINAL_STUDY_DIR (not part of the release).

Output per run: results/trained_model_predictions/{R}_{I}__{target}.npz and
               .../{R}_{I}__{target}.config.json
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env_path, PRED_DIR, TARGETS  # noqa: E402

SRC = env_path("ORIGINAL_ROOT", "original project holding the trained runs") / "study_canonical"
KEEP = ["input_size", "horizon", "max_steps", "n_windows", "learning_rate", "batch_size", "splits",
        "split_scheme", "noise_levels", "sub_horizons", "exog", "target_channel", "target_kind",
        "protocol", "dropped_as_leakage", "models", "seed", "n_rows"]


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in (1, 2, 3):
        for r in (1, 2, 3):
            for t in TARGETS:
                src = SRC / f"Iteration_{i}" / f"Region_{r}" / t / "results"
                stem = f"R{r}_I{i}__{t}"
                shutil.copyfile(src / "predictions.npz", PRED_DIR / f"{stem}.npz")
                c = json.loads((src / "run_config.json").read_text())
                ds = re.sub(r"^.*_(R\d_I\d)$", r"\1", c["dataset"])
                clean = {k: c[k] for k in KEEP if k in c}
                clean.update(dataset=ds, region=f"R{r}", iteration=i,
                             note="'iteration' corrected here; the original training config "
                                  "hard-coded iteration=1 for every run (metadata bug only)")
                (PRED_DIR / f"{stem}.config.json").write_text(json.dumps(clean, indent=2))
                n += 1
    print(f"[ok] imported {n} runs")


if __name__ == "__main__":
    main()
