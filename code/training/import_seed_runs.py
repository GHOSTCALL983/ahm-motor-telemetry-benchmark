"""Copy the extra-seed trained runs into the release (neutral names, sanitised configs).

The original study re-trained the 14 iteration-1 runs at two further seeds
(101 and 202), with everything else identical to the stored seed-42 runs, to
measure ordinary training variance. Region 1 / self_x_ac was not repeated: it
was the first full run, launched on its own as a pre-launch check, and the
seed queue reused the queue's target list, which did not contain it.
Source: ORIGINAL_ROOT/study_canonical/Iteration_1/Region_r/<target>/results_seed<s>/

Output: results/trained_model_predictions_extra_seeds/R{r}_I1__{target}__seed{s}.npz (+ .config.json)
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, TARGETS, env_path  # noqa: E402

SRC = env_path("ORIGINAL_ROOT", "original project holding the trained runs") / "study_canonical" / "Iteration_1"
DST = RESULTS / "trained_model_predictions_extra_seeds"
SEEDS = (101, 202)
KEEP = ["input_size", "horizon", "max_steps", "n_windows", "learning_rate", "batch_size", "splits",
        "split_scheme", "noise_levels", "sub_horizons", "exog", "target_channel", "target_kind",
        "protocol", "dropped_as_leakage", "models", "seed", "n_rows"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    n, missing = 0, []
    for r in (1, 2, 3):
        for t in TARGETS:
            for s in SEEDS:
                src = SRC / f"Region_{r}" / t / f"results_seed{s}"
                stem = f"R{r}_I1__{t}__seed{s}"
                if not (src / "predictions.npz").exists():
                    missing.append(stem)
                    continue
                shutil.copyfile(src / "predictions.npz", DST / f"{stem}.npz")
                c = json.loads((src / "run_config.json").read_text())
                clean = {k: c[k] for k in KEEP if k in c}
                clean.update(dataset=f"R{r}_I1", region=f"R{r}", iteration=1)
                (DST / f"{stem}.config.json").write_text(json.dumps(clean, indent=2))
                n += 1
    print(f"[ok] imported {n} extra-seed runs; not present: {missing}")


if __name__ == "__main__":
    main()
