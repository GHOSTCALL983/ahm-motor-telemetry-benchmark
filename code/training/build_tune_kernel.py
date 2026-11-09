"""Rebuild the hyperparameter-check kernel exactly as it was run:
   train_forecasters.py up to (not including) its METRICS section
 + train_forecasters.py's SPLITS / WINDOWS section (up to MODEL FACTORY)
 + tune_tail.py
Folds, origins, preprocessing and leakage handling are therefore identical to
the stored trained runs. Output: tune_forecasters.py (next to this file).
On Kaggle, the queue runner prepends DS / TARGET_CH / TUNE_MODELS settings.
"""
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "train_forecasters.py").read_text(encoding="utf-8").splitlines(keepends=True)


def marker(title):
    for i, line in enumerate(src):
        if line.strip() == f"# {title}" and src[i - 1].startswith("# ====="):
            return i - 1
    raise SystemExit(f"section {title!r} not found")


part1 = src[:marker("METRICS")]
part2 = src[marker("SPLITS / WINDOWS"):marker("MODEL FACTORY")]
out = "".join(part1 + part2) + (HERE / "tune_tail.py").read_text(encoding="utf-8")
(HERE / "tune_forecasters.py").write_text(out, encoding="utf-8")
print(f"[ok] tune_forecasters.py: {out.count(chr(10))} lines")
