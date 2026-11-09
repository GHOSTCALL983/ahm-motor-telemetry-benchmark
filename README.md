# Steel-plant motor telemetry: dataset, forecasting results and audit

Minute-level condition-monitoring data from electric motors at a steel plant in
Eastern India (site and devices anonymised), the forecasts of 13 trained neural
models, training-free and pretrained baselines, and a full audit of every
earlier claim. Start with **`AUDIT_REPORT.md`**, then **`results/KEY_RESULTS.md`**.

## Folder structure

```
data/                              released dataset (9 files) + documentation
  R{1,2,3}_I{1,2,3}.csv            region x 18-day iteration, 1-minute rows
  manifest.csv, schema.json        per-file statistics; column list and pruning record
  DEVICE_LABELS.csv                anonymised devices, roles, exclusions with reasons
  anomaly_intervals.csv            motor stops, vibration spikes, temperature jumps
                                   (flagged by fixed rules; NOT removed from the files)
code/
  common.py                        shared constants and paths
  data_build/01_..., 02_...        raw export -> released files (needs private raw data)
  training/train_forecasters.py    neural training script exactly as run (identifiers redacted)
  training/transfer_*.py           cross-domain transfer training / evaluation, as run
  training/import_trained_runs.py  brings the 45 trained runs into results/
  training/import_seed_runs.py     brings the 28 extra-seed runs (seeds 101, 202) into results/
  training/tune_tail.py, build_tune_kernel.py, tune_queue_runner.py, import_tuning_runs.py
                                   hyperparameter check (Kaggle GPU), as run
  analysis/b01 ... b10             new experiments, tables, figures, key results
                                   (b09 = tuning check, b10 = training-seed variability)
  audit/a01 ... a08                verification of the data and of every earlier claim
                                   (a06 = final independent pass, a07 = diverged forecasts,
                                   a08 = pilot that set the roster)
  paper/make_paper_tables.py       LaTeX tables of the paper and its supplement, from results/
  paper/check_paper_numbers.py     checks every number quoted in the manuscript against results/
results/
  trained_model_predictions/       45 runs: forecasts of 13 neural models + 2 naive, 5 folds
  trained_model_predictions_extra_seeds/
                                   14 iteration-1 runs re-trained at seeds 101 and 202
  training_free_predictions/       same origins: naive, closed-form, pretrained forecasts
  tuning/                          18 runs: TiDE/TimeXer/NHITS x 8 configurations, validation
                                   and test forecasts, logs, configs; JOB_ACCOUNTS.csv
  tables/                          every result table (CSV)
  audit/                           every audit check (CSV)
  transfer/                        transfer matrices (merge table flagged; see audit)
  KEY_RESULTS.md                   generated summary of the main numbers
figures/                           fig1-fig6 (generated from the tables)
AUDIT_REPORT.md                    what was verified, what was wrong, what changed
```

## The data in one paragraph

Three regions (thermal zones: hot, mid, cool) x three 18-day windows
(Jun 24-Jul 12, Jul 12-Jul 30, Aug 15-Sep 2, 2025), 157,707 one-minute rows in
gap-free segments. Coverage inside a window depends on device uptime: 58-96% of
the window's minutes are usable (`results/audit/iteration_windows.csv`), and the
retained rows of each file are listed in `data/manifest.csv` (for example,
Region 3's iteration-3 file ends on Aug 26). Each row carries, for the
region's reference motor ("self") and its neighbours ("nb1", "nb2"): drive-end
bearing temperature
(`temperature_one`), motor-body temperature (`temperature_two`), 3-axis vibration
AC amplitude (`x_ac`, `y_ac`, `z_ac`, m/s2, trailing 3-min standard deviation),
static acceleration (`vib_dc`), plus causal rolling features. Column `y` equals
`self_temperature_one`. Region 3 has one neighbour; its `nb2_*` columns are zero.
Temperatures are corrected for a firmware PT100 nominal-resistance error;
accelerations are calibrated per device against standard gravity.

## Reproduce

All analysis runs on a CPU from the released files:

```
pip install -r requirements.txt
cd code/analysis
python b01_training_free_forecasts.py                      # ~8 min; skips runs already written,
                                                           # delete results/training_free_predictions/*.npz to redo
# optional, not used in the reported results (about 6 CPU-hours):
# python b01_training_free_forecasts.py --chronos2-sensors
python b02_accuracy.py
python b03_recalibration_causal.py
python b04_error_decomposition.py
python b05_predictability.py
python b06_anomaly_sensitivity.py    # anomaly flags + results without them
python b07_figures.py
python b09_tuning_check.py           # uses the stored tuning forecasts in results/tuning
python b10_seed_variability.py       # uses the stored extra-seed forecasts
python ../audit/a07_divergence_robustness.py
python b08_key_results.py
python ../audit/a06_final_audit.py   # privacy scan, independent recomputation, report numbers
python ../paper/make_paper_tables.py # tables for the manuscript (written to paper_tables/)
```

Re-running the hyperparameter check itself needs a GPU (about 1.5 GPU-hours per
run on a Kaggle T4, 18 runs): `code/training/build_tune_kernel.py`, then
`tune_queue_runner.py` (see its header).

Before sharing a copy of this folder, delete every `__pycache__` directory:
Python's compiled files record the absolute local path of the source files.

Checks that compare against private material need environment variables:
`RAW_CONFIG` (label-to-raw-file mapping), `PRIVATE_CACHE` (stage-1 cache) and
`ORIGINAL_ROOT` (original working project). They are not needed to reproduce
any result above.

## Evaluation protocol (applies to every number)

* Five time-ordered folds per run; each tests a disjoint 10% block of the
  record (50-60%, ..., 90-100%) with 80 forecast origins, so 400 per run.
* Every forecast uses only the 128 minutes before its origin; every window lies
  in one gap-free segment.
* Anything fitted (closed-form coefficients, recalibration, ridge penalty)
  uses only rows or folds that precede the test block.
* Statistical unit: the run (45 = 9 datasets x 5 targets), never a horizon or
  a fold; Holm correction across comparisons.

## Licence

Code: MIT licence (`LICENSE`). Data, stored forecasts and result tables:
Creative Commons Attribution 4.0 International (`DATA_LICENSE`). Raw sensor
exports and the mapping from device labels to plant assets are not released.
