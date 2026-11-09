# Audit report

Scope: every step from the raw sensor exports to every number in the earlier
drafts (dataset card, reviewer-defence notes, paper draft, strategy notes) was
re-derived from data, independently of what had been written before. Nothing
was accepted because a document said so. Each check is a script in
`code/audit/` or `code/analysis/`, and its output is a CSV in `results/`.

Severity scale: **S1** invalidates a reported result. **S2** claim not supported
by the data, must be dropped or reworded. **S3** factual or wording correction,
or a disclosure that is missing. **OPEN** needs information only the authors have.

---

## 1. What was verified and holds

| Item | How it was checked | Result |
|---|---|---|
| Entire data pipeline (windows, PT100 correction, gravity calibration, per-minute features, role columns, pruning) | Re-implemented from scratch with anonymised labels (`code/data_build/`), run on the raw exports, compared value by value with the files the models were trained on (`a01`) | All 9 files identical: same shape, columns, timestamps; max abs difference 2.4e-7 (CSV rounding). 157,707 rows. No gaps inside segments. No overlap between iterations. |
| Gravity constants k_d (previously no script existed for this step) | Recomputed from raw (`01_raw_index_and_calibration.py`) | All 9 match to 4 decimals with identical sample counts |
| Calibrated resultant equals g within 1.22% over 24 device-dataset pairs | `a02`, claim C3 | PASS (max 1.22%, median 0.37%) |
| PT100 0 degC round trip 100.0000 ohm; correction -8.38 / -10.74 degC | `a02`, C4-C5 | PASS (the temperatures are the *reported* readings; see S3) |
| Window choice uses coverage only, decided before any model ran | Code read; file timestamps (windows 03:21, first model results 05:54, same day) | PASS |
| Exclusions: AHM-11 1.0% coverage / 21,728 rows / 833.51 degC; AHM-10 835.51 degC and y axis exactly 0.0 in 50.98% of samples; AHM-09 costs 46% of Region 3 usable minutes (34,248 vs 63,461) | `a02`, `a05` (private raw index) | PASS |
| 45 trained-model runs (3 iterations x 3 regions x 5 targets, 13 models + 2 naive) | `a03`: ground truth equals data at stored origins; every origin in its fold's test block; every window inside one segment; test blocks disjoint; no NaN predictions | PASS on all 45 |
| No look-ahead in the neural training/evaluation | Code read: inputs are the 128 rows before the origin; standardisation uses training rows; early stopping uses a separate validation block; leaking target columns dropped (listed per run) | PASS |
| Pilot that removed LSTM/GRU/FEDformer | Recomputed from the stored pilot forecasts (`a08`) | Means reproduce (LSTM 6.06x, GRU 6.00x, TabNet 6.07x persistence MSE; FEDformer NaN) - see S3 for wording |
| Transfer-matrix marginal means (vibration, R2_I3 excluded as target) | `a04` | Reproduce exactly (TiDE 0.01470/0.01484/0.01493/0.01479; TimeXer 0.01591/0.01624/0.01672/0.01629) |
| Exact MSE identity on every cell | `b04` asserts it | PASS |

---

## 2. Problems found

### S1 - results that were invalid as reported

1. **Rescaling baselines B1, B2, B3 and conformal B5 used future data.** For each
   test fold the constants were fitted on all *other* folds, including later
   ones. A deployed system cannot do this. B1 was also clipped to [0.02, 0.98],
   so it could only shrink. **Redone causally** in `b03` (fit on earlier folds
   only). Result: the original B1 idea gives median MSE ratio 0.94 on vibration
   (19/27 runs better, p = 0.06, not significant) and **worsens temperature in
   18/18 runs** (ratio 1.04).
2. **SCAS selected on the test set.** Whenever SCAS scored worse than B1 *on
   the test fold*, B1's number was reported instead, so SCAS could not lose by
   construction. All SCAS numbers are withdrawn. SCAS is not used.
3. **Weight-merging weights were chosen on the test predictions.** Only the
   three un-merged checkpoints (M1, M2, M3) are valid results
   (`results/transfer/merge_results_FLAGGED.csv`).
4. **Floor-aware training used test information.** The variance floor came from
   lambda computed on the full record, test blocks included.
5. **p-values were inflated.** Earlier tests treated target x model x horizon
   cells as independent (for example "p = 2e-39", "58% win rate over 208
   cells"). Horizons, folds and models within a run are not independent. All
   tests are now at run level (45 runs) or target level (26 targets for
   floor-aware training), with Holm correction where several are made.

### S2 - claims the data do not support

1. **"Predictability law" (miscalibration scales with lambda).** Lambda only
   separates the two channel types: temperature 1.000-1.004 (uncapped; the old
   code capped it at 1), vibration 0.67-0.98. *Within* vibration (27 runs) it
   does not predict overconfidence (Spearman -0.27, p = 0.17) or how many
   trained models fail (-0.00, p = 0.99). It does correlate, at the nominal 5%
   level only, with the best skill reached in a run (+0.40, p = 0.040) and with
   the anchored-recalibration gain (-0.42, p = 0.031); neither survives Holm
   correction over the four outcomes tested (both p = 0.12). Within
   temperature, no relation (all four p >= 0.27). (`b05`)
2. **"Model diversity is largely explained by miscalibration."** No. The
   unexplained term dominates the error; rescaling does not change which
   trained model is best (27/27 vibration and 17-18/18 temperature runs
   unchanged under shrink/affine). (`b03`, `b04`)
3. **"SCAS gain 10.6% vibration / 2.3% temperature."** Withdrawn (S1.1-S1.2).
4. **Floor-aware training effect.** At target level: TSMixerx -2.8% median
   (18/26 targets better, p = 0.15); TiDE +0.2% (p = 0.71). Not significant.
5. **"R2_I3 is intrinsically hard."** It has a cause: the reference motor was
   stopped from Aug 17 20:30 to Aug 22 22:10 UTC (vibration near zero, bearing
   temperature down to about 14 degC), and the final test block contains an
   episode of vibration three to four times the normal level
   (`figures/fig5_R2_I3_shutdown.png`). The authors identify these periods as
   anomalies and spikes. They are kept in the data and in the primary
   analysis; they are flagged by fixed sensor-only rules
   (`data/anomaly_intervals.csv`) and every main comparison is repeated
   without them (`b06`). The conclusions do not change (see
   `results/KEY_RESULTS.md`, sensitivity section).
6. **The MSE decomposition is not new.** It is Murphy (1988) (bias,
   conditional bias, correlation terms; see also Taylor 2001). It must be cited
   as such, not presented as a contribution.
7. **"143,787 forecast origins"** is the capacity of the files. Each model was
   evaluated on 400 origins per run (5 folds x 80).

### S3 - corrections and missing disclosures

1. PT100: the -8.38 / -10.74 degC corrections apply at *reported* readings of
   22 / 94 degC; the firmware over-reports by 8-11 degC.
2. Pilot: 6.00-6.07x are means over three regions; per region the ratios range
   from 3.6x to 7.7x. "In every region" is wrong.
3. k_d statistics "across nine devices" include the excluded AHM-09; the
   release has eight devices (mean 1.8179).
4. Released files have 56 feature columns each (Region 3: 16 zero-filled
   neighbour-2 columns). The "42-61 features" figures describe older files.
5. Coverage percentages (22.3%, 30.2%, 32.8%) depend on an undocumented
   denominator; against each device's own span they are 21.7%, 29.5%, 32.7%.
6. A retained device (AHM-07) also has a temperature spike (one start-up
   sample, 821.9 degC, removed by the envelope filter). Exclusions rest on
   coverage, not on spikes.
7. Roster history must be disclosed in full: models were added after the first
   results ("second wave": NBEATSx, MLP, TCN, DilatedRNN, KAN, DeepNPTS) and
   LSTM/GRU/FEDformer were dropped after a pilot (bearing temperature,
   iteration 1, earlier S1-S5 split scheme; `a08`). TabNet did as badly in the
   pilot (6.07x persistence) as LSTM (6.06x) and GRU (6.00x) but was kept: the
   training code records that it was re-admitted after it was seen to rank
   mid-pack on vibration x. LSTM and GRU were never tried on vibration. This is
   a post-hoc roster decision and must be stated as such.
8. Operating horizons (vibration 3 min, temperature 15 min) were chosen after
   viewing skill-decay curves. All horizons 1-15 are reported (`fig2`).
9. The original study did no hyperparameter tuning and trained every model for
   a 15-step horizon (vibration evaluated at 3). Every reported number uses
   seed 42; 14 of the 45 runs were also trained at seeds 101 and 202 (S3.19).
   A tuning check was added after this audit (section 3, `b09`).
10. During training, windows may span a gap between segments (synthetic
    contiguous time axis). Test windows never do.
11. Logged metric files round MSE to 4 decimals (up to 12% relative error on
    vibration). Every number here is recomputed from stored predictions.
12. Training configs record `iteration = 1` for every run (metadata bug, fixed
    in the released configs). Some code comments are stale (a "P2 trains only
    TiDE + TSMixerx" note, although all 13 were trained; a scaler comment).
13. k_d uses each device's full record (test periods included). This cannot
    affect any model, because every input is re-standardised with training
    statistics, but it should be stated. The same applies to the
    neighbour ordering (taken from iteration-1 data).
14. Transfer cells: one model, one seed, fold F5 only, 80 origins each. Half of
    the temporal-shift cells transfer *backwards* in time (reported separately
    in `a04_transfer_summary.csv`).
15. Fine-tuning through R2_I3 (M3) is worse on R2_I3 than the never-adapted M1
    (0.288 vs 0.209). This is one run with one seed, and R2_I3's training part
    contains the motor stop.
16. "Persistence R2 is about -0.5 on vibration and about 0.99 on temperature"
    is not reproduced. At the operating horizons over the 45 runs, persistence
    R2 is -0.18 to 0.49 on vibration (median 0.04) and 0.91 to 0.99 on
    temperature (median 0.955). On vibration persistence is still not the
    strongest naive baseline (EWMA60 usually is), which is why skill is always
    measured against the strongest one.
17. **Diverged forecasts are part of the stored results** (found while checking
    the tuning re-runs, after the first pass of this audit). In 41 of 2,925
    model x run x fold cells the operating-horizon MSE is more than 10x that of
    persistence on the same fold (10 cells above 20x, 2 above 50x). The worst is
    NHITS on Region 3, iteration 2 (R3_I2), `self_temperature_one`, fold F2: 526x,
    forecasts up to 92 degC while the truth stays below 37 degC. By model:
    TabNet 14, DilatedRNN 11, KAN 9, TCN 3, BiTCN 2, NHITS 1, MLP 1; none for
    TiDE. The cells are kept in every primary result, because they are what the
    models produced. With them removed (`a07`), every headline comparison is
    unchanged; the only visible shift is the anchored recalibration gain on
    temperature (median ratio 0.399 -> 0.449, still 18/18 runs). A per-step
    median of the 13 trained models (`Ensemble-Median-13`) is no better than the
    mean (1.243x / 1.553x TiDE on vibration / temperature).
18. **Re-training on different hardware does not reproduce some models.** Re-running the
    original configuration with the same settings and seed (tuning check,
    configuration 0) gives a test MSE of 0.96-1.02 times the stored value for
    TiDE, but 0.67-1.93 for TimeXer and 0.00-1.82 for NHITS (the 0.00 is the
    diverged fold of item 17). The re-run used a different GPU and torch
    version (Tesla T4 / torch 2.11.0 instead of Tesla P100 / torch 2.7.1), so
    this mixes hardware/library differences with run-to-run non-determinism;
    the two cannot be separated here. The seed-only variance, on identical
    hardware, is measured in `b10` (S3.19).
19. **Extra-seed runs exist and were not reported.** An earlier version of this
    report said only seed 42 was trained. The final audit found 28 complete
    runs at seeds 101 and 202 for 14 iteration-1 runs (all 13 models, all 5
    folds; same code, data, origins and hardware, Tesla P100 / torch 2.7.1),
    trained on the same day as the main runs to measure training variance.
    Region 1 / `self_x_ac` was not repeated: it was the first full run,
    launched on its own as a check before the queue, and the seed queue
    reused the queue's target list (its configuration is identical to the
    other runs). They are now in the release (`results/
    trained_model_predictions_extra_seeds/`) and analysed in `b10`, with the
    analysis fixed before their results were looked at. Seed 42 was not a
    favourable draw: it is the best of the three seeds in 38% (vibration) and
    37% (temperature) of run x model cells, against 33% expected. Every
    headline conclusion holds at each seed separately: on vibration
    Ridge-Univariate is 1.024-1.034x TiDE and TiDE is the best trained model in
    8/8 runs at every seed; on temperature Ridge-Univariate beats TiDE in 6/6
    runs at every seed (0.823-0.840x); the anchored recalibration improves every
    run at every seed. TiDE barely moves with the seed (median coefficient of
    variation 0.003 / 0.010); TabNet (0.340) and TCN (0.250) move most on
    temperature.
20. **Scope of the anchored recalibration.** Its headline gain (median MSE
    ratio 0.709 vibration, 0.399 temperature) is a median over the 13 trained
    models and comes from the miscalibrated ones. It does not improve the best
    trained model: TiDE alone gets 1.008 (vibration, improved in 7/27 runs) and
    1.006 (temperature, 7/18), and Ridge-Univariate does not gain from it either.
    It does not change which method is best: after recalibration, no trained
    model beats raw Ridge-Univariate in a majority of temperature runs, and even
    the best recalibrated trained model per run, picked with hindsight, is
    1.146x ridge there (better in 3/18 runs). On vibration TiDE remains the only
    trained model ahead of ridge (21/27 runs). It should be presented as a
    repair for overconfident models, not as a way to beat the closed-form
    baseline.

### Note from the authors - PT100 readings during the stop

During the R2_I3 motor stop, the *corrected* bearing temperature settles at
13.9-17 degC, below typical August ambient for the region (uncorrected about
22 degC). The authors state that these readings are anomalies and spikes, not
normal operation, so they are not evidence against the 97 -> 100 ohm
correction. This audit could not verify the correction independently of the
firmware fact reported by the hardware owner. The correction is nearly linear over the operating range
(slope 0.97 between reported 22 and 94 degC), so it rescales temperature errors
almost uniformly and model comparisons should change very little; this has not
been tested by retraining. Absolute temperatures would change by up to about 11 degC.

---

## 3. Experiments added (all causal)

Training-free and pretrained forecasters at the exact same 400 origins per
run, causal recalibration, error decomposition, predictability test, a
recheck of floor-aware training and transfer, a hyperparameter check (the only
new neural training), a robustness check for diverged forecasts (`a07`), and
the training-seed analysis of the extra-seed runs (`b10`, S3.19).
Results: `results/KEY_RESULTS.md` (generated by script).

Design choices, stated so they can be judged:

* Every new method sees the same 128-minute history as the neural models.
  The pretrained models could use longer histories; they were limited to 128
  for parity, which may understate them.
* Pretrained models (Chronos-Bolt-small, Chronos-2) are used as downloaded,
  with no tuning, on the target signal only. A variant giving Chronos-2 the
  same past exogenous channels as the neural models is implemented
  (`b01 --chronos2-sensors`) but was not run to completion (about 6 CPU-hours
  per pass); it is not reported anywhere, and no partial result was kept.
* Ridge penalty: chosen from a fixed grid on each fold's validation block,
  which precedes the test block. AR coefficients: moments of training rows.
* Naive-EWMA60 is computed within the current segment. The original study's
  EWMA ran across gaps; persistence and MA60 match the stored values exactly.
* Ensemble-Mean-13: unweighted mean of the 13 trained models (no selection).
* Recalibration is fitted on earlier folds only, so all recalibration numbers
  use folds 2-5 for every method, raw included.
* Statistical unit: run (45). Runs share datasets (5 targets per dataset), so
  they are not fully independent; the run-level p-values should be read as
  approximate. Win counts (for example 18/18) are reported next to every
  p-value for that reason.

### Hyperparameter check (`b09`)

Question: would tuning the neural models change the conclusions?

* Models: TiDE (best trained model), TimeXer and NHITS. 8 configurations each:
  configuration 0 is the original setting; configurations 1-7 were drawn from a
  fixed grid (hidden size or width, dropout, learning rate, and for TiDE the
  number of encoder layers) with a fixed seed (2026), before any run.
* Runs: 9 datasets x {`self_x_ac`, `self_temperature_one`} = 18 runs, all 5
  folds, so 3 x 8 x 5 = 120 trainings per run and 2,160 in total. Everything else
  is identical to the stored runs: the kernel is the original training script
  plus `tune_tail.py` (`code/training/build_tune_kernel.py`).
* Selection rule, fixed before any tuned result was seen: for each fold and
  model, the configuration with the lowest MSE on that fold's validation block
  at the operating horizon. "Tuned-Any" picks, per fold, the best of all 24
  model x configuration pairs on validation. A test-selected "Oracle" is
  reported for reference only and is never used for a claim.
* One comparison was added after the first two runs returned and before any
  tuned result was looked at: tuned vs the configuration-0 re-run from the same
  session, because the re-run did not reproduce the stored NHITS/TimeXer
  forecasts (S3.18). The reason is recorded in `b09_tuning_check.py`.
* Integrity: test origins and ground truth identical to the stored runs in
  90/90 run-folds; every validation target precedes every test target.
  2 of 2,160 trainings failed (TimeXer configuration 7, learning rate 3e-3,
  loss became NaN on one fold of two runs); a failed configuration is not a
  candidate on that fold.
* Compute: Kaggle GPU, Tesla T4 with torch 2.11.0+cu130 (from the job logs;
  the original runs used Tesla P100 with torch 2.7.1+cu126; the neuralforecast
  version was not logged in either case). The 18 jobs were split across three accounts
  belonging to three of the authors (first author: 10 jobs, all of iteration 1
  plus R1_I2 and R3_I3; co-author B: R1_I3 and R2_I3; co-author C: R2_I2 and
  R3_I2), each person running their own jobs. Each account attached its own
  private copy of the data, md5-identical to the others (checked before use).

Results (9 runs per kind; full list in `results/KEY_RESULTS.md`):

* Temperature: tuning helps TiDE (median MSE ratio 0.896, 9/9 runs) and TimeXer
  (0.883, 9/9), but Ridge-Univariate still beats every tuned model in every run
  (tuned TiDE 1.147x ridge, 0/9). Even the test-selected TiDE configuration is
  1.147x ridge.
* Vibration: tuning does not help TiDE (1.004, better in 2/9 runs). The best
  validation-selected model is level with Ridge-Univariate (0.996, 7/9 runs,
  p = 0.50).
* The conclusions of the main analysis do not change.

## 4. Not in the release

Raw exports, the label-to-file mapping (`RAW_CONFIG`), and the stage-1 cache
contain site identifiers and stay private. Everything else in this folder can
be rebuilt from `data/` and `results/trained_model_predictions/`.
