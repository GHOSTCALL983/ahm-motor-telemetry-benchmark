"""Audit 6 - second, independent pass over the finished release.

1. Privacy: no site, device-id, person or account identifier anywhere in the
   folder (text files, CSV cells, JSON, and the key names inside .npz files).
2. Every earlier audit CSV still passes.
3. Independent recomputation (does NOT use analysis/load.py): for a random
   sample of runs, MSE at the operating horizon of TiDE, Ridge-Univariate and
   persistence, and the anchored recalibration of TiDE, recomputed from the raw
   stored arrays and compared with the result tables.
4. Hyperparameter check and diverged forecasts: integrity table, validation-only
   selection recomputed from the raw tuning arrays for random runs, the worst
   diverged cell recomputed from the stored forecasts.
5. Numbers quoted in AUDIT_REPORT.md are compared with the tables they cite.
Writes results/audit/a06_final_audit.csv and exits non-zero on any failure.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FOLDS, PRED_DIR, RESULTS, ROOT, TABLES  # noqa: E402

# Patterns are assembled from fragments so this file does not itself contain them.
FORBIDDEN = [r"(?i)" + a + b for a, b in (("ja", "y ?balaji"), ("jb", "sp"), ("ie", "ma"), ("601", "-2"),
                                         ("shre", "yan"), ("kun", "du"), ("unr", "ul"), ("gm", "ail"))]
OP_H = {"vibration": 3, "temperature": 15}
checks = []


def check(name, ok, detail=""):
    checks.append(dict(check=name, ok=bool(ok), detail=str(detail)[:300]))


def privacy():
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        texts = [str(p.relative_to(ROOT))]
        if p.suffix in (".py", ".md", ".csv", ".json", ".txt"):
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        elif p.suffix == ".npz":
            texts.append(" ".join(np.load(p).files))
        for t in texts:
            for pat in FORBIDDEN:
                m = re.search(pat, t)
                if m:
                    hits.append(f"{p.relative_to(ROOT)}: '{m.group(0)}'")
    check("privacy: no identifiers in any file or file name", not hits, hits[:10])


def earlier_audits():
    a01 = pd.read_csv(RESULTS / "audit" / "a01_rebuild_check.csv")
    check("a01 rebuild identical", a01[["same_shape", "same_columns", "same_timestamps"]].all().all()
          and a01.max_abs_value_diff.max() < 1e-5 and (a01.within_segment_gaps == 0).all(),
          f"max diff {a01.max_abs_value_diff.max():.2e}")
    a02 = pd.read_csv(RESULTS / "audit" / "a02_dataset_claims.csv")
    check("a02 no FAIL verdicts", not a02.verdict.str.contains("FAIL").any(), a02.verdict.value_counts().to_dict())
    a03 = pd.read_csv(RESULTS / "audit" / "a03_trained_runs.csv")
    cols = ["all_13_trained", "truth_matches_data", "origins_in_test_block", "windows_within_segment",
            "test_blocks_disjoint"]
    check("a03 all 45 runs pass", len(a03) == 45 and a03[cols].all().all() and a03.nan_prediction_cells.sum() == 0)


def recompute(n_sample=8, seed=7):
    rl = pd.read_csv(TABLES / "b02_run_level.csv")
    rec = pd.read_csv(TABLES / "b03_recal_run_level.csv")
    runs = sorted(rl.run.unique())
    rng = np.random.default_rng(seed)
    worst = 0.0
    for run in rng.choice(runs, n_sample, replace=False):
        kind = "temperature" if "temperature" in run else "vibration"
        H = OP_H[kind]
        z = np.load(PRED_DIR / f"{run}.npz")
        tf = np.load(RESULTS / "training_free_predictions" / f"{run}.npz")
        for m, src, key in (("TiDE", z, "TiDE|{f}|0.0"), ("Naive-Persistence", z, "Naive-Persistence|{f}|0.0"),
                            ("Ridge-Univariate", tf, "Ridge-Univariate|{f}")):
            mse = np.mean([np.mean((src[key.format(f=f)][:, :H].astype(float)
                                    - z[f"ytrue|{f}"][:, :H].astype(float)) ** 2) for f in FOLDS])
            tab = rl[(rl.run == run) & (rl.method == m) & (rl.h == H)].mse.iloc[0]
            worst = max(worst, abs(mse - tab) / tab)
        # anchored recalibration of TiDE, written independently
        mses = []
        for k in range(1, 5):
            P = np.vstack([z[f"TiDE|{f}|0.0"] for f in FOLDS[:k]]).astype(float)
            Y = np.vstack([z[f"ytrue|{f}"] for f in FOLDS[:k]]).astype(float)
            Lp = np.vstack([z[f"Naive-Persistence|{f}|0.0"][:, :1] for f in FOLDS[:k]]).astype(float)
            f = FOLDS[k]
            Pt, Yt = z[f"TiDE|{f}|0.0"].astype(float), z[f"ytrue|{f}"].astype(float)
            Lt = z[f"Naive-Persistence|{f}|0.0"][:, :1].astype(float)
            b = ((P - Lp) * (Y - Lp)).sum(0) / ((P - Lp) ** 2).sum(0)
            mses.append(np.mean((Lt + b * (Pt - Lt) - Yt)[:, :H] ** 2))
        tab = rec[(rec.run == run) & (rec.method == "TiDE") & (rec.variant == "anchored")].mse_folds2to5.iloc[0]
        worst = max(worst, abs(np.mean(mses) - tab) / tab)
    check("independent recomputation matches tables (8 random runs)", worst < 1e-6, f"max rel diff {worst:.2e}")


def tuning_and_divergence(n_sample=5, seed=11):
    integ = pd.read_csv(TABLES / "b09_tuning_integrity.csv")
    check("b09 tuning integrity: 90/90 run-folds, origins & truth identical, validation before test",
          len(integ) == 90 and integ.origins_identical.all() and integ.truth_max_abs_diff.max() == 0
          and integ.val_before_test.all())
    sel = pd.read_csv(TABLES / "b09_tuning_selection.csv")
    tdir = RESULTS / "tuning"
    runs = sorted(p.stem for p in tdir.glob("*.npz"))
    check("tuning runs present (18) with account record", len(runs) == 18
          and len(pd.read_csv(tdir / "JOB_ACCOUNTS.csv")) == 18)
    rng = np.random.default_rng(seed)
    worst, mism = 0.0, []
    for run in rng.choice(runs, n_sample, replace=False):
        H = OP_H["temperature" if "temperature" in run else "vibration"]
        z = np.load(tdir / f"{run}.npz")
        for m in ("TiDE", "TimeXer", "NHITS"):
            for f in FOLDS:
                yv, yt = z[f"ytrue_val|{f}"].astype(float), z[f"ytrue_test|{f}"].astype(float)
                vm = {}
                for k in range(8):
                    key = f"{m}|cfg{k}|{f}|val"
                    if key in z.files:
                        vm[k] = np.mean((z[key][:, :H].astype(float) - yv[:, :H]) ** 2)
                k = min(vm, key=vm.get)
                t = np.mean((z[f"{m}|cfg{k}|{f}|test"][:, :H].astype(float) - yt[:, :H]) ** 2)
                row = sel[(sel.run == run) & (sel.model == m) & (sel.fold == f)].iloc[0]
                if int(row.cfg_selected) != k:
                    mism.append(f"{run} {m} {f}")
                worst = max(worst, abs(t - row.test_mse_selected) / row.test_mse_selected)
    check(f"tuning selection recomputed independently ({n_sample} random runs): same configuration, same test MSE",
          not mism and worst < 1e-6, f"mismatches {mism[:5]}, max rel diff {worst:.2e}")
    cells = pd.read_csv(RESULTS / "audit" / "a07_divergent_cells.csv")
    top = cells.iloc[0]
    z = np.load(PRED_DIR / f"{top.run}.npz")
    H = OP_H[top.kind]
    y = z[f"ytrue|{top.fold}"][:, :H].astype(float)
    tf = np.load(RESULTS / "training_free_predictions" / f"{top.run}.npz")
    # tables use the float64 persistence recomputed from data/ (b01); the trained-run file
    # stores a float32 copy of the same forecast
    pers64 = np.mean((tf[f"Naive-Persistence|{top.fold}"][:, :H] - y) ** 2)
    pers32 = np.mean((z[f"Naive-Persistence|{top.fold}|0.0"][:, :H].astype(float) - y) ** 2)
    ratio = np.mean((z[f"{top.method}|{top.fold}|0.0"][:, :H].astype(float) - y) ** 2) / pers64
    check("worst diverged cell recomputed from raw arrays", abs(ratio - top.x_persistence) / ratio < 1e-6
          and abs(pers32 - pers64) / pers64 < 1e-5,
          f"{top.method} {top.run} {top.fold}: {ratio:.1f}x; float32 vs float64 persistence rel diff "
          f"{abs(pers32 - pers64) / pers64:.1e}")


def seeds():
    integ = pd.read_csv(TABLES / "b10_seed_integrity.csv")
    sdir = RESULTS / "trained_model_predictions_extra_seeds"
    check("b10 extra-seed runs: 28 present, origins & truth identical to seed 42, no NaN",
          len(list(sdir.glob("*.npz"))) == 28 and len(integ) == 28 and integ.origins_identical.all()
          and integ.truth_max_abs_diff.max() == 0 and integ.nan_cells.sum() == 0)
    # independent: Ridge-Univariate / TiDE median ratio per seed, vibration, from raw arrays
    co = pd.read_csv(TABLES / "b10_seed_conclusions.csv")
    worst = 0.0
    for s in (42, 101, 202):
        lr = []
        for f in sorted(sdir.glob("*__seed101.npz")):
            run = f.name.replace("__seed101.npz", "")
            if "temperature" in run:
                continue
            z42 = np.load(PRED_DIR / f"{run}.npz")
            zs = z42 if s == 42 else np.load(sdir / f"{run}__seed{s}.npz")
            tf = np.load(RESULTS / "training_free_predictions" / f"{run}.npz")
            m_t = np.mean([np.mean((zs[f"TiDE|{fo}|0.0"][:, :3].astype(float) - z42[f"ytrue|{fo}"][:, :3]) ** 2)
                           for fo in FOLDS])
            m_r = np.mean([np.mean((tf[f"Ridge-Univariate|{fo}"][:, :3] - z42[f"ytrue|{fo}"][:, :3]) ** 2) for fo in FOLDS])
            lr.append(np.log(m_r / m_t))
        tab = co[(co.kind == "vibration") & (co.seed == str(s))].ridge_vs_tide_median_ratio.iloc[0]
        worst = max(worst, abs(np.exp(np.median(lr)) - tab) / tab)
    check("seed comparison recomputed independently (vibration, Ridge-Univariate vs TiDE, each seed)",
          worst < 1e-6, f"max rel diff {worst:.2e}")


def report_numbers():
    rep = (ROOT / "AUDIT_REPORT.md").read_text(encoding="utf-8")
    kn = json.loads((TABLES / "key_numbers.json").read_text())
    fl = pd.read_csv(RESULTS / "audit" / "a04_floor_summary.csv").set_index("model")
    rt = pd.read_csv(TABLES / "b03_recal_tests.csv")
    stab = pd.read_csv(TABLES / "b03_recal_best_model_stability.csv")
    cor = pd.read_csv(TABLES / "b05_correlations.csv")
    pr = pd.read_csv(TABLES / "b05_predictability.csv")
    a05 = pd.read_csv(RESULTS / "audit" / "a05_region3_configurations.csv").set_index("configuration")
    r2 = pd.read_csv(TABLES / "b02_r2_operating_horizon.csv")
    pv = r2[(r2.method == "Naive-Persistence") & (r2.kind == "vibration")].r2_fold_mean
    pt = r2[(r2.method == "Naive-Persistence") & (r2.kind == "temperature")].r2_fold_mean
    vs = rt[(rt.kind == "vibration") & (rt.variant == "shrink")].iloc[0]
    ts = rt[(rt.kind == "temperature") & (rt.variant == "shrink")].iloc[0]
    cv = cor[(cor.scope == "vibration") & (cor.score == "lambda_uncapped")].set_index("outcome")
    vstab = stab[(stab.kind == "vibration") & stab.variant.isin(["shrink", "affine"])]
    tstab = stab[(stab.kind == "temperature") & stab.variant.isin(["shrink", "affine"])]
    lt = pr[pr.kind == "temperature"].lambda_uncapped
    lv = pr[pr.kind == "vibration"].lambda_uncapped
    expected = {
        "ratio 0.94 on vibration": f"{vs.median_ratio:.2f}" == "0.94",
        "(19/27 runs better, p = 0.06": int(vs.runs_improved) == 19 and f"{vs.p_wilcoxon:.2f}" == "0.06",
        "**worsens temperature in\n   18/18 runs** (ratio 1.04)": int(ts.runs_improved) == 0 and f"{ts.median_ratio:.2f}" == "1.04",
        "Spearman -0.27, p = 0.17": f"{cv.loc['median_overconf','spearman']:.2f}" == "-0.27" and f"{cv.loc['median_overconf','p']:.2f}" == "0.17",
        "(-0.00, p = 0.99)": f"{cv.loc['share_trained_beat_naive','p']:.2f}" == "0.99",
        "temperature 1.000-1.004": f"{lt.min():.3f}" == "1.000" and f"{lt.max():.3f}" == "1.004",
        "vibration 0.67-0.98": f"{lv.min():.2f}" == "0.67" and f"{lv.max():.2f}" == "0.98",
        "(27/27 vibration and 17-18/18 temperature": (vstab.best_trained_model_unchanged == 27).all()
        and set(tstab.best_trained_model_unchanged) <= {17, 18},
        "TSMixerx -2.8% median": f"{fl.loc['TSMixerx','median_delta_pct']:.1f}" == "-2.8",
        "(18/26 targets better, p = 0.15)": int(fl.loc["TSMixerx", "targets_improved"]) == 18 and f"{fl.loc['TSMixerx','p_wilcoxon']:.2f}" == "0.15",
        "TiDE +0.2% (p = 0.71)": f"{fl.loc['TiDE','median_delta_pct']:.1f}" == "0.2" and f"{fl.loc['TiDE','p_wilcoxon']:.2f}" == "0.71",
        "(34,248 vs 63,461)": int(a05.loc["AHM-08+AHM-07+AHM-09", "usable_minutes"]) == 34248
        and int(a05.loc["AHM-08+AHM-07", "usable_minutes"]) == 63461,
        "R2 is -0.18 to 0.49 on vibration (median 0.04)": f"{pv.min():.2f}" == "-0.18"
        and f"{pv.max():.2f}" == "0.49" and f"{pv.median():.2f}" == "0.04",
        "0.91 to 0.99 on\n    temperature (median 0.955)": f"{pt.min():.2f}" == "0.91"
        and f"{pt.max():.2f}" == "0.99" and f"{pt.median():.3f}" == "0.955",
        "costs 46% of Region 3":f"{a05.loc['AHM-08+AHM-07+AHM-09','loss_vs_retained_pair_pct']:.0f}" == "46",
    }
    # hyperparameter check and diverged forecasts (S3.17-18, section 3)
    cells = pd.read_csv(RESULTS / "audit" / "a07_divergent_cells.csv")
    rob = pd.read_csv(RESULTS / "audit" / "a07_robustness.csv")
    sel = pd.read_csv(TABLES / "b09_tuning_selection.csv")
    tt = pd.read_csv(TABLES / "b09_tuning_tests.csv")
    t9 = pd.read_csv(TABLES / "b09_tuning_run_level.csv")
    logs = pd.concat([pd.read_csv(f) for f in (RESULTS / "tuning").glob("*.log.csv")])

    def rv(kind, claim, removed):
        g = rob[(rob.kind == kind) & (rob.claim == claim)]
        return g[g.variant.str.contains("removed") == removed].iloc[0]

    def tv(kind, a, b):
        return tt[(tt.kind == kind) & (tt.a == a) & (tt.b == b)].iloc[0]

    def vs_ridge(kind, a):
        p = t9[t9.kind == kind].pivot(index="run", columns="method", values="mse")
        x = p[a] / p["Ridge-Univariate"]
        return f"{x.median():.3f}", int((x < 1).sum())

    sp = sel.assign(r=sel.test_mse_cfg0_rerun / sel.test_mse_original_stored).groupby("model").r
    anc_t = (rv("temperature", "recalibration 'anchored': median MSE ratio vs raw", False),
             rv("temperature", "recalibration 'anchored': median MSE ratio vs raw", True))
    expected.update({
        "In 41 of 2,925": int((cells.x_persistence > 10).sum()) == 41,
        "(10 cells above 20x, 2 above 50x)": int((cells.x_persistence > 20).sum()) == 10
        and int((cells.x_persistence > 50).sum()) == 2,
        "fold F2: 526x": f"{cells.x_persistence.max():.0f}" == "526",
        "(median ratio 0.399 -> 0.449, still 18/18 runs)": f"{anc_t[0].value:.3f}" == "0.399"
        and f"{anc_t[1].value:.3f}" == "0.449" and int(anc_t[1].runs_better) == 18,
        "(1.243x / 1.553x TiDE": f"{rv('vibration', 'Ensemble-Median-13 vs TiDE: median MSE ratio', False).value:.3f}" == "1.243"
        and f"{rv('temperature', 'Ensemble-Median-13 vs TiDE: median MSE ratio', False).value:.3f}" == "1.553",
        "0.96-1.02 times the stored value": f"{sp.min()['TiDE']:.2f}-{sp.max()['TiDE']:.2f}" == "0.96-1.02",
        "0.67-1.93 for TimeXer and 0.00-1.82 for NHITS": f"{sp.min()['TimeXer']:.2f}-{sp.max()['TimeXer']:.2f}" == "0.67-1.93"
        and f"{sp.min()['NHITS']:.2f}-{sp.max()['NHITS']:.2f}" == "0.00-1.82",
        "2 of 2,160 trainings failed": len(logs) == 2160 and int((logs.status != "ok").sum()) == 2,
        "(median MSE ratio 0.896, 9/9 runs)": f"{tv('temperature', 'TiDE-Tuned', 'TiDE-Original').median_ratio_a_over_b:.3f}" == "0.896"
        and int(tv('temperature', 'TiDE-Tuned', 'TiDE-Original').runs_a_better) == 9,
        "(0.883, 9/9)": f"{tv('temperature', 'TimeXer-Tuned', 'TimeXer-Original').median_ratio_a_over_b:.3f}" == "0.883",
        "(tuned TiDE 1.147x ridge, 0/9)": vs_ridge("temperature", "TiDE-Tuned") == ("1.147", 0),
        "test-selected TiDE configuration is\n  1.147x ridge": vs_ridge("temperature", "TiDE-Oracle(test-selected, reference only)")[0] == "1.147",
        "(1.004, better in 2/9 runs)": f"{tv('vibration', 'TiDE-Tuned', 'TiDE-Original').median_ratio_a_over_b:.3f}" == "1.004"
        and int(tv('vibration', 'TiDE-Tuned', 'TiDE-Original').runs_a_better) == 2,
        "(0.996, 7/9 runs,\n  p = 0.50)": f"{tv('vibration', 'Tuned-Any', 'Ridge-Univariate').median_ratio_a_over_b:.3f}" == "0.996"
        and int(tv('vibration', 'Tuned-Any', 'Ridge-Univariate').runs_a_better) == 7
        and f"{tv('vibration', 'Tuned-Any', 'Ridge-Univariate').p_wilcoxon:.2f}" == "0.50",
    })
    # predictability correlations not supporting the law, and the seed analysis (S2.1, S3.19)
    sp = pd.read_csv(TABLES / "b10_seed_spread.csv").set_index(["kind", "model"])
    pil = pd.read_csv(RESULTS / "audit" / "a08_pilot_roster.csv").set_index("model")
    co = pd.read_csv(TABLES / "b10_seed_conclusions.csv")
    cs = co[co.seed != "seed-mean"]
    vv, tt_ = cs[cs.kind == "vibration"], cs[cs.kind == "temperature"]
    ct = cor[(cor.scope == "temperature") & (cor.score == "lambda_uncapped")]
    expected.update({
        "(+0.40, p = 0.040)": f"{cv.loc['best_skill','spearman']:+.2f}" == "+0.40" and f"{cv.loc['best_skill','p']:.3f}" == "0.040",
        "(-0.42, p = 0.031)": f"{cv.loc['median_anchor_ratio','spearman']:+.2f}" == "-0.42"
        and f"{cv.loc['median_anchor_ratio','p']:.3f}" == "0.031",
        "(both p = 0.12)": f"{cv.loc['best_skill','p_holm_4_outcomes']:.2f}" == "0.12"
        and f"{cv.loc['median_anchor_ratio','p_holm_4_outcomes']:.2f}" == "0.12",
        "(all four p >= 0.27)": len(ct) == 4 and ct.p.min() >= 0.27 - 5e-3,
        "28 complete": len(pd.read_csv(TABLES / "b10_seed_integrity.csv")) == 28,
        "best of the three seeds in 38% (vibration) and\n    37% (temperature)":
            f"{100 * sp.loc[('vibration', 'ALL 13'), 'share_seed42_best']:.0f}" == "38"
            and f"{100 * sp.loc[('temperature', 'ALL 13'), 'share_seed42_best']:.0f}" == "37",
        "1.024-1.034x TiDE": f"{vv.ridge_vs_tide_median_ratio.min():.3f}-{vv.ridge_vs_tide_median_ratio.max():.3f}" == "1.024-1.034"
        and (vv.tide_best_trained_model_runs == 8).all() and (vv.ridge_better_than_tide == 0).all(),
        "(0.823-0.840x)": f"{tt_.ridge_vs_tide_median_ratio.min():.3f}-{tt_.ridge_vs_tide_median_ratio.max():.3f}" == "0.823-0.840"
        and (tt_.ridge_better_than_tide == 6).all(),
        "improves every\n    run at every seed": (cs.anchored_runs_improved == cs.n_runs).all(),
        "(median coefficient of\n    variation 0.003 / 0.010)": f"{sp.loc[('vibration', 'TiDE'), 'median_cv']:.3f}" == "0.003"
        and f"{sp.loc[('temperature', 'TiDE'), 'median_cv']:.3f}" == "0.010",
        "TiDE alone gets 1.008 (vibration, improved in 7/27 runs) and\n    1.006 (temperature, 7/18)":
            f"{kn['vibration']['recal_anchored_scope']['tide_median']:.3f}" == "1.008"
            and kn["vibration"]["recal_anchored_scope"]["tide_improved"] == 7
            and f"{kn['temperature']['recal_anchored_scope']['tide_median']:.3f}" == "1.006"
            and kn["temperature"]["recal_anchored_scope"]["tide_improved"] == 7,
        "1.146x ridge there (better in 3/18 runs)":
            f"{kn['temperature']['recal_anchored_scope']['hindsight_best_vs_ridge']:.3f}" == "1.146"
            and kn["temperature"]["recal_anchored_scope"]["hindsight_best_better"] == 3
            and kn["temperature"]["recal_anchored_scope"]["models_beating_ridge_majority"] == [],
        "trained model ahead of ridge (21/27 runs)":
            kn["vibration"]["recal_anchored_scope"]["models_beating_ridge_majority"] == ["TiDE"],
        "(LSTM 6.06x, GRU 6.00x, TabNet 6.07x persistence MSE; FEDformer NaN)":
            f"{pil.loc['LSTM', 'mean_3_regions']:.2f}" == "6.06" and f"{pil.loc['GRU', 'mean_3_regions']:.2f}" == "6.00"
            and f"{pil.loc['TabNet', 'mean_3_regions']:.2f}" == "6.07" and pil.loc[["FEDformer"]].mean_3_regions.isna().all(),
        "ratios range\n   from 3.6x to 7.7x": f"{pil.loc[['LSTM', 'GRU', 'TabNet'], ['R1', 'R2', 'R3']].min().min():.1f}" == "3.6"
            and f"{pil.loc[['LSTM', 'GRU', 'TabNet'], ['R1', 'R2', 'R3']].max().max():.1f}" == "7.7",
        "TabNet did as badly in the\n   pilot (6.07x persistence) as LSTM (6.06x) and GRU (6.00x)":
            f"{pil.loc['TabNet', 'mean_3_regions']:.2f}" == "6.07",
        "TabNet (0.340) and TCN (0.250)":f"{sp.loc[('temperature', 'TabNet'), 'median_cv']:.3f}" == "0.340"
        and f"{sp.loc[('temperature', 'TCN'), 'median_cv']:.3f}" == "0.250",
    })
    for snippet, ok in expected.items():
        check(f"report text present & matches table: {snippet!r}", (snippet in rep) and ok,
              "text missing" if snippet not in rep else "value mismatch" if not ok else "")
    check("key_numbers.json present", bool(kn))


if __name__ == "__main__":
    privacy()
    earlier_audits()
    recompute()
    tuning_and_divergence()
    seeds()
    report_numbers()
    d = pd.DataFrame(checks)
    d.to_csv(RESULTS / "audit" / "a06_final_audit.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(d.to_string(index=False))
    sys.exit(0 if d.ok.all() else 1)
