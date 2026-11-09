"""P2 - check that every number quoted in the manuscript text matches the result files.

Each entry recomputes a value from results/ or data/, formats it the way the text
quotes it, and checks that the formatted string occurs in the manuscript.
Usage: python check_paper_numbers.py path/to/main.tex
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, TABLES, TRAINED_MODELS  # noqa: E402

AUD = RESULTS / "audit"


def pct(x):
    return f"{100 * x:.1f}"


def main(tex_path):
    tex = Path(tex_path).read_text(encoding="utf-8")
    K = json.loads((TABLES / "key_numbers.json").read_text())
    s2 = pd.read_csv(TABLES / "b02_summary.csv")
    t2 = pd.read_csv(TABLES / "b02_tests.csv")
    b4 = pd.read_csv(TABLES / "b04_overconfidence_by_model.csv")
    b5 = pd.read_csv(TABLES / "b05_correlations.csv")
    b5p = pd.read_csv(TABLES / "b05_predictability.csv")
    b9 = pd.read_csv(TABLES / "b09_tuning_tests.csv")
    b9s = pd.read_csv(TABLES / "b09_tuning_selection.csv")
    b10 = pd.read_csv(TABLES / "b10_seed_spread.csv")
    a7 = pd.read_csv(AUD / "a07_divergent_cells.csv")
    a8 = pd.read_csv(AUD / "a08_pilot_roster.csv").set_index("model")
    tr = pd.read_csv(AUD / "a04_transfer_summary.csv")
    man = pd.read_csv(DATA / "manifest.csv")
    a2 = pd.read_csv(AUD / "a02_dataset_claims.csv").set_index("id")
    raw = pd.read_csv(AUD / "device_raw_summary.csv")

    def test(kind, m, ref):
        return t2[(t2.kind == kind) & (t2.method == m) & (t2.reference == ref)].iloc[0]

    def tune(kind, a, b):
        return b9[(b9.kind == kind) & (b9.a == a) & (b9.b == b)].iloc[0]

    def wins(kind, m):
        return int(s2[(s2.kind == kind) & (s2.method == m)].runs_beating_best_naive.iloc[0])

    rv, rt = test("vibration", "Ridge-Univariate", "TiDE"), test("temperature", "Ridge-Univariate", "TiDE")
    oc = b4[b4.method.isin(TRAINED_MODELS)]
    ocv = oc[oc.kind == "vibration"].set_index("method")
    weak = ocv[ocv.median_overconf >= 1.8]
    temp_r = b4[b4.kind == "temperature"].set_index("method").median_r
    trained_r = temp_r[[m for m in TRAINED_MODELS if m != "TiDE"]]
    sel = b9s.assign(rr=b9s.test_mse_cfg0_rerun / b9s.test_mse_original_stored).groupby("model").rr
    cv = b10.set_index(["kind", "model"]).median_cv
    corr = b5[(b5.scope == "vibration") & (b5.score == "lambda_uncapped")].set_index("outcome")
    tp = b5[(b5.scope == "temperature") & (b5.score == "lambda_uncapped")].p.min()
    lam = b5p.groupby("kind").lambda_uncapped.agg(["min", "max"])
    acf = b5p[b5p.kind == "temperature"].acf_h
    tt = tr[(tr.model == "TiDE") & (~tr.excl_R2_I3_target) & (tr["shift"].isin(["spatial-only", "temporal-only", "both"]))]
    ttv = tt[tt.channel == "self_x_ac"].median_ratio_vs_in_domain
    ttt = tt[tt.channel == "self_temperature_one"].median_ratio_vs_in_domain
    spm = (raw.raw_rows / raw.distinct_minutes).median()

    checks = [
        ("rows", f"{man.rows.sum():,}".replace(",", "{,}")),
        ("ridge vs TiDE vibration", f"{pct(rv.median_mse_ratio - 1)}\\%"),
        ("ridge vs TiDE temperature", f"{pct(1 - rt.median_mse_ratio)}\\%"),
        ("ridge wins vibration", f"won {int(rv.runs_method_better)} of 27"),
        ("ridge Holm vibration", f"$p={rv.p_holm:.2f}$"),
        ("ridge Holm temperature", f"{rt.p_holm:.1e}".replace("e-0", r"\times10^{-") + "}"),
        ("anchored vibration gain", f"{round(100 * (1 - K['vibration']['recal_anchored']['median_ratio']))}\\% on vibration"),
        ("anchored temperature gain", f"{round(100 * (1 - K['temperature']['recal_anchored']['median_ratio']))}\\% on temperature"),
        ("TiDE/TimeXer wins vibration", f"in {wins('vibration', 'TiDE')} of 27 runs and TimeXer in {wins('vibration', 'TimeXer')}"),
        ("TiDE wins temperature", f"({wins('temperature', 'TiDE')} of 18)"),
        ("TimeXer wins temperature", f"followed by TimeXer ({wins('temperature', 'TimeXer')})"),
        ("ridge naive wins", f"{wins('vibration', 'Ridge-Univariate')}/27 vibration and {wins('temperature', 'Ridge-Univariate')}/18"),
        ("ensemble vs TiDE", f"{round(100 * (test('vibration', 'Ensemble-Mean-13', 'TiDE').median_mse_ratio - 1))}\\% (vibration) and "
                             f"{round(100 * (test('temperature', 'Ensemble-Mean-13', 'TiDE').median_mse_ratio - 1))}\\% (temperature)"),
        ("ridge-S wins temperature", f"Ridge-Sensors won {int(test('temperature', 'Ridge-Sensors', 'TiDE').runs_method_better)}"),
        ("rho/r range vibration", f"{ocv.loc['TFT'].median_overconf:.1f} (TFT) to {ocv.median_overconf.max():.1f} (TSMixerx)"),
        ("cond bias share", f"{weak.median_cond_bias_pct.min():.0f}--{weak.median_cond_bias_pct.max():.0f}\\%"),
        ("n weak models", "eight models" if len(weak) == 8 else f"{len(weak)} models"),
        ("rho/r temperature", f"({b4[b4.kind == 'temperature'].median_overconf.min():.2f}--{b4[b4.kind == 'temperature'].median_overconf.max():.2f})"),
        ("trained r vs persistence", f"({trained_r.min():.2f}--{trained_r.max():.3f} against {temp_r['Naive-Persistence']:.3f})"),
        ("acf temperature", f"{acf.min():.2f}--{acf.max():.2f}"),
        ("TiDE anchored", f"({K['vibration']['recal_anchored_scope']['tide_median']:.3f} and "
                          f"{K['temperature']['recal_anchored_scope']['tide_median']:.3f}, better in "
                          f"{K['vibration']['recal_anchored_scope']['tide_improved']}/27 and "
                          f"{K['temperature']['recal_anchored_scope']['tide_improved']}/18"),
        ("hindsight best vs ridge", f"{pct(K['temperature']['recal_anchored_scope']['hindsight_best_vs_ridge'] - 1)}\\% worse"),
        ("hindsight best wins", f"only {K['temperature']['recal_anchored_scope']['hindsight_best_better']} of 18"),
        ("tuning TiDE/TimeXer temp", f"{tune('temperature', 'TiDE-Tuned', 'TiDE-Original').median_ratio_a_over_b:.3f} and "
                                     f"{tune('temperature', 'TimeXer-Tuned', 'TimeXer-Original').median_ratio_a_over_b:.3f}"),
        ("tuning Holm", f"$p_{{\\mathrm{{Holm}}}}={tune('temperature', 'TiDE-Tuned', 'TiDE-Original').p_holm:.3f}$"),
        ("oracle TiDE vs ridge", f"{pct(K['tuning']['temperature:TiDE-Oracle(test-selected, reference only)/Ridge-Univariate'] - 1)}\\% behind"),
        ("tuning TiDE vibration", f"({tune('vibration', 'TiDE-Tuned', 'TiDE-Original').median_ratio_a_over_b:.3f})"),
        ("tuned-any vs ridge vib", f"({tune('vibration', 'Tuned-Any', 'Ridge-Univariate').median_ratio_a_over_b:.3f}, "
                                   f"{tune('vibration', 'Tuned-Any', 'Ridge-Univariate').runs_a_better}/9 runs, "
                                   f"$p={tune('vibration', 'Tuned-Any', 'Ridge-Univariate').p_wilcoxon:.2f}$"),
        ("rerun TiDE", f"{sel.min()['TiDE']:.2f}--{sel.max()['TiDE']:.2f}"),
        ("rerun TimeXer", f"{sel.min()['TimeXer']:.2f}--{sel.max()['TimeXer']:.2f}"),
        ("rerun NHITS", f"{sel.min()['NHITS']:.2f}--{sel.max()['NHITS']:.2f}"),
        ("TiDE seed CV", f"{cv[('vibration', 'TiDE')]:.3f} on vibration and {cv[('temperature', 'TiDE')]:.3f}"),
        ("TabNet/TCN seed CV", f"TabNet ({cv[('temperature', 'TabNet')]:.2f}) and TCN ({cv[('temperature', 'TCN')]:.2f})"),
        ("diverged cells", f"{int(a7.over_10x.sum())} of the 2{{,}}925"),
        ("worst diverged", f"({a7.x_persistence.max():.0f} times)"),
        ("lambda temperature", f"$\\lambda={lam.loc['temperature', 'min']:.3f}$--${lam.loc['temperature', 'max']:.3f}$"),
        ("lambda vibration", f"vibration ${lam.loc['vibration', 'min']:.2f}$--${lam.loc['vibration', 'max']:.2f}$"),
        ("lambda best skill", f"(${corr.loc['best_skill'].spearman:+.2f}$, $p={corr.loc['best_skill'].p:.3f}$)"),
        ("lambda anchor", f"(${corr.loc['median_anchor_ratio'].spearman:+.2f}$, $p={corr.loc['median_anchor_ratio'].p:.3f}$)"),
        ("lambda Holm", f"($p={corr.loc['best_skill'].p_holm_4_outcomes:.2f}$ each)"),
        ("lambda temperature p", f"all $p\\ge{tp:.2f}$"),
        ("transfer vibration", f"{ttv.min():.2f}--{ttv.max():.2f}"),
        ("transfer temperature", f"{pct(ttt.min() - 1)}--{pct(ttt.max() - 1)}\\%"),
        ("pilot", f"LSTM ({a8.loc['LSTM', 'mean_3_regions']:.2f} times persistence MSE), GRU ({a8.loc['GRU', 'mean_3_regions']:.2f})"),
        ("pilot TabNet", f"in that pilot ({a8.loc['TabNet', 'mean_3_regions']:.2f})"),
        ("gravity", "within 1.22\\% of $g$ in all 24 device--dataset pairs (median deviation 0.37\\%)"
         if "1.22%" in a2.loc["C3", "recomputed"] and "0.37%" in a2.loc["C3", "recomputed"] else "MISMATCH"),
        ("samples per minute", f"about {spm:.0f} samples per minute"),
    ]
    bad = 0
    for name, s in checks:
        ok = s in tex
        bad += not ok
        print(f"{'ok  ' if ok else 'MISS'} {name}: {s}")
    print(f"\n{len(checks) - bad}/{len(checks)} quoted numbers found in the manuscript")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
