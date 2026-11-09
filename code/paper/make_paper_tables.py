"""P1 - LaTeX tables for the paper and its supplementary material.

Every table is generated from the result CSVs in results/ (nothing is typed by
hand), so the manuscript cannot drift from the data. Re-run after any analysis
script has been re-run.

Usage:
    python make_paper_tables.py [--main DIR] [--supp DIR]
Defaults: <release>/paper_tables/main and <release>/paper_tables/supp
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, TABLES, TRAINED_MODELS  # noqa: E402

AUD = RESULTS / "audit"
OP_H = {"vibration": 3, "temperature": 15}
N_RUNS = {"vibration": 27, "temperature": 18}


# ---------------------------------------------------------------- helpers
def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


def pv(p):
    """p-value in compact scientific form."""
    if pd.isna(p):
        return "--"
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print("  wrote", path.name)


def tabular(cols, header, rows, size=r"\footnotesize"):
    out = [size, r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule"]
    out += header + [r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def longtab(cols, header, rows, caption, label):
    """Supplementary long table (longtable package)."""
    out = [r"{\footnotesize\setlength{\tabcolsep}{4pt}", rf"\begin{{longtable}}{{{cols}}}",
           rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule"] + header + [r"\midrule", r"\endfirsthead",
           r"\toprule"] + header + [r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    out += rows + [r"\end{longtable}}"]
    return "\n".join(out) + "\n"


def nice(m):
    return {"ChronosBolt-Small": "Chronos-Bolt-S", "Ensemble-Mean-13": "Ensemble mean (13)",
            "Ensemble-Median-13": "Ensemble median (13)", "Ridge-Univariate": "Ridge (univariate)",
            "Ridge-Sensors": "Ridge (+sensors)", "Naive-Persistence": "Persistence",
            "Naive-EWMA60": "EWMA-60", "Naive-MA60": "MA-60", "Naive-MA15": "MA-15",
            "Chronos2": "Chronos-2"}.get(m, m)


FAMILY = {"trained": "trained", "closed-form": "closed form", "pretrained": "pretrained",
          "naive": "naive", "ensemble": "ensemble"}


# ---------------------------------------------------------------- main tables
def t_data(out):
    man = pd.read_csv(DATA / "manifest.csv")
    win = pd.read_csv(AUD / "iteration_windows.csv")
    win["dataset"] = win.region + "_I" + win.iteration.astype(str)
    d = man.merge(win[["dataset", "usable_pct"]], on="dataset")
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{tex(r.dataset)} & {r.zone} & {pd.Timestamp(r.start):%d/%m}--{pd.Timestamp(r.end):%d/%m} & "
                    f"{r.rows:,} & {r.segments} & {r.usable_pct:.1f} & {r.y_mean:.1f} $\\pm$ {r.y_std:.1f} \\\\")
    rows.append(r"\midrule")
    rows.append(f"Total & & & {d.rows.sum():,} & {d.segments.sum()} & & \\\\")
    body = tabular("@{}llcrrrc@{}",
                   [r"Data & Zone & Span (2025) & Rows & Seg. & Use \% & $T_b$ ($^\circ$C) \\"],
                   rows)
    write(out / "tab_data.tex", body)


def t_accuracy(out):
    s = pd.read_csv(TABLES / "b02_summary.csv")
    v = s[s.kind == "vibration"].set_index("method")
    t = s[s.kind == "temperature"].set_index("method")
    order = v.sort_values("mean_rank").index
    rows = []
    for m in order:
        a, b = v.loc[m], t.loc[m]
        rows.append(f"{tex(nice(m))} & {FAMILY[a.family]} & {a.median_skill:+.3f} & {int(a.runs_beating_best_naive)} & "
                    f"{a.mean_rank:.1f} & {b.median_skill:+.3f} & {int(b.runs_beating_best_naive)} & {b.mean_rank:.1f} \\\\")
    hdr = [r" & & \multicolumn{3}{c}{Vibration, $h=3$ (27 runs)} & \multicolumn{3}{c}{Temperature, $h=15$ (18 runs)} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"Method & Family & Skill & Wins & Rank & Skill & Wins & Rank \\"]
    write(out / "tab_accuracy.tex", tabular("@{}llrrrrrr@{}", hdr, rows))


def t_pairs(out):
    s = pd.read_csv(TABLES / "b02_tests.csv")
    meths = ["Ridge-Univariate", "Ridge-Sensors", "AR-Local", "ChronosBolt-Small", "Chronos2", "Ensemble-Mean-13"]
    rows = []
    for m in meths:
        cells = []
        for kind in ("vibration", "temperature"):
            for ref in ("TiDE", "median trained model"):
                r = s[(s.kind == kind) & (s.method == m) & (s.reference == ref)].iloc[0]
                star = r"$^{*}$" if r.p_holm < 0.05 else ""
                cells.append(f"{r.median_mse_ratio:.3f}{star} ({int(r.runs_method_better)})")
        rows.append(f"{tex(nice(m))} & " + " & ".join(cells) + r" \\")
    hdr = [r" & \multicolumn{2}{c}{Vibration (27 runs)} & \multicolumn{2}{c}{Temperature (18 runs)} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Method & vs TiDE & vs median & vs TiDE & vs median \\"]
    write(out / "tab_pairs.tex", tabular("@{}lcccc@{}", hdr, rows))


def _anchored_alone(method, kind):
    r = pd.read_csv(TABLES / "b03_recal_run_level.csv")
    r = r[(r.kind == kind) & (r.method == method) & (r.variant == "anchored") & (r.h == OP_H[kind])]
    return float(r.ratio_vs_raw.median()), int((r.ratio_vs_raw < 1).sum()), len(r)


def t_recal(out):
    s = pd.read_csv(TABLES / "b03_recal_tests.csv").set_index(["kind", "variant"])
    k = json.loads((TABLES / "key_numbers.json").read_text())
    rows = []
    for var, lab in (("shrink", "Shrink"), ("affine", "Affine"),
                     ("anchored", "Anchored")):
        cells = []
        for kind in ("vibration", "temperature"):
            r = s.loc[(kind, var)]
            cells.append(f"{r.median_ratio:.3f} & {int(r.runs_improved)}/{N_RUNS[kind]} & {pv(r.p_wilcoxon)}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    for m, lab in (("TiDE", "Anch., TiDE only"), ("Ridge-Univariate", "Anch., ridge only")):
        cells = []
        for kind in ("vibration", "temperature"):
            med, imp, n = _anchored_alone(m, kind)
            cells.append(f"{med:.3f} & {imp}/{n} & --")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    cells = []
    for kind in ("vibration", "temperature"):
        sc = k[kind]["recal_anchored_scope"]
        cells.append(f"{sc['hindsight_best_vs_ridge']:.3f} & {sc['hindsight_best_better']}/{N_RUNS[kind]} & --")
    rows.append(r"Best anch.$^{\dagger}$/ridge & " + " & ".join(cells) + r" \\")
    hdr = [r" & \multicolumn{3}{c}{Vibration} & \multicolumn{3}{c}{Temperature} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Variant & Ratio & Impr. & $p$ & Ratio & Impr. & $p$ \\"]
    write(out / "tab_recal.tex", tabular("@{}lrrrrrr@{}", hdr, rows))


def t_robust(out):
    a7 = pd.read_csv(AUD / "a07_robustness.csv")
    b6 = pd.read_csv(TABLES / "b06_sensitivity_summary.csv")
    b10 = pd.read_csv(TABLES / "b10_seed_conclusions.csv")
    b9 = pd.read_csv(TABLES / "b09_tuning_tests.csv")

    def a7v(kind, claim, variant):
        r = a7[(a7.kind == kind) & (a7.claim == claim) & (a7.variant == variant)].iloc[0]
        return r.value, r.runs_better, r.n_runs

    def b6v(kind, comp):
        r = b6[(b6.kind == kind) & (b6.comparison == comp)].iloc[0]
        return r.value, r.runs_method_better, r.n_runs

    rows = []
    for variant, lab in (("as reported", "Primary"),
                         ("diverged cells (>10x persistence) removed", "Diverged removed")):
        cells = []
        for kind in ("vibration", "temperature"):
            nb = a7v(kind, "trained models beating best naive in a majority of runs", variant)[0]
            rv, rw, n = a7v(kind, "Ridge-Univariate vs TiDE: median MSE ratio", variant)
            an = a7v(kind, "recalibration 'anchored': median MSE ratio vs raw", variant)[0]
            cells.append(f"{int(nb)} & {rv:.3f} ({int(rw)}/{int(n)}) & {an:.3f}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    cells = []
    for kind in ("vibration", "temperature"):
        nb = b6v(kind, "trained models beating best naive in majority of runs")[0]
        rv, rw, n = b6v(kind, "Ridge-Univariate vs TiDE: median MSE ratio")
        an = b6v(kind, "anchored recalibration of trained models: median MSE ratio")[0]
        cells.append(f"{int(nb)} & {rv:.3f} ({int(rw)}/{int(n)}) & {an:.3f}")
    rows.append("Anomalies removed & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    for seed, lab in (("42", "Seed 42"), ("101", "Seed 101"),
                      ("202", "Seed 202"), ("seed-mean", "Seed mean")):
        cells = []
        for kind in ("vibration", "temperature"):
            r = b10[(b10.kind == kind) & (b10.seed.astype(str) == seed)].iloc[0]
            cells.append(f"{int(r.trained_beating_best_naive_in_majority)} & {r.ridge_vs_tide_median_ratio:.3f} "
                         f"({int(r.ridge_better_than_tide)}/{int(r.n_runs)}) & {r.anchored_median_ratio:.3f}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    cells = []
    for kind in ("vibration", "temperature"):
        r = b9[(b9.kind == kind) & (b9.a == "Tuned-Any") & (b9.b == "Ridge-Univariate")].iloc[0]
        cells.append(f"-- & {1 / r.median_ratio_a_over_b:.3f} ({int(r.n_runs - r.runs_a_better)}/{int(r.n_runs)}) & --")
    rows.append(r"Tuned-Any$^{\ddagger}$ & " + " & ".join(cells) + r" \\")
    hdr = [r" & \multicolumn{3}{c}{Vibration} & \multicolumn{3}{c}{Temperature} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Condition & $N_{>}$ & Ridge/TiDE & Anch. & $N_{>}$ & Ridge/TiDE & Anch. \\"]
    write(out / "tab_robust.tex", tabular("@{}lrcrrcr@{}", hdr, rows))


def t_tuning(out):
    b9 = pd.read_csv(TABLES / "b09_tuning_tests.csv")
    pairs = [("TiDE-Tuned", "TiDE-Original"), ("TimeXer-Tuned", "TimeXer-Original"), ("NHITS-Tuned", "NHITS-Original"),
             ("TiDE-Tuned", "TiDE-cfg0-rerun"), ("TimeXer-Tuned", "TimeXer-cfg0-rerun"),
             ("NHITS-Tuned", "NHITS-cfg0-rerun"), ("Tuned-Any", "Best-Naive"), ("Tuned-Any", "Ridge-Univariate"),
             ("Tuned-Any", "Ridge-Sensors"), ("Tuned-Any", "Chronos2")]
    lab = {"TiDE-cfg0-rerun": "TiDE re-run", "TimeXer-cfg0-rerun": "TimeXer re-run", "NHITS-cfg0-rerun": "NHITS re-run",
           "Best-Naive": "naive", "Ridge-Univariate": "ridge-U", "Ridge-Sensors": "ridge-S",
           "Chronos2": "Chronos-2"}
    rows = []
    for a, b in pairs:
        cells = []
        for kind in ("vibration", "temperature"):
            r = b9[(b9.kind == kind) & (b9.a == a) & (b9.b == b)].iloc[0]
            cells.append(f"{r.median_ratio_a_over_b:.3f} ({int(r.runs_a_better)}) & {r.p_holm:.2f}")
        rows.append(f"{tex(a.replace('-Tuned', '').replace('Tuned-Any', 'Any'))} vs {tex(lab.get(b, b.replace('-Original', ' orig.')))} & "
                    + " & ".join(cells) + r" \\")
    hdr = [r" & \multicolumn{2}{c}{Vibration} & \multicolumn{2}{c}{Temperature} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Comparison & Ratio (wins) & $p_{\mathrm{Holm}}$ & Ratio (wins) & $p_{\mathrm{Holm}}$ \\"]
    write(out / "tab_tuning.tex", tabular("@{}lcccc@{}", hdr, rows))


# ---------------------------------------------------------------- supplementary
def s_tables(out):
    parts = []
    # S: devices
    dl = pd.read_csv(DATA / "DEVICE_LABELS.csv").fillna("")
    rows = [f"{r.label} & {r.region} & {r.zone} & {tex(r.role_in_files)} & {r.status} & "
            rf"\parbox[t]{{6.2cm}}{{\raggedright {tex(r.reason)}}} \\" for _, r in dl.iterrows()]
    parts.append(longtab("llllll", [r"Device & Region & Zone & Role & Status & Note \\"], rows,
                         "Anonymised devices, roles and exclusions.", "stab:devices"))
    # S: raw device summary
    ds = pd.read_csv(AUD / "device_raw_summary.csv")
    gc = pd.read_csv(AUD / "gravity_calibration.csv").set_index("device")
    rows = [f"{r.device} & {r.raw_rows:,} & {r.raw_rows / r.distinct_minutes:.1f} & {r.coverage_of_own_span_pct:.1f} & "
            f"{gc.loc[r.device, 'med']:.4f} & {gc.loc[r.device, 'iqr_pct']:.2f} \\\\" for _, r in ds.iterrows()]
    parts.append(longtab("lrrrrr", [r"Device & Raw samples & Samples/min & Coverage (\%) & $k_d$ & IQR (\%) \\"], rows,
                         r"Raw exports per device: sample count, mean samples per active minute, coverage of the "
                         r"device's own span, and the gravity constant $k_d$ (median static resultant in raw units) "
                         r"with its inter-quartile range in percent of $k_d$.", "stab:raw"))
    # S: manifest
    man = pd.read_csv(DATA / "manifest.csv")
    rows = [f"{tex(r.dataset)} & {r.rows:,} & {r.segments} & {r.seg_min} & {r.seg_max:,} & {r.features} & "
            f"{r.origins_capacity:,} & {r.y_min:.1f} & {r.y_max:.1f} \\\\" for _, r in man.iterrows()]
    parts.append(longtab("lrrrrrrrr", [r"Dataset & Rows & Seg. & Min seg. & Max seg. & Feat. & Origin cap. & "
                                       r"$y_{\min}$ & $y_{\max}$ \\"], rows,
                         r"Released files. Segment lengths in minutes; origin capacity counts every admissible "
                         r"forecast origin (the evaluation uses 400 per run); $y$ is the reference-motor bearing "
                         r"temperature in $^\circ$C.", "stab:manifest"))
    # S: all paired tests
    s = pd.read_csv(TABLES / "b02_tests.csv").sort_values(["kind", "reference", "median_mse_ratio"])
    rows = [f"{r.kind} & {tex(nice(r.method))} & {tex(r.reference)} & {r.median_mse_ratio:.3f} & "
            f"{int(r.runs_method_better)}/{r.n_runs} & {pv(r.p_wilcoxon)} & {pv(r.p_holm)} \\\\" for _, r in s.iterrows()]
    parts.append(longtab("lllrrrr", [r"Kind & Method & Reference & Ratio & Wins & $p$ & $p_{\mathrm{Holm}}$ \\"], rows,
                         "All run-level paired comparisons at the operating horizon (median ratio of MSE, method "
                         "over reference; Wilcoxon signed-rank; Holm within kind).", "stab:tests"))
    # S: R2 and skill per method
    r2 = pd.read_csv(TABLES / "b02_r2_operating_horizon.csv")
    r2m = r2.groupby(["kind", "method"]).r2_fold_mean.median().unstack(0)
    s = pd.read_csv(TABLES / "b02_summary.csv")
    sv = s.pivot(index="method", columns="kind", values="mean_skill")
    rows = [f"{tex(nice(m))} & {r2m.loc[m, 'vibration']:.3f} & {sv.loc[m, 'vibration']:+.3f} & "
            f"{r2m.loc[m, 'temperature']:.3f} & {sv.loc[m, 'temperature']:+.3f} \\\\"
            for m in r2m.sort_values("vibration", ascending=False).index]
    parts.append(longtab("lrrrr", [r"Method & Vib. median $R^2$ & Vib. mean skill & Temp. median $R^2$ & "
                                   r"Temp. mean skill \\"], rows,
                         r"Median $R^2$ over runs (fold-averaged) and mean skill at the operating horizon. Mean "
                         r"skill is pulled down by diverged forecasts; medians are reported in the paper.",
                         "stab:r2"))
    # S: per-run skill of key methods
    rl = pd.read_csv(TABLES / "b02_run_level.csv")
    rl = rl[rl.h == rl.kind.map(OP_H)]
    key = ["TiDE", "TimeXer", "Ridge-Univariate", "Ridge-Sensors", "ChronosBolt-Small", "Chronos2"]
    piv = rl.pivot(index="run", columns="method", values="skill_vs_best_naive")
    med = rl[rl.method.isin(TRAINED_MODELS)].groupby("run").skill_vs_best_naive.median()
    nb = rl[rl.method.str.startswith("Naive")].sort_values("mse").groupby("run").first().method
    rows = []
    for run in piv.index:
        rows.append(f"{tex(run.replace('__self_', ' '))} & " + " & ".join(f"{piv.loc[run, m]:+.3f}" for m in key) +
                    f" & {med[run]:+.3f} & {tex(nice(nb[run]))} \\\\")
    parts.append(longtab("lrrrrrrrl", [r"Run & TiDE & TimeXer & Ridge-U & Ridge-S & Bolt-S & Chr.-2 & Med. trained & "
                                       r"Best naive \\"], rows,
                         "Skill against the strongest naive baseline for every run at the operating horizon "
                         "(vibration $h=3$, temperature $h=15$).", "stab:perrun"))
    # S: skill by horizon
    rl2 = pd.read_csv(TABLES / "b02_run_level.csv")
    for kind in ("vibration", "temperature"):
        g = rl2[rl2.kind == kind]
        mt = g[g.method.isin(TRAINED_MODELS)].groupby(["run", "h"]).skill_vs_best_naive.median().groupby("h").median()
        tab = g[g.method.isin(["TiDE", "Ridge-Univariate", "ChronosBolt-Small", "Chronos2", "Naive-Persistence"])] \
            .groupby(["h", "method"]).skill_vs_best_naive.median().unstack()
        rows = [f"{h} & {tab.loc[h, 'TiDE']:+.3f} & {tab.loc[h, 'Ridge-Univariate']:+.3f} & "
                f"{tab.loc[h, 'ChronosBolt-Small']:+.3f} & {tab.loc[h, 'Chronos2']:+.3f} & "
                f"{tab.loc[h, 'Naive-Persistence']:+.3f} & {mt[h]:+.3f} \\\\" for h in tab.index]
        parts.append(longtab("rrrrrrr", [r"$h$ & TiDE & Ridge-U & Bolt-S & Chronos-2 & Persistence & Med. trained \\"],
                             rows, f"Median skill by horizon, {kind} (cumulative MSE over steps $1..h$).",
                             f"stab:horizon_{kind}"))
    # S: decomposition
    dc = pd.read_csv(TABLES / "b04_overconfidence_by_model.csv").sort_values(["kind", "median_overconf"])
    rows = [f"{r.kind} & {tex(nice(r.method))} & {r.median_overconf:.3f} & {r.median_r:.3f} & {r.median_bias_pct:.1f} & "
            f"{r.median_cond_bias_pct:.1f} & {r.median_unexplained_pct:.1f} \\\\" for _, r in dc.iterrows()]
    parts.append(longtab("llrrrrr", [r"Kind & Method & $\rho/r$ & $r$ & Bias (\%) & Cond. bias (\%) & Unexpl. (\%) \\"],
                         rows, r"Murphy decomposition of MSE (median over runs). $\rho/r$ is the ratio of forecast to "
                               r"observed standard deviation divided by their correlation; 1 means no conditional "
                               r"bias, $>1$ means the forecast swings too much for its correlation.", "stab:decomp"))
    # S: recalibration by family and per trained model
    rs = pd.read_csv(TABLES / "b03_recal_summary.csv")
    rows = [f"{r.kind} & {r.family} & {r.variant} & {r.cells} & {r.median_ratio:.3f} & {100 * r.share_improved:.1f} \\\\"
            for _, r in rs.iterrows()]
    parts.append(longtab("lllrrr", [r"Kind & Family & Variant & Cells & Median ratio & Improved (\%) \\"], rows,
                         "Causal recalibration by method family (cells = run $\\times$ method; folds 2--5).",
                         "stab:recal_family"))
    rr = pd.read_csv(TABLES / "b03_recal_run_level.csv")
    rr = rr[(rr.variant == "anchored") & (rr.h == rr.kind.map(OP_H)) & (rr.method.isin(TRAINED_MODELS))]
    pm = rr.groupby(["method", "kind"]).ratio_vs_raw.agg(["median", lambda x: int((x < 1).sum())]).unstack()
    rows = [f"{m} & {pm.loc[m, ('median', 'vibration')]:.3f} & {int(pm.loc[m, ('<lambda_0>', 'vibration')])}/27 & "
            f"{pm.loc[m, ('median', 'temperature')]:.3f} & {int(pm.loc[m, ('<lambda_0>', 'temperature')])}/18 \\\\"
            for m in pm.sort_values(("median", "vibration")).index]
    parts.append(longtab("lrrrr", [r"Model & Vib. ratio & Improved & Temp. ratio & Improved \\"], rows,
                         "Anchored recalibration per trained model (MSE after / before, folds 2--5).",
                         "stab:recal_model"))
    st = pd.read_csv(TABLES / "b03_recal_best_model_stability.csv")
    rows = [f"{r.kind} & {r.variant} & {r.best_trained_model_unchanged}/{r.runs} \\\\" for _, r in st.iterrows()]
    parts.append(longtab("llr", [r"Kind & Variant & Best trained model unchanged \\"], rows,
                         "Does recalibration change which trained model is best in a run?", "stab:recal_stab"))
    # S: predictability
    c = pd.read_csv(TABLES / "b05_correlations.csv")
    rows = [f"{r.scope} & {tex(r.score.replace('_uncapped', ''))} & {tex(r.outcome)} & {r.n} & {r.spearman:+.3f} & "
            f"{pv(r.p)} & {pv(r.p_holm_4_outcomes)} \\\\" for _, r in c.iterrows()]
    parts.append(longtab("lllrrrr", [r"Scope & Score & Outcome & $n$ & Spearman & $p$ & $p_{\mathrm{Holm}}$ \\"], rows,
                         r"Predictability scores computed on training rows only versus model outcomes. "
                         r"$p_{\mathrm{Holm}}$ is adjusted over the four outcomes of each scope and score.",
                         "stab:lambda"))
    # S: anomaly sensitivity
    b6 = pd.read_csv(TABLES / "b06_sensitivity_summary.csv")
    rows = [f"{r.kind} & {tex(r.comparison)} & {r.value:.3f} & "
            f"{'' if pd.isna(r.runs_method_better) else int(r.runs_method_better)} & {pv(r.p)} \\\\"
            for _, r in b6.iterrows()]
    parts.append(longtab("llrrr", [r"Kind & Comparison & Value & Wins & $p$ \\"], rows,
                         "Main comparisons after dropping every origin whose input or forecast window touches a "
                         "flagged anomaly (17,315 of 18,000 origins kept).", "stab:anom"))
    # S: divergent cells
    dv = pd.read_csv(AUD / "a07_divergent_cells.csv").sort_values("x_persistence", ascending=False)
    rows = [f"{tex(r.run.replace('__self_', ' '))} & {tex(r.fold[:2])} & {r.method} & {r.x_persistence:.1f} \\\\"
            for _, r in dv.iterrows()]
    parts.append(longtab("lllr", [r"Run & Fold & Model & MSE / persistence MSE \\"], rows,
                         "Trained-model cells whose operating-horizon MSE exceeds ten times that of persistence on "
                         "the same fold (41 of 2,925).", "stab:diverged"))
    a7 = pd.read_csv(AUD / "a07_robustness.csv")
    rows = []
    for (kind, claim), g in a7.groupby(["kind", "claim"], sort=False):
        a = g[g.variant == "as reported"].iloc[0]
        b = g[g.variant != "as reported"].iloc[0]
        fw = lambda r: "" if pd.isna(r.runs_better) else f" ({int(r.runs_better)})"  # noqa: E731
        rows.append(f"{kind} & {tex(claim)} & {a.value:.3f}{fw(a)} & {b.value:.3f}{fw(b)} \\\\")
    parts.append(longtab("llrr", [r"Kind & Claim & As reported & Diverged removed \\"], rows,
                         "Every headline comparison with and without the diverged cells.", "stab:diverged_rob"))
    # S: tuning
    t9 = pd.read_csv(TABLES / "b09_tuning_summary.csv").sort_values(["kind", "median_ratio_to_tide_original"])
    rows = [f"{r.kind} & {tex(r.method.replace('(test-selected, reference only)', ' (ref.)'))} & {r.median_mse:.4f} & "
            f"{r.median_ratio_to_tide_original:.3f} & {r.runs_beating_best_naive}/9 \\\\" for _, r in t9.iterrows()]
    parts.append(longtab("llrrr", [r"Kind & Method & Median MSE & Ratio to TiDE orig. & Beats naive \\"], rows,
                         "Hyperparameter check: every method on the 18 tuning runs. Oracle rows select on the test "
                         "block and are shown for reference only.", "stab:tune_sum"))
    t9t = pd.read_csv(TABLES / "b09_tuning_tests.csv")
    rows = [f"{r.kind} & {tex(r.a)} & {tex(r.b)} & {r.median_ratio_a_over_b:.3f} & {r.runs_a_better}/9 & "
            f"{pv(r.p_wilcoxon)} & {r.p_holm:.2f} \\\\" for _, r in t9t.iterrows()]
    parts.append(longtab("lllrrrr", [r"Kind & A & B & Ratio A/B & A better & $p$ & $p_{\mathrm{Holm}}$ \\"], rows,
                         "Hyperparameter check: all pre-registered paired tests.", "stab:tune_tests"))
    sel = pd.read_csv(TABLES / "b09_tuning_selection.csv")
    g = sel.groupby(["kind", "model"]).agg(folds=("fold", "size"), orig=("cfg_selected", lambda x: (x == 0).mean()),
                                           same_as_oracle=("cfg_selected", "size")).reset_index()
    sel["agree"] = sel.cfg_selected == sel.cfg_oracle_REFERENCE_ONLY
    ag = sel.groupby(["kind", "model"]).agree.mean()
    sel["rr"] = sel.test_mse_cfg0_rerun / sel.test_mse_original_stored
    rr_ = sel.groupby(["kind", "model"]).rr.agg(["min", "median", "max"])
    rows = [f"{r.kind} & {r.model} & {r.folds} & {100 * r.orig:.0f} & {100 * ag[(r.kind, r.model)]:.0f} & "
            f"{rr_.loc[(r.kind, r.model), 'min']:.2f} / {rr_.loc[(r.kind, r.model), 'median']:.2f} / "
            f"{rr_.loc[(r.kind, r.model), 'max']:.2f} \\\\" for _, r in g.iterrows()]
    parts.append(longtab("llrrrc", [r"Kind & Model & Folds & Orig. chosen (\%) & = oracle (\%) & Re-run/stored "
                                    r"min/med/max \\"], rows,
                         "Configuration selection on validation blocks, and reproducibility of the original "
                         "configuration when re-trained on different hardware (test MSE ratio per run-fold).",
                         "stab:tune_sel"))
    # S: seeds
    sp = pd.read_csv(TABLES / "b10_seed_spread.csv").sort_values(["kind", "median_cv"])
    rows = [f"{r.kind} & {r.model} & {r.n_runs} & {r.median_cv:.3f} & {r.median_max_over_min:.3f} & "
            f"{r.worst_max_over_min:.2f} & {100 * r.share_seed42_best:.0f} \\\\" for _, r in sp.iterrows()]
    parts.append(longtab("llrrrrr", [r"Kind & Model & Runs & Median CV & Median max/min & Worst max/min & "
                                     r"Seed 42 best (\%) \\"], rows,
                         "Training-seed variability (seeds 42, 101, 202; 14 iteration-1 runs).", "stab:seeds"))
    sc = pd.read_csv(TABLES / "b10_seed_conclusions.csv")
    rows = [f"{r.kind} & {r.seed} & {r.trained_beating_best_naive_in_majority} & {r.ridge_vs_tide_median_ratio:.3f} "
            f"({r.ridge_better_than_tide}/{r.n_runs}) & {r.ridge_vs_median_trained_ratio:.3f} & "
            f"{r.tide_best_trained_model_runs}/{r.n_runs} & {r.anchored_median_ratio:.3f} \\\\"
            for _, r in sc.iterrows()]
    parts.append(longtab("llrrrrr", [r"Kind & Seed & $N_{>}$ & Ridge/TiDE & Ridge/median & TiDE best & Anchored \\"],
                         rows, "Headline conclusions recomputed at each seed and for the per-model mean over seeds.",
                         "stab:seed_concl"))
    # S: transfer, floor, pilot
    tr = pd.read_csv(AUD / "a04_transfer_summary.csv")
    tr = tr[~tr.excl_R2_I3_target]
    rows = [f"{tex(r.channel)} & {r.model} & {r['shift']} & {r.cells} & {r.mean_mse:.4f} & {r.median_ratio_vs_in_domain:.3f} \\\\"
            for _, r in tr.iterrows()]
    parts.append(longtab("lllrrr", [r"Channel & Model & Shift & Cells & Mean MSE & Median ratio to in-domain \\"], rows,
                         "Zero-shot cross-domain transfer (fold F5, 80 origins per cell). Mean MSE is in "
                         "standardised units of the source domain.", "stab:transfer"))
    fl = pd.read_csv(AUD / "a04_floor_summary.csv")
    rows = [f"{r.model} & {r.n_targets} & {r.median_delta_pct:+.2f} & {r.targets_improved}/{r.n_targets} & "
            f"{r.p_wilcoxon:.2f} \\\\" for _, r in fl.iterrows()]
    parts.append(longtab("lrrrr", [r"Model & Targets & Median change (\%) & Improved & $p$ \\"], rows,
                         "Variance-floored Gaussian likelihood training (iteration 1, unit = target). The floor was "
                         "derived from the full record and is therefore not a causal method; reported for "
                         "completeness.", "stab:floor"))
    pi = pd.read_csv(AUD / "a08_pilot_roster.csv")
    rows = [f"{r.model} & {f2(r.R1)} & {f2(r.R2)} & {f2(r.R3)} & {f2(r.mean_3_regions)} \\\\" for _, r in pi.iterrows()]
    parts.append(longtab("lrrrr", [r"Model & R1 & R2 & R3 & Mean \\"], rows,
                         "Pilot run that fixed the roster (bearing temperature, iteration 1, earlier overlapping "
                         "split scheme): MSE divided by persistence MSE. FEDformer produced no finite forecasts.",
                         "stab:pilot"))
    # S: verification checklist
    a6 = pd.read_csv(AUD / "a06_final_audit.csv")
    rows = [rf"\parbox[t]{{12.5cm}}{{\raggedright {tex(r.check)}}} & {'pass' if r.ok else 'FAIL'} \\"
            for _, r in a6.iterrows()]
    parts.append(longtab("lr", [r"Automated check & Result \\"], rows,
                         "Final automated verification run (script \\texttt{a06\\_final\\_audit.py}).", "stab:a06"))
    write(out / "supp_tables.tex", "\n".join(parts))


def s_static(out):
    """Settings copied from the training scripts, and the column dictionary from schema.json."""
    parts = []
    models = [
        ("NHITS", "library defaults (three MLP stacks)"),
        ("TiDE", "hidden size 256; other settings library defaults"),
        ("BiTCN", "hidden size 32"),
        ("TFT", "hidden size 64, 4 heads, dropout 0.1, window batch 64"),
        ("TSMixerx", "2 blocks, feed-forward size 64"),
        ("TimeXer", "hidden size 128, 4 heads, 2 encoder layers, patch length 16"),
        ("NBEATSx", "library defaults"),
        ("MLP", "hidden size 256, 3 layers"),
        ("TCN", "encoder and decoder hidden size 128"),
        ("DilatedRNN", "encoder hidden size 128"),
        ("KAN", "library defaults"),
        ("DeepNPTS", "hidden size 128"),
        ("TabNet", r"pytorch-tabnet regressor on the flattened 128-minute history; $n_d=n_a=32$, 4 steps, "
                   r"$\gamma=1.5$, batch 512 (virtual 128), at most 100 epochs, patience 15"),
    ]
    rows = [rf"{m} & \parbox[t]{{10cm}}{{\raggedright {d}}} \\" for m, d in models]
    common = (r"Shared settings (NeuralForecast models): input 128 min, 15-step output, learning rate $10^{-3}$ "
              r"(Adam), batch size 32, window batch 256 where applicable, at most 3000 steps, validation every 50 "
              r"steps, early stopping after 15 checks without improvement, gradient clipping 1.0, per-window robust "
              r"scaler, seed 42, all non-target channels as historical covariates.")
    parts.append(longtab("ll", [r"Model & Architecture settings \\"], rows,
                         "Settings of the 13 trained models, fixed once without tuning. " + common, "stab:models"))
    grid = [("TiDE", "256 / 0.3 / $10^{-3}$ / 1 layer", r"hidden \{128, 256, 512\}; dropout \{0.1, 0.3\}; "
             r"learning rate \{$3\times10^{-4}$, $10^{-3}$, $3\times10^{-3}$\}; encoder layers \{1, 2\}"),
            ("NHITS", "512 / 0.0 / $10^{-3}$", r"MLP width \{256, 512, 1024\}; dropout on $\theta$ \{0.0, 0.2\}; "
             r"learning rate \{$3\times10^{-4}$, $10^{-3}$, $3\times10^{-3}$\}"),
            ("TimeXer", "128 / 0.1 / $10^{-3}$", r"hidden \{64, 128, 256\}; dropout \{0.1, 0.3\}; "
             r"learning rate \{$3\times10^{-4}$, $10^{-3}$, $3\times10^{-3}$\}")]
    rows = [rf"{m} & {o} & \parbox[t]{{8.5cm}}{{\raggedright {g}}} \\" for m, o, g in grid]
    parts.append(longtab("lll", [r"Model & Original (config.~0) & Grid \\"], rows,
                         "Hyperparameter check: configuration 0 is the original setting; configurations 1--7 were "
                         "drawn without replacement from the grid with random seed 2026 before any run.",
                         "stab:grid"))
    desc = {
        "y_roll_mean_15": "trailing 15-min mean of $y$", "y_roll_iqr_60": "trailing 60-min inter-quartile range of $y$",
        "temperature_one": r"drive-end bearing temperature ($^\circ$C, PT100, corrected)",
        "temperature_two": r"motor-body temperature ($^\circ$C)", "t_grad": "bearing minus body temperature",
        "t1_diff_15": "15-min change of bearing temperature",
        "vib_ac": r"vibration magnitude: Euclidean norm of the three axis amplitudes (m/s$^2$)",
        "vib_dc": r"static acceleration: norm of the per-minute mean of each axis (m/s$^2$, $\approx g$)",
        "d_self_nb1": "self minus nb1 bearing temperature", "d_self_nb2": "self minus nb2 bearing temperature",
        "tod_sin": "time of day, sine", "tod_cos": "time of day, cosine"}
    sch = json.loads((DATA / "schema.json").read_text())
    rows = [r"\texttt{unique\_id}, \texttt{ds} & series identifier and UTC timestamp (1-min) \\",
            r"\texttt{segment\_id} & index of the gap-free segment \\",
            r"\texttt{y} & copy of \texttt{self\_temperature\_one} (the original target column) \\"]
    for c in sch["kept_columns"]:
        if c in desc:
            d = desc[c]
        elif c.startswith("y_roll_std_"):
            d = f"trailing {c.split('_')[-1]}-min standard deviation of $y$"
        elif c.startswith("y_diff_"):
            d = f"{c.split('_')[-1]}-min difference of $y$"
        else:
            role, rest = c.split("_", 1)
            if rest in desc:
                d = desc[rest]
            elif rest.endswith("_mean_60"):
                d = f"trailing 60-min mean of the {rest[0]}-axis amplitude"
            elif rest.endswith("_std_60"):
                d = f"trailing 60-min standard deviation of the {rest[0]}-axis amplitude"
            elif rest in ("x_ac", "y_ac", "z_ac"):
                d = rf"{rest[0]}-axis vibration amplitude: trailing 3-min standard deviation of raw samples (m/s$^2$)"
            else:
                raise KeyError(c)
            d = f"{role}: {d}"
        rows.append(rf"\texttt{{{tex(c)}}} & \parbox[t]{{9.5cm}}{{\raggedright {d}}} \\")
    parts.append(longtab("ll", [r"Column & Meaning \\"], rows,
                         "Columns of the released files (56 feature columns plus identifiers). Prefixes: self = "
                         "reference motor, nb1/nb2 = neighbours. All rolling quantities are trailing and computed "
                         "within segments. Region 3 has no second neighbour; its nb2 columns are zero.",
                         "stab:columns"))
    write(out / "supp_static.tex", "\n".join(parts))


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(root / "paper_tables" / "main"))
    ap.add_argument("--supp", default=str(root / "paper_tables" / "supp"))
    a = ap.parse_args()
    m, s = Path(a.main), Path(a.supp)
    for fn in (t_data, t_accuracy, t_pairs, t_recal, t_robust, t_tuning):
        fn(m)
    s_tables(s)
    s_static(s)


if __name__ == "__main__":
    main()
