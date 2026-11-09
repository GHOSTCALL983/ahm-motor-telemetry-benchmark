"""B7 - figures, all drawn from the result tables (no number is typed by hand).

fig1_skill_by_method.png       skill vs strongest naive, operating horizon, every run
fig2_skill_vs_horizon.png      median skill across runs, horizons 1..15
fig3_overconfidence.png        sd_ratio / r of each trained model, by channel kind
fig4_anchored_recalibration.png  causal anchored recalibration, per trained model
fig5_R2_I3_shutdown.png        the shutdown inside dataset R2_I3
fig6_transfer.png              cross-domain transfer cost (ratio to in-domain MSE)
Colour = model family (validated categorical slots 1-3 + neutral grey for naive);
families also differ by marker shape and every method is labelled in text.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, FIGURES, RESULTS, TABLES, TRAINED_MODELS  # noqa: E402

FAM_COLOR = {"trained": "#2a78d6", "ensemble": "#2a78d6", "closed-form": "#eb6834", "pretrained": "#1baf7a",
             "naive": "#8a8984"}
FAM_MARK = {"trained": "o", "ensemble": "*", "closed-form": "s", "pretrained": "D", "naive": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OP_H = {"vibration": 3, "temperature": 15}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig1(rl):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.2))
    for ax, kind in zip(axes, ("vibration", "temperature")):
        g = rl[(rl.kind == kind) & (rl.h == OP_H[kind])]
        order = g.groupby("method").skill_vs_best_naive.median().sort_values()
        lo = -1.0
        for y, m in enumerate(order.index):
            gg = g[g.method == m]
            fam = gg.family.iloc[0]
            v = gg.skill_vs_best_naive.clip(lower=lo)
            ax.scatter(v, np.full(len(v), y) + np.random.default_rng(y).uniform(-.18, .18, len(v)),
                       s=10, color=FAM_COLOR[fam], alpha=.45, marker=FAM_MARK[fam], lw=0)
            ax.scatter([max(order[m], lo)], [y], s=46, color=FAM_COLOR[fam], marker=FAM_MARK[fam],
                       edgecolor="#fcfcfb", lw=1.5, zorder=3)
        ax.set_yticks(range(len(order)), order.index)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(lo - .05, max(.45, g.skill_vs_best_naive.max() + .05))
        ax.set_xlabel(f"skill vs strongest naive baseline (1 - MSE ratio), h = {OP_H[kind]} min")
        ax.set_title(f"{kind.capitalize()}: {g.run.nunique()} runs", loc="left", color=INK)
        _grid(ax)
    handles = [plt.Line2D([], [], color=c, marker=FAM_MARK[f], ls="", label=f) for f, c in FAM_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.text(0.01, 0.005, "Large dot = median over runs; small dots = individual runs (values below -1 drawn at -1). "
             "Right of 0 = beats the best of persistence / MA15 / MA60 / EWMA60.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig1_skill_by_method.png")
    plt.close(fig)


def fig2(rl):
    keep = {"TiDE": "TiDE (best trained)", "Ridge-Univariate": "Ridge (closed form)",
            "ChronosBolt-Small": "Chronos-Bolt (zero-shot)", "Chronos2-Sensors": "Chronos-2 + sensors (zero-shot)",
            "Naive-Persistence": "Persistence"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    ylims = {"vibration": (-1.05, 0.4), "temperature": (-1.05, 0.4)}
    for ax, kind in zip(axes, ("vibration", "temperature")):
        g = rl[rl.kind == kind]
        lo = ylims[kind][0]
        top = g[g.method.isin(keep)].groupby(["method", "h"]).skill_vs_best_naive.median().max()
        hi = max(ylims[kind][1], top + 0.08)
        med_tr = g[g.method.isin(TRAINED_MODELS)].groupby(["run", "h"]).skill_vs_best_naive.median()             .groupby("h").median()
        ax.plot(med_tr.index, med_tr.clip(lower=lo).values, color=FAM_COLOR["trained"], lw=2, ls="--",
                label="median of 13 trained")
        if (med_tr < lo).any():
            below = med_tr[med_tr < lo]
            ax.text(1.2, lo + .06, f"median of 13 trained is below -1 at h = {int(below.index.min())}-"
                    f"{int(below.index.max())} ({below.iloc[0]:.1f} at h=1, {med_tr.iloc[-1]:.1f} at h=15); "
                    "drawn at the axis edge", fontsize=7.5, color=INK2)
        for m, lab in keep.items():
            gg = g[g.method == m]
            if gg.empty:
                continue
            s = gg.groupby("h").skill_vs_best_naive.median()
            ax.plot(s.index, s.clip(lower=lo).values, color=FAM_COLOR[gg.family.iloc[0]], lw=2,
                    marker=FAM_MARK[gg.family.iloc[0]], ms=4, label=lab)
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(OP_H[kind], color=GRID, lw=1)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0.6, 15.4)
        ax.set_xticks([1, 3, 5, 10, 15])
        ax.set_xlabel("forecast horizon h (min, cumulative MSE over steps 1..h)")
        ax.set_ylabel("median skill vs strongest naive")
        ax.set_title(kind.capitalize(), loc="left", color=INK)
        _grid(ax, "y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig2_skill_vs_horizon.png")
    plt.close(fig)


def fig3(oc):
    t = oc[oc.family == "trained"].pivot(index="method", columns="kind", values="median_overconf")
    t = t.sort_values("vibration")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for y, (m, row) in enumerate(t.iterrows()):
        ax.plot([row.temperature, row.vibration], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter(row.vibration, y, color="#2a78d6", s=40, zorder=2, marker="o")
        ax.scatter(row.temperature, y, color="#eb6834", s=40, zorder=2, marker="s")
    ax.set_yticks(range(len(t)), t.index)
    ax.axvline(1, color=INK2, lw=1)
    ax.text(1.02, len(t) - .4, "calibrated (= 1)", color=INK2, fontsize=8)
    ax.set_xlabel("median over runs of  sd(forecast)/sd(truth) / r   (> 1: swings too large)")
    ax.legend(handles=[plt.Line2D([], [], color="#2a78d6", marker="o", ls="", label="vibration (h = 3)"),
                       plt.Line2D([], [], color="#eb6834", marker="s", ls="", label="temperature (h = 15)")],
              frameon=False, loc="lower right")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_overconfidence.png")
    plt.close(fig)


def fig4(rec):
    g = rec[(rec.family == "trained") & (rec.variant == "anchored")]
    t = g.groupby(["method", "kind"]).ratio_vs_raw.median().unstack().sort_values("vibration")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for y, (m, row) in enumerate(t.iterrows()):
        ax.scatter(row.vibration, y, color="#2a78d6", s=40, marker="o")
        ax.scatter(row.temperature, y, color="#eb6834", s=40, marker="s")
    ax.set_yticks(range(len(t)), t.index)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xlabel("MSE after / before anchored recalibration (median over runs; < 1 = better)")
    ax.legend(handles=[plt.Line2D([], [], color="#2a78d6", marker="o", ls="", label="vibration"),
                       plt.Line2D([], [], color="#eb6834", marker="s", ls="", label="temperature")],
              frameon=False, loc="lower right")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_anchored_recalibration.png")
    plt.close(fig)


def fig5():
    d = pd.read_csv(DATA / "R2_I3.csv", usecols=["ds", "y", "self_vib_ac"])
    d["ds"] = pd.to_datetime(d.ds)
    d = d.set_index("ds").resample("10min").mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    a1.plot(d.index, d.y, color="#2a78d6", lw=1.2)
    a1.set_ylabel("bearing temp. (degC)")
    a2.plot(d.index, d.self_vib_ac, color="#2a78d6", lw=1.2)
    a2.set_ylabel("vibration AC (m/s2)")
    quiet = d.index[d.self_vib_ac < 0.1]          # 10-min mean vibration below 0.1 m/s2 = not running
    s0, s1 = quiet.min(), quiet.max()
    for a in (a1, a2):
        a.axvspan(s0, s1, color=GRID, lw=0)
        _grid(a, "y")
    import matplotlib.dates as mdates
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a1.set_title(f"Dataset R2_I3 (reference sensor): motor stopped {s0:%b %d %H:%M} - {s1:%b %d %H:%M} UTC "
                 "(shaded: 10-min mean vibration < 0.1 m/s2)", loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_R2_I3_shutdown.png")
    plt.close(fig)


def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ch in zip(axes, ("self_x_ac", "self_temperature_one")):
        t = pd.read_csv(RESULTS / "transfer" / f"transfer_matrix_{ch}.csv")
        t = t[t.model == "TiDE"]
        diag = t[t["shift"] == "diagonal"].set_index("target").MSE
        t["ratio"] = t.MSE / t.target.map(diag)
        doms = sorted(t.source.unique(), key=lambda s: (s[-1], s[:2]))
        m = t.pivot(index="source", columns="target", values="ratio").loc[doms, doms]
        im = ax.imshow(m.values, cmap="Blues", vmin=1, vmax=max(1.5, np.nanpercentile(m.values, 95)))
        for i in range(len(doms)):
            for j in range(len(doms)):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#fcfcfb" if v > 1.3 else INK)
        ax.set_xticks(range(len(doms)), doms, rotation=60)
        ax.set_yticks(range(len(doms)), doms)
        ax.set_xlabel("target domain (test)")
        ax.set_ylabel("source domain (training)")
        ax.set_title(f"TiDE, {'vibration x' if 'x_ac' in ch else 'bearing temperature'}: MSE / in-domain MSE",
                     loc="left", color=INK, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_transfer.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    rl = pd.read_csv(TABLES / "b02_run_level.csv")
    fig1(rl)
    fig2(rl)
    fig3(pd.read_csv(TABLES / "b04_overconfidence_by_model.csv"))
    fig4(pd.read_csv(TABLES / "b03_recal_run_level.csv"))
    fig5()
    fig6()
    print("[ok] figures written to", FIGURES)
