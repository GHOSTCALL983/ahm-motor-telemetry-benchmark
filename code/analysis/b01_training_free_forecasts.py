"""B1 - training-free forecasts at exactly the trained models' forecast origins.

For each of the 45 runs and each of its 5 folds, forecasts for horizons 1..15
are produced at the SAME origins the 13 neural models were scored on (read
from the run's .npz). Nothing here uses a test-block value as input except the
history strictly before the origin, which is what every model receives.

Methods (none involves iterative/gradient training):
  Naive-Persistence   last observed value
  Naive-MA15/MA60     mean of the last 15/60 minutes
  Naive-EWMA60        exponentially weighted mean (span 60), within the segment
  AR-Global           y = mu + phi_h (y_last - mu); mu and phi_h are moments of
                      the fold's TRAINING rows (closed form)
  AR-Local            same, but anchored at the mean of the 128-min input window
  Ridge-Univariate    closed-form ridge, 128 lags (relative to last value) ->
                      15 horizons; penalty chosen on the fold's VALIDATION block
  Ridge-Sensors       as above plus the last value of every exogenous channel the
                      neural models received (same leakage exclusions)
  ChronosBolt-Small   pretrained, zero-shot, univariate, 128-min context
  Chronos2            pretrained, zero-shot, univariate, 128-min context
  Chronos2-Sensors    pretrained, zero-shot, 128-min context + the same past
                      exogenous channels the neural models received

Output: results/training_free_predictions/{run}.npz  keys "{method}|{fold}"
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, FOLD_FRAC, FOLDS, INPUT_SIZE, PRED_DIR, RESULTS, TARGETS  # noqa: E402

H = 15
OUT = RESULTS / "training_free_predictions"
RIDGE_GRID = [1e-2, 1e-1, 1, 10, 100, 1000]
STRIDE = 2                     # subsample training windows for the ridge design matrix
SKIP_CHRONOS = "--no-chronos" in sys.argv
SENSORS_PASS = "--chronos2-sensors" in sys.argv     # separate slow pass -> {run}__c2s.npz
ONLY_REGION = next((a.split("=")[1] for a in sys.argv if a.startswith("--region=")), None)
THREADS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--threads=")), "0"))


def valid_starts(seg, lo, hi):
    """Origins t with [t-128, t+H) inside one segment and lo <= t, t+H <= hi."""
    t = np.arange(max(lo, INPUT_SIZE), hi - H + 1)
    if t.size == 0:
        return t
    ok = (seg[t - INPUT_SIZE] == seg[t + H - 1]) & (seg[t] == seg[t - INPUT_SIZE])
    return t[ok]


def lag_matrix(y, starts):
    idx = starts[:, None] + np.arange(-INPUT_SIZE, 0)[None, :]
    return y[idx]                                   # [n, 128], last column = y[t-1]


def fut_matrix(y, starts):
    return y[starts[:, None] + np.arange(H)[None, :]]


def ridge_fit(X, Y, alpha):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return lambda Z: (Z - mx) @ W + my


def ewma_last(y, seg, o, span=60):
    s0 = np.flatnonzero(seg[:o] != seg[o - 1])
    start = s0[-1] + 1 if s0.size else 0
    return pd.Series(y[start:o]).ewm(span=span, adjust=False).mean().iloc[-1]


def forecasts_for_fold(y, X_ex, seg, origins, N, fold):
    tr_end, va_end, te_end = (int(N * f) for f in FOLD_FRAC[fold])
    lag_te = lag_matrix(y, origins)
    last = lag_te[:, -1:]
    out = {"Naive-Persistence": np.repeat(last, H, 1),
           "Naive-MA15": np.repeat(lag_te[:, -15:].mean(1, keepdims=True), H, 1),
           "Naive-MA60": np.repeat(lag_te[:, -60:].mean(1, keepdims=True), H, 1),
           "Naive-EWMA60": np.repeat(np.array([[ewma_last(y, seg, o)] for o in origins]), H, 1)}

    tr = valid_starts(seg, 0, tr_end)               # training windows end before tr_end
    va = valid_starts(seg, tr_end, va_end)          # validation windows (selection only)
    Ltr, Ftr = lag_matrix(y, tr), fut_matrix(y, tr)

    # AR-Global: anchor = training mean
    mu = y[:tr_end].mean()
    a = Ltr[:, -1] - mu
    phi = (a[:, None] * (Ftr - mu)).sum(0) / (a ** 2).sum()
    out["AR-Global"] = mu + phi[None, :] * (last - mu)
    # AR-Local: anchor = mean of the 128-min input window
    m_tr = Ltr.mean(1, keepdims=True)
    a = Ltr[:, -1:] - m_tr
    phi_l = (a * (Ftr - m_tr)).sum(0) / (a[:, 0] ** 2).sum()
    m_te = lag_te.mean(1, keepdims=True)
    out["AR-Local"] = m_te + phi_l[None, :] * (last - m_te)

    # Ridge (NLinear-style: everything relative to the last observed value)
    def design(L, starts, with_ex):
        rel = L - L[:, -1:]
        if not with_ex:
            return rel[:, :-1]
        return np.hstack([rel[:, :-1], X_ex[starts - 1]])
    for name, with_ex in (("Ridge-Univariate", False), ("Ridge-Sensors", True)):
        if with_ex and X_ex is None:
            continue
        trs = tr[::STRIDE]
        Xtr = design(lag_matrix(y, trs), trs, with_ex)
        Ytr = fut_matrix(y, trs) - lag_matrix(y, trs)[:, -1:]
        Xva = design(lag_matrix(y, va), va, with_ex)
        Yva = fut_matrix(y, va) - lag_matrix(y, va)[:, -1:]
        best = None
        for al in RIDGE_GRID:
            f = ridge_fit(Xtr, Ytr, al)
            e = float(((f(Xva) - Yva) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, al)
        f = ridge_fit(Xtr, Ytr, best[1])
        out[name] = f(design(lag_te, origins, with_ex)) + last
        out[name + "_alpha"] = np.array([best[1]])
    return out


def chronos_forecasts(pipes, y, ex_df, origins, sensors=False):
    """sensors=False: univariate Chronos-Bolt and Chronos-2.
    sensors=True : Chronos-2 with past covariates only (slow; separate pass)."""
    import torch
    res = {}
    ctx = [y[o - INPUT_SIZE:o].astype("float32") for o in origins]
    if not sensors:
        _, m = pipes["bolt"].predict_quantiles([torch.from_numpy(c) for c in ctx],
                                               prediction_length=H, quantile_levels=[0.5])
        res["ChronosBolt-Small"] = m.numpy().astype(float)
        _, m = pipes["c2"].predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        res["Chronos2"] = np.stack([t[0].numpy() for t in m]).astype(float)
    elif ex_df is not None:
        exv = ex_df.to_numpy("float32")
        inputs = [{"target": c, "past_covariates": {n: exv[o - INPUT_SIZE:o, j]
                                                    for j, n in enumerate(ex_df.columns)}}
                  for c, o in zip(ctx, origins)]
        _, m = pipes["c2"].predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.5])
        res["Chronos2-Sensors"] = np.stack([t[0].numpy() for t in m]).astype(float)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipes = None
    if not SKIP_CHRONOS:
        import torch
        from chronos import BaseChronosPipeline
        torch.manual_seed(0)
        if THREADS:
            torch.set_num_threads(THREADS)
        pipes = {"bolt": BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu",
                                                             torch_dtype=torch.float32),
                 "c2": BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu",
                                                           torch_dtype=torch.float32)}
    for r in (1, 2, 3):
        if ONLY_REGION and f"R{r}" != ONLY_REGION:
            continue
        for i in (1, 2, 3):
            df = pd.read_csv(DATA / f"R{r}_I{i}.csv")
            N, seg = len(df), df.segment_id.to_numpy()
            for t in TARGETS:
                run = f"R{r}_I{i}__{t}"
                target_file = OUT / (f"{run}__c2s.npz" if SENSORS_PASS else f"{run}.npz")
                if target_file.exists():
                    continue
                t0 = time.time()
                cfg = json.loads((PRED_DIR / f"{run}.config.json").read_text())
                col = "y" if t == "self_temperature_one" else t
                y = df[col].to_numpy(float)
                ex_cols = [c for c in cfg["exog"] if c in df.columns]
                z = np.load(PRED_DIR / f"{run}.npz")
                store = {}
                for fold in FOLDS:
                    o = z[f"origins|{fold}"].astype(int)
                    tr_end = int(N * FOLD_FRAC[fold][0])
                    ex = df[ex_cols].to_numpy(float)
                    mu_ex, sd_ex = ex[:tr_end].mean(0), ex[:tr_end].std(0)
                    sd_ex[sd_ex == 0] = 1.0
                    X_ex = (ex - mu_ex) / sd_ex
                    if SENSORS_PASS:
                        fc = chronos_forecasts(pipes, y, df[ex_cols], o, sensors=True)
                    else:
                        fc = forecasts_for_fold(y, X_ex, seg, o, N, fold)
                        if pipes is not None:
                            fc.update(chronos_forecasts(pipes, y, df[ex_cols], o))
                    for k, v in fc.items():
                        store[f"{k}|{fold}"] = np.asarray(v, dtype=np.float64)
                np.savez_compressed(target_file, **store)
                print(f"{run}: {time.time() - t0:.1f}s", flush=True)
    print("[ok] training-free forecasts written")


if __name__ == "__main__":
    main()
