

# ============================================================
# HYPERPARAMETER CHECK (appended; everything above is verbatim
# from train_forecasters.py so folds, origins and preprocessing
# are identical to the stored runs)
# ============================================================
# For each model, config 0 is the ORIGINAL setting; configs 1..7 are drawn
# without replacement from a fixed grid with a fixed seed (decided before any
# run). Every config is trained on each fold exactly as in the original study,
# then predicts (a) the fold's VALIDATION block and (b) the fold's TEST origins.
# Selection happens OFFLINE and uses ONLY the validation predictions.
import itertools  # noqa: E402
import random  # noqa: E402

if os.environ.get("TUNE_OUT"):                     # keep tuning output away from the stored runs
    OUT = Path(os.environ["TUNE_OUT"])
    OUT.mkdir(parents=True, exist_ok=True)
TUNE_MODELS = [m.strip() for m in os.environ.get("TUNE_MODELS", "TiDE,TimeXer,NHITS").split(",") if m.strip()]
N_CONFIGS = 2 if SMOKE else int(os.environ.get("N_CONFIGS", "8"))

ORIGINAL = {
    "TiDE": dict(hidden_size=256, dropout=0.3, learning_rate=1e-3, num_encoder_layers=1),
    "NHITS": dict(width=512, dropout_prob_theta=0.0, learning_rate=1e-3),
    "TimeXer": dict(hidden_size=128, dropout=0.1, learning_rate=1e-3),
}
GRID = {
    "TiDE": dict(hidden_size=[128, 256, 512], dropout=[0.1, 0.3], learning_rate=[3e-4, 1e-3, 3e-3],
                 num_encoder_layers=[1, 2]),
    "NHITS": dict(width=[256, 512, 1024], dropout_prob_theta=[0.0, 0.2], learning_rate=[3e-4, 1e-3, 3e-3]),
    "TimeXer": dict(hidden_size=[64, 128, 256], dropout=[0.1, 0.3], learning_rate=[3e-4, 1e-3, 3e-3]),
}


def configs_for(model):
    keys = list(GRID[model])
    allc = [dict(zip(keys, v)) for v in itertools.product(*[GRID[model][k] for k in keys])]
    allc = [c for c in allc if c != ORIGINAL[model]]
    random.Random(2026).shuffle(allc)
    return [ORIGINAL[model]] + allc[:N_CONFIGS - 1]


def build_tuned(model, cfg):
    from neuralforecast.models import NHITS, TiDE, TimeXer
    base = dict(h=HORIZON, input_size=INPUT_SIZE, max_steps=MAX_STEPS, learning_rate=cfg["learning_rate"],
                batch_size=BATCH_SIZE, val_check_steps=VAL_CHECK_STEPS,
                early_stop_patience_steps=EARLY_STOP_PATIENCE, scaler_type="robust", random_seed=RANDOM_SEED,
                enable_progress_bar=False, logger=False, enable_model_summary=False,
                accelerator=ACCELERATOR, devices=1, gradient_clip_val=1.0,
                windows_batch_size=WINDOWS_BATCH_SIZE, hist_exog_list=EXOG)
    if model == "TiDE":
        return TiDE(**base, hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
                    num_encoder_layers=cfg["num_encoder_layers"], alias="TiDE")
    if model == "NHITS":
        w = cfg["width"]
        return NHITS(**base, mlp_units=[[w, w]] * 3, dropout_prob_theta=cfg["dropout_prob_theta"], alias="NHITS")
    if model == "TimeXer":
        return TimeXer(**base, n_series=1, hidden_size=cfg["hidden_size"], n_heads=4, e_layers=2,
                       patch_len=16, dropout=cfg["dropout"], alias="TimeXer")
    raise ValueError(model)


def predict(nf, model, src, origins, y_mu, y_sd):
    if model in MULTIVARIATE:
        rows = []
        for o in origins:
            w = src.iloc[o - INPUT_SIZE:o].copy()
            w["unique_id"] = DS_NAME
            w["ds"] = BASE_TS + pd.to_timedelta(np.arange(INPUT_SIZE), unit="m")
            p = nf.predict(df=w, verbose=False)
            p = p.reset_index() if "unique_id" not in p.columns else p
            rows.append(p[model].to_numpy(dtype=np.float64)[:HORIZON])
        return np.stack(rows) * y_sd + y_mu
    p = nf.predict(df=history_batch(src, origins), verbose=False)
    p = p.reset_index() if "unique_id" not in p.columns else p
    return collect(p, model, len(origins)) * y_sd + y_mu


def tune_main():
    from neuralforecast import NeuralForecast
    store, log = {}, []
    t_start = time.time()
    cfgs = {m: configs_for(m) for m in TUNE_MODELS}
    for split_name, (tr_pct, va_pct, te_pct) in SPLITS.items():
        train_end, val_end, test_end = split_bounds(tr_pct, va_pct, te_pct)
        o_te = make_origins(val_end, test_end)
        o_va = make_origins(train_end, val_end)       # same rule, applied to the validation block
        store[f"origins_test|{split_name}"] = o_te.astype(np.int32)
        store[f"origins_val|{split_name}"] = o_va.astype(np.int32)
        store[f"ytrue_test|{split_name}"] = truth_matrix(o_te).astype(np.float32)
        store[f"ytrue_val|{split_name}"] = truth_matrix(o_va).astype(np.float32)
        dfs, y_mu, y_sd = standardize(train_end)
        train_val = dfs.iloc[:val_end]
        for model in TUNE_MODELS:
            for k, cfg in enumerate(cfgs[model]):
                if time.time() - t_start > TIME_BUDGET_S:
                    print("[!] time budget reached", flush=True)
                    break
                t0 = time.time()
                try:
                    nf = NeuralForecast(models=[build_tuned(model, cfg)], freq="1min")
                    nf.fit(df=train_val, val_size=val_end - train_end, verbose=False)
                    store[f"{model}|cfg{k}|{split_name}|val"] = predict(nf, model, dfs, o_va, y_mu, y_sd).astype(np.float32)
                    store[f"{model}|cfg{k}|{split_name}|test"] = predict(nf, model, dfs, o_te, y_mu, y_sd).astype(np.float32)
                    status = "ok"
                except Exception as exc:  # noqa: BLE001
                    status = f"FAILED: {exc}"
                    traceback.print_exc()
                log.append(dict(model=model, cfg=k, fold=split_name, seconds=round(time.time() - t0, 1),
                                status=status, **cfg))
                print(f"{split_name} {model} cfg{k} {cfg} {status} {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT / "tuning_predictions.npz", **store)
                pd.DataFrame(log).to_csv(OUT / "tuning_log.csv", index=False)
    with open(OUT / "tuning_configs.json", "w") as f:
        json.dump(dict(dataset=DS_NAME, target_channel=TARGET_CHANNEL, configs=cfgs, seed=RANDOM_SEED,
                       input_size=INPUT_SIZE, horizon=HORIZON, max_steps=MAX_STEPS, n_windows=N_WINDOWS,
                       splits=SPLITS, exog=EXOG, dropped_as_leakage=sorted(_leak)), f, indent=2)
    print(f"[DONE] {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    tune_main()
