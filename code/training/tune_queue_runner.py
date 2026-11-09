# RELEASE NOTE: the queue runner used for the hyperparameter check, with account
# names and dataset slugs removed (they are now required arguments). As run, the
# 18 jobs were split over the first author's account and two co-authors' accounts,
# each running its own jobs on its own private, md5-identical copy of the data
# (Kaggle GPU Tesla T4, torch 2.11.0+cu130). Which job ran where: AUDIT_REPORT.md.
"""Hyperparameter check on Kaggle: per account, at most 2 concurrent GPU sessions,
quota-aware, resumable (re-running skips finished jobs and adopts running ones).

Jobs: 9 canonical datasets x 2 targets (self_x_ac = the original vibration
headline channel, self_temperature_one = the original temperature headline
target) = 18 kernels. Each kernel runs tune_forecasters.py (built by build_tune_kernel.py): TiDE, TimeXer, NHITS x
8 configs (config 0 = original) x 5 folds.

Results are pulled to --out (default ./tuning_raw)/<DS>/<channel>/.
Usage: python tune_queue_runner.py --owner=USER --token=PATH --dataset=USER/SLUG [--iterations=1,2,3] [--only=slug,...]
       add --dry-run to print the plan only
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT_ROOT = Path(ARGS.get("out", TRAIN / "tuning_raw"))   # then: import_tuning_runs.py (ORIGINAL_ROOT layout)
# Each account is run by its own holder (one account per person, Kaggle rules):
#   --owner=<kaggle user>  --token=<path to that user's access_token>  --iterations=1,2
OWNER = ARGS["owner"]                       # Kaggle user name of the person running this
TOKEN = Path(ARGS["token"]).read_text().strip()   # that user's own API token file
# each account attaches its own private copy; copies verified byte-identical (md5) before use
DATASET = ARGS["dataset"]                   # private Kaggle copy of data/ (the 9 CSV files)
MAX_ACTIVE = 2
POLL_S = 120
EST_H = 2.0                      # conservative per-kernel estimate (3 models x 8 cfgs x 5 folds)
MARGIN_H = 0.5
DRY = "--dry-run" in sys.argv
ITERS = [int(x) for x in ARGS.get("iterations", "1,2,3").split(",")]
JOBS = [(f"R{r}_I{i}", ch) for i in ITERS for r in (1, 2, 3)
        for ch in ("self_x_ac", "self_temperature_one")]
ENV = {**os.environ, "KAGGLE_API_TOKEN": TOKEN, "PYTHONUTF8": "1"}


def slug(ds, ch):
    return "tune-" + ds.lower().replace("_", "-") + "-" + \
        ("vibx" if ch == "self_x_ac" else "temp1")


def out_dir(ds, ch):
    return OUT_ROOT / ds / ch


def done(ds, ch):
    return (out_dir(ds, ch) / "tuning_configs.json").exists()


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=ENV)
    return (r.stdout + r.stderr).strip(), r.returncode


def quota_h():
    out, _ = kaggle("quota")
    for line in out.splitlines():
        if line.strip().startswith("GPU"):
            m = re.findall(r"(\d+\.\d+)h", line)
            if len(m) >= 2:
                return float(m[1])
    return None


def push(ds, ch):
    s = slug(ds, ch)
    stage = TRAIN / f".stage_{s}"
    stage.mkdir(exist_ok=True)
    header = ("# ---- injected by tune_queue_runner.py ----\nimport os\n"
              f"os.environ['SMOKE'] = '0'\nos.environ['DS'] = '{ds}'\n"
              f"os.environ['TARGET_CH'] = '{ch}'\nos.environ['PROTOCOL'] = 'reduced'\n"
              "os.environ['CANONICAL'] = '1'\nos.environ['TUNE_MODELS'] = 'TiDE,TimeXer,NHITS'\n"
              "# -------------------------------------------\n\n")
    (stage / "tune_forecasters.py").write_text(header + (TRAIN / "tune_forecasters.py").read_text(encoding="utf-8"),
                                          encoding="utf-8")
    meta = dict(id=f"{OWNER}/{s}", title=s, code_file="tune_forecasters.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False,
                enable_internet=True, competition_sources=[], dataset_sources=[DATASET],
                kernel_sources=[], model_sources=[])
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out, _ = kaggle("kernels", "push", "-p", str(stage))
    ok = "successfully pushed" in out.lower()
    print(f"[{'PUSH' if ok else 'PUSH-REJECTED'}] {s}: {out[-200:] if not ok else ''}", flush=True)
    return ok


def pull(ds, ch):
    s = slug(ds, ch)
    tmp = TRAIN / f".pull_{s}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    out, rc = kaggle("kernels", "output", f"{OWNER}/{s}", "-p", str(tmp))
    d = out_dir(ds, ch)
    d.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        shutil.copy2(f, d / f.name)
    shutil.rmtree(tmp)
    ok = (d / "tuning_configs.json").exists() and (d / "tuning_predictions.npz").exists()
    print(f"[{'DONE' if ok else 'PULL-INCOMPLETE'}] {s} -> {d}", flush=True)
    return ok


def main():
    only = set(ARGS["only"].split(",")) if "only" in ARGS else None      # e.g. --only=tune-r2-i2-vibx,...
    jobs = [j for j in JOBS if only is None or slug(*j) in only]
    pending = [j for j in jobs if not done(*j)]
    print(f"[PLAN] account={OWNER} dataset={DATASET} jobs={len(jobs)} pending={len(pending)} "
          f"max_active={MAX_ACTIVE} est={EST_H}h/job", flush=True)
    for ds, ch in pending:
        print(f"   {slug(ds, ch)}  ({ds}, {ch})")
    print(f"[QUOTA] GPU hours remaining this week: {quota_h()}", flush=True)
    if DRY:
        return
    active, failed = [], []
    for ds, ch in list(pending):                   # resume: adopt jobs already on Kaggle
        st, _ = kaggle("kernels", "status", f"{OWNER}/{slug(ds, ch)}")
        su = st.upper()
        if "RUNNING" in su or "QUEUED" in su:
            pending.remove((ds, ch))
            active.append((ds, ch))
            print(f"[ADOPT] {slug(ds, ch)} already running on Kaggle", flush=True)
        elif "COMPLETE" in su:
            pending.remove((ds, ch))
            if not pull(ds, ch):
                failed.append((ds, ch))
    while active or pending:
        still = []
        for ds, ch in active:
            st, _ = kaggle("kernels", "status", f"{OWNER}/{slug(ds, ch)}")
            su = st.upper()
            if "COMPLETE" in su:
                if not pull(ds, ch):
                    failed.append((ds, ch))
            elif "ERROR" in su or "CANCEL" in su:
                print(f"[FAILED] {slug(ds, ch)}: {st}", flush=True)
                pull(ds, ch)                       # keep partial output for inspection
                failed.append((ds, ch))
            else:
                still.append((ds, ch))
        active = still
        while len(active) < MAX_ACTIVE and pending:
            q = quota_h()
            if q is None or q < EST_H + MARGIN_H:
                print(f"[QUOTA-PAUSE] remaining={q}; waiting for weekly reset. Re-run later; "
                      "finished jobs are skipped.", flush=True)
                pending = [] if not active else pending
                break
            ds, ch = pending.pop(0)
            if push(ds, ch):
                active.append((ds, ch))
            else:
                pending.insert(0, (ds, ch))
                break
        print(f"[STATUS] active={len(active)} pending={len(pending)} failed={len(failed)}", flush=True)
        if active:
            time.sleep(POLL_S)
        elif pending:
            break
    print(f"[END] failed={failed}", flush=True)


if __name__ == "__main__":
    main()
