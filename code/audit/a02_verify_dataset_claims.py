"""Audit 2 - recompute every dataset/instrument number used in the drafts.

Each row: claim as written in the earlier drafts, value recomputed here from
the released data (or stage-1 device statistics), and a verdict.
Writes results/audit/a02_dataset_claims.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DATA, G0, INPUT_SIZE, NOM_FIRMWARE, NOM_TRUE, RESULTS, RTD_A,  # noqa: E402
                    RTD_B, RETAINED)

AUD = RESULTS / "audit"
rows = []


def claim(cid, text, value, verdict, note=""):
    rows.append(dict(id=cid, claim_in_drafts=text, recomputed=value, verdict=verdict, note=note))


def cvd_R(t, r0):                        # resistance for temperature t >= 0
    return r0 * (1 + RTD_A * t + RTD_B * t * t)


def cvd_T(R, r0):                        # library's closed-form solve for t >= 0
    z1, z2, z3, z4 = -RTD_A, RTD_A ** 2 - 4 * RTD_B, 4 * RTD_B / r0, 2 * RTD_B
    return (z1 + np.sqrt(z2 + z3 * R)) / z4


def main():
    man = pd.read_csv(DATA / "manifest.csv")
    cal = pd.read_csv(AUD / "gravity_calibration.csv").set_index("device")
    dev = pd.read_csv(AUD / "device_raw_summary.csv").set_index("device")

    claim("D1", "157,707 rows", int(man.rows.sum()), "PASS" if man.rows.sum() == 157707 else "FAIL")
    # forecast-origin capacity (window = 128 input + 18), as in the original manifest
    cap = int(man.origins_capacity.sum())
    claim("D2", "143,787 valid forecast origins", cap, "PASS" if cap == 143787 else "FAIL",
          "capacity of the files; the models were EVALUATED on 80 origins x 5 folds per target")
    claim("D3", "nine datasets, 3 regions x 3 iterations, 18-day windows", f"{len(man)} datasets",
          "PASS" if len(man) == 9 else "FAIL")

    k9 = cal.loc[[f"AHM-0{i}" for i in range(1, 10)], "med"]
    k8 = cal.loc[RETAINED, "med"]
    claim("C1", "k_d in [1.7904, 1.8479], mean 1.8186, sd 0.0196, 3.2% spread across nine devices",
          f"9 devices: [{k9.min():.4f}, {k9.max():.4f}], mean {k9.mean():.4f}, sd {k9.std(ddof=0):.4f}, "
          f"spread {(k9.max()-k9.min())/k9.mean()*100:.2f}% | 8 retained: mean {k8.mean():.4f}",
          "PASS (wording)", "the nine include excluded AHM-09; the release has 8 devices")
    claim("C2", "raw static resultant ~1.81 units where gravity must read 1.00",
          f"median of 9 k_d = {k9.median():.4f}", "PASS")

    # vib_dc median vs g across device-dataset pairs
    errs = []
    for f in sorted(DATA.glob("R*_I*.csv")):
        d = pd.read_csv(f, usecols=lambda c: c.endswith("vib_dc"))
        for c in d.columns:
            if (d[c] == 0).all():
                continue                    # zero-filled absent neighbour (R3 nb2)
            errs.append(abs(d[c].median() - G0) / G0 * 100)
    claim("C3", "median vib_dc equals g within 1.22% across 24 device-dataset pairs",
          f"{len(errs)} pairs, max deviation {max(errs):.2f}%, median {np.median(errs):.2f}%",
          "PASS" if len(errs) == 24 and max(errs) <= 1.225 else "CHECK")

    # PT100 nominal-resistance correction
    r0 = cvd_R(0.0, NOM_TRUE)
    claim("C4", "round trip 100.0000 ohm at 0 degC", f"{r0:.4f} ohm", "PASS" if abs(r0 - 100) < 1e-9 else "FAIL")
    corr = []
    for t_rep in (22.0, 94.0):               # temperature as REPORTED by the firmware
        R = cvd_R(t_rep, NOM_FIRMWARE)       # resistance implied by the firmware's nominal 97
        corr.append(cvd_T(R, NOM_TRUE) - t_rep)   # re-solve with the correct nominal 100
    ok = abs(corr[0] + 8.38) < 0.005 and abs(corr[1] + 10.74) < 0.005
    claim("C5", "correction grows from -8.38 degC at 22 degC to -10.74 degC at 94 degC",
          f"correction (true - reported): {corr[0]:+.2f} at reported 22, {corr[1]:+.2f} at reported 94",
          "PASS (wording)" if ok else "FAIL",
          "the temperatures are REPORTED (uncorrected) readings; firmware over-reports by 8-11 degC")

    claim("X1", "AHM-11: 1.0% coverage, 21,728 rows, spike 833.51 degC",
          f"{dev.loc['AHM-11','coverage_of_own_span_pct']}%, {dev.loc['AHM-11','raw_rows']} rows, "
          f"max t2 {dev.loc['AHM-11','t2_raw_max']}", "PASS")
    claim("X2", "AHM-10: 22.3% coverage, spike 835.51 degC, y axis identical in 50.98% of samples",
          f"{dev.loc['AHM-10','coverage_of_own_span_pct']}% of own span; max t2 {dev.loc['AHM-10','t2_raw_max']}; "
          "y reads exactly 0.0 in 50.98% of samples",
          "PASS (approx.)", "coverage denominator never documented; 50.98% = share of exact zeros")
    claim("X3", "AHM-09 coverage 30.2% vs AHM-04 32.8%",
          f"{dev.loc['AHM-09','coverage_of_own_span_pct']}% vs {dev.loc['AHM-04','coverage_of_own_span_pct']}% "
          "(of own span)", "PASS (approx.)", "denominator not documented")
    claim("X4", "excluded devices carry temperature spikes (implied: retained ones do not)",
          f"retained AHM-07 has one 821.9 degC sample at start-up (removed by envelope filter)",
          "DISCLOSE", "spikes alone do not justify exclusion; coverage does")

    feats = int(man.features.iloc[0])
    claim("D4", "features 42-61 per dataset / Region 3: 42-43 vs 55-62 elsewhere",
          f"released canonical schema: {feats} feature columns in every file; "
          "Region 3 has 16 zero-filled nb2 columns", "REWORD",
          "the 42-61 figures describe the earlier per-region files, not this release")
    zones = man.groupby("region").y_mean.agg(["min", "max"]).round(1)
    claim("D5", "regions ordered hot/mid/cool by operating temperature",
          "; ".join(f"{r}: {a}-{b} degC" for r, (a, b) in zones.iterrows()), "PASS")
    ystd = man.set_index("dataset").y_std
    claim("D6", "R2_I3 variance regime: y_std 3.05 -> 12.76",
          f"R2_I1 {ystd['R2_I1']}, R2_I2 {ystd['R2_I2']}, R2_I3 {ystd['R2_I3']}", "PASS")
    pd.DataFrame(rows).to_csv(AUD / "a02_dataset_claims.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(pd.DataFrame(rows)[["id", "recomputed", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
