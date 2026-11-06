#!/usr/bin/env python3
"""RQNN forecast of the peak >10 MeV flux on the clean sample. Target: log10(NOAA peak pfu). Same inputs, issue time and
leave-one-year-out protocol as the onset study; also a holdout of cycle 25 and a fluence-only baseline in the spirit of Garcia/Balch."""
from paths import *
import json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd
GROUPS = ("G0_flare", "G2_flare_cme", "G3_full", "G7_flare_cme_proton", "G8_all_precursors")
MODELS = ("ridge", "gb", "rf", "mlp", "rqnn")
NSEED = 3


def job(a):
    import adv_lib as A
    scheme, fold, g, m = a
    d = A.load(); d["y"] = np.log10(d.peak_pfu)
    if scheme == "loyo":
        tr, te = d[d.year != fold], d[d.year == fold]
    else:
        c = pd.Timestamp("2020-01-01"); tr, te = d[d.begin < c], d[d.begin >= c]
    y = tr.y.to_numpy()
    if m == "clim":
        p, pt = np.full(len(te), np.median(y)), np.full(len(tr), np.median(y))
    elif m == "fluence":
        k = np.polyfit(tr.log_flu, y, 1); p, pt = np.polyval(k, te.log_flu), np.polyval(k, tr.log_flu)
    else:
        feats = A.GROUPS[g]
        ms = [A.make(m, feats, s).fit(tr, y) for s in range(NSEED if m in ("rqnn", "rqnn_noent", "mlp") else 1)]
        p = np.mean([x.predict(te) for x in ms], axis=0); pt = np.mean([x.predict(tr) for x in ms], axis=0)
    return {"scheme": scheme, "fold": str(fold), "g": g, "m": m, "idx": [int(i) for i in te.index], "p": [float(v) for v in p], "train_mae": float(np.abs(pt - y).mean()), "ntr": len(y)}


def main():
    d = pd.read_csv(INP / "events_final.csv"); years = sorted(int(v) for v in d.year.unique())
    folds = [("chrono", 0)] + [("loyo", y) for y in years]
    jobs = [(s, f, "none", m) for s, f in folds for m in ("clim", "fluence")]
    jobs += [(s, f, g, m) for s, f in folds for g in GROUPS for m in MODELS]
    jobs += [(s, f, "G3_full", "rqnn_noent") for s, f in folds]
    jobs.sort(key=lambda j: (j[3] not in ("rqnn", "rqnn_noent"), j[2] not in ("G3_full", "G8_all_precursors")))
    print(len(jobs), flush=True)
    out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            out.append(r)
            if i % 25 == 0:
                print(i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(RES / "peak_predictions.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
