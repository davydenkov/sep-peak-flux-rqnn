#!/usr/bin/env python3
"""RQNN on the same X-ray quantities as Balch (2008) and Garcia (2004): measured integrated flux instead of the triangular proxy, and the Garcia inputs."""
from paths import *
import json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd
D = RES
NSEED = 3
GR = {"Q_balch": ["log_flu", "connect"], "Q_garcia": ["logEM", "log_flux", "dur25"], "Q_both": ["connect", "log_flu", "log_flux", "logEM", "dur25"]}
MODELS = ("rqnn", "ridge", "rf")


def data(A):
    d = A.load(); q = pd.read_csv(D / "xrs_features.csv")
    d["log_flu"] = np.log10(q.x_int.to_numpy()); d["log_flux"] = np.log10(q.x_max.to_numpy()); d["logEM"] = q.logEM_pk.to_numpy(); d["dur25"] = np.log10(q.dur25_h.to_numpy())
    d["y"] = np.log10(d.peak_pfu)
    return d[d.log_flu.notna()]


def job(a):
    import adv_lib as A
    A.STAGE.update({"logEM": 1, "dur25": 1})
    base = A.E.entangling
    def ent(feats):
        idx = {f: i for i, f in enumerate(feats)}
        return base(feats) + [(idx[u], idx[v]) for u, v in (("log_flux", "logEM"), ("logEM", "dur25")) if u in idx and v in idx]
    A.E.entangling = ent
    scheme, fold, g, m = a
    feats = GR.get(g) or A.GROUPS[g.replace("x", "", 1) if False else {"G0x": "G0_flare", "G3x": "G3_full"}[g]]
    d = data(A)
    if scheme == "loyo":
        tr, te = d[d.year != fold], d[d.year == fold]
    else:
        c = pd.Timestamp("2020-01-01"); tr, te = d[d.begin < c], d[d.begin >= c]
    if not len(te): return None
    y = tr.y.to_numpy()
    ms = [A.make(m, feats, s).fit(tr, y) for s in range(NSEED if m == "rqnn" else 1)]
    p = np.mean([x.predict(te) for x in ms], axis=0); pt = np.mean([x.predict(tr) for x in ms], axis=0)
    return {"scheme": scheme, "fold": str(fold), "g": g, "m": m, "idx": [int(i) for i in te.index], "p": [float(v) for v in p], "train_mae": float(np.abs(pt - y).mean()), "ntr": len(tr)}


def main():
    e = pd.read_csv(INP / "events_final.csv"); years = sorted(int(v) for v in e.year.unique())
    folds = [("chrono", 0)] + [("loyo", y) for y in years]
    jobs = [(s, f, g, m) for s, f in folds for g in ("G3x", "G0x", "Q_both", "Q_garcia", "Q_balch") for m in MODELS]
    jobs.sort(key=lambda j: (j[3] != "rqnn", j[2] != "G3x"))
    print(len(jobs), flush=True); out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            if r: out.append(r)
            if i % 20 == 0: print(i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(D / "rqnn_gb_predictions.json", "w")); print("done", flush=True)


if __name__ == "__main__":
    main()
