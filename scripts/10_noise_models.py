#!/usr/bin/env python3
"""Robustness of the Garcia-Balch forms and the RQNN to errors in the inputs, under leave-one-year-out.
Same noise procedure as the onset study; all models share the same noise draws per input (common random numbers)."""
from paths import *
import json, multiprocessing as mp, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
D = RES
LEVELS = (.1, .2, .3, .5); NREP = 20; NSEED = 3
XR = ["log_flu", "log_flux", "logEM", "dur25", "log_dur", "log_rise"]
SCALE = float(np.log10(1 / .7))
Q = {"Q_balch": ["log_flu", "connect"], "Q_garcia": ["logEM", "log_flux", "dur25"]}
JOBS = [("balch_pub", "-"), ("fluence_fit", "-"), ("lon_fit", "-"), ("garcia_fit", "-"), ("ridge", "G3x"), ("rf", "G3x"), ("ridge", "Q_balch"), ("rqnn", "Q_balch"), ("rqnn", "G0x"), ("rqnn", "G3x"), ("rqnn_noent", "G3x")]


def data(A):
    d = A.load(); q = pd.read_csv(D / "xrs_features.csv"); pr = pd.read_csv(D / "prior_flare.csv")
    d["tri"] = d.log_flu
    d["log_flu"] = np.log10(q.x_int.to_numpy()); d["log_flux"] = np.log10(q.x_max.to_numpy()); d["logEM"] = q.logEM_pk.to_numpy(); d["dur25"] = np.log10(q.dur25_h.to_numpy()); d["dur_h"] = q.dur25_h.to_numpy()
    x = pr.x_pf.to_numpy(); a = np.where(x > .08, (x / .167) ** 1.146, 1.0); d["log_alpha"] = np.log10(np.where(np.isnan(a), 1.0, a))
    d["lonfac"] = np.exp(-((d.lon_w.to_numpy() - 52.5) ** 2) / (2 * 12.0**2))
    d["y"] = np.log10(d.peak_pfu)
    return d[d.log_flu.notna()]


class Lin:
    """Least-squares law in the given columns; a missing fluence falls back on the triangular catalogue proxy, as an operator would."""
    def __init__(self, cols): self.c = cols
    def _x(self, df):
        z = df[self.c].copy()
        if "log_flu" in z: z["log_flu"] = z.log_flu.fillna(df.tri)
        return z.to_numpy()
    def fit(self, tr, y):
        self.b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), self._x(tr)]), y, rcond=None)[0]; return self
    def predict(self, df): return np.column_stack([np.ones(len(df)), self._x(df)]) @ self.b


class Pub:
    def fit(self, tr, y): return self
    def predict(self, df): return 1 + df.log_alpha.to_numpy() + .82 * (df.log_flu.fillna(df.tri).to_numpy() - np.log10(.00987))


def job(a):
    import adv_lib as A
    A.STAGE.update({"logEM": 1, "dur25": 1})
    base = A.E.entangling
    def ent(feats):
        idx = {f: i for i, f in enumerate(feats)}
        return base(feats) + [(idx[u], idx[v]) for u, v in (("log_flux", "logEM"), ("logEM", "dur25")) if u in idx and v in idx]
    A.E.entangling = ent
    yr, m, g = a
    d = data(A); tr, te = d[d.year != yr], d[d.year == yr]
    if not len(te): return None
    y = tr.y.to_numpy()
    if m == "balch_pub": ms, feats = [Pub()], ["log_flu"]
    elif m == "fluence_fit": ms, feats = [Lin(["log_flu"]).fit(tr, y)], ["log_flu"]
    elif m == "lon_fit": ms, feats = [Lin(["log_flu", "lonfac"]).fit(tr, y)], ["log_flu"]
    elif m == "garcia_fit": ms, feats = [Lin(["logEM", "log_flux", "dur_h"]).fit(tr, y)], ["logEM", "log_flux", "dur25"]
    else:
        feats = Q.get(g) or A.GROUPS[{"G0x": "G0_flare", "G3x": "G3_full"}[g]]
        ms = [A.make(m, feats, s).fit(tr, y) for s in range(NSEED if m.startswith("rqnn") else 1)]
    cont = [f for f in feats if f != "prior_sep_72h"]
    allc = ["connect", "log_flu", "log_flux", "log_rise", "log_dur", "logEM", "dur25", "cme_v", "cme_w", "DE4_rise", "imf_align_mean", "imf_stability", "imf_theta_abs"]
    sd = tr[allc].std()
    Z = {c: np.random.default_rng([yr, zlib.crc32(c.encode())]).normal(size=(NREP, len(te))) for c in allc}
    cases, frames = [("clean", 0, 0)], [te]
    for sc, cols in (("all", cont), ("xray", [c for c in cont if c in XR])):
        if not cols: continue
        for L in LEVELS:
            for k in range(NREP):
                q = te.copy()
                for c in cols:
                    q[c] = q[c].to_numpy() + L * sd[c] * Z[c][k]
                q["dur_h"] = 10 ** q.dur25
                cases.append((sc, L, k)); frames.append(q)
    q = te.copy(); q["log_flu"] = np.nan; cases.append(("missing", 0, 0)); frames.append(q)
    q = te.copy(); q["log_flu"] += SCALE; q["log_flux"] += SCALE; q["logEM"] += SCALE; cases.append(("scale", 0, 0)); frames.append(q)
    big = pd.concat(frames, ignore_index=True)
    p = np.mean([x.predict(big) for x in ms], axis=0).reshape(len(cases), len(te))
    return {"yr": int(yr), "m": m, "g": g, "idx": [int(i) for i in te.index], "cases": cases, "p": p.round(5).tolist()}


def main():
    e = pd.read_csv(INP / "events_final.csv"); years = sorted(int(v) for v in e.year.unique())
    jobs = [(y, m, g) for y in years for m, g in JOBS]
    jobs.sort(key=lambda j: (-(j[2] == "G3x" and j[1].startswith("rqnn")), -(j[1] == "rqnn")))
    print(len(jobs), flush=True); out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            if r: out.append(r)
            if i % 20 == 0: print(i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(D / "noise_predictions.json", "w")); print("done", flush=True)


if __name__ == "__main__":
    main()
