#!/usr/bin/env python3
"""Increase of the error under input errors, paired by event, for the Garcia-Balch forms and the RQNN."""
from paths import *
import json, zlib
from pathlib import Path
import numpy as np, pandas as pd
D = RES
J = json.load(open(D / "noise_predictions.json"))
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv"); y = np.log10(e.peak_pfu.to_numpy()); n = len(y)
LV = (.1, .2, .3, .5); NREP = 20; SC = float(np.log10(1 / .7))
# per model: dict scenario -> array [case, event]
P = {}
for r in J:
    k = f"{r['m']}:{r['g']}"; p = np.array(r["p"]); t = P.setdefault(k, {})
    for c, row in zip(r["cases"], p):
        key = (c[0], c[1]) if c[0] in ("all", "xray") else (c[0],)
        t.setdefault(key, {}).setdefault(c[2], np.full(n, np.nan))[r["idx"]] = row
# parabolic law through the same noise draws
ok = q.x_int.notna().to_numpy(); lx = np.log10(q.x_int.to_numpy()); tri = e.log_flu.to_numpy(); yrs = e.year.to_numpy()
Pq = {}
for yr in np.unique(yrs):
    te = (yrs == yr) & ok; tr = (yrs != yr) & ok
    if not te.any(): continue
    b = np.polyfit(lx[tr], y[tr], 2); sd = pd.Series(lx[tr]).std()
    Z = np.random.default_rng([int(yr), zlib.crc32(b"log_flu")]).normal(size=(NREP, te.sum()))
    Pq.setdefault(("clean",), {}).setdefault(0, np.full(n, np.nan))[te] = np.polyval(b, lx[te])
    for s in ("all", "xray"):
        for L in LV:
            for k in range(NREP):
                Pq.setdefault((s, L), {}).setdefault(k, np.full(n, np.nan))[te] = np.polyval(b, lx[te] + L * sd * Z[k])
    Pq.setdefault(("missing",), {}).setdefault(0, np.full(n, np.nan))[te] = np.polyval(b, tri[te])
    Pq.setdefault(("scale",), {}).setdefault(0, np.full(n, np.nan))[te] = np.polyval(b, lx[te] + SC)
P["quad:-"] = Pq
def err(k, key):
    """Mean absolute error per event, averaged over noise draws."""
    t = P[k][key]; return np.mean([np.abs(v - y) for v in t.values()], axis=0)
def shift(k, key):
    c = P[k][("clean",)][0]; return np.nanmean(np.mean([np.abs(v - c) for v in P[k][key].values()], axis=0)[ok])
for k, t in P.items():
    if ("xray", .3) not in t: print("no xray", k)
M = ["balch_pub:-", "fluence_fit:-", "quad:-", "lon_fit:-", "garcia_fit:-", "ridge:Q_balch", "vqr:Q_balch", "ridge:G3x", "rf:G3x", "vqr:G0x", "vqr:G3x", "vqr_noent:G3x"]
rows = []
for k in M:
    c = err(k, ("clean",))[ok]; r = {"model": k, "clean": c.mean()}
    for s in ("all", "xray"):
        for L in LV:
            if (s, L) in P[k]:
                r[f"d_{s}_{L}"] = err(k, (s, L))[ok].mean() - c.mean(); r[f"sh_{s}_{L}"] = shift(k, (s, L))
    r["missing"] = err(k, ("missing",))[ok].mean(); r["scale"] = err(k, ("scale",))[ok].mean()
    rows.append(r)
T = pd.DataFrame(rows).set_index("model"); T.to_csv(D / "noise_scores.csv"); pd.set_option("display.width", 250, "display.max_columns", 40)
print(T.round(3).T.to_string())
def pair(a, b, key, deg=True):
    da = err(a, key)[ok] - (err(a, ("clean",))[ok] if deg else 0); db = err(b, key)[ok] - (err(b, ("clean",))[ok] if deg else 0)
    d = da - db; rng = np.random.default_rng(21); s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)
R = []
for a in ("vqr:G3x", "vqr:G0x", "vqr:Q_balch"):
    for b in ("balch_pub:-", "fluence_fit:-", "quad:-", "rf:G3x", "ridge:G3x"):
        for key in (("all", .3), ("all", .5), ("xray", .3), ("xray", .5)):
            for deg in (True, False):
                R.append(dict(a=a, b=b, key=str(key), kind="degradation" if deg else "mae", **dict(zip(("d", "lo", "hi"), pair(a, b, key, deg)))))
        for key in (("missing",), ("scale",)):
            R.append(dict(a=a, b=b, key=str(key), kind="mae", **dict(zip(("d", "lo", "hi"), pair(a, b, key, False)))))
for key in (("all", .3), ("all", .5), ("xray", .3), ("xray", .5)):
    R.append(dict(a="vqr:G3x", b="vqr_noent:G3x", key=str(key), kind="degradation", **dict(zip(("d", "lo", "hi"), pair("vqr:G3x", "vqr_noent:G3x", key)))))
    d = np.mean([np.abs(v - P["vqr:G3x"][("clean",)][0]) for v in P["vqr:G3x"][key].values()], axis=0)[ok] - np.mean([np.abs(v - P["vqr_noent:G3x"][("clean",)][0]) for v in P["vqr_noent:G3x"][key].values()], axis=0)[ok]
    rng = np.random.default_rng(22); s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    R.append(dict(a="vqr:G3x", b="vqr_noent:G3x", key=str(key), kind="shift", d=d.mean(), lo=np.percentile(s, 2.5), hi=np.percentile(s, 97.5)))
R = pd.DataFrame(R); R.to_csv(D / "noise_paired.csv", index=False)
sig = R[(R.lo > 0) | (R.hi < 0)]
print(R[R.b.isin(["balch_pub:-", "fluence_fit:-", "quad:-"]) & R.a.isin(["vqr:G3x", "vqr:Q_balch"])].round(3).to_string(index=False))
print("significant:"); print(sig.round(3).to_string(index=False))
print(R[R.b == "vqr_noent:G3x"].round(3).to_string(index=False))
# check that the clean run reproduces the main forecasts
F = pd.read_csv(D / "forecasts_same_inputs.csv")
print("reproduce G3x", np.nanmax(np.abs(P["vqr:G3x"][("clean",)][0] - F["vqr:G3x"].to_numpy())))
