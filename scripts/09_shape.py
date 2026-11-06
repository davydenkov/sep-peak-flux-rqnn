#!/usr/bin/env python3
"""What does the two-input network learn? LOYO low-order classical laws in fluence and connection, and the response of the network."""
from paths import *
from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
D = RES
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv"); F = pd.read_csv(D / "forecasts_loyo.csv"); S = pd.read_csv(D / "forecasts_same_inputs.csv")
y = F.y.to_numpy(); ok = q.x_int.notna().to_numpy(); x = np.log10(q.x_int.to_numpy()); c = e.connect.to_numpy(); yr = e.year.to_numpy()
def loyo(cols):
    p = np.full(len(y), np.nan)
    for v in np.unique(yr):
        te = (yr == v) & ok; tr = (yr != v) & ok
        if not te.any(): continue
        A = np.column_stack([np.ones(tr.sum())] + [k[tr] for k in cols]); b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        p[te] = np.column_stack([np.ones(te.sum())] + [k[te] for k in cols]) @ b
    return p
xc = np.where(ok, x, 0)
M = {"x": [xc], "x,c": [xc, c], "x,x2": [xc, xc**2], "x,x2,c": [xc, xc**2, c], "x,c,xc": [xc, c, xc * c], "x,x2,c,c2": [xc, xc**2, c, c**2], "x,x2,c,c2,xc": [xc, xc**2, c, c**2, xc * c]}
rq = S["vqr:Q_balch"].to_numpy(); a = np.abs(rq[ok] - y[ok]); P = {}
for k, v in M.items():
    p = loyo(v); P[k] = p; b = np.abs(p[ok] - y[ok]); d = a - b; rng = np.random.default_rng(1); s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    print(f"{k:16s} MAE {b.mean():.3f}  r {np.corrcoef(p[ok], y[ok])[0,1]:.2f}  rqnn-this {d.mean():+.3f} [{np.percentile(s,2.5):+.3f},{np.percentile(s,97.5):+.3f}]")
print("rqnn Q_balch MAE", a.mean())
# binned residual of the refitted law vs fluence and connection
res = y - F.balch_fit.to_numpy()
qs = pd.qcut(x[ok], 5); print(pd.Series(res[ok]).groupby(qs, observed=True).agg(["mean", "count"]).round(2).to_string())
qs = pd.cut(c[ok], [-1.01, 0, .5, .8, .95, 1.01]); print(pd.Series(res[ok]).groupby(qs, observed=True).agg(["mean", "count"]).round(2).to_string())
print("rqnn resp vs connect (rho at fixed x):", spearmanr(c[ok], rq[ok] - np.polyval(np.polyfit(x[ok], rq[ok], 1), x[ok])))
print("rf vs rqnn Q_balch corr", np.corrcoef(S["rf:Q_balch"][ok], rq[ok])[0, 1], "MAE rf", np.abs(S["rf:Q_balch"][ok] - y[ok]).mean())
