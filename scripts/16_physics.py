#!/usr/bin/env python3
"""Physical organization of the events: longitude, CME speed, and fluence by time of proton maximum, and by solar cycle."""
from paths import *
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(RES / "xrs_features.csv", parse_dates=["t_peak"]); F = pd.read_csv(RES / "forecasts_loyo.csv")
ok = q.x_int.notna().to_numpy(); y = F.y.to_numpy()
rise = ((pd.to_datetime(e.tmax, utc=True, format="mixed") - q.t_peak).dt.total_seconds() / 3600).to_numpy()
M, rows = {}, []
fm = lambda v, d=2: f"{v:.{d}f}".replace("-", "$-$")
lon = lambda v: f"W{v:.0f}" if v >= 0 else f"E{-v:.0f}"
for k, lab, m in (("Early", "Within 12 h", rise <= 12), ("Mid", "12--24 h", (rise > 12) & (rise <= 24)), ("Late", "After 24 h", rise > 24)):
    m = m & ok; d = e[m]
    M[f"{k}Lon"] = lon(d.lon_w.median()); M[f"{k}East"] = f"{(d.lon_w < 0).mean() * 100:.0f}"; M[f"{k}Big"] = str(int((d.peak_pfu >= 1000).sum())); M[f"{k}Cme"] = f"{d.cme_v.median():.0f}"; M[f"{k}X"] = f"{q.x_int[m].median():.2f}"
    rows.append(f"{lab} & {m.sum()} & {lon(d.lon_w.median())} & {(d.lon_w < 0).mean() * 100:.0f} & {q.x_int[m].median():.2f} & {d.cme_v.median():.0f} & {d.peak_pfu.median():.0f} & {int((d.peak_pfu >= 1000).sum())} & {fm((F.balch - y)[m].mean())} & {fm((F['vqr:G3x'] - y)[m].mean())} \\\\")
r = spearmanr(e.lon_w[ok], rise[ok]); M["RhoRiseLon"] = fm(r[0]); M["PRiseLon"] = f"{r[1]:.4f}"
for c in (23, 24, 25):
    m = ok & (e.cycle == c).to_numpy(); M[f"Pfu{'ABC'[c - 23]}"] = f"{e.peak_pfu[m].median():.0f}"; M[f"Cme{'ABC'[c - 23]}"] = f"{e.cme_v[m].median():.0f}"; M[f"Xint{'ABC'[c - 23]}"] = f"{q.x_int[m].median():.2f}"
json.dump({"macros": M, "table": "\n".join(rows) + "\n"}, open(RES / "physics.json", "w"), indent=1); print(M)
