#!/usr/bin/env python3
"""SOHO/EPHIN proton intensities (P8: 7.8-25 MeV, P25: 25-41 MeV, P41: 41-53 MeV) around the NOAA begin time, from the yearly L3I 1-min CDFs.
Needs the local EPHIN cache, so it is kept for provenance only; output: data/input/ephin_early.csv."""
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd, cdflib
D = Path(sys.argv[1]); E = sys.argv[2]; O = sys.argv[3]
files = {int(re.search(r"1min_(\d{4})", f)[1]): f for f in sorted(glob.glob(str(D / "soho_costep-ephin_l3i-1min_*.cdf")))}
cache = {}
def year(y):
    if y not in cache:
        if y not in files: cache[y] = None
        else:
            c = cdflib.CDF(files[y]); t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")), utc=True)
            d = pd.DataFrame(c.varget("P_int").astype(float), index=t, columns=["P4", "P8", "P25", "P41"]); cache[y] = d.where(d > 0).sort_index()
    return cache[y]
e = pd.read_csv(E); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e["tmax"] = pd.to_datetime(e.tmax, utc=True, format="mixed"); e = e.sort_values("begin")
rows = []
for r in e.itertuples():
    t0, t1 = r.begin - pd.Timedelta(hours=3), r.tmax + pd.Timedelta(hours=2)
    ds = [year(y) for y in sorted({t0.year, t1.year})]; rec = {"noaa_id": r.noaa_id}
    if all(x is not None for x in ds):
        d = pd.concat(ds)
        for h in (1, 3, 6):
            w = d[(d.index >= r.begin + pd.Timedelta(minutes=60 * h - 30)) & (d.index <= r.begin + pd.Timedelta(minutes=60 * h + 30))]
            for ch in ("P8", "P25", "P41"):
                rec[f"{ch}_{h}h"] = w[ch].mean() if w[ch].notna().sum() >= 10 else np.nan
        w = d[(d.index >= r.begin) & (d.index <= t1)]; rec["P8_peak"] = w.P8.rolling(30, min_periods=10).mean().max() if len(w) else np.nan
    rows.append(rec)
o = pd.DataFrame(rows); o.to_csv(O, index=False); print(o.notna().sum().to_string())
