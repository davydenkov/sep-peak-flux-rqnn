#!/usr/bin/env python3
"""Reduce the local GOES XRS cache (3.6 GB) to the 1-min series actually used: one window per target flare and per preceding flare in the same region.
Needs the local cache, so it is kept for provenance only; the output is data/input/xrs_windows.csv.gz."""
import importlib, sys
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
from xrs_load import Cache
F = importlib.import_module("15_xrs_features")
OUT = sys.argv[1]; FLARES = sys.argv[2]
SITE = "../../../struminsky/paper/sep_forecast_site/data/flares_catalog.csv"
cols = ["event_starttime", "event_peaktime", "event_endtime", "goes_class_score", "flare_class", "noaa_srs_region_number", "hek_sxr_fluence_proxy_wm2s"]
fc = pd.read_csv(SITE, usecols=cols)
for c_ in cols[:3]: fc[c_] = pd.to_datetime(fc[c_], utc=True, format="ISO8601")
q = pd.read_csv("../data/xrs_features.csv", parse_dates=["t_start", "t_peak", "t_end"])
c = Cache(); rows = []; keep = []
def grab(r):
    t0, t1 = r.t_start - pd.Timedelta(minutes=60), r.t_end + pd.Timedelta(hours=10)
    w = c.window(t0, t1)
    p = F.pick(w, r) if w else None
    if p is None: return
    (sat, src), d = p
    d = d[~d.index.duplicated()]
    rows.append(pd.DataFrame({"t0": t0.isoformat(), "t1": t1.isoformat(), "sat": sat, "src": src, "t": d.index.strftime("%Y-%m-%dT%H:%M:%S.%f"), "short": d.Flux_Short.to_numpy(), "long": d.Flux_Long.to_numpy()}))
for r in q.itertuples():
    grab(r)
    try:
        reg = int(float(r.region)); reg = reg if reg < 10000 else reg % 10000
    except (TypeError, ValueError):
        continue
    ff = fc[((fc.noaa_srs_region_number % 10000) == reg) & (fc.event_peaktime < r.t_start) & (fc.event_peaktime > r.t_start - pd.Timedelta(days=14))].sort_values("event_peaktime")
    keep.append(ff)
    if len(ff):
        p = ff.iloc[-1]; grab(NS(t_start=p.event_starttime, t_peak=p.event_peaktime, t_end=p.event_endtime, score=p.goes_class_score))
o = pd.concat(rows).drop_duplicates(["t0", "t1", "t"]); o.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
k = pd.concat(keep).drop_duplicates(); k.to_csv(FLARES, index=False)
print(len(rows), "windows", len(o), "rows;", len(k), "catalogue flares")
