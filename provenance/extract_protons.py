#!/usr/bin/env python3
"""Reduce the local GOES proton cache to one >10 MeV proxy series per event (begin-2 h to max+6 h). Provenance only: needs the local cache."""
import pickle, sqlite3, sys, zlib, re
from pathlib import Path
import numpy as np, pandas as pd
DB = Path(__file__).resolve().parents[3] / "data/goes_protons_cache/goes_proton_series_cache.sqlite"
e = pd.read_csv("../../rqnn_jaa/data/events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e["tmax"] = pd.to_datetime(e.tmax, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
c = sqlite3.connect(DB); k = pd.read_sql("select cache_key,span_a,span_b from sunpy_series_blob", c)
k["a"] = pd.to_datetime(k.span_a, utc=True); k["b"] = pd.to_datetime(k.span_b, utc=True) + pd.Timedelta(days=1); k["be"] = k.cache_key.str.split("|").str[5]
def proxy(d, be):
    """>10 MeV integral proxy in pfu from the channels of each instrument."""
    if be == "cdaweb": return d["CDAWEB_P1"].astype(float)
    if be == "ncei_sem_epead":   # GOES-13/15 EPS differential, 1/(cm2 s sr MeV): 9-15, 15-40, 40-80, 80-165, 165-500 MeV
        return sum(d[f"SEM_P{i}"].clip(lower=0).astype(float) * w for i, w in ((3, 6), (4, 25), (5, 40), (6, 85), (7, 335)))
    cols = {}
    for x in d.columns:
        m = re.match(r"P(\d+)_([\d.]+)-([\d.]+)MeV", x)
        if m: cols[x] = (float(m[2]), float(m[3]))
    s = 0
    for x, (lo, hi) in cols.items():
        if hi > 10:
            w = (hi - max(lo, 10)) * 1000.0   # GOES-R L2 differential flux per keV
            s = s + d[x].clip(lower=0).astype(float) * w
    return s
rows = []; series = []
for i, r in e.iterrows():
    t0, t1 = r.begin - pd.Timedelta(hours=2), r.tmax + pd.Timedelta(hours=6)
    best = None
    kk = k[(k.a <= t0) & (k.b >= t1)]
    if not len(kk): kk = k[(k.a <= r.begin) & (k.b >= r.tmax)]
    for key, be in zip(kk.cache_key, kk.be):
        o = pickle.loads(zlib.decompress(c.execute("select payload from sunpy_series_blob where cache_key=?", (key,)).fetchone()[0]))
        d = pd.DataFrame(o)
        if not len(d): continue
        d["t"] = pd.to_datetime(d.DateTime, utc=True); d = d[(d.t >= t0) & (d.t <= t1)]
        if len(d) < 20: continue
        p = proxy(d, be)
        if isinstance(p, int): continue
        s = pd.Series(np.asarray(p, float), index=d.t).replace([np.inf, -np.inf], np.nan).dropna(); s = s[s >= 0]
        if len(s) < 20: continue
        sc = len(s) - 200 * abs(np.log10(max(s.max(), 1e-3)) - np.log10(r.peak_pfu))
        if best is None or sc > best[0]: best = (sc, be, s, key)
    if best is None: rows.append(dict(i=i, be=None)); continue
    _, be, s, key = best; s = s[~s.index.duplicated()].sort_index()
    rows.append(dict(i=i, be=be, n=len(s), pmax=s.max(), noaa=r.peak_pfu, ratio=np.log10(s.max() / r.peak_pfu), sat=key.split("|")[4]))
    series.append(pd.DataFrame({"noaa_id": r.noaa_id, "t": s.index.strftime("%Y-%m-%dT%H:%M:%S"), "pfu_proxy": s.values, "backend": be}))
q = pd.DataFrame(rows); print(q.be.value_counts(dropna=False).to_string()); print(q.groupby("be").ratio.describe()[["count", "50%", "min", "max", "std"]].round(2).to_string())
pd.concat(series).to_csv(sys.argv[1], index=False, compression={"method": "gzip", "mtime": 0}); q.to_csv("../data/proton_extract_log.csv", index=False)
