#!/usr/bin/env python3
"""Near-relativistic electron / ion response at L1 known at the issue time (flare end), from ACE/EPAM 5-min data.
Pre-flare window: 3 h before flare start. Response window: flare start .. flare end. Both end before or at the issue time."""
import glob, os, re
from pathlib import Path
import numpy as np, pandas as pd, cdflib

ROOT = Path(__file__).resolve().parents[1]
DIRS = [Path.home() / "graviton/data/ace_epam_sunpy_cache/AC_H1_EPM", Path.home() / "graviton/struminsky/data_paths/AC_H1_EPM"]
FILES = {}
for d in DIRS:
    for f in glob.glob(str(d / "*.cdf")):
        FILES[re.search(r"_(\d{8})_", os.path.basename(f))[1]] = f
VARS = ["DE1", "DE4", "P1p", "P5p"]


def day(dstr, cache={}):
    if dstr not in cache:
        f = FILES.get(dstr)
        if f is None:
            cache[dstr] = None
        else:
            c = cdflib.CDF(f)
            t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")), utc=True)
            d = pd.DataFrame({v: c.varget(v).astype(float) for v in VARS}, index=t)
            cache[dstr] = d.where(d > 0)
    return cache[dstr]


def main():
    e = pd.read_csv(ROOT / "data/events_stage1.csv")
    for c in ("t_start", "t_peak", "t_end"):
        e[c] = pd.to_datetime(e[c], utc=True, format="mixed")
    rows, missing = [], []
    for r in e.itertuples():
        t1 = min(r.t_start, r.t_peak); t2 = max(r.t_end, r.t_peak + pd.Timedelta(minutes=5)); t0 = t1 - pd.Timedelta(hours=3)
        ds = sorted({f"{x:%Y%m%d}" for x in (t0, t1, t2)})
        parts = [day(x) for x in ds]
        if any(p is None for p in parts):
            missing.append(r.noaa_id[5:]); rows.append({"noaa_id": r.noaa_id}); continue
        d = pd.concat(parts).sort_index()
        pre, post = d[(d.index >= t0) & (d.index < t1)], d[(d.index >= t1) & (d.index <= t2)]
        rec = {"noaa_id": r.noaa_id, "epam_npre": len(pre), "epam_npost": len(post)}
        for v in VARS:
            a, b = pre[v].median(), post[v].max()
            rec[f"{v}_bg"] = np.log10(a) if a > 0 else np.nan
            rec[f"{v}_rise"] = np.log10(b / a) if a > 0 and b > 0 else np.nan
        rows.append(rec)
    o = pd.DataFrame(rows); o.to_csv(ROOT / "data/epam_local.csv", index=False)
    print(o.describe().T[["count", "mean", "min", "50%", "max"]].round(2).to_string())
    print("missing", len(missing), " ".join(missing))


main()
