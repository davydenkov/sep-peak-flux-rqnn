#!/usr/bin/env python3
"""SOHO/EPHIN proton response known at the issue time (flare start .. flare end) relative to the 3-h pre-flare median, yearly L3I 1-min CDFs."""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd, cdflib
ROOT = Path(__file__).resolve().parents[1]
D = ROOT.parents[1] / "data/ephin_costep_cache"
files = {int(re.search(r"1min_(\d{4})", f)[1]): f for f in sorted(glob.glob(str(D / "soho_costep-ephin_l3i-1min_*.cdf")))}
cache = {}


def year(y):
    if y not in cache:
        if y not in files:
            cache[y] = None
        else:
            c = cdflib.CDF(files[y])
            t = pd.to_datetime(cdflib.cdfepoch.to_datetime(c.varget("Epoch")), utc=True)
            p = c.varget("P_int").astype(float)
            d = pd.DataFrame(p, index=t, columns=["P4", "P8", "P25", "P41"])
            cache[y] = d.where(d > 0).sort_index()
            if y == 2012:
                print("labels", c.varget("P_E_label"), c.varget("P_energy"))
    return cache[y]


def main():
    e = pd.read_csv(ROOT / "data/events_final.csv")
    for c in ("t_start", "t_peak", "t_end"):
        e[c] = pd.to_datetime(e[c], utc=True, format="mixed")
    rows, miss = [], []
    for r in e.itertuples():
        t1, t2 = min(r.t_start, r.t_peak), r.t_end
        t0 = t1 - pd.Timedelta(hours=3)
        ds = [year(y) for y in sorted({t0.year, t2.year})]
        rec = {"noaa_id": r.noaa_id}
        if any(x is None for x in ds):
            miss.append(f"{r.noaa_id[5:]}:{t1:%Y%m%d%H%M}:{t2:%Y%m%d%H%M}")
        else:
            d = pd.concat(ds)
            pre, post = d[(d.index >= t0) & (d.index < t1)], d[(d.index >= t1) & (d.index <= t2)]
            rec.update(eph_npre=len(pre), eph_npost=len(post))
            for ch in ("P8", "P25", "P41"):
                a, b = pre[ch].median(), post[ch].max()
                rec[f"{ch}_bg"] = a; rec[f"{ch}_max"] = b
        rows.append(rec)
    o = pd.DataFrame(rows); o.to_csv(ROOT / "data/ephin_local.csv", index=False)
    print(o.describe().T[["count", "min", "50%", "max"]].round(4).to_string())
    print("no local year:", len(miss)); print(" ".join(miss))


main()
