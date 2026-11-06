#!/usr/bin/env python3
"""Microwave burst from the RSTN fixed-frequency network (local day files), known at the issue time: flare start .. flare end."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DAYS = ROOT.parents[1] / "data/targeted_rstn_cache/days"
COLS = ["Radio_245MHz", "Radio_410MHz", "Radio_610MHz", "Radio_1415MHz", "Radio_2695MHz", "Radio_4995MHz", "Radio_8800MHz", "Radio_15400MHz"]


def main():
    e = pd.read_csv(ROOT / "data/events_final.csv")
    for c in ("t_start", "t_peak", "t_end"):
        e[c] = pd.to_datetime(e[c], utc=True, format="mixed")
    rows = []
    for r in e.itertuples():
        t1, t2 = min(r.t_start, r.t_peak), r.t_end
        ds = sorted({f"{x:%Y%m%d}" for x in (t1 - pd.Timedelta(minutes=30), t2)})
        fs = [DAYS / f"{x}.csv" for x in ds]
        rec = {"noaa_id": r.noaa_id, "rstn_file": int(all(f.exists() for f in fs))}
        if rec["rstn_file"]:
            d = pd.concat([pd.read_csv(f) for f in fs])
            d["DateTime"] = pd.to_datetime(d.DateTime, utc=True, errors="coerce", format="mixed")
            d = d.dropna(subset=["DateTime"]).sort_values("DateTime").set_index("DateTime")
            pre = d[(d.index >= t1 - pd.Timedelta(minutes=30)) & (d.index < t1)]
            post = d[(d.index >= t1) & (d.index <= t2)]
            rec["rstn_npre"], rec["rstn_npost"] = len(pre), len(post)
            for c in COLS:
                if c in d and len(pre) >= 5 and len(post) >= 3:
                    a = pd.to_numeric(pre[c], errors="coerce").median(); b = pd.to_numeric(post[c], errors="coerce")
                    ex = (b - a).clip(lower=0)
                    rec[c.replace("Radio_", "mw_peak_")] = float(ex.max()); rec[c.replace("Radio_", "mw_int_")] = float(ex.sum())
        rows.append(rec)
    o = pd.DataFrame(rows); o.to_csv(ROOT / "data/rstn_features.csv", index=False)
    print(len(o), "day files", int(o.rstn_file.sum()), "with window data", int(o.get("mw_int_2695MHz", pd.Series(dtype=float)).notna().sum()))
    print(o.filter(like="mw_int").notna().sum().to_dict())
    m = e.merge(o, on="noaa_id"); from scipy.stats import spearmanr
    for c in ("mw_int_1415MHz", "mw_int_2695MHz", "mw_int_8800MHz", "mw_peak_2695MHz"):
        x = m[[c, "log_delay"]].dropna(); x = x[x[c] > 0]
        print(c, len(x), np.round(spearmanr(x[c], x.log_delay), 4))
    print(m.groupby("cycle")["mw_int_2695MHz"].apply(lambda s: s.notna().sum()).to_dict(), m[m.mw_int_2695MHz.isna()].year.value_counts().sort_index().to_dict())


main()
