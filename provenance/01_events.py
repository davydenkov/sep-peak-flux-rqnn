#!/usr/bin/env python3
"""Clean SEP onset sample. Target: NOAA >10 MeV 10-pfu begin minus the SXR peak of the NOAA-attributed flare.
No quantity derived from the proton onset is used to define the reference time."""
import re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT.parents[1] / "struminsky/paper/sep_forecast_site"
OUT = ROOT / "data"


def flare_time(begin, meta):
    m = re.search(r"(\d{2})/(\d{2})\s+(\d{4})", str(meta))
    if not m:
        return pd.NaT
    mm, dd, hm = int(m[1]), int(m[2]), m[3]
    y = begin.year - (1 if (mm == 12 and begin.month == 1) else 0)
    try:
        return pd.Timestamp(year=y, month=mm, day=dd, hour=int(hm[:2]), minute=int(hm[2:]), tz="UTC")
    except ValueError:
        return pd.NaT


def lon_lat(loc):
    m = re.search(r"([NS])\s*(\d{1,2})\s*([EW])\s*(\d{1,3})", str(loc).upper())
    if not m:
        return np.nan, np.nan
    return (1 if m[3] == "W" else -1) * float(m[4]), (1 if m[1] == "N" else -1) * float(m[2])


def main():
    n = pd.read_csv(SITE / "data/sep_events_noaa.csv")
    n["begin"] = pd.to_datetime(n.begin_utc, utc=True); n["tmax"] = pd.to_datetime(n.max_utc, utc=True)
    n = n[n.begin >= pd.Timestamp("1996-08-01", tz="UTC")].copy()
    log = {"NOAA list since 1996-08": len(n)}
    n["t_flare"] = [flare_time(b, f) for b, f in zip(n.begin, n.flare)]
    ll = n.location.map(lon_lat); n["lon_w"] = [a for a, _ in ll]; n["lat"] = [b for _, b in ll]
    n["flare_class_noaa"] = n.flare.astype(str).str.extract(r"\b([CMX]\d+(?:\.\d+)?)", flags=re.I)[0]
    n = n[n.t_flare.notna() & n.flare_class_noaa.notna()]
    log["with attributed flare (time, class)"] = len(n)
    fl = pd.read_csv(SITE / "data/flares_catalog.csv", low_memory=False,
                     usecols=["hek_event_id", "event_starttime", "event_peaktime", "event_endtime", "flare_class", "goes_class_score",
                              "hek_cme_flag", "hek_cme_speed_kms", "hek_cme_width_deg", "hek_cme_time_delta_min"])
    for c in ("event_starttime", "event_peaktime", "event_endtime"):
        fl[c] = pd.to_datetime(fl[c], utc=True, format="mixed", errors="coerce")
    rows = []
    for r in n.itertuples():
        d = (fl.event_peaktime - r.t_flare).abs()
        cand = fl[d <= pd.Timedelta(minutes=30)]
        rec = r._asdict(); rec.pop("Index")
        if len(cand):
            f = cand.sort_values("goes_class_score").iloc[-1]
            rec.update(hek=f.hek_event_id, t_start=f.event_starttime, t_peak=f.event_peaktime, t_end=f.event_endtime, hek_class=f.flare_class,
                       score=f.goes_class_score, cme_flag=f.hek_cme_flag, cme_v=f.hek_cme_speed_kms, cme_w=f.hek_cme_width_deg, cme_dt=f.hek_cme_time_delta_min)
        rows.append(rec)
    e = pd.DataFrame(rows)
    log["matched to HEK flare within 30 min"] = int(e.hek.notna().sum())
    e = e[e.hek.notna()].copy()
    e["delay_min"] = (e.begin - e.t_peak).dt.total_seconds() / 60
    e["cls_letter_ok"] = e.flare_class_noaa.str[0].str.upper() == e.hek_class.str[0].str.upper()
    log["class letter agrees NOAA vs HEK"] = int(e.cls_letter_ok.sum())
    print(log); print(e.delay_min.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(0).to_dict())
    print("end<peak", int((e.t_end <= e.t_peak).sum()), "dup flares", int(e.hek.duplicated().sum()), "onset before flare end", int((e.begin <= e.t_end).sum()))
    print(e[e.delay_min <= 0][["noaa_id", "flare", "t_peak", "delay_min"]].to_string())
    print(e[~e.cls_letter_ok][["noaa_id", "flare", "hek_class"]].head(12).to_string())
    print(e[e.hek.duplicated(keep=False)][["noaa_id", "flare", "delay_min", "peak_pfu"]].to_string())
    e.to_csv(OUT / "events_stage1.csv", index=False)
    pd.Series(log).to_csv(OUT / "selection_log_stage1.csv")


main()
