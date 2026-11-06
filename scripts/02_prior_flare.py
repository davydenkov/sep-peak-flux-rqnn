"""Most recent earlier flare in the same active region, for the alpha_PF term of Balch (2008), Equation (2)."""
from paths import *
import numpy as np, pandas as pd, importlib
from types import SimpleNamespace as NS
from xrs_load import Cache
F = importlib.import_module("01_xrs_features")
SITE = INP / "flares_regions.csv"
fc = pd.read_csv(SITE, usecols=["event_starttime", "event_peaktime", "event_endtime", "goes_class_score", "flare_class", "noaa_srs_region_number", "hek_sxr_fluence_proxy_wm2s"])
for c_ in ["event_starttime", "event_peaktime", "event_endtime"]:
    fc[c_] = pd.to_datetime(fc[c_], utc=True, format="ISO8601")
q = pd.read_csv(RES / "xrs_features.csv", parse_dates=["t_start", "t_peak", "t_end"])
print("catalog region fill", fc.noaa_srs_region_number.notna().mean(), "events region fill", q.region.notna().mean())
print(q.region.head(8).tolist())
c = Cache()
out = []
for r in q.itertuples():
    reg = r.region
    res = dict(idx=r.idx, prev_t=pd.NaT, prev_class=None, x_pf=np.nan, x_pf_src="none", n_prev=0)
    try:
        reg = int(float(reg)); reg = reg if reg < 10000 else reg % 10000
    except (TypeError, ValueError):
        reg = None
    if reg:
        ff = fc[((fc.noaa_srs_region_number % 10000) == reg) & (fc.event_peaktime < r.t_start) & (fc.event_peaktime > r.t_start - pd.Timedelta(days=14))].sort_values("event_peaktime")
        res["n_prev"] = len(ff)
        if len(ff):
            p = ff.iloc[-1]
            res.update(prev_t=p.event_peaktime, prev_class=p.flare_class)
            ns = NS(t_start=p.event_starttime, t_peak=p.event_peaktime, t_end=p.event_endtime, score=p.goes_class_score)
            m = F.measure(c, ns)
            if m and m["n"] >= 3:
                res.update(x_pf=m["x_int"], x_pf_src="xrs")
            else:
                res.update(x_pf=p.hek_sxr_fluence_proxy_wm2s * (0.7 if r.t_peak.year >= 2020 else 1.0), x_pf_src="proxy")
    out.append(res)
o = pd.DataFrame(out); o.to_csv(RES / "prior_flare.csv", index=False)
print(o.x_pf_src.value_counts().to_string()); print((o.x_pf > 0.08).sum(), "with X_pf>0.08")
print(o.sort_values("x_pf", ascending=False).head(8).to_string())
