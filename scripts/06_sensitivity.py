#!/usr/bin/env python3
"""Sensitivity of the Garcia-Balch scores to the flux scale and integration, lead times, and residual physics of the fluence law."""
from paths import *
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
D = RES
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv", parse_dates=["t_start", "t_peak", "t_end", "t_half", "t25"]); F = pd.read_csv(D / "forecasts_loyo.csv")
y = np.log10(e.peak_pfu.to_numpy()); ok = q.x_int.notna().to_numpy()
lx = np.log10(q.x_int.to_numpy())
def sc(p, m=ok):
    a = np.abs(p[m] - y[m]); return dict(N=int(m.sum()), MAE=round(a.mean(), 3), bias=round((p[m] - y[m]).mean(), 3), fac=round(10 ** np.median(a), 2), w3=round((a <= np.log10(3)).mean() * 100, 1), r=round(pearsonr(p[m], y[m])[0], 3))
B = lambda l: 1 + 0.82 * (l - np.log10(0.00987))
c25 = e.cycle.to_numpy() == 25; pre = ~c25
S = {}
S["operational scale"] = sc(B(lx))
S["background subtracted"] = sc(B(np.log10(q.x_int_bs.to_numpy())))
S["true irradiance (x1/0.7 all)"] = sc(B(lx - np.log10(0.7)))
tr = np.where(e.begin >= pd.Timestamp("2020-01-01", tz="UTC"), lx - np.log10(0.7), lx)
S["no rescaling after 2020"] = sc(B(tr))
S["triangular proxy"] = sc(B(e.log_flu.to_numpy()))
S["visible disk, |lon|<=90"] = sc(B(lx), ok & (e.lon_w.abs().to_numpy() <= 90))
S["X_int above Balch minimum"] = sc(B(lx), ok & (q.x_int.to_numpy() > 0.00993))
S["c25 operational"] = sc(B(lx), ok & c25); S["c25 no rescaling"] = sc(B(tr), ok & c25)
G = lambda em, x, dt: -115.67 + 2.34 * em - 1055 * x + 0.44 * dt
EM, X, DT = q.logEM_pk.to_numpy(), q.x_max.to_numpy(), q.dur25_h.to_numpy()
S["garcia"] = sc(G(EM, X, DT))
S["garcia EM at maximum"] = sc(G(q.logEM_max.to_numpy(), X, DT))
S["garcia true flux (EM+0.155, X/0.7)"] = sc(G(EM + 0.155, X / 0.7, DT))
lxm = np.log10(X); Tn = 57.9 + 9.52 * lxm + 0.06 * lxm**2
cold = ok & (q.T_pk.to_numpy() < Tn)
S["garcia cool flares"] = sc(G(EM, X, DT), cold); S["balch on cool flares"] = sc(B(lx), cold); S["balch on hot flares"] = sc(B(lx), ok & ~cold)
off = np.mean(y[cold] - G(EM, X, DT)[cold])
S["garcia offset removed (in-sample)"] = sc(G(EM, X, DT) + np.mean((y - G(EM, X, DT))[ok]))
S["garcia cool, offset"] = dict(offset=round(off, 3), EM_equiv=round(off / 2.34, 3))
print(pd.DataFrame(S).T.to_string())
# lead times
tmax = pd.to_datetime(e.tmax, utc=True, format="mixed"); iss = pd.to_datetime(e.issue, utc=True, format="mixed"); beg = e.begin
L = {}
for n, t in (("balch t_half", q.t_half), ("garcia t25", q.t25), ("rqnn issue", iss)):
    t = pd.to_datetime(t, utc=True, format="mixed")
    L[n] = dict(after_peak_min=np.nanmedian((t - q.t_peak).dt.total_seconds() / 60), to_onset_h=np.nanmedian((beg - t).dt.total_seconds() / 3600), to_max_h=np.nanmedian((tmax - t).dt.total_seconds() / 3600), before_onset_pct=np.nanmean((beg > t)[ok]) * 100, before_max_pct=np.nanmean((tmax > t)[ok]) * 100)
print(pd.DataFrame(L).T.round(2).to_string())
# residuals of the LOYO refit law against physical quantities (Spearman; Holm over the listed set)
res = (y - F.balch_fit.to_numpy())
V = {"Connection factor": e.connect, "West longitude": e.lon_w, "Absolute latitude": e.lat.abs(), "T (GOES)": q.T_pk, "Peak 1-8 A flux": np.log10(q.x_max), "Rise time": e.log_rise, "Duration (25 %)": np.log10(q.dur25_h), "Prior SEP 72 h": e.prior_sep_72h, "CME speed": e.cme_v, "CME width": e.cme_w,
     "CME speed (pre-2019)": e.cme_v.where(e.year < 2019), "Near-rel. electrons": e.DE4_rise, "Type III power": e.III_int, "Microwave 2.7 GHz": e.mw_log, "Early proton rise": e.Rp_P25, "Background J10": np.log10(e.J10_bg.where(e.J10_bg > 0)), "Delay": e.log_delay}
rows = []
for k, v in V.items():
    v = np.asarray(v, float); m = ok & np.isfinite(v) & np.isfinite(res)
    r, p = spearmanr(v[m], res[m]); rng = np.random.default_rng(2); ii = rng.integers(0, m.sum(), (3000, m.sum()))
    bs = [spearmanr(v[m][i], res[m][i])[0] for i in ii[:1500]]
    rows.append(dict(q=k, N=int(m.sum()), rho=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p=p))
R = pd.DataFrame(rows).sort_values("p"); n = len(R); R["p_holm"] = np.minimum(1, np.maximum.accumulate(R.p.to_numpy() * (n - np.arange(n)))); R.to_csv(D / "residual_physics.csv", index=False); print(R.round(3).to_string(index=False))
json.dump({"sens": S, "lead": {k: {a: float(b) for a, b in v.items()} for k, v in L.items()}}, open(D / "sensitivity.json", "w"), indent=1, default=float)
# how many events in each NOAA S-level are forecast at the right level
lev = lambda v: np.digitize(v, [2, 3, 4])
for n in ("balch", "balch_fit", "rqnn:G3_full", "rf:G3_full", "clim:none"):
    p = F[n].to_numpy()[ok]; t = y[ok]; print(n, "level hit %", round((lev(p) == lev(t)).mean() * 100, 1), " max pred pfu", round(10 ** p.max()), " pred>=100:", int((p >= 2).sum()), "POD100", round(((p >= 2) & (t >= 2)).sum() / (t >= 2).sum(), 2), "FAR100", round(((p >= 2) & (t < 2)).sum() / max((p >= 2).sum(), 1), 2), "POD1000", round(((p >= 3) & (t >= 3)).sum() / (t >= 3).sum(), 2))
print(q.assign(cyc=e.cycle)[["cyc","T_pk","logEM_pk","x_int","x_max"]].groupby("cyc").median().round(4).to_string()); print("extremes", e.assign(p=F["rqnn:G3_full"], b=F.balch, bf=F.balch_fit).sort_values("peak_pfu", ascending=False).head(8)[["begin", "hek_class", "peak_pfu", "lon_w", "p", "b", "bf"]].assign(p=lambda d: 10**d.p, b=lambda d: 10**d.b, bf=lambda d: 10**d.bf).round(0).to_string())
