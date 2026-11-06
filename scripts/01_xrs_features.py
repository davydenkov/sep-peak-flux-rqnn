"""Flare X-ray quantities of Balch (2008) and Garcia (2004) from 1-min GOES XRS, on the operational (pre-GOES-16) flux scale."""
from paths import *
import numpy as np, pandas as pd, json
from xrs_load import Cache
E0 = INP / "events_final.csv"
FAC_L, FAC_S = 0.70, 0.85           # SWPC operational scaling of the 1-8 and 0.5-4 A channels (true -> operational)

def thomas(short, long_):
    """Isothermal T [MK] and EM [cm^-3] from the two GOES channels, Thomas, Starr & Crannell (1985)."""
    R = short / long_
    T = 3.15 + 77.2 * R - 164 * R**2 + 205 * R**3
    b = -3.86 + 1.17 * T - 0.0131 * T**2 + 0.000178 * T**3
    return T, 1e55 * long_ / b

def pick(w, r):
    """Series with full minute coverage of the flare whose peak agrees best with the catalogued class on either flux scale."""
    best, sc = None, -1e9
    span = max((r.t_end - r.t_start).total_seconds() / 60, 1)
    for k, d in w.items():
        x = d.loc[r.t_start:r.t_end]
        if len(x) < 3: continue
        cov = min(x.index.floor("min").nunique() / (span + 1), 1)
        rt = x.Flux_Long.max() / (r.score * 1e-8)
        s = cov - min(abs(np.log(rt)), abs(np.log(rt / 1.43)))
        if s > sc: best, sc = (k, d), s
    return best

def measure(c, r):
    w = c.window(r.t_start - pd.Timedelta(minutes=60), r.t_end + pd.Timedelta(hours=10))
    if not w: return None
    p = pick(w, r)
    if p is None: return None
    k, d = p
    d = d[~d.index.duplicated()].copy()
    fl = d.loc[r.t_start:r.t_end]
    if len(fl) < 3: return None
    ratio = fl.Flux_Long.max() / (r.score * 1e-8)
    era_true = r.t_peak >= pd.Timestamp("2020-01-01", tz="UTC")
    true_units = era_true or ratio > 1.25 or (r.score >= 1.7e5 and ratio > 0.8)   # reprocessed series keep the flux above the operational saturation
    if true_units:
        d["Flux_Long"] *= FAC_L; d["Flux_Short"] *= FAC_S
    pre = d.loc[r.t_start - pd.Timedelta(minutes=30):r.t_start]
    bl = pre.Flux_Long.quantile(0.1) if len(pre) >= 3 else d.Flux_Long.iloc[:5].min()
    bs = pre.Flux_Short.quantile(0.1) if len(pre) >= 3 else d.Flux_Short.iloc[:5].min()
    tp = d.loc[r.t_start:r.t_end].Flux_Long.idxmax()
    xm = d.Flux_Long[tp]
    post = d.loc[tp:]
    half = bl + 0.5 * (xm - bl)
    below = post[post.Flux_Long <= half]
    t_half = below.index[0] if len(below) else post.index[-1]
    t_half = min(t_half, r.t_end) if r.t_end > tp else t_half   # the catalogued end uses the same half-power definition; stops the integral running into the next flare
    seg = d.loc[r.t_start:t_half]
    dt = np.diff(seg.index.values).astype("timedelta64[s]").astype(float)
    x_int = float(np.sum(0.5 * (seg.Flux_Long.values[1:] + seg.Flux_Long.values[:-1]) * dt))
    x_int_bs = float(np.sum(0.5 * ((seg.Flux_Long.values[1:] + seg.Flux_Long.values[:-1]) - 2 * bl) * dt))
    # Garcia: duration from onset until 0.5-4 A flux falls to 25 % of its peak
    sh = d.loc[r.t_start:r.t_end + pd.Timedelta(hours=10)]
    tps = d.loc[r.t_start:r.t_end].Flux_Short.idxmax(); sm = d.Flux_Short[tps]
    dec = sh.loc[tps:]; dec = dec[dec.Flux_Short <= 0.25 * sm]
    t25 = dec.index[0] if len(dec) else sh.index[-1]
    # emission measure at the 1-8 A peak, background-subtracted
    Ls, Ss = max(xm - bl, 1e-12), max(d.Flux_Short[tp] - bs, 1e-12)
    T, EM = thomas(Ss, Ls)
    # peak EM over rise-to-peak window
    z = d.loc[r.t_start:tp]
    Tz, Ez = thomas((z.Flux_Short - bs).clip(lower=1e-12), (z.Flux_Long - bl).clip(lower=1e-12))
    gaps = float(np.max(dt)) / 60 if len(dt) else np.nan
    return dict(src=f"{k[0]}/{k[1]}", ratio=ratio, true_units=true_units, x_max=xm, bg=bl, t_half=t_half, x_int=x_int, x_int_bs=x_int_bs,
                t25=t25, dur25_h=(t25 - r.t_start).total_seconds() / 3600, T_pk=T, logEM_pk=np.log10(EM), logEM_max=np.log10(np.nanmax(Ez[(Tz > 3)])) if (Tz > 3).any() else np.nan,
                gap_min=gaps, n=len(seg), peak_shift_min=(tp - r.t_peak).total_seconds() / 60, cov=d.loc[r.t_start:r.t_end].index.floor("min").nunique() / (max((r.t_end - r.t_start).total_seconds() / 60, 1) + 1))

if __name__ == "__main__":
    c = Cache()
    e = pd.read_csv(E0, parse_dates=["t_start", "t_peak", "t_end", "begin"]).sort_values("begin").reset_index(drop=True)
    # HEK entry for the X2.3 flare of 2000 June 6 is a 6 s fragment 10 min after the true maximum; use the GOES light curve (maximum 15:25 UT)
    fix = e.noaa_id == "NOAA-200006071310"
    e.loc[fix, "t_start"] = pd.Timestamp("2000-06-06 14:58", tz="UTC"); e.loc[fix, "t_peak"] = pd.Timestamp("2000-06-06 15:25", tz="UTC"); e.loc[fix, "t_end"] = pd.Timestamp("2000-06-06 15:40", tz="UTC")
    rows = []
    for i, r in enumerate(e.itertuples()):
        m = measure(c, r) or {}
        rows.append(dict(idx=i, noaa_id=r.noaa_id, t_start=r.t_start, t_peak=r.t_peak, t_end=r.t_end, hek_class=r.hek_class, score=r.score, region=r.region, lon_w=r.lon_w, peak_pfu=r.peak_pfu, **m))
    q = pd.DataFrame(rows); q.to_csv(RES / "xrs_features.csv", index=False)
    print(q.x_int.notna().sum(), "of", len(q))
    print(q[["x_max","x_int","x_int_bs","dur25_h","T_pk","logEM_pk","logEM_max","gap_min","ratio"]].describe().T.to_string())
    print(q[q.x_int.isna()][["t_peak","hek_class"]].to_string())
    print(q.true_units.value_counts().to_string())
    print(q[q.peak_shift_min.abs() > 5][["t_peak", "hek_class", "src", "peak_shift_min", "x_max"]].to_string())
    print(q[(q["cov"] < .8) | (q.ratio < .9) | ((q.ratio > 1.2) & (q.ratio < 1.33)) | (q.ratio > 1.65)][["t_peak", "t_end", "hek_class", "src", "ratio", "cov", "x_max", "x_int"]].to_string())
