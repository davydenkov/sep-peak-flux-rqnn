#!/usr/bin/env python3
"""IMF / solar-wind state in the 6 h before SXR flare start, from OMNI HRO 1-min CDFs (L1, time-shifted to the bow-shock nose)."""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd, cdflib

ROOT = Path(__file__).resolve().parents[1]
OMNI_DIR = Path.home() / "sunpy/data"
OMEGA = 2.86533e-6  # rad/s, sidereal (Carrington-rate) solar rotation
AU_KM = 1.495978707e8
FILL = {"BX_GSE": 9000, "BY_GSE": 9000, "BZ_GSE": 9000, "flow_speed": 90000, "proton_density": 900}


def load_month(ym):
    fs = sorted(glob.glob(str(OMNI_DIR / f"omni_hro_1min_{ym}01_v*.cdf")))
    if not fs:
        return None
    c = cdflib.CDF(fs[-1])
    t = cdflib.cdfepoch.to_datetime(c.varget("Epoch"))
    d = pd.DataFrame({k: c.varget(k).astype(float) for k in FILL}, index=pd.to_datetime(t, utc=True))
    for k, v in FILL.items():
        d.loc[d[k].abs() >= v, k] = np.nan
    return d


def imf_features(w):
    """Directional statistics of the field line relative to the nominal Parker spiral (unsigned line direction)."""
    w = w.dropna(subset=["BX_GSE", "BY_GSE", "BZ_GSE"])
    out = {"imf_n": len(w)}
    if len(w) < 60:
        return out
    V = np.nanmedian(w.flow_speed)
    b = w[["BX_GSE", "BY_GSE", "BZ_GSE"]].to_numpy()
    B = np.linalg.norm(b, axis=1); ok = B > 0
    bh = b[ok] / B[ok, None]
    psi = np.arctan(OMEGA * AU_KM / V)                    # nominal spiral angle from the radial direction
    s = np.array([-np.cos(psi), np.sin(psi), 0.0])        # 'away' sector unit vector in GSE
    dot = bh @ s
    pol = np.sign(np.nanmedian(dot))                      # +1 away, -1 toward (Parker-projected polarity)
    m = (bh * np.sign(dot)[:, None]).mean(axis=0)         # sign-folded mean direction
    R = np.linalg.norm(m)
    mh = m / R
    phi_line = np.degrees(np.arctan2(mh[1], mh[0]))       # azimuth of the folded line in GSE
    psi_meas = np.degrees(np.arctan2(abs(mh[1]), abs(mh[0])))
    out.update(sw_v=V, sw_n=np.nanmedian(w.proton_density), imf_b=np.nanmean(B), imf_align=float(np.abs(dot).mean()), imf_align_mean=float(abs(mh @ s)),
               imf_delta_deg=float(np.degrees(np.arccos(min(1.0, abs(mh @ s))))), imf_polarity=float(pol), imf_stability=float(R),
               imf_phi_deg=float(phi_line), imf_psi_meas_deg=float(psi_meas), imf_psi_parker_deg=float(np.degrees(psi)),
               imf_theta_deg=float(np.degrees(np.arcsin(np.clip(mh[2], -1, 1)))), imf_bz_mean=float(np.nanmean(w.BZ_GSE)),
               imf_away_fraction=float((dot > 0).mean()))
    return out


def main():
    e = pd.read_csv(ROOT / "data/events_stage1.csv")
    for c in ("begin", "t_start", "t_peak", "t_end"):
        e[c] = pd.to_datetime(e[c], utc=True, format="mixed")
    cache = {}
    rec = []
    for r in e.itertuples():
        t1 = min(r.t_start, r.t_peak); t0 = t1 - pd.Timedelta(hours=6)
        parts = []
        for ym in sorted({f"{t0:%Y%m}", f"{t1:%Y%m}"}):
            if ym not in cache:
                cache[ym] = load_month(ym)
            if cache[ym] is not None:
                parts.append(cache[ym])
        if not parts:
            rec.append({"noaa_id": r.noaa_id, "omni_file": 0}); continue
        d = pd.concat(parts).sort_index()
        rec.append({"noaa_id": r.noaa_id, "omni_file": 1, **imf_features(d[(d.index >= t0) & (d.index < t1)])})
    o = pd.DataFrame(rec)
    o.to_csv(ROOT / "data/imf_local.csv", index=False)
    print(len(o), "files", int(o.omni_file.sum()), "imf ok", int(o.imf_align.notna().sum()) if "imf_align" in o else 0)
    print(o.describe().T[["count", "mean", "min", "50%", "max"]].round(2).to_string())
    print(sorted({f"{pd.Timestamp(t):%Y%m}" for t, m in zip(e.t_start, o.omni_file) if not m}))
    print("no imf:", o[o.omni_file.eq(1) & o.imf_align.isna()].noaa_id.tolist())


main()
