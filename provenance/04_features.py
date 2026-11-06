#!/usr/bin/env python3
"""Final event table and selection log. Every input is known at the issue time (flare end; CME inputs at first C2 detection)."""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "data"
OMEGA, AU = 2.86533e-6, 1.495978707e8
IMF_COLS = ["noaa", "imf_rows", "imf_n", "v_n", "sw_v", "sw_n", "imf_b", "imf_align", "imf_align_mean", "imf_delta_deg", "imf_polarity", "imf_stability",
            "imf_phi_deg", "imf_psi_meas_deg", "imf_psi_parker_deg", "imf_theta_deg", "imf_bz_mean", "imf_away_fraction"]
EP_COLS = ["noaa", "epam_npre", "epam_npost", "DE1_bg", "DE1_rise", "DE4_bg", "DE4_rise", "P1p_bg", "P1p_rise", "P5p_bg", "P5p_rise"]


def table(path, cols):
    rows = [x.split(",") for x in path.read_text().split()]
    rows = [r + [""] * (len(cols) - len(r)) for r in rows]
    d = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["noaa_id"] = "NOAA-" + d.noaa
    return d.drop(columns="noaa")


def fill(a, b):
    a = a.set_index("noaa_id"); b = b.set_index("noaa_id")
    for c in b.columns:
        if c not in a:
            a[c] = np.nan
        a[c] = a[c].where(a[c].notna(), b[c].reindex(a.index))
    return a.reset_index()


def main():
    e = pd.read_csv(R / "events_stage1.csv")
    for c in ("begin", "tmax", "t_start", "t_peak", "t_end"):
        e[c] = pd.to_datetime(e[c], utc=True, format="mixed")
    imf = fill(pd.read_csv(R / "imf_local.csv"), table(R / "imf_hapi.txt", IMF_COLS))
    ep = fill(pd.read_csv(R / "epam_local.csv"), table(R / "epam_hapi.txt", EP_COLS))
    pb = table(R / "omni2_proton_background.txt", ["noaa", "J10_bg", "J30_bg", "J60_bg"])
    e = e.merge(imf, on="noaa_id", how="left").merge(ep, on="noaa_id", how="left").merge(pb, on="noaa_id", how="left")
    log = [("NOAA events since 1996-08 (live list, 2026-10-30)", 176), ("attributed flare with class and time", 157), ("matched to a HEK GOES flare within 30 min", len(e))]

    def keep(mask, why):
        nonlocal e
        e = e[mask].copy(); log.append((why, len(e)))

    keep(e.delay_min > 0, "onset after the flare peak")
    e = e.sort_values("begin")
    keep(~e.hek.duplicated(keep="first"), "first NOAA line per flare (re-crossings removed)")
    keep((e.t_end > e.t_peak) & (e.t_peak >= e.t_start), "valid HEK start/peak/end order")
    keep(e.lon_w.notna(), "NOAA heliographic location available")
    keep(e.imf_align.notna() & e.sw_v.notna(), "OMNI IMF and speed in the 6 h pre-flare window")
    log.append(("  of which delay <= 48 h", int((e.delay_min <= 2880).sum())))
    keep(e.delay_min <= 1440, "delay <= 24 h (primary sample)")

    e["year"] = e.begin.dt.year
    e["cycle"] = np.where(e.begin < pd.Timestamp("2008-12-01", tz="UTC"), 23, np.where(e.begin < pd.Timestamp("2019-12-01", tz="UTC"), 24, 25))
    e["log_delay"] = np.log10(e.delay_min)
    e["log_flux"] = np.log10(e.score * 1e-8)
    dur = (e.t_end - e.t_start).dt.total_seconds() / 60
    rise = ((e.t_peak - e.t_start).dt.total_seconds() / 60).clip(lower=1)
    e["log_rise"] = np.log10(rise); e["log_dur"] = np.log10(dur)
    e["log_flu"] = np.log10(10 ** e.log_flux * dur * 60 / 2)
    foot = np.degrees(OMEGA * AU / e.sw_v)
    e["foot_w"] = foot
    e["sep_angle"] = (e.lon_w - foot).abs()
    e["connect"] = np.cos(np.radians(e.lon_w - foot))
    e["lat_abs"] = e.lat.abs()
    e["cme_v_catalogue"] = e.cme_v.where(e.cme_v > 0); e["cme_w_catalogue"] = e.cme_w.where(e.cme_w > 0)
    # a CME is used only if its reference time is at most 60 min after the SXR peak: a deadline that does not depend on the target
    usable = e.cme_dt.notna() & (e.cme_dt <= 60)
    e["cme_v"] = e.cme_v_catalogue.where(usable); e["cme_w"] = e.cme_w_catalogue.where(usable)
    e["issue"] = e.t_end
    e["lead_min"] = (e.begin - e.issue).dt.total_seconds() / 60
    n = pd.read_csv(R.parents[2] / "struminsky/paper/sep_forecast_site/data/sep_events_noaa.csv")
    b = pd.to_datetime(n.begin_utc, utc=True).sort_values().to_numpy()
    idx = np.searchsorted(b, e.t_start.to_numpy()) - 1
    e["hours_since_sep"] = np.where(idx >= 0, (e.t_start.to_numpy() - b[np.clip(idx, 0, None)]) / np.timedelta64(1, "h"), np.nan)
    e["prior_sep_72h"] = (e.hours_since_sep <= 72).astype(float)
    # direction features: signed Parker-frame azimuth offset, out-of-ecliptic angle, measured-spiral connection
    e["imf_dphi"] = e.imf_psi_meas_deg - e.imf_psi_parker_deg
    e["connect_imf"] = np.cos(np.radians(e.lon_w - e.imf_psi_meas_deg * (foot / e.imf_psi_parker_deg).clip(0.5, 2)))
    w = table(R / "waves_rad1_hapi.txt", ["noaa", "w_st", "w_npre", "w_npost", "w_bg", "w_int", "w_pk", "w_t3"])
    e = e.merge(w, on="noaa_id", how="left")
    e["III_int"] = np.log10(e.w_int.where(e.w_int > 0))   # log excess 0.9-1.04 MHz power integrated from t_s-10 min to the flare end
    mw = pd.read_csv(R / "microwave_features.csv")  # RSTN 2695 MHz excess over the 30-min pre-flare median, flare start to flare end (sfu min)
    e = e.merge(mw, on="noaa_id", how="left")
    e["mw_log"] = np.log10(e.mw2695.clip(lower=1))
    e = e.merge(pd.read_csv(R / "ephin_features.csv"), on="noaa_id", how="left")  # SOHO/EPHIN 25-41 MeV proton rise, flare start to flare end
    e.to_csv(R / "events_final.tmp", index=False); (R / "events_final.tmp").replace(R / "events_final.csv")
    pd.DataFrame(log, columns=["step", "n"]).to_csv(R / "selection_log.csv", index=False)
    print(pd.DataFrame(log).to_string(header=False, index=False))
    cols = ["delay_min", "log_flux", "log_rise", "log_dur", "connect", "sep_angle", "sw_v", "imf_align_mean", "imf_delta_deg", "imf_stability", "imf_theta_deg", "imf_polarity", "cme_v", "cme_w", "DE1_rise", "DE4_rise", "P5p_rise", "J10_bg", "prior_sep_72h", "lead_min"]
    print(e[cols].describe().T[["count", "mean", "min", "50%", "max"]].round(2).to_string())
    print(e.groupby("cycle").size().to_dict(), e.groupby("year").size().to_dict())


main()
