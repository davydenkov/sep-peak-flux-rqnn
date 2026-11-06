#!/usr/bin/env python3
"""Electronic Supplementary Material: one row per event with the measured X-ray quantities and the forecasts."""
from paths import *
from pathlib import Path
import numpy as np, pandas as pd
D = RES
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv"); pr = pd.read_csv(D / "prior_flare.csv"); F = pd.read_csv(D / "forecasts_loyo.csv"); S = pd.read_csv(D / "forecasts_same_inputs.csv")
o = pd.DataFrame({"noaa_begin_utc": e.begin.dt.strftime("%Y-%m-%d %H:%M"), "noaa_max_utc": pd.to_datetime(e.tmax, utc=True, format="mixed").dt.strftime("%Y-%m-%d %H:%M"), "peak_pfu": e.peak_pfu, "cycle": e.cycle, "flare_class": e.hek_class, "flare_peak_utc": q.t_peak.str[:16], "lon_w_deg": e.lon_w,
                  "noaa_region": e.region, "x_max_operational_wm2": q.x_max, "x_int_jm2": q.x_int, "x_int_background_subtracted_jm2": q.x_int_bs, "duration_25pct_h": q.dur25_h, "T_MK": q.T_pk, "log10_EM_cm3": q.logEM_pk, "x_prior_jm2": pr.x_pf, "flux_rescaled_from_true": q.true_units})
for k, n in (("balch", "balch2008_published_pfu"), ("garcia", "garcia2004_published_pfu"), ("balch_fit", "fluence_refit_pfu"), ("garcia_fit", "garcia_refit_pfu"), ("vqr:G3x", "quantum_regression_12_inputs_pfu"), ("rf:G3x", "random_forest_12_inputs_pfu")):
    o[n] = (10 ** F[k]).round(1)
o["quantum_regression_2_inputs_pfu"] = (10 ** S["vqr:Q_balch"]).round(1)
o.to_csv(RES / "esm_events.csv", index=False); print(o.shape, o.isna().sum()[lambda v: v > 0].to_dict())
