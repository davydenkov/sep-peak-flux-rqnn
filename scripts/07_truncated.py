#!/usr/bin/env python3
"""Exponent of the peak-flux/fluence law allowing for the 10 pfu selection threshold, and scores by time of maximum."""
from paths import *
from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr
D = RES
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv", parse_dates=["t_peak"]); F = pd.read_csv(D / "forecasts_loyo.csv")
y = np.log10(e.peak_pfu.to_numpy()); ok = q.x_int.notna().to_numpy(); x = np.log10(q.x_int.to_numpy())
def nll(t, xx, yy):
    a, b, s = t[0], t[1], np.exp(t[2]); m = a + b * xx
    return -np.sum(norm.logpdf(yy, m, s) - norm.logsf(1.0, m, s))
def fit(xx, yy):
    return minimize(nll, [2.5, .8, -.3], args=(xx, yy), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6)).x
t = fit(x[ok], y[ok]); rng = np.random.default_rng(4); xx, yy = x[ok], y[ok]; bs = []
for _ in range(400):
    i = rng.integers(0, len(xx), len(xx)); bs.append(fit(xx[i], yy[i]))
bs = np.array(bs); print("truncated a,b,s", t[0], t[1], np.exp(t[2]), "b CI", np.percentile(bs[:, 1], [2.5, 97.5]), "a CI", np.percentile(bs[:, 0], [2.5, 97.5]))
print("balch intercept", 1 - .82 * np.log10(.00987))
tmax = pd.to_datetime(e.tmax, utc=True, format="mixed"); rise = (tmax - q.t_peak).dt.total_seconds() / 3600
print(rise.describe().round(1).to_dict(), "rho rise vs y", spearmanr(rise[ok], y[ok]))
R = {}
for n, m in (("<=12 h", ok & (rise <= 12)), ("12-24 h", ok & (rise > 12) & (rise <= 24)), (">24 h", ok & (rise > 24))):
    R[n] = {"N": int(m.sum()), "median_pfu": float(np.median(e.peak_pfu[m]))}
    for k in ("clim:none", "balch", "balch_fit", "rqnn:G3_full", "rf:G3_full"):
        d = F[k].to_numpy()[m] - y[m]; R[n][k] = (round(np.abs(d).mean(), 3), round(d.mean(), 3))
print(pd.DataFrame(R).T.to_string()); json.dump({"trunc": [float(t[0]), float(t[1]), float(np.exp(t[2]))], "b_ci": list(map(float, np.percentile(bs[:, 1], [2.5, 97.5]))), "by_rise": R}, open(D / "truncated.json", "w"), indent=1)
