#!/usr/bin/env python3
"""Peak-flux forecasts updated after the proton onset: the flare inputs plus the observed >10 MeV flux 1, 3, and 6 h after the NOAA begin time.
Only events whose maximum is still ahead at the issue time are scored. Flux source: OMNI hourly to 2017 and GOES-R afterwards, with SOHO/EPHIN filling the gaps."""
from paths import *
import json
import numpy as np, pandas as pd
import adv_lib as A
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
d = A.load(); q = pd.read_csv(RES / "xrs_features.csv"); d["log_flu"] = np.log10(q.x_int.to_numpy()); d["y"] = np.log10(d.peak_pfu); d["begin"] = pd.to_datetime(d.begin, utc=True); d["tmax"] = pd.to_datetime(d.tmax, utc=True, format="mixed")
o = pd.read_csv(INP / "omni_early_proton.csv"); o["begin"] = pd.to_datetime(o.begin, utc=True); d = d.merge(o, on="begin", how="left")
w = pd.read_csv(INP / "goes_proton_windows.csv.gz"); w["t"] = pd.to_datetime(w.t, utc=True); w = w[w.backend == "ncei"]
H = (1, 3, 6)
for i, r in d[d.begin >= pd.Timestamp("2020-01-01", tz="UTC")].iterrows():
    s = w[w.noaa_id == r.noaa_id].set_index("t").pfu_proxy
    for h in H:
        x = s[(s.index >= r.begin + pd.Timedelta(minutes=60 * h - 30)) & (s.index <= r.begin + pd.Timedelta(minutes=60 * h + 30))]
        if len(x): d.loc[i, f"J10_{h}h"] = x.mean()
# SOHO/EPHIN fills the gaps of OMNI; its 7.8-25 MeV intensity is put on the >10 MeV scale by the median offset from events with both
eph = pd.read_csv(INP / "ephin_early.csv"); d = d.merge(eph, on="noaa_id", how="left")
off = np.nanmedian(np.concatenate([np.log10(d[f"J10_{h}h"]) - np.log10(d[f"P8_{h}h"]) for h in H])); print("EPHIN offset dex", round(off, 2))
SRC = {}
for h in H:
    c = d[f"J10_{h}h"]; SRC[h] = int(c.isna().sum() - (c.isna() & d[f"P8_{h}h"].isna()).sum())
    d[f"J10_{h}h"] = c.fillna(10 ** off * d[f"P8_{h}h"])
print("filled by EPHIN", SRC)
d = d[d.log_flu.notna()].reset_index(drop=True)
def loyo(fit, tr_all, feats):
    p = pd.Series(np.nan, index=tr_all.index)
    for yr in tr_all.year.unique():
        tr, te = tr_all[tr_all.year != yr], tr_all[tr_all.year == yr]
        if len(tr) > 10: p[te.index] = fit(tr, te, feats)
    return p.to_numpy()
lin = lambda tr, te, f: LinearRegression().fit(tr[f], tr.y).predict(te[f])
def rf(tr, te, f): return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=.5, random_state=0, n_jobs=1).fit(tr[f], tr.y).predict(te[f])
def mlp(tr, te, f): return np.mean([make_pipeline(StandardScaler(), MLPRegressor((6,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=s)).fit(tr[f], tr.y).predict(te[f]) for s in range(3)], axis=0)
def vqr(tr, te, f):
    A.STAGE.update({"logJ": 2}); return np.mean([A.make("vqr", f, s).fit(tr, tr.y.to_numpy()).predict(te) for s in range(3)], axis=0)
rng = np.random.default_rng(51); rows = []; pairs = []
for h in H:
    x = d.copy(); x["logJ"] = np.log10(x[f"J10_{h}h"]); issue = x.begin + pd.Timedelta(hours=h)
    x = x[x.logJ.notna() & (x.tmax > issue)].reset_index(drop=True); y = x.y.to_numpy()
    late = ((x.tmax - x.begin).dt.total_seconds() / 3600 > 12).to_numpy()
    P = {"Balch published": (1 + .82 * (x.log_flu - np.log10(.00987))).to_numpy(), "persistence": x.logJ.to_numpy(), "fluence law": loyo(lin, x, ["log_flu"]), "early flux, linear": loyo(lin, x, ["logJ"]), "fluence + early flux, linear": loyo(lin, x, ["log_flu", "logJ"]), "fluence + connection + early flux, linear": loyo(lin, x, ["log_flu", "connect", "logJ"]),
         "forest": loyo(rf, x, ["log_flu", "connect", "logJ"]), "small network": loyo(mlp, x, ["log_flu", "connect", "logJ"]), "VQR": loyo(vqr, x, ["log_flu", "connect", "logJ"])}
    for k, p in P.items():
        m = ~np.isnan(p); a = np.abs(p[m] - y[m])
        rows.append(dict(h=h, model=k, N=int(m.sum()), mae=a.mean(), bias=(p[m] - y[m]).mean(), r=np.corrcoef(p[m], y[m])[0, 1], late_mae=np.abs(p - y)[m & late].mean(), nlate=int((m & late).sum()), big_mae=np.abs(p - y)[m & (y >= 3)].mean(), nbig=int((m & (y >= 3)).sum())))
    for a_, b_ in (("fluence + early flux, linear", "fluence law"), ("fluence + early flux, linear", "Balch published"), ("VQR", "fluence + early flux, linear"), ("VQR", "fluence law"), ("small network", "fluence + early flux, linear")):
        m = ~np.isnan(P[a_]) & ~np.isnan(P[b_]); dd = np.abs(P[a_][m] - y[m]) - np.abs(P[b_][m] - y[m]); s = dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1)
        pairs.append(dict(h=h, a=a_, b=b_, d=dd.mean(), lo=np.percentile(s, 2.5), hi=np.percentile(s, 97.5)))
R = pd.DataFrame(rows); Q = pd.DataFrame(pairs); R.to_csv(RES / "early_protons.csv", index=False); Q.to_csv(RES / "early_protons_paired.csv", index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False)); print(Q.round(3).to_string(index=False))
