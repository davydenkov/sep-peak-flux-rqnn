#!/usr/bin/env python3
"""Garcia-Balch forecasts of the >10 MeV peak flux and their paired comparison with the RQNN. All scores use the 111 events with GOES XRS coverage."""
from paths import *
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon
D = RES
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv"); pr = pd.read_csv(D / "prior_flare.csv")
assert (q.noaa_id == e.noaa_id).all()
y = np.log10(e.peak_pfu.to_numpy())
ok = q.x_int.notna().to_numpy()
lx = np.log10(q.x_int.to_numpy())
xpf = pr.x_pf.to_numpy()
alpha = np.where(xpf > 0.08, (xpf / 0.167) ** 1.146, 1.0); alpha = np.where(np.isnan(alpha), 1.0, alpha)
P = {}
P["balch"] = 1 + np.log10(alpha) + 0.82 * (lx - np.log10(0.00987))           # Balch (2008), Equations (1)-(2)
P["balch_a1"] = 1 + 0.82 * (lx - np.log10(0.00987))
EM, X, DT = q.logEM_pk.to_numpy(), q.x_max.to_numpy(), q.dur25_h.to_numpy()
P["garcia"] = -115.67 + 2.34 * EM - 1055 * X + 0.44 * DT                    # Garcia (2004), Equation (6)
lf = np.exp(-((e.lon_w.to_numpy() - 52.5) ** 2) / (2 * 12.0**2))

def ols(cols, tr, te):
    A = np.column_stack([np.ones(tr.sum())] + [c[tr] for c in cols]); b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    return np.column_stack([np.ones(te.sum())] + [c[te] for c in cols]) @ b

years = e.year.to_numpy()
REF = {"balch_fit": [lx], "balch_quad": [lx, lx**2], "balch_lon_fit": [lx, lf], "garcia_fit": [EM, X, DT]}
for k, cols in REF.items():
    p = np.full(len(y), np.nan)
    for yr in np.unique(years):
        te = (years == yr) & ok; tr = (years != yr) & ok
        if te.any(): p[te] = ols(cols, tr, te)
    P[k] = p
ch = {}
tr = (e.begin < pd.Timestamp("2020-01-01", tz="UTC")).to_numpy() & ok; te = (e.begin >= pd.Timestamp("2020-01-01", tz="UTC")).to_numpy() & ok
for k, cols in REF.items():
    p = np.full(len(y), np.nan); p[te] = ols(cols, tr, te); ch[k] = p
Jn = json.load(open(RES / "peak_predictions.json"))
for r in Jn:
    n = f"{r['m']}:{r['g']}"
    tgt = P if r["scheme"] == "loyo" else ch
    tgt.setdefault(n, np.full(len(y), np.nan))[r["idx"]] = r["p"]
for k in ("balch", "balch_a1", "garcia"):
    ch[k] = np.where(te, P[k], np.nan)

def score(p, m):
    p, t = p[m], y[m]; er = p - t; a = np.abs(er)
    rng = np.random.default_rng(5); b = a[rng.integers(0, len(a), (5000, len(a)))].mean(1)
    return dict(N=int(m.sum()), MAE=a.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), bias=er.mean(), rmse=np.sqrt((er**2).mean()), sd=er.std(ddof=1), fac=10 ** np.median(a),
                w3=(a <= np.log10(3)).mean() * 100, w10=(a <= 1).mean() * 100, r=pearsonr(p, t)[0] if np.std(p) > 0 else np.nan, rho=spearmanr(p, t)[0] if np.std(p) > 0 else np.nan, slope=np.polyfit(t, p, 1)[0])
NAMES = ["clim:none", "balch", "balch_a1", "garcia", "balch_fit", "balch_quad", "balch_lon_fit", "garcia_fit", "fluence:none", "ridge:G0_flare", "rf:G0_flare", "rqnn:G0_flare", "rqnn:G2_flare_cme", "rqnn:G3_full", "rqnn_noent:G3_full", "rqnn:G8_all_precursors", "ridge:G3_full", "rf:G3_full", "gb:G3_full", "rf:G8_all_precursors"]
c25 = e.cycle.to_numpy() == 25
SUB = {"all": ok, "c23": ok & (e.cycle.to_numpy() == 23), "c24": ok & (e.cycle.to_numpy() == 24), "c25": ok & c25, "west": ok & (e.lon_w.to_numpy() >= 0) & (e.lon_w.to_numpy() <= 90), "east": ok & (e.lon_w.to_numpy() < 0), "limb": ok & (e.lon_w.to_numpy() > 90) | ok & (e.lon_w.to_numpy() < -90), "ge100": ok & (e.peak_pfu.to_numpy() >= 100), "ge1000": ok & (e.peak_pfu.to_numpy() >= 1000)}
rows = [dict(subset=s, model=n, **score(P[n], m)) for s, m in SUB.items() for n in NAMES if not np.isnan(P[n][m]).any()]
T = pd.DataFrame(rows); T.to_csv(D / "scores_loyo.csv", index=False)
rows = [dict(model=n, **score(ch[n], ok & te)) for n in NAMES if n in ch and not np.isnan(ch[n][ok & te]).any()]
C = pd.DataFrame(rows); C.to_csv(D / "scores_chrono.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 300)
print(T[T.subset == "all"].round(3).to_string(index=False)); print(C.round(3).to_string(index=False))
print(T[T.model.isin(["balch", "balch_fit", "garcia_fit", "rqnn:G3_full", "rqnn:G0_flare", "rf:G3_full", "clim:none"]) & (T.subset != "all")][["subset", "model", "N", "MAE", "bias", "fac", "w3", "rho"]].round(3).to_string(index=False))

def paired(a, b, m, src=P):
    d = np.abs(src[a][m] - y[m]) - np.abs(src[b][m] - y[m])
    rng = np.random.default_rng(11); bs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    # cluster bootstrap over calendar years
    ys = years[m]; u = np.unique(ys); idx = {v: np.where(ys == v)[0] for v in u}
    cb = np.array([d[np.concatenate([idx[v] for v in rng.choice(u, len(u))])].mean() for _ in range(5000)])
    return dict(a=a, b=b, N=int(m.sum()), d=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), clo=np.percentile(cb, 2.5), chi=np.percentile(cb, 97.5), p_w=wilcoxon(d).pvalue, win=(d < 0).mean() * 100)
pairs = [("rqnn:G3_full", b) for b in ("balch", "garcia", "balch_fit", "balch_lon_fit", "garcia_fit", "clim:none", "rf:G3_full")] + [("rqnn:G0_flare", b) for b in ("balch", "balch_fit", "balch_lon_fit", "garcia_fit")] + [("rqnn:G8_all_precursors", "balch_fit"), ("balch_fit", "clim:none"), ("balch_lon_fit", "balch_fit"), ("garcia_fit", "balch_fit"), ("balch", "clim:none"), ("balch_quad", "balch_fit"), ("balch_quad", "balch"), ("rqnn:G3_full", "balch_quad"), ("rf:G8_all_precursors", "balch_quad")]
R = pd.DataFrame([dict(s="loyo", **paired(a, b, ok)) for a, b in pairs] + [dict(s="c25_loyo", **paired(a, b, ok & c25)) for a, b in pairs[:6] + [("balch_quad", "balch_fit"), ("balch_quad", "balch")]] + [dict(s="chrono", **paired(a, b, ok & te, ch)) for a, b in pairs[:6] + [("balch_quad", "balch_fit"), ("balch_quad", "balch")]])
R.to_csv(D / "paired.csv", index=False); print(R.round(3).to_string(index=False))
out = pd.DataFrame({"noaa_id": e.noaa_id, "y": y, **{k: P[k] for k in NAMES}}); out.to_csv(D / "forecasts_loyo.csv", index=False)
# exponent of the fluence scaling
m = ok; A = np.column_stack([np.ones(m.sum()), lx[m]]); rng = np.random.default_rng(3); bs = []
for _ in range(5000):
    i = rng.integers(0, m.sum(), m.sum()); bs.append(np.linalg.lstsq(A[i], y[m][i], rcond=None)[0])
bs = np.array(bs); b = np.linalg.lstsq(A, y[m], rcond=None)[0]
print("fit all", b, np.percentile(bs, [2.5, 97.5], axis=0).T)
for c in (23, 24, 25):
    mm = ok & (e.cycle.to_numpy() == c); print(c, mm.sum(), np.polyfit(lx[mm], y[mm], 1), pearsonr(lx[mm], y[mm])[0])
A2 = np.column_stack([np.ones(ok.sum()), lx[ok], lx[ok]**2]); b2 = np.linalg.lstsq(A2, y[ok], rcond=None)[0]; bs2 = []
for _ in range(5000):
    i = rng.integers(0, ok.sum(), ok.sum()); bs2.append(np.linalg.lstsq(A2[i], y[ok][i], rcond=None)[0])
bs2 = np.array(bs2); print("quad", b2, np.percentile(bs2, [2.5, 97.5], axis=0).T, "vertex log X", -b2[1] / 2 / b2[2], "chrono coef fit <2020", np.polyfit(lx[tr], y[tr], 2)); json.dump({"b": list(map(float, b2)), "lo": list(map(float, np.percentile(bs2, 2.5, axis=0))), "hi": list(map(float, np.percentile(bs2, 97.5, axis=0)))}, open(D / "quad.json", "w"))
print("r logX", pearsonr(lx[ok], y[ok]), spearmanr(lx[ok], y[ok])[0], "alpha!=1", (alpha != 1).sum())
