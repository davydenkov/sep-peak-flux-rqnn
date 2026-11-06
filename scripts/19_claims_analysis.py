#!/usr/bin/env python3
"""Tables of the measurements in results/claims.json."""
from paths import *
import json
import numpy as np, pandas as pd
C = json.load(open(RES / "claims.json")); R = pd.DataFrame(C["runs"])
F = pd.read_csv(RES / "forecasts_loyo.csv"); Sx = pd.read_csv(RES / "forecasts_same_inputs.csv"); y = F.y.to_numpy(); ok = F.balch.notna().to_numpy()
pd.set_option("display.width", 200, "display.max_rows", 200)
o = R[R.kind == "ood"][["split", "m", "g", "ntr", "nte", "mae", "bias", "pmax", "ymax_tr", "above", "sec"]].sort_values(["split", "mae"]); print(o.round(3).to_string(index=False))
l = R[R.kind == "lc"]; t = l.groupby(["g", "m", "fr"])[["ntr", "test", "train"]].mean(); t["gap"] = t.test - t.train; print(t.round(3).to_string())
piv = l.pivot_table(index=["rep", "g"], columns=["m", "fr"], values="test")
for g in ("Q2", "G0"):
    for fr in (.25, .5, 1.0):
        for b in ("ridge", "rf", "mlp_small"):
            d = (l[(l.g == g) & (l.fr == fr) & (l.m == "vqr")].set_index("rep").test - l[(l.g == g) & (l.fr == fr) & (l.m == b)].set_index("rep").test)
            print(g, fr, "rqnn-" + b, round(d.mean(), 3), "+-", round(d.std(ddof=1) / np.sqrt(len(d)), 3), "wins", int((d < 0).sum()), "/", len(d))
s = R[R.kind == "smooth"][["m", "tv", "rough"]]; print(s.round(3).to_string(index=False))
T = pd.DataFrame(C["trainability"]); print(T.pivot(index="n", columns="mode", values="var").to_string()); print(T.groupby("mode").apply(lambda d: np.polyfit(d.n, np.log10(d["var"]), 1)[0]))
M = {}
rng = np.random.default_rng(41)
fm = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
for k, v in C["surrogate"].items():
    p = np.full(len(y), np.nan); p[v["idx"]] = v["p"]; g, K = k.split(":"); ref = {"Q2": "vqr:Q_balch", "G0": "vqr:G0x", "G3": "vqr:G3x"}[g]
    d = np.abs(Sx[ref].to_numpy()[ok] - y[ok]) - np.abs(p[ok] - y[ok]); s_ = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    print(k, "MAE", round(np.abs(p[ok] - y[ok]).mean(), 3), "rqnn-this", round(d.mean(), 3), np.round(np.percentile(s_, [2.5, 97.5]), 3)); nm = "Sur" + {"Q2": "Two", "G0": "Six", "G3": "Twelve"}[g] + {"K1": "One", "K2": "Har", "K3": "Tri"}[K]; M[nm] = f"{np.abs(p[ok] - y[ok]).mean():.3f}"; M[nm + "D"] = f"{fm(d.mean())} [{fm(np.percentile(s_, 2.5))}, {fm(np.percentile(s_, 97.5))}]"
json.dump(M, open(RES / "claims_macros.json", "w"), indent=1)

# small MLP with one year left out, and the Fourier surrogate against the relations
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(RES / "xrs_features.csv"); X = np.column_stack([np.log10(q.x_int), e.connect]); pm = np.full(len(y), np.nan)
for yr in e.year.unique():
    te = ok & (e.year == yr).to_numpy(); tr = ok & (e.year != yr).to_numpy()
    if te.any(): pm[te] = np.mean([make_pipeline(StandardScaler(), MLPRegressor((6,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=s)).fit(X[tr], y[tr]).predict(X[te]) for s in range(3)], axis=0)
s1 = C["surrogate"]["Q2:K1"]; ps = np.full(len(y), np.nan); ps[s1["idx"]] = s1["p"]
def pr(a, b):
    d = np.abs(a[ok] - y[ok]) - np.abs(b[ok] - y[ok]); s_ = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return f"{fm(d.mean())} [{fm(np.percentile(s_, 2.5))}, {fm(np.percentile(s_, 97.5))}]"
rq = Sx["vqr:Q_balch"].to_numpy()
M.update(MlpQ=f"{np.abs(pm[ok] - y[ok]).mean():.3f}", RqVsMlp=pr(rq, pm), RqVsSur=pr(rq, ps), SurVsBalch=pr(ps, F.balch.to_numpy()), SurVsFit=pr(ps, F.balch_fit.to_numpy()), SurVsQuad=pr(ps, F.balch_quad.to_numpy()), SurR=f"{np.corrcoef(ps[ok], y[ok])[0, 1]:.2f}")
g = lambda s, m, gg, c: R[(R.kind == "ood") & (R.split == s) & (R.m == m) & (R.g == gg)].iloc[0][c]
lcv = lambda gg, m, fr, c: l[(l.g == gg) & (l.m == m) & (l.fr == fr)][c].mean()
for n, (s, m, gg) in {"OPkRq": ("peak", "vqr", "Q2"), "OPkBalch": ("peak", "balch", "-"), "OPkQuad": ("peak", "quad", "-"), "OPkRf": ("peak", "rf", "Q2"), "OFlRq": ("fluence", "vqr", "Q2"), "OFlRqSix": ("fluence", "vqr", "G0"), "OFlBalch": ("fluence", "balch", "-"), "OFlLaw": ("fluence", "law", "-"), "OFlRf": ("fluence", "rf", "Q2"), "OFlGb": ("fluence", "gb", "G3")}.items():
    M[n] = f"{g(s, m, gg, 'mae'):.2f}"; M[n + "Max"] = f"{10 ** g(s, m, gg, 'pmax'):.0f}"
M["OPkTrainMax"] = f"{10 ** g('peak', 'vqr', 'Q2', 'ymax_tr'):.0f}"; M["OAbove"] = f"{R[(R.kind == 'ood') & R.m.isin(['vqr'])].above.max() * 100:.0f}"
for n, (gg, m) in {"LcRq": ("Q2", "vqr"), "LcRidge": ("Q2", "ridge"), "LcRf": ("Q2", "rf"), "LcMlp": ("Q2", "mlp_small"), "LcLaw": ("-", "law"), "LcRqSix": ("G0", "vqr"), "LcRfSix": ("G0", "rf")}.items():
    M[n + "Small"] = f"{lcv(gg, m, .25, 'test'):.3f}"; M[n + "Full"] = f"{lcv(gg, m, 1.0, 'test'):.3f}"; M[n + "Gap"] = f"{lcv(gg, m, 1.0, 'test') - lcv(gg, m, 1.0, 'train'):.3f}"
sm = R[R.kind == "smooth"].set_index("m")
for n, m in (("SmRq", "vqr"), ("SmRf", "rf"), ("SmMlp", "mlp_small"), ("SmQuad", "quad")): M[n] = f"{sm.loc[m].rough:.2f}"
tv = T.set_index(["mode", "n"])["var"]; M["GvTwo"] = f"{tv[('project', 2)]:.3f}"; M["GvTwelve"] = f"{tv[('project', 12)]:.4f}"; M["GvRand"] = f"{tv[('random', 12)]:.3f}"
M["GvFactor"] = f"{10 ** (-np.polyfit(T[T['mode'] == 'project'].n, np.log10(T[T['mode'] == 'project']['var']), 1)[0]):.1f}"
M["SecRq"] = f"{g('peak', 'vqr', 'G3', 'sec'):.0f}"; M["SecRf"] = f"{g('peak', 'rf', 'G3', 'sec'):.1f}"
print({k: M[k] for k in ("MlpQ", "RqVsMlp", "RqVsSur", "SurVsBalch", "SurVsFit", "SurVsQuad", "SurR", "GvFactor", "OPkRqMax", "OPkBalchMax", "OPkTrainMax")})
json.dump(M, open(RES / "claims_macros.json", "w"), indent=1)
