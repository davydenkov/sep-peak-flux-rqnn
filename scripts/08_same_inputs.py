#!/usr/bin/env python3
"""Scores of the RQNN and classical models given exactly the X-ray quantities of Balch (2008) and Garcia (2004)."""
from paths import *
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
D = RES
F = pd.read_csv(D / "forecasts_loyo.csv"); y = F.y.to_numpy(); n = len(y)
J = json.load(open(D / "rqnn_gb_predictions.json")) + [dict(r, m="vqr") for r in json.load(open(D / "vqc_predictions.json")) if r["m"] == "vqc3"]
P, CH = {}, {}
for r in J:
    t = P if r["scheme"] == "loyo" else CH
    t.setdefault(f"{r['m']}:{r['g']}", np.full(n, np.nan))[r["idx"]] = r["p"]
ok = F.balch.notna().to_numpy()
for k in ("balch", "balch_fit", "balch_quad", "balch_lon_fit", "garcia_fit", "clim:none"):
    P[k] = F[k].to_numpy()
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
c25 = (e.cycle == 25).to_numpy() & ok
def sc(k, m=ok, src=P):
    p = src[k][m]; a = np.abs(p - y[m]); return a.mean(), (p - y[m]).mean(), 10 ** np.median(a), (a <= np.log10(3)).mean() * 100, np.corrcoef(p, y[m])[0, 1]
def pair(a, b, m=ok):
    d = np.abs(P[a][m] - y[m]) - np.abs(P[b][m] - y[m]); rng = np.random.default_rng(13); s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5), wilcoxon(d).pvalue
rows = []
G = (("Q_balch", "Fluence, connection", 2), ("Q_garcia", "Emission measure, peak flux, duration", 3), ("Q_both", "All five X-ray and connection inputs", 5), ("G0x", "Six flare inputs, measured fluence", 6), ("G3x", "Twelve inputs, measured fluence", 12))
for g, lab, nq in G:
    v = [sc(f"{m}:{g}") for m in ("vqr", "ridge", "rf")]; c = sc(f"vqr:{g}", c25)
    rows.append(f"{lab} & {nq} & {v[0][0]:.3f} & {v[1][0]:.3f} & {v[2][0]:.3f} & {c[0]:.3f} \\\\")
    print(g, [round(x, 3) for x in v[0]], "ridge", round(v[1][0], 3), "rf", round(v[2][0], 3), "c25", round(c[0], 3), "bias c25", round(c[1], 3))
fmt = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
M = {}
for name, a, b in (("QBalchVsLon", "vqr:Q_balch", "balch_lon_fit"), ("QGarVsFit", "vqr:Q_garcia", "garcia_fit"), ("QBothVsFit", "vqr:Q_both", "balch_fit"), ("GzVsFit", "vqr:G0x", "balch_fit"), ("GtVsFit", "vqr:G3x", "balch_fit"), ("GtVsBalch", "vqr:G3x", "balch"), ("GtVsOld", "vqr:G3x", "rqnn:G3x"), ("GzVsOld", "vqr:G0x", "rqnn:G0x"), ("GtVsRidge", "vqr:G3x", "ridge:G3x"), ("GtVsRf", "vqr:G3x", "rf:G3x"), ("QBalchVsRidge", "vqr:Q_balch", "ridge:Q_balch"), ("QBalchVsPub", "vqr:Q_balch", "balch"), ("QBalchVsQuad", "vqr:Q_balch", "balch_quad"), ("GzVsQuad", "vqr:G0x", "balch_quad"), ("GtVsQuad", "vqr:G3x", "balch_quad"), ("RfQVsQuad", "rf:Q_balch", "balch_quad"), ("QBalchVsFit", "vqr:Q_balch", "balch_fit"), ("QBalchVsRf", "vqr:Q_balch", "rf:Q_balch"), ("RfQVsLon", "rf:Q_balch", "balch_lon_fit"), ("GzVsBalch", "vqr:G0x", "balch")):
    r = pair(a, b); M[name] = f"{fmt(r[0])}$\\,$[{fmt(r[1])},\\,{fmt(r[2])}]"; M[name + "P"] = f"{r[3]:.2f}"; print(name, np.round(r, 3))
for name, a, b in (("GtVsBalchCyc", "vqr:G3x", "balch"), ("GtVsFitCyc", "vqr:G3x", "balch_fit"), ("QBalchVsPubCyc", "vqr:Q_balch", "balch"), ("QBalchVsFitCyc", "vqr:Q_balch", "balch_fit")):
    r = pair(a, b, c25); M[name] = f"{fmt(r[0])}$\\,$[{fmt(r[1])},\\,{fmt(r[2])}]"; print(name, np.round(r, 3))
for g, k in (("Q_balch", "QBalch"), ("Q_garcia", "QGar"), ("Q_both", "QBoth"), ("G0x", "Gz"), ("G3x", "Gt")):
    v = sc(f"vqr:{g}"); M[k + "Mae"] = f"{v[0]:.3f}"; M[k + "Fac"] = f"{v[2]:.1f}"; M[k + "R"] = f"{v[4]:.2f}"; M[k + "Bias"] = f"{v[1]:.2f}".replace("-", "$-$"); M[k + "Cyc"] = f"{sc(f'vqr:{g}', c25)[0]:.3f}"
tm = {}
for r in J:
    if r["scheme"] == "loyo": tm.setdefault((r["m"], r["g"]), []).append((r["train_mae"], len(r["idx"])))
gap = {k: np.average([a for a, _ in v], weights=[w for _, w in v]) for k, v in tm.items()}
for g in ("G3x", "Q_balch"):
    for m in ("vqr", "rf", "ridge"):
        print("gap", g, m, round(sc(f"{m}:{g}")[0] - gap[(m, g)], 3))
M["RfQMae"] = f"{sc('rf:Q_balch')[0]:.3f}"; M["RidgeQMae"] = f"{sc('ridge:Q_balch')[0]:.3f}"
M["QCycBias"] = f"{sc('vqr:Q_balch', c25)[1]:+.2f}".replace("+", "$+$").replace("-", "$-$")
M["GapQ"] = f"{sc('vqr:Q_balch')[0] - gap[('vqr', 'Q_balch')]:.2f}"; M["GapQRf"] = f"{sc('rf:Q_balch')[0] - gap[('rf', 'Q_balch')]:.2f}"; M["RfQCyc"] = f"{sc('rf:Q_balch', c25)[0]:.3f}"
M["GapRq"] = f"{sc('vqr:G3x')[0] - gap[('vqr', 'G3x')]:.2f}"; M["GapRf"] = f"{sc('rf:G3x')[0] - gap[('rf', 'G3x')]:.2f}"; M["GapRidge"] = f"{sc('ridge:G3x')[0] - gap[('ridge', 'G3x')]:.2f}"
json.dump({"macros": M, "table": "\n".join(rows) + "\n"}, open(D / "same_inputs.json", "w"), indent=1)
out = pd.DataFrame({"noaa_id": F.noaa_id, **{k: v for k, v in P.items() if k.endswith(("G3x", "G0x", "Q_balch", "Q_garcia", "Q_both"))}}); out.to_csv(D / "forecasts_same_inputs.csv", index=False)
