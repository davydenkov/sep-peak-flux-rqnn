#!/usr/bin/env python3
"""Scores of the regression circuits against the staged RQNN with the same inputs."""
from paths import *
import json
import numpy as np, pandas as pd
F = pd.read_csv(RES / "forecasts_loyo.csv"); Sx = pd.read_csv(RES / "forecasts_same_inputs.csv"); y = F.y.to_numpy(); n = len(y); ok = F.balch.notna().to_numpy()
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True); c25 = ok & (e.cycle == 25).to_numpy()
P, NP_, TR = {}, {}, {}
for r in json.load(open(RES / "vqc_predictions.json")):
    if r["scheme"] != "loyo": continue
    k = f"{r['m']}:{r['g']}"; P.setdefault(k, np.full(n, np.nan))[r["idx"]] = r["p"]; NP_[k] = r["npar"]; TR.setdefault(k, []).append((r["train_mae"], len(r["idx"])))
for k in ("rqnn:Q_balch", "rqnn:G0x", "rqnn:G3x", "rf:Q_balch", "rf:G0x", "rf:G3x"): P[k] = Sx[k].to_numpy()
P["balch_fit"] = F.balch_fit.to_numpy(); P["balch"] = F.balch.to_numpy(); P["balch_quad"] = F.balch_quad.to_numpy()
mae = lambda k, m=ok: np.abs(P[k][m] - y[m]).mean()
fmt = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
def pair(a, b):
    d = np.abs(P[a][ok] - y[ok]) - np.abs(P[b][ok] - y[ok]); rng = np.random.default_rng(31); s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return f"{fmt(d.mean())} [{fmt(np.percentile(s, 2.5))}, {fmt(np.percentile(s, 97.5))}]"
rows, M = [], {}
for g, lab in (("Q_balch", "Fluence, connection"), ("G0x", "Six flare inputs"), ("G3x", "Twelve inputs")):
    rows.append(f"{lab} & {mae(f'rqnn:{g}'):.3f} & {mae(f'vqc3:{g}'):.3f} & {mae(f'vqc1:{g}'):.3f} & {mae(f'rf:{g}'):.3f} & {NP_[f'vqc1:{g}']} & {NP_[f'vqc3:{g}']} \\\\")
    M[f"VqThree{g.replace('_', '').replace('0', 'Z').replace('3', 'T')}"] = f"{mae(f'vqc3:{g}'):.3f}"; M[f"VqOne{g.replace('_', '').replace('0', 'Z').replace('3', 'T')}"] = f"{mae(f'vqc1:{g}'):.3f}"
    M[f"VqDiff{g.replace('_', '').replace('0', 'Z').replace('3', 'T')}"] = pair(f"vqc3:{g}", f"rqnn:{g}"); M[f"VqOneDiff{g.replace('_', '').replace('0', 'Z').replace('3', 'T')}"] = pair(f"vqc1:{g}", f"rqnn:{g}")
for k, nm in (("rqnn_ent:Q_balch", "EntStaged"), ("vqc3_ent:Q_balch", "EntVq")):
    M[nm] = f"{mae(k):.3f}"; M[nm + "Diff"] = pair(k, "vqc3:Q_balch" if k.startswith("vqc3") else "rqnn:Q_balch"); M[nm + "Cyc"] = f"{mae(k, c25):.3f}"
best = min(((mae(k), k) for k in P if k.startswith(("vqc", "rqnn"))))
M["VqBest"] = f"{best[0]:.3f}"; M["VqBestVsFit"] = pair(best[1], "balch_fit"); M["VqBestVsBalch"] = pair(best[1], "balch"); M["VqBestVsQuad"] = pair(best[1], "balch_quad")
print(best); print(json.dumps(M, indent=0, ensure_ascii=False)); print("\n".join(rows))
for k in sorted(P):
    if k.startswith(("vqc", "rqnn")): print(k, round(mae(k), 3), "c25", round(mae(k, c25), 3))
json.dump({"macros": M, "table": "\n".join(rows) + "\n"}, open(RES / "vqc.json", "w"), indent=1)
