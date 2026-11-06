#!/usr/bin/env python3
"""Numbers quoted in the manuscript (macros.tex) and its tables, generated from the result files so that nothing is typed by hand."""
from paths import *
import json
from pathlib import Path
import numpy as np, pandas as pd
D = RES
T = pd.read_csv(D / "scores_loyo.csv"); C = pd.read_csv(D / "scores_chrono.csv"); R = pd.read_csv(D / "paired.csv"); X = json.load(open(D / "same_inputs.json"))
S = json.load(open(D / "sensitivity.json")); TR = json.load(open(D / "truncated.json")); RP = pd.read_csv(D / "residual_physics.csv")
m = {}
key = {"clim:none": "Clim", "balch": "Balch", "garcia": "Garcia", "balch_fit": "BalchFit", "balch_quad": "Quad", "balch_lon_fit": "BalchLon", "garcia_fit": "GarciaFit", "rqnn:G3_full": "Rq", "rf:G3_full": "Rf", "rqnn:G0_flare": "RqZero", "rf:G8_all_precursors": "RfAll", "rqnn:G8_all_precursors": "RqAll"}
for n, k in key.items():
    a = T[(T.subset == "all") & (T.model == n)].iloc[0]
    for f, s in (("MAE", "Mae"), ("lo", "Lo"), ("hi", "Hi"), ("bias", "Bias"), ("rmse", "Rmse"), ("fac", "Fac"), ("w3", "Wthree"), ("w10", "Wten"), ("r", "R"), ("rho", "Rho")):
        d = 1 if f in ("w3", "w10", "fac") else 3 if f in ("MAE", "lo", "hi") else 2
        v = a[f]
        m[f"{k}{s}"] = f"{v:.{d}f}".replace("-", "$-$") if f in ("bias",) else f"{v:.{d}f}"
    for sub, s in (("c25", "Cyc"), ("c23", "Cyt")):
        b = T[(T.subset == sub) & (T.model == n)]
        if len(b): m[f"{k}{s}Mae"] = f"{b.iloc[0].MAE:.3f}"; m[f"{k}{s}Bias"] = f"{b.iloc[0].bias:.2f}".replace("-", "$-$")
def pr(s, a, b):
    r = R[(R.s == s) & (R.a == a) & (R.b == b)].iloc[0]
    f = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
    return f"{f(r.d)}$\\,$[{f(r.lo)},\\,{f(r.hi)}]"
for name, s, a, b in (("PRqBalch", "loyo", "rqnn:G3_full", "balch"), ("PRqGarcia", "loyo", "rqnn:G3_full", "garcia"), ("PRqBalchFit", "loyo", "rqnn:G3_full", "balch_fit"), ("PRqLon", "loyo", "rqnn:G3_full", "balch_lon_fit"), ("PRqGarciaFit", "loyo", "rqnn:G3_full", "garcia_fit"), ("PRqClim", "loyo", "rqnn:G3_full", "clim:none"), ("PRqRf", "loyo", "rqnn:G3_full", "rf:G3_full"),
                      ("PFitClim", "loyo", "balch_fit", "clim:none"), ("PQuadFit", "loyo", "balch_quad", "balch_fit"), ("PQuadBalch", "loyo", "balch_quad", "balch"), ("PQuadCycBalch", "c25_loyo", "balch_quad", "balch"), ("PRqQuad", "loyo", "rqnn:G3_full", "balch_quad"), ("PBalchClim", "loyo", "balch", "clim:none"), ("PLonFit", "loyo", "balch_lon_fit", "balch_fit"), ("PGarFit", "loyo", "garcia_fit", "balch_fit"), ("PCycBalch", "c25_loyo", "rqnn:G3_full", "balch"), ("PCycFit", "c25_loyo", "rqnn:G3_full", "balch_fit"), ("PChronoBalch", "chrono", "rqnn:G3_full", "balch"), ("PChronoFit", "chrono", "rqnn:G3_full", "balch_fit")):
    m[name] = pr(s, a, b)
Qd = json.load(open(D / "quad.json")); b = Qd["b"]
for i, n in enumerate("ABC"):
    m[f"QuadCo{n}"] = f"{b[i]:.2f}"; m[f"QuadCo{n}Lo"] = f"{Qd['lo'][i]:.2f}"; m[f"QuadCo{n}Hi"] = f"{Qd['hi'][i]:.2f}"
m["QuadExpMid"] = f"{b[1] + 2 * b[2] * (-1):.2f}"; m["QuadExpTop"] = f"{b[1]:.2f}"; m["QuadVertex"] = f"{10 ** (-b[1] / 2 / b[2]):.3f}"
for k, v in X["macros"].items():
    m[k] = v
s = S["sens"]
for k, n in (("operational scale", "SenOp"), ("background subtracted", "SenBg"), ("true irradiance (x1/0.7 all)", "SenTrue"), ("triangular proxy", "SenTri"), ("visible disk, |lon|<=90", "SenDisk"), ("X_int above Balch minimum", "SenMin"), ("balch on cool flares", "SenCool"), ("garcia cool flares", "SenGarCool"), ("garcia true flux (EM+0.155, X/0.7)", "SenGarTrue"), ("c25 no rescaling", "SenCycNo")):
    m[n + "Mae"] = f"{s[k]['MAE']:.3f}"; m[n + "Bias"] = f"{s[k]['bias']:.2f}".replace("-", "$-$"); m[n + "N"] = str(int(s[k]["N"]))
m["GarOffset"] = f"{s['garcia cool, offset']['offset']:.2f}"; m["GarEMEquiv"] = f"{s['garcia cool, offset']['EM_equiv']:.2f}"
m["TruncB"] = f"{TR['trunc'][1]:.2f}"; m["TruncLo"] = f"{TR['b_ci'][0]:.2f}"; m["TruncHi"] = f"{TR['b_ci'][1]:.2f}"; m["TruncS"] = f"{TR['trunc'][2]:.2f}"
for n, k in (("<=12 h", "Early"), ("12-24 h", "Mid"), (">24 h", "Late")):
    r = TR["by_rise"][n]; m[k + "N"] = str(r["N"]); m[k + "Pfu"] = f"{r['median_pfu']:.0f}"
    for mo, q in (("balch", "Balch"), ("balch_fit", "Fit"), ("rqnn:G3_full", "Rq")):
        m[k + q + "Mae"] = f"{r[mo][0]:.2f}"; m[k + q + "Bias"] = f"{r[mo][1]:.2f}".replace("-", "$-$")
L = S["lead"]
m["LeadPeak"] = f"{L['rqnn issue']['after_peak_min']:.0f}"; m["LeadOnset"] = f"{L['rqnn issue']['to_onset_h']:.1f}"; m["LeadMax"] = f"{L['rqnn issue']['to_max_h']:.1f}"; m["LeadBefore"] = f"{L['rqnn issue']['before_onset_pct']:.0f}"; m["LeadHalf"] = f"{L['balch t_half']['after_peak_min']:.0f}"; m["LeadGar"] = f"{L['garcia t25']['after_peak_min']:.0f}"
for q, n in (("CME speed", "ResCme"), ("CME speed (pre-2019)", "ResCmePre"), ("West longitude", "ResLon"), ("Delay", "ResDelay")):
    r = RP[RP.q == q].iloc[0]; m[n] = f"{r.rho:.2f}".replace("-", "$-$"); m[n + "P"] = f"{r.p:.3f}"; m[n + "Holm"] = f"{r.p_holm:.2f}"; m[n + "N"] = str(int(r.N))

N = pd.read_csv(D / "noise_scores.csv").set_index("model"); NP = pd.read_csv(D / "noise_paired.csv")
def npair(a, b, key, kind):
    r = NP[(NP.a == a) & (NP.b == b) & (NP.key == key) & (NP.kind == kind)].iloc[0]; f = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
    return f"{f(r.d)}$\\,$[{f(r.lo)},\\,{f(r.hi)}]"
nm = {"NzBalchAllThree": ("rqnn:G3x", "balch_pub:-", "('all', 0.3)", "degradation"), "NzBalchAllFive": ("rqnn:G3x", "balch_pub:-", "('all', 0.5)", "degradation"), "NzBalchXrFive": ("rqnn:G3x", "balch_pub:-", "('xray', 0.5)", "degradation"), "NzFitAllFive": ("rqnn:G3x", "fluence_fit:-", "('all', 0.5)", "degradation"),
      "NzQuadAllFive": ("rqnn:G3x", "quad:-", "('all', 0.5)", "degradation"), "NzQuadMaeFive": ("rqnn:G3x", "quad:-", "('all', 0.5)", "mae"), "NzBalchMaeFive": ("rqnn:G3x", "balch_pub:-", "('all', 0.5)", "mae"), "NzRidgeXrFive": ("rqnn:G3x", "ridge:G3x", "('xray', 0.5)", "degradation"), "NzQbFitFive": ("rqnn:Q_balch", "fluence_fit:-", "('all', 0.5)", "degradation"),
      "NzEntDegFive": ("rqnn:G3x", "rqnn_noent:G3x", "('all', 0.5)", "degradation"), "NzEntShiftThree": ("rqnn:G3x", "rqnn_noent:G3x", "('all', 0.3)", "shift"), "NzEntShiftFive": ("rqnn:G3x", "rqnn_noent:G3x", "('all', 0.5)", "shift"), "NzMissQuad": ("rqnn:G3x", "quad:-", "('missing',)", "mae"), "NzMissBalch": ("rqnn:G3x", "balch_pub:-", "('missing',)", "mae")}
for k, v in nm.items(): m[k] = npair(*v)
for k, n in (("balch_pub:-", "Balch"), ("fluence_fit:-", "Fit"), ("quad:-", "Quad"), ("rqnn:G3x", "Gt"), ("rqnn:Q_balch", "Qb"), ("ridge:G3x", "Ridge"), ("rf:G3x", "Rf"), ("rqnn_noent:G3x", "Noent")):
    r = N.loc[k]; m[f"Nd{n}Five"] = f"{r['d_all_0.5']:.3f}"; m[f"Nd{n}Three"] = f"{r['d_all_0.3']:.3f}"; m[f"Nd{n}Miss"] = f"{r['missing']:.3f}"; m[f"Nd{n}Scale"] = f"{r['scale']:.3f}"
m["NoiseSd"] = f"{pd.Series(np.log10(pd.read_csv(D / 'xrs_features.csv').x_int.dropna())).std():.2f}"
LN = {"balch_pub:-": "Balch (2008), published", "fluence_fit:-": "Fluence law, refitted", "quad:-": "Parabolic fluence law", "garcia_fit:-": "Garcia variables, refitted", "ridge:G3x": "Ridge regression, 12 inputs", "rf:G3x": "Random forest, 12 inputs", "rqnn:Q_balch": "RQNN, 2 inputs", "rqnn:G0x": "RQNN, 6 inputs", "rqnn:G3x": "RQNN, 12 inputs", "rqnn_noent:G3x": "RQNN, no entanglement"}
(LTX / "table_noise.tex").write_text("\n".join(f"{v} & {N.loc[k].clean:.3f} & {N.loc[k]['d_all_0.3']:.3f} & {N.loc[k]['d_all_0.5']:.3f} & {N.loc[k]['d_xray_0.5']:.3f} & {N.loc[k].missing:.3f} & {N.loc[k].scale:.3f} \\\\" for k, v in LN.items()) + "\n")
m["NoiseHalf"] = f"{float(m['NoiseSd']) / 2:.2f}"
m = {k: v.replace("$\\,$[", " [").replace(",\\,", ", ") for k, v in m.items()}
(LTX / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))

LAB = {"clim:none": "Climatology (training median)", "balch": "Balch (2008), published", "garcia": "Garcia (2004), published", "balch_fit": "Fluence law, refitted", "balch_quad": "Parabolic fluence law, refitted", "balch_lon_fit": "Fluence and longitude, refitted", "garcia_fit": "Garcia variables, refitted",
       "rqnn:G0_flare": "RQNN, flare inputs (6 qubits)", "rqnn:G3_full": "RQNN, all inputs (12 qubits)", "rf:G3_full": "Random forest, all inputs", "ridge:G3_full": "Ridge regression, all inputs"}
def row(a):
    return f"{LAB[a.model]} & {a.MAE:.3f} [{a.lo:.2f}, {a.hi:.2f}] & {a.bias:+.2f} & {a.rmse:.2f} & {a.fac:.1f} & {a.w3:.0f} & {a.r:.2f} \\\\".replace("-0.", "$-$0.").replace("+0.", "$+$0.").replace("-1.", "$-$1.")
A = T[T.subset == "all"].set_index("model")
rows = []
for grp, names in (("No flare information", ["clim:none"]), ("Published, no fitting to this sample", ["balch", "garcia"]), ("Refitted, leave one year out", ["balch_fit", "balch_quad", "balch_lon_fit", "garcia_fit"]), ("Multi-input models, leave one year out", ["rqnn:G0_flare", "rqnn:G3_full", "rf:G3_full", "ridge:G3_full"])):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{grp}}}}} \\\\")
    rows += [row(A.loc[n].rename(n).to_frame().T.assign(model=n).iloc[0]) for n in names]
rows = [r.replace("$-$0.37 \\\\", "$\\cdots$ \\\\") if r.startswith("Climatology") else r for r in rows]
(LTX / "table_main.tex").write_text("\n".join(rows) + "\n")
rows = []
for n in ("clim:none", "balch", "garcia", "balch_fit", "balch_quad", "garcia_fit", "rqnn:G3_full", "rf:G3_full"):
    v = [T[(T.subset == s) & (T.model == n)].iloc[0] for s in ("c23", "c24", "c25")]; c = C[C.model == n].iloc[0]
    rows.append(f"{LAB[n]} & " + " & ".join(f"{x.MAE:.2f} ({x.bias:+.2f})" for x in v) + f" & {c.MAE:.2f} ({c.bias:+.2f}) \\\\")
(LTX / "table_cycles.tex").write_text("\n".join(r.replace("-0.", "$-$0.").replace("+0.", "$+$0.").replace("-1.", "$-$1.").replace("+1.", "$+$1.") for r in rows) + "\n")
(LTX / "table_same.tex").write_text(X["table"])
rows = []
for r in RP.sort_values("p").itertuples():
    rows.append(f"{r.q.replace('%', chr(92) + '%')} & {r.N} & {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] & {r.p:.3f} & {r.p_holm:.2f} \\\\".replace("-0.", "$-$0.").replace("+0.", "$+$0."))
(LTX / "table_resid.tex").write_text("\n".join(rows) + "\n")
print(len(m), "macros")
