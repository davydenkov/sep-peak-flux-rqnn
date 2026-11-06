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
key = {"clim:none": "Clim", "balch": "Balch", "garcia": "Garcia", "balch_fit": "BalchFit", "balch_quad": "Quad", "balch_lon_fit": "BalchLon", "garcia_fit": "GarciaFit", "vqr:G3x": "Rq", "rf:G3x": "Rf", "vqr:G0x": "RqZero", "rf:G8_all_precursors": "RfAll"}
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
for name, s, a, b in (("PRqBalch", "loyo", "vqr:G3x", "balch"), ("PRqGarcia", "loyo", "vqr:G3x", "garcia"), ("PRqBalchFit", "loyo", "vqr:G3x", "balch_fit"), ("PRqLon", "loyo", "vqr:G3x", "balch_lon_fit"), ("PRqGarciaFit", "loyo", "vqr:G3x", "garcia_fit"), ("PRqClim", "loyo", "vqr:G3x", "clim:none"), ("PRqRf", "loyo", "vqr:G3x", "rf:G3x"),
                      ("PFitClim", "loyo", "balch_fit", "clim:none"), ("PQuadFit", "loyo", "balch_quad", "balch_fit"), ("PQuadBalch", "loyo", "balch_quad", "balch"), ("PQuadCycBalch", "c25_loyo", "balch_quad", "balch"), ("PRqQuad", "loyo", "vqr:G3x", "balch_quad"), ("PBalchClim", "loyo", "balch", "clim:none"), ("PLonFit", "loyo", "balch_lon_fit", "balch_fit"), ("PGarFit", "loyo", "garcia_fit", "balch_fit"), ("PCycBalch", "c25_loyo", "vqr:G3x", "balch"), ("PCycFit", "c25_loyo", "vqr:G3x", "balch_fit"), ("PChronoBalch", "chrono", "vqr:G3x", "balch"), ("PChronoFit", "chrono", "vqr:G3x", "balch_fit")):
    m[name] = pr(s, a, b)
Qd = json.load(open(D / "quad.json")); b = Qd["b"]
for i, n in enumerate("ABC"):
    m[f"QuadCo{n}"] = f"{b[i]:.2f}"; m[f"QuadCo{n}Lo"] = f"{Qd['lo'][i]:.2f}"; m[f"QuadCo{n}Hi"] = f"{Qd['hi'][i]:.2f}"
m["QuadExpMid"] = f"{b[1] + 2 * b[2] * (-1):.2f}"; m["QuadExpTop"] = f"{b[1]:.2f}"; m["QuadVertex"] = f"{10 ** (-b[1] / 2 / b[2]):.3f}"
ER = pd.read_csv(D / "early_protons.csv"); EP = pd.read_csv(D / "early_protons_paired.csv")
eg = lambda h, k, c="mae": ER[(ER.h == h) & (ER.model == k)].iloc[0][c]
ev = (("Balch published", "Balch (2008), published"), ("fluence law", "Fluence law, refitted"), ("early flux, linear", "Early flux, linear"), ("fluence + early flux, linear", "Fluence and early flux, linear"), ("fluence + connection + early flux, linear", "With connection, linear"), ("forest", "Random forest"), ("small network", "Small network"), ("VQR", "VQR, 3 inputs"))
(LTX / "table_early.tex").write_text(f"Events & {eg(1, 'VQR', 'N'):.0f} & {eg(3, 'VQR', 'N'):.0f} & {eg(6, 'VQR', 'N'):.0f} & {eg(6, 'VQR', 'nbig'):.0f} \\\\\n\\midrule\n" + "\n".join(f"{lab} & {eg(1, k):.2f} & {eg(3, k):.2f} & {eg(6, k):.2f} & {eg(6, k, 'big_mae'):.2f} \\\\" for k, lab in ev) + "\n")
for h, w_ in ((1, "One"), (3, "Three"), (6, "Six")):
    r_ = EP[(EP.h == h) & (EP.a == "fluence + early flux, linear") & (EP.b == "fluence law")].iloc[0]; f_ = lambda v: f"{v:+.2f}".replace("-", "$-$").replace("+", "$+$")
    m[f"Early{w_}"] = f"{f_(r_.d)} [{f_(r_.lo)}, {f_(r_.hi)}]"; m[f"EarlyN{w_}"] = f"{eg(h, 'VQR', 'N'):.0f}"
    for k, n_ in (("fluence law", "Law"), ("fluence + early flux, linear", "Lin"), ("small network", "Net"), ("VQR", "Vq")):
        m[f"Ea{n_}{w_}"] = f"{eg(h, k):.2f}"; m[f"Ea{n_}{w_}Big"] = f"{eg(h, k, 'big_mae'):.2f}"
r_ = EP[(EP.h == 1) & (EP.a == "VQR") & (EP.b == "fluence + early flux, linear")].iloc[0]; m["EaVqVsLinOne"] = f"{r_.d:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]".replace("-", "$-$").replace("+", "$+$")
PH = json.load(open(D / "physics.json")); VQ = json.load(open(D / "vqc.json")); CL = json.load(open(D / "claims_macros.json"))
for k, v in {**PH["macros"], **VQ["macros"], **CL}.items():
    m[k] = v
(LTX / "table_rise.tex").write_text(PH["table"]); (LTX / "table_vqc.tex").write_text(VQ["table"])
for k, v in X["macros"].items():
    m[k] = v
s = S["sens"]
for k, n in (("operational scale", "SenOp"), ("background subtracted", "SenBg"), ("true irradiance (x1/0.7 all)", "SenTrue"), ("triangular proxy", "SenTri"), ("visible disk, |lon|<=90", "SenDisk"), ("X_int above Balch minimum", "SenMin"), ("balch on cool flares", "SenCool"), ("garcia cool flares", "SenGarCool"), ("garcia true flux (EM+0.155, X/0.7)", "SenGarTrue"), ("c25 no rescaling", "SenCycNo")):
    m[n + "Mae"] = f"{s[k]['MAE']:.3f}"; m[n + "Bias"] = f"{s[k]['bias']:.2f}".replace("-", "$-$"); m[n + "N"] = str(int(s[k]["N"]))
m["GarOffset"] = f"{s['garcia cool, offset']['offset']:.2f}"; m["GarEMEquiv"] = f"{s['garcia cool, offset']['EM_equiv']:.2f}"
m["TruncB"] = f"{TR['trunc'][1]:.2f}"; m["TruncLo"] = f"{TR['b_ci'][0]:.2f}"; m["TruncHi"] = f"{TR['b_ci'][1]:.2f}"; m["TruncS"] = f"{TR['trunc'][2]:.2f}"
EX = pd.read_csv(D / "extrapolation.csv").set_index(["split", "model"])
for n_, k_ in (("truncated power law", "ExTrunc"), ("truncated law + circuit residual", "ExHyb"), ("truncated law + small MLP residual", "ExMlp")):
    r_ = EX.loc[("peak>=1000", n_)]; m[k_] = f"{r_.mae:.2f}"; m[k_ + "Above"] = f"{r_.above:.0f}"
m["ExR"] = f"{EX.loc[('peak>=1000', 'power law')].r:.2f}"
for n, k in (("<=12 h", "Early"), ("12-24 h", "Mid"), (">24 h", "Late")):
    r = TR["by_rise"][n]; m[k + "N"] = str(r["N"]); m[k + "Pfu"] = f"{r['median_pfu']:.0f}"
    for mo, q in (("balch", "Balch"), ("balch_fit", "Fit"), ("vqr:G3x", "Rq")):
        m[k + q + "Mae"] = f"{r[mo][0]:.2f}"; m[k + q + "Bias"] = f"{r[mo][1]:.2f}".replace("-", "$-$")
L = S["lead"]
m["LeadPeak"] = f"{L['rqnn issue']['after_peak_min']:.0f}"; m["LeadOnset"] = f"{L['rqnn issue']['to_onset_h']:.1f}"; m["LeadMax"] = f"{L['rqnn issue']['to_max_h']:.1f}"; m["LeadBefore"] = f"{L['rqnn issue']['before_onset_pct']:.0f}"; m["LeadHalf"] = f"{L['balch t_half']['after_peak_min']:.0f}"; m["LeadGar"] = f"{L['garcia t25']['after_peak_min']:.0f}"
for q, n in (("CME speed", "ResCme"), ("CME speed (pre-2019)", "ResCmePre"), ("West longitude", "ResLon"), ("Delay", "ResDelay")):
    r = RP[RP.q == q].iloc[0]; m[n] = f"{r.rho:.2f}".replace("-", "$-$"); m[n + "P"] = f"{r.p:.3f}"; m[n + "Holm"] = f"{r.p_holm:.2f}"; m[n + "N"] = str(int(r.N))

N = pd.read_csv(D / "noise_scores.csv").set_index("model"); NP = pd.read_csv(D / "noise_paired.csv")
def npair(a, b, key, kind):
    r = NP[(NP.a == a) & (NP.b == b) & (NP.key == key) & (NP.kind == kind)].iloc[0]; f = lambda v: f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")
    return f"{f(r.d)}$\\,$[{f(r.lo)},\\,{f(r.hi)}]"
nm = {"NzBalchAllThree": ("vqr:G3x", "balch_pub:-", "('all', 0.3)", "degradation"), "NzBalchAllFive": ("vqr:G3x", "balch_pub:-", "('all', 0.5)", "degradation"), "NzBalchXrFive": ("vqr:G3x", "balch_pub:-", "('xray', 0.5)", "degradation"), "NzFitAllFive": ("vqr:G3x", "fluence_fit:-", "('all', 0.5)", "degradation"),
      "NzQuadAllFive": ("vqr:G3x", "quad:-", "('all', 0.5)", "degradation"), "NzQuadMaeFive": ("vqr:G3x", "quad:-", "('all', 0.5)", "mae"), "NzBalchMaeFive": ("vqr:G3x", "balch_pub:-", "('all', 0.5)", "mae"), "NzRidgeXrFive": ("vqr:G3x", "ridge:G3x", "('xray', 0.5)", "degradation"), "NzQbFitFive": ("vqr:Q_balch", "fluence_fit:-", "('all', 0.5)", "degradation"),
      "NzEntDegFive": ("vqr:G3x", "vqr_noent:G3x", "('all', 0.5)", "degradation"), "NzEntShiftThree": ("vqr:G3x", "vqr_noent:G3x", "('all', 0.3)", "shift"), "NzEntShiftFive": ("vqr:G3x", "vqr_noent:G3x", "('all', 0.5)", "shift"), "NzMissQuad": ("vqr:G3x", "quad:-", "('missing',)", "mae"), "NzMissBalch": ("vqr:G3x", "balch_pub:-", "('missing',)", "mae")}
for k, v in nm.items(): m[k] = npair(*v)
for k, n in (("balch_pub:-", "Balch"), ("fluence_fit:-", "Fit"), ("quad:-", "Quad"), ("vqr:G3x", "Gt"), ("vqr:Q_balch", "Qb"), ("ridge:G3x", "Ridge"), ("rf:G3x", "Rf"), ("vqr_noent:G3x", "Noent")):
    r = N.loc[k]; m[f"Nd{n}Five"] = f"{r['d_all_0.5']:.3f}"; m[f"Nd{n}Three"] = f"{r['d_all_0.3']:.3f}"; m[f"Nd{n}Miss"] = f"{r['missing']:.3f}"; m[f"Nd{n}Scale"] = f"{r['scale']:.3f}"
m["NoiseSd"] = f"{pd.Series(np.log10(pd.read_csv(D / 'xrs_features.csv').x_int.dropna())).std():.2f}"
LN = {"balch_pub:-": "Balch (2008), published", "fluence_fit:-": "Fluence law, refitted", "quad:-": "Parabolic fluence law", "garcia_fit:-": "Garcia variables, refitted", "ridge:G3x": "Ridge regression, 12 inputs", "rf:G3x": "Random forest, 12 inputs", "vqr:Q_balch": "Quantum regression, 2 inputs", "vqr:G0x": "Quantum regression, 6 inputs", "vqr:G3x": "Quantum regression, 12 inputs", "vqr_noent:G3x": "Quantum regression, no entanglement"}
(LTX / "table_noise.tex").write_text("\n".join(f"{v} & {N.loc[k].clean:.3f} & {N.loc[k]['d_all_0.3']:.3f} & {N.loc[k]['d_all_0.5']:.3f} & {N.loc[k]['d_xray_0.5']:.3f} & {N.loc[k].missing:.3f} & {N.loc[k].scale:.3f} \\\\" for k, v in LN.items()) + "\n")
m["NoiseHalf"] = f"{float(m['NoiseSd']) / 2:.2f}"
m = {k: v.replace("$\\,$[", " [").replace(",\\,", ", ") for k, v in m.items()}
(LTX / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))

LAB = {"clim:none": "Climatology (training median)", "balch": "Balch (2008), published", "garcia": "Garcia (2004), published", "balch_fit": "Fluence law, refitted", "balch_quad": "Parabolic fluence law, refitted", "balch_lon_fit": "Fluence and longitude, refitted", "garcia_fit": "Garcia variables, refitted",
       "vqr:Q_balch": "Quantum regression, 2 inputs", "vqr:G0x": "Quantum regression, 6 inputs", "vqr:G3x": "Quantum regression, 12 inputs", "rf:G3x": "Random forest, 12 inputs", "ridge:G3x": "Ridge regression, 12 inputs"}
def row(a):
    return f"{LAB[a.model]} & {a.MAE:.3f} [{a.lo:.2f}, {a.hi:.2f}] & {a.bias:+.2f} & {a.rmse:.2f} & {a.fac:.1f} & {a.w3:.0f} & {a.r:.2f} \\\\".replace("-0.", "$-$0.").replace("+0.", "$+$0.").replace("-1.", "$-$1.")
A = T[T.subset == "all"].set_index("model")
rows = []
for grp, names in (("No flare information", ["clim:none"]), ("Published, no fitting to this sample", ["balch", "garcia"]), ("Refitted, leave one year out", ["balch_fit", "balch_quad", "balch_lon_fit", "garcia_fit"]), ("Non-linear and multi-input models, leave one year out", ["vqr:Q_balch", "vqr:G0x", "vqr:G3x", "rf:G3x", "ridge:G3x"])):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{grp}}}}} \\\\")
    rows += [row(A.loc[n].rename(n).to_frame().T.assign(model=n).iloc[0]) for n in names]
rows = [r.replace("$-$0.37 \\\\", "$\\cdots$ \\\\") if r.startswith("Climatology") else r for r in rows]
(LTX / "table_main.tex").write_text("\n".join(rows) + "\n")
rows = []
for n in ("clim:none", "balch", "garcia", "balch_fit", "balch_quad", "garcia_fit", "vqr:G3x", "rf:G3x"):
    v = [T[(T.subset == s) & (T.model == n)].iloc[0] for s in ("c23", "c24", "c25")]; c = C[C.model == n].iloc[0]
    rows.append(f"{LAB[n]} & " + " & ".join(f"{x.MAE:.2f} ({x.bias:+.2f})" for x in v) + f" & {c.MAE:.2f} ({c.bias:+.2f}) \\\\")
(LTX / "table_cycles.tex").write_text("\n".join(r.replace("-0.", "$-$0.").replace("+0.", "$+$0.").replace("-1.", "$-$1.").replace("+1.", "$+$1.") for r in rows) + "\n")
(LTX / "table_same.tex").write_text(X["table"])
rows = []
for r in RP.sort_values("p").itertuples():
    rows.append(f"{r.q.replace('%', chr(92) + '%')} & {r.N} & {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] & {r.p:.3f} & {r.p_holm:.2f} \\\\".replace("-0.", "$-$0.").replace("+0.", "$+$0."))
(LTX / "table_resid.tex").write_text("\n".join(rows) + "\n")
print(len(m), "macros")
