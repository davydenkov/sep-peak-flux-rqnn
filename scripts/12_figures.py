#!/usr/bin/env python3
"""Figures of the Solar Physics manuscript. Each is written as EPS and PDF; every series also differs by marker or line style."""
from paths import *
from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = RES; O = FIG
C1, C2, C3, C4, INK, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "axes.edgecolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE, "axes.labelcolor": INK, "legend.frameon": False, "ps.fonttype": 42, "pdf.fonttype": 42})
e = pd.read_csv(INP / "events_final.csv"); e["begin"] = pd.to_datetime(e.begin, utc=True, format="mixed"); e = e.sort_values("begin").reset_index(drop=True)
q = pd.read_csv(D / "xrs_features.csv", parse_dates=["t_peak"]); F = pd.read_csv(D / "forecasts_loyo.csv"); Sx = pd.read_csv(D / "forecasts_same_inputs.csv"); T = pd.read_csv(D / "scores_loyo.csv"); TR = json.load(open(D / "truncated.json"))
ok = q.x_int.notna().to_numpy(); y = F.y.to_numpy(); lx = np.log10(q.x_int.to_numpy()); cyc = e.cycle.to_numpy()
MK = {23: ("o", C1), 24: ("s", C2), 25: ("^", C3)}
def save(f, n):
    f.savefig(O / f"{n}.pdf", bbox_inches="tight"); f.savefig(O / f"{n}.eps", bbox_inches="tight"); plt.close(f)

# Figure 1: the scaling law
f, a = plt.subplots(figsize=(5.0, 3.5))
for c, (m, col) in MK.items():
    s = ok & (cyc == c); a.scatter(10 ** lx[s], 10 ** y[s], marker=m, s=18, facecolor="none" if c == 23 else col, edgecolor=col, lw=.8, label=f"Cycle {c} ({s.sum()})", zorder=3)
xx = np.linspace(-3.4, .5, 50)
a.plot(10 ** xx, 10 ** (1 + .82 * (xx - np.log10(.00987))), color=INK, lw=1.6, label="Balch (2008)")
b = np.polyfit(lx[ok], y[ok], 1); a.plot(10 ** xx, 10 ** np.polyval(b, xx), color=MUTE, lw=1.3, ls="--", label=f"Least squares, exponent {b[0]:.2f}")
a.plot(10 ** xx, 10 ** (TR["trunc"][0] + TR["trunc"][1] * xx), color=MUTE, lw=1.3, ls=":", label=f"Truncated fit, exponent {TR['trunc'][1]:.2f}")
Qd = json.load(open(D / "quad.json"))["b"]; a.plot(10 ** xx, 10 ** (Qd[0] + Qd[1] * xx + Qd[2] * xx**2), color=C4, lw=1.5, ls="-.", label="Parabolic fit")
a.axhline(10, color=MUTE, lw=.6); a.text(4.5e-4, 11.5, "10 pfu selection limit", color=MUTE, fontsize=7.5)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(4e-4, 3.5); a.set_ylim(6, 6e4); a.set_xlabel(r"Integrated 1$-$8 Å flux, $X_{\rm int}$ (J m$^{-2}$)"); a.set_ylabel(r"Peak $>$10 MeV flux (pfu)"); a.legend(loc="upper left", fontsize=7.5, ncol=1)
save(f, "fig1")

# Figure 3: predicted against observed
pan = (("balch", "Balch (2008), published"), ("balch_fit", "Fluence law, refitted"), ("vqr:G3x", "Quantum regression, 12 inputs"))
f, A = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
for a, (k, t) in zip(A, pan):
    p = F[k].to_numpy()
    for c, (m, col) in MK.items():
        s = ok & (cyc == c); a.scatter(10 ** y[s], 10 ** p[s], marker=m, s=13, facecolor="none" if c == 23 else col, edgecolor=col, lw=.7, label=f"Cycle {c}", zorder=3)
    g = np.array([5, 1e5]); a.plot(g, g, color=INK, lw=.9); a.plot(g, g * 3, color=MUTE, lw=.7, ls="--"); a.plot(g, g / 3, color=MUTE, lw=.7, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(7, 6e4); a.set_ylim(7, 6e4); a.set_title(t, fontsize=8.5, color=INK); a.set_xlabel("Observed peak flux (pfu)"); a.set_aspect("equal")
A[0].set_ylabel("Forecast peak flux (pfu)"); A[0].legend(loc="upper left", fontsize=7, handletextpad=.1)
save(f, "fig3")

# Figure 4: scores with intervals, all events and cycle 25
L = (("clim:none", "Climatology"), ("garcia", "Garcia (2004), published"), ("balch", "Balch (2008), published"), ("garcia_fit", "Garcia variables, refitted"), ("balch_fit", "Fluence law, refitted"), ("balch_quad", "Parabolic fluence law, refitted"), ("balch_lon_fit", "Fluence and longitude, refitted"), ("vqr:Q_balch", "Quantum regression, 2 inputs"), ("vqr:G3x", "Quantum regression, 12 inputs"), ("rf:G3x", "Random forest, 12 inputs"))
f, a = plt.subplots(figsize=(5.2, 3.3)); yy = np.arange(len(L))[::-1]
for s, m, col, off, lab in (("all", "o", C1, .14, "All events"), ("c25", "^", C2, -.14, "Cycle 25")):
    r = T[T.subset == s].set_index("model")
    for i, (k, _) in zip(yy, L):
        v = r.loc[k]; a.plot([v.lo, v.hi], [i + off] * 2, color=col, lw=1.1, solid_capstyle="round"); a.scatter(v.MAE, i + off, marker=m, s=22, color=col, zorder=3, label=lab if k == L[0][0] else None)
a.set_yticks(yy); a.set_yticklabels([n for _, n in L], color=INK); a.set_xlabel("Mean absolute error (dex)"); a.set_xlim(.3, 1.8); a.legend(loc="lower right", fontsize=7.5); a.grid(axis="x", color="#e6e5e0", lw=.5); a.set_axisbelow(True)
save(f, "fig4")

# Figure 2: forecast error against the time of maximum
tmax = pd.to_datetime(e.tmax, utc=True, format="mixed"); rise = ((tmax - q.t_peak).dt.total_seconds() / 3600).to_numpy()
f, A = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for a, (k, t, col, m) in zip(A, (("balch", "Balch (2008), published", C1, "o"), ("vqr:G3x", "Quantum regression, 12 inputs", C2, "^"))):
    d = (F[k].to_numpy() - y); a.scatter(rise[ok], d[ok], marker=m, s=13, facecolor="none", edgecolor=col, lw=.7, zorder=2)
    bins = np.array([0, 6, 12, 24, 48, 600]); c = []; md = []
    for i in range(len(bins) - 1):
        s = ok & (rise > bins[i]) & (rise <= bins[i + 1]); c.append(np.median(rise[s])); md.append(np.median(d[s]))
    a.plot(c, md, color=INK, lw=1.4, marker="D", ms=3.5, zorder=3, label="Binned median"); a.axhline(0, color=MUTE, lw=.7); a.set_xscale("log"); a.set_xlabel("Flare maximum to proton maximum (h)"); a.set_title(t, fontsize=8.5, color=INK); a.set_xlim(.6, 600)
A[0].set_ylabel(r"Forecast $-$ observed (dex)"); A[0].legend(loc="lower left", fontsize=7.5)
save(f, "fig2")
print("ok")
