#!/usr/bin/env python3
"""Can a model forecast a peak above everything in its training set? Bounded circuits against physics-anchored hybrids.
A hybrid keeps a power law for the extrapolating part and lets a bounded model correct the residual."""
from paths import *
import json
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from scipy.stats import norm
import adv_lib as A
from rqnn_torch import Circuit, fit_params
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
d = A.load(); q = pd.read_csv(RES / "xrs_features.csv"); d["log_flu"] = np.log10(q.x_int.to_numpy()); d["y"] = np.log10(d.peak_pfu); d = d[d.log_flu.notna()].reset_index(drop=True)
def tobit(x, y, hi):
    """Power law fitted to a sample truncated at 10 pfu below and at `hi` above."""
    def nll(t):
        m, s = t[0] + t[1] * x, np.exp(t[2]); return -np.sum(norm.logpdf(y, m, s) - np.log(np.clip(norm.cdf(hi, m, s) - norm.cdf(1.0, m, s), 1e-12, None)))
    return minimize(nll, [2.5, .8, -.3], method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-4, fatol=1e-6)).x
def circuit_residual(xtr, ctr, rtr, xte, cte, bounds, entangle=True):
    """Bounded correction from a 2-qubit circuit. Linear (not quantile) encoding: inputs outside the training range still move the angle."""
    lo, hi = bounds; enc = lambda x, c: np.repeat(np.clip(np.column_stack([(x - lo) / (hi - lo), (c + 1) / 2]), 0, 1.5)[:, None, :], 3, 1)
    U, V = enc(xtr, ctr), enc(xte, cte); ps = []
    for s in range(3):
        cc = Circuit(2, [(0, 1)], [(0, 1)], entangle=entangle); p = fit_params(cc, U, np.tanh(rtr / max(rtr.std(), 1e-6)), seed=42 + s, maxiter=40)
        with torch.no_grad(): Z, W = cc.expz(p, torch.as_tensor(U)).numpy(), cc.expz(p, torch.as_tensor(V)).numpy()
        ps.append(RidgeCV(alphas=A.ALPHAS).fit(Z, rtr).predict(W))
    return np.mean(ps, axis=0)
rows = []
for name, tr, te, top in (("peak>=1000", d[d.peak_pfu < 1000], d[d.peak_pfu >= 1000], 3.0), ("top fluence", d[d.log_flu < d.log_flu.quantile(.75)], d[d.log_flu >= d.log_flu.quantile(.75)], 9.0)):
    x, y, c = tr.log_flu.to_numpy(), tr.y.to_numpy(), tr.connect.to_numpy(); xt, yt, ct = te.log_flu.to_numpy(), te.y.to_numpy(), te.connect.to_numpy()
    b = np.polyfit(x, y, 1); t = tobit(x, y, top); B = 1 + .82 * (xt - np.log10(.00987)); B0 = 1 + .82 * (x - np.log10(.00987))
    P = {"Balch published": B, "power law": np.polyval(b, xt), "truncated power law": t[0] + t[1] * xt}
    law, lawt = t[0] + t[1] * x, t[0] + t[1] * xt; r = y - law; bd = (x.min(), x.max())
    P["truncated law + circuit residual"] = lawt + circuit_residual(x, c, r, xt, ct, bd)
    P["truncated law + circuit, no entanglement"] = lawt + circuit_residual(x, c, r, xt, ct, bd, entangle=False)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.column_stack([x, c]))
    P["truncated law + small MLP residual"] = lawt + np.mean([MLPRegressor((6,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=s).fit(sc.transform(np.column_stack([x, c])), r).predict(sc.transform(np.column_stack([xt, ct]))) for s in range(3)], axis=0)
    for k, p in P.items():
        e = p - yt; rows.append(dict(split=name, model=k, n=len(te), mae=np.abs(e).mean(), bias=e.mean(), pmax=10 ** p.max(), above=(p > y.max()).mean() * 100, r=np.corrcoef(p, yt)[0, 1]))
    print(name, "train max pfu", round(10 ** y.max()), "truncated exponent", round(t[1], 2), "obs max", round(10 ** yt.max()))
R = pd.DataFrame(rows); R.to_csv(RES / "extrapolation.csv", index=False); print(R.round(3).to_string(index=False))
