#!/usr/bin/env python3
"""Measurements of properties often attributed to quantum neural networks, on the events of this study:
extrapolation beyond the training range, small-sample behavior, classical Fourier surrogates, smoothness, trainability, and cost."""
from paths import *
import json, multiprocessing as mp, time
import numpy as np, pandas as pd
FE = {"Q2": ["log_flu", "connect"]}
NR, FR = 10, (.25, .5, 1.0)


def data(A):
    d = A.load(); q = pd.read_csv(RES / "xrs_features.csv")
    d["log_flu"] = np.log10(q.x_int.to_numpy()); d["log_flux"] = np.log10(q.x_max.to_numpy()); d["y"] = np.log10(d.peak_pfu)
    return d[d.log_flu.notna()].reset_index(drop=False)


def feats(A, g):
    return FE.get(g) or A.GROUPS[{"G0": "G0_flare", "G3": "G3_full"}[g]]


def build(A, m, g, tr, y, nseed=3):
    """Returns a predict function."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if m == "balch": return lambda df: 1 + .82 * (df.log_flu.to_numpy() - np.log10(.00987))
    if m in ("law", "quad"):
        b = np.polyfit(tr.log_flu, y, 1 if m == "law" else 2); return lambda df: np.polyval(b, df.log_flu.to_numpy())
    f = feats(A, g)
    if m == "mlp_small":   # about as many weights as the circuit of the two-input case
        ms = [make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor((6,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=s)).fit(tr[f], y) for s in range(nseed)]
        return lambda df: np.mean([x.predict(df[f]) for x in ms], axis=0)
    ms = [A.make(m, f, s).fit(tr, y) for s in range(nseed if m == "vqr" else 1)]
    return lambda df: np.mean([x.predict(df) for x in ms], axis=0)


def job_ood(a):
    import adv_lib as A
    split, m, g = a
    d = data(A)
    if split == "fluence": cut = d.log_flu.quantile(.75); tr, te = d[d.log_flu < cut], d[d.log_flu >= cut]
    else: tr, te = d[d.peak_pfu < 1000], d[d.peak_pfu >= 1000]
    y = tr.y.to_numpy(); t0 = time.time(); p = build(A, m, g, tr, y)(te); er = p - te.y.to_numpy()
    return dict(kind="ood", split=split, m=m, g=g, ntr=len(tr), nte=len(te), mae=float(np.abs(er).mean()), bias=float(er.mean()), pmax=float(p.max()), ymax_tr=float(y.max()), above=float((p > y.max()).mean()), sec=time.time() - t0)


def job_lc(a):
    import adv_lib as A
    rep, fr, m, g = a
    d = data(A); rng = np.random.default_rng(100 + rep); yrs = d.year.unique(); ty = rng.choice(yrs, int(len(yrs) * .3), replace=False)
    te, pool = d[d.year.isin(ty)], d[~d.year.isin(ty)]
    tr = pool.sample(max(int(len(pool) * fr), 8), random_state=rep); y = tr.y.to_numpy()
    f = build(A, m, g, tr, y, nseed=2)
    return dict(kind="lc", rep=rep, fr=fr, m=m, g=g, ntr=len(tr), nte=len(te), test=float(np.abs(f(te) - te.y).mean()), train=float(np.abs(f(tr) - y).mean()))


def job_smooth(m):
    import adv_lib as A
    d = data(A); y = d.y.to_numpy(); f = build(A, m, "Q2", d, y)
    g = pd.DataFrame({"log_flu": np.linspace(d.log_flu.quantile(.02), d.log_flu.quantile(.98), 200), "connect": d.connect.median()})
    p = f(g); return dict(kind="smooth", m=m, tv=float(np.abs(np.diff(p)).sum()), rough=float(np.abs(np.diff(p, 2)).sum()), curve=[float(v) for v in p[::20]])


def surrogate():
    """Classical additive Fourier model on the same quantile-mapped inputs, one year left out."""
    import adv_lib as A
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import QuantileTransformer
    d = data(A); out = {}
    for g in ("Q2", "G0", "G3"):
        f = feats(A, g)
        for K in (1, 2, 3):
            p = np.full(len(d), np.nan)
            for yr in d.year.unique():
                tr, te = d[d.year != yr], d[d.year == yr]
                im = SimpleImputer(strategy="median").fit(tr[f]); q = QuantileTransformer(n_quantiles=min(len(tr), 32), output_distribution="uniform", random_state=0).fit(im.transform(tr[f]))
                ph = lambda z: np.column_stack([fn(k * np.pi * q.transform(im.transform(z[f]))) for k in range(1, K + 1) for fn in (np.sin, np.cos)])
                p[te.index] = RidgeCV(alphas=A.ALPHAS).fit(ph(tr), tr.y).predict(ph(te))
            out[f"{g}:K{K}"] = dict(idx=[int(i) for i in d["index"]], p=[float(v) for v in p])
    return out


def trainability():
    """Variance of the gradient of <Z_0> at random parameters against the number of qubits (barren-plateau diagnostic)."""
    import torch, adv_lib as A
    from rqnn_torch import Circuit
    rows = []
    for n in (2, 4, 6, 8, 10, 12):
        pairs = [(i, i + 1) for i in range(n - 1)]; c = Circuit(n, pairs, pairs); rng = np.random.default_rng(n)
        for mode in ("random", "project"):
            G = []
            for s in range(60):
                p0 = c.init(s) if mode == "project" else np.concatenate([rng.uniform(0, 2.5, c.n_gamma), rng.uniform(-np.pi, np.pi, c.n_local + c.n_ent)])
                p = torch.tensor(p0, requires_grad=True); u = torch.as_tensor(rng.uniform(.1, .9, (1, 3, n)))
                c.expz(p, u)[0, 0].backward(); G.append(p.grad.numpy().copy())
            rows.append(dict(n=n, mode=mode, var=float(np.var(np.array(G), axis=0).mean()), npar=c.n_par))
            print(rows[-1], flush=True)
    return rows


def main():
    jobs = [(s, m, g) for s in ("fluence", "peak") for m, g in (("vqr", "G3"), ("vqr", "G0"), ("vqr", "Q2"), ("rf", "G3"), ("rf", "Q2"), ("ridge", "G3"), ("ridge", "Q2"), ("mlp", "G3"), ("mlp_small", "Q2"), ("gb", "G3"), ("balch", "-"), ("law", "-"), ("quad", "-"))]
    lc = [(r, f, m, g) for r in range(NR) for f in FR for g in ("Q2", "G0") for m in ("vqr", "ridge", "rf", "mlp_small")] + [(r, f, "law", "-") for r in range(NR) for f in FR]
    lc.sort(key=lambda j: (j[2] != "vqr", j[3] != "G0"))
    t0 = time.time(); R = []
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for r in pool.imap_unordered(job_ood, jobs): R.append(r)
        print("ood", f"{time.time() - t0:.0f}s", flush=True)
        for i, r in enumerate(pool.imap_unordered(job_lc, lc)):
            R.append(r)
            if i % 50 == 0: print("lc", i, len(lc), f"{time.time() - t0:.0f}s", flush=True)
        for r in pool.imap_unordered(job_smooth, ["vqr", "rf", "mlp_small", "quad", "ridge"]): R.append(r)
    out = {"runs": R, "surrogate": surrogate(), "trainability": trainability()}
    json.dump(out, open(RES / "claims.json", "w")); print("done", flush=True)


if __name__ == "__main__":
    main()
