#!/usr/bin/env python3
"""Regression circuits without the staged recurrence, on the same events and inputs as the RQNN.
vqc3: every input known from the first step, three data re-uploading layers. vqc1: one encoding and one trainable layer.
*_ent: the fluence and connection qubits are coupled, which the a-priori pair list of the RQNN does not do."""
from paths import *
import json, multiprocessing as mp, sys, time
import numpy as np, pandas as pd
NSEED = 3
GR = {"Q_balch": ["log_flu", "connect"], "Q_garcia": ["logEM", "log_flux", "dur25"], "Q_both": ["connect", "log_flu", "log_flux", "logEM", "dur25"]}
VAR = (("vqc3", "Q_balch"), ("vqc3", "G0x"), ("vqc3", "G3x"), ("vqc1", "Q_balch"), ("vqc1", "G0x"), ("vqc1", "G3x"), ("rqnn_ent", "Q_balch"), ("vqc3_ent", "Q_balch"), ("vqc3", "Q_garcia"), ("vqc3", "Q_both"))


def data(A):
    d = A.load(); q = pd.read_csv(RES / "xrs_features.csv")
    d["log_flu"] = np.log10(q.x_int.to_numpy()); d["log_flux"] = np.log10(q.x_max.to_numpy()); d["logEM"] = q.logEM_pk.to_numpy(); d["dur25"] = np.log10(q.dur25_h.to_numpy()); d["y"] = np.log10(d.peak_pfu)
    return d[d.log_flu.notna()]


def job(a):
    import torch, adv_lib as A
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from rqnn_torch import Circuit, fit_params

    class V(A.RQNN):
        """Same training, teacher, and readout as the RQNN; only the input schedule, depth, and pair list differ."""
        def __init__(self, feats, seed, T, staged, couple):
            super().__init__(feats, seed); self.T, self.staged, self.couple = T, staged, couple
        def _enc(self, df):
            u = np.clip(self.q.transform(self.imp.transform(df[self.f])), 1e-3, 1 - 1e-3)
            s = np.repeat(u[:, None, :], self.T, 1)
            if self.staged:
                st = np.array([A.STAGE[x] for x in self.f])
                for t in range(self.T): s[:, t, st > t] = .5
            return s
        def fit(self, tr, y):
            X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(tr[self.f])
            r = RidgeCV(alphas=A.ALPHAS).fit(X, y)
            g = GradientBoostingRegressor(max_depth=1, n_estimators=45, learning_rate=.05, min_samples_leaf=5, subsample=.8, random_state=7).fit(X, y)
            teach = .7 * r.predict(X) + .3 * g.predict(X); med, sd = float(np.median(y)), float(np.std(y)) or 1.0
            self.imp = SimpleImputer(strategy="median").fit(tr[self.f])
            self.q = QuantileTransformer(n_quantiles=min(len(tr), 32), output_distribution="uniform", random_state=0).fit(self.imp.transform(tr[self.f]))
            U = self._enc(tr); pairs = A.E.entangling(self.f) + ([(0, 1)] if self.couple else [])
            self.c = Circuit(len(self.f), pairs, pairs, n_steps=self.T, entangle=True)
            self.p = fit_params(self.c, U, np.tanh((teach - med) / sd), seed=self.seed, maxiter=self.maxiter)
            with torch.no_grad(): Z = self.c.expz(self.p, torch.as_tensor(U)).numpy()
            self.ro = RidgeCV(alphas=A.ALPHAS).fit(Z, y); self.npar = self.c.n_par
            return self
    A.STAGE.update({"logEM": 1, "dur25": 1})
    base = A.E.entangling
    def ent(feats):
        idx = {f: i for i, f in enumerate(feats)}
        return base(feats) + [(idx[u], idx[v]) for u, v in (("log_flux", "logEM"), ("logEM", "dur25")) if u in idx and v in idx]
    A.E.entangling = ent
    scheme, fold, m, g = a
    feats = GR.get(g) or A.GROUPS[{"G0x": "G0_flare", "G3x": "G3_full"}[g]]
    d = data(A)
    if scheme == "loyo": tr, te = d[d.year != fold], d[d.year == fold]
    else:
        c = pd.Timestamp("2020-01-01"); tr, te = d[d.begin < c], d[d.begin >= c]
    if not len(te): return None
    T = 1 if m == "vqc1" else 3; staged = m.startswith("rqnn"); couple = m.endswith("_ent"); y = tr.y.to_numpy()
    ms = [V(feats, 42 + s, T, staged, couple).fit(tr, y) for s in range(NSEED)]
    p = np.mean([x.predict(te) for x in ms], axis=0); pt = np.mean([x.predict(tr) for x in ms], axis=0)
    return {"scheme": scheme, "fold": str(fold), "g": g, "m": m, "idx": [int(i) for i in te.index], "p": [float(v) for v in p], "train_mae": float(np.abs(pt - y).mean()), "ntr": len(tr), "npar": int(ms[0].npar)}


def main():
    e = pd.read_csv(INP / "events_final.csv"); years = sorted(int(v) for v in e.year.unique())
    folds = [("chrono", 0)] + [("loyo", y) for y in years]
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None   # e.g. Q_garcia,Q_both adds these to an existing result file
    jobs = [(s, f, m, g) for s, f in folds for m, g in VAR if not only or g in only]; jobs.sort(key=lambda j: (j[3] != "G3x", j[2] == "vqc1"))
    print(len(jobs), flush=True); out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            if r: out.append(r)
            if i % 20 == 0: print(i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    if only: out = [r for r in json.load(open(RES / "vqc_predictions.json")) if r["g"] not in only] + out
    json.dump(out, open(RES / "vqc_predictions.json", "w")); print("done", flush=True)


if __name__ == "__main__":
    main()
