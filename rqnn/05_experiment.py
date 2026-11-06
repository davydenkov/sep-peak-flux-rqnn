#!/usr/bin/env python3
"""Leave-one-year-out and chronological evaluation of the onset-delay models with and without the IMF-direction inputs.
Target: log10(NOAA 10-pfu onset - SXR peak, min). All scalers, teachers and readouts are fitted inside the training fold only.
Usage: 05_experiment.py [--seeds 3] [--maxiter 40] [--tag main]"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from rqnn_torch import Circuit, fit_params

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
R = Path(__file__).resolve().parents[1] / "data" / "input"
ALPHAS = np.logspace(-2, 3, 16)

# causal stage of each input: 0 interplanetary state before the flare, 1 flare, 2 eruption / L1 response
STAGE = {"connect": 0, "prior_sep_72h": 0, "imf_align_mean": 0, "imf_theta_abs": 0, "imf_stability": 0,
         "log_flux": 1, "log_flu": 1, "log_rise": 1, "log_dur": 1, "cme_v": 2, "cme_w": 2, "DE4_rise": 2, "III_int": 2, "mw_log": 2, "Rp_P25": 2}
BASE = ["connect", "log_flux", "log_flu", "log_rise", "log_dur", "prior_sep_72h"]
ERUPT = ["cme_v", "cme_w", "DE4_rise"]
IMF = ["imf_align_mean", "imf_stability", "imf_theta_abs"]
GROUPS = {"G0_flare": BASE, "G1_flare_imf": BASE + IMF, "G2_flare_cme": BASE + ERUPT, "G3_full": BASE + ERUPT + IMF,
          "G4_flare_cme_radio": BASE + ERUPT + ["III_int"], "G5_flare_radio": BASE + ["III_int"], "G6_flare_cme_mw": BASE + ERUPT + ["mw_log"],
          "G7_flare_cme_proton": BASE + ERUPT + ["Rp_P25"], "G8_all_precursors": BASE + ERUPT + ["III_int", "mw_log", "Rp_P25"]}


def entangling(feats):
    """A-priori physical links (no tuning on the data): connection-field, connection-CME, flare profile, CME-electrons, field-stability."""
    want = [("connect", "imf_align_mean"), ("connect", "cme_v"), ("log_flux", "log_rise"), ("cme_v", "DE4_rise"), ("imf_align_mean", "imf_stability"), ("connect", "log_flux"), ("connect", "III_int"), ("log_flux", "mw_log"), ("connect", "Rp_P25")]
    idx = {f: i for i, f in enumerate(feats)}
    return [(idx[a], idx[b]) for a, b in want if a in idx and b in idx]


def encode(tr, te, feats, mode):
    """Quantile-to-uniform angles, fitted on the training fold; three re-uploading steps."""
    imp = SimpleImputer(strategy="median").fit(tr[feats])
    a, b = imp.transform(tr[feats]), imp.transform(te[feats])
    q = QuantileTransformer(n_quantiles=min(len(tr), 32), output_distribution="uniform", random_state=0).fit(a)
    ua, ub = np.clip(q.transform(a), 1e-3, 1 - 1e-3), np.clip(q.transform(b), 1e-3, 1 - 1e-3)
    st = np.array([STAGE[f] for f in feats])

    def seq(u):
        s = np.repeat(u[:, None, :], 3, 1)
        if mode == "staged":      # an input is neutral (0.5) until the step at which it physically becomes known
            for t in range(3):
                s[:, t, st > t] = .5
        elif mode == "profile":   # synthetic growth profile used by the first version of the model
            for t, k in enumerate((.35, .65, 1.0)):
                s[:, t, :] = .5 + (u - .5) * k
        return s
    return seq(ua), seq(ub)


def rqnn_predict(tr, te, feats, y, seed, maxiter, mode="staged", entangle=True, train=True):
    Xa = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    A = Xa.fit_transform(tr[feats])
    ridge = RidgeCV(alphas=ALPHAS).fit(A, y)
    gb = GradientBoostingRegressor(max_depth=1, n_estimators=45, learning_rate=.05, min_samples_leaf=5, subsample=.8, random_state=7).fit(A, y)
    teacher = .7 * ridge.predict(A) + .3 * gb.predict(A)
    med, sd = float(np.median(y)), float(np.std(y)) or 1.0
    Ua, Ub = encode(tr, te, feats, mode)
    pairs = entangling(feats)
    c = Circuit(len(feats), pairs, pairs, entangle=entangle)
    p = fit_params(c, Ua, np.tanh((teacher - med) / sd), seed=seed, maxiter=maxiter, train=train)
    with torch.no_grad():
        Za, Zb = c.expz(p, torch.as_tensor(Ua)).numpy(), c.expz(p, torch.as_tensor(Ub)).numpy()
    return RidgeCV(alphas=ALPHAS).fit(Za, y).predict(Zb)


def classical(name, tr, te, feats, y, seed=0):
    pre = [SimpleImputer(strategy="median"), StandardScaler()]
    m = {"ridge": lambda: RidgeCV(alphas=ALPHAS),
         "gb": lambda: GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=.04, min_samples_leaf=5, subsample=.8, random_state=seed),
         "rf": lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=.5, random_state=seed, n_jobs=2),
         "mlp": lambda: MLPRegressor((8,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=seed)}[name]()
    return make_pipeline(*pre, m).fit(tr[feats], y).predict(te[feats])


def folds(d, scheme):
    if scheme == "loyo":
        for yv in sorted(d.year.unique()):
            yield str(yv), d.index[d.year != yv], d.index[d.year == yv]
    else:
        yield "cycle25", d.index[d.begin < "2020-01-01"], d.index[d.begin >= "2020-01-01"]


def run_fold(job):
    torch.set_num_threads(1)
    scheme, fold, itr, ite, seeds, maxiter, groups, tag = job
    d = load()
    tr, te = d.loc[itr], d.loc[ite]
    y = tr.log_delay.to_numpy()
    rows, t0 = [], time.time()

    def put(model, grp, seed, p):
        for i, v in zip(ite, p):
            rows.append((scheme, fold, int(i), model, grp, seed, float(v)))
    put("climatology", "none", 0, np.full(len(te), np.median(y)))
    for g in groups:
        f = GROUPS[g]
        for m in ("ridge", "gb", "rf", "mlp"):
            put(m, g, 0, classical(m, tr, te, f, y))
        for s in range(seeds):
            put("rqnn", g, s, rqnn_predict(tr, te, f, y, 42 + s, maxiter))
            if g in ("G3_full", "G0_flare"):
                put("rqnn_noent", g, s, rqnn_predict(tr, te, f, y, 42 + s, maxiter, entangle=False))
                put("rqnn_random", g, s, rqnn_predict(tr, te, f, y, 42 + s, maxiter, train=False))
                put("rqnn_profile", g, s, rqnn_predict(tr, te, f, y, 42 + s, maxiter, mode="profile"))
    print(scheme, fold, len(te), f"{time.time() - t0:.0f}s", flush=True)
    return rows


def load():
    d = pd.read_csv(R / "events_final.csv")
    d["imf_theta_abs"] = d.imf_theta_deg.abs()
    d["begin"] = pd.to_datetime(d.begin, utc=True).dt.tz_localize(None)
    return d.sort_values("begin").reset_index(drop=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--maxiter", type=int, default=40); ap.add_argument("--tag", default="main")
    ap.add_argument("--groups", default=",".join(GROUPS)); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    d = load()
    print("N", len(d), flush=True)
    jobs = [(sc, f, list(i), list(j), a.seeds, a.maxiter, a.groups.split(","), a.tag) for sc in ("chrono", "loyo") for f, i, j in folds(d, sc)]
    jobs.sort(key=lambda j: -len(j[3]))
    out = []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(run_fold, jobs):
            out += r
            pd.DataFrame(out, columns=["scheme", "fold", "idx", "model", "group", "seed", "pred"]).to_csv(R / f"predictions_{a.tag}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
