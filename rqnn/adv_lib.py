"""Fit-once models with a common predict() for the property experiments (robustness, extrapolation, sample size)."""
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp05", HERE / "05_experiment.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from rqnn_torch import Circuit, fit_params
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
ALPHAS = E.ALPHAS; STAGE = E.STAGE; GROUPS = E.GROUPS; load = E.load


class RQNN:
    def __init__(self, feats, seed, maxiter=40, entangle=True):
        self.f, self.seed, self.maxiter, self.ent = feats, seed, maxiter, entangle

    def _enc(self, df):
        u = np.clip(self.q.transform(self.imp.transform(df[self.f])), 1e-3, 1 - 1e-3)
        s = np.repeat(u[:, None, :], 3, 1)
        st = np.array([STAGE[x] for x in self.f])
        for t in range(3):
            s[:, t, st > t] = .5
        return s

    def fit(self, tr, y):
        A = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(tr[self.f])
        r = RidgeCV(alphas=ALPHAS).fit(A, y)
        g = GradientBoostingRegressor(max_depth=1, n_estimators=45, learning_rate=.05, min_samples_leaf=5, subsample=.8, random_state=7).fit(A, y)
        teach = .7 * r.predict(A) + .3 * g.predict(A)
        med, sd = float(np.median(y)), float(np.std(y)) or 1.0
        self.imp = SimpleImputer(strategy="median").fit(tr[self.f])
        self.q = QuantileTransformer(n_quantiles=min(len(tr), 32), output_distribution="uniform", random_state=0).fit(self.imp.transform(tr[self.f]))
        U = self._enc(tr)
        pairs = E.entangling(self.f)
        self.c = Circuit(len(self.f), pairs, pairs, entangle=self.ent)
        self.p = fit_params(self.c, U, np.tanh((teach - med) / sd), seed=self.seed, maxiter=self.maxiter)
        with torch.no_grad():
            Z = self.c.expz(self.p, torch.as_tensor(U)).numpy()
        self.ro = RidgeCV(alphas=ALPHAS).fit(Z, y)
        return self

    def predict(self, df):
        with torch.no_grad():
            Z = self.c.expz(self.p, torch.as_tensor(self._enc(df))).numpy()
        return self.ro.predict(Z)


class Cl:
    def __init__(self, name, feats, seed):
        self.f = feats
        self.m = E.classical.__globals__  # namespace of the experiment module
        self.name, self.seed = name, seed

    def fit(self, tr, y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        s = self.seed
        est = {"ridge": lambda: RidgeCV(alphas=ALPHAS),
               "gb": lambda: GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=.04, min_samples_leaf=5, subsample=.8, random_state=s),
               "rf": lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=.5, random_state=s, n_jobs=1),
               "mlp": lambda: MLPRegressor((8,), alpha=1.0, max_iter=4000, solver="lbfgs", random_state=s)}[self.name]()
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est).fit(tr[self.f], y)
        return self

    def predict(self, df):
        return self.pipe.predict(df[self.f])


class VQR(RQNN):
    """Regression circuit: every input is known from the first step, three data re-uploading layers."""
    def _enc(self, df):
        u = np.clip(self.q.transform(self.imp.transform(df[self.f])), 1e-3, 1 - 1e-3)
        return np.repeat(u[:, None, :], 3, 1)


def make(name, feats, seed):
    if name == "vqr":
        return VQR(feats, 42 + seed)
    if name == "vqr_noent":
        return VQR(feats, 42 + seed, entangle=False)
    if name == "rqnn":
        return RQNN(feats, 42 + seed)
    if name == "rqnn_noent":
        return RQNN(feats, 42 + seed, entangle=False)
    return Cl(name, feats, seed)
