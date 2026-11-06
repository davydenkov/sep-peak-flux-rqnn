#!/usr/bin/env python3
"""Compare the PyTorch state-vector RQNN with PennyLane default.qubit on random parameters and inputs."""
import sys, time
from pathlib import Path
import numpy as np, torch
import pennylane as qml
sys.path.insert(0, str(Path(__file__).parent))
from rqnn_torch import Circuit

n, T, zz, ed = 9, 3, ((0, 5), (0, 3), (3, 7)), ((0, 5), (0, 3), (3, 7), (7, 2))
c = Circuit(n, zz, ed)
r = np.random.default_rng(1)
p = r.uniform(-1.5, 1.5, c.n_par); p[:c.n_gamma] = r.uniform(0, 2.5, c.n_gamma)
U = r.uniform(.02, .98, (6, T, n))
dev = qml.device("default.qubit", wires=n)

@qml.qnode(dev)
def f(p, s):
    a = c.n_gamma
    g = p[:a].reshape(T, n, 2); l = p[a:a + c.n_local].reshape(T, n, 2); e = p[a + c.n_local:].reshape(T, len(ed), 2)
    for t in range(T):
        u = s[t]; rx = u * np.pi / 2; rz = np.arccos(np.clip(2 * u - 1, -1, 1))
        for q in range(n):
            qml.RX(g[t, q, 0] * rx[q], wires=q); qml.RZ(g[t, q, 1] * rz[q], wires=q)
        for i, j in zz:
            qml.IsingZZ(g[t, i, 0] * g[t, j, 0] * u[i] * u[j] * np.pi, wires=[i, j])
        for q in range(n):
            qml.RY(l[t, q, 0], wires=q); qml.RZ(l[t, q, 1], wires=q)
        for k, (a_, b_) in enumerate(ed):
            qml.CRY(e[t, k, 0], wires=[a_, b_]); qml.CRZ(e[t, k, 1], wires=[a_, b_])
    return [qml.expval(qml.PauliZ(i)) for i in range(n)]

t0 = time.time(); ref = np.array([np.array(f(p, s)) for s in U]); t1 = time.time()
got = c.expz(torch.tensor(p), torch.tensor(U)).numpy(); t2 = time.time()
print("max |dZ| =", np.abs(ref - got).max(), f"pennylane {t1 - t0:.2f}s torch {t2 - t1:.3f}s")
