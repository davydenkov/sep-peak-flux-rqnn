"""Batched exact state-vector implementation of the RQNN circuit (same gate sequence as the PennyLane model on the forecast site).
Trained by L-BFGS with autograd; validated against PennyLane in code/validate_simulator.py."""
from __future__ import annotations
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
C = torch.complex128


def _one(state, n, q, m):
    """Apply a batched 2x2 matrix m[B,2,2] to qubit q (wire 0 = most significant bit)."""
    B = state.shape[0]
    s = state.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    return torch.einsum("bij,bljr->blir", m, s).reshape(B, -1)


def _rx(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([torch.complex(c, z), torch.complex(z, -s)], -1), torch.stack([torch.complex(z, -s), torch.complex(c, z)], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(C)


def _rz(t):
    z = torch.zeros_like(t)
    a, b = torch.complex(torch.cos(t / 2), -torch.sin(t / 2)), torch.complex(torch.cos(t / 2), torch.sin(t / 2))
    o = torch.complex(z, z)
    return torch.stack([torch.stack([a, o], -1), torch.stack([o, b], -1)], -2)


class Circuit:
    def __init__(self, n, zz_pairs, edges, n_steps=3, entangle=True):
        self.n, self.zz, self.edges, self.T, self.entangle = n, tuple(zz_pairs), tuple(edges), n_steps, entangle
        idx = torch.arange(2 ** n)
        self.bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], 0).to(torch.float64)  # [n, 2^n]
        self.sign = 1 - 2 * self.bits
        self.n_gamma = n_steps * n * 2
        self.n_local = n_steps * n * 2
        self.n_ent = n_steps * len(self.edges) * 2
        self.n_par = self.n_gamma + self.n_local + self.n_ent

    def split(self, p):
        a = self.n_gamma
        g = p[:a].reshape(self.T, self.n, 2)
        l = p[a:a + self.n_local].reshape(self.T, self.n, 2)
        e = p[a + self.n_local:].reshape(self.T, len(self.edges), 2)
        return g, l, e

    def _controlled(self, state, ctl, tgt, m):
        """Controlled single-qubit gate with a scalar 2x2 matrix m."""
        n, B = self.n, state.shape[0]
        i, j = min(ctl, tgt), max(ctl, tgt)
        s = state.reshape(B, 2 ** i, 2, 2 ** (j - i - 1), 2, 2 ** (n - j - 1))
        if ctl < tgt:
            s1 = torch.einsum("ij,blkjr->blkir", m, s[:, :, 1])
            return torch.stack([s[:, :, 0], s1], 2).reshape(B, -1)
        s1 = torch.einsum("ij,bljkr->blikr", m, s[:, :, :, :, 1])
        return torch.stack([s[:, :, :, :, 0], s1], 4).reshape(B, -1)

    def expz(self, p, U):
        """p: parameters, U: [B, T, n] encoded angles in (0,1). Returns <Z_q> [B, n]."""
        g, l, e = self.split(p)
        B, n = U.shape[0], self.n
        st = torch.zeros(B, 2 ** n, dtype=C)
        st[:, 0] = 1
        for t in range(self.T):
            u = U[:, t]
            rx = u * (math.pi / 2)
            rz = torch.arccos(torch.clamp(2 * u - 1, -1, 1))
            for q in range(n):  # RZ.RX fused into one batched 2x2 matrix
                st = _one(st, n, q, _rz(g[t, q, 1] * rz[:, q]) @ _rx(g[t, q, 0] * rx[:, q]))
            if self.entangle:
                for i, j in self.zz:
                    th = g[t, i, 0] * g[t, j, 0] * u[:, i] * u[:, j] * math.pi
                    ph = self.sign[i] * self.sign[j]  # exp(-i th/2 Z Z)
                    st = st * torch.exp(torch.complex(torch.zeros_like(th)[:, None] * ph, -(th[:, None] / 2) * ph))
            for q in range(n):  # data-independent RZ.RY: one shared 2x2 matrix
                m = _rz(l[t, q, 1][None])[0] @ _ry(l[t, q, 0][None])[0]
                sh = st.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
                st = torch.einsum("ij,bljr->blir", m, sh).reshape(B, -1)
            if self.entangle:
                for k, (c, tg) in enumerate(self.edges):
                    st = self._controlled(st, c, tg, _ry(e[t, k, 0][None])[0])
                    st = self._controlled(st, c, tg, _rz(e[t, k, 1][None])[0])
        pr = (st.real ** 2 + st.imag ** 2)
        return pr @ self.sign.T

    def init(self, seed):
        r = np.random.default_rng(seed)
        return np.concatenate([1 + r.normal(0, .04, self.n_gamma), r.uniform(-.08, .08, self.n_local), r.uniform(-.08, .08, self.n_ent)])


def fit_params(circ, U, target, seed=42, maxiter=60, late_weight=3.0, delta=.12, temp=.06, train=True):
    """Same objective as the site model: asymmetric pseudo-Huber on mean_q <Z_q> vs a tanh-scaled teacher."""
    U = torch.as_tensor(U); y = torch.as_tensor(target)
    p0 = circ.init(seed)
    p = torch.tensor(p0, requires_grad=True)
    if not train:
        return p.detach()
    lo = torch.cat([torch.zeros(circ.n_gamma), torch.full((circ.n_local + circ.n_ent,), -math.pi)])
    hi = torch.cat([torch.full((circ.n_gamma,), 2.5), torch.full((circ.n_local + circ.n_ent,), math.pi)])
    opt = torch.optim.LBFGS([p], lr=.5, max_iter=maxiter, history_size=12, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-8)

    def closure():
        opt.zero_grad()
        pc = torch.max(torch.min(p, hi), lo)
        z = circ.expz(pc, U).mean(1)
        err = z - y
        w = 1 + (late_weight - 1) / (1 + torch.exp(-err / temp))
        ph = delta ** 2 * (torch.sqrt(1 + (err / delta) ** 2) - 1)
        g, l, e = circ.split(pc)
        loss = (w * ph).mean() + .002 * ((g - 1) ** 2).mean() + .0005 * (l ** 2).mean() + (.0005 * (e ** 2).mean() if e.numel() else 0.0)
        loss.backward()
        return loss

    opt.step(closure)
    return torch.max(torch.min(p, hi), lo).detach()
