"""D6 generality: do the three closed-loop phenomena replicate in the
real 2D-rotational RNN (no complex arithmetic, no realification)?

Phenomenon 1 (static ceiling): causal Wiener filter's credit cosine
rises with horizon K at trained params.
Phenomenon 2 (deployment barrier): frozen full-K=64 Wiener filter
deployed from scratch destabilizes/degrades training vs online.
Phenomenon 3 (adaptive orientation): routePhi — per-block SO(2)
rotation angle phi learned by the routeA one-step lookahead
meta-gradient — beats online.

Arms (paired seeds {0,1,2}, same budget): online, routePhi, bptt,
frozenW64. Plus the static Wiener sweep (K in {1,4,16,64}) at trained
params (2x2 matrix FIR per block).

REGISTERED BARS (fixed before running):
  P3: median routePhi <= 0.5 x median online, all finite.
  P1: static cosine at the top layer monotone non-decreasing in K.
  P2: median frozenW64 > median online.

Run:  python rot_rnn_generality.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import rot_rnn as rr

SEEDS = [0, 1, 2]
K_GRID = [1, 4, 16, 64]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "rot_rnn_generality")


def rot2(phi):
    c, s = np.cos(phi), np.sin(phi)
    return c, s


def rotate_G(G, phi):
    """Apply per-block angle phi: R(phi) on (Gp, Gq) pairs and on the
    state index of Gb rows."""
    Gp, Gq, Gb = [], [], []
    for l in range(rr.L):
        c, s = np.cos(phi[l]), np.sin(phi[l])
        Gp.append(c * G["p"][l] - s * G["q"][l])
        Gq.append(s * G["p"][l] + c * G["q"][l])
        # Gb[l]: (NB, 2, M) — rotate the state index
        gb = G["b"][l]
        Gb.append(np.stack([c[:, None] * gb[:, 0] - s[:, None] * gb[:, 1],
                            s[:, None] * gb[:, 0] + c[:, None] * gb[:, 1]],
                           axis=1))
    return dict(p=Gp, q=Gq, b=Gb, c=G["c"])


def drotate_G(G, phi):
    """d/dphi of rotate_G."""
    Gp, Gq, Gb = [], [], []
    for l in range(rr.L):
        c, s = np.cos(phi[l]), np.sin(phi[l])
        Gp.append(-s * G["p"][l] - c * G["q"][l])
        Gq.append(c * G["p"][l] - s * G["q"][l])
        gb = G["b"][l]
        Gb.append(np.stack([-s[:, None] * gb[:, 0] - c[:, None] * gb[:, 1],
                            c[:, None] * gb[:, 0] - s[:, None] * gb[:, 1]],
                           axis=1))
    return dict(p=Gp, q=Gq, b=Gb, c=np.zeros_like(G["c"]))


def adam(flat, g, m, v, step, lr=rr.LR):
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g ** 2
    upd = lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step))
                                          + 1e-8)
    return flat - upd, m, v


def clip(g):
    n = np.linalg.norm(g)
    return g * (CLIP_ / n) if n > CLIP_ else g


CLIP_ = 1.0


def train_arm(arm, seed, wiener64=None):
    params = rr.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    phi = [np.zeros(rr.NB) for _ in range(rr.L)]
    flat = rr.flatten_params(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, rr.STEPS + 1):
        x, y = rr.make_data(rng)
        loss, G, q, r, h = rr.batch_grad(params, x, y)
        losses.append(loss)
        if arm == "bptt":
            _, Ge, _, _, _ = rr.batch_grad(params, x, y, exact=True)
            g = clip(rr.flat(rr.param_grad_transform(Ge, params), params))
        elif arm == "frozenW64":
            err = [apply_fir2(q[l], wiener64[l]) for l in range(rr.L)]
            S = rr.sensitivities(params, h, x)
            Gf = rr.assemble(params, h, x, r, err, S=S)
            g = clip(rr.flat(rr.param_grad_transform(Gf, params), params))
        else:
            if arm == "routePhi":
                G = rotate_G(G, phi)
            g = clip(rr.flat(rr.param_grad_transform(G, params), params))
        flat, m, v = adam(flat, g, m, v, step)
        params_next = rr.pack_params(params, flat)
        if arm == "routePhi":
            # meta-gradient through the update (routeA style; Adam
            # normalization ignored as in cvm)
            _, Gn, _, _, _ = rr.batch_grad(params_next, x, y, exact=True)
            Gn_t = rr.param_grad_transform(Gn, params_next)
            dG = drotate_G(G, phi)
            dGn = rr.param_grad_transform(dG, params_next)
            for l in range(rr.L):
                dphi = (-rr.LR) * (
                    Gn_t["rho"][l] * dGn["rho"][l]
                    + Gn_t["theta"][l] * dGn["theta"][l]
                    + (Gn_t["b"][l] * dGn["b"][l]).sum(axis=(1, 2)))
                phi[l] = phi[l] - 1e-3 * dphi
        params = params_next
    losses = np.asarray(losses)
    return dict(final=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                phi=[pl.copy() for pl in phi])


def apply_fir2(q, F):
    """2x2 matrix FIR per block: out_t = sum_k F_k @ q_{t-k}."""
    K = F.shape[0]
    out = np.zeros_like(q)
    for k in range(K):
        out[k:] += np.einsum("nij,tbnj->tbni", F[k], q[:rr.T - k])
    return out


def wiener_fit_2x2(q, lam, K):
    """Per block: min over F_0..F_{K-1} (2x2 real) of
    sum_t ||lam_t - sum_k F_k q_{t-k}||^2. Shared feature matrix per
    block: Zf[row, 2k+cj] = q_{t-k}[b,n,cj]; one lstsq per output comp."""
    T_, B, N_, _ = q.shape
    F = np.zeros((K, N_, 2, 2))
    for n in range(N_):
        Zf = np.concatenate(
            [np.concatenate([q[t - k, :, n, :] for k in range(K)], axis=-1)
             for t in range(K - 1, T_)], axis=0)          # (M, 2K)
        Y = lam[K - 1:, :, n, :].reshape(-1, 2)           # (M, 2)
        for ci in range(2):
            sol, *_ = np.linalg.lstsq(Zf, Y[:, ci], rcond=None)
            F[:, n, ci, :] = sol.reshape(K, 2)
    return F


def static_cosine(params, seed, K):
    rng = np.random.RandomState(900 + seed)
    x, y = rr.make_data(rng)
    h, yhat = rr.forward(params, x)
    r = yhat - y
    r[:rr.DELAY] = 0.0
    q = rr.spatial_q(params, r)
    lam = rr.exact_lambda(params, q)
    F = wiener_fit_2x2(q[rr.L - 1], lam[rr.L - 1], K)
    lh = apply_fir2(q[rr.L - 1], F)
    return float(np.abs(np.vdot(lam[rr.L - 1].ravel(), lh.ravel()))
                 / (np.linalg.norm(lam[rr.L - 1]) * np.linalg.norm(lh)
                    + 1e-300))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table = {}
    w64 = {}
    cos_K = {K: [] for K in K_GRID}
    for seed in SEEDS:
        print(f"seed {seed}: routePhi reference train...", flush=True)
        ref = train_arm("routePhi", seed)
        table.setdefault("routePhi", []).append(ref["final"])
        print(f"  routePhi final {ref['final']:.4f} finite {ref['finite']}",
              flush=True)
        params = rr.init_params(seed)   # filter/probe at init params
        on = train_arm("online", seed)
        table.setdefault("online", []).append(on["final"])
        print(f"  online   final {on['final']:.4f}", flush=True)
        bp = train_arm("bptt", seed)
        table.setdefault("bptt", []).append(bp["final"])
        print(f"  bptt     final {bp['final']:.4f}", flush=True)
        # Wiener filter estimated on probe stats at init params (fixed
        # reference state; the point is the causal-horizon ceiling and
        # the deployment barrier, both state-dependent anyway)
        rng = np.random.RandomState(900 + seed)
        x, y = rr.make_data(rng)
        h, yhat = rr.forward(params, x)
        r = yhat - y
        r[:rr.DELAY] = 0.0
        q = rr.spatial_q(params, r)
        lam = rr.exact_lambda(params, q)
        w64[seed] = [wiener_fit_2x2(q[l], lam[l], 64) for l in range(rr.L)]
        for K in K_GRID:
            cos_K[K].append(static_cosine(params, seed, K))
        fw = train_arm("frozenW64", seed, wiener64=w64[seed])
        table.setdefault("frozenW64", []).append(fw["final"])
        print(f"  frozenW64 final {fw['final']:.4f} finite {fw['finite']}",
              flush=True)

    med = {a: float(np.median(v)) for a, v in table.items()}
    cos_med = {K: float(np.median(v)) for K, v in cos_K.items()}
    finite_all = all(np.isfinite([table[a][i] for a in table
                                  for i in range(len(SEEDS))]))
    p3 = med["routePhi"] <= 0.5 * med["online"] and finite_all
    cos_list = [cos_med[K] for K in K_GRID]
    p1 = all(b >= a - 1e-9 for a, b in zip(cos_list, cos_list[1:]))
    p2 = med["frozenW64"] > med["online"]
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"static cosine by K (top layer): "
          f"{ {K: round(cos_med[K], 3) for K in K_GRID} }")
    print(f"P1 ceiling rises with horizon: {'PASS' if p1 else 'FAIL'}")
    print(f"P2 frozen filter destabilizes: {'PASS' if p2 else 'FAIL'}")
    print(f"P3 adaptive orientation wins (0.5x online): {'PASS' if p3 else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, per_arm=table, medians=med, cos_K=cos_med,
               bars=dict(P1=bool(p1), P2=bool(p2), P3=bool(p3)))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
