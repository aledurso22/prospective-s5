"""Decompose the learned metric w vs the exact-credit correction alpha,
AT THE FINAL TRAINED PARAMS (proper version of the init-params probe).

For each seed: train routeA on delayed copy, checkpoint params at init,
step 750, and final. At each checkpoint, on a fixed probe batch, compute
per (layer, mode):
    u_j   = online gradient block (S-slot)
    v_j   = exact-BPTT gradient block (J-slot)
    alpha_j = <v_j, u_j> / ||u_j||^2  (per-mode exact correction)
and compare the metric w_j learned by routeA at the same checkpoint:
    |w|/|alpha|   (magnitude ratio; 1.0 = magnitude matches credit correction)
    phase(w) - phase(alpha)  (radians; 0 = phase matches credit correction)
plus correlations of |alpha| with |a_j| and ||u_j|| (spectral-optimizer
control).

Run:  python decompose_w_final.py
"""
from __future__ import annotations

import copy

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm

SEEDS = [0, 1]
STEPS = 1500
CHECKPOINTS = [1, 750, 1500]


def make_data(rng, batch=32):
    x = rng.randn(tcg.T, batch)
    y = np.concatenate([np.zeros((tcg.DELAY, batch)), x[:-tcg.DELAY]],
                       axis=0)
    return x, y


def probe_blocks(params, rng):
    x, y = make_data(rng)
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    lam = tcg.exact_lambda(params, q)
    Sa, Sb = tcg.sensitivities(params, h, x)
    xs = [x[..., None]] + [h[l].real for l in range(tcg.L)]
    out = []
    for l in range(tcg.L):
        h_prev = np.concatenate([np.zeros((1, tcg.BATCH, tcg.N)),
                                 h[l][:-1]], axis=0)
        for j in range(tcg.N):
            u_a = np.sum(np.conj(q[l][:, :, j]) * Sa[l][:, :, j])
            u_b = np.einsum("tb,tbm->m", np.conj(q[l][:, :, j]),
                            Sb[l][:, :, j, :])
            v_a = np.sum(np.conj(lam[l][:, :, j]) * h_prev[:, :, j])
            v_b = np.einsum("tb,tbm->m", np.conj(lam[l][:, :, j]), xs[l])
            u = np.concatenate([[u_a], u_b])
            v = np.concatenate([[v_a], v_b])
            alpha = np.conj(np.vdot(v, u) / max(np.vdot(u, u).real, 1e-300))
            out.append((l, j, u, v, alpha))
    return out


def decompose(params, w, seed):
    rng = np.random.RandomState(900 + seed)
    for (l, j, u, v, alpha) in probe_blocks(params, rng):
        wj = w[l][j]
        yield dict(layer=l, mode=j,
                   ratio=abs(wj) / max(abs(alpha), 1e-300),
                   dphi=float(np.angle(wj) - np.angle(alpha)),
                   amag=abs(alpha), unorm=float(np.linalg.norm(u)),
                   amode=float(np.abs(params["a"][l][j])))


def train_and_decompose(seed):
    tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 128, 50, 1, 32
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    ckpts = {}
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
        # meta-gradient for w (routeA form)
        G_next = cvm.exact_grad(params_next, x, y)
        gN = tcg.flat_grads(G_next, params_next)
        off = 0
        for l in range(tcg.L):
            th = params["theta"][l]
            u_mode = tcg.sig(params["rho"][l])
            sigp = u_mode * (1 - u_mode)
            A = G["a"][l] * np.exp(1j * th)
            Gb = G["b"][l]
            M_ = Gb.shape[1]
            gN_rho = gN[off:off + tcg.N]
            gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
            gN_bre = gN[off + 2 * tcg.N:off + 2 * tcg.N + tcg.N * M_].reshape(
                tcg.N, M_)
            gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                        off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
            off += 2 * tcg.N + 2 * tcg.N * M_
            du = (gN_rho * sigp * A.real
                  + gN_theta * (-u_mode) * A.imag
                  + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
            dv = (gN_rho * sigp * A.imag
                  + gN_theta * (u_mode) * A.real
                  + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        params = params_next
        if step in CHECKPOINTS:
            ckpts[step] = (copy.deepcopy(params), [wl.copy() for wl in w])
    final_loss = float(np.mean([0.0]))  # loss curve not needed here
    return ckpts


def main() -> None:
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        ckpts = train_and_decompose(seed)
        for step in CHECKPOINTS:
            params, w = ckpts[step]
            rows = list(decompose(params, w, seed))
            for l in range(tcg.L):
                rs = [r for r in rows if r["layer"] == l]
                ratio = np.median([r["ratio"] for r in rs])
                dphi = np.median([r["dphi"] for r in rs])
                ca = np.corrcoef([r["amag"] for r in rs],
                                 [r["amode"] for r in rs])[0, 1]
                cu = np.corrcoef([r["amag"] for r in rs],
                                 [r["unorm"] for r in rs])[0, 1]
                print(f"  step {step:>5} L{l}: |w|/|a| {ratio:6.2f}  "
                      f"dphi {dphi:+.2f} rad   corr(amag,|a|) {ca:+.2f}  "
                      f"corr(amag,||u||) {cu:+.2f}")


if __name__ == "__main__":
    main()
