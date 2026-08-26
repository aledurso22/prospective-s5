"""Cross-task transfer of the learned metric (the MAML-style decisive test).

Meta-train the per-mode metric w across a FAMILY of delayed-copy tasks
(delays {25, 50, 100}, T=128), jointly with the model. Then freeze w and
deploy on an UNSEEN delay (200): does the geometry learned on the family
transfer, or is it a per-task overfit?

Arms on the unseen task (identical model class and budget):
  online     w = 1 fixed (plain online RTRL)
  frozen_w   w = the meta-trained geometry, frozen, no updates
  bptt       exact adjoint (reference ceiling)

Bar (registered before running): frozen_w median final loss <= 0.5 x
online median on the unseen task, all runs finite.

Config: meta-train 1500 steps, batch 32, delays sampled uniformly per
batch; deploy 1500 steps on D=200; seeds {0,1,2} for meta-train and for
evaluation.

Run:  python transfer_m.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm

SEEDS = [0, 1, 2]
META_DELAYS = [25, 50, 100]
EVAL_DELAY = 200
STEPS = 1500
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "transfer_m")


def make_data(rng, delay, batch=32, T=None):
    T = tcg.T if T is None else T
    x = rng.randn(T, batch)
    y = np.concatenate([np.zeros((delay, batch)), x[:-delay]], axis=0)
    return x, y


def meta_train(seed):
    tcg.T, tcg.M_IN, tcg.BATCH = 128, 1, 32
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    for step in range(1, STEPS + 1):
        D = int(rng.choice(META_DELAYS))
        tcg.DELAY = D
        x, y = make_data(rng, D)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
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
    return w


def deploy(arm, seed, w_frozen):
    tcg.T = 256  # unseen horizon (meta-train was T=128)
    tcg.DELAY = EVAL_DELAY
    params = tcg.init_params(seed)
    rng = np.random.RandomState(2000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)] \
        if arm == "online" else [wl.copy() for wl in w_frozen]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng, EVAL_DELAY)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:EVAL_DELAY] = 0.0
        losses.append(0.5 * float(np.mean(r ** 2)))
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        if arm == "bptt":
            G = tcg.assemble(params, h, x, r, tcg.exact_lambda(params, q),
                             Sa, Sb, direct=True)
        else:
            G = tcg.assemble(params, h, x, r,
                             [q[l] * w[l][None, None, :]
                              for l in range(tcg.L)], Sa, Sb)
        g = cvm.clip(tcg.flat_grads(G, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))))


def main() -> None:
    tcg.L, tcg.N = 4, 16
    print("=" * 78)
    print("Transfer: metric learned on delays {25,50,100}, deployed on 200")
    print("=" * 78)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for seed in SEEDS:
        w = meta_train(seed)
        for arm in ["online", "frozen_w", "bptt"]:
            out = deploy(arm, seed, w)
            results[f"{arm}/s{seed}"] = out
            print(f"  {arm:<9s} seed {seed}: final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}", flush=True)
    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final_loss"]
                                 for s in SEEDS]))
           for arm in ["online", "frozen_w", "bptt"]}
    finite_all = all(results[f"{arm}/s{s}"]["finite"]
                     for arm in ["online", "frozen_w", "bptt"]
                     for s in SEEDS)
    win = med["frozen_w"] <= 0.5 * med["online"] and finite_all
    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"BAR: frozen_w <= 0.5x online on unseen delay, all finite -> "
          f"{'WIN' if win else 'NO WIN'}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(meta_delays=META_DELAYS, eval_delay=EVAL_DELAY,
                           steps=STEPS, seeds=SEEDS),
               medians=med, win=win)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
