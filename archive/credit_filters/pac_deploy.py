"""PAC deployment — the causal law, end-to-end. No meta-gradient, no BPTT.

pac_probe2 established: the learned Route-A phase is the optimal scalar
projection of exact onto causal credit (up to its own seed-reliability),
the structure is task+architecture (online-params control), and the
AR(1) closure K = 1/(1 - conj(a) rho(1)) predicts w as well as or
better than the optimal projection. This script deploys that law as the
preconditioner, with rho(1) estimated by causal EMA over the online
credit signal q during training — a fully causal adaptive rule with no
meta-gradient and no exact credit anywhere.

  beta_j  <- (1-gamma) beta_j + gamma * rho1_step(q_j)
  K_j     = 1 / (1 - conj(a_j) beta_j)      (|conj(a) beta| clipped to 0.95)
  err     = scale_by_w(G, K)                 (same convention as routeA's w:
           the probe measured R(arg c_AR1, arg w) in this convention)

Arms (paired seeds {0,1,2}, same init/streams, 1500 steps):
  online    w = 1
  pac_g05   the law, gamma = 0.05
  pac_g01   the law, gamma = 0.01
  routeA    live meta-gradient reference (paired)

REGISTERED BAR (P4, fixed before running): the better pac arm closes
>= 50% of the online -> routeA gap on median final loss:
(online - pac) / (online - routeA) >= 0.5. Per-seed paired deltas
reported. < 20% => directionally right but not load-bearing.

Run:  python pac_deploy.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.train_cell import STEPS
from toyrig.probes import make_data

SEEDS = [0, 1, 2]
ARMS = ["online", "pac_g05", "pac_g01", "routeA"]
GAMMAS = {"pac_g05": 0.05, "pac_g01": 0.01}
CLIP_RHO = 0.95
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "pac_deploy")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def rho1_of(q):
    """One-step lag-1 autocorrelation of q per mode (same lag direction
    as the probe: E[q_t conj(q_{t-1})] / E|q_{t-1}|^2)."""
    num = np.mean(q[1:] * np.conj(q[:-1]), axis=(0, 1))
    den = np.mean(np.abs(q[:-1]) ** 2, axis=(0, 1)) + 1e-300
    return num / den


def train_arm(arm, seed):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    beta = [np.zeros(tcg.N, np.complex128) for _ in range(tcg.L)]
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    losses = []
    gamma = GAMMAS.get(arm)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        if gamma is not None:
            for l in range(tcg.L):
                beta[l] = (1 - gamma) * beta[l] + gamma * rho1_of(q[l])
                z = np.conj(params["a"][l]) * beta[l]
                mag = np.abs(z)
                over = mag > CLIP_RHO
                z[over] *= CLIP_RHO / mag[over]
                w[l] = 1.0 / (1.0 - z)
        G_use = G if arm == "online" else cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
        if arm == "routeA":
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
                gN_bre = gN[off + 2 * tcg.N:
                            off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
                gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                            off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        params = params_next
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses))), w


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table, w_pac = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            fl, fin, w = train_arm(arm, seed)
            table.setdefault(arm, []).append(fl)
            if arm == "pac_g05":
                w_pac[seed] = [wl.copy() for wl in w]
            print(f"  seed {seed} {arm:<8s} final {fl:.4f} finite {fin}",
                  flush=True)
    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    gap = med["online"] - med["routeA"]
    fracs = {arm: ((med["online"] - med[arm]) / gap if gap > 0 else None)
             for arm in ("pac_g05", "pac_g01")}
    best = max(fracs, key=lambda a: fracs[a])
    win = fracs[best] >= 0.5
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"gap online->routeA {gap:.4f}; closed: "
          f"pac_g05 {fracs['pac_g05']:.2f}  pac_g01 {fracs['pac_g01']:.2f}")
    print("paired deltas per seed (online - arm): "
          f"{ {a: [round(table['online'][i] - table[a][i], 4) for i in range(len(SEEDS))] for a in ARMS[1:]} }")
    print(f"BAR P4: best pac closes >= 50%  ->  "
          f"{'CAUSAL LAW HOLDS' if win else 'NO WIN'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(steps=STEPS, seeds=SEEDS,
                                    gammas=GAMMAS, clip_rho=CLIP_RHO,
                                    bar="P4: >= 50% of online->routeA gap"),
               per_arm=table, medians=med, fracs=fracs, win=bool(win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "w_pac_g05.npy"),
            np.array([w_pac[s] for s in SEEDS], dtype=object))
    print("wrote summary.json")


if __name__ == "__main__":
    main()
