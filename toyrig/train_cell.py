"""Depth law: does the learned metric's magnitude follow a law in depth/delay?

The agent's hypothesis: |w_l| ~ 1 / (effective online-credit transmission
at depth l). If lower layers lose more future/cross-layer credit, the
optimal meta-geometry compensates with larger gain. This experiment
measures, for each (L, delay) cell, at trained params:

    |w_l|      median per-mode learned metric magnitude at layer l
    |alpha_l|  median per-mode exact-credit correction magnitude
    ratio      |w_l| / |alpha_l|

and checks for a reproducible law across L in {1,2,4,8} and delay in
{25,50,100}, 2 seeds, delayed copy, T=128.

Run:  python depth_law.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.probes import probe_blocks, make_data

L_SWEEP = [1, 2, 4, 8]
D_SWEEP = [25, 50, 100]
SEEDS = [0, 1]
STEPS = 1500
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "depth_law")


def train_cell(L, delay, seed):
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        L, 16, 128, delay, 1, 32
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
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
    return params, w


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for L in L_SWEEP:
        for D in D_SWEEP:
            for seed in SEEDS:
                params, w = train_cell(L, D, seed)
                rng = np.random.RandomState(900 + seed)
                rows_pb = probe_blocks(params, rng)
                for l in range(L):
                    wabs = [abs(w[l][j]) for j in range(tcg.N)]
                    aabs = [ab for (ll, j, u, v, ab) in rows_pb
                            if ll == l and j < tcg.N]
                    rows.append(dict(L=L, D=D, seed=seed, layer=l,
                                     w_med=float(np.median(wabs)),
                                     alpha_med=float(np.median(aabs))))
            # per-cell summary over seeds
            for l in range(L):
                rs = [r for r in rows if r["L"] == L and r["D"] == D
                      and r["layer"] == l]
                wm = float(np.median([r["w_med"] for r in rs]))
                am = float(np.median([r["alpha_med"] for r in rs]))
                print(f"L={L} D={D:<4} layer {l}: |w| {wm:8.3f}  "
                      f"|alpha| {am:8.3f}  ratio {wm / max(am, 1e-9):8.2f}",
                      flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(L_sweep=L_SWEEP, D_sweep=D_SWEEP,
                                    seeds=SEEDS, steps=STEPS), rows=rows)
    with open(os.path.join(RESULTS_DIR, "depth_law.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote depth_law.json")


if __name__ == "__main__":
    main()
