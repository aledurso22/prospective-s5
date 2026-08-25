"""Optimum tracking diagnostic — does the instantaneous optimal phase
phi*(theta_n) move along training? (Correction to phase_track, which
followed the LEARNED phase, not the optimum.)

At checkpoints n in {1,100,250,500,750,1000,1500} of a routeA run,
freeze theta_n and estimate the instantaneous optimal per-mode phase
via the exact-credit scalar projection (arg c*, the probe-validated
oracle for this diagnostic) on 8 probe batches:

  phi*(theta_n) = arg E[lam conj(q)]/E|q|^2   per (layer, mode)

Measure per (layer, mode):
  Var_training[phi*(theta_n)]   over checkpoints (does the optimum move?)
  Var_batch[phi_hat(theta_n,B)] over probe batches (estimation noise)
  |arg w_learned - phi*(theta_n)| over checkpoints (learned-vs-optimum)

READING (fixed before running): if median Var_training << Var_batch
(ratio < 0.2) across layers, the optimum is static relative to noise
and Simonetto is unnecessary — close it. If Var_training is comparable
or larger, the optimum moves and phase_track only bounded momentum-
following of the learned path.

Run:  python optimum_track.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
CKPTS = [1, 100, 250, 500, 750, 1000, 1500]
N_PROBE = 8
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "optimum_track")


def cstar_phase(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    lam = tcg.exact_lambda(params, q)
    out = []
    for l in range(tcg.L):
        num = np.mean(lam[l] * np.conj(q[l]), axis=(0, 1))
        den = np.mean(np.abs(q[l]) ** 2, axis=(0, 1)) + 1e-300
        out.append(np.angle(num / den))
    return out                                    # (L, N)


def train_with_ckpts(seed):
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    opt_phases, learned_phases = {}, {}
    probe_rng = np.random.RandomState(7000 + seed)
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
            gN_bre = gN[off + 2 * tcg.N:
                        off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
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
        if step in CKPTS:
            ph = np.stack([np.stack(cstar_phase(params, *make_data(probe_rng)))
                           for _ in range(N_PROBE)], axis=0)  # (P, L, N)
            opt_phases[step] = ph
            learned_phases[step] = [np.angle(wl) for wl in w]
    return opt_phases, learned_phases


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agg = []
    for seed in SEEDS:
        print(f"seed {seed}: routeA + checkpoint optima...", flush=True)
        opt, learned = train_with_ckpts(seed)
        ph_mean = np.stack([opt[c].mean(axis=0) for c in CKPTS])  # (Ck,L,N)
        # circular variance over training of the batch-mean phase
        z = np.exp(1j * ph_mean)
        var_train = 1.0 - np.abs(z.mean(axis=0))          # (L,N)
        # circular variance over probe batches within checkpoints
        zb = np.exp(1j * np.stack([opt[c] for c in CKPTS]))  # (Ck,P,L,N)
        var_batch = 1.0 - np.abs(zb.mean(axis=1))         # (Ck,L,N)
        # learned vs instantaneous optimum distance
        dist = []
        for c in CKPTS:
            d = np.angle(np.exp(1j * (np.array(learned[c])
                                      - ph_mean[CKPTS.index(c)])))
            dist.append(np.abs(d))
        agg.append(dict(
            var_train=float(np.median(var_train)),
            var_batch=float(np.median(var_batch)),
            dist=float(np.median(dist)),
            var_train_layers=[float(np.median(var_train[l]))
                              for l in range(tcg.L)],
            var_batch_layers=[float(np.median(var_batch[:, l, :]))
                              for l in range(tcg.L)],
        ))
        print(f"  seed {seed}: Var_train {agg[-1]['var_train']:.4f}  "
              f"Var_batch {agg[-1]['var_batch']:.4f}  "
              f"|learned-opt| {agg[-1]['dist']:.4f} rad", flush=True)
        print(f"    per-layer Var_train/Var_batch ratio: "
              f"{[round(agg[-1]['var_train_layers'][l] / max(agg[-1]['var_batch_layers'][l], 1e-12), 3) for l in range(tcg.L)]}",
              flush=True)
    vt = np.median([a["var_train"] for a in agg])
    vb = np.median([a["var_batch"] for a in agg])
    print("-" * 70)
    print(f"medians: Var_training {vt:.4f}  Var_batch {vb:.4f}  "
          f"ratio {vt / max(vb, 1e-12):.3f}")
    print(f"reading: ratio < 0.2 => optimum static relative to noise, "
          f"Simonetto unnecessary; >= 0.2 => the optimum moves")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, per_seed=agg, var_train=vt, var_batch=vb,
                       ratio=float(vt / max(vb, 1e-12))), f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
