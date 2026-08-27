"""B7B -- verify g_full_causal == g_BPTT at training snapshots (init,
early, middle, late), for BOTH the "a" block (already verified
throughout Phase A/B1-B4 via credit_memory/teacher.py) and the NEW "b"
block generalization (credit_memory/full_causal.py), before any task
loss is interpreted.

Uses the same bitwise-reproducible replay of the ONLINE arm's own
trajectory as B5.1/B6 (not a new experiment -- re-deriving intermediate
parameter state at predetermined checkpoints).

Run:  python -m credit_memory.b7_verify_exact
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.full_causal import full_causal_gradient
from credit_memory.b5_train import set_config, draw_task_batch, loss_of, LR
from credit_memory.b5_1_action_utility import adam_step

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b7")

L, N, T, DELAY, BATCH = 2, 6, 60, 20, 8
SEEDS = list(range(3))          # verification only, small is enough
CHECKPOINTS = [0, 100, 300, 600]   # 0 = initialization


def replay_online_to(seed, target_step):
    set_config()
    params = tcg.init_params(seed)
    if target_step == 0:
        return params
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    for step in range(1, target_step + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        g = tcg.flat_grads(G, params)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
    return params


def cos_np(u, v):
    return float(np.abs(np.vdot(v, u))
                / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def relerr_np(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-300))


def main() -> None:
    print("=" * 90)
    print("Phase B7B: full-causal vs BPTT exactness check (a AND b blocks)")
    print("=" * 90)

    rows = []
    for seed in SEEDS:
        for ckpt in CHECKPOINTS:
            set_config()
            params = replay_online_to(seed, ckpt)
            fresh_rng = np.random.RandomState(424242 + seed * 1000 + ckpt)
            x, y = draw_task_batch(fresh_rng)
            _, h, r = loss_of(params, x, y)
            q = tcg.spatial_q(params, h, r)
            Sa, Sb = tcg.sensitivities(params, h, x)

            lam = tcg.exact_lambda(params, q)
            G_bptt = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
            Ga_bptt, Gb_bptt = G_bptt["a"][0], G_bptt["b"][0]

            a1, B1 = params["a"][1], params["b"][1]
            Ga_full, Gb_full = full_causal_gradient(a1, B1, N, q[1],
                                                     Sa[0], Sb[0])

            err_a_vs_bptt = relerr_np(Ga_full, Ga_bptt)
            err_b_vs_bptt = relerr_np(Gb_full, Gb_bptt)
            cos_a = cos_np(Ga_full, Ga_bptt)
            cos_b = cos_np(Gb_full, Gb_bptt)
            norm_ratio_a = float(np.linalg.norm(Ga_full)
                                 / (np.linalg.norm(Ga_bptt) + 1e-300))
            norm_ratio_b = float(np.linalg.norm(Gb_full)
                                 / (np.linalg.norm(Gb_bptt) + 1e-300))

            rows.append(dict(seed=seed, checkpoint=ckpt,
                             a_block=dict(cos=cos_a, rel_err=err_a_vs_bptt,
                                         norm_ratio=norm_ratio_a),
                             b_block=dict(cos=cos_b, rel_err=err_b_vs_bptt,
                                         norm_ratio=norm_ratio_b)))
            print(f"seed={seed} ckpt={ckpt:4d}: a_block rel_err="
                  f"{err_a_vs_bptt:.2e} cos={cos_a:.12f}   "
                  f"b_block rel_err={err_b_vs_bptt:.2e} cos={cos_b:.12f}")

    all_pass = all(row["a_block"]["rel_err"] < 1e-8
                   and row["b_block"]["rel_err"] < 1e-8 for row in rows)
    print("-" * 90)
    print(f"ALL EXACT (a and b blocks, rel_err < 1e-8): {all_pass}")
    if not all_pass:
        print("STOP: full-causal reconstruction is not exact -- diagnose "
              "before running B7C/D/E.")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(L=L, N=N, T=T, DELAY=DELAY, BATCH=BATCH,
                                    seeds=SEEDS, checkpoints=CHECKPOINTS),
              rows=rows, all_pass=bool(all_pass))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b7b_verify_exact_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
