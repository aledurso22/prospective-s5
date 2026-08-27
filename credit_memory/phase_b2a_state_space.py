"""B2A -- verify the exact state-space formulation.

F=diag(A, conj(A)), d=[1;1] (see credit_memory/hankel.py docstring for
the exact construction). Confirms the vectorized recurrence/readout
reproduces both Phase-A's P/Q implementation (credit_memory/teacher.py,
unedited) and the trusted BPTT reference to machine precision. Pure
convention check -- no new claim.

Run:  python -m credit_memory.phase_b2a_state_space
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
from credit_memory.hankel import build_F, build_c_t, propagate

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def state_space_gradient(a1, B1, Sa0, q1):
    """Full (non-reduced) state-space computation of G_causal, per mode,
    via x_u = F x_{u-1} + d Sa0_u[m]; g_t[m] = c_t^dagger x_t."""
    N = a1.shape[0]
    f_diag = build_F(a1)                      # (2N,)
    d = np.ones(2 * N, np.complex128)
    G = np.zeros(N, np.complex128)
    for m in range(N):
        x = propagate(np.diag(f_diag), d, Sa0[:, :, m])   # (T,BATCH,2N)
        c_t = build_c_t(q1, B1[:, m])                     # (T,BATCH,2N)
        g_t = np.sum(np.conj(c_t) * x, axis=-1)
        G[m] = g_t.sum()
    return G


def main() -> None:
    N, T, BATCH = 6, 40, 8
    rows = []
    with set_l2_config(N, T, BATCH):
        for seed in range(5):
            params = tcg.init_params(seed)
            rng = np.random.RandomState(60000 + seed)
            x, r = draw_trajectory(params, rng, T, BATCH)
            out = compute_teacher(params, x, r)
            G_ss = state_space_gradient(out["a1"], out["B1"], out["Sa0"],
                                        out["q1"])
            err_vs_causal = np.linalg.norm(G_ss - out["G_causal"]) \
                / max(np.linalg.norm(out["G_causal"]), 1e-300)
            err_vs_bptt = np.linalg.norm(G_ss - out["G_bptt"]) \
                / max(np.linalg.norm(out["G_bptt"]), 1e-300)
            rows.append(dict(seed=seed,
                             state_space_vs_causal_PQ_rel_err=float(
                                 err_vs_causal),
                             state_space_vs_bptt_rel_err=float(err_vs_bptt)))
            print(f"seed {seed}: state-space vs P/Q rel_err="
                  f"{err_vs_causal:.3e}  state-space vs BPTT rel_err="
                  f"{err_vs_bptt:.3e}")

    all_pass = all(row["state_space_vs_causal_PQ_rel_err"] < 1e-10
                   and row["state_space_vs_bptt_rel_err"] < 1e-10
                   for row in rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH), rows=rows,
              all_pass=bool(all_pass))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_b2a_state_space_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"ALL PASS: {all_pass}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
