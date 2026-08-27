"""B1.0 -- deterministic probe exposing the exact teacher cleanly.

Confirms credit_memory.teacher's P/Q contraction still agrees with the
trusted BPTT reference (toyrig.ssm_rig.assemble(..., direct=True)) to
machine precision, and reports the baseline online/exact gap this probe
exposes to Phase B1's compression ladder.

Does NOT change the Phase-A equations. Run:
  python -m credit_memory.phase_b1_0_probe
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def cos(u, v):
    u = np.ravel(u); v = np.ravel(v)
    return float(np.abs(np.vdot(v, u)) / (np.linalg.norm(u)
                                          * np.linalg.norm(v) + 1e-300))


def main() -> None:
    N, T, BATCH = 6, 40, 8
    rows = []
    with set_l2_config(N, T, BATCH):
        for seed in range(5):
            params = tcg.init_params(seed)
            rng = np.random.RandomState(30000 + seed)
            x, r = draw_trajectory(params, rng, T, BATCH)
            out = compute_teacher(params, x, r)
            err = np.linalg.norm(out["G_causal"] - out["G_bptt"]) \
                / max(np.linalg.norm(out["G_bptt"]), 1e-300)
            c_on = cos(out["G_online"], out["G_bptt"])
            c_ex = cos(out["G_causal"], out["G_bptt"])
            rows.append(dict(seed=seed,
                             causal_vs_bptt_rel_err=float(err),
                             causal_vs_bptt_cos_check=float(cos(
                                 out["G_causal"], out["G_bptt"])),
                             online_cos_vs_bptt=c_on, causal_cos_vs_bptt=c_ex,
                             G_bptt_norm=float(np.linalg.norm(out["G_bptt"])),
                             G_online_norm=float(np.linalg.norm(
                                 out["G_online"]))))
            print(f"seed {seed}: causal-vs-BPTT rel_err={err:.3e}  "
                  f"online cos={c_on:.3f}  causal(exact) cos={c_ex:.3f}")

    all_exact = all(row["causal_vs_bptt_rel_err"] < 1e-10 for row in rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH), rows=rows,
              all_exact_pass=bool(all_exact))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_b1_0_probe_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"ALL EXACT (rel_err<1e-10): {all_exact}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
