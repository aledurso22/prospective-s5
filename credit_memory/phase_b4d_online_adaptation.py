"""B4D -- fully online adaptation diagnostic. The relevance statistic is
updated continuously (EMA, credit_memory/streaming.py's "ema" mode) while
replaying a long fixed-architecture trajectory stream; no task-training
parameter updates, no BPTT in the update (BPTT is evaluation-only, used
periodically to score whichever channel is currently selected against a
FIXED held-out test set).

Focuses on the 4 hardest B3/B4C seeds (0, 3, 4, 5 -- all under 0.90
median cos in B4C) plus 2 easy ones (1, 6) for contrast, at several EMA
time constants gamma.

Run:  python -m credit_memory.phase_b4d_online_adaptation
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from credit_memory.hankel import build_F, build_c_t
from credit_memory.streaming import StreamingRelevance
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, N_TEST_TRAJ, collect_rows, cos_np, relerr_np)
from credit_memory.phase_b4c_streaming_rank1 import deploy_selected_channel

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

SEEDS = [0, 3, 4, 5, 1, 6]           # 4 hard (per B4C) + 2 easy
HARD_SEEDS = {0, 3, 4, 5}
GAMMAS = [0.005, 0.02, 0.08]
N_STREAM_TRAJ = 40                    # long replay stream
SNAPSHOT_EVERY = 4                    # trajectories between evaluations


def evaluate_current_selection(f_diag, top_j_by_mode, B1, test_rows):
    rows = []
    for row in test_rows:
        G_hat = np.zeros(N, np.complex128)
        for m in range(N):
            G_hat[m] = deploy_selected_channel(f_diag, top_j_by_mode[m],
                                               B1[:, m], row, m)
        c_hat = cos_np(G_hat, row["G_bptt"])
        rows.append(c_hat)
    return float(np.median(rows))


def main() -> None:
    print("=" * 78)
    print(f"Phase B4D: online EMA adaptation, seeds={SEEDS}, "
          f"gammas={GAMMAS}")
    print("=" * 78)

    results = {}
    for seed in SEEDS:
        _, stream_rows = collect_rows(seed, N_STREAM_TRAJ, offset=50000)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = stream_rows[0]["a1"], stream_rows[0]["B1"]
        f_diag = build_F(a1)

        for gamma in GAMMAS:
            estimators = {m: StreamingRelevance(f_diag, BATCH, mode="ema",
                                                gamma=gamma)
                         for m in range(N)}
            traj_idx = []
            cos_over_time = []
            top_channel_over_time = {m: [] for m in range(N)}
            n_channel_switches = {m: 0 for m in range(N)}
            prev_top = {m: None for m in range(N)}

            for i, row in enumerate(stream_rows):
                u_traj = row["Sa0"]                        # (T,BATCH,N)
                c_traj = {m: build_c_t(row["q1"], B1[:, m])
                         for m in range(N)}
                for t in range(T):
                    for m in range(N):
                        estimators[m].step(u_traj[t, :, m], c_traj[m][t])

                if (i + 1) % SNAPSHOT_EVERY == 0 or i == len(stream_rows) - 1:
                    top_j_by_mode = {}
                    for m in range(N):
                        tj = int(estimators[m].top_channel(1)[0])
                        top_j_by_mode[m] = tj
                        top_channel_over_time[m].append(tj)
                        if prev_top[m] is not None and tj != prev_top[m]:
                            n_channel_switches[m] += 1
                        prev_top[m] = tj
                    med_cos = evaluate_current_selection(f_diag,
                                                         top_j_by_mode, B1,
                                                         test_rows)
                    traj_idx.append(i + 1)
                    cos_over_time.append(med_cos)

            final_var = float(np.var(cos_over_time[-5:])) \
                if len(cos_over_time) >= 5 else float(np.var(cos_over_time))
            results[(seed, gamma)] = dict(
                traj_idx=traj_idx, cos_over_time=cos_over_time,
                final_cos=cos_over_time[-1],
                converged_cos_tail_var=final_var,
                n_channel_switches=n_channel_switches,
                total_switches=sum(n_channel_switches.values()))
            print(f"seed {seed} gamma={gamma}: final_cos="
                  f"{cos_over_time[-1]:.4f}  tail_var={final_var:.5f}  "
                  f"switches={sum(n_channel_switches.values())}")

    print("-" * 78)
    hard_finals = [results[(s, g)]["final_cos"] for s in SEEDS
                  for g in GAMMAS if s in HARD_SEEDS]
    easy_finals = [results[(s, g)]["final_cos"] for s in SEEDS
                  for g in GAMMAS if s not in HARD_SEEDS]
    print(f"hard-seed final cos: median={np.median(hard_finals):.4f}  "
          f"range=[{min(hard_finals):.3f},{max(hard_finals):.3f}]")
    print(f"easy-seed final cos: median={np.median(easy_finals):.4f}  "
          f"range=[{min(easy_finals):.3f},{max(easy_finals):.3f}]")
    for gamma in GAMMAS:
        finals = [results[(s, gamma)]["final_cos"] for s in SEEDS]
        tail_vars = [results[(s, gamma)]["converged_cos_tail_var"]
                    for s in SEEDS]
        print(f"gamma={gamma}: median final cos={np.median(finals):.4f}  "
              f"median tail var={np.median(tail_vars):.6f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                         hard_seeds=sorted(HARD_SEEDS), gammas=GAMMAS,
                         n_stream_traj=N_STREAM_TRAJ,
                         snapshot_every=SNAPSHOT_EVERY,
                         n_test_traj=N_TEST_TRAJ),
              results={f"seed{s}_gamma{g}": results[(s, g)]
                      for s in SEEDS for g in GAMMAS})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b4d_online_adaptation_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
