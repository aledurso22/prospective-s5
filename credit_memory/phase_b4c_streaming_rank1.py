"""B4C -- reproduce B3's rank-1 (R1) result using ONLY the streaming
estimator (credit_memory/streaming.py): no batch numpy reduction over a
stored (T,BATCH,...) array, no BPTT, no exact P/Q teacher state -- one
timestep at a time, O(2N) running state per lower mode during
calibration, frozen to O(1) per selected channel for test.

Same 8 seeds, same 4 calibration / 4 test trajectories, same N/T/BATCH
as B2/B3 (imported directly for an exact apples-to-apples comparison).

First sanity-checks that the streaming estimator's frozen rho exactly
matches B3's batch-computed g_p (same numbers, different code path) --
this must hold since both implement the identical sum, just via a loop
vs. vectorized ops.

Run:  python -m credit_memory.phase_b4c_streaming_rank1
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.streaming import StreamingRelevance, run_windowed_calibration
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np,
    relerr_np)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def deploy_selected_channel(f_diag, top_j, B1_col, test_row, m):
    """Genuinely O(1)-per-selected-channel forward pass: only needs
    Sa0[:, :, m] (existing eligibility, already computed by the online
    rule) and ONE component of q1 (q1[:, :, j_orig], not the full
    N-dim q1 vector -- see PHASE_B4.md's B4E accounting)."""
    n2 = f_diag.shape[0]
    j_orig = top_j % N
    is_Q = top_j >= N
    u_t = test_row["Sa0"][:, :, m]                     # (T,BATCH)
    q1_j = test_row["q1"][:, :, j_orig]                 # (T,BATCH), ONE mode
    Bcoef = B1_col[j_orig]
    # matches credit_memory/hankel.py:build_c_t exactly (P-block: 0.5
    # conj(B) q1; Q-block: 0.5 B conj(q1)), specialized to one component
    c_j = (0.5 * Bcoef * np.conj(q1_j) if is_Q
          else 0.5 * np.conj(Bcoef) * q1_j)
    lam = f_diag[top_j]
    Tn, Bn = u_t.shape
    x = np.zeros((Tn, Bn), np.complex128)
    prev = np.zeros(Bn, np.complex128)
    for t in range(Tn):
        prev = lam * prev + u_t[t]
        x[t] = prev
    g_t = np.conj(c_j) * x
    return g_t.sum()


def main() -> None:
    print("=" * 78)
    print(f"Phase B4C: streaming-only rank-1, {len(SEEDS)} seeds")
    print("=" * 78)

    # sanity check: streaming rho matches B3's batch g_p exactly
    _, cal0 = collect_rows(0, N_CAL_TRAJ, offset=0)
    a1_0, B1_0 = cal0[0]["a1"], cal0[0]["B1"]
    f_diag_0 = build_F(a1_0)
    d0 = np.ones(2 * N, np.complex128)
    m0 = 0
    est0 = run_windowed_calibration(f_diag_0, cal0, m0)
    c_pool = np.concatenate([build_c_t(row["q1"], B1_0[:, m0])
                             for row in cal0], axis=0)
    u_pool = np.concatenate([row["Sa0"][:, :, m0] for row in cal0], axis=0)
    g_p_batch, _ = per_coordinate_contribution(f_diag_0, d0, c_pool, u_pool)
    sanity_err = float(np.linalg.norm(est0.rho - g_p_batch)
                       / (np.linalg.norm(g_p_batch) + 1e-300))
    print(f"sanity check (streaming rho vs B3 batch g_p, seed 0 mode 0): "
          f"rel_err={sanity_err:.3e}  {'PASS' if sanity_err < 1e-10 else 'FAIL'}")

    rows = []
    top_channels_by_seed_mode = {}
    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)

        top_j_by_mode = {}
        for m in range(N):
            est = run_windowed_calibration(f_diag, cal_rows, m)
            top_j = int(est.top_channel(1)[0])
            top_j_by_mode[m] = top_j
        top_channels_by_seed_mode[seed] = top_j_by_mode

        for t_idx, row in enumerate(test_rows):
            G_hat = np.zeros(N, np.complex128)
            for m in range(N):
                G_hat[m] = deploy_selected_channel(
                    f_diag, top_j_by_mode[m], B1[:, m], row, m)
            G_bptt, G_online = row["G_bptt"], row["G_online"]
            c_hat, c_on = cos_np(G_hat, G_bptt), cos_np(G_online, G_bptt)
            gap = max(1.0 - c_on, 1e-12)
            rows.append(dict(seed=seed, test_traj=t_idx, cos=c_hat,
                             cos_online=c_on,
                             rel_err=relerr_np(G_hat, G_bptt),
                             norm_ratio=float(np.linalg.norm(G_hat)
                                              / (np.linalg.norm(G_bptt)
                                                 + 1e-300)),
                             frac_gap_recovered=float((c_hat - c_on)
                                                      / gap)))
        print(f"seed {seed}: median cos="
              f"{np.median([x['cos'] for x in rows if x['seed'] == seed]):.4f}")

    med_cos = float(np.median([x["cos"] for x in rows]))
    med_online = float(np.median([x["cos_online"] for x in rows]))
    med_frac = float(np.median([x["frac_gap_recovered"] for x in rows]))
    print("-" * 78)
    print(f"B4C (streaming, no teacher) TEST median cos: {med_cos:.4f}")
    print(f"C0 online baseline median cos (same test set): {med_online:.4f}")
    print(f"median frac_gap_recovered: {med_frac:.4f}")
    print(f"Compare: B3 R1 (batch, same data) 0.926; B3 R3 0.940; "
          f"exact-teacher rank-1 diagnostic (B3B) 0.992")
    gate_pass = med_cos >= 0.90
    print(f"PRIMARY GATE median cos >= 0.90: {'PASS' if gate_pass else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                         n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ),
              sanity_streaming_vs_batch_rel_err=sanity_err,
              rows=rows, median_cos=med_cos,
              median_online_cos=med_online,
              median_frac_gap_recovered=med_frac,
              gate_pass=bool(gate_pass),
              top_channels_by_seed_mode=top_channels_by_seed_mode)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b4c_streaming_rank1_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
