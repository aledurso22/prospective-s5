"""B8C -- static fixed-architecture held-out benchmark, exactly B3/B4's
protocol and seeds, comparing S0 (current unnormalized |rho|^2 score) vs
S1 (resource-normalized R_j = |rho|^2/(E+eps) score) for rank-1 channel
selection. No training; offline, causal-calibration-only selection,
evaluated against BPTT on held-out trajectories (BPTT used only to
SCORE the two selectors here, never inside either selection algorithm).

Run:  python -m credit_memory.b8_static_benchmark
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from credit_memory.hankel import build_F
from credit_memory.b8_normalized_selector import (
    run_windowed_calibration_normalized, EPS)
from credit_memory.phase_b4c_streaming_rank1 import deploy_selected_channel
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np,
    relerr_np)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def main() -> None:
    print("=" * 90)
    print(f"Phase B8C: S0 (unnormalized) vs S1 (resource-normalized) "
          f"static rank-1 selector, {len(SEEDS)} seeds")
    print("=" * 90)

    rows = []
    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)

        top_j_s0, top_j_s1 = {}, {}
        per_mode_info = {}
        for m in range(N):
            est = run_windowed_calibration_normalized(f_diag, cal_rows, m)
            j_s0 = int(np.argmax(np.abs(est.rho) ** 2))
            j_s1 = int(est.top_channel_normalized(1)[0])
            top_j_s0[m] = j_s0
            top_j_s1[m] = j_s1
            per_mode_info[m] = dict(
                j_s0=j_s0, j_s1=j_s1, disagree=bool(j_s0 != j_s1),
                lambda_s0=complex(f_diag[j_s0]).__repr__(),
                lambda_s1=complex(f_diag[j_s1]).__repr__(),
                abs_lambda_s0=float(np.abs(f_diag[j_s0])),
                abs_lambda_s1=float(np.abs(f_diag[j_s1])),
                raw_rho2_s0=float(np.abs(est.rho[j_s0]) ** 2),
                raw_rho2_s1=float(np.abs(est.rho[j_s1]) ** 2),
                E_s0=float(est.E[j_s0]), E_s1=float(est.E[j_s1]),
                R_s0=float(est.R()[j_s0]), R_s1=float(est.R()[j_s1]))

        for r_idx, row in enumerate(test_rows):
            G_bptt, G_online = row["G_bptt"], row["G_online"]
            # deploy_selected_channel returns a scalar Ghat directly per
            # (top_j, B1_col, test_row, m) -- call per mode, per test row
            G_s0 = np.array([deploy_selected_channel(
                f_diag, top_j_s0[m], B1[:, m], row, m) for m in range(N)])
            G_s1 = np.array([deploy_selected_channel(
                f_diag, top_j_s1[m], B1[:, m], row, m) for m in range(N)])
            c_s0, c_s1 = cos_np(G_s0, G_bptt), cos_np(G_s1, G_bptt)
            c_on = cos_np(G_online, G_bptt)
            rows.append(dict(
                seed=seed, test_traj=r_idx,
                cos_s0=c_s0, cos_s1=c_s1, cos_online=c_on,
                rel_err_s0=relerr_np(G_s0, G_bptt),
                rel_err_s1=relerr_np(G_s1, G_bptt)))
        n_disagree = sum(1 for m in range(N) if per_mode_info[m]["disagree"])
        print(f"seed {seed}: S0/S1 disagree on {n_disagree}/{N} modes; "
              f"median cos this seed: S0="
              f"{np.median([x['cos_s0'] for x in rows if x['seed'] == seed]):.3f}"
              f"  S1="
              f"{np.median([x['cos_s1'] for x in rows if x['seed'] == seed]):.3f}")
        rows[-1]["per_mode_info"] = per_mode_info    # attach once/seed

    print("-" * 90)
    med_s0 = np.median([x["cos_s0"] for x in rows])
    med_s1 = np.median([x["cos_s1"] for x in rows])
    med_on = np.median([x["cos_online"] for x in rows])
    total_disagree = sum(
        1 for row in rows if "per_mode_info" in row
        for m in row["per_mode_info"] if row["per_mode_info"][m]["disagree"])
    total_modes = sum(N for row in rows if "per_mode_info" in row)
    print(f"S0 (unnormalized) median cos: {med_s0:.4f}")
    print(f"S1 (normalized)   median cos: {med_s1:.4f}")
    print(f"online baseline   median cos: {med_on:.4f}")
    print(f"S0/S1 disagreement rate: {total_disagree}/{total_modes} modes "
          f"({100 * total_disagree / total_modes:.1f}%)")
    print("(compare: streaming-S0 benchmark from B4C = 0.926 median; "
          "exact-teacher-optimized rank-1 diagnostic from B3B = 0.992)")

    gate_meaningful = (med_s1 - med_s0) > 0.02   # a small, honest bar:
    # "meaningfully improves" is interpreted as more than a rounding-
    # level difference; not tuned post-hoc, stated before reporting
    # results below
    print(f"gate: S1 - S0 = {med_s1 - med_s0:+.4f}  "
          f"meaningful improvement (>0.02): {gate_meaningful}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ,
                                    n_test_traj=N_TEST_TRAJ, eps=EPS),
              rows=rows, median_cos_s0=med_s0, median_cos_s1=med_s1,
              median_cos_online=med_on,
              disagreement_rate=total_disagree / total_modes,
              gate_meaningful_improvement=bool(gate_meaningful))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b8c_static_benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
