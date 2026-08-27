"""B9.1 -- selector/predictability diagnostic (Parts 2-5). Diagnostic
only: no prediction-correction/resurrection mechanism, no training-loop
change, no S5. Builds on B9 (credit_memory/b9_oracle_utility_audit.py)
and the Part-1 leak fix (StreamingRelevance.reset_filter, applied in
credit_memory/streaming.py, b5_train.py, b6_prospective_tracking.py).

Same 8 seeds, N=6, T=60, BATCH=8, N_CAL_TRAJ=4, N_TEST_TRAJ=4 static
protocol as B3/B4/B8/B9 (imported directly).

S is taken as the EMPTY set (not "the current rank-1 pick" as in B9's
own script) so U_j(S) = |G|^2 - |G-gamma_j|^2 is the utility of
selecting candidate j AS THE SOLE rank-1 channel from scratch -- the
actual decision this whole line of work makes, and non-degenerate for
every candidate (B9's own S={top_j} definition makes U_{top_j} a
self-referential, hard-to-interpret quantity). The identity check is
re-verified under this S=empty definition.

Run:  python -m credit_memory.b9_1_selector_predictability
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.streaming import StreamingRelevance, run_windowed_calibration
from credit_memory.phase_b4c_streaming_rank1 import deploy_selected_channel
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np,
    relerr_np)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

TOPK = (1, 3, 5)


def lag1_autocorr(u_t):
    x0 = np.concatenate([u_t[:-1].real.ravel(), u_t[:-1].imag.ravel()])
    x1 = np.concatenate([u_t[1:].real.ravel(), u_t[1:].imag.ravel()])
    if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def exact_gamma_no_leak(f_diag, d, cal_rows, B1_col, m):
    """Sum of per-row (fresh-state) per_coordinate_contribution -- the
    B9.1-corrected, non-cross-trajectory-leaking exact oracle."""
    gamma = np.zeros(2 * N, np.complex128)
    for row in cal_rows:
        c_row = build_c_t(row["q1"], B1_col)
        u_row = row["Sa0"][:, :, m]
        g_row, _ = per_coordinate_contribution(f_diag, d, c_row, u_row)
        gamma += g_row
    return gamma


def topk_recall(order_desc, j_oracle, k):
    return bool(j_oracle in order_desc[:k])


def main() -> None:
    print("=" * 90)
    print(f"Phase B9.1: selector/predictability diagnostic, {len(SEEDS)} seeds")
    print("=" * 90)

    rows = []
    identity_errs = []
    cost_time = dict(rho=0.0, cheap=0.0)

    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)
        d = np.ones(2 * N, np.complex128)

        # cheap, shared-across-modes architecture quantities (Part 3/4)
        abs_lambda = np.abs(f_diag)                          # (2N,)
        j_orig_of = np.arange(2 * N) % N
        q1_cal_pooled = np.concatenate(
            [row["q1"].reshape(-1, N) for row in cal_rows], axis=0)
        t0 = time.perf_counter()
        abs_q_upper = np.sqrt(np.mean(np.abs(q1_cal_pooled) ** 2, axis=0))  # (N,)
        B_row_norm_upper = np.linalg.norm(B1, axis=1)                       # (N,)
        abs_q = abs_q_upper[j_orig_of]              # (2N,) broadcast P/Q
        B_row_norm = B_row_norm_upper[j_orig_of]     # (2N,)
        cost_time["cheap"] += time.perf_counter() - t0   # one-time, all modes

        top_j_rho_by_mode = {}
        top_j_oracle_by_mode = {}
        rho_by_mode = {}

        for m in range(N):
            B1_col = B1[:, m]

            t0 = time.perf_counter()
            est = run_windowed_calibration(f_diag, cal_rows, m)
            rho = est.rho
            cost_time["rho"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            gamma = exact_gamma_no_leak(f_diag, d, cal_rows, B1_col, m)
            E_m = float(np.sum(np.abs(
                np.concatenate([row["Sa0"][:, :, m] for row in cal_rows],
                               axis=0)) ** 2))
            ac1 = lag1_autocorr(np.concatenate(
                [row["Sa0"][:, :, m] for row in cal_rows], axis=0))
            cost_time["cheap"] += time.perf_counter() - t0

            G = np.sum(gamma)
            U = 2 * np.real(np.conj(G) * gamma) - np.abs(gamma) ** 2  # S=empty
            U_direct = np.abs(G) ** 2 - np.abs(G - gamma) ** 2
            identity_errs.append(float(np.max(np.abs(U - U_direct))))

            j_oracle = int(np.argmax(U))
            j_rho = int(np.argmax(np.abs(rho)))
            rng = np.random.RandomState(seed * 100 + m)
            j_random = int(rng.randint(2 * N))

            order_rho = list(np.argsort(-np.abs(rho)))
            spearman_rho = stats.spearmanr(np.abs(rho), U).statistic

            top_j_rho_by_mode[m] = j_rho
            top_j_oracle_by_mode[m] = j_oracle
            rho_by_mode[m] = rho

            # cheap scores (Part 3) -- NO x_j/P_j/Q_j feature used
            scores = dict(
                A_random=rng.rand(2 * N),
                B_abs_lambda=abs_lambda,
                C_inv_ctrb=1.0 / (1.0 - abs_lambda ** 2 + 1e-12),
                D_q2B2=abs_q ** 2 * B_row_norm ** 2,
                E_full=(abs_q ** 2 * B_row_norm ** 2 * E_m
                       / (1.0 - abs_lambda ** 2 + 1e-12)),
            )
            score_metrics = {}
            for name, sc in scores.items():
                order = list(np.argsort(-sc))
                sp = stats.spearmanr(sc, U).statistic
                regret = float(U[j_oracle] - U[order[0]])
                score_metrics[name] = dict(
                    spearman=float(sp) if not np.isnan(sp) else None,
                    top1_hit=topk_recall(order, j_oracle, 1),
                    top3_hit=topk_recall(order, j_oracle, 3),
                    top5_hit=topk_recall(order, j_oracle, 5),
                    regret=regret)

            rows.append(dict(
                seed=seed, mode=m, j_oracle=j_oracle, j_rho=j_rho,
                j_random=j_random,
                U=U.tolist(), U_max=float(U.max()), U_min=float(U.min()),
                U_gap_oracle_minus_median=float(U[j_oracle]
                                                - np.median(U)),
                spearman_rho=float(spearman_rho)
                if not np.isnan(spearman_rho) else None,
                top1_hit_rho=topk_recall(order_rho, j_oracle, 1),
                top3_hit_rho=topk_recall(order_rho, j_oracle, 3),
                top5_hit_rho=topk_recall(order_rho, j_oracle, 5),
                regret_rho=float(U[j_oracle] - U[j_rho]),
                regret_random=float(U[j_oracle] - U[j_random]),
                E_m=E_m, ac1=ac1,
                cheap_scores=score_metrics))

        # actual gradient quality on held-out test trajectories:
        # oracle-pick vs rho-pick vs online baseline vs BPTT
        for row in test_rows:
            G_hat_rho = np.zeros(N, np.complex128)
            G_hat_oracle = np.zeros(N, np.complex128)
            for m in range(N):
                G_hat_rho[m] = deploy_selected_channel(
                    f_diag, top_j_rho_by_mode[m], B1[:, m], row, m)
                G_hat_oracle[m] = deploy_selected_channel(
                    f_diag, top_j_oracle_by_mode[m], B1[:, m], row, m)
            G_bptt, G_online = row["G_bptt"], row["G_online"]
            rows.append(dict(
                seed=seed, mode=None, test_grad_row=True,
                cos_rho=cos_np(G_hat_rho, G_bptt),
                cos_oracle=cos_np(G_hat_oracle, G_bptt),
                cos_online=cos_np(G_online, G_bptt),
                relerr_rho=relerr_np(G_hat_rho, G_bptt),
                relerr_oracle=relerr_np(G_hat_oracle, G_bptt)))

        # Part 5: shared-active-set-across-modes, report-only analysis.
        # aggregate score per candidate j: sum over m of |rho[m][j]|^2
        agg = np.zeros(2 * N)
        for m in range(N):
            agg += np.abs(rho_by_mode[m]) ** 2
        j_shared = int(np.argmax(agg))
        for row in test_rows:
            G_hat_shared = np.zeros(N, np.complex128)
            for m in range(N):
                G_hat_shared[m] = deploy_selected_channel(
                    f_diag, j_shared, B1[:, m], row, m)
            rows.append(dict(seed=seed, mode=None, shared_grad_row=True,
                             j_shared=j_shared,
                             cos_shared=cos_np(G_hat_shared, row["G_bptt"])))

        print(f"seed {seed}: per-mode rho picks {top_j_rho_by_mode}  "
              f"oracle picks {top_j_oracle_by_mode}  shared j*={j_shared}")

    # ---------------- aggregate summary ----------------
    per_mode_rows = [r for r in rows if r.get("mode") is not None]
    test_rows_ = [r for r in rows if r.get("test_grad_row")]
    shared_rows_ = [r for r in rows if r.get("shared_grad_row")]

    max_identity_err = max(identity_errs)
    print("-" * 90)
    print(f"Identity check (S=empty): max abs err = {max_identity_err:.2e} "
         f"({'PASS' if max_identity_err < 1e-9 else 'FAIL'})")

    def med(key, src=per_mode_rows):
        return float(np.median([r[key] for r in src]))

    print(f"median Spearman(|rho|, U): {med('spearman_rho'):.3f}")
    print(f"top1/3/5 oracle hit rate (rho ranking): "
         f"{np.mean([r['top1_hit_rho'] for r in per_mode_rows]):.3f} / "
         f"{np.mean([r['top3_hit_rho'] for r in per_mode_rows]):.3f} / "
         f"{np.mean([r['top5_hit_rho'] for r in per_mode_rows]):.3f}")
    print(f"median regret_rho: {med('regret_rho'):.4f}  "
         f"median regret_random: {med('regret_random'):.4f}")
    print(f"median U gap (oracle - median candidate): "
         f"{med('U_gap_oracle_minus_median'):.4f}")

    print(f"held-out gradient quality: median cos_rho={med('cos_rho', test_rows_):.4f}"
         f"  cos_oracle={med('cos_oracle', test_rows_):.4f}"
         f"  cos_online={med('cos_online', test_rows_):.4f}")

    print(f"shared-active-set (Part 5): median cos_shared="
         f"{med('cos_shared', shared_rows_):.4f}  vs per-mode-optimal "
         f"cos_rho={med('cos_rho', test_rows_):.4f}")

    print("-" * 90)
    print("cheap-score summary (median across all (seed,mode) rows):")
    score_names = list(per_mode_rows[0]["cheap_scores"].keys())
    cheap_summary = {}
    for name in score_names:
        sps = [r["cheap_scores"][name]["spearman"] for r in per_mode_rows
              if r["cheap_scores"][name]["spearman"] is not None]
        t1 = np.mean([r["cheap_scores"][name]["top1_hit"] for r in per_mode_rows])
        t3 = np.mean([r["cheap_scores"][name]["top3_hit"] for r in per_mode_rows])
        t5 = np.mean([r["cheap_scores"][name]["top5_hit"] for r in per_mode_rows])
        reg = np.median([r["cheap_scores"][name]["regret"] for r in per_mode_rows])
        cheap_summary[name] = dict(median_spearman=float(np.median(sps)),
                                   top1=float(t1), top3=float(t3),
                                   top5=float(t5), median_regret=float(reg))
        print(f"  {name:12s} spearman={np.median(sps):+.3f}  "
             f"top1/3/5={t1:.2f}/{t3:.2f}/{t5:.2f}  median_regret={reg:.4f}")

    print("-" * 90)
    print(f"measured wall-clock (all seeds*modes): rho(O(2N) propagation)="
         f"{cost_time['rho']:.3f}s   cheap scores (architecture/shared-only)="
         f"{cost_time['cheap']:.4f}s   ratio={cost_time['rho']/max(cost_time['cheap'],1e-9):.1f}x")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                         n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ),
              identity_check_pass=bool(max_identity_err < 1e-9),
              max_identity_err=max_identity_err,
              median_spearman_rho=med('spearman_rho'),
              top1_hit_rho=float(np.mean([r['top1_hit_rho'] for r in per_mode_rows])),
              top3_hit_rho=float(np.mean([r['top3_hit_rho'] for r in per_mode_rows])),
              top5_hit_rho=float(np.mean([r['top5_hit_rho'] for r in per_mode_rows])),
              median_regret_rho=med('regret_rho'),
              median_regret_random=med('regret_random'),
              median_cos_rho=med('cos_rho', test_rows_),
              median_cos_oracle=med('cos_oracle', test_rows_),
              median_cos_online=med('cos_online', test_rows_),
              median_cos_shared=med('cos_shared', shared_rows_),
              cheap_summary=cheap_summary,
              measured_time_seconds=cost_time,
              rows=rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_1_selector_predictability_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
