"""B3C -- cross-correlation/cross-spectrum relevance constructions R0-R3,
tested at r in {1,2,4} on the same seeds/calibration/test split as B2/B3B.

R0  existing baseline: standard B2 controllability/observability balanced
    truncation (Wc analytic, Wo from readout energy S=E[c c^dagger]).
    Reused from phase_b2bc_hankel_truncation's own machinery, recomputed
    here on this file's own calibration draw for a self-contained
    artifact (numerically identical protocol to B2C).

R1  zero-lag cross relevance: rank the 2N ORIGINAL (P,Q) coordinates
    directly by their own exact contribution g_p = sum_t conj(c_t[p])
    x_t[p] (a genuine zero-lag empirical cross-covariance between the
    causal state coordinate and the readout, NOT a readout-energy-only
    score); keep the top-r coordinates AS-IS (no basis rotation -- they
    are already F's own eigenbasis, so this is the cheapest possible
    construction).

R2  lagged cross relevance: cross-Gramian M_cross = sum_k outer(F^k d,
    r_k) (credit_memory/lagcorr.py), whose trace is exactly G; eigen-
    decompose, keep the r eigenpairs with largest |eigenvalue|,
    Galerkin-project (F,d) onto that (generally rotated, non-orthogonal)
    subspace. Depends on BOTH u and c jointly at every lag, not just
    readout energy.

R3  frequency-domain consistency check: recomputes R1's per-coordinate
    g_p via the cross-spectrum between u and c and the known transfer
    function d[p]/(1-f_p e^{-iw}), using a circular (T-point) DFT.
    Reports agreement with R1's time-domain g_p (expected to differ at
    the "finite-window/circular-convolution" level given |a1[j]| up to
    ~0.995, i.e. slow decay relative to T); evaluated at the same
    r-ladder using R3's own (frequency-recomputed) coordinate ranking.

No free learned poles anywhere in R0-R3 (F is always the architecture's
own a1/conj(a1)); R2/R3 only ever rotate/select within the space spanned
by (F^k d) directions, never introduce new dynamics.

Run:  python -m credit_memory.phase_b3c_relevance
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from credit_memory.hankel import (build_F, build_c_t, analytic_Wc,
                                  estimate_S, solve_Wo, balanced_transform,
                                  reduced_system, reduced_gradient)
from credit_memory.lagcorr import (lagged_r_k, per_coordinate_contribution,
                                   cross_gramian, top_r_eigen_reduction,
                                   reduced_gradient_general, freq_domain_g_p)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np,
    relerr_np)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

R_LADDER = [1, 2, 4]
KMAX_R2 = T - 1        # full-horizon cross-Gramian (no lag truncation)


def r0_reduction(f_diag, Wc, S, r):
    Wo = solve_Wo(f_diag, S)
    T_bal, Tinv_bal, _ = balanced_transform(Wc, Wo)
    F_r, d_r = reduced_system(f_diag, np.ones(2 * N, np.complex128),
                              T_bal, Tinv_bal, r)
    return dict(kind="balanced", F_r=F_r, d_r=d_r, T_bal=T_bal)


def r1_reduction(f_diag, g_p, r):
    order = np.argsort(-np.abs(g_p))[:r]
    F_r = np.diag(f_diag[order])
    d_r = np.ones(r, np.complex128)
    return dict(kind="coordinate_select", order=order, F_r=F_r, d_r=d_r)


def r2_reduction(f_diag, M_cross, r):
    F = np.diag(f_diag)
    d = np.ones(2 * N, np.complex128)
    F_r, d_r, V_r, eigvals = top_r_eigen_reduction(M_cross, F, d, r)
    return dict(kind="cross_gramian", F_r=F_r, d_r=d_r, V_r=V_r,
               eigvals=eigvals)


def eval_reduction(red, m, B1_col, Sa0_m, q1, G_bptt_m, G_online_m):
    kind = red["kind"]
    if kind == "balanced":
        _, G_hat = reduced_gradient(red["F_r"], red["d_r"], red["T_bal"],
                                    Sa0_m, q1, B1_col,
                                    red["d_r"].shape[0])
    elif kind == "coordinate_select":
        order = red["order"]
        r = len(order)
        z = np.zeros((Sa0_m.shape[0], Sa0_m.shape[1], r), np.complex128)
        z_prev = np.zeros((Sa0_m.shape[1], r), np.complex128)
        for t in range(Sa0_m.shape[0]):
            z_prev = red["F_r"] @ z_prev.T
            z_prev = z_prev.T + Sa0_m[t][:, None] * red["d_r"][None, :]
            z[t] = z_prev
        c_t = build_c_t(q1, B1_col)                     # (T,B,2N)
        c_sel = c_t[:, :, order]                          # (T,B,r)
        g_t = np.sum(np.conj(c_sel) * z, axis=-1)
        G_hat = g_t.sum()
    elif kind == "cross_gramian":
        c_t = build_c_t(q1, B1_col)
        _, G_hat = reduced_gradient_general(red["F_r"], red["d_r"],
                                            red["V_r"], Sa0_m, c_t)
    else:
        raise ValueError(kind)
    c_hat = float(np.abs(np.conj(G_hat) * G_bptt_m)
                  / (abs(G_hat) * abs(G_bptt_m) + 1e-300))
    return G_hat, c_hat


def main() -> None:
    print("=" * 78)
    print(f"Phase B3C: relevance constructions R0-R3, {len(SEEDS)} seeds, "
          f"r in {R_LADDER}")
    print("=" * 78)

    rows = []       # one row per (rule, r, seed, test_traj), full-vector metrics
    freq_agreement = []

    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)
        Wc = analytic_Wc(f_diag)
        d = np.ones(2 * N, np.complex128)

        q1_cal_pooled = np.concatenate(
            [row["q1"].reshape(-1, N) for row in cal_rows], axis=0)
        Sa0_cal_pooled = {m: np.concatenate(
            [row["Sa0"][:, :, m].reshape(-1) for row in cal_rows])
            for m in range(N)}

        per_mode_red = {r: {} for r in R_LADDER}
        for m in range(N):
            S = estimate_S(q1_cal_pooled, B1[:, m])

            c_t_pool = np.concatenate(
                [build_c_t(row["q1"], B1[:, m]) for row in cal_rows], axis=0)
            u_t_pool = np.concatenate(
                [row["Sa0"][:, :, m] for row in cal_rows], axis=0)
            g_p, _ = per_coordinate_contribution(f_diag, d, c_t_pool,
                                                 u_t_pool)
            rk = lagged_r_k(c_t_pool, u_t_pool, K=min(KMAX_R2,
                                                       c_t_pool.shape[0] - 1))
            M_cross = cross_gramian(f_diag, d, rk)

            g_p_freq = freq_domain_g_p(f_diag, d, u_t_pool, c_t_pool)
            agree = float(np.linalg.norm(g_p_freq - g_p)
                          / (np.linalg.norm(g_p) + 1e-300))
            freq_agreement.append(dict(seed=seed, mode=m,
                                       rel_err_freq_vs_time=agree))

            for r in R_LADDER:
                per_mode_red[r][m] = dict(
                    R0=r0_reduction(f_diag, Wc, S, r),
                    R1=r1_reduction(f_diag, g_p, r),
                    R2=r2_reduction(f_diag, M_cross, r),
                    R3=r1_reduction(f_diag, g_p_freq, r))

        for r in R_LADDER:
            for rule in ("R0", "R1", "R2", "R3"):
                for t_idx, row in enumerate(test_rows):
                    G_hat = np.zeros(N, np.complex128)
                    for m in range(N):
                        red = per_mode_red[r][m][rule]
                        G_hat[m], _ = eval_reduction(
                            red, m, B1[:, m], row["Sa0"][:, :, m],
                            row["q1"], row["G_bptt"][m], row["G_online"][m])
                    G_bptt, G_online = row["G_bptt"], row["G_online"]
                    c_hat = cos_np(G_hat, G_bptt)
                    c_on = cos_np(G_online, G_bptt)
                    gap = max(1.0 - c_on, 1e-12)
                    rows.append(dict(
                        seed=seed, rule=rule, r=r, test_traj=t_idx,
                        cos=c_hat, cos_online=c_on,
                        rel_err=relerr_np(G_hat, G_bptt),
                        norm_ratio=float(np.linalg.norm(G_hat)
                                         / (np.linalg.norm(G_bptt)
                                            + 1e-300)),
                        frac_gap_recovered=float((c_hat - c_on) / gap)))
        print(f"seed {seed}: R0 r=4 cos="
              f"{np.median([x['cos'] for x in rows if x['seed'] == seed and x['rule'] == 'R0' and x['r'] == 4]):.3f}"
              f"  R1 r=1 cos="
              f"{np.median([x['cos'] for x in rows if x['seed'] == seed and x['rule'] == 'R1' and x['r'] == 1]):.3f}"
              f"  R2 r=1 cos="
              f"{np.median([x['cos'] for x in rows if x['seed'] == seed and x['rule'] == 'R2' and x['r'] == 1]):.3f}")

    print("-" * 78)
    agg = {}
    for rule in ("R0", "R1", "R2", "R3"):
        for r in R_LADDER:
            sub = [x for x in rows if x["rule"] == rule and x["r"] == r]
            agg[(rule, r)] = dict(
                median_cos=float(np.median([x["cos"] for x in sub])),
                median_rel_err=float(np.median([x["rel_err"] for x in sub])),
                median_norm_ratio=float(np.median(
                    [x["norm_ratio"] for x in sub])),
                median_frac_gap_recovered=float(np.median(
                    [x["frac_gap_recovered"] for x in sub])))
            print(f"{rule} r={r}: median cos={agg[(rule, r)]['median_cos']:.4f}"
                  f"  frac_gap={agg[(rule, r)]['median_frac_gap_recovered']:.4f}")

    print(f"R1-vs-R3 (time vs frequency domain) median rel disagreement: "
          f"{np.median([x['rel_err_freq_vs_time'] for x in freq_agreement]):.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(
        git=git,
        config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                   n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ,
                   r_ladder=R_LADDER, kmax_r2=KMAX_R2),
        rows=rows,
        aggregate={f"{rule}_r{r}": agg[(rule, r)]
                  for rule in ("R0", "R1", "R2", "R3") for r in R_LADDER},
        freq_domain_agreement=freq_agreement,
        median_freq_vs_time_disagreement=float(np.median(
            [x["rel_err_freq_vs_time"] for x in freq_agreement])),
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_b3c_relevance_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
