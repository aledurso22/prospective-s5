"""Phase B10.2 -- two theory-to-algorithm gaps left open by B10/B10.1.
No new training algorithm, no prediction-correction, no event triggers,
no feedback alignment, no new persistent training arm, no S5.

Part I: why does the practical selector |rho[j,m]| rank candidates
almost as well as the ideal single-channel utility U[j,m]?

Part II: can the low-dimensional routed-relevance geometry (B10/B10.1)
be recovered from O(r(M+N)) structured row/column measurements instead
of the full O(MN) calibration, without ever using unseen matrix
entries to choose which rows/columns to reveal?

D_m (the "exact missing cross-layer correction," per-timestep, before
summing over t/batch) is teacher.py's own g_t_bptt[:,:,m] -- the exact
BPTT per-(t,b) contribution. gamma[j,m] (per-timestep, before summing)
is the same per-(t,b) product per_coordinate_contribution already sums
internally: conj(c_{j,t,b,m}) * x_{j,t,b,m}. Both are TB-length vectors
(TB = N_CAL_TRAJ*T*BATCH, each trajectory's own filter independently
zero-initialized, then concatenated -- concatenation is safe here
because gamma/D are plain per-step signals, not recursive states, so
there is no cross-trajectory leakage to guard against for this
computation specifically). rho[j,m] = sum(gamma[j,m]) recovers the
established scalar exactly.

Run:  python -m credit_memory.b10_2_selector_and_calibration_theory
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.b9_2_shared_pool import best_pool_exact, pool_most_frequent
from credit_memory.b10_tangent_adjoint_theory import (
    forward_filter, low_rank_trunc, decision_metrics, effective_ranks,
    qr_pivot_indices, cur_reconstruct, part_e_margin_test, build_factors,
    direct_routed)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b10_2")

K_LIST = [1, 2, 4, 6]
R_LIST = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Part I.A: D_m, gamma[j,m] as TB-length vectors; |rho|, ||gamma||,
# alignment cos(theta), ideal utility U, single-channel error e.
# ---------------------------------------------------------------------------
def exact_gamma_no_leak(f_diag, d, rows, B1_col, m):
    """Scalar per-candidate contribution gamma[j,m] (2N,), leak-fixed
    (B9.1): each calibration row's own filter state starts at zero, no
    cross-trajectory leakage. gamma[j,m] IS rho[j,m] -- the same object
    used throughout B9-B10 -- summed here from a fresh per-row filter
    for consistency with the leak fix."""
    gamma = np.zeros(2 * N, np.complex128)
    for row in rows:
        c_row = build_c_t(row["q1"], B1_col)
        u_row = row["Sa0"][:, :, m]
        g_row, _ = per_coordinate_contribution(f_diag, d, c_row, u_row)
        gamma += g_row
    return gamma


def compute_D_and_gamma(rows, f_diag, B1, m):
    """SCALAR framework (matches B9.1/B9.4's own S=empty oracle utility
    exactly): D_m = G[m] = sum_j gamma[j,m] (the exact full-bank
    gradient), gamma[j,m] = exact_gamma_no_leak's per-candidate output
    -- literally the same scalar as rho[j,m] used throughout B9-B10.
    cos_theta here is PHASE alignment between two complex scalars, not
    a high-dimensional vector angle."""
    d = np.ones(2 * N, np.complex128)
    gamma = exact_gamma_no_leak(f_diag, d, rows, B1[:, m], m)   # (2N,)
    D_m = gamma.sum()
    return D_m, gamma


def part_a_stats(rows, f_diag, B1):
    """Per-mode arrays over all 2N candidates: rho (=gamma, scalar),
    norm_gamma(=|rho|, kept as a separate name only to match the task's
    requested quantity list -- see note in PHASE_B10_2.md), inner,
    cos_theta (phase alignment of two complex scalars), U, e."""
    out = {}
    for m in range(N):
        D_m, gamma = compute_D_and_gamma(rows, f_diag, B1, m)      # gamma: (2N,)
        rho = gamma                                                # same object
        norm_gamma = np.abs(gamma)
        norm_D = np.abs(D_m)
        inner = np.real(gamma * np.conj(D_m))       # Re[conj(D_m) gamma[j]]
        cos_theta = inner / (norm_D * norm_gamma + 1e-300)
        U = 2 * inner - norm_gamma ** 2
        e = np.sqrt(np.maximum(norm_D ** 2 - 2 * inner + norm_gamma ** 2, 0.0))
        out[m] = dict(rho=rho, abs_rho=np.abs(rho), norm_gamma=norm_gamma,
                      norm_D=norm_D, inner=inner, cos_theta=cos_theta,
                      U=U, e=e)
    return out


def safe_spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r = stats.spearmanr(a, b).statistic
    return None if np.isnan(r) else float(r)


def safe_pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r = np.corrcoef(a, b)[0, 1]
    return None if np.isnan(r) else float(r)


def pool_regret_from_score(U_true, score, K):
    pool = best_pool_exact(score, K)
    true_util = float(sum(U_true[list(best_pool_exact(U_true, K)), m].max()
                          for m in range(N)))
    achieved = float(sum(U_true[list(pool), m].max() for m in range(N)))
    return true_util - achieved


def part_a_correlations(stats_by_mode, K=4):
    rows_out = []
    U_true_mat = np.stack([stats_by_mode[m]["U"] for m in range(N)], axis=1)
    for m in range(N):
        s = stats_by_mode[m]
        rows_out.append(dict(
            mode=m,
            spearman_absrho_U=safe_spearman(s["abs_rho"], s["U"]),
            pearson_absrho_U=safe_pearson(s["abs_rho"], s["U"]),
            spearman_absrho_normgamma=safe_spearman(s["abs_rho"], s["norm_gamma"]),
            spearman_absrho_costheta=safe_spearman(s["abs_rho"], s["cos_theta"]),
            spearman_normgamma_U=safe_spearman(s["norm_gamma"], s["U"]),
            top1_agree_absrho_U=bool(np.argmax(s["abs_rho"]) == np.argmax(s["U"])),
            top1_agree_normgamma_U=bool(np.argmax(s["norm_gamma"]) == np.argmax(s["U"])),
            topK_recall_absrho_U=len(set(np.argsort(-s["abs_rho"])[:K])
                                     & set(np.argsort(-s["U"])[:K])) / K))
    pool_regret = dict(
        absrho=pool_regret_from_score(U_true_mat, np.stack(
            [stats_by_mode[m]["abs_rho"] for m in range(N)], axis=1), K),
        norm_gamma=pool_regret_from_score(U_true_mat, np.stack(
            [stats_by_mode[m]["norm_gamma"] for m in range(N)], axis=1), K),
        ideal=0.0)
    return rows_out, pool_regret


# ---------------------------------------------------------------------------
# Part I.B: narrow positive cone hypothesis
# ---------------------------------------------------------------------------
def part_b_cone(stats_by_mode, top_frac=0.25):
    all_cos, top_cos, bottom_cos = [], [], []
    ratio_gamma_D = []
    overshoot_frac = []
    for m in range(N):
        s = stats_by_mode[m]
        n2 = len(s["abs_rho"])
        order = np.argsort(-s["abs_rho"])
        n_top = max(1, int(round(top_frac * n2)))
        top_idx, bot_idx = order[:n_top], order[-n_top:]
        all_cos.extend(s["cos_theta"].tolist())
        top_cos.extend(s["cos_theta"][top_idx].tolist())
        bottom_cos.extend(s["cos_theta"][bot_idx].tolist())
        ratio_gamma_D.extend((s["norm_gamma"] / (s["norm_D"] + 1e-300)).tolist())
        overshoot_frac.append(float(np.mean(s["U"] < 0)))
    return dict(
        all_cos_theta_mean=float(np.mean(all_cos)), all_cos_theta_std=float(np.std(all_cos)),
        top_cos_theta_mean=float(np.mean(top_cos)), top_cos_theta_std=float(np.std(top_cos)),
        bottom_cos_theta_mean=float(np.mean(bottom_cos)), bottom_cos_theta_std=float(np.std(bottom_cos)),
        mean_ratio_gamma_over_D=float(np.mean(ratio_gamma_D)),
        median_ratio_gamma_over_D=float(np.median(ratio_gamma_D)),
        mean_frac_negative_utility=float(np.mean(overshoot_frac)))


# ---------------------------------------------------------------------------
# Part I.C: decompose |rho|-vs-U ranking disagreements
# ---------------------------------------------------------------------------
def part_c_misrank_decomposition(stats_by_mode, f_diag):
    records = []
    for m in range(N):
        s = stats_by_mode[m]
        n2 = len(s["abs_rho"])
        for j, k in itertools.combinations(range(n2), 2):
            rho_says_j = s["abs_rho"][j] > s["abs_rho"][k]
            U_says_j = s["U"][j] > s["U"][k]
            if rho_says_j == U_says_j:
                continue
            d_cos = float(s["cos_theta"][j] - s["cos_theta"][k])
            d_mag = float(s["norm_gamma"][j] - s["norm_gamma"][k])
            same_upper_mode = (j % N) == (k % N)   # P/Q pair of the same pole
            records.append(dict(mode=m, j=j, k=k, d_cos_theta=d_cos,
                                d_norm_gamma=d_mag,
                                d_overshoot=float(s["norm_gamma"][j] ** 2
                                                  - s["norm_gamma"][k] ** 2),
                                same_upper_mode_PQ_pair=same_upper_mode))
    if not records:
        return dict(n_misranked_pairs=0)
    d_cos = np.array([r["d_cos_theta"] for r in records])
    d_mag = np.array([r["d_norm_gamma"] for r in records])
    frac_pq_pair = float(np.mean([r["same_upper_mode_PQ_pair"] for r in records]))
    # crude attribution: which factor's sign more often "explains" why
    # rho and U disagree (i.e. cos_theta difference points opposite to
    # the magnitude difference, so the two effects fight each other)
    opposing_signs = float(np.mean(np.sign(d_cos) != np.sign(d_mag)))
    return dict(n_misranked_pairs=len(records),
               mean_abs_d_cos_theta=float(np.mean(np.abs(d_cos))),
               mean_abs_d_norm_gamma=float(np.mean(np.abs(d_mag))),
               frac_same_upper_mode_PQ_pair=frac_pq_pair,
               frac_opposing_cos_and_magnitude_signs=opposing_signs)


# ---------------------------------------------------------------------------
# Part I.D: conditional ranking theorem, derived and tested empirically.
#
# U_j - U_k = (||gamma_j||-||gamma_k||)[2||D||c - (||gamma_j||+||gamma_k||)]
#            + 2||D||(delta_j ||gamma_j|| - delta_k ||gamma_k||)
# where cos_theta_i = c + delta_i, |delta_i|<=delta, ||gamma_i||<=g_max.
# If 2||D||c - 2 g_max > 0 (the "not yet overshooting" regime) then the
# bracket term has the same sign as (||gamma_j||-||gamma_k||), and the
# error term is bounded by 4||D|| delta g_max. So under bounded angular
# spread and a magnitude regime below the overshoot threshold,
# ||gamma_j||>||gamma_k|| => U_j>U_k unless the bracket-term margin is
# smaller than the error bound.
# ---------------------------------------------------------------------------
def part_d_conditional_theorem(stats_by_mode):
    all_cos, all_g, all_D = [], [], []
    for m in range(N):
        s = stats_by_mode[m]
        all_cos.append(s["cos_theta"])
        all_g.append(s["norm_gamma"])
        all_D.append(np.full_like(s["cos_theta"], s["norm_D"]))
    cos_all = np.concatenate(all_cos)
    g_all = np.concatenate(all_g)
    D_all = np.concatenate(all_D)

    c = float(np.median(cos_all))
    delta = float(np.percentile(np.abs(cos_all - c), 90))
    g_max = float(np.percentile(g_all, 90))
    D_typ = float(np.median(D_all))
    margin_condition = 2 * D_typ * c - 2 * g_max   # >0 needed

    n_checked, n_theorem_applies, n_theorem_correct = 0, 0, 0
    for m in range(N):
        s = stats_by_mode[m]
        n2 = len(s["abs_rho"])
        for j, k in itertools.combinations(range(n2), 2):
            gj, gk = s["norm_gamma"][j], s["norm_gamma"][k]
            cj, ck = s["cos_theta"][j], s["cos_theta"][k]
            if abs(cj - c) > delta or abs(ck - c) > delta:
                continue
            if gj > g_max or gk > g_max:
                continue
            n_checked += 1
            bracket = 2 * D_typ * c - (gj + gk)
            error_bound = 4 * D_typ * delta * g_max
            predicted_margin = abs(gj - gk) * bracket
            if predicted_margin <= error_bound:
                continue    # theorem doesn't confidently apply here
            n_theorem_applies += 1
            predicted_sign = np.sign((gj - gk) * bracket)
            actual_sign = np.sign(s["U"][j] - s["U"][k])
            if predicted_sign == actual_sign:
                n_theorem_correct += 1

    return dict(c=c, delta=delta, g_max=g_max, D_typ=D_typ,
               margin_condition_2Dc_minus_2gmax=margin_condition,
               regime_is_pre_overshoot=bool(margin_condition > 0),
               n_pairs_checked=n_checked, n_theorem_applies=n_theorem_applies,
               n_theorem_correct=n_theorem_correct,
               theorem_accuracy=(n_theorem_correct / n_theorem_applies
                                if n_theorem_applies else None))


# ===========================================================================
# PART II -- calibration theory: can the routed relevance matrix be
# recovered from O(r(M+N)) row/column measurements instead of O(MN)?
# ===========================================================================
def full_state_cost(n_rows, n_cols):
    return n_rows * n_cols   # M x N_lower


def sampled_state_cost(r_rows, r_cols, n_rows, n_cols):
    return r_rows * n_cols + r_cols * n_rows


def eval_reconstruction(R_true, R_hat, K=4):
    s_true, s_hat = np.abs(R_true), np.abs(R_hat)
    fro = float(np.linalg.norm(s_hat - s_true) / (np.linalg.norm(s_true) + 1e-300))
    dm = decision_metrics(s_true, s_hat, K)
    margin = part_e_margin_test(s_true, s_hat)
    return dict(fro_rel_err=fro, **dm,
               frac_certified_by_margin=margin["frac_certified_by_margin"],
               frac_preserved=margin["frac_preserved_overall"])


# --- II.A: oracle sample-complexity curve (full-matrix QR pivots, used
# only to characterize the geometry -- not claimed as a cheap method) ---
def part_iiA_sample_complexity(R, K=4):
    n_rows, n_cols = R.shape
    curve = []
    for r_rows in range(1, n_rows + 1):
        for r_cols in range(1, n_cols + 1):
            row_idx = qr_pivot_indices(R, r_rows, axis="rows")
            col_idx = qr_pivot_indices(R, r_cols, axis="cols")
            R_hat = cur_reconstruct(R, row_idx, col_idx)
            ev = eval_reconstruction(R, R_hat, K)
            curve.append(dict(r_rows=r_rows, r_cols=r_cols, **ev))
    def first_satisfying(pred):
        ok = [c for c in curve if pred(c)]
        if not ok:
            return None
        best = min(ok, key=lambda c: sampled_state_cost(c["r_rows"], c["r_cols"], n_rows, n_cols))
        return dict(r_rows=best["r_rows"], r_cols=best["r_cols"],
                   state_cost_frac=sampled_state_cost(best["r_rows"], best["r_cols"],
                                                      n_rows, n_cols)
                   / full_state_cost(n_rows, n_cols))
    return dict(
        min_for_95pct_fro=first_satisfying(lambda c: c["fro_rel_err"] <= 0.05),
        min_for_99pct_fro=first_satisfying(lambda c: c["fro_rel_err"] <= 0.01),
        min_for_95pct_winner=first_satisfying(lambda c: c["winner_preserved"] >= 0.95),
        min_for_95pct_topk=first_satisfying(lambda c: c["mean_topk_recall"] >= 0.95),
        min_for_near_zero_regret=first_satisfying(lambda c: c["pool_regret"] <= 1e-6))


# --- II.B: cheap, no-leakage pivot discovery ---
def aca_no_leakage(R, r, rng, start_row=None):
    n_rows, n_cols = R.shape
    if start_row is None:
        start_row = int(rng.randint(n_rows))
    rows_sel, cols_sel = [], []
    cur_row = start_row
    for _ in range(r):
        row_vals = R[cur_row, :]
        approx_row = (cur_reconstruct(R, rows_sel, cols_sel)[cur_row, :]
                     if rows_sel and cols_sel else np.zeros(n_cols, complex))
        resid_row = row_vals - approx_row
        avail_cols = [c for c in range(n_cols) if c not in cols_sel]
        j_star = avail_cols[int(np.argmax(np.abs(resid_row[avail_cols])))]
        rows_sel.append(cur_row)
        cols_sel.append(j_star)
        col_vals = R[:, j_star]
        approx_col = cur_reconstruct(R, rows_sel, cols_sel)[:, j_star]
        resid_col = col_vals - approx_col
        avail_rows = [rr for rr in range(n_rows) if rr not in rows_sel]
        if avail_rows:
            cur_row = avail_rows[int(np.argmax(np.abs(resid_col[avail_rows])))]
    return rows_sel, cols_sel


def part_iiB_pivot_methods(R, U, f_diag, j_pool_deployed, rng, r=4, K=4):
    n_rows, n_cols = R.shape
    methods = {}
    methods["B0_oracle_QR"] = (qr_pivot_indices(R, r, axis="rows"),
                               qr_pivot_indices(R, r, axis="cols"))
    methods["B1_random"] = (sorted(rng.choice(n_rows, r, replace=False).tolist()),
                            sorted(rng.choice(n_cols, r, replace=False).tolist()))
    methods["B3_ACA_no_leakage"] = aca_no_leakage(R, r, rng)
    deployed_rows = list(j_pool_deployed)[:r]
    if len(deployed_rows) < r:
        extra = [j for j in range(n_rows) if j not in deployed_rows]
        deployed_rows += rng.choice(extra, r - len(deployed_rows), replace=False).tolist()
    methods["B4_deployed_pool_seeded"] = (
        sorted(deployed_rows), qr_pivot_indices(R, r, axis="cols"))
    Uc, Sc, Uch = np.linalg.svd(U, full_matrices=False)
    col_idx_from_U = qr_pivot_indices(U, r, axis="cols") if U.shape[1] >= r else list(range(U.shape[1]))
    methods["B5_U_seeded_cols"] = (qr_pivot_indices(R, r, axis="rows"),
                                   sorted(col_idx_from_U[:r]))

    out = {}
    for name, (row_idx, col_idx) in methods.items():
        R_hat = cur_reconstruct(R, row_idx, col_idx)
        ev = eval_reconstruction(R, R_hat, K)
        out[name] = dict(rows=row_idx, cols=col_idx,
                         state_cost=sampled_state_cost(len(set(row_idx)), len(set(col_idx)),
                                                       n_rows, n_cols),
                         state_cost_frac=sampled_state_cost(len(set(row_idx)), len(set(col_idx)),
                                                            n_rows, n_cols)
                         / full_state_cost(n_rows, n_cols), **ev)
    return out


# --- II.E: generic low-rank matrix completion baseline (same observed
# entries as CUR's row/col union, small ALS, diagnostic only) ---
def matrix_completion_als(R, observed_mask, r, iters=200, reg=1e-3, seed=0):
    rng = np.random.RandomState(seed)
    n_rows, n_cols = R.shape
    Xr = (rng.randn(n_rows, r) + 1j * rng.randn(n_rows, r)) * 0.1
    Yr = (rng.randn(n_cols, r) + 1j * rng.randn(n_cols, r)) * 0.1
    Robs = np.where(observed_mask, R, 0.0)
    for _ in range(iters):
        for i in range(n_rows):
            cols = np.where(observed_mask[i, :])[0]
            if len(cols) == 0:
                continue
            A = Yr[cols, :]
            b = Robs[i, cols]
            Xr[i, :] = np.linalg.lstsq(np.conj(A).T @ A + reg * np.eye(r),
                                       np.conj(A).T @ b, rcond=None)[0]
        for jc in range(n_cols):
            rws = np.where(observed_mask[:, jc])[0]
            if len(rws) == 0:
                continue
            A = Xr[rws, :]
            b = Robs[rws, jc]
            Yr[jc, :] = np.linalg.lstsq(np.conj(A).T @ A + reg * np.eye(r),
                                       np.conj(A).T @ b, rcond=None)[0]
    return Xr @ np.conj(Yr).T


def part_iiE_matrix_completion(R, row_idx, col_idx, r, K=4):
    n_rows, n_cols = R.shape
    mask = np.zeros((n_rows, n_cols), bool)
    mask[row_idx, :] = True
    mask[:, col_idx] = True
    R_mc = matrix_completion_als(R, mask, r)
    R_cur = cur_reconstruct(R, row_idx, col_idx)
    return dict(matrix_completion=eval_reconstruction(R, R_mc, K),
               cur_same_entries=eval_reconstruction(R, R_cur, K),
               n_observed_entries=int(mask.sum()), total_entries=n_rows * n_cols)


# --- II.F: temporal reuse of pivots across calibration events (warm-
# started linear algebra, NOT a predictor) ---
def run_online_training_snapshot(seed, target_step, N_, T_, BATCH_, DELAY_, LR_):
    from toyrig import ssm_rig as tcg
    from credit_memory.b5_train import draw_task_batch, loss_of
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = 2, N_, T_, DELAY_, BATCH_
    rng = np.random.RandomState(1000 + seed)
    params = tcg.init_params(seed)
    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    for step in range(1, target_step + 1):
        x, y = draw_task_batch(rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        g = tcg.flat_grads(G, params)
        m_ = b1_ * m_ + (1 - b1_) * g
        v_ = b2_ * v_ + (1 - b2_) * g ** 2
        flat = flat - LR_ * (m_ / (1 - b1_ ** step)) / (np.sqrt(v_ / (1 - b2_ ** step)) + eps)
        params = tcg.pack(params, flat)
    return params


def build_R_from_params(seed, params, cal_rng_seed=777):
    from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
    with set_l2_config(N, T, BATCH):
        rows = []
        for k in range(N_CAL_TRAJ):
            rng = np.random.RandomState(cal_rng_seed + seed * 1000 + k)
            x, r = draw_trajectory(params, rng, T, BATCH)
            rows.append(compute_teacher(params, x, r))
    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)
    from credit_memory.b10_tangent_adjoint_theory import direct_routed
    R = direct_routed(rows, f_diag, B1)
    return R, rows, f_diag, B1


def part_iiF_temporal_reuse(seed, r=4, gap=100, K=4):
    from credit_memory.b5_train import N as N5, T as T5, BATCH as B5, DELAY as D5, LR as LR5
    params0 = run_online_training_snapshot(seed, 0, N5, T5, B5, D5, LR5)
    R0, _, _, _ = build_R_from_params(seed, params0)
    params1 = run_online_training_snapshot(seed, gap, N5, T5, B5, D5, LR5)
    R1, _, _, _ = build_R_from_params(seed, params1)

    rng = np.random.RandomState(seed)
    rows0, cols0 = aca_no_leakage(R0, r, rng)

    rng_fresh = np.random.RandomState(seed + 500)
    rows_fresh, cols_fresh = aca_no_leakage(R1, r, rng_fresh)
    ev_fresh = eval_reconstruction(R1, cur_reconstruct(R1, rows_fresh, cols_fresh), K)

    # warm start: begin ACA at calibration n+gap from the PREVIOUS
    # calibration's own pivot rows (reuse) then let it proceed as usual
    start_row = rows0[0]
    rows_warm, cols_warm = aca_no_leakage(R1, r, rng_fresh, start_row=start_row)
    ev_warm = eval_reconstruction(R1, cur_reconstruct(R1, rows_warm, cols_warm), K)

    pivot_overlap_rows = len(set(rows0) & set(rows_warm)) / r
    return dict(fresh=ev_fresh, warm_started=ev_warm,
               pivot_overlap_rows_frac=pivot_overlap_rows)


# ===========================================================================
# PART III -- K_epsilon(r): temporal-coupling rank -> physical pool size
# ===========================================================================
def part_iii_K_epsilon_curve(R, eps_fracs=(0.02, 0.05, 0.10)):
    S = np.abs(R)
    F_full = float(sum(S[:, m].max() for m in range(N)))
    n2 = S.shape[0]
    curve = {}
    for r in range(1, N + 1):
        S_r = low_rank_trunc(S, r)
        row = {}
        for eps in eps_fracs:
            eps_abs = eps * F_full
            min_K = n2
            for K in range(1, n2 + 1):
                P_hat = best_pool_exact(S_r, K)
                achieved = float(sum(S[list(P_hat), m].max() for m in range(N)))
                if F_full - achieved <= eps_abs:
                    min_K = K
                    break
            row[str(eps)] = min_K
        curve[str(r)] = row
    j_rho = {m: int(np.argmax(S[:, m])) for m in range(N)}
    rho_pool_K4 = pool_most_frequent(j_rho, S, min(4, n2))
    rho_pool_util = float(sum(S[list(rho_pool_K4), m].max() for m in range(N)))
    return dict(curve=curve, F_full=F_full,
               rho_guided_K4_regret_frac=(F_full - rho_pool_util) / (F_full + 1e-300))


def main() -> None:
    print("=" * 90)
    print(f"Phase B10.2: selector theory + calibration theory, {len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(a_corr=[], a_regret=[], b_cone=[], c_misrank=[], d_theorem=[],
                    iiA=[], iiB=[], iiE=[], iii=[])
    rng_master = np.random.RandomState(2024)

    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)

        sbm = part_a_stats(rows, f_diag, B1)
        corr_rows, pool_regret = part_a_correlations(sbm)
        per_seed["a_corr"].append(corr_rows)
        per_seed["a_regret"].append(pool_regret)
        per_seed["b_cone"].append(part_b_cone(sbm))
        per_seed["c_misrank"].append(part_c_misrank_decomposition(sbm, f_diag))
        per_seed["d_theorem"].append(part_d_conditional_theorem(sbm))

        U, V0_P, V0_Q = build_factors(rows, a1)
        R = direct_routed(rows, f_diag, B1)

        per_seed["iiA"].append(part_iiA_sample_complexity(R))
        j_pool_deployed = sorted(pool_most_frequent(
            {m: int(np.argmax(np.abs(R[:, m]))) for m in range(N)}, np.abs(R), 4))
        b_res = part_iiB_pivot_methods(R, U, f_diag, j_pool_deployed, rng_master)
        per_seed["iiB"].append(b_res)
        row_idx, col_idx = b_res["B3_ACA_no_leakage"]["rows"], b_res["B3_ACA_no_leakage"]["cols"]
        per_seed["iiE"].append(part_iiE_matrix_completion(R, row_idx, col_idx, r=4))
        per_seed["iii"].append(part_iii_K_epsilon_curve(R))

        print(f"seed {seed}: median spearman(absrho,U)="
             f"{np.median([r['spearman_absrho_U'] for r in corr_rows if r['spearman_absrho_U'] is not None]):.2f}  "
             f"n_theorem_applies={per_seed['d_theorem'][-1]['n_theorem_applies']}  "
             f"B3_ACA winner={b_res['B3_ACA_no_leakage']['winner_preserved']:.2f}")

    # temporal reuse (Part II.F) -- separate, cheaper subset of seeds
    iiF_results = [part_iiF_temporal_reuse(seed) for seed in SEEDS[:4]]

    def med(lst, key_fn):
        vals = [key_fn(x) for x in lst]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    all_corr = [r for run in per_seed["a_corr"] for r in run]
    a_summary = dict(
        median_spearman_absrho_U=med(all_corr, lambda r: r["spearman_absrho_U"]),
        median_pearson_absrho_U=med(all_corr, lambda r: r["pearson_absrho_U"]),
        median_spearman_absrho_normgamma=med(all_corr, lambda r: r["spearman_absrho_normgamma"]),
        median_spearman_absrho_costheta=med(all_corr, lambda r: r["spearman_absrho_costheta"]),
        frac_top1_agree=float(np.mean([r["top1_agree_absrho_U"] for r in all_corr])),
        mean_topK_recall=float(np.mean([r["topK_recall_absrho_U"] for r in all_corr])),
        median_pool_regret_absrho=med(per_seed["a_regret"], lambda r: r["absrho"]))

    b_summary = dict(
        top_cos_theta_mean=med(per_seed["b_cone"], lambda r: r["top_cos_theta_mean"]),
        bottom_cos_theta_mean=med(per_seed["b_cone"], lambda r: r["bottom_cos_theta_mean"]),
        mean_frac_negative_utility=med(per_seed["b_cone"], lambda r: r["mean_frac_negative_utility"]))

    c_summary = dict(
        median_n_misranked=med(per_seed["c_misrank"], lambda r: r.get("n_misranked_pairs")),
        median_frac_pq_pair=med(per_seed["c_misrank"], lambda r: r.get("frac_same_upper_mode_PQ_pair")),
        median_frac_opposing_signs=med(per_seed["c_misrank"], lambda r: r.get("frac_opposing_cos_and_magnitude_signs")))

    total_applies = sum(r["n_theorem_applies"] for r in per_seed["d_theorem"])
    total_correct = sum(r["n_theorem_correct"] for r in per_seed["d_theorem"])
    d_summary = dict(total_pairs_checked=sum(r["n_pairs_checked"] for r in per_seed["d_theorem"]),
                     total_theorem_applies=total_applies, total_theorem_correct=total_correct,
                     theorem_accuracy_when_applies=(total_correct / total_applies
                                                    if total_applies else None),
                     frac_seeds_pre_overshoot_regime=float(np.mean(
                         [r["regime_is_pre_overshoot"] for r in per_seed["d_theorem"]])))

    iiA_summary = {key: dict(
        median_r_rows=med(per_seed["iiA"], lambda r, k=key: (r[k] or {}).get("r_rows")),
        median_r_cols=med(per_seed["iiA"], lambda r, k=key: (r[k] or {}).get("r_cols")),
        median_state_cost_frac=med(per_seed["iiA"], lambda r, k=key: (r[k] or {}).get("state_cost_frac")))
        for key in ("min_for_95pct_fro", "min_for_99pct_fro", "min_for_95pct_winner",
                   "min_for_95pct_topk", "min_for_near_zero_regret")}

    iiB_summary = {name: dict(
        median_state_cost_frac=med(per_seed["iiB"], lambda r, n=name: r[n]["state_cost_frac"]),
        median_fro_rel_err=med(per_seed["iiB"], lambda r, n=name: r[n]["fro_rel_err"]),
        median_winner_preserved=med(per_seed["iiB"], lambda r, n=name: r[n]["winner_preserved"]),
        median_pool_regret=med(per_seed["iiB"], lambda r, n=name: r[n]["pool_regret"]),
        median_frac_certified=med(per_seed["iiB"], lambda r, n=name: r[n]["frac_certified_by_margin"]))
        for name in ("B0_oracle_QR", "B1_random", "B3_ACA_no_leakage",
                    "B4_deployed_pool_seeded", "B5_U_seeded_cols")}

    iiE_summary = dict(
        matrix_completion=dict(
            median_fro_rel_err=med(per_seed["iiE"], lambda r: r["matrix_completion"]["fro_rel_err"]),
            median_winner_preserved=med(per_seed["iiE"], lambda r: r["matrix_completion"]["winner_preserved"]),
            median_pool_regret=med(per_seed["iiE"], lambda r: r["matrix_completion"]["pool_regret"])),
        cur_same_entries=dict(
            median_fro_rel_err=med(per_seed["iiE"], lambda r: r["cur_same_entries"]["fro_rel_err"]),
            median_winner_preserved=med(per_seed["iiE"], lambda r: r["cur_same_entries"]["winner_preserved"]),
            median_pool_regret=med(per_seed["iiE"], lambda r: r["cur_same_entries"]["pool_regret"])))

    iiF_summary = dict(
        median_fresh_fro=med(iiF_results, lambda r: r["fresh"]["fro_rel_err"]),
        median_warm_fro=med(iiF_results, lambda r: r["warm_started"]["fro_rel_err"]),
        median_fresh_winner=med(iiF_results, lambda r: r["fresh"]["winner_preserved"]),
        median_warm_winner=med(iiF_results, lambda r: r["warm_started"]["winner_preserved"]),
        median_pivot_overlap=med(iiF_results, lambda r: r["pivot_overlap_rows_frac"]))

    iii_summary = {}
    for r in range(1, N + 1):
        iii_summary[str(r)] = {
            eps: med(per_seed["iii"], lambda x, r=r, eps=eps: x["curve"][str(r)][eps])
            for eps in ("0.02", "0.05", "0.1")}

    print("-" * 90)
    print("PART I SUMMARY:")
    print("  A:", json.dumps(a_summary, indent=1))
    print("  B (cone):", json.dumps(b_summary, indent=1))
    print("  C (misrank):", json.dumps(c_summary, indent=1))
    print("  D (theorem):", json.dumps(d_summary, indent=1))
    print("PART II SUMMARY:")
    print("  IIA (oracle sample complexity):", json.dumps(iiA_summary, indent=1))
    print("  IIB (pivot methods):", json.dumps(iiB_summary, indent=1))
    print("  IIE (matrix completion vs CUR):", json.dumps(iiE_summary, indent=1))
    print("  IIF (temporal reuse):", json.dumps(iiF_summary, indent=1))
    print("PART III (K_epsilon(r)):", json.dumps(iii_summary, indent=1))

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ),
              part_i=dict(a_summary=a_summary, b_summary=b_summary,
                         c_summary=c_summary, d_summary=d_summary),
              part_ii=dict(iiA_summary=iiA_summary, iiB_summary=iiB_summary,
                          iiE_summary=iiE_summary, iiF_summary=iiF_summary),
              part_iii=iii_summary,
              per_seed=per_seed, iiF_results=iiF_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b10_2_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
