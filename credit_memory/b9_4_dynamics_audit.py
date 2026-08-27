"""B9.4 -- dynamics/mechanism audit of the moving credit-pool geometry,
using dense per-step logging on top of B9.3's arms. Diagnostic only:
no prediction-correction, no prospective coding, no feedback alignment/
PAL-style probing, no new training arm, no S5.

Two questions:
  1. Is the moving credit-pool geometry predictable from cheap
     (non-candidate-propagated) quantities?
  2. Does a fixed small credit pool spontaneously become more aligned
     with the network as it trains, or does it just go stale?

Terminology note (addressed per user correction): no dormant-state
resurrection is required anywhere in this setup (established in
PHASE_B9.md); any future sparse mechanism informed by this audit would
use TEMPORARY CANDIDATE PROBES, not persistent per-candidate state.

Run:  python -m credit_memory.b9_4_dynamics_audit
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.full_causal import full_causal_gradient
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.b5_train import set_config, draw_task_batch, loss_of, N, T, BATCH, DELAY, LR
from credit_memory.b6_prospective_tracking import (
    causal_prefix_selection, hysteretic_select, single_batch_observation,
    T2_GAMMA, HYSTERESIS_MARGIN)
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b9_2_shared_pool import pool_most_frequent, best_pool_exact
from credit_memory.b9_3_pool_training import (
    build_pool_and_selection, pool_batch_observation)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b9_4")

SEEDS = list(range(8))
STEPS = 600
K_PRIMARY = 4
D_SUBSAMPLE = 5          # Part D tangent-recursion evaluated every 5 steps
TRAIN_FRAC = 400.0 / 599.0   # Part C train/test split, in-time, no leakage


# ---------------------------------------------------------------------------
# Part D helper: tangent recursion for a single candidate's own pole
# ---------------------------------------------------------------------------
def x_and_tangent(lam, u_t):
    """x_t = lam x_{t-1} + u_t;  r_t = dx_t/dlam = x_{t-1} + lam r_{t-1}.
    u_t: (T,BATCH) complex. Returns x (T,BATCH), r (T,BATCH)."""
    Tn, Bn = u_t.shape
    x = np.zeros((Tn, Bn), np.complex128)
    r = np.zeros((Tn, Bn), np.complex128)
    prevx = np.zeros(Bn, np.complex128)
    prevr = np.zeros(Bn, np.complex128)
    for t in range(Tn):
        newx = lam * prevx + u_t[t]
        newr = prevx + lam * prevr
        x[t], r[t] = newx, newr
        prevx, prevr = newx, newr
    return x, r


def verify_tangent_recursion_once(seed=0):
    """One finite-difference spot check (not run every step): confirms
    the analytic tangent recursion above matches d x_t/d lambda to
    numerical precision, before it is trusted in Part D."""
    rng = np.random.RandomState(seed)
    u_t = rng.randn(T, BATCH) + 1j * rng.randn(T, BATCH)
    lam = 0.93 * np.exp(1j * 0.7)
    eps = 1e-6
    x0, r0 = x_and_tangent(lam, u_t)
    x1, _ = x_and_tangent(lam + eps, u_t)
    fd = (x1 - x0) / eps
    return float(np.max(np.abs(fd - r0)) / (np.max(np.abs(r0)) + 1e-300))


def lag1_autocorr(u_t):
    x0 = np.concatenate([u_t[:-1].real.ravel(), u_t[:-1].imag.ravel()])
    x1 = np.concatenate([u_t[1:].real.ravel(), u_t[1:].imag.ravel()])
    if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def effective_ranks(M, fracs=(0.90, 0.95, 0.99)):
    s = np.linalg.svd(M, compute_uv=False)
    sq = s ** 2
    total = sq.sum()
    if total <= 0:
        return {str(f): int(M.shape[1]) for f in fracs}
    cum = np.cumsum(sq) / total
    return {str(f): int(np.searchsorted(cum, f) + 1) for f in fracs}


def z_hard(abs_rho_n, winners_n):
    """abs_rho_n: (2N,N). winners_n: (N,) int -- current per-mode top
    winners (reference pool P_n = set of winners). z_j = sum_m relu(
    |rho_j,m| - max_{k in P_n} |rho_k,m|)."""
    P = np.unique(winners_n)
    ref = np.max(abs_rho_n[P, :], axis=0)          # (N,) per-mode best-in-P
    return np.sum(np.maximum(abs_rho_n - ref[None, :], 0.0), axis=1)  # (2N,)


def z_smooth(abs_rho_n, winners_n, tau):
    """Smooth surrogate (log-sum-exp / softplus) of the same quantity,
    for Taylor/local-linear fitting ONLY -- hard z_hard is still used
    for all final pool-regret/top-K evaluation."""
    P = np.unique(winners_n)
    ref = tau * np.log(np.sum(np.exp(abs_rho_n[P, :] / tau), axis=0))  # smooth-max, (N,)
    diff = (abs_rho_n - ref[None, :]) / tau
    sp = tau * np.log1p(np.exp(np.clip(diff, -30, 30)))    # softplus, (2N,N)
    return np.sum(sp, axis=1)


# ---------------------------------------------------------------------------
# Part A/B/C/D dense run: unrestricted reactive_full arm
# ---------------------------------------------------------------------------
def dense_reactive_full_run(seed):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)
    d = np.ones(2 * N, np.complex128)

    rho_cur, top_j_by_mode = causal_prefix_selection(params, cal_rng, f_diag)

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)

    log = dict(rho_ema=[], gamma_inst=[], lambda_full=[], B_row_norm=[],
              q_mag=[], elig_energy=[], elig_ac1=[], delta_theta_norm=[],
              winners=[], d_sens=[], d_step=[])

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        # instantaneous exact per-candidate decomposition (fresh state,
        # single batch -- diagnostic only, not used by any training rule)
        gamma_n = np.zeros((2 * N, N), np.complex128)
        for m in range(N):
            c_m = build_c_t(q[1], B1[:, m])
            g_p, _ = per_coordinate_contribution(f_diag, d, c_m, Sa[0][:, :, m])
            gamma_n[:, m] = g_p
            r_obs = single_batch_observation(f_diag, Sa[0][:, :, m], c_m)
            rho_cur[m] = (1 - T2_GAMMA) * rho_cur[m] + T2_GAMMA * r_obs
            new_sel, _ = hysteretic_select(rho_cur[m], top_j_by_mode.get(m),
                                           HYSTERESIS_MARGIN)
            top_j_by_mode[m] = new_sel

        rho_mat = np.stack([rho_cur[m] for m in range(N)], axis=1)  # (2N,N)
        winners = np.array([top_j_by_mode[m] for m in range(N)])

        q_mag_upper = np.sqrt(np.mean(np.abs(q[1]) ** 2, axis=(0, 1)))  # (N,)
        elig_energy = np.array([np.sum(np.abs(Sa[0][:, :, m]) ** 2)
                               for m in range(N)])
        elig_ac1 = np.array([lag1_autocorr(Sa[0][:, :, m]) for m in range(N)])
        j_orig = np.arange(2 * N) % N
        B_row_norm = np.linalg.norm(B1, axis=1)[j_orig]
        q_mag = q_mag_upper[j_orig]

        log["rho_ema"].append(rho_mat.copy())
        log["gamma_inst"].append(gamma_n.copy())
        log["lambda_full"].append(f_diag.copy())
        log["B_row_norm"].append(B_row_norm)
        log["q_mag"].append(q_mag)
        log["elig_energy"].append(elig_energy)
        log["elig_ac1"].append(elig_ac1)
        log["winners"].append(winners)

        if step % D_SUBSAMPLE == 0:
            sens = np.zeros((2 * N, N), np.complex128)
            for m in range(N):
                c_m = build_c_t(q[1], B1[:, m])
                for j in range(2 * N):
                    _, rr = x_and_tangent(f_diag[j], Sa[0][:, :, m])
                    sens[j, m] = np.sum(np.conj(c_m[:, :, j]) * rr)
            log["d_sens"].append(sens)
            log["d_step"].append(step)

        # A0-style training update (reactive_full's own gradient rule)
        G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
        Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                      q[1], Sa[0], Sb[0])
        G = dict(a=[Ga0] + G_online["a"][1:],
                 b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        g = tcg.flat_grads(G, params)
        log["delta_theta_norm"].append(float(np.linalg.norm(g)))

        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)

    return {k: np.array(v) for k, v in log.items()}


# ---------------------------------------------------------------------------
# Part B: temporal structure of the moving credit geometry
# ---------------------------------------------------------------------------
def analyze_part_b(log):
    rho_ema = log["rho_ema"]
    abs_rho = np.abs(rho_ema)
    winners = log["winners"]
    Tn = rho_ema.shape[0]

    flat = abs_rho.reshape(Tn, -1)
    autocorrs = {}
    for lag in (1, 5, 10, 20, 50):
        if lag >= Tn:
            continue
        a = flat[:-lag] - flat[:-lag].mean(axis=0, keepdims=True)
        b = flat[lag:] - flat[lag:].mean(axis=0, keepdims=True)
        num = np.sum(a * b, axis=0)
        den = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0)) + 1e-300
        autocorrs[str(lag)] = float(np.median(num / den))

    run_lengths = []
    for m in range(winners.shape[1]):
        w = winners[:, m]
        cur, length = w[0], 1
        for t in range(1, len(w)):
            if w[t] == cur:
                length += 1
            else:
                run_lengths.append(length)
                cur, length = w[t], 1
        run_lengths.append(length)
    run_lengths = np.array(run_lengths)

    rel_change = (np.linalg.norm((rho_ema[1:] - rho_ema[:-1]).reshape(Tn - 1, -1), axis=1)
                 / (np.linalg.norm(rho_ema[:-1].reshape(Tn - 1, -1), axis=1) + 1e-300))

    delta_rho = (rho_ema[1:] - rho_ema[:-1]).reshape(Tn - 1, -1)
    er_rho = effective_ranks(delta_rho)

    z_series = np.array([z_hard(abs_rho[n], winners[n]) for n in range(Tn)])
    delta_z = z_series[1:] - z_series[:-1]
    er_z = effective_ranks(delta_z)

    return dict(
        autocorr_by_lag=autocorrs,
        median_winner_lifetime=float(np.median(run_lengths)),
        lifetime_p10_p50_p90=[float(np.percentile(run_lengths, p))
                              for p in (10, 50, 90)],
        n_switch_events=int(len(run_lengths) - winners.shape[1]),
        median_rel_step_change=float(np.median(rel_change)),
        effective_rank_delta_rho=er_rho,
        effective_rank_delta_z=er_z), z_series


# ---------------------------------------------------------------------------
# Part C: predictor audit (P0 persistence, P1 secant, P2 context-linear,
# and a low-rank latent AR(1) model if Part B's rank supports it)
# ---------------------------------------------------------------------------
def build_features(log, n):
    lam0, lam1 = log["lambda_full"][n], log["lambda_full"][n + 1]
    d_abs_lam = np.abs(lam1) - np.abs(lam0)
    d_phase = np.angle(lam1 / lam0)
    d_Brow = log["B_row_norm"][n + 1] - log["B_row_norm"][n]
    d_q = log["q_mag"][n + 1] - log["q_mag"][n]
    d_elig = float(np.mean(log["elig_energy"][n + 1] - log["elig_energy"][n]))
    d_ac1 = float(np.mean(log["elig_ac1"][n + 1] - log["elig_ac1"][n]))
    upd_norm = float(log["delta_theta_norm"][n])
    return np.stack([d_abs_lam, d_phase, d_Brow, d_q,
                     np.full(2 * N, d_elig), np.full(2 * N, d_ac1),
                     np.full(2 * N, upd_norm)], axis=1)


def fit_p2(log, target_series, train_end):
    Xs, ys = [], []
    for n in range(train_end):
        Xs.append(build_features(log, n))
        ys.append(target_series[n + 1] - target_series[n])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    g, *_ = np.linalg.lstsq((X - mu) / sd, y, rcond=None)
    return g, mu, sd


def fit_latent_ar(z_series, train_end, r):
    Ztr = z_series[:train_end + 1]
    mean_z = Ztr.mean(axis=0)
    U, S, Vt = np.linalg.svd(Ztr - mean_z, full_matrices=False)
    V_r = Vt[:r].T
    H = (Ztr - mean_z) @ V_r
    Xh = np.concatenate([H[:-1], np.ones((H.shape[0] - 1, 1))], axis=1)
    AB, *_ = np.linalg.lstsq(Xh, H[1:], rcond=None)
    return V_r, mean_z, AB[:-1], AB[-1]


def topk_set(v, k):
    return set(np.argsort(-v)[:k].tolist())


def run_predictor_comparison(log, target_series, smooth_target_series,
                             effective_rank_90, K=K_PRIMARY):
    """Shared P0/P1/P2/latent comparison harness, usable for either the
    (sparse) z_hard target or the (smoother) per-pole aggregated-relevance
    fallback target -- see analyze_part_c."""
    Tn = target_series.shape[0]
    train_end = int(round(TRAIN_FRAC * (Tn - 1)))
    gamma_inst = log["gamma_inst"]

    g2, mu2, sd2 = fit_p2(log, smooth_target_series, train_end)
    r_lat = max(1, min(effective_rank_90, 4))
    V_r, mean_z, A_lat, b_lat = fit_latent_ar(target_series, train_end, r_lat)

    def predict_p2(n):
        F = build_features(log, n)
        return target_series[n] + ((F - mu2) / sd2) @ g2

    def predict_latent(n):
        h = (target_series[n] - mean_z) @ V_r
        return mean_z + (h @ A_lat + b_lat) @ V_r.T

    def U_inst_at(n):
        G = gamma_inst[n].sum(axis=0)
        return 2 * np.real(np.conj(G)[None, :] * gamma_inst[n]) - np.abs(gamma_inst[n]) ** 2

    results = {name: dict(mse=[], corr=[], spearman=[], topk_recall=[],
                          regret=[]) for name in ("P0", "P1", "P2", "latent")}
    anticipation = dict(p2_top3=0, persistence_top3=0, n_switches=0)

    for n in range(train_end, Tn - 1):
        z_true = target_series[n + 1]
        preds = dict(P0=target_series[n],
                    P1=target_series[n] + (target_series[n] - target_series[n - 1]),
                    P2=predict_p2(n), latent=predict_latent(n))
        U_true = U_inst_at(n + 1)
        true_pool = best_pool_exact(U_true, K)
        true_util = sum(U_true[list(true_pool), m].max() for m in range(N))
        true_top = topk_set(z_true, K)

        for name, zhat in preds.items():
            results[name]["mse"].append(float(np.mean((zhat - z_true) ** 2)))
            if np.std(zhat) > 1e-12 and np.std(z_true) > 1e-12:
                results[name]["corr"].append(float(np.corrcoef(zhat, z_true)[0, 1]))
                sp = stats.spearmanr(zhat, z_true).statistic
                if not np.isnan(sp):
                    results[name]["spearman"].append(float(sp))
            pred_pool = topk_set(zhat, K)
            results[name]["topk_recall"].append(len(pred_pool & true_top) / K)
            achieved = sum(U_true[list(pred_pool), m].max() for m in range(N))
            results[name]["regret"].append(true_util - achieved)

        # anticipation check: does P2 (using only info through step n) rank
        # the NEW winner (revealed at step n+1) in its own top-3 more often
        # than persistence (target_series[n] itself) does?
        for m in range(N):
            if log["winners"][n + 1][m] != log["winners"][n][m]:
                new_w = log["winners"][n + 1][m]
                anticipation["n_switches"] += 1
                if new_w in topk_set(preds["P2"], 3):
                    anticipation["p2_top3"] += 1
                if new_w in topk_set(preds["P0"], 3):
                    anticipation["persistence_top3"] += 1

    def agg(v):
        return float(np.median(v)) if len(v) else None

    summary = {name: dict(median_mse=agg(v["mse"]), median_corr=agg(v["corr"]),
                          median_spearman=agg(v["spearman"]),
                          n_corr_defined=len(v["corr"]),
                          mean_topk_recall=float(np.mean(v["topk_recall"])),
                          median_regret=agg(v["regret"]))
              for name, v in results.items()}
    summary["anticipation"] = anticipation
    summary["latent_rank_used"] = r_lat
    return summary


def analyze_part_c(log, z_series, effective_rank_z_90, K=K_PRIMARY):
    """Primary target z_hard (per addendum: pool regret / top-K recall /
    rank correlation are primary; raw MSE secondary). z_hard is heavily
    sparse across candidates at a single step (most candidates never
    approach unseating the current winner), which degenerates
    cross-sectional corr/spearman at many individual steps (filtered via
    n_corr_defined above) -- so a smoother fallback target,
    rho_agg[j,n] = max_m |rho[j,m,n]| (aggregated per-pole relevance,
    the spec's own documented fallback), is evaluated the same way for
    comparison."""
    Tn = z_series.shape[0]
    tau = float(np.std(np.abs(log["rho_ema"])) + 1e-8)
    z_smooth_series = np.array([
        z_smooth(np.abs(log["rho_ema"][n]), log["winners"][n], tau)
        for n in range(Tn)])
    z_result = run_predictor_comparison(log, z_series, z_smooth_series,
                                        effective_rank_z_90, K)

    rho_agg_series = np.max(np.abs(log["rho_ema"]), axis=2)      # (Tn,2N)
    rho_agg_smooth = tau * np.log1p(np.exp(
        np.clip((rho_agg_series - rho_agg_series.mean()) / tau, -30, 30)))
    er_rho_agg = effective_ranks(rho_agg_series[1:] - rho_agg_series[:-1])
    rho_agg_result = run_predictor_comparison(
        log, rho_agg_series, rho_agg_smooth, int(er_rho_agg["0.9"]), K)

    return dict(z_hard_target=z_result, rho_agg_target=rho_agg_result)


# ---------------------------------------------------------------------------
# Part D: analytic pole-sensitivity, real (magnitude/phase) coordinates
# ---------------------------------------------------------------------------
def analyze_part_d(log):
    d_steps = log["d_step"]
    lam = log["lambda_full"]
    gamma = log["gamma_inst"]
    obs, ana, feat_targets, feats = [], [], [], []

    for idx, n in enumerate(d_steps):
        n = int(n)
        n2 = n + D_SUBSAMPLE
        if n2 >= gamma.shape[0]:
            continue
        sens = log["d_sens"][idx]                # (2N,N) drho/dlambda at n
        lam_n, lam_n2 = lam[n], lam[n2]
        d_abs = np.abs(lam_n2) - np.abs(lam_n)     # (2N,) real
        d_phase = np.angle(lam_n2 / lam_n)          # (2N,) real
        rho_n = gamma[n]                            # (2N,N) complex, |rho| basis
        abs_rho_n = np.abs(rho_n) + 1e-300

        # real chain rule: d|rho|/dm = Re[conj(rho)/|rho| * s * e^{i theta}]
        #                  d|rho|/dtheta = Re[conj(rho)/|rho| * s * i * lambda]
        theta = np.angle(lam_n)
        unit = np.exp(1j * theta)[:, None]
        dabs_dm = np.real(np.conj(rho_n) / abs_rho_n * sens * unit)
        dabs_dtheta = np.real(np.conj(rho_n) / abs_rho_n * sens
                             * 1j * lam_n[:, None])
        analytic_dabs = dabs_dm * d_abs[:, None] + dabs_dtheta * d_phase[:, None]
        observed_dabs = np.abs(gamma[n2]) - np.abs(gamma[n])

        obs.append(observed_dabs.ravel())
        ana.append(analytic_dabs.ravel())

    obs = np.concatenate(obs)
    ana = np.concatenate(ana)
    corr_analytic = float(np.corrcoef(ana, obs)[0, 1])
    r2_analytic = 1.0 - np.sum((obs - ana) ** 2) / (np.sum((obs - obs.mean()) ** 2) + 1e-300)
    r2_persistence = 0.0   # predicts zero change by construction
    return dict(n_pairs=int(len(obs)),
               analytic_corr_vs_observed=corr_analytic,
               analytic_r2_vs_observed=float(r2_analytic),
               persistence_r2=r2_persistence)


# ---------------------------------------------------------------------------
# Part E: fixed frozen K=4 pool -- does it become more or less useful?
# ---------------------------------------------------------------------------
def dense_pool_frozen_run(seed, K=K_PRIMARY):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    diag_rng = np.random.RandomState(55555 + seed)
    pool_rng = np.random.RandomState(424242 + seed)
    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)
    d = np.ones(2 * N, np.complex128)

    P, j_pool_by_mode, rho_mat0, top_j0 = build_pool_and_selection(
        params, cal_rng, f_diag, B1, "most_frequent", K, pool_rng)
    rho_cur_pool = {m: rho_mat0[P, m].copy() for m in range(N)}
    current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)

    log = dict(cos_a0=[], cos_whole=[], coverage=[], avg_rho_pool=[],
              avg_rho_nonpool=[], avg_Brow_pool=[], avg_Brow_nonpool=[],
              avg_lambda_pool=[], avg_lambda_nonpool=[])

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        # diagnostic-only full snapshot (never used to drive training)
        gamma_n = np.zeros((2 * N, N), np.complex128)
        for m in range(N):
            c_m = build_c_t(q[1], B1[:, m])
            g_p, _ = per_coordinate_contribution(f_diag, d, c_m, Sa[0][:, :, m])
            gamma_n[:, m] = g_p
        abs_rho_n = np.abs(gamma_n)
        winners_n = np.argmax(abs_rho_n, axis=0)
        coverage = float(np.mean([winners_n[m] in P for m in range(N)]))
        mask = np.zeros(2 * N, bool)
        mask[P] = True
        j_orig = np.arange(2 * N) % N
        B_row_norm = np.linalg.norm(B1, axis=1)[j_orig]
        log["coverage"].append(coverage)
        log["avg_rho_pool"].append(float(np.mean(abs_rho_n[mask, :])))
        log["avg_rho_nonpool"].append(float(np.mean(abs_rho_n[~mask, :])))
        log["avg_Brow_pool"].append(float(np.mean(B_row_norm[mask])))
        log["avg_Brow_nonpool"].append(float(np.mean(B_row_norm[~mask])))
        log["avg_lambda_pool"].append(float(np.mean(np.abs(f_diag)[mask])))
        log["avg_lambda_nonpool"].append(float(np.mean(np.abs(f_diag)[~mask])))

        # within-pool reactive per-mode selection (unchanged pool_frozen design)
        f_pool = f_diag[P]
        for m in range(N):
            c_m_pool = build_c_t(q[1], B1[:, m])[:, :, P]
            r_obs = pool_batch_observation(f_pool, Sa[0][:, :, m], c_m_pool)
            rho_cur_pool[m] = (1 - T2_GAMMA) * rho_cur_pool[m] + T2_GAMMA * r_obs
            new_local, _ = hysteretic_select(rho_cur_pool[m], current_local.get(m),
                                             HYSTERESIS_MARGIN)
            current_local[m] = new_local
            j_pool_by_mode[m] = P[new_local]

        G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
        Ga0, Gb0 = b4_layer0_gradient(f_diag, j_pool_by_mode, B1, N,
                                      q[1], Sa[0], Sb[0])
        G = dict(a=[Ga0] + G_online["a"][1:],
                 b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        g = tcg.flat_grads(G, params)

        lam = tcg.exact_lambda(params, q)
        G_bptt = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        g_bptt = tcg.flat_grads(G_bptt, params)
        cos_whole = float(np.abs(np.vdot(g_bptt, g))
                          / (np.linalg.norm(g) * np.linalg.norm(g_bptt) + 1e-300))
        cos_a0 = float(np.abs(np.vdot(G_bptt["a"][0], Ga0))
                      / (np.linalg.norm(Ga0) * np.linalg.norm(G_bptt["a"][0]) + 1e-300))
        log["cos_whole"].append(cos_whole)
        log["cos_a0"].append(cos_a0)

        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)

    return {k: np.array(v) for k, v in log.items()}, P


def analyze_part_e(log):
    Tn = len(log["cos_a0"])
    windows = dict(early=(0, Tn // 3), mid=(Tn // 3, 2 * Tn // 3),
                  late=(2 * Tn // 3, Tn))
    out = {}
    for name, (a, b) in windows.items():
        out[name] = dict(
            median_cos_a0=float(np.median(log["cos_a0"][a:b])),
            median_cos_whole=float(np.median(log["cos_whole"][a:b])),
            median_coverage=float(np.median(log["coverage"][a:b])),
            avg_rho_pool_over_nonpool=float(
                np.mean(log["avg_rho_pool"][a:b])
                / (np.mean(log["avg_rho_nonpool"][a:b]) + 1e-300)),
            avg_Brow_pool_over_nonpool=float(
                np.mean(log["avg_Brow_pool"][a:b])
                / (np.mean(log["avg_Brow_nonpool"][a:b]) + 1e-300)),
            avg_lambda_pool_over_nonpool=float(
                np.mean(log["avg_lambda_pool"][a:b])
                / (np.mean(log["avg_lambda_nonpool"][a:b]) + 1e-300)))
    return out


def main() -> None:
    print("=" * 90)
    print(f"Phase B9.4: dynamics/mechanism audit, {len(SEEDS)} seeds")
    print("=" * 90)

    fd_check = verify_tangent_recursion_once()
    print(f"Part D tangent-recursion finite-difference check: "
         f"rel_err={fd_check:.2e} (PASS if ~1e-6, matches eps)")

    per_seed = dict(b=[], c=[], d=[], e=[])
    for seed in SEEDS:
        log = dense_reactive_full_run(seed)
        b_summary, z_series = analyze_part_b(log)
        c_summary = analyze_part_c(log, z_series,
                                   b_summary["effective_rank_delta_z"]["0.9"])
        d_summary = analyze_part_d(log)
        per_seed["b"].append(b_summary)
        per_seed["c"].append(c_summary)
        per_seed["d"].append(d_summary)
        del log

        e_log, pool = dense_pool_frozen_run(seed, K=K_PRIMARY)
        e_summary = analyze_part_e(e_log)
        e_summary["pool"] = pool
        per_seed["e"].append(e_summary)
        del e_log

        print(f"seed {seed}: lifetime={b_summary['median_winner_lifetime']:.0f} "
             f"rank_delta_rho90={b_summary['effective_rank_delta_rho']['0.9']} "
             f"P0_topk={c_summary['z_hard_target']['P0']['mean_topk_recall']:.2f} "
             f"P2_topk={c_summary['z_hard_target']['P2']['mean_topk_recall']:.2f} "
             f"D_r2={d_summary['analytic_r2_vs_observed']:.3f} "
             f"E_cos_a0(early->late)={e_summary['early']['median_cos_a0']:.2f}->"
             f"{e_summary['late']['median_cos_a0']:.2f}")

    def med(lst, *keys):
        vals = []
        for d in lst:
            v = d
            for k in keys:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    summary = dict(
        part_b=dict(
            median_lifetime=med(per_seed["b"], "median_winner_lifetime"),
            median_rel_step_change=med(per_seed["b"], "median_rel_step_change"),
            median_er_delta_rho_90=med(per_seed["b"], "effective_rank_delta_rho", "0.9"),
            median_er_delta_rho_95=med(per_seed["b"], "effective_rank_delta_rho", "0.95"),
            median_er_delta_z_90=med(per_seed["b"], "effective_rank_delta_z", "0.9"),
            median_autocorr_lag1=med(per_seed["b"], "autocorr_by_lag", "1"),
            median_autocorr_lag10=med(per_seed["b"], "autocorr_by_lag", "10"),
            median_autocorr_lag50=med(per_seed["b"], "autocorr_by_lag", "50")),
        part_c={
            target: {
                arm: dict(
                    median_topk_recall=med(per_seed["c"], target, arm, "mean_topk_recall")
                    if arm != "P0-baseline" else None,
                    median_regret=med(per_seed["c"], target, arm, "median_regret"),
                    median_spearman=med(per_seed["c"], target, arm, "median_spearman"))
                for arm in ("P0", "P1", "P2", "latent")}
            for target in ("z_hard_target", "rho_agg_target")},
        part_d=dict(
            median_analytic_r2=med(per_seed["d"], "analytic_r2_vs_observed"),
            median_analytic_corr=med(per_seed["d"], "analytic_corr_vs_observed"),
            fd_check_rel_err=fd_check),
        part_e={
            window: dict(
                median_cos_a0=med(per_seed["e"], window, "median_cos_a0"),
                median_coverage=med(per_seed["e"], window, "median_coverage"),
                median_rho_pool_over_nonpool=med(
                    per_seed["e"], window, "avg_rho_pool_over_nonpool"))
            for window in ("early", "mid", "late")})

    # paired seed-level check: does cos_a0 rise early->late, per seed?
    diffs = [e["late"]["median_cos_a0"] - e["early"]["median_cos_a0"]
            for e in per_seed["e"]]
    summary["part_e"]["late_minus_early_cos_a0_per_seed"] = diffs
    summary["part_e"]["n_seeds_rising"] = int(sum(d_ > 0 for d_ in diffs))

    # paired seed-level check: P2 vs P0 on z_hard target (regret, topk)
    p2_regret = [c["z_hard_target"]["P2"]["median_regret"] for c in per_seed["c"]]
    p0_regret = [c["z_hard_target"]["P0"]["median_regret"] for c in per_seed["c"]]
    p2_topk = [c["z_hard_target"]["P2"]["mean_topk_recall"] for c in per_seed["c"]]
    p0_topk = [c["z_hard_target"]["P0"]["mean_topk_recall"] for c in per_seed["c"]]
    summary["part_c"]["p2_vs_p0_regret_diff_per_seed"] = \
        [p0 - p2 for p0, p2 in zip(p0_regret, p2_regret)]
    summary["part_c"]["p2_vs_p0_topk_diff_per_seed"] = \
        [p2 - p0 for p2, p0 in zip(p2_topk, p0_topk)]

    print("-" * 90)
    print("PART B (median over seeds):", json.dumps(summary["part_b"], indent=1))
    print("PART C z_hard (median over seeds):",
         json.dumps(summary["part_c"]["z_hard_target"], indent=1))
    print("PART C P2 vs P0 regret diff per seed:",
         summary["part_c"]["p2_vs_p0_regret_diff_per_seed"])
    print("PART C P2 vs P0 topk diff per seed:",
         summary["part_c"]["p2_vs_p0_topk_diff_per_seed"])
    print("PART D (median over seeds):", json.dumps(summary["part_d"], indent=1))
    print("PART E cos_a0 by window (median over seeds):",
         {w: summary["part_e"][w]["median_cos_a0"] for w in ("early", "mid", "late")})
    print("PART E late-early diff per seed:", diffs,
         " n_rising=", summary["part_e"]["n_seeds_rising"], "/", len(SEEDS))

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, DELAY=DELAY,
                                    seeds=SEEDS, steps=STEPS, K=K_PRIMARY,
                                    d_subsample=D_SUBSAMPLE,
                                    train_frac=TRAIN_FRAC),
              summary=summary, per_seed=per_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_4_dynamics_audit_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
