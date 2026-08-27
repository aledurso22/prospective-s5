"""B9.5 -- staleness DETECTION (not next-state prediction). B9.4 showed
persistence beats any tested predictor of the next pool/winner/rho
vector -- so this phase does not try to predict WHERE the geometry
moves. It tests the weaker, control-theoretic question: can CHEAP,
deployable, active-pool-only signals detect WHEN the currently
deployed K-pool has gone stale enough to justify paying for a full
recalibration -- an event-triggered recalibration test, explicitly NOT
prospective coding or prediction-correction.

No new training ALGORITHM (gradient rule / pool-construction method):
this only tests different RECALIBRATION-TIMING policies plugged into
B9.3's already-accepted periodic-pool mechanism.

Run:  python -m credit_memory.b9_5_staleness_detection
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
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.b5_train import set_config, draw_task_batch, loss_of, N, T, BATCH, DELAY, LR
from credit_memory.b6_prospective_tracking import hysteretic_select, T2_GAMMA, HYSTERESIS_MARGIN
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b9_2_shared_pool import best_pool_exact
from credit_memory.b9_3_pool_training import build_pool_and_selection, pool_batch_observation

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b9_5")

SEEDS = list(range(8))
STEPS = 600
K_PRIMARY = 4
TRAIN_FRAC = 400.0 / 599.0
HORIZONS = (1, 5, 10, 25)
TAU_COS_LIST = (0.85, 0.75, 0.65)
TAU_REGRET_PCTL = (75, 90, 95)     # percentile-of-own-run thresholds


def lag1_autocorr(u_t):
    x0 = np.concatenate([u_t[:-1].real.ravel(), u_t[:-1].imag.ravel()])
    x1 = np.concatenate([u_t[1:].real.ravel(), u_t[1:].imag.ravel()])
    if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def entropy_over_pool(abs_rho_pool_col):
    p = abs_rho_pool_col / (abs_rho_pool_col.sum() + 1e-300)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))   # normalized [0,1]


# ---------------------------------------------------------------------------
# Step 1: aging-pool labeled trajectory (pool built once, never refreshed
# -- reuses B9.3/B9.4's pool_frozen mechanism) with BOTH oracle staleness
# LABELS (full-bank/BPTT, diagnostic only) and CHEAP in-pool FEATURES
# (the only things any trigger is allowed to see).
# ---------------------------------------------------------------------------
def dense_labeled_run(seed, K=K_PRIMARY):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    pool_rng = np.random.RandomState(424242 + seed)
    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)
    d = np.ones(2 * N, np.complex128)

    P, j_pool_by_mode, rho_mat0, _ = build_pool_and_selection(
        params, cal_rng, f_diag, B1, "most_frequent", K, pool_rng)
    rho_cur_pool = {m: rho_mat0[P, m].copy() for m in range(N)}
    current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}
    prev_rho_pool = np.stack([rho_cur_pool[m] for m in range(N)], axis=1).copy()

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)

    log = dict(cos_a0=[], regret=[], coverage=[],
              margin=[], drift=[], entropy=[], switch_flag=[],
              q_mag_pool=[], Brow_pool=[], elig_energy=[], elig_ac1=[],
              upd_norm=[])

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        # --- oracle labels: full-bank exact decomposition (LABEL ONLY) ---
        gamma_n = np.zeros((2 * N, N), np.complex128)
        for m in range(N):
            c_m = build_c_t(q[1], B1[:, m])
            g_p, _ = per_coordinate_contribution(f_diag, d, c_m, Sa[0][:, :, m])
            gamma_n[:, m] = g_p
        G_n = gamma_n.sum(axis=0)
        U_n = 2 * np.real(np.conj(G_n)[None, :] * gamma_n) - np.abs(gamma_n) ** 2
        best_pool = best_pool_exact(U_n, K)
        util_best = sum(U_n[list(best_pool), m].max() for m in range(N))
        util_cur = sum(U_n[P, m].max() for m in range(N))
        regret = util_best - util_cur
        winners_full = np.argmax(np.abs(gamma_n), axis=0)
        coverage = float(np.mean([winners_full[m] in P for m in range(N)]))

        # --- cheap, deployable in-pool features (predictor-visible) ---
        f_pool = f_diag[P]
        margins, switched, entropies = [], 0, []
        for m in range(N):
            c_m_pool = build_c_t(q[1], B1[:, m])[:, :, P]
            r_obs = pool_batch_observation(f_pool, Sa[0][:, :, m], c_m_pool)
            rho_cur_pool[m] = (1 - T2_GAMMA) * rho_cur_pool[m] + T2_GAMMA * r_obs
            new_local, sw = hysteretic_select(rho_cur_pool[m], current_local.get(m),
                                              HYSTERESIS_MARGIN)
            if sw:
                switched += 1
            current_local[m] = new_local
            j_pool_by_mode[m] = P[new_local]

            abs_col = np.abs(rho_cur_pool[m])
            sorted_col = np.sort(abs_col)[::-1]
            margins.append(float(sorted_col[0] - sorted_col[1]) if len(sorted_col) > 1 else 0.0)
            entropies.append(entropy_over_pool(abs_col))

        rho_pool_mat = np.stack([rho_cur_pool[m] for m in range(N)], axis=1)
        drift = float(np.linalg.norm(rho_pool_mat - prev_rho_pool)
                     / (np.linalg.norm(prev_rho_pool) + 1e-300))
        prev_rho_pool = rho_pool_mat.copy()

        j_orig_pool = [j % N for j in P]
        q_mag_pool = float(np.mean(np.sqrt(np.mean(
            np.abs(q[1][:, :, j_orig_pool]) ** 2, axis=(0, 1)))))
        Brow_pool = float(np.mean(np.linalg.norm(B1, axis=1)[j_orig_pool]))
        elig_energy = float(np.mean([np.sum(np.abs(Sa[0][:, :, m]) ** 2)
                                    for m in range(N)]))
        elig_ac1 = float(np.mean([lag1_autocorr(Sa[0][:, :, m]) for m in range(N)]))

        # --- deployed gradient + BPTT reference (D_n label only) ---
        G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
        Ga0, Gb0 = b4_layer0_gradient(f_diag, j_pool_by_mode, B1, N, q[1], Sa[0], Sb[0])
        G = dict(a=[Ga0] + G_online["a"][1:], b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        g = tcg.flat_grads(G, params)
        lam = tcg.exact_lambda(params, q)
        G_bptt = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        cos_a0 = float(np.abs(np.vdot(G_bptt["a"][0], Ga0))
                      / (np.linalg.norm(Ga0) * np.linalg.norm(G_bptt["a"][0]) + 1e-300))

        log["cos_a0"].append(cos_a0)
        log["regret"].append(regret)
        log["coverage"].append(coverage)
        log["margin"].append(float(np.mean(margins)))
        log["drift"].append(drift)
        log["entropy"].append(float(np.mean(entropies)))
        log["switch_flag"].append(int(switched > 0))
        log["q_mag_pool"].append(q_mag_pool)
        log["Brow_pool"].append(Brow_pool)
        log["elig_energy"].append(elig_energy)
        log["elig_ac1"].append(elig_ac1)
        log["upd_norm"].append(float(np.linalg.norm(g)))

        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)

    return {k: np.array(v) for k, v in log.items()}


# ---------------------------------------------------------------------------
# Staleness labels + secondary AUROC/AUPRC diagnostics
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["margin", "drift", "entropy", "switch_flag", "q_mag_pool",
                 "Brow_pool", "elig_energy", "elig_ac1", "upd_norm", "age"]


def auroc(scores, labels):
    labels = np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return None
    ranks = stats.rankdata(scores)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores, labels):
    labels = np.asarray(labels, bool)
    if not labels.any():
        return None
    order = np.argsort(-scores)
    lab_sorted = labels[order]
    tp = np.cumsum(lab_sorted)
    fp = np.cumsum(~lab_sorted)
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))


def build_stale_labels(log, tau_cos, tau_regret):
    D = log["cos_a0"]
    return dict(cos=(D < tau_cos).astype(int),
               regret=(log["regret"] > tau_regret).astype(int))


def horizon_label(stale_now, h):
    """stale within the next h steps, from the current step's own info
    forward -- a legitimate forward-looking EVALUATION label (never
    seen by any online trigger, which only sees the past)."""
    Tn = len(stale_now)
    out = np.zeros(Tn, dtype=int)
    for n in range(Tn):
        out[n] = int(stale_now[n:n + h + 1].any())
    return out


def logistic_fit(X, y, l2=1e-2, iters=500, lr=0.5):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def logistic_predict(X, w):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def analyze_staleness_signals(log):
    Tn = len(log["cos_a0"])
    age = np.arange(Tn, dtype=float)     # this run's pool never refreshes
    feats = {name: (age if name == "age" else log[name]) for name in FEATURE_NAMES}

    out = dict(per_tau={}, auroc_by_feature={}, auprc_by_feature={})
    for tau in TAU_COS_LIST:
        labels = build_stale_labels(log, tau, np.inf)["cos"]
        frac_stale = float(labels.mean())
        aurocs, auprcs = {}, {}
        for h in HORIZONS:
            hl = horizon_label(labels, h)
            for name, f in feats.items():
                sign = -1.0 if name in ("margin",) else 1.0  # low margin -> stale
                a = auroc(sign * f, hl)
                p = auprc(sign * f, hl)
                aurocs.setdefault(name, {})[str(h)] = a
                auprcs.setdefault(name, {})[str(h)] = p
        out["per_tau"][str(tau)] = dict(frac_stale=frac_stale)
        out["auroc_by_feature"][str(tau)] = aurocs
        out["auprc_by_feature"][str(tau)] = auprcs
    return out


def fit_b3_logistic(log, tau_cos, h, train_end):
    Tn = len(log["cos_a0"])
    age = np.arange(Tn, dtype=float)
    feats = {name: (age if name == "age" else log[name]) for name in FEATURE_NAMES}
    X_all = np.stack([feats[n] for n in FEATURE_NAMES], axis=1)
    mu, sd = X_all[:train_end].mean(0), X_all[:train_end].std(0) + 1e-8
    Xn = (X_all - mu) / sd
    labels = build_stale_labels(log, tau_cos, np.inf)["cos"]
    y = horizon_label(labels, h)
    w = logistic_fit(Xn[:train_end], y[:train_end])
    p_test = logistic_predict(Xn[train_end:], w)
    y_test = y[train_end:]
    return dict(auroc=auroc(p_test, y_test), auprc=auprc(p_test, y_test),
               weights=dict(zip(FEATURE_NAMES + ["bias"], w.tolist())))


# ---------------------------------------------------------------------------
# Generalized event-triggered pool_periodic trainer: identical mechanism
# to B9.3's pool_periodic, but the recalibration DECISION comes from a
# `trigger_fn(cheap_state) -> bool` callback instead of a fixed interval.
# `cheap_state` contains ONLY quantities visible online from the active
# pool -- no full-bank, no BPTT, no oracle utility.
# ---------------------------------------------------------------------------
def train_event_triggered(seed, trigger_fn, K=K_PRIMARY, pool_method="most_frequent"):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    diag_rng = np.random.RandomState(55555 + seed)
    pool_rng = np.random.RandomState(424242 + seed)
    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)

    P, j_pool_by_mode, rho_mat0, _ = build_pool_and_selection(
        params, cal_rng, f_diag, B1, pool_method, K, pool_rng)
    rho_cur_pool = {m: rho_mat0[P, m].copy() for m in range(N)}
    current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}
    prev_rho_pool = np.stack([rho_cur_pool[m] for m in range(N)], axis=1).copy()
    age = 0
    n_recal = 0
    from credit_memory.b7_full_causal_training import cos_np, relerr_np, block_slices
    CHECKPOINTS = [1, 100, 300, 600]

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses, diagnostics = [], []
    finite = True

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        f_pool = f_diag[P]
        margins, switched, entropies = [], 0, []
        for m in range(N):
            c_m_pool = build_c_t(q[1], B1[:, m])[:, :, P]
            r_obs = pool_batch_observation(f_pool, Sa[0][:, :, m], c_m_pool)
            rho_cur_pool[m] = (1 - T2_GAMMA) * rho_cur_pool[m] + T2_GAMMA * r_obs
            new_local, sw = hysteretic_select(rho_cur_pool[m], current_local.get(m),
                                              HYSTERESIS_MARGIN)
            if sw:
                switched += 1
            current_local[m] = new_local
            j_pool_by_mode[m] = P[new_local]
            abs_col = np.abs(rho_cur_pool[m])
            sorted_col = np.sort(abs_col)[::-1]
            margins.append(float(sorted_col[0] - sorted_col[1]) if len(sorted_col) > 1 else 0.0)
            entropies.append(entropy_over_pool(abs_col))

        rho_pool_mat = np.stack([rho_cur_pool[m] for m in range(N)], axis=1)
        drift = float(np.linalg.norm(rho_pool_mat - prev_rho_pool)
                     / (np.linalg.norm(prev_rho_pool) + 1e-300))
        prev_rho_pool = rho_pool_mat.copy()
        j_orig_pool = [j % N for j in P]
        cheap_state = dict(
            age=float(age), margin=float(np.mean(margins)), drift=drift,
            entropy=float(np.mean(entropies)), switch_flag=int(switched > 0),
            q_mag_pool=float(np.mean(np.sqrt(np.mean(
                np.abs(q[1][:, :, j_orig_pool]) ** 2, axis=(0, 1))))),
            Brow_pool=float(np.mean(np.linalg.norm(B1, axis=1)[j_orig_pool])),
            elig_energy=float(np.mean([np.sum(np.abs(Sa[0][:, :, m]) ** 2)
                                       for m in range(N)])),
            elig_ac1=float(np.mean([lag1_autocorr(Sa[0][:, :, m]) for m in range(N)])))

        G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
        Ga0, Gb0 = b4_layer0_gradient(f_diag, j_pool_by_mode, B1, N, q[1], Sa[0], Sb[0])
        G = dict(a=[Ga0] + G_online["a"][1:], b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        g = tcg.flat_grads(G, params)
        cheap_state["upd_norm"] = float(np.linalg.norm(g))

        nrm = np.linalg.norm(g)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)
        age += 1

        losses.append(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(g)):
            finite = False
            break

        if step in CHECKPOINTS:
            x_d, y_d = draw_task_batch(diag_rng)
            _, h_d, r_d = loss_of(params, x_d, y_d)
            q_d = tcg.spatial_q(params, h_d, r_d)
            Sa_d, Sb_d = tcg.sensitivities(params, h_d, x_d)
            lam_d = tcg.exact_lambda(params, q_d)
            G_bptt_d = tcg.assemble(params, h_d, x_d, r_d, lam_d, Sa_d, Sb_d, direct=True)
            g_bptt_d = tcg.flat_grads(G_bptt_d, params)
            G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
            Ga0_d, Gb0_d = b4_layer0_gradient(f_diag, j_pool_by_mode, B1, N,
                                              q_d[1], Sa_d[0], Sb_d[0])
            G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                      b=[Gb0_d] + G_online_d["b"][1:], c=G_online_d["c"])
            g_train_d = tcg.flat_grads(G_d, params)
            slices, _ = block_slices(params)
            def sl(vec, key):
                a, b = slices[key]
                return vec[a:b]
            diagnostics.append(dict(step=step, cos_whole=cos_np(g_train_d, g_bptt_d),
                                    cos_a0=cos_np(sl(g_train_d, "a0"), sl(g_bptt_d, "a0"))))

        if trigger_fn(cheap_state):
            P, j_pool_by_mode, rho_mat_new, _ = build_pool_and_selection(
                params, cal_rng, f_diag, B1, pool_method, K, pool_rng)
            rho_cur_pool = {m: rho_mat_new[P, m].copy() for m in range(N)}
            current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}
            prev_rho_pool = np.stack([rho_cur_pool[m] for m in range(N)], axis=1).copy()
            age = 0
            n_recal += 1

    return dict(seed=seed, finite=finite, steps_run=len(losses),
               median_late_loss=float(np.median(losses[-100:]))
               if len(losses) >= 100 else float(np.median(losses)),
               n_recal=n_recal, diagnostics=diagnostics)


def late_avg_cos(diagnostics, key="cos_a0"):
    return float(np.mean([d[key] for d in diagnostics if d["step"] >= 300]))


def make_min_age_threshold_trigger(feature, threshold, direction, min_age=20):
    def trig(state):
        if state["age"] < min_age:
            return False
        v = state[feature]
        return (v < threshold) if direction == "below" else (v > threshold)
    return trig


def make_logistic_trigger(w, mu, sd, p_thresh=0.5, min_age=20):
    def trig(state):
        if state["age"] < min_age:
            return False
        x = np.array([state[n] for n in FEATURE_NAMES])
        xn = (x - mu) / sd
        z = np.dot(xn, w[:-1]) + w[-1]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return bool(p > p_thresh)
    return trig


# B0 reference numbers, reused directly from B9.3 (no rerun needed):
# median cos_a0 (late-avg) / n_recal, K=4, 8 seeds.
B0_REFERENCE = {
    "refresh=50": dict(cos_a0=0.620, n_recal=11),
    "refresh=100": dict(cos_a0=0.731, n_recal=5),
    "refresh=200": dict(cos_a0=0.625, n_recal=2),
    "refresh=600(frozen)": dict(cos_a0=0.484, n_recal=0),
}


def main() -> None:
    print("=" * 90)
    print(f"Phase B9.5: staleness-detection diagnostic, {len(SEEDS)} seeds")
    print("=" * 90)

    # --- Step 1: labeled aging-pool trajectories, AUROC diagnostics, B3 fit ---
    labeled_logs = [dense_labeled_run(seed) for seed in SEEDS]
    print("labeled trajectories collected.")

    signal_reports = [analyze_staleness_signals(log) for log in labeled_logs]
    agg_auroc = {}
    for tau in TAU_COS_LIST:
        agg_auroc[str(tau)] = {}
        for name in FEATURE_NAMES:
            for h in HORIZONS:
                vals = [r["auroc_by_feature"][str(tau)][name][str(h)]
                        for r in signal_reports
                        if r["auroc_by_feature"][str(tau)][name][str(h)] is not None]
                agg_auroc[str(tau)].setdefault(name, {})[str(h)] = (
                    float(np.median(vals)) if vals else None)
    print("median AUROC (tau=0.75, h=10) by feature:",
         {k: v.get("10") for k, v in agg_auroc["0.75"].items()})

    train_end = int(round(TRAIN_FRAC * (STEPS - 1)))
    b3_fits = [fit_b3_logistic(log, 0.75, 10, train_end) for log in labeled_logs]
    b3_auroc_median = float(np.median([f["auroc"] for f in b3_fits if f["auroc"] is not None]))
    print(f"B3 logistic (tau=0.75,h=10) median held-out AUROC: {b3_auroc_median:.3f}")

    # pick a representative fit (seed 0's, standardized on its own train
    # portion) to deploy the SAME logistic trigger online across seeds --
    # a fixed, pre-trained trigger, exactly as any deployed rule would be.
    rep_log = labeled_logs[0]
    age0 = np.arange(STEPS, dtype=float)
    feats0 = {n: (age0 if n == "age" else rep_log[n]) for n in FEATURE_NAMES}
    X0 = np.stack([feats0[n] for n in FEATURE_NAMES], axis=1)
    mu0, sd0 = X0[:train_end].mean(0), X0[:train_end].std(0) + 1e-8
    y0 = horizon_label(build_stale_labels(rep_log, 0.75, np.inf)["cos"], 10)
    w0 = logistic_fit((X0[:train_end] - mu0) / sd0, y0[:train_end])

    # percentile thresholds for B2a/B2b, pooled across all labeled runs
    all_margin = np.concatenate([lg["margin"] for lg in labeled_logs])
    all_drift = np.concatenate([lg["drift"] for lg in labeled_logs])
    margin_p25, margin_p40 = np.percentile(all_margin, [25, 40])
    drift_p75, drift_p90 = np.percentile(all_drift, [75, 90])
    print(f"margin p25/p40={margin_p25:.3f}/{margin_p40:.3f}  "
         f"drift p75/p90={drift_p75:.3f}/{drift_p90:.3f}")

    # --- Step 2: actual event-triggered training for each candidate policy ---
    policies = {
        "B2a_margin_p25": make_min_age_threshold_trigger("margin", margin_p25, "below"),
        "B2a_margin_p40": make_min_age_threshold_trigger("margin", margin_p40, "below"),
        "B2b_drift_p75": make_min_age_threshold_trigger("drift", drift_p75, "above"),
        "B2b_drift_p90": make_min_age_threshold_trigger("drift", drift_p90, "above"),
        "B3_logistic": make_logistic_trigger(w0, mu0, sd0, p_thresh=0.5),
    }

    policy_results = {}
    for name, trig in policies.items():
        runs = [train_event_triggered(seed, trig) for seed in SEEDS]
        cos_a0s = [late_avg_cos(r["diagnostics"]) for r in runs if r["diagnostics"]]
        n_recals = [r["n_recal"] for r in runs]
        policy_results[name] = dict(median_cos_a0=float(np.median(cos_a0s)),
                                    median_n_recal=float(np.median(n_recals)),
                                    n_recal_per_seed=n_recals,
                                    cos_a0_per_seed=cos_a0s)
        print(f"{name:20s} median cos_a0(late)={np.median(cos_a0s):.3f}  "
             f"median n_recal={np.median(n_recals):.1f}  per-seed n_recal={n_recals}")

    print("-" * 90)
    print("Pareto comparison (median cos_a0 late-avg vs median n_recal):")
    for name, d in B0_REFERENCE.items():
        print(f"  B0 {name:22s} cos_a0={d['cos_a0']:.3f}  n_recal={d['n_recal']}")
    for name, d in policy_results.items():
        print(f"  {name:25s} cos_a0={d['median_cos_a0']:.3f}  n_recal={d['median_n_recal']:.1f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, DELAY=DELAY, seeds=SEEDS,
                         steps=STEPS, K=K_PRIMARY, tau_cos_list=TAU_COS_LIST,
                         horizons=HORIZONS,
                         margin_p25=float(margin_p25), margin_p40=float(margin_p40),
                         drift_p75=float(drift_p75), drift_p90=float(drift_p90)),
              agg_auroc_by_feature=agg_auroc,
              b3_median_auroc=b3_auroc_median,
              b3_fits=b3_fits,
              b0_reference=B0_REFERENCE,
              policy_results=policy_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_5_staleness_detection_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
