"""B9.3 -- periodically recalibrated K-pool CCM: the first real
end-to-end training arm built on B9.2's shared-candidate-pool result.
Toy system only, no S5.

Core algorithm (per the task spec):
  1. At a calibration event, evaluate the FULL 2N candidate bank on
     N_CAL_TRAJ calibration trajectories (credit_memory/
     b6_prospective_tracking.py::causal_prefix_selection, already B9.1
     leak-fixed) and construct a shared pool P of K upper channels.
  2. Pool construction uses the cheapest B9.2 method that matched the
     exact pool well: "most-frequently-winning channels"
     (credit_memory/b9_2_shared_pool.py::pool_most_frequent). An
     oracle-utility pool is ALSO computed at every calibration event,
     but ONLY for the (diagnostic-only) coverage/regret log -- never
     as a deployable arm's gradient source.
  3. Between calibration events, relevance is tracked reactively
     (EMA, matching B6's T2 exactly) for ONLY the K candidates in P --
     genuine O(K) per-mode state/compute, not O(2N).
  4. Each lower mode independently runs hysteretic_select within the
     pool every step -- no forced shared channel across modes.
  5. No dormant-state resurrection: the deployed gradient
     (credit_memory/b4_deploy.py::b4_layer0_gradient) resets its own
     filter state fresh every step regardless of which pool channel is
     currently selected, exactly as established in PHASE_B9.md Part 1.

Arms: A0 online, A1 reactive_full (unrestricted O(2N) reactive |rho|,
identical to b7_full_causal_training.py's "a1_rank1"), A2 pool_frozen
(pool + per-mode pick built once at step 0, frozen), A3 pool_periodic
(this phase's new arm), A4 full_causal (exact, uncompressed P/Q,
credit_memory/full_causal.py, unmodified), Bref bptt.

No prospective/prediction-correction of any kind (B6 already found
"reactive suffices"; this phase does not revisit that).

Run:  python -m credit_memory.b9_3_pool_training
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import time

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.full_causal import full_causal_gradient
from credit_memory.b5_train import (set_config, draw_task_batch, loss_of,
                                    N, T, DELAY, BATCH, LR, N_CAL_TRAJ,
                                    CHECKPOINTS)
from credit_memory.b6_prospective_tracking import (
    causal_prefix_selection, hysteretic_select, T2_GAMMA, HYSTERESIS_MARGIN)
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b7_full_causal_training import cos_np, relerr_np, block_slices
from credit_memory.b9_2_shared_pool import (
    pool_most_frequent, pool_largest_lambda, pool_uniform_coverage,
    best_pool_exact)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b9_3")

SEEDS = list(range(8))
STEPS = 600
K_PRIMARY = 4
K_LIST = [1, 2, 4, 8]
REFRESH_GRID_FOR_K4 = [50, 100, 200, 600]     # 600 == effectively frozen
REFRESH_DEFAULT = 100                          # used for K != 4 and for
                                                # the random-pool ablation


def pool_batch_observation(f_diag_pool, Sa0_m, c_m_pool):
    """Same construction as b6_prospective_tracking.single_batch_
    observation, reimplemented for an arbitrary K-length pool (that
    function hardcodes d=ones(2N) via the module-global N)."""
    d = np.ones(f_diag_pool.shape[0], np.complex128)
    g_p, _ = per_coordinate_contribution(f_diag_pool, d, c_m_pool, Sa0_m)
    return g_p


def exact_gamma_no_leak(f_diag, cal_rng_rows, B1_col, m):
    d = np.ones(f_diag.shape[0], np.complex128)
    gamma = np.zeros(f_diag.shape[0], np.complex128)
    for row in cal_rng_rows:
        c_row = build_c_t(row["q1"], B1_col)
        u_row = row["Sa0"][:, :, m]
        g_row, _ = per_coordinate_contribution(f_diag, d, c_row, u_row)
        gamma += g_row
    return gamma


def calibration_rows(params, rng, n_traj):
    """Rows in the shape exact_gamma_no_leak/oracle diagnostics need,
    built from REAL task batches (unlike the static-benchmark scripts'
    arbitrary-r protocol) -- consistent with how this arm is actually
    trained/calibrated."""
    rows = []
    for _ in range(n_traj):
        x, y = draw_task_batch(rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, _ = tcg.sensitivities(params, h, x)
        rows.append(dict(q1=q[1], Sa0=Sa[0]))
    return rows


def build_pool_and_selection(params, cal_rng, f_diag, B1, method, K,
                             pool_rng=None):
    """One full calibration event: FULL O(2N) causal_prefix_selection,
    then pool construction (deployable methods: 'most_frequent',
    'random'; diagnostic-only: 'oracle_exact' is computed separately in
    the caller, never here), then per-mode selection-within-pool.
    Returns (P, j_pool_by_mode, rho_by_mode, top_j_unrestricted)."""
    rho_by_mode, top_j_unrestricted = causal_prefix_selection(
        params, cal_rng, f_diag)
    rho_mat = np.stack([rho_by_mode[m] for m in range(N)], axis=1)
    if method == "most_frequent":
        P = pool_most_frequent(top_j_unrestricted, rho_mat, K)
    elif method == "random":
        P = set(pool_rng.choice(2 * N, size=K, replace=False).tolist())
    elif method == "largest_lambda":
        P = pool_largest_lambda(f_diag, K)
    elif method == "uniform_coverage":
        P = pool_uniform_coverage(f_diag, K)
    else:
        raise ValueError(method)
    P_sorted = sorted(P)
    j_pool_by_mode = {m: P_sorted[int(np.argmax(np.abs(rho_mat[P_sorted, m])))]
                      for m in range(N)}
    return P_sorted, j_pool_by_mode, rho_mat, top_j_unrestricted


def oracle_pool_diagnostic(params, cal_rng_oracle, f_diag, B1, K):
    """DIAGNOSTIC ONLY: exact-search oracle-utility pool, used solely to
    log coverage/regret. Never returned as, or used to build, a
    deployable arm's gradient."""
    rows = calibration_rows(params, cal_rng_oracle, N_CAL_TRAJ)
    U_mat = np.zeros((2 * N, N))
    for m in range(N):
        gamma = exact_gamma_no_leak(f_diag, rows, B1[:, m], m)
        G = gamma.sum()
        U_mat[:, m] = 2 * np.real(np.conj(G) * gamma) - np.abs(gamma) ** 2
    P_oracle = best_pool_exact(U_mat, K)
    j_oracle_unrestricted = {m: int(np.argmax(U_mat[:, m])) for m in range(N)}
    return P_oracle, U_mat, j_oracle_unrestricted


def train(arm, seed, clip=0.0, K=None, refresh=None, pool_method="most_frequent",
          log_pool_events=True):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    oracle_rng = np.random.RandomState(313131 + seed)
    diag_rng = np.random.RandomState(55555 + seed)
    pool_rng = np.random.RandomState(424242 + seed)

    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)

    timing = dict(calibration_s=0.0, deploy_s=0.0, reactive_track_s=0.0)
    pool_events = []
    top_j_unrestricted_last = None
    P = j_pool_by_mode = rho_cur_pool = current_local = None

    if arm == "reactive_full":
        t0 = time.perf_counter()
        rho_cur, top_j_by_mode = causal_prefix_selection(params, cal_rng, f_diag)
        timing["calibration_s"] += time.perf_counter() - t0
    elif arm in ("pool_frozen", "pool_periodic"):
        t0 = time.perf_counter()
        P, j_pool_by_mode, rho_mat, top_j_unrestricted_last = \
            build_pool_and_selection(params, cal_rng, f_diag, B1,
                                     pool_method, K, pool_rng)
        timing["calibration_s"] += time.perf_counter() - t0
        rho_cur_pool = {m: rho_mat[P, m].copy() for m in range(N)}
        current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}
        if log_pool_events:
            P_oracle, U_mat, j_oracle_unrestricted = oracle_pool_diagnostic(
                params, oracle_rng, f_diag, B1, K)
            coverage = sum(1 for m in range(N)
                          if top_j_unrestricted_last[m] in P) / N
            pool_events.append(dict(
                step=0, pool=list(P), coverage_frac=coverage,
                jaccard_vs_prev=None,
                oracle_pool=sorted(P_oracle),
                jaccard_pool_vs_oracle=(len(set(P) & P_oracle)
                                       / len(set(P) | P_oracle))))

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses, diagnostics = [], []
    finite = True

    for step in range(1, STEPS + 1):
        # --- periodic pool recalibration (A3 only) ---
        if arm == "pool_periodic" and step > 1 and (step - 1) % refresh == 0:
            t0 = time.perf_counter()
            P_new, j_pool_new, rho_mat, top_j_unrestricted_last = \
                build_pool_and_selection(params, cal_rng, f_diag, B1,
                                         pool_method, K, pool_rng)
            timing["calibration_s"] += time.perf_counter() - t0
            if log_pool_events:
                P_oracle, U_mat, j_oracle_unrestricted = \
                    oracle_pool_diagnostic(params, oracle_rng, f_diag, B1, K)
                coverage = sum(1 for m in range(N)
                              if top_j_unrestricted_last[m] in P_new) / N
                jacc = (len(set(P_new) & set(P)) / len(set(P_new) | set(P))
                       if P else None)
                pool_events.append(dict(
                    step=step, pool=list(P_new), coverage_frac=coverage,
                    jaccard_vs_prev=jacc, oracle_pool=sorted(P_oracle),
                    jaccard_pool_vs_oracle=(len(set(P_new) & P_oracle)
                                           / len(set(P_new) | P_oracle))))
            P, j_pool_by_mode = P_new, j_pool_new
            rho_cur_pool = {m: rho_mat[P, m].copy() for m in range(N)}
            current_local = {m: P.index(j_pool_by_mode[m]) for m in range(N)}

        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        if arm == "online":
            G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        elif arm == "bptt":
            lam = tcg.exact_lambda(params, q)
            G = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        elif arm == "full_causal":
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = full_causal_gradient(a1, B1, N, q[1], Sa[0], Sb[0])
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        elif arm == "reactive_full":
            from credit_memory.b6_prospective_tracking import single_batch_observation
            t0 = time.perf_counter()
            for m in range(N):
                c_m = build_c_t(q[1], B1[:, m])
                r_obs = single_batch_observation(f_diag, Sa[0][:, :, m], c_m)
                rho_cur[m] = (1 - T2_GAMMA) * rho_cur[m] + T2_GAMMA * r_obs
                new_sel, _ = hysteretic_select(rho_cur[m],
                                               top_j_by_mode.get(m),
                                               HYSTERESIS_MARGIN)
                top_j_by_mode[m] = new_sel
            timing["reactive_track_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                          q[1], Sa[0], Sb[0])
            timing["deploy_s"] += time.perf_counter() - t0
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        elif arm in ("pool_frozen", "pool_periodic"):
            t0 = time.perf_counter()
            f_pool = f_diag[P]
            for m in range(N):
                c_m_pool = build_c_t(q[1], B1[:, m])[:, :, P]
                r_obs = pool_batch_observation(f_pool, Sa[0][:, :, m], c_m_pool)
                rho_cur_pool[m] = ((1 - T2_GAMMA) * rho_cur_pool[m]
                                   + T2_GAMMA * r_obs)
                new_local, _ = hysteretic_select(rho_cur_pool[m],
                                                 current_local.get(m),
                                                 HYSTERESIS_MARGIN)
                current_local[m] = new_local
                j_pool_by_mode[m] = P[new_local]
            timing["reactive_track_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = b4_layer0_gradient(f_diag, j_pool_by_mode, B1, N,
                                          q[1], Sa[0], Sb[0])
            timing["deploy_s"] += time.perf_counter() - t0
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        else:
            raise ValueError(arm)

        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if clip > 0 and nrm > clip:
            g = g * (clip / nrm)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)

        losses.append(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(g)):
            finite = False
            break

        if step in CHECKPOINTS or step == 1:
            x_d, y_d = draw_task_batch(diag_rng)
            _, h_d, r_d = loss_of(params, x_d, y_d)
            q_d = tcg.spatial_q(params, h_d, r_d)
            Sa_d, Sb_d = tcg.sensitivities(params, h_d, x_d)
            lam_d = tcg.exact_lambda(params, q_d)
            G_bptt_d = tcg.assemble(params, h_d, x_d, r_d, lam_d, Sa_d,
                                    Sb_d, direct=True)
            g_bptt_d = tcg.flat_grads(G_bptt_d, params)

            if arm == "online":
                G_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
            elif arm == "bptt":
                G_d = G_bptt_d
            elif arm == "full_causal":
                G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
                Ga0_d, Gb0_d = full_causal_gradient(a1, B1, N, q_d[1],
                                                    Sa_d[0], Sb_d[0])
                G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:], c=G_online_d["c"])
            elif arm == "reactive_full":
                G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
                Ga0_d, Gb0_d = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                                  q_d[1], Sa_d[0], Sb_d[0])
                G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:], c=G_online_d["c"])
            elif arm in ("pool_frozen", "pool_periodic"):
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

            diagnostics.append(dict(
                step=step,
                cos_whole=cos_np(g_train_d, g_bptt_d),
                rel_err_whole=relerr_np(g_train_d, g_bptt_d),
                cos_a0=cos_np(sl(g_train_d, "a0"), sl(g_bptt_d, "a0")),
                cos_b0=cos_np(sl(g_train_d, "b0"), sl(g_bptt_d, "b0"))))

    n_candidate_states = (2 * N * N if arm in ("reactive_full", "full_causal")
                         else (K * N if arm in ("pool_frozen", "pool_periodic")
                              else 0))

    return dict(arm=arm, seed=seed, clip=clip, K=K, refresh=refresh,
               pool_method=pool_method, finite=finite,
               steps_run=len(losses),
               final_loss=float(losses[-1]) if losses else None,
               median_late_loss=float(np.median(losses[-100:]))
               if len(losses) >= 100 else
               (float(np.median(losses)) if losses else None),
               n_candidate_states=n_candidate_states,
               timing=timing, pool_events=pool_events,
               diagnostics=diagnostics)


def summarize(runs):
    fin = [r for r in runs if r["finite"]]
    late_cos = [r["diagnostics"][-1]["cos_whole"] for r in fin
               if r["diagnostics"]]
    late_cos_a0 = [r["diagnostics"][-1]["cos_a0"] for r in fin
                  if r["diagnostics"]]
    return dict(
        n_seeds=len(runs), n_finite=len(fin),
        median_final_cos_whole=float(np.median(late_cos)) if late_cos else None,
        median_final_cos_a0=float(np.median(late_cos_a0)) if late_cos_a0 else None,
        median_final_loss=float(np.median(
            [r["final_loss"] for r in fin if r["final_loss"] is not None])),
        median_late_loss=float(np.median(
            [r["median_late_loss"] for r in fin
             if r["median_late_loss"] is not None])),
        mean_calibration_s=float(np.mean(
            [r["timing"]["calibration_s"] for r in fin])),
        mean_deploy_s=float(np.mean([r["timing"]["deploy_s"] for r in fin])),
        mean_reactive_track_s=float(np.mean(
            [r["timing"]["reactive_track_s"] for r in fin])),
        n_candidate_states=fin[0]["n_candidate_states"] if fin else None,
        median_pool_coverage_frac=float(np.median(
            [ev["coverage_frac"] for r in fin for ev in r["pool_events"]]))
        if any(r["pool_events"] for r in fin) else None,
        median_pool_jaccard_vs_prev=float(np.median(
            [ev["jaccard_vs_prev"] for r in fin for ev in r["pool_events"]
             if ev["jaccard_vs_prev"] is not None]))
        if any(ev["jaccard_vs_prev"] is not None
              for r in fin for ev in r["pool_events"]) else None,
        median_pool_jaccard_vs_oracle=float(np.median(
            [ev["jaccard_pool_vs_oracle"] for r in fin
             for ev in r["pool_events"]]))
        if any(r["pool_events"] for r in fin) else None)


def main() -> None:
    print("=" * 90)
    print(f"Phase B9.3: periodically recalibrated K-pool CCM training, "
         f"{len(SEEDS)} seeds x {STEPS} steps")
    print("=" * 90)

    configs = []
    for arm in ("online", "bptt", "full_causal", "reactive_full"):
        configs.append(dict(arm=arm))
    for K in K_LIST:
        configs.append(dict(arm="pool_frozen", K=K))
    for refresh in REFRESH_GRID_FOR_K4:
        configs.append(dict(arm="pool_periodic", K=K_PRIMARY, refresh=refresh))
    for K in K_LIST:
        if K == K_PRIMARY:
            continue
        configs.append(dict(arm="pool_periodic", K=K, refresh=REFRESH_DEFAULT))
    configs.append(dict(arm="pool_periodic", K=K_PRIMARY, refresh=REFRESH_DEFAULT,
                        pool_method="random"))

    all_results = {}
    t_wall0 = time.perf_counter()
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        runs = [train(seed=seed, **cfg) for seed in SEEDS]
        all_results[key] = runs
        summ = summarize(runs)
        print(f"{key:80s} cos={summ['median_final_cos_whole']:.4f}  "
             f"late_loss={summ['median_late_loss']:.4f}  "
             f"states={summ['n_candidate_states']}")
    print(f"total wall time: {time.perf_counter() - t_wall0:.1f}s")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, DELAY=DELAY, seeds=SEEDS,
                         steps=STEPS, k_primary=K_PRIMARY, k_list=K_LIST,
                         refresh_grid_k4=REFRESH_GRID_FOR_K4,
                         refresh_default=REFRESH_DEFAULT),
              summaries={key: summarize(runs)
                        for key, runs in all_results.items()},
              raw={key: [{k: v for k, v in r.items() if k != "diagnostics"}
                        | dict(diagnostics=r["diagnostics"])
                        for r in runs]
                  for key, runs in all_results.items()})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_3_pool_training_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
