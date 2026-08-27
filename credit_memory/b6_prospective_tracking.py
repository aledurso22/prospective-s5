"""B6: prospective CCM tracking validation. Tests whether repairing the
B5.1-measured calibration-staleness problem requires only periodic
causal recalibration (T1), or genuinely benefits from prospective
prediction-correction (T3), versus a purely reactive EMA (T2), against
the frozen B5 baseline (T0).

Does NOT change the underlying CCM temporal-credit mechanism (Phase A's
(E1)/(E2), unedited); does NOT use BPTT/exact adjoint/exact P/Q teacher
state anywhere in the T0-T3 training algorithms (BPTT is
evaluation-only, exactly as in B5/B5.1).

Sequence time (per-training-batch, UNCHANGED CCM channel dynamics):
  x_{j,t} = lambda_j x_{j,t-1} + u_t
Optimizer/meta time (NEW in B6, operates once per training step or once
per K steps -- a slower, separate timescale from sequence time):
  rho^-_{n+1} = rho_n + beta (rho_n - rho_{n-1})              (T3 predict)
  rho_{n+1}   = rho^-_{n+1} + K (r_{n+1} - rho^-_{n+1})        (T3 correct)
r_{n+1} is the "ordinary B4 streaming statistic" -- the raw per-batch
cross term (credit_memory.lagcorr.per_coordinate_contribution, reused
unmodified), reset fresh each training batch (no cross-batch state,
matching the toy's own forward() convention of zero initial recurrent
state per batch).

Fixed hyperparameters (no sweep, per instruction): T3 beta=0.5, K=0.3;
T2 gamma=0.08 (B4D's best-performing EMA rate); hysteresis margin=0.15
(relative); T1 recalibration period K_period=100 (primary), 50
(secondary, clip=0 only).

Run:  python -m credit_memory.b6_prospective_tracking
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.streaming import StreamingRelevance
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.b5_train import (set_config, draw_task_batch, loss_of,
                                    architecture_only_selector,
                                    L, N, T, DELAY, BATCH, LR, N_CAL_TRAJ,
                                    CHECKPOINTS)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b6")

SEEDS = list(range(8))
STEPS = 600
B1_, B2_, EPS = 0.9, 0.999, 1e-8

# fixed hyperparameters, per instruction (no sweep)
T3_BETA = 0.5
T3_K = 0.3
T2_GAMMA = 0.08
HYSTERESIS_MARGIN = 0.15
T1_PERIOD_PRIMARY = 100
T1_PERIOD_SECONDARY = 50


def causal_prefix_selection(params, cal_rng, f_diag):
    """Same protocol as b5_train.causal_calibration_selector -- a
    short causal, parameter-update-free calibration prefix. Reused
    verbatim (not reimplemented) for T0's initial calibration and T1's
    periodic recalibration, so both use the IDENTICAL statistic B5
    already validated."""
    estimators = {m: StreamingRelevance(f_diag, BATCH, mode="windowed")
                 for m in range(N)}
    for _ in range(N_CAL_TRAJ):
        x, y = draw_task_batch(cal_rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, _ = tcg.sensitivities(params, h, x)
        for m in range(N):
            c_m = build_c_t(q[1], params["b"][1][:, m])
            for t in range(T):
                estimators[m].step(Sa[0][t, :, m], c_m[t])
    return ({m: estimators[m].rho.copy() for m in range(N)},
           {m: int(estimators[m].top_channel(1)[0]) for m in range(N)})


def single_batch_observation(f_diag, Sa0_m, c_m):
    """r_n: the raw, fresh, single-training-batch cross-statistic
    (credit_memory.lagcorr.per_coordinate_contribution, unmodified),
    reset each batch -- the "ordinary B4 streaming statistic" observation
    at optimizer step n."""
    d = np.ones(2 * N, np.complex128)
    g_p, _ = per_coordinate_contribution(f_diag, d, c_m, Sa0_m)
    return g_p


def hysteretic_select(rho_vec, current, margin):
    """Stay on `current` unless another channel exceeds it by `margin`
    (relative). current=None picks the argmax outright."""
    absr = np.abs(rho_vec)
    best = int(np.argmax(absr))
    if current is None:
        return best, True
    if absr[best] > absr[current] * (1.0 + margin):
        return best, (best != current)
    return current, False


def adam_step(flat, m_, v_, g, step, lr):
    m_ = B1_ * m_ + (1 - B1_) * g
    v_ = B2_ * v_ + (1 - B2_) * g ** 2
    flat_new = flat - lr * (m_ / (1 - B1_ ** step)) / (
        np.sqrt(v_ / (1 - B2_ ** step)) + EPS)
    return flat_new, m_, v_


def train(arm, seed, clip, t1_period=T1_PERIOD_PRIMARY):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    diag_rng = np.random.RandomState(55555 + seed)

    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)

    if arm == "T0":
        _, top_j_by_mode = causal_prefix_selection(params, cal_rng, f_diag)
    elif arm == "T1":
        _, top_j_by_mode = causal_prefix_selection(params, cal_rng, f_diag)
    elif arm in ("T2", "T3"):
        # bootstrap: one causal prefix, same as T0/T1, to seed rho_0
        rho0, top_j_by_mode = causal_prefix_selection(params, cal_rng,
                                                       f_diag)
        rho_prev = {m: None for m in range(N)}   # rho_{n-1}, for T3
        rho_cur = rho0                             # rho_n
    else:
        raise ValueError(arm)

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses, diagnostics, tracking_log = [], [], []
    switch_count = {m: 0 for m in range(N)}
    dwell_since_switch = {m: 0 for m in range(N)}
    finite = True

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        # -- T1 periodic recalibration (pauses nothing; uses a
        # SEPARATE cal_rng stream, no training-batch data reused as
        # calibration data)
        if arm == "T1" and step > 1 and (step - 1) % t1_period == 0:
            _, top_j_by_mode = causal_prefix_selection(params, cal_rng,
                                                        f_diag)

        # -- T2/T3 per-step relevance tracking (uses THIS step's own
        # batch data -- no extra forward passes)
        if arm in ("T2", "T3"):
            for m in range(N):
                c_m = build_c_t(q[1], B1[:, m])
                r_obs = single_batch_observation(f_diag, Sa[0][:, :, m], c_m)
                if arm == "T2":
                    rho_cur[m] = (1 - T2_GAMMA) * rho_cur[m] \
                        + T2_GAMMA * r_obs
                    pred, resid = None, None
                else:  # T3
                    if rho_prev[m] is None:
                        pred = rho_cur[m]
                    else:
                        pred = rho_cur[m] + T3_BETA * (rho_cur[m]
                                                       - rho_prev[m])
                    resid = r_obs - pred
                    rho_new = pred + T3_K * resid
                    rho_prev[m] = rho_cur[m]
                    rho_cur[m] = rho_new
                new_sel, switched = hysteretic_select(
                    rho_cur[m], top_j_by_mode.get(m), HYSTERESIS_MARGIN)
                if switched:
                    switch_count[m] += 1
                    dwell_since_switch[m] = 0
                else:
                    dwell_since_switch[m] += 1
                top_j_by_mode[m] = new_sel
                if step <= 20 or step % 50 == 0:
                    absr = np.abs(rho_cur[m])
                    margin = float((np.sort(absr)[-1] - np.sort(absr)[-2])
                                   / (np.sort(absr)[-1] + 1e-30))
                    entry = dict(step=step, mode=m, selected=new_sel,
                               switched=bool(switched), margin=margin,
                               dwell=dwell_since_switch[m])
                    if arm == "T3":
                        entry["persistence_err"] = float(np.linalg.norm(
                            r_obs - rho_prev[m])) if rho_prev[m] is not None \
                            else None
                        entry["prediction_err"] = float(np.linalg.norm(resid))
                    tracking_log.append(entry)

        G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
        Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N, q[1],
                                      Sa[0], Sb[0])
        G = dict(a=[Ga0] + G_online["a"][1:], b=[Gb0] + G_online["b"][1:],
                c=G_online["c"])

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

            G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
            Ga0_d, Gb0_d = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                              q_d[1], Sa_d[0], Sb_d[0])
            G_ccm_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:],
                          c=G_online_d["c"])
            g_ccm_d = tcg.flat_grads(G_ccm_d, params)

            def cos(u, v):
                return float(np.abs(np.vdot(v, u))
                            / (np.linalg.norm(u) * np.linalg.norm(v)
                               + 1e-300))
            def relerr(u, v):
                return float(np.linalg.norm(u - v)
                            / (np.linalg.norm(v) + 1e-300))

            # per-block a0/b0 cosine, matching B5.1's convention
            idx = 0
            block_sl = {}
            for l in range(L):
                block_sl[f"a{l}"] = (idx, idx + 2 * N); idx += 2 * N
                m_sz = params["b"][l].size
                block_sl[f"b{l}"] = (idx, idx + 2 * m_sz); idx += 2 * m_sz
            def sl(vec, key):
                a, b = block_sl[key]
                return vec[a:b]

            diagnostics.append(dict(
                step=step,
                cos_whole=cos(g_ccm_d, g_bptt_d),
                rel_err_whole=relerr(g_ccm_d, g_bptt_d),
                cos_a0=cos(sl(g_ccm_d, "a0"), sl(g_bptt_d, "a0")),
                cos_b0=cos(sl(g_ccm_d, "b0"), sl(g_bptt_d, "b0"))))

    out = dict(arm=arm, seed=seed, clip=clip, t1_period=t1_period,
              finite=finite, steps_run=len(losses),
              final_loss=float(losses[-1]) if losses else None,
              median_late_loss=float(np.median(losses[-100:]))
              if len(losses) >= 100 else
              (float(np.median(losses)) if losses else None),
              diagnostics=diagnostics,
              switch_count=switch_count if arm in ("T2", "T3") else None,
              tracking_log=tracking_log if arm in ("T2", "T3") else None)
    return out


def main() -> None:
    print("=" * 90)
    print("Phase B6: prospective CCM tracking validation")
    print("=" * 90)

    all_runs = []
    for clip in [0.0, 1.0]:
        arms = (["T0", "T1", "T2", "T3"] if clip == 0.0
               else ["T0", "T2", "T3"])   # clip=1 secondary, T1 skipped
                                           # for budget (T1's mechanism
                                           # does not depend on clip)
        for arm in arms:
            for seed in SEEDS:
                out = train(arm, seed, clip)
                all_runs.append(out)
                d600 = next((d for d in out["diagnostics"]
                            if d["step"] == 600), None)
                print(f"[{arm} s{seed} clip{clip}] final_loss="
                      f"{out['final_loss']:.4f}  "
                      f"cos_a0@600={d600['cos_a0'] if d600 else float('nan'):.3f}"
                      f"  finite={out['finite']}")
        if clip == 0.0:
            for seed in SEEDS:
                out = train("T1", seed, clip, t1_period=T1_PERIOD_SECONDARY)
                out["arm"] = "T1_K50"
                all_runs.append(out)

    print("-" * 90)
    print("Staleness-curve check (median cos_a0/cos_b0 by step, clip=0):")
    for arm in ["T0", "T1", "T2", "T3", "T1_K50"]:
        rows = [r for r in all_runs if r["arm"] == arm and r["clip"] == 0.0]
        if not rows:
            continue
        for step in CHECKPOINTS:
            a0s = [d["cos_a0"] for r in rows for d in r["diagnostics"]
                  if d["step"] == step]
            b0s = [d["cos_b0"] for r in rows for d in r["diagnostics"]
                  if d["step"] == step]
            if a0s:
                print(f"  {arm:8s} step={step}: median cos_a0="
                      f"{np.median(a0s):.3f}  median cos_b0="
                      f"{np.median(b0s):.3f}")

    print("-" * 90)
    print("Task loss (median final, clip=0):")
    for arm in ["T0", "T1", "T2", "T3", "T1_K50"]:
        rows = [r for r in all_runs if r["arm"] == arm and r["clip"] == 0.0
               and r["finite"]]
        if rows:
            print(f"  {arm:8s}: median final_loss="
                  f"{np.median([r['final_loss'] for r in rows]):.4f}  "
                  f"n_finite={len(rows)}/{len(SEEDS)}")

    print("-" * 90)
    print("T3 prediction vs persistence error (median over all logged "
          "steps, clip=0):")
    t3_rows = [r for r in all_runs if r["arm"] == "T3" and r["clip"] == 0.0]
    pred_errs, pers_errs = [], []
    for r in t3_rows:
        for e in (r["tracking_log"] or []):
            if e.get("prediction_err") is not None \
                    and e.get("persistence_err") is not None:
                pred_errs.append(e["prediction_err"])
                pers_errs.append(e["persistence_err"])
    if pred_errs:
        print(f"  median |pred_err|={np.median(pred_errs):.4f}  "
              f"median |persistence_err|={np.median(pers_errs):.4f}  "
              f"prediction better on {np.mean(np.array(pred_errs) < np.array(pers_errs)) * 100:.1f}% of logged steps")

    print("-" * 90)
    print("Switch counts (median total over 6 modes, clip=0):")
    for arm in ["T2", "T3"]:
        rows = [r for r in all_runs if r["arm"] == arm and r["clip"] == 0.0]
        totals = [sum(r["switch_count"].values()) for r in rows]
        print(f"  {arm}: median total switches over 600 steps="
              f"{np.median(totals):.1f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(L=L, N=N, T=T, DELAY=DELAY, BATCH=BATCH,
                         steps=STEPS, lr=LR, seeds=SEEDS,
                         t3_beta=T3_BETA, t3_K=T3_K, t2_gamma=T2_GAMMA,
                         hysteresis_margin=HYSTERESIS_MARGIN,
                         t1_period_primary=T1_PERIOD_PRIMARY,
                         t1_period_secondary=T1_PERIOD_SECONDARY,
                         checkpoints=CHECKPOINTS),
              runs=all_runs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b6_prospective_tracking_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
