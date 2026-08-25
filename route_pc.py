"""routePC — fully causal prediction-correction orientation learning.

PRE-CLUSTER MECHANISM GATE (exploratory; the registered benchmark protocol
is unchanged). Route A's w is meta-learned from a BPTT teacher:

    w <- w - LR_M * (-LR) * (du + i dv),   du/dv = [d_w (M_w g)]^dagger gN,
    gN = exact BPTT gradient at params_next on the SAME batch
         (co_variational_metric.py, FD-gated convention).

routePC removes the BPTT teacher entirely. The ONLY substitution is

    gN  ->  h_n := g_online(theta_n, newly arrived batch n)

i.e. the realized online gradient at the post-update parameters on the new
batch corrects the geometry that produced that model step (one step late —
delayed correction). NO exact_grad / exact_lambda / reverse temporal
cotangent call occurs anywhere in the PC arms (audited, see below).

Timing per batch n (causal ordering, per the directive):
  1. compute the normal online gradient blocks G_n at theta_n;
  2. correct the geometry that generated the previous model step:
         w_corr = w_pred_prev - LR_M * (-LR) * (du + i dv)
     with du/dv built from the STORED previous unscaled blocks G_{n-1},
     theta_{n-1} and the current h_n — the identical analytic u/v chain
     as co_variational_metric.py (no new complex convention);
  3. predict (PC1): w_pred = w_corr + beta * (w_corr - w_corr_prev),
     displacement clamped elementwise to |disp| <= DELTA_MAX (the clamp
     exists so prediction can never be the source of an unstable win;
     its fire count is logged);
     PC0 is beta = 0 (correction only);
  4. main update: Adam on clip(flat(scale_by_w(G_n, w_pred))) — identical
     to Route A's model optimizer.

Arms: online, routeA (both reused verbatim from cvm.train_route), pc0,
pc1_b025, pc1_b050. Toy delayed-copy rig: T=128, D=50, L=4, N=16,
batch=32, STEPS=1500, LR=LR_M=1e-3, CLIP=1.0, seeds {0..4}, paired data
streams (same rng per seed across arms).

REGISTERED EXPLORATORY GATE (fixed before running, from the directive):
  * PC is interesting iff the best PC arm beats online on >= 4/5 paired
    seeds AND median R_gap >= 0.30, where per-seed
        R_gap = (L_online - L_arm) / (L_online - L_routeA).
  * Prediction is load-bearing only if the better fixed beta in {0.25,
    0.5} beats PC0 on >= 4/5 paired seeds AND improves the median R_gap.
    Otherwise retain correction-only; no Simonetto prediction story.
  * Strict audit: BPTT_CALLS(routePC) = 0 (counted wrappers on
    cvm.exact_grad and tcg.exact_lambda; asserted per arm).

If the gate fails: stop here, continue the registered Route-A cluster
plan. If it passes: port the winning causal arm into the JAX pipeline as
an exploratory supplemental arm (primary bars unchanged).

Run:  python route_pc.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2, 3, 4]
BETAS = [0.0, 0.25, 0.5]           # fixed, preregistered; 0.0 IS PC0
DELTA_MAX = 0.2                    # prediction-displacement clamp (|dw|)
LR_M = cvm.LR_M
LR = cvm.LR
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "route_pc")

# ---- BPTT audit: counting wrappers on the two non-causal entry points ----
BPTT_CALLS = {"exact_grad": 0, "exact_lambda": 0}
_orig_exact_grad = cvm.exact_grad
_orig_exact_lambda = tcg.exact_lambda


def _count_grad(*a, **k):
    BPTT_CALLS["exact_grad"] += 1
    return _orig_exact_grad(*a, **k)


def _count_lambda(*a, **k):
    BPTT_CALLS["exact_lambda"] += 1
    return _orig_exact_lambda(*a, **k)


cvm.exact_grad = _count_grad
tcg.exact_lambda = _count_lambda


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_pc(seed, beta):
    """Prediction-correction arm. beta = 0 -> PC0 (correction only)."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)     # paired with cvm arms
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    w_corr_prev = None
    prev = None            # (G_blocks, theta, u_mode, sigp) of step n-1
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    amax_hist = []
    phase_track_ = []
    corr_mag, pred_mag = [], []
    clamp_fires = 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        h_n = tcg.flat_grads(G, params)      # realized online gradient

        # ---- (2) delayed correction of the previous geometry ----
        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            off = 0
            w_corr = []
            for l in range(tcg.L):
                th = th_all[l]
                u_mode = u_all[l]
                sigp = sig_all[l]
                A = Gp["a"][l] * np.exp(1j * th)           # (N,)
                Gb = Gp["b"][l]                            # (N, M)
                M_ = Gb.shape[1]
                gN_rho = h_n[off:off + tcg.N]
                gN_theta = h_n[off + tcg.N:off + 2 * tcg.N]
                gN_bre = h_n[off + 2 * tcg.N:
                             off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
                gN_bim = h_n[off + 2 * tcg.N + tcg.N * M_:
                             off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                 tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w_corr.append(w_pred[l] - LR_M * (-LR) * (du + 1j * dv))
            corr_mag.append(float(np.mean(
                [np.abs(wc - wp).mean() for wc, wp in zip(w_corr, w_pred)])))

            # ---- (3) prediction (PC1; PC0 is beta = 0) ----
            if beta > 0.0 and w_corr_prev is not None:
                disp = [beta * (wc - wcp)
                        for wc, wcp in zip(w_corr, w_corr_prev)]
                over = [np.abs(d) > DELTA_MAX for d in disp]
                if any(np.any(o) for o in over):
                    clamp_fires += 1
                    disp = [np.where(o, d * (DELTA_MAX / np.abs(d)), d)
                            for d, o in zip(disp, over)]
                pred_mag.append(float(np.mean(
                    [np.abs(d).mean() for d in disp])))
                w_pred = [wc + d for wc, d in zip(w_corr, disp)]
            else:
                w_pred = w_corr
            w_corr_prev = [wc.copy() for wc in w_corr]

        # ---- store this step's pre-update blocks for the next correction --
        # (the u/v chain pairs G_{n-1} with the params it was evaluated at)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(tcg.L)])

        # ---- (4) main update with the predicted geometry ----
        G_use = cvm.scale_by_w(G, w_pred)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)

        if step % 200 == 0:
            amax = max(float(np.abs(aa).max()) for aa in params["a"])
            amax_hist.append(amax)
            print(f"      pc(b={beta}) s{seed} step {step}: loss "
                  f"{loss:.4f}  max|a| {amax:.4f}", flush=True)
        if step % 25 == 0:
            phase_track_.append([np.angle(wl) for wl in w_pred])

    losses = np.asarray(losses)
    return dict(arm=f"pc_b{beta}", seed=seed, beta=beta,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                w_final=[wl.copy() for wl in w_pred],
                amax_end=amax_hist[-1] if amax_hist else None,
                w_abs_mean=float(np.mean([np.abs(wl).mean()
                                          for wl in w_pred])),
                argw_final=[np.angle(wl).tolist() for wl in w_pred],
                corr_step_mean=float(np.mean(corr_mag)) if corr_mag else 0.0,
                pred_disp_mean=float(np.mean(pred_mag)) if pred_mag else 0.0,
                clamp_fires=int(clamp_fires),
                wall_time_sec=time.time() - t0)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 78)
    print("routePC — causal prediction-correction orientation (BPTT-free)")
    print("=" * 78)
    results = {}
    audit = {}

    for arm in ["online", "routeA"]:
        before = dict(BPTT_CALLS)
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            out = cvm.train_route(arm, seed)
            out = {k: v for k, v in out.items() if k != "w_final"}
            results[f"{arm}/s{seed}"] = out
            print(f"  {arm} s{seed}: final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}", flush=True)
        audit[arm] = {k: BPTT_CALLS[k] - before[k] for k in BPTT_CALLS}

    for beta in BETAS:
        arm = f"pc_b{beta}"
        before = dict(BPTT_CALLS)
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            out = train_pc(seed, beta)
            results[f"{arm}/s{seed}"] = out
            np.save(os.path.join(RESULTS_DIR, f"w_{arm}_s{seed}.npy"),
                    np.array(out["w_final"]))
            out = {k: v for k, v in out.items() if k != "w_final"}
            results[f"{arm}/s{seed}"] = out
            print(f"  {arm} s{seed}: final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  "
                  f"|w| {out['w_abs_mean']:.3f}  "
                  f"corr|dw| {out['corr_step_mean']:.2e}  "
                  f"pred|dw| {out['pred_disp_mean']:.2e}  "
                  f"clamps {out['clamp_fires']}", flush=True)
        audit[arm] = {k: BPTT_CALLS[k] - before[k] for k in BPTT_CALLS}

    # ---- strict audit: PC arms must have made ZERO BPTT calls ----
    for arm in audit:
        if arm.startswith("pc_"):
            assert audit[arm]["exact_grad"] == 0
            assert audit[arm]["exact_lambda"] == 0
    print(f"BPTT audit (calls per arm): {audit}")

    # ---- paired statistics ----
    med = {}
    arms = ["online", "routeA"] + [f"pc_b{b}" for b in BETAS]
    finals = {arm: {s: results[f"{arm}/s{s}"]["final_loss"]
                    for s in SEEDS} for arm in arms}
    for arm in arms:
        med[arm] = float(np.median([finals[arm][s] for s in SEEDS]))
    rgap = {arm: {s: (finals["online"][s] - finals[arm][s])
                  / (finals["online"][s] - finals["routeA"][s])
                  for s in SEEDS} for arm in arms if arm != "online"}
    beats_online = {arm: sum(finals[arm][s] < finals["online"][s]
                             for s in SEEDS) for arm in arms}

    pc_arms = [f"pc_b{b}" for b in BETAS]
    best_pc = max(pc_arms, key=lambda a: float(np.median(
        [rgap[a][s] for s in SEEDS])))
    best_pc_med = float(np.median([rgap[best_pc][s] for s in SEEDS]))
    gate_pass = (beats_online[best_pc] >= 4 and best_pc_med >= 0.30)

    pc1_arms = [a for a in pc_arms if a != "pc_b0.0"]
    best_pc1 = max(pc1_arms, key=lambda a: float(np.median(
        [rgap[a][s] for s in SEEDS])))
    pc1_beats_pc0 = sum(finals[best_pc1][s] < finals["pc_b0.0"][s]
                        for s in SEEDS)
    pc1_med = float(np.median([rgap[best_pc1][s] for s in SEEDS]))
    pc0_med = float(np.median([rgap["pc_b0.0"][s] for s in SEEDS]))
    prediction_loadbearing = (pc1_beats_pc0 >= 4 and pc1_med > pc0_med)

    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    for arm in arms:
        if arm == "online":
            continue
        print(f"  {arm:<10s} R_gap per seed "
              f"{['%.2f' % rgap[arm][s] for s in SEEDS]}  "
              f"median {np.median([rgap[arm][s] for s in SEEDS]):.2f}")
    print(f"beats online (of 5): "
          f"{ {a: beats_online[a] for a in arms if a != 'online'} }")
    print(f"GATE: best PC = {best_pc} (median R_gap {best_pc_med:.2f}, "
          f"beats online {beats_online[best_pc]}/5)  ->  "
          f"{'PASS — port winning arm to JAX as exploratory supplement'
            if gate_pass else 'FAIL — continue registered Route-A plan'}")
    print(f"prediction load-bearing: best PC1 = {best_pc1} "
          f"(beats PC0 {pc1_beats_pc0}/5, median R_gap {pc1_med:.2f} vs "
          f"PC0 {pc0_med:.2f})  ->  "
          f"{'YES' if prediction_loadbearing else 'NO — correction-only'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, clip=cvm.CLIP,
                           seeds=SEEDS, betas=BETAS, delta_max=DELTA_MAX,
                           L=4, N=16, T=128, delay=50, batch=32,
                           bar=("best PC beats online >=4/5 paired seeds "
                                "and median R_gap >= 0.30")),
               audit=audit, finals=finals, medians=med,
               rgap={a: {str(s): rgap[a][s] for s in SEEDS}
                     for a in rgap},
               best_pc=best_pc, gate_pass=bool(gate_pass),
               prediction_loadbearing=bool(prediction_loadbearing),
               per_run=results)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
