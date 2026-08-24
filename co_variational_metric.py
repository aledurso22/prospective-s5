"""The co-variational metric — the prospective action's second block.

The grid cell the unification (THEORY.md) left open: the mass matrix of
the prospective action is never *given* — it can be part of the
variational problem. Freeing it,

    R[q', M] = 1/2 q'^T M q' + (Phi(q + tau q') - Phi(q))/tau,

gives two Euler-Lagrange blocks: the flow (metric descent) and a learning
rule for the metric itself. This script tests the two honest realizations
of the metric block for online SSM learning, under the design rules the
credit lane's failures imposed:

  * the metric acts on the DESCENT field (per-mode preconditioner of the
    online gradient — never a filter on the error signal);
  * modes bounded by construction (a = sigmoid(rho) e^{i theta});
  * same-batch one-step lookahead (no cross-batch noise in the metric
    signal).

The metric: per-(layer, mode) complex gain w_j, applied as e_t = w_j q_t
(equivalently: conj(w_j) preconditioning of the mode's gradient block —
same linear object, see optimal_credit_filter.py). Init w = 1 (online
RTRL). The two routes for LEARNING w online:

  route A (meta-gradient): w descends the one-step-lookahead loss
      L(params - eta*g_w(batch)) on the SAME batch, via the analytic
      chain through the update. (Auxiliary note: the lookahead evaluation
      uses exact credit for the OUTER signal; the model's updates remain
      fully online/causal.)
  route B (consistency residual): the metric must make the realized
      same-batch energy change match the first-order prediction
      Re(G^H Delta). w descends ||DeltaL_real - DeltaL_pred||^2 — no
      exact credit anywhere. (First order constrains Re(w) only; Im(w)
      is free and held at 0 in this route.)

Arms: online (w=1 fixed), routeA, routeB, bptt (exact ceiling).
Task: delayed continuous copy, y_t = x_{t-50}, T=128, per-step loss.
Model: L=4, N=16, |a| init (0.90, 0.995) sigmoid-constrained.
Budget: batch 32, Adam lr 1e-3 (params) / 1e-3 (metric), clip 1.0,
1500 steps, seeds {0..4}.

REGISTERED BAR (fixed before running): the better metric route WINS iff
its median final loss <= 0.5 x online's median, all runs finite. max|a|
trajectories are reported throughout (the metric must not win by pushing
modes to the boundary — sigmoid prevents crossing, but we watch).

Run:  python co_variational_metric.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg

SEEDS = [0, 1, 2, 3, 4]
STEPS = 1500
LR = 1e-3
LR_M = 1e-3
CLIP = 1.0
ARMS = ["online", "routeA", "routeB", "bptt"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "co_variational_metric")


def setup():
    tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 128, 50, 1, 32


def make_data(rng, batch=32):
    x = rng.randn(tcg.T, batch)
    y = np.concatenate([np.zeros((tcg.DELAY, batch)), x[:-tcg.DELAY]],
                       axis=0)
    return x, y


def batch_grad(params, x, y):
    """Forward + online (S-slot) gradient, UNSCALED (metric applied
    separately by scale_by_w)."""
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    G = tcg.assemble(params, h, x, r, q, Sa, Sb)
    loss = 0.5 * float(np.mean(r ** 2))
    return loss, G, q, r, h


def scale_by_w(G, w):
    """Metric as descent-field preconditioner: conj(w) per mode on the
    gradient blocks (same linear object as filtering q by w)."""
    return dict(a=[np.conj(w[l]) * G["a"][l] for l in range(tcg.L)],
                b=[np.conj(w[l])[:, None] * G["b"][l]
                   for l in range(tcg.L)],
                c=G["c"])


def exact_grad(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    err = tcg.exact_lambda(params, q)
    return tcg.assemble(params, h, x, r, err, Sa, Sb, direct=True)


def adam(flat, g, m, v, step, lr=LR):
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g ** 2
    upd = lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step))
                                          + 1e-8)
    return flat - upd, m, v


def clip(g):
    n = np.linalg.norm(g)
    return g * (CLIP / n) if n > CLIP else g


def train_route(arm, seed):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    amax_hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = batch_grad(params, x, y)
        losses.append(loss)
        if arm == "bptt":
            g = tcg.flat_grads(exact_grad(params, x, y), params)
        else:
            G_use = G if arm == "online" else scale_by_w(G, w)
            g = tcg.flat_grads(G_use, params)
        g = clip(g)
        new_flat, m, v = adam(flat, g, m, v, step)
        params_next = tcg.pack(params, new_flat)

        # ---- metric learning (only routeA/routeB) ----
        if arm == "routeA":
            # one-step-lookahead exact gradient at params_next, same batch
            G_next = exact_grad(params_next, x, y)
            gN = tcg.flat_grads(G_next, params_next)
            # per-mode analytic meta-gradient through the update
            off = 0
            for l in range(tcg.L):
                th = params["theta"][l]
                u_mode = tcg.sig(params["rho"][l])
                sigp = u_mode * (1 - u_mode)
                A = G["a"][l] * np.exp(1j * th)          # (N,)
                Gb = G["b"][l]                             # (N, M_l)
                M_ = Gb.shape[1]
                gN_rho = gN[off:off + tcg.N]
                gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
                gN_bre = gN[off + 2 * tcg.N:
                            off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
                gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                            off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w[l] = w[l] - LR_M * (-LR) * (du + 1j * dv)
        elif arm == "routeB":
            # consistency residual: realized vs predicted same-batch dE
            _, yhat2 = tcg.forward(params_next, x)
            r2 = yhat2 - y
            r2[:tcg.DELAY] = 0.0
            loss_after = 0.5 * float(np.mean(r2 ** 2))
            dL_real = loss_after - loss
            # first-order predicted change: dE = -LR sum_j Re(w_j) blk2_j
            # (metric's own model; Adam normalization ignored by design)
            dpred = 0.0
            blk = []
            for l in range(tcg.L):
                blk2_j = np.abs(G["a"][l]) ** 2 + (
                    np.abs(G["b"][l]) ** 2).sum(axis=1)        # (N,)
                blk.append(blk2_j)
                dpred += (-LR * w[l].real * blk2_j).sum()
            resid = dL_real - dpred
            for l in range(tcg.L):
                grad_w = 2 * resid * LR * blk[l]               # (N,)
                w[l] = w[l] - LR_M * grad_w
        params = params_next
        if step % 200 == 0:
            amax = max(float(np.abs(aa).max()) for aa in params["a"])
            amax_hist.append(amax)
            print(f"      {arm} s{seed} step {step}: loss {loss:.4f}  "
                  f"max|a| {amax:.4f}", flush=True)

    losses = np.asarray(losses)
    return dict(arm=arm, seed=seed,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                amax_end=amax_hist[-1] if amax_hist else None,
                w_abs_mean=float(np.mean([np.abs(wl).mean() for wl in w])),
                wall_time_sec=time.time() - t0)


def main() -> None:
    setup()
    print("=" * 78)
    print("Co-variational metric — routes A (meta-grad) and B (residual)")
    print("=" * 78)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for arm in ARMS:
        finals = []
        for seed in SEEDS:
            out = train_route(arm, seed)
            finals.append(out["final_loss"])
            results[f"{arm}/s{seed}"] = out
            with open(os.path.join(RESULTS_DIR, f"{arm}_s{seed}.json"),
                      "w") as f:
                json.dump(out, f, indent=2)
        print(f"  {arm:<8s} finals {['%.4f' % x for x in finals]}  "
              f"median {np.median(finals):.4f}", flush=True)

    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final_loss"]
                                 for s in SEEDS])) for arm in ARMS}
    best_metric = min(med["routeA"], med["routeB"])
    finite_all = all(results[f"{arm}/s{s}"]["finite"]
                     for arm in ARMS for s in SEEDS)
    win = best_metric <= 0.5 * med["online"] and finite_all
    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"  routeA w|mean| {results['routeA/s0']['w_abs_mean']:.3f}  "
          f"routeB w|mean| {results['routeB/s0']['w_abs_mean']:.3f}")
    print(f"BAR: best route <= 0.5x online, all finite  ->  "
          f"{'WIN' if win else 'NO WIN'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, clip=CLIP,
                           seeds=SEEDS, bar="best route <= 0.5x online"),
               medians=med, win=win)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
