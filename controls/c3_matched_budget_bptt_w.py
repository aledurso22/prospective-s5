"""C3 — matched-headroom BPTT+w control (addendum control 3 of 3).

The final-step 2x2 control floors BPTT near 1e-4, so BPTT+w ~= BPTT
there has a headroom limitation: maybe w cannot help simply because
BPTT has nothing left to gain. This control re-tests the interaction
at EARLIER common budgets K, while BPTT still has room to improve —
without tuning BPTT to be bad, without arm-specific early stopping,
without separately tuned learning rates.

Protocol: retrain the four arms of the frozen 2x2 (online, PC0, BPTT,
BPTT+w) with per-step loss recording, identical paired streams and
optimizers, same frozen rig. Each arm is a verbatim replica of the
frozen implementation (cvm.train_route for online/bptt;
control_2x2_normmatch for PC0/BPTT+w). GATE: every arm's final loss
must reproduce the stored finals BITWISE (same-protocol proof):
online/PC0 vs results/route_pc/summary.json, BPTT vs
results/co_variational_metric/bptt_s*.json, BPTT+w vs
results/control_2x2_normmatch/summary.json.

Budget evaluation: L(K) = mean of the 25 steps ending at K (K >= 25),
fixed for all arms.

K SELECTION RULE (registered before running): let B_med(K) be the
median over seeds of L_BPTT(K), and L_on_med the stored online median
final (0.02242807737868163). For each target in
{2.0x, 1.0x, 0.25x} x L_on_med, pick K = the first step where
B_med(K) <= target. Also report K = 1500 (the original floored budget)
for reference.

At each K report per seed:
  L_online,i(K), L_PC0,i(K), L_BPTT,i(K), L_BPTT+w,i(K),
  Delta_credit,i(K) = L_online,i - L_BPTT,i  (precondition),
  I_i(K) = [L_online,i - L_PC0,i] - [L_BPTT,i - L_BPTT+w,i],
plus medians. If w is generic preconditioning, BPTT+w should improve
over BPTT when headroom exists (I_i(K) -> 0). If the benefit is
credit-specific, I_i(K) stays large even with headroom.

Run:  python -m controls.c3_matched_budget_bptt_w
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig import routepc as rp
from toyrig.train_cell import STEPS
from toyrig.probes import make_data
from diagnostics.prospective_kappa import chain_c_stored

SEEDS = [0, 1, 2, 3, 4]
LR, LR_M = cvm.LR, cvm.LR_M
WIN = 25                      # L(K) = mean of the WIN steps ending at K
TARGETS = [2.0, 1.0, 0.25]    # x stored online median final
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "c3_matched_budget_bptt_w")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def _prev_blocks(G, params):
    return (dict(a=[ga.copy() for ga in G["a"]],
                 b=[gb.copy() for gb in G["b"]]),
            [th.copy() for th in params["theta"]],
            [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
            [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
             for l in range(tcg.L)])


def train_arm(arm, seed):
    """Verbatim replicas of the frozen arms, + per-step loss recording.

    online/bptt: cvm.train_route's loop (no metric / exact gradient).
    PC0: control_2x2_normmatch.train_pc0_variant(norm_matched=False).
    BPTT+w: control_2x2_normmatch.train_bptt_w.
    """
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = np.empty(STEPS)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses[step - 1] = loss
        if arm == "online":
            g = cvm.clip(tcg.flat_grads(G, params))
        elif arm == "bptt":
            g = cvm.clip(tcg.flat_grads(cvm.exact_grad(params, x, y),
                                      params))
        elif arm == "pc0":
            h_n = tcg.flat_grads(G, params)
            if prev is not None:
                Gp, th_all, u_all, sig_all = prev
                r = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
                w = [wl - LR_M * (-LR) * rl for wl, rl in zip(w, r)]
            g = cvm.clip(tcg.flat_grads(cvm.scale_by_w(G, w), params))
        else:  # bptt_w
            h_on = tcg.flat_grads(G, params)
            G_ex = cvm.exact_grad(params, x, y)
            if prev is not None:
                Gp, th_all, u_all, sig_all = prev
                r = chain_c_stored(Gp, th_all, u_all, sig_all, h_on)
                w = [wl - LR_M * (-LR) * rl for wl, rl in zip(w, r)]
            g = cvm.clip(tcg.flat_grads(cvm.scale_by_w(G_ex, w), params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        if arm in ("pc0", "bptt_w"):
            prev = _prev_blocks(G, params)
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    return losses


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c2 = json.load(open(os.path.join(ROOT, "results",
                                     "control_2x2_normmatch",
                                     "summary.json")))
    stored = {
        "online": {s: rp_ref["finals"]["online"][str(s)] for s in SEEDS},
        "pc0": {s: rp_ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS},
        "bptt": {s: json.load(open(os.path.join(
            ROOT, "results", "co_variational_metric",
            f"bptt_s{s}.json")))["final_loss"] for s in SEEDS},
        "bptt_w": {s: c2["finals"]["bptt_w"][str(s)] for s in SEEDS},
    }

    curves = {arm: {} for arm in stored}
    for arm in ["online", "pc0", "bptt", "bptt_w"]:
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            curves[arm][seed] = train_arm(arm, seed)
            print(f"  final {curves[arm][seed][-100:].mean():.4f}",
                  flush=True)
        np.save(os.path.join(RESULTS_DIR, f"curve_{arm}.npy"),
                np.stack([curves[arm][s] for s in SEEDS]))

    # ---- bitwise same-protocol gates ----
    gates = {}
    for arm in stored:
        gates[arm] = max(abs(curves[arm][s][-100:].mean()
                             - stored[arm][s]) for s in SEEDS)
        print(f"GATE {arm}: max |dfinal| vs stored {gates[arm]:.2e}  "
              f"{'PASS' if gates[arm] == 0.0 else 'FAIL'}")
    assert all(g == 0.0 for g in gates.values())

    L = lambda arm, K: {s: curves[arm][s][K - WIN:K].mean()
                        for s in SEEDS}
    bmed_curve = np.median(np.stack([curves["bptt"][s] for s in SEEDS]),
                           axis=0)
    bmed_smooth = np.convolve(bmed_curve, np.ones(WIN) / WIN,
                              mode="valid")          # index K-WIN -> L(K)
    l_on_med = float(np.median([stored["online"][s] for s in SEEDS]))
    Ks = {}
    for t in TARGETS:
        idx = np.nonzero(bmed_smooth <= t * l_on_med)[0]
        Ks[f"{t}x_online_med"] = int(idx[0] + WIN) if len(idx) else None
    Ks["final"] = STEPS

    print("-" * 78)
    print(f"selected budgets: {Ks}")
    rows = {}
    for tag, K in Ks.items():
        if K is None:
            print(f"{tag}: BPTT median never reaches target — skipped")
            rows[tag] = None
            continue
        Lo, Lc = L("online", K), L("pc0", K)
        Lb, Lw = L("bptt", K), L("bptt_w", K)
        D = {s: Lo[s] - Lb[s] for s in SEEDS}
        I = {s: (Lo[s] - Lc[s]) - (Lb[s] - Lw[s]) for s in SEEDS}
        iv = np.array([I[s] for s in SEEDS])
        rows[tag] = dict(K=K,
                         online=Lo, pc0=Lc, bptt=Lb, bptt_w=Lw,
                         delta_credit=D, interaction=I,
                         interaction_median=float(np.median(iv)),
                         precondition_positive=int(sum(D[s] > 0
                                                       for s in SEEDS)))
        print(f"\nK={K} ({tag}):")
        print(f"  online  {['%.5f' % Lo[s] for s in SEEDS]}")
        print(f"  pc0     {['%.5f' % Lc[s] for s in SEEDS]}")
        print(f"  bptt    {['%.5f' % Lb[s] for s in SEEDS]}")
        print(f"  bptt+w  {['%.5f' % Lw[s] for s in SEEDS]}")
        print(f"  Delta_credit {['%+.5f' % D[s] for s in SEEDS]}  "
              f"(positive {rows[tag]['precondition_positive']}/5)")
        print(f"  I(K)    {['%+.5f' % I[s] for s in SEEDS]}  "
              f"median {np.median(iv):+.5f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, seeds=SEEDS,
                           win=WIN, targets=TARGETS,
                           k_rule=("first K where median L_BPTT(K) <= "
                                   "target x stored online median final")),
               gates=gates, budgets=Ks,
               rows={t: (r if r is None else
                         {k: (v if not isinstance(v, dict) else
                              {str(s): vv for s, vv in v.items()})
                          for k, v in r.items()})
                     for t, r in rows.items()})
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json (+ curve_*.npy)")


if __name__ == "__main__":
    main()
