"""G3X — matched-headroom C3 crossover WITHOUT clipping (priority 5).

G3 found the phase/PC0 closed-loop benefit is entirely clip-dependent
and the real-gain benefit is clip-independent. The C3 credit-specificity
crossover (M_w helps defective online credit, hurts exact BPTT credit)
was run under the clipped regime. This reruns it with clipping made
NONBINDING (CLIP = 1e30, same code path, no LR retuning — registered):
is M_w still a credit-repair-specific object when the normalization
channel is removed?

Arms: online, pc0, bptt, bptt_w — the verbatim frozen replicas from
controls/c3_matched_budget_bptt_w.py with clip parameterized. Curves
recorded per step. Budgets reselected by the SAME rule on the no-clip
curves: first K where median L_BPTT(K) <= {2x, 1x, 0.25x} x median
no-clip online final, plus K=1500. L(K) = mean of the 25 steps ending
at K. Report L per arm, Delta_credit, and the interaction I_i(K) per
seed at every budget.

Run:  python -m controls.g3x_c3_noclip
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.train_cell import STEPS
from toyrig.probes import make_data
from diagnostics.prospective_kappa import chain_c_stored

SEEDS = [0, 1, 2, 3, 4]
LR, LR_M = cvm.LR, cvm.LR_M
WIN = 25
TARGETS = [2.0, 1.0, 0.25]
NOCLIP = 1e30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


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


def clip_nc(g):
    n = np.linalg.norm(g)
    return g * (NOCLIP / n) if n > NOCLIP else g


def train_arm(arm, seed):
    """c3 replicas with nonbinding clip (identical update order)."""
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
            g = clip_nc(tcg.flat_grads(G, params))
        elif arm == "bptt":
            g = clip_nc(tcg.flat_grads(cvm.exact_grad(params, x, y),
                                       params))
        elif arm == "pc0":
            h_n = tcg.flat_grads(G, params)
            if prev is not None:
                Gp, th_all, u_all, sig_all = prev
                r = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
                w = [wl - LR_M * (-LR) * rl for wl, rl in zip(w, r)]
            g = clip_nc(tcg.flat_grads(cvm.scale_by_w(G, w), params))
        else:  # bptt_w
            h_on = tcg.flat_grads(G, params)
            G_ex = cvm.exact_grad(params, x, y)
            if prev is not None:
                Gp, th_all, u_all, sig_all = prev
                r = chain_c_stored(Gp, th_all, u_all, sig_all, h_on)
                w = [wl - LR_M * (-LR) * rl for wl, rl in zip(w, r)]
            g = clip_nc(tcg.flat_grads(cvm.scale_by_w(G_ex, w), params))
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
    os.makedirs(OUT, exist_ok=True)
    curves = {}
    for arm in ["online", "pc0", "bptt", "bptt_w"]:
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            curves[(arm, seed)] = train_arm(arm, seed)
            print(f"  final {curves[(arm, seed)][-100:].mean():.4f}",
                  flush=True)

    L = lambda arm, K: {s: curves[(arm, s)][K - WIN:K].mean()
                        for s in SEEDS}
    bmed = np.median(np.stack([curves[("bptt", s)] for s in SEEDS]),
                     axis=0)
    bmed_s = np.convolve(bmed, np.ones(WIN) / WIN, mode="valid")
    on_med = float(np.median([curves[("online", s)][-100:].mean()
                              for s in SEEDS]))
    Ks = {}
    for t in TARGETS:
        idx = np.nonzero(bmed_s <= t * on_med)[0]
        Ks[f"{t}x_online_med"] = int(idx[0] + WIN) if len(idx) else None
    Ks["final"] = STEPS
    print(f"no-clip online median final {on_med:.4f}; budgets {Ks}")

    rows = {}
    for tag, K in Ks.items():
        if K is None:
            rows[tag] = None
            continue
        Lo, Lc = L("online", K), L("pc0", K)
        Lb, Lw = L("bptt", K), L("bptt_w", K)
        D = {s: Lo[s] - Lb[s] for s in SEEDS}
        I = {s: (Lo[s] - Lc[s]) - (Lb[s] - Lw[s]) for s in SEEDS}
        rows[tag] = dict(K=K, online=Lo, pc0=Lc, bptt=Lb, bptt_w=Lw,
                         delta_credit=D, interaction=I,
                         interaction_median=float(np.median(list(
                             I.values()))))
        print(f"\nK={K} ({tag}):")
        print(f"  online  {['%.5f' % Lo[s] for s in SEEDS]}")
        print(f"  pc0     {['%.5f' % Lc[s] for s in SEEDS]}")
        print(f"  bptt    {['%.5f' % Lb[s] for s in SEEDS]}")
        print(f"  bptt+w  {['%.5f' % Lw[s] for s in SEEDS]}")
        print(f"  I(K)    {['%+.5f' % I[s] for s in SEEDS]}  median "
              f"{np.median(list(I.values())):+.5f}  "
              f"D_credit>0 {sum(D[s] > 0 for s in SEEDS)}/5")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, noclip=True, budgets=Ks,
               rows={t: (r if r is None else
                         {k: (v if not isinstance(v, dict) else
                              {str(s): vv for s, vv in v.items()})
                          for k, v in r.items()})
                     for t, r in rows.items()})
    with open(os.path.join(OUT, "g3x_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g3x_summary.json")


if __name__ == "__main__":
    main()
