"""STAGE A — offline gate for the analytic prospective operator.

Frozen program: PC0/RoutePC/benchmark bars/train_bench.py untouched.
This measures, OFFLINE, whether the fixed analytic prospective operator
(prospective_ops.py) improves the ACTUAL eligibility-projected layer
parameter gradient against exact BPTT, at online-trained params
(the deployment regime), per seed, over 8 independent probe batches.

PRIMARY ENDPOINT (per layer, per arm):
    cos(g^l_surrogate, g^l_exact),
    ||g^l_surrogate - g^l_exact|| / ||g^l_exact||
Pooled = cos over the full concatenated gradient (automatically
weighted by exact-gradient energy; NO |a_j| weighting anywhere).
Secondary: signal-level cos(qv, lam), phase error, norm ratio,
max amplitude.

Arms (layers 0..L-2 only; top layer untouched — its online gradient is
already exact): base / gain / raw / ema rho in {0.5, 0.9, 0.99} /
matched (rho_j = |a_j|) / oppphase (same-magnitude, opposite-phase
causal control).

PRE-REGISTERED MECHANISTIC SUCCESS (fixed before running, from the
directive):
  1. the stable prospective arm improves pooled projected-gradient
     alignment over analytic gain-only;
  2. the lead-sign arm beats the opposite-phase causal control;
  3. no nonfinite values.
Failure of either claim is scientifically informative. The best global
rho (by pooled alignment) is recorded for Stage B's "bestGlobal" arm.

Run:  python prospective_offline.py
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
from prospective_ops import build_err

SEEDS = [0, 1, 2, 3, 4]
BATCHES = 8
RHOS = [0.5, 0.9, 0.99]
ARMS = ["base", "gain", "raw", "ema0.5", "ema0.9", "ema0.99",
        "matched", "oppphase"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "prospective_offline")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_online(seed):
    """Plain online training to the probe params (deployment regime)."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        g = cvm.clip(tcg.flat_grads(G, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    return params, float(np.asarray(losses)[-100:].mean())


def blocks_vec(G, l):
    return np.concatenate([G["a"][l].ravel(), G["b"][l].ravel()])


def probe(params, rng):
    rows = []
    for _ in range(BATCHES):
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        lam = tcg.exact_lambda(params, q)
        G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        a_list = [np.asarray(params["a"][l]) for l in range(tcg.L)]
        row = {}
        for arm in ARMS:
            rho = (float(arm[3:]) if arm.startswith("ema") else None)
            err = build_err(q, a_list, arm if not arm.startswith("ema")
                            else "ema", rho)
            G_sur = tcg.assemble(params, h, x, r, err, Sa, Sb)
            finite = all(np.all(np.isfinite(blocks_vec(G_sur, l)))
                         for l in range(tcg.L))
            per_layer = []
            for l in range(tcg.L):
                gs, ge = blocks_vec(G_sur, l), blocks_vec(G_ex, l)
                cosl = float(np.abs(np.vdot(ge, gs))
                             / (np.linalg.norm(ge) * np.linalg.norm(gs)
                                + 1e-30))
                rel = float(np.linalg.norm(gs - ge)
                            / (np.linalg.norm(ge) + 1e-30))
                per_layer.append((cosl, rel))
            gs_full = np.concatenate([blocks_vec(G_sur, l)
                                      for l in range(tcg.L)])
            ge_full = np.concatenate([blocks_vec(G_ex, l)
                                      for l in range(tcg.L)])
            cos_full = float(np.abs(np.vdot(ge_full, gs_full))
                             / (np.linalg.norm(ge_full)
                                * np.linalg.norm(gs_full) + 1e-30))
            rel_full = float(np.linalg.norm(gs_full - ge_full)
                             / (np.linalg.norm(ge_full) + 1e-30))
            # secondary: signal-level vs exact lambda (layers 0..L-2)
            qhat = np.concatenate([np.ravel(err[l])
                                   for l in range(tcg.L - 1)])
            lm = np.concatenate([np.ravel(lam[l])
                                 for l in range(tcg.L - 1)])
            cos_sig = float(np.abs(np.vdot(lm, qhat))
                            / (np.linalg.norm(lm) * np.linalg.norm(qhat)
                               + 1e-30))
            phase_err = float(np.abs(np.angle(
                np.vdot(lm, qhat))))
            nrat = float(np.linalg.norm(qhat)
                         / (np.linalg.norm(lm) + 1e-30))
            amax = float(np.max(np.abs(qhat)))
            row[arm] = dict(per_layer=per_layer, cos=cos_full,
                            rel=rel_full, cos_sig=cos_sig,
                            phase_err=phase_err, norm_ratio=nrat,
                            amax=amax, finite=finite)
        rows.append(row)
    # median over batches
    out = {}
    for arm in ARMS:
        out[arm] = dict(
            per_layer=[(float(np.median([b[arm]["per_layer"][l][0]
                                         for b in rows])),
                        float(np.median([b[arm]["per_layer"][l][1]
                                         for b in rows])))
                       for l in range(tcg.L)],
            cos=float(np.median([b[arm]["cos"] for b in rows])),
            rel=float(np.median([b[arm]["rel"] for b in rows])),
            cos_sig=float(np.median([b[arm]["cos_sig"] for b in rows])),
            phase_err=float(np.median([b[arm]["phase_err"]
                                       for b in rows])),
            norm_ratio=float(np.median([b[arm]["norm_ratio"]
                                        for b in rows])),
            amax=float(np.max([b[arm]["amax"] for b in rows])),
            finite=all(b[arm]["finite"] for b in rows))
    return out


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agg = {arm: [] for arm in ARMS}
    finals = {}
    for seed in SEEDS:
        print(f"seed {seed}: online training to probe params...",
              flush=True)
        params, fin = train_online(seed)
        finals[seed] = fin
        prng = np.random.RandomState(777000 + seed)
        out = probe(params, prng)
        print(f"  online final {fin:.4f}; pooled cos vs exact:",
          flush=True)
        for arm in ARMS:
            o = out[arm]
            agg[arm].append(o)
            print(f"    {arm:<9s} cos {o['cos']:.3f}  rel {o['rel']:.3f}  "
                  f"per-layer cos "
                  f"{[round(c, 3) for c, _ in o['per_layer']]}  "
                  f"sig {o['cos_sig']:.3f}  phi {o['phase_err']:.3f}  "
                  f"|qhat|/|lam| {o['norm_ratio']:.2f}  "
                  f"amax {o['amax']:.1e}  fin {o['finite']}", flush=True)

    med = {arm: dict(cos=float(np.median([o["cos"] for o in agg[arm]])),
                     rel=float(np.median([o["rel"] for o in agg[arm]])),
                     cos_sig=float(np.median([o["cos_sig"]
                                              for o in agg[arm]])),
                     per_layer=[float(np.median(
                         [o["per_layer"][l][0] for o in agg[arm]]))
                         for l in range(tcg.L)],
                     finite=all(o["finite"] for o in agg[arm]))
           for arm in ARMS}
    print("-" * 78)
    print("medians over seeds (pooled cos / per-layer cos):")
    for arm in ARMS:
        print(f"  {arm:<9s} {med[arm]['cos']:.3f}   "
              f"{[round(c, 3) for c in med[arm]['per_layer']]}")

    pro_arms = ["raw", "ema0.5", "ema0.9", "ema0.99", "matched"]
    best_pro = max(pro_arms, key=lambda a: med[a]["cos"])
    best_rho = max([a for a in pro_arms if a.startswith("ema")],
                   key=lambda a: med[a]["cos"])
    bar1 = med[best_pro]["cos"] > med["gain"]["cos"]
    bar2 = med[best_pro]["cos"] > med["oppphase"]["cos"]
    bar3 = all(med[a]["finite"] for a in ARMS)
    print(f"best prospective arm: {best_pro} (pooled cos "
          f"{med[best_pro]['cos']:.3f}); gain {med['gain']['cos']:.3f}; "
          f"oppphase {med['oppphase']['cos']:.3f}")
    print(f"best global rho for Stage B: {best_rho}")
    print(f"BAR1 prospective > gain-only: {bar1}")
    print(f"BAR2 prospective > oppphase:  {bar2}")
    print(f"BAR3 all finite:              {bar3}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, batches=BATCHES,
                           rhos=RHOS),
               online_finals={str(s): finals[s] for s in SEEDS},
               per_arm=med, best_prospective=best_pro,
               best_global_rho=best_rho,
               bars=dict(pro_beats_gain=bool(bar1),
                         pro_beats_oppphase=bool(bar2),
                         all_finite=bool(bar3)))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
