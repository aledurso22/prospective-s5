"""STAGE B — training with the fixed analytic prospective operator at the
Zucchet-approximation site (layers 0..L-2, before eligibility
projection; top recurrent layer untouched).

Stage A (prospective_offline.py) registered verdict: BOTH bars FAILED —
every fixed prospective arm DEGRADED pooled projected-gradient
alignment vs exact BPTT at trained params (base 0.440 > gain 0.307 >
oppphase 0.269 > matched 0.249 > ema0.99 0.226 > raw 0.174; raw lead
norm ratio ~700 from c1* = r/(1-r)^2). Per the program's own D3 lesson
(static alignment does not determine deployment), Stage B runs the
training comparison as registered rather than inferring it from A.

Arms (paired 5 seeds, identical optimizer/streams as every prior arm):
  online        raw q everywhere (reference; stored finals)
  gain          c0* q'
  raw           c0* q' + c1* (q'_t - q'_{t-1})
  matched       ema lead with rho_j = |a_j|
  bestGlobal    ema lead with rho = 0.99 (best global rho from Stage A)
  oppphase      same-magnitude/opposite-phase causal control
  PC0           stored causal-correction reference

Report: final loss per seed/arm + per-layer BPTT alignment at the end
of training. No new hyperparameters; clip + Adam as in every prior arm.

Run:  python prospective_train.py
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
from archive.prospective_ops import build_err

SEEDS = [0, 1, 2, 3, 4]
ARMS = ["online", "gain", "raw", "matched", "bestGlobal", "oppphase"]
RHO_BEST = 0.99          # Stage A's best global rho
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "prospective_train")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_arm(seed, arm):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        loss = 0.5 * float(np.mean(r ** 2))
        losses.append(loss)
        if arm == "online":
            G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        else:
            a_list = [np.asarray(params["a"][l]) for l in range(tcg.L)]
            op = {"gain": "gain", "raw": "raw", "matched": "matched",
                  "bestGlobal": "ema", "oppphase": "oppphase"}[arm]
            err = build_err(q, a_list, op,
                            RHO_BEST if arm == "bestGlobal" else None)
            G = tcg.assemble(params, h, x, r, err, Sa, Sb)
        g = cvm.clip(tcg.flat_grads(G, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    losses = np.asarray(losses)

    # per-layer BPTT alignment at the final params (same operator state)
    prng = np.random.RandomState(999000 + seed)
    x, y = make_data(prng)
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    lam = tcg.exact_lambda(params, q)
    G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
    if arm == "online":
        G_al = tcg.assemble(params, h, x, r, q, Sa, Sb)
    else:
        a_list = [np.asarray(params["a"][l]) for l in range(tcg.L)]
        op = {"gain": "gain", "raw": "raw", "matched": "matched",
              "bestGlobal": "ema", "oppphase": "oppphase"}[arm]
        err = build_err(q, a_list, op,
                        RHO_BEST if arm == "bestGlobal" else None)
        G_al = tcg.assemble(params, h, x, r, err, Sa, Sb)
    align = []
    for l in range(tcg.L):
        gs = np.concatenate([G_al["a"][l].ravel(), G_al["b"][l].ravel()])
        ge = np.concatenate([G_ex["a"][l].ravel(), G_ex["b"][l].ravel()])
        align.append(float(np.abs(np.vdot(ge, gs))
                           / (np.linalg.norm(ge) * np.linalg.norm(gs)
                              + 1e-30)))
    return dict(arm=arm, seed=seed,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                align=align)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "route_pc", "summary.json")))
    finals = {"PC0": {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}}
    aligns = {}
    for arm in ARMS:
        finals[arm] = {}
        aligns[arm] = {}
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            out = train_arm(seed, arm)
            finals[arm][seed] = out["final_loss"]
            aligns[arm][seed] = out["align"]
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  "
                  f"align {[round(a, 3) for a in out['align']]}",
                  flush=True)

    print("-" * 78)
    med = {a: float(np.median([finals[a][s] for s in SEEDS]))
           for a in ["online"] + ARMS[1:] + ["PC0"]}
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    for a in ["online", "gain", "raw", "matched", "bestGlobal",
              "oppphase", "PC0"]:
        print(f"  {a:<10s} {['%.4f' % finals[a][s] for s in SEEDS]}")
    print("per-layer alignment at final params (medians):")
    for a in ARMS:
        per_l = [float(np.median([aligns[a][s][l] for s in SEEDS]))
                 for l in range(tcg.L)]
        print(f"  {a:<10s} {[round(c, 3) for c in per_l]}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, rho_best=RHO_BEST,
                           stageA="BOTH BARS FAILED (see "
                                  "results/prospective_offline)"),
               finals={a: {str(s): finals[a][s] for s in SEEDS}
                       for a in finals},
               medians=med,
               aligns={a: {str(s): aligns[a][s] for s in SEEDS}
                       for a in aligns})
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
