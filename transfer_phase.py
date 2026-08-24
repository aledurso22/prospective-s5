"""Phase transfer: the decisive test of operator-derived credit geometry.

factorize_w.py found phase-only closes 113% of the online->full gap on
the training task (D=50, T=128) — the phase is the mechanism. Two
confounds remain before claiming the phase is DERIVED structure (the
credit operator's matched-filter phase) rather than learned task noise:

  1. specificity: maybe ANY per-mode rotation helps. Control: random
     unit-modulus phases, same protocol.
  2. transfer: operator-derived phase depends on the mode a_j and the
     credit operator, not on the task loss; meta-learned task noise
     would not transfer. The gyroscopic-action reading predicts phase
     transfers, gain does not (transfer_m.py already showed frozen FULL
     w does not transfer — but that froze gain and phase together).

Protocol mirrors transfer_m.py: deploy on the UNSEEN task D=200, T=256,
1500 steps, frozen metrics, seeds {0,1,2}. Arms:

  online        w = 1
  phase         w = exp(i arg w_full)   — learned on D=50/T=128
  mag           w = |w_full|            — gain alone
  random_phase  w = exp(i U(0, 2pi))    — specificity control

w_full from results/factorize_w/w_full_s{seed}.npy (saved by
factorize_w.py).

REGISTERED BAR (fixed before running): the phase is derived, specific
structure iff median(phase) <= 0.5 x median(online) AND
median(random_phase) > 0.5 x median(online).

Run:  python transfer_phase.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
ARMS = ["online", "phase", "mag", "random_phase"]
EVAL_DELAY = 200
EVAL_T = 256
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "transfer_phase")
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "factorize_w")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, EVAL_T, EVAL_DELAY, 1, 32


def deploy(seed, w_frozen):
    """Online training with a frozen metric on the unseen task."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(2000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        G_use = cvm.scale_by_w(G, w_frozen)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses)))


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table = {}
    for seed in SEEDS:
        w_full = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        rng_ph = np.random.RandomState(7000 + seed)
        rand_w = [np.exp(1j * rng_ph.uniform(0, 2 * np.pi, tcg.N))
                  for _ in range(tcg.L)]
        variants = {
            "online": [np.ones(tcg.N, np.complex128)
                       for _ in range(tcg.L)],
            "phase": [np.exp(1j * np.angle(wl)) for wl in w_full],
            "mag": [np.abs(wl).astype(np.complex128) for wl in w_full],
            "random_phase": rand_w,
        }
        for arm in ARMS:
            fl, fin = deploy(seed, variants[arm])
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<13s} final {fl:.4f} "
                  f"finite {fin}", flush=True)

    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    win = (med["phase"] <= 0.5 * med["online"]
           and med["random_phase"] > 0.5 * med["online"])
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"BAR: phase <= 0.5x online AND random_phase > 0.5x online  ->  "
          f"{'DERIVED + SPECIFIC' if win else 'NO WIN'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, eval_delay=EVAL_DELAY,
                           eval_t=EVAL_T,
                           bar="phase <= 0.5x online, random_phase not"),
               per_arm=table, medians=med, win=win)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
