"""Factorize the learned metric: is the win in the phase or the gain?

The matrix recheck (recheck_curvature_matrix.py) killed the curvature-
mass story twice over: the rotational part of W is unreachable by any
symmetric mobility (structural floor), and the scale profile is ANTI-
correlated with (I + tau H)^{-1} (corr down to -0.9). But the phase of
w aligns with the exact-credit defect alpha (decompose_w_final.py), so
the win may live in credit-orientation repair (phase) rather than in
the task-specific gain (magnitude) — the reviewer's factorization
hypothesis:

    w_j = rho_j e^{i psi_j};   phase ~ prospective credit geometry,
                               gain  ~ task-specific optimization.

Four arms, all with the metric FROZEN and the model trained online
(same protocol, init, and data streams as co_variational_metric.py):

  online      w = 1
  phase_only  w = exp(i arg w_full)    — credit orientation, no gain
  mag_only    w = |w_full|             — gain, no orientation
  full_frozen w = w_full               — the trained metric as-is

w_full comes from a fresh RouteA run per seed (train_cell), seeds
{0,1,2}; w_full is saved to results/factorize_w/ for reuse.

REGISTERED BAR (fixed before running): the phase is the mechanism iff
phase_only closes >= 50% of the online->full gap on median final loss,
i.e. (L_online - L_phase) / (L_online - L_full) >= 0.5.

Run:  python factorize_w.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import train_cell, STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
ARMS = ["online", "phase_only", "mag_only", "full_frozen"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "factorize_w")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_frozen(seed, w_frozen):
    """Online training with a fixed metric — identical loop to
    depth_law.train_cell minus the meta-gradient (w never updates)."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
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
        print(f"seed {seed}: training routeA for w_full...", flush=True)
        _, w_full = train_cell(4, 50, seed)
        np.save(os.path.join(RESULTS_DIR, f"w_full_s{seed}.npy"),
                np.array(w_full))
        variants = {
            "online": [np.ones(tcg.N, np.complex128)
                       for _ in range(tcg.L)],
            "phase_only": [np.exp(1j * np.angle(wl)) for wl in w_full],
            "mag_only": [np.abs(wl).astype(np.complex128) for wl in w_full],
            "full_frozen": w_full,
        }
        for arm in ARMS:
            fl, fin = train_frozen(seed, variants[arm])
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<12s} final {fl:.4f} "
                  f"finite {fin}", flush=True)

    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    gap = med["online"] - med["full_frozen"]
    frac_phase = ((med["online"] - med["phase_only"]) / gap
                  if gap > 0 else float("nan"))
    frac_mag = ((med["online"] - med["mag_only"]) / gap
                if gap > 0 else float("nan"))
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"gap online->full {gap:.4f}; closed by phase {frac_phase:.2f}, "
          f"by magnitude {frac_mag:.2f}")
    win = frac_phase >= 0.5
    print(f"BAR: phase closes >= 50% of the gap  ->  "
          f"{'PHASE IS THE MECHANISM' if win else 'PHASE ALONE IS NOT IT'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS,
                           bar="phase closes >= 50% of online->full gap"),
               per_arm=table, medians=med,
               frac_phase=frac_phase, frac_mag=frac_mag, win=win)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
