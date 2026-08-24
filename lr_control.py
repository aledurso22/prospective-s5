"""LR control for covariant_adam: is the win covariance or just rate?

covariant_adam seed 0: online_cov 0.0028 vs online 0.0284. The boring
explanation: shared-v Adam is a different effective learning rate for
the complex entries, and standard Adam at the right LR would match.
Control: online arm, standard Adam, LR swept over 4 values spanning
2 orders of magnitude. If none reaches the covAdam neighborhood, the
win is structural (covariance), not rate.

Arms: online with LR in {3e-4, 1e-3, 3e-3, 1e-2}, paired seeds {0,1,2}.
Reference: covAdam 0.0028 (seed 0; full numbers when its run lands).

Run:  python lr_control.py
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
LRS = [3e-4, 1e-3, 3e-3, 1e-2]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "lr_control")


def train_online(seed, lr):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        g = cvm.clip(tcg.flat_grads(G, params))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - lr * (m / (1 - b1 ** step)) / (
            np.sqrt(v / (1 - b2 ** step)) + eps)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses)))


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table = {}
    for lr in LRS:
        for seed in SEEDS:
            fl, fin = train_online(seed, lr)
            table.setdefault(lr, []).append(fl)
            print(f"  LR {lr:g} seed {seed}: final {fl:.4f} finite {fin}",
                  flush=True)
    med = {lr: float(np.median(v)) for lr, v in table.items()}
    print("-" * 70)
    print(f"medians: { {f'{lr:g}': round(v, 4) for lr, v in med.items()} }")
    best = min(med, key=med.get)
    print(f"best standard-Adam LR {best:g} -> {med[best]:.4f}; "
          f"covAdam seed-0 reference 0.0028")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, per_arm={str(k): v for k, v in table.items()},
                       medians={str(k): v for k, v in med.items()}),
                  f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
