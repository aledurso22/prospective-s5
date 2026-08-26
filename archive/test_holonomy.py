"""Holonomy test (directive 04, Test B) — is the shallow phase a
discrete parallel transport down the stack?

Model: arg w_j^l = sum_{m>=l} delta_m with delta shared across modes
(4 dof vs 64 phases per seed — heavily overdetermined, can genuinely
fail). Sharpest form: per-mode layer increments
Delta^l_j = arg w^l_j - arg w^{l+1}_j must be mode-INDEPENDENT
(concentrated) for additivity to hold.

Data: results/factorize_w/w_full_s{0,1,2}.npy (on disk, no retraining).

BAR (fixed in docstring before running): holonomy consistent iff every
increment concentration R_l > 0.7 AND cumulative reconstruction
R > 0.8 in every layer, in >= 2 of 3 seeds. Else the metriplectic/
connection reading of the shallow phase is decoration, not prediction.

Run:  python test_holonomy.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

SEEDS = [0, 1, 2]
W_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "factorize_w")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "holonomy")


def circ_mean(angles):
    z = np.exp(1j * np.asarray(angles)).mean()
    return float(np.angle(z)), float(np.abs(z))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    L = 4
    out = {}
    for seed in SEEDS:
        w = np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy"))
        arg = [np.angle(wl) for wl in w]
        inc_R, deltas = [], []
        for l in range(L - 1):
            d, R = circ_mean(arg[l] - arg[l + 1])
            deltas.append(d)
            inc_R.append(R)
        d_last, _ = circ_mean(arg[L - 1])
        deltas_full = deltas + [d_last]
        recon_R = []
        for l in range(L):
            pred = sum(deltas_full[l:])
            recon_R.append(circ_mean(arg[l] - pred)[1])
        out[seed] = dict(inc_R=inc_R, deltas=deltas_full, recon_R=recon_R)
        print(f"  seed {seed}: increment R {['%.3f' % r for r in inc_R]}  "
              f"deltas {['%+.3f' % d for d in deltas_full]}  "
              f"recon R {['%.3f' % r for r in recon_R]}", flush=True)
    passes = sum(
        all(r > 0.7 for r in out[s]["inc_R"])
        and all(r > 0.8 for r in out[s]["recon_R"]) for s in SEEDS)
    win = passes >= 2
    print("-" * 70)
    print(f"BAR: all increments R>0.7 and recon R>0.8 in >=2 seeds  ->  "
          f"{'HOLONOMY CONSISTENT' if win else 'NO HOLONOMY — connection reading is decoration'}"
          f"  ({passes}/3 seeds)")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, per_seed=out, win=bool(win)), f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
