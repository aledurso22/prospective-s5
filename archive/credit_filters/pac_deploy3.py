"""PAC deploy 3 — the simplest statistic (EXPLORATORY, post-Analysis-A).

Analysis A (preregistered, FAIL for the resolvent framing) found that
arg rho(1) ALONE predicts the learned phase better than the resolvent
combination -arg(1 - conj(a) rho(1)) at L2/L3. This arm deploys the
simpler law: w = e^{i arg rho(1)} with rho(1) from the full batch each
step (oracle) or EMA. Labeled EXPLORATORY: motivated by post-hoc
analysis, not preregistered; the comb-phase reference (0.0188 median)
comes from pac_deploy2 by determinism (identical protocol).

Exploratory bar: median <= comb-phase median (0.0188) AND >= 50% of
the online -> routeA gap (refs from pac_deploy/summary.json).

Run:  python pac_deploy3.py
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
from pac_deploy2 import rho1_of, setup

SEEDS = [0, 1, 2]
EMA_GAMMA = 0.05
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "pac_deploy3")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")
PAC2_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy2", "summary.json")


def train_arm(arm, seed):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    beta = [np.zeros(tcg.N, np.complex128) for _ in range(tcg.L)]
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        if arm == "rho_oracle":
            b_now = [rho1_of(ql) for ql in q]
        else:
            beta = [(1 - EMA_GAMMA) * b + EMA_GAMMA * rho1_of(ql)
                    for b, ql in zip(beta, q)]
            b_now = beta
        w = [np.exp(1j * np.angle(bl)) for bl in b_now]
        # step 1: beta = 0 -> angle(0) = 0 -> w = 1 = online (clean start)
        w = [np.where(np.abs(bl) > 0, wl, 1.0)
             for bl, wl in zip(b_now, w)]
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses)))


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    with open(PAC2_SUMMARY) as f:
        comb_med = json.load(f)["medians"]["pac_phase_oracle"]
    table = {}
    for seed in SEEDS:
        for arm in ["rho_oracle", "rho_ema"]:
            fl, fin = train_arm(arm, seed)
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<11s} final {fl:.4f} finite {fin}",
                  flush=True)
    med = {a: float(np.median(v)) for a, v in table.items()}
    gap = ref["online"] - ref["routeA"]
    fracs = {a: (ref["online"] - med[a]) / gap for a in med}
    print("-" * 70)
    print(f"medians { {k: round(v, 4) for k, v in med.items()} }  "
          f"comb-phase ref {comb_med:.4f}")
    print(f"fracs { {k: round(v, 2) for k, v in fracs.items()} }")
    win = {a: med[a] <= comb_med and fracs[a] >= 0.5 for a in med}
    print(f"exploratory bar (<= comb median AND >= 50% gap): "
          f"{ {k: ('PASS' if v else 'no') for k, v in win.items()} }")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, note="EXPLORATORY arm post Analysis A",
               refs=ref, comb_med=comb_med, per_arm=table,
               medians=med, fracs=fracs, win={k: bool(v) for k, v in win.items()})
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
