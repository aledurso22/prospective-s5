"""PAC deploy 4 — the horizon-1 form, estimation rate, and the
stability-barrier test.

Analysis B (preregistered, PASS) found the FIRST-ORDER truncation
c(1) = 1 + conj(a) rho(1) matches the learned phase best (monotone
decline from H=1) — but pac_deploy2 deployed the AR(1) CLOSURE
K = 1/(1 - conj(a) beta). The best static form was never deployed.
This script deploys c(1)'s phase and tests the named deployment barrier
(stability vs lag) with a hard prediction.

Arms (paired seeds {0,1,2}, same protocol):
  c1_oracle     w = e^{i arg(1 + conj(a) rho(1))}, rho(1) from the full
                batch each step
  c1_ema01      same, EMA gamma = 0.01
  c1_ema005     same, EMA gamma = 0.005
  c1_frozen200  same, re-estimated every 200 steps, w frozen between
                updates (stability test: if the barrier is variance,
                this beats EMA; if it's lag, this loses)

References online/routeA from pac_deploy/summary.json; comb-phase-oracle
(0.0188) from pac_deploy2/summary.json.

REGISTERED BARS (fixed before running):
  P4: best arm closes >= 50% of the online -> routeA gap.
  STABILITY: c1_frozen200 > c1_ema01 (median) iff the barrier is
  variance (not lag).

Run:  python pac_deploy4.py
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
from pac_deploy2 import rho1_of, setup

SEEDS = [0, 1, 2]
ARMS = ["c1_oracle", "c1_ema01", "c1_ema005", "c1_frozen200"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pac_deploy4")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")
PAC2_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy2", "summary.json")


def c1_phase(params, beta):
    return [np.exp(1j * np.angle(1.0 + np.conj(params["a"][l]) * beta[l]))
            for l in range(tcg.L)]


def train_arm(arm, seed):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    beta = [np.zeros(tcg.N, np.complex128) for _ in range(tcg.L)]
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    losses = []
    gamma = {"c1_ema01": 0.01, "c1_ema005": 0.005}.get(arm)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        if arm == "c1_oracle":
            w = c1_phase(params, [rho1_of(ql) for ql in q])
        elif gamma is not None:
            beta = [(1 - gamma) * b + gamma * rho1_of(ql)
                    for b, ql in zip(beta, q)]
            w = c1_phase(params, beta)
        elif arm == "c1_frozen200" and (step % 200 == 1):
            w = c1_phase(params, [rho1_of(ql) for ql in q])
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
        for arm in ARMS:
            fl, fin = train_arm(arm, seed)
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<13s} final {fl:.4f} finite {fin}",
                  flush=True)
    med = {a: float(np.median(v)) for a, v in table.items()}
    gap = ref["online"] - ref["routeA"]
    fracs = {a: (ref["online"] - med[a]) / gap for a in med}
    best = max(fracs, key=lambda a: fracs[a])
    p4 = fracs[best] >= 0.5
    stability_is_variance = med["c1_frozen200"] < med["c1_ema01"]
    print("-" * 70)
    print(f"medians { {k: round(v, 4) for k, v in med.items()} }  "
          f"comb-oracle ref {comb_med:.4f}")
    print(f"fracs { {k: round(v, 2) for k, v in fracs.items()} }")
    print(f"BAR P4 (best >= 50%): {'CAUSAL LAW HOLDS' if p4 else 'NO WIN'}")
    print(f"BAR STABILITY (frozen200 beats ema01): "
          f"{'barrier is VARIANCE' if stability_is_variance else 'barrier is LAG'}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, refs=ref, comb_med=comb_med, per_arm=table,
               medians=med, fracs=fracs, p4=bool(p4),
               stability_variance=bool(stability_is_variance))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
