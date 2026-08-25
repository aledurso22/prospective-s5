"""Matched-function-class control — horizon vs estimation rate, in
routeA's exact function class.

Correction to orient_wiener: D2 deployed a noisy PER-TIMESTEP phase
(destructive), while routeA deploys a CONSTANT per-mode rotation — and
frozen learned phase already preserves ~86% of routeA's improvement, so
live adaptation is at most the last ~14%. To separate horizon from
estimator granularity, convert each Wiener K-filter into a single
constant per-mode rotation:

  w^K_j = exp(i arg c^K_j),  c^K_j = E[lambda_hat^K conj(q)]/E|q|^2,
  lambda_hat^K = (K-filter) * q   (estimated with an exact-credit teacher)

Deployed exactly like routeA's w (scale_by_w). Arms (paired seeds
{0,1,2}):
  frozenK, K in {1,4,16,32,64,96}   — estimated once at trained params
  refreshK, K in {1,64}             — teacher every 200 steps
  perbatchK1                        — teacher every step (K=1 only;
                                      per-step Wiener fit at K=64 is
                                      routeA-cost anyway)

Refs: online/routeA from pac_deploy/summary.json; frozen learned phase
0.0053 (factorize_w).

REGISTERED BARS (fixed before running):
  HORIZON: frozenK64 <= 0.5 x frozenK1 (median) => long-history
  orientation adds value in the matched class. (If not, horizon does
  not matter in this function class.)
  ANCHOR: frozenK64 vs 0.0053 (frozen learned phase) — is the derived
  filter phase as good as the learned one?
  RATE: perbatchK1 vs refreshK1 vs frozenK1 ladder — the adaptation-rate
  component at fixed function class.

Run:  python matched_phase.py
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
from wiener_oracle import wiener_fit
from orient_wiener import apply_filter

SEEDS = [0, 1, 2]
K_FROZEN = [1, 4, 16, 32, 64, 96]
K_RATE = [1, 64]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "matched_phase")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")


def phase_of_filter(q, lam, K):
    """Per-mode constant rotation from the K-filtered credit."""
    w = []
    for l in range(tcg.L):
        lh = apply_filter(q[l], wiener_fit(q[l], lam[l], K)[0])
        num = np.mean(lh * np.conj(q[l]), axis=(0, 1))
        den = np.mean(np.abs(q[l]) ** 2, axis=(0, 1)) + 1e-300
        w.append(np.exp(1j * np.angle(num / den)))
    return w


def deploy(arm, seed, w_frozen):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    w = w_frozen
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        if arm.startswith("refresh") and step % 200 == 1:
            lam = tcg.exact_lambda(params, q)
            K = int(arm[7:])
            w = phase_of_filter(q, lam, K)
        elif arm.startswith("perbatch"):
            lam = tcg.exact_lambda(params, q)
            w = phase_of_filter(q, lam, 1)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses)))


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    table = {}
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + filter phases...", flush=True)
        params, _ = train_cell(4, 50, seed)
        rng = np.random.RandomState(900 + seed)
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        lam = tcg.exact_lambda(params, q)
        frozen = {K: phase_of_filter(q, lam, K) for K in set(K_FROZEN)}
        arms = [f"frozen{K}" for K in K_FROZEN]
        for arm in arms:
            fl, fin = deploy(arm, seed, frozen[int(arm[6:])])
            table.setdefault(arm, []).append(fl)
            print(f"  {arm:<10s} final {fl:.4f} finite {fin}", flush=True)
        for K in K_RATE:
            fl, fin = deploy(f"refresh{K}", seed, None)
            table.setdefault(f"refresh{K}", []).append(fl)
            print(f"  refresh{K}  final {fl:.4f} finite {fin}", flush=True)
        fl, fin = deploy("perbatch1", seed, None)
        table.setdefault("perbatch1", []).append(fl)
        print(f"  perbatch1  final {fl:.4f} finite {fin}", flush=True)

    med = {a: float(np.median(v)) for a, v in table.items()}
    gap = ref["online"] - ref["routeA"]
    fracs = {a: (ref["online"] - med[a]) / gap for a in med}
    horizon_win = med["frozen64"] <= 0.5 * med["frozen1"]
    print("-" * 70)
    print(f"refs: online {ref['online']:.4f}  routeA {ref['routeA']:.4f}  "
          f"frozen-learned-phase 0.0053")
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"fracs: { {k: round(v, 2) for k, v in fracs.items()} }")
    print(f"BAR HORIZON (frozen64 <= 0.5x frozen1): "
          f"{'HORIZON ADDS VALUE' if horizon_win else 'NO'}")
    print(f"BAR ANCHOR: frozen64 {med['frozen64']:.4f} vs frozen-learned "
          f"0.0053 -> {'DERIVED MATCHES LEARNED' if med['frozen64'] <= 2 * 0.0053 else 'learned phase still better'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, refs=ref, per_arm=table, medians=med,
               fracs=fracs, horizon_win=bool(horizon_win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
