"""PAC analyses A & B (directive 03) — run AFTER preregistration commit
50879e3 (results/pac_analysis/PREDICTIONS.md).

A (pairing test): does the resolvent combination -arg(1 - conj(a) rho(1))
beat its factors arg rho(1) and arg a separately, weighted by E|q|^2,
in layers where the learned phase is replicable (R_w > 0.5)?

B (horizon test): R(c(H), arg w) over H in {1,2,4,8,16,32,T-1},
c(H) = sum_{k=0}^{H} conj(a)^k rho(k). Monotone decline from H=1 is the
one-step-horizon signature; rise-then-fall is bias-variance.

Run:  python pac_analysis.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
from depth_law import train_cell
from decompose_w_final import make_data
from pac_probe2 import autocorr, circ_R

SEEDS = [0, 1, 2]
H_GRID = [1, 2, 4, 8, 16, 32, None]          # None = full (k up to T-1)
R_W = {0: 0.649, 1: 0.380, 2: 0.630, 3: 0.995}   # pac_probe2 ceilings
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pac_analysis")
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "factorize_w")


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    A, B = {}, {}
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + probe...", flush=True)
        params, w = train_cell(4, 50, seed)
        w_saved = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        det = max(float(np.max(np.abs(w[l] - w_saved[l])))
                  for l in range(tcg.L))
        assert det < 1e-12, f"determinism gate failed: {det}"
        rng = np.random.RandomState(900 + seed)
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        for l in range(tcg.L):
            rho, denom = autocorr(q[l], tcg.T)
            a = params["a"][l]
            aw = np.angle(w[l])
            comb = -np.angle(1.0 - np.conj(a) * rho[1])
            s_rho = np.angle(rho[1])
            s_a = np.angle(a)
            A.setdefault(l, {})[seed] = dict(
                comb=circ_R(comb - aw, denom),
                s_rho=circ_R(s_rho - aw, denom),
                s_a=circ_R(s_a - aw, denom))
            ak = np.conj(a)
            cH = {}
            for H in H_GRID:
                hmax = tcg.T - 1 if H is None else H
                c = np.sum(np.stack([ak ** k for k in range(hmax + 1)])
                           * rho[:hmax + 1], axis=0)
                cH["full" if H is None else H] = circ_R(np.angle(c) - aw,
                                                        denom)
            B.setdefault(l, {})[seed] = cH
            print(f"  L{l}: A comb {A[l][seed]['comb']:.3f} vs rho "
                  f"{A[l][seed]['s_rho']:.3f} vs a {A[l][seed]['s_a']:.3f}   "
                  f"B H1 {cH[1]:.3f} H4 {cH[4]:.3f} H16 {cH[16]:.3f} "
                  f"full {cH['full']:.3f}", flush=True)

    print("-" * 70)
    # verdicts per preregistration
    layers_A_ok = []
    for l in range(tcg.L):
        if R_W[l] <= 0.5:
            print(f"  L{l}: reliability {R_W[l]:.3f} <= 0.5, A test skips")
            continue
        med = {k: float(np.median([A[l][s][k] for s in SEEDS]))
               for k in ("comb", "s_rho", "s_a")}
        ok = med["comb"] > med["s_rho"] and med["comb"] > med["s_a"]
        layers_A_ok.append(ok)
        print(f"  L{l} (R_w {R_W[l]:.3f}): comb {med['comb']:.3f} vs "
              f"rho {med['s_rho']:.3f} vs a {med['s_a']:.3f}  "
              f"{'beats both' if ok else 'does NOT beat both'}")
    A_pass = bool(layers_A_ok) and all(layers_A_ok)

    mono_layers = 0
    rise_layers = 0
    for l in range(tcg.L):
        meds = [float(np.median([B[l][s][h] for s in SEEDS]))
                for h in [1, 2, 4, 8, 16, 32, "full"]]
        diffs = np.diff(meds)
        if np.all(diffs <= 0.02):
            mono_layers += 1
        if np.max(meds[1:]) - meds[0] > 0.05:
            rise_layers += 1
        print(f"  L{l}: R by H = {['%.3f' % m for m in meds]}")
    B_pass = mono_layers >= 3 and rise_layers < 2

    print(f"A pairing test: {'PASS' if A_pass else 'FAIL -> drop resolvent framing'}")
    print(f"B horizon test: {'PASS (monotone decline, one-step horizon)' if B_pass else 'FAIL'}"
          f"  (mono {mono_layers}/4, rise {rise_layers}/4)")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, prereg="50879e3", A=A, B=B,
               verdicts=dict(A=bool(A_pass), B=bool(B_pass),
                             mono_layers=mono_layers,
                             rise_layers=rise_layers))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2, default=str)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
