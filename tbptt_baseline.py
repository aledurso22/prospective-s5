"""Truncated-BPTT baseline — the publication-blocking control.

The reviewer question: does Route A's streaming O(1)-memory rule buy
anything over simply buffering W steps and running exact BPTT in the
buffer? Delayed copy D=50 predicts the shape of the answer: windows
W < 50 should fail (the credit never reaches the responsible input),
W >= 50 should work. Arms (paired seeds {0,1,2}, same protocol):

  online, tbptt W in {1, 4, 16, 64}, bptt (full)

tbptt credit: forward carried, backward truncated — exact adjoint
recursion inside each window with zero terminal lambda at the window
end (state detached between windows), gradients assembled with the
J-slot formula per window.

Route A reference is read from results/pac_deploy/summary.json if
present (identical protocol), else noted from the registered
co_variational_metric median.

No win/loss bar: this is a baseline characterization. Predeclared
reading: W=16 ~ online (fails), W=64 ~ bptt (works); the scientifically
interesting cell is routeA vs tbptt64 (streaming rule vs 64-step
buffer at matched task).

Run:  python tbptt_baseline.py
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
WINDOWS = [1, 4, 16, 64]
ARMS = ["online"] + [f"tbptt{w}" for w in WINDOWS] + ["bptt"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "tbptt_baseline")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def tbptt_lambda(params, q, W):
    """Exact adjoint within each length-W window, zero terminal lambda."""
    a, B = params["a"], params["b"]
    lam = [np.zeros((tcg.T, q[0].shape[1], tcg.N), np.complex128)
           for _ in range(tcg.L)]
    for t1 in range(tcg.T, 0, -W):
        t0 = max(0, t1 - W)
        lam_next = [np.zeros((q[0].shape[1], tcg.N), np.complex128)
                    for _ in range(tcg.L)]
        for t in range(t1 - 1, t0 - 1, -1):
            lam_next[tcg.L - 1] = (q[tcg.L - 1][t]
                                   + np.conj(a[tcg.L - 1])
                                   * lam_next[tcg.L - 1])
            for l in range(tcg.L - 2, -1, -1):
                up = np.einsum("jm,bj->bm", B[l + 1],
                               np.conj(lam_next[l + 1])).real
                lam_next[l] = up + np.conj(a[l]) * lam_next[l]
            for l in range(tcg.L):
                lam[l][t] = lam_next[l]
    return lam


def train_arm(arm, seed):
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
        if arm == "online":
            G_use = G
        elif arm == "bptt":
            lam = tcg.exact_lambda(params, q)
            Sa, Sb = tcg.sensitivities(params, h, x)
            G_use = tcg.assemble(params, h, x, r, lam, Sa, Sb,
                                 direct=True)
        else:
            W = int(arm[5:])
            lam = tbptt_lambda(params, q, W)
            Sa, Sb = tcg.sensitivities(params, h, x)
            G_use = tcg.assemble(params, h, x, r, lam, Sa, Sb,
                                 direct=True)
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
        for arm in ARMS:
            fl, fin = train_arm(arm, seed)
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<8s} final {fl:.4f} finite {fin}",
                  flush=True)
    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    routeA_med = None
    pac_summary = os.path.join(os.path.dirname(RESULTS_DIR),
                               "pac_deploy", "summary.json")
    if os.path.exists(pac_summary):
        with open(pac_summary) as f:
            routeA_med = json.load(f)["medians"].get("routeA")
    print(f"routeA reference median: {routeA_med}")
    print("predeclared reading: W=16 ~ online (fail), W=64 ~ bptt (work);"
          " key cell: routeA vs tbptt64")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(steps=STEPS, seeds=SEEDS,
                                    windows=WINDOWS),
               per_arm=table, medians=med, routeA_ref=routeA_med)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
