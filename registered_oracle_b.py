"""REGISTERED CONFIRMATION — oracle_B vs online (per-mode gains on B only).

Post-hoc observation in trained_credit_gains.py v2 (delayed copy, 3 seeds,
constrained modes): oracle_B median final loss 0.0096 vs online 0.0238 —
a 2.5x gap, stable everywhere. This experiment registers the test BEFORE
running it, on TWO tasks and FIVE seeds.

Mechanism under test: the online rule's per-mode credit defect is
mode-dependent and data-stable (spectral defect law). A per-mode complex
gain on the B-gradient channel re-weights the write-in strengths toward
their exact values WITHOUT touching mode placement — the a-channel is
where corrections destabilize (v1 explosion / v2 saturation), the
B-channel cannot move |a| at all.

Arms:
  online    e_t = q_t, exact per-module RTRL sensitivities (S-slot)
  oracle_B  per-mode gains w_j (least-squares fit against the exact
            gradient on ONE probe batch at init) applied to B-gradients
            only; a-gradient is the plain online one
  bptt      exact adjoint gradient (reference ceiling; not online)

Tasks:
  copy     delayed continuous copy: y_t = x_{t-50}, T=128, per-step loss
  adding   adding problem: x_t = (u_t, m_t), u~U(0,1), two marked
           positions, target = sum of marked u; final-step loss, T=96

Config: L=4, N=16, |a| init (0.90, 0.995) with a = sigmoid(rho) e^{i
theta}, batch 32, Adam lr 1e-3, clip 1.0, 1500 steps, seeds {0..4}.

REGISTERED BAR (fixed 2026-08-24, before any run):
  WIN iff median final loss of oracle_B <= 0.6 x online's on BOTH tasks
  AND every run is finite. Anything else closes the B-gain hypothesis.

Usage:
  python registered_oracle_b.py --smoke
  python registered_oracle_b.py --grid
  python registered_oracle_b.py --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg

SEEDS = [0, 1, 2, 3, 4]
TASKS = ["copy", "adding"]
ARMS = ["online", "oracle_B", "bptt"]
STEPS = 1500
CLIP = 1.0
LR = 1e-3
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "registered_oracle_b")


# ---------------------------------------------------------------------------
# Task data and residuals
# ---------------------------------------------------------------------------

def task_setup(task):
    if task == "copy":
        tcg.T, tcg.DELAY, tcg.M_IN = 128, 50, 1
    else:
        tcg.T, tcg.DELAY, tcg.M_IN = 96, 0, 2
    tcg.BATCH = 32


def make_data(task, rng, batch=32):
    if task == "copy":
        x = rng.randn(tcg.T, batch)
        y = np.concatenate([np.zeros((tcg.DELAY, batch)), x[:-tcg.DELAY]],
                           axis=0)
    else:
        u = rng.uniform(0, 1, (tcg.T, batch))
        m = np.zeros((tcg.T, batch))
        for b in range(batch):
            pos = rng.choice(tcg.T - 1, size=2, replace=False)
            m[pos, b] = 1.0
        x = np.stack([u, m], axis=-1)                     # (T, B, 2)
        y = (u * m).sum(axis=0)                           # (B,)
    return x, y


def task_residual(task, yhat, y):
    r = np.zeros_like(yhat)
    if task == "copy":
        r = yhat - y
        r[:tcg.DELAY] = 0.0
    else:
        r[tcg.T - 1] = yhat[tcg.T - 1] - y
    return r


def task_loss(task, r):
    if task == "copy":
        return 0.5 * float(np.mean(r ** 2))
    return 0.5 * float(np.mean(r[tcg.T - 1] ** 2))


# ---------------------------------------------------------------------------
# Training (arms share everything but the error signal / pairing)
# ---------------------------------------------------------------------------

def fit_w(params, task, rng):
    x, y = make_data(task, rng)
    h, yhat = tcg.forward(params, x)
    r = task_residual(task, yhat, y)
    return tcg.fit_gains(params, x, r)


def train_arm(task, arm, seed):
    task_setup(task)
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    probe_rng = np.random.RandomState(77)
    w = fit_w(params, task, probe_rng) if arm == "oracle_B" else None

    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_data(task, rng)
        h, yhat = tcg.forward(params, x)
        r = task_residual(task, yhat, y)
        losses.append(task_loss(task, r))
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        if arm == "bptt":
            err = tcg.exact_lambda(params, q)
            G = tcg.assemble(params, h, x, r, err, Sa, Sb, direct=True)
        elif arm == "oracle_B":
            G_on = tcg.assemble(params, h, x, r, q, Sa, Sb)
            err_w = [q[l] * w[l][None, None, :] for l in range(tcg.L)]
            G_w = tcg.assemble(params, h, x, r, err_w, Sa, Sb)
            G = dict(a=G_on["a"], b=G_w["b"], c=G_on["c"])
        else:
            G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if nrm > CLIP:
            g = g * (CLIP / nrm)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - LR * (m / (1 - b1 ** step)) / (
            np.sqrt(v / (1 - b2 ** step)) + eps)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return dict(arm=arm, task=task, seed=seed,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                wall_time_sec=time.time() - t0)


# ---------------------------------------------------------------------------
# Grid + registered evaluation
# ---------------------------------------------------------------------------

def _path(task, arm, seed):
    return os.path.join(RESULTS_DIR, f"{arm}_{task}_s{seed}.json")


def run_grid():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for task in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                out = train_arm(task, arm, seed)
                with open(_path(task, arm, seed), "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[done] {arm:<9s} {task:<7s} s{seed}  "
                      f"final {out['final_loss']:.4f}  finite {out['finite']}",
                      flush=True)


def summarize():
    print("=" * 78)
    print("REGISTERED CONFIRMATION — evaluation")
    print("=" * 78)
    wins = 0
    for task in TASKS:
        meds = {}
        for arm in ARMS:
            finals = []
            fins = []
            for seed in SEEDS:
                with open(_path(task, arm, seed)) as f:
                    out = json.load(f)
                finals.append(out["final_loss"])
                fins.append(out["finite"])
            meds[arm] = float(np.median(finals))
            print(f"  {task:<7s} {arm:<9s}: "
                  f"{['%.4f' % x for x in finals]}  median {meds[arm]:.4f}  "
                  f"all-finite {all(fins)}")
        ratio = meds["oracle_B"] / meds["online"]
        ok = ratio <= 0.6
        wins += int(ok)
        print(f"  {task}: oracle_B/online = {ratio:.3f} "
              f"(need <= 0.60)  {'WIN' if ok else 'NO WIN'}")
    allfinite = all(json.load(open(_path(t, a, s)))["finite"]
                    for t in TASKS for a in ARMS for s in SEEDS)
    print("-" * 78)
    print("VERDICT: " + ("WIN — the B-gain hypothesis holds"
                           if (wins == len(TASKS) and allfinite) else
                           "NO WIN — the B-gain hypothesis closes"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        global STEPS
        STEPS = 60
        print("SMOKE (not part of the registered grid)")
        for task in TASKS:
            for arm in ARMS:
                out = train_arm(task, arm, 0)
                print(f"  {arm:<9s} {task:<7s} final {out['final_loss']:.4f}"
                      f"  finite {out['finite']}")
        return
    if args.grid:
        run_grid()
        return
    if args.summarize:
        summarize()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
