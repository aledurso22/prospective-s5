"""bench_smnist — the headroom gate on a recognized benchmark (CPU scale).

Sequential MNIST (downsampled 14x14 -> T=196) on the program's
LRU-family diagonal complex module (tcg), with a dense mean-pooled
readout + linear 10-class head. Mean pooling distributes the loss
densely over time, so the online rule (S-slot: RTRL sensitivities +
instantaneous spatial error, the online_full structure) is causal and
well-defined.

Per-step per-mode top-layer credit: pooled = mean_t Re(h^L_t), so
dL/dRe(h^L_t) = (dlog @ W)/T at every t — the dense error. Layer-0
credit via the rig's spatial routing. The exact arm applies the
adjoint (exact_lambda) to the same q. W-head gradient is exact
instantaneous for both arms.

GATE (fixed before running): headroom h = (L_online - L_bptt)/L_online
>= 0.2 on mean train loss over the last 50 of 800 steps. If h < 0.2 the
cell is capacity-limited and no benchmark claim is made (the D6 lesson).

Paired: same init, same W init, same batch stream, plain SGD (no Adam)
for the gate's cleanest credit comparison.

Run:  python bench_smnist.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from toyrig import ssm_rig as tcg
from train import load_mnist

SEED = 0
STEPS = 800
SUBSET = 5000
N_CLASSES = 10
LR = 1e-3
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "bench_smnist")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.M_IN, tcg.BATCH = 2, 32, 196, 1, 32


def loss_and_err(params, W, x, y):
    """Cross-entropy on the mean-pooled readout; returns loss, states,
    and dpooled = dL/d(pooled) (B, N)."""
    h, _ = tcg.forward(params, x)
    pooled = h[-1].real.mean(axis=0)              # (B, N)
    logits = pooled @ W.T                         # (B, 10)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    B = x.shape[1]
    loss = float(-np.log(p[np.arange(B), y] + 1e-300).mean())
    dlog = p
    dlog[np.arange(B), y] -= 1.0
    dlog /= B
    return loss, h, dlog @ W, pooled, dlog


def q_of(params, dpooled):
    """Per-step per-mode credit: dL/dRe(h^L_t) = dpooled/T every step.
    q_top complex (real values); q_0 via the rig's spatial routing."""
    T, B = tcg.T, tcg.BATCH
    q1 = np.broadcast_to((dpooled / T)[None, :, :],
                         (T, B, tcg.N)).copy().astype(np.complex128)
    q0 = np.einsum("jm,tbj->tbm", params["b"][1],
                   np.conj(q1)).real.astype(np.complex128)
    return [q0, q1]


def batch_grad(params, W, x, y, exact=False):
    loss, h, dpooled, pooled, dlog = loss_and_err(params, W, x, y)
    q = q_of(params, dpooled)
    Sa, Sb = tcg.sensitivities(params, h, x)
    if exact:
        err = tcg.exact_lambda(params, q)
        G = tcg.assemble(params, h, x, np.zeros((tcg.T, tcg.BATCH)),
                         err, Sa, Sb, direct=True)
    else:
        G = tcg.assemble(params, h, x, np.zeros((tcg.T, tcg.BATCH)),
                         q, Sa, Sb)
    GW = dlog.T @ pooled                            # (10, N)
    return loss, G, GW


def evaluate(params, W, x, y):
    errs = []
    for i in range(0, len(x), 256):
        xb = x[i:i + 256].T
        h, _ = tcg.forward(params, xb)
        logits = h[-1].real.mean(axis=0) @ W.T
        errs.append((logits.argmax(axis=1) != y[i:i + 256]).mean())
    return float(np.mean(errs))


def train_arm(arm, x_train, y_train):
    params = tcg.init_params(SEED)
    W = np.random.RandomState(100 + SEED).randn(N_CLASSES, tcg.N) \
        / np.sqrt(tcg.N)
    rng = np.random.RandomState(1000 + SEED)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bi = rng.randint(0, SUBSET, tcg.BATCH)
        x = x_train[bi].T.copy()
        y = y_train[bi]
        loss, G, GW = batch_grad(params, W, x, y, exact=(arm == "bptt"))
        losses.append(loss)
        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if nrm > 1.0:
            g = g * (1.0 / nrm)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        flat = flat - LR * (m / (1 - 0.9 ** step)) / (
            np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        mW = 0.9 * mW + 0.1 * GW
        vW = 0.999 * vW + 0.001 * GW ** 2
        W = W - LR * (mW / (1 - 0.9 ** step)) / (
            np.sqrt(vW / (1 - 0.999 ** step)) + 1e-8)
        params = tcg.pack(params, flat)
        if step % 200 == 0:
            print(f"    {arm} step {step}: loss {loss:.4f}", flush=True)
    return dict(train_loss=float(np.mean(losses[-50:])),
                wall=time.time() - t0,
                params=params, W=W)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist(downsample=2)
    rng_np = np.random.RandomState(SEED)
    idx = rng_np.choice(len(x_train), SUBSET, replace=False)
    x_train, y_train = x_train[idx], y_train[idx]
    idx = rng_np.choice(len(x_test), 1000, replace=False)
    x_ev, y_ev = x_test[idx], y_test[idx]
    print(f"sMNIST gate: {SUBSET} train, T={tcg.T}, N={tcg.N}, L={tcg.L}",
          flush=True)

    results = {}
    for arm in ["online", "bptt"]:
        out = train_arm(arm, x_train, y_train)
        err = evaluate(out["params"], out["W"], x_ev, y_ev)
        results[arm] = dict(train_loss=out["train_loss"], test_err=err,
                            wall=out["wall"])
        print(f"  {arm}: train_loss {out['train_loss']:.4f}  "
              f"test_err {err:.4f}  ({out['wall']:.0f}s)", flush=True)

    h = (results["online"]["train_loss"] - results["bptt"]["train_loss"]) \
        / max(results["online"]["train_loss"], 1e-300)
    print("-" * 70)
    print(f"GATE: h = {h:.2f} (need >= 0.2)  ->  "
          f"{'HEADROOM EXISTS' if h >= 0.2 else 'NO HEADROOM — do not benchmark'}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, results=results, headroom=h,
                       config=dict(steps=STEPS, subset=SUBSET, N=tcg.N,
                                   L=tcg.L, T=tcg.T, lr=LR)),
                  f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
