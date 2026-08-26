"""bench_copy — the copy-task headroom gate and first benchmark cell.

The copy task from the prior online-learning suite (Zucchet et al.
NeurIPS 2023 family): ENCODE phase (20 steps of white signal), MEMORY
gap of D steps (zeros), READOUT phase (20 steps: reproduce the encoded
signal, dense per-step loss only in the readout window). Genuine long
memory + dense causal credit — the regime where online algorithms are
meant to compete with BPTT.

Rig: the program's LRU-family diagonal complex module (tcg), L=4, N=16 —
online arm = S-slot (RTRL sensitivities + instantaneous spatial error,
the online_full structure), exact arm = BPTT adjoint. Arms:
  online, bptt        (gate: headroom h = (L_online - L_bptt)/L_online)
  routeA, frozenPhase, scalarOnly   (if gate passes)

routeA/frozenPhase: per-(layer, mode) complex w learned by the routeA
meta-gradient (frozenPhase deploys arg w frozen from a fresh run —
strictly causal deployment). scalarOnly: real w (the magnitude-only
ablation — real counterpart of the complex arm).

PRE-REGISTERED (fixed before running):
  GATE: h >= 0.3 at the chosen delay (else the cell is capacity-limited
  and no mechanism arm runs — the D6 lesson).
  BAR A (online improvement): median routeA < median online on all
  seeds (paired).
  BAR B (orientation specificity): median routeA < median scalarOnly.
  BAR C (deployment): median frozenPhase retains >= 50% of routeA's
  live gap-closure.

Run:  python bench_copy.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.probes import make_data as _mk  # unused; task below

SEEDS = [0, 1]
DELAYS = [50, 100]
T_ENC, T_OUT = 20, 20
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "bench_copy")


def setup(delay):
    tcg.L, tcg.N = 4, 16
    tcg.T = T_ENC + delay + T_OUT
    tcg.DELAY = T_ENC + delay        # readout starts here
    tcg.M_IN, tcg.BATCH = 1, 32


def make_batch(rng):
    """Encode phase signal + gap; loss only in the readout window."""
    T = tcg.T
    x = np.zeros((T, tcg.BATCH))
    x[:T_ENC] = rng.randn(T_ENC, tcg.BATCH)
    y = np.zeros((T, tcg.BATCH))
    y[tcg.DELAY:tcg.DELAY + T_OUT] = x[:T_OUT]
    return x, y


def batch_loss_grad(params, x, y):
    """Per-step loss restricted to the readout window."""
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    r[tcg.DELAY + T_OUT:] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    G = tcg.assemble(params, h, x, r, q, Sa, Sb)
    loss = 0.5 * float(np.mean(r ** 2))
    return loss, G, q, r, h


def exact_grad(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    r[tcg.DELAY + T_OUT:] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    err = tcg.exact_lambda(params, q)
    return tcg.assemble(params, h, x, r, err, Sa, Sb, direct=True)


def eval_loss(params, seed, n=8):
    rng = np.random.RandomState(5000 + seed)
    tot = 0.0
    for _ in range(n):
        x, y = make_batch(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        tot += 0.5 * float(np.mean(r ** 2))
    return tot / n


def train_arm(arm, seed, w_frozen=None, steps=1500):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)] \
        if w_frozen is None else [wl.copy() for wl in w_frozen]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        x, y = make_batch(rng)
        loss, G, q, r, h = batch_loss_grad(params, x, y)
        losses.append(loss)
        if arm == "bptt":
            g = tcg.flat_grads(exact_grad(params, x, y), params)
        elif arm == "scalarOnly":
            Gw = cvm.scale_by_w(G, [wl.real.astype(np.complex128)
                                    for wl in w])
            g = tcg.flat_grads(Gw, params)
        else:
            Gw = cvm.scale_by_w(G, w)
            g = tcg.flat_grads(Gw, params)
        g = cvm.clip(g)
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
        if arm == "routeA":
            G_next = exact_grad(params_next, x, y)
            gN = tcg.flat_grads(G_next, params_next)
            off = 0
            for l in range(tcg.L):
                th = params["theta"][l]
                u_mode = tcg.sig(params["rho"][l])
                sigp = u_mode * (1 - u_mode)
                A = G["a"][l] * np.exp(1j * th)
                Gb = G["b"][l]
                M_ = Gb.shape[1]
                gN_rho = gN[off:off + tcg.N]
                gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
                gN_bre = gN[off + 2 * tcg.N:
                            off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
                gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                            off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        params = params_next
    wall = time.time() - t0
    return dict(final=eval_loss(params, seed),
                finite=bool(np.all(np.isfinite(losses))),
                w=[wl.copy() for wl in w], wall=wall)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    chosen = None
    print("HEADROOM gate (online vs bptt):", flush=True)
    for D in DELAYS:
        setup(D)
        on = np.median([train_arm("online", s, steps=800)["final"]
                        for s in [0]])
        bp = np.median([train_arm("bptt", s, steps=800)["final"]
                        for s in [0]])
        h = (on - bp) / max(on, 1e-300)
        print(f"  D={D}: online {on:.4f}  bptt {bp:.4f}  headroom {h:.2f}",
              flush=True)
        if h >= 0.3:
            chosen = D
    if chosen is None:
        print("NO HEADROOM >= 0.3 at any delay — stop (capacity-limited)")
        return
    print(f"chosen delay D={chosen}; full arms, seeds {SEEDS}")
    setup(chosen)
    arms = ["online", "bptt", "routeA", "scalarOnly"]
    for arm in arms:
        for seed in SEEDS:
            out = train_arm(arm, seed)
            results[f"{arm}/s{seed}"] = dict(final=out["final"],
                                             finite=out["finite"],
                                             wall=out["wall"])
            if arm == "routeA":
                np.save(os.path.join(RESULTS_DIR, f"w_routeA_s{seed}.npy"),
                        np.array(out["w"]))
            print(f"  {arm:<10s} s{seed} final {out['final']:.4f} "
                  f"finite {out['finite']} ({out['wall']:.0f}s)", flush=True)
    # frozen phase deployment (from routeA's w, fresh trajectory)
    for seed in SEEDS:
        w_full = list(np.load(os.path.join(RESULTS_DIR,
                                           f"w_routeA_s{seed}.npy"),
                              allow_pickle=True))
        w_ph = [np.exp(1j * np.angle(wl)) for wl in w_full]
        out = train_arm("frozenPhase", seed, w_frozen=w_ph)
        results[f"frozenPhase/s{seed}"] = dict(final=out["final"],
                                               finite=out["finite"],
                                               wall=out["wall"])
        print(f"  frozenPhase s{seed} final {out['final']:.4f}", flush=True)

    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final"]
                                 for s in SEEDS]))
           for arm in ["online", "bptt", "routeA", "scalarOnly",
                       "frozenPhase"]}
    gap = med["online"] - med["bptt"]
    rg = {arm: (med["online"] - med[arm]) / gap
          for arm in ["routeA", "scalarOnly", "frozenPhase"]}
    barA = all(results[f"routeA/s{s}"]["final"]
               < results[f"online/s{s}"]["final"] for s in SEEDS)
    barB = med["routeA"] < med["scalarOnly"]
    barC = rg["frozenPhase"] >= 0.5 * rg["routeA"]
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"R_gap: { {k: round(v, 2) for k, v in rg.items()} }")
    print(f"BAR A (routeA < online all seeds): {'PASS' if barA else 'FAIL'}")
    print(f"BAR B (routeA < scalarOnly): {'PASS' if barB else 'FAIL'}")
    print(f"BAR C (frozenPhase >= 50% of routeA's closure): "
          f"{'PASS' if barC else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, delay=chosen, seeds=SEEDS, results=results,
               medians=med, R_gap=rg,
               bars=dict(A=bool(barA), B=bool(barB), C=bool(barC)))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
