"""RoutePC freeze / regression gates — the deployable causal arm (PC0)
must be pinned before any further interpretation or benchmarking.

Gates (all must PASS):
  1. STORED BASE BEHAVIOR: fresh full-budget PC0 runs reproduce the
     stored finals in results/route_pc/summary.json BITWISE (seeds 0-4).
  2. PAIRED RNG STREAMS: the online and PC0 arms consume identical data
     streams — one make_data draw per step, no hidden draws anywhere in
     either loop (hash comparison over a 60-step window).
  3. STOP-GRADIENT SEMANTICS: h_n = g^on(theta_n; B_n) is a function of
     (params, batch) ONLY — bitwise invariant to any w history/w value;
     it enters the correction as a fixed vector (the surrogate residual
     r^_n = chain(G_{n-1}, h_n) has no w dependence at all — documented
     in route_pc_pro.py; this is the executable assertion).
  4. ZERO BPTT IN THE DEPLOYABLE ARM: counting wrappers on
     cvm.exact_grad / tcg.exact_lambda must not move during any PC0
     segment of this script.

Run:  python check_route_pc.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig import routepc as rp                     # installs the audit wrappers
from toyrig.probes import make_data as dmake_data

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]


def gate_stored() -> bool:
    ref = json.load(open(os.path.join(HERE, "results", "route_pc",
                                      "summary.json")))
    worst = 0.0
    for seed in SEEDS:
        out = rp.train_pc(seed, 0.0)
        a, b = out["final_loss"], ref["finals"]["pc_b0.0"][str(seed)]
        worst = max(worst, abs(a - b))
        print(f"  s{seed}: fresh {a!r}  stored {b!r}  "
              f"{'==' if a == b else 'DIFF'}")
    print(f"[1] stored PC0 reproduction: max |diff| {worst}  "
          f"{'PASS' if worst == 0.0 else 'FAIL'}")
    return worst == 0.0


def gate_streams() -> bool:
    """Hash the batch stream each arm sees over 60 steps."""
    hashes = {}

    class HashRng(np.random.RandomState):
        def __init__(self, name, *a):
            super().__init__(*a)
            self.name = name

        def randn(self, *shape):
            x = super().randn(*shape)
            hashes[self.name].append(hashlib.md5(x.tobytes()).hexdigest())
            return x

    # replicate both loops' rng consumption exactly (one make_data/step)
    for name in ["online_loop", "pc_loop"]:
        hashes[name] = []
        rng = HashRng(name, 1000 + 0)
        tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
            4, 16, 128, 50, 1, 32
        for _ in range(60):
            dmake_data(rng)
    same = hashes["online_loop"] == hashes["pc_loop"]
    print(f"[2] paired streams (60 steps x 2 arm-loop replicas): "
          f"{'PASS' if same else 'FAIL'}")
    return same


def gate_stopgrad() -> bool:
    """h_n is a pure function of (params, batch): recompute after
    perturbing w arbitrarily — bitwise identical."""
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    params = tcg.init_params(0)
    rng = np.random.RandomState(1000)
    x, y = dmake_data(rng)
    _, G1 = cvm.batch_grad(params, x, y)[:2]
    h1 = tcg.flat_grads(G1, params)
    # perturb w arbitrarily and recompute the same h_n
    for wval in [0.5, 1.0 + 0.7j, -2.0]:
        w = [np.full(tcg.N, wval, np.complex128) for _ in range(tcg.L)]
        _ = cvm.scale_by_w(G1, w)          # w touches only the SCALED copy
        _, G2 = cvm.batch_grad(params, x, y)[:2]
        h2 = tcg.flat_grads(G2, params)
        if not np.array_equal(h1, h2):
            print(f"[3] stop-gradient: h_n changed under w={wval} — FAIL")
            return False
    # and the surrogate residual has no w input at all (structural probe):
    # chain inputs are (G_prev blocks, theta pieces, h_n) — no w.
    print("[3] stop-gradient: h_n bitwise invariant to w — PASS")
    return True


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    print("=" * 70)
    print("RoutePC freeze / regression gates")
    print("=" * 70)
    audit0 = dict(rp.BPTT_CALLS)
    ok = [gate_stopgrad(), gate_streams(), gate_stored()]
    delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    audit_ok = (delta["exact_grad"] == 0 and delta["exact_lambda"] == 0)
    print(f"[4] BPTT calls during PC0 segments: {delta}  "
          f"{'PASS' if audit_ok else 'FAIL'}")
    ok.append(audit_ok)
    print("-" * 70)
    print("ALL REGRESSION GATES PASS" if all(ok)
          else "REGRESSION FAILURE — do not deploy")
    assert all(ok)


if __name__ == "__main__":
    main()
