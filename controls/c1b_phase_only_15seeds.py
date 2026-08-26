"""C1b — phase-only RoutePC, 15-seed extension (addendum C1, part 2).

Triggered by the registered C1 decision rule: pcPhase was COMPETITIVE on
seeds 0..4 (median 0.0085 <= 1.5x PC0 0.0110, beats online 4/5). This
script runs the NEW seeds 5..14 for all three arms (online via
cvm.train_route; PC0 replay and pcPhase via controls.c1's train_pc),
then merges with the stored seeds 0..4 finals (route_pc summary for
online/PC0, c1 summary for pcPhase) for the 15-seed failure-rate
estimate.

REGISTERED FAILURE DEFINITION (fixed before running): a seed is an arm
failure iff L_arm/L_online > 1 on that paired seed (arm loses to the
deployable baseline on its own stream). Report failure counts over 15
seeds, median paired ratios, and the per-seed table; seed 3 remains a
genuine outcome.

Run:  python -m controls.c1b_phase_only_15seeds
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import route_a as cvm
from toyrig import routepc as rp
from controls.c1_phase_only_routepc import train_pc, setup

NEW_SEEDS = list(range(5, 15))
ALL_SEEDS = list(range(15))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "c1_phase_only_routepc")


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fO, fC, fP = {}, {}, {}
    audit0 = dict(rp.BPTT_CALLS)
    for seed in NEW_SEEDS:
        out = cvm.train_route("online", seed)
        fO[seed] = out["final_loss"]
        print(f"online s{seed}: final {fO[seed]:.4f}", flush=True)
        fC[seed] = train_pc(seed, phase_only=False)["final_loss"]
        print(f"PC0    s{seed}: final {fC[seed]:.4f}", flush=True)
        fP[seed] = train_pc(seed, phase_only=True)["final_loss"]
        print(f"pcPhase s{seed}: final {fP[seed]:.4f}", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                      "summary.json")))
    c1 = json.load(open(os.path.join(RESULTS_DIR, "summary.json")))
    for s in range(5):
        fO[s] = ref["finals"]["online"][str(s)]
        fC[s] = ref["finals"]["pc_b0.0"][str(s)]
        fP[s] = c1["finals"]["pcphase"][str(s)]

    med = lambda f: float(np.median([f[s] for s in ALL_SEEDS]))
    print("-" * 78)
    hdr = "".join(f"{s:>8d}" for s in ALL_SEEDS)
    print(f"{'arm':<10s}{hdr}{'median':>9s}")
    for name, f in [("online", fO), ("PC0", fC), ("pcPhase", fP)]:
        print(f"{name:<10s}" + "".join(f"{f[s]:>8.3f}" for s in ALL_SEEDS)
              + f"{med(f):>9.4f}")
    rat_c = {s: fC[s] / fO[s] for s in ALL_SEEDS}
    rat_p = {s: fP[s] / fO[s] for s in ALL_SEEDS}
    rat_pc = {s: fP[s] / fC[s] for s in ALL_SEEDS}
    print(f"PC0/online ratios    : {['%.2f' % rat_c[s] for s in ALL_SEEDS]}"
          f"  median {np.median(list(rat_c.values())):.3f}")
    print(f"pcPhase/online ratios: {['%.2f' % rat_p[s] for s in ALL_SEEDS]}"
          f"  median {np.median(list(rat_p.values())):.3f}")
    print(f"pcPhase/PC0 ratios   : {['%.2f' % rat_pc[s] for s in ALL_SEEDS]}"
          f"  median {np.median(list(rat_pc.values())):.3f}")
    fail_c = [s for s in ALL_SEEDS if rat_c[s] > 1.0]
    fail_p = [s for s in ALL_SEEDS if rat_p[s] > 1.0]
    print(f"FAILURES (ratio > 1): PC0 {len(fail_c)}/15 seeds {fail_c}  |  "
          f"pcPhase {len(fail_p)}/15 seeds {fail_p}")
    beats_c = sum(fP[s] < fC[s] for s in ALL_SEEDS)
    beats_o = sum(fP[s] < fO[s] for s in ALL_SEEDS)
    print(f"pcPhase beats PC0 on {beats_c}/15 paired seeds; beats online "
          f"on {beats_o}/15")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(new_seeds=NEW_SEEDS,
                           failure_rule="L_arm/L_online > 1 on the seed"),
               finals=dict(online={str(s): fO[s] for s in ALL_SEEDS},
                           pc0={str(s): fC[s] for s in ALL_SEEDS},
                           pcphase={str(s): fP[s] for s in ALL_SEEDS}),
               medians=dict(online=med(fO), pc0=med(fC), pcphase=med(fP)),
               paired_ratios=dict(
                   pc0_over_online={str(s): rat_c[s] for s in ALL_SEEDS},
                   pcphase_over_online={str(s): rat_p[s]
                                        for s in ALL_SEEDS},
                   pcphase_over_pc0={str(s): rat_pc[s]
                                     for s in ALL_SEEDS}),
               failures=dict(pc0=fail_c, pcphase=fail_p),
               beats=dict(pcphase_vs_pc0=beats_c,
                          pcphase_vs_online=beats_o),
               bptt_calls=audit)
    with open(os.path.join(RESULTS_DIR, "summary_15seeds.json"),
              "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary_15seeds.json")


if __name__ == "__main__":
    main()
