"""Benchmark aggregator — headroom gates + registered bars over
results/bench/metrics_{task}_{arm}*_s{seed}.json.

Usage:
  python bench_report.py --gate     # per-(task, seed) headroom h; writes
                                    # results/bench/gate.json; the grid's
                                    # mechanism arms run only where PASS
  python bench_report.py            # full table + registered bars A/B/C

Headroom (per task, seed):   h = (L_online - L_baseline) / L_online,
PASS iff h >= 0.2 (registered; the D6 lesson — no headroom, no mechanism
test).

Registered bars (per task, over paired seeds):
  A (online improvement):  routeA beats online on every paired seed.
  B (orientation specificity): median R_gap(routeA) > median
     R_gap(scalarLive), per-seed paired.
  C (deployment): median closure retained by frozenPhase
     (L_online - L_frozenPhase) / (L_online - L_routeA) >= 0.5.
  routePC is reported as the exploratory causal supplement (vs routeA):
     R_gap measured against the SAME seed's baseline/online.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "bench")
GATE_H = 0.2

ARMS = ["baseline", "online", "tbptt", "routeA", "scalarLive", "routePC",
        "routePCreal", "frozenPhase", "frozenMag"]


def collect():
    """{task: {arm: {seed: final_test_loss}}} (+ acc + paths)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              "metrics_*.json"))):
        with open(path) as f:
            m = json.load(f)
        cfg = m["config"]
        task, arm, seed = cfg["task"], cfg["arm"], cfg["seed"]
        out.setdefault(task, {}).setdefault(arm, {})[seed] = dict(
            loss=m["final_test_loss"], acc=m["final_test_acc"],
            wall=m["wall_time_sec"], steps_per_sec=m["steps_per_sec"],
            path=path)
    return out


def gate(data):
    gates = {}
    print("headroom gate (h = (L_online - L_baseline)/L_online, "
          f"bar {GATE_H}):")
    for task, arms in data.items():
        for seed in sorted(arms.get("online", {})):
            if seed not in arms.get("baseline", {}):
                print(f"  {task} s{seed}: baseline missing — SKIP")
                continue
            lo = arms["online"][seed]["loss"]
            lb = arms["baseline"][seed]["loss"]
            h = (lo - lb) / lo
            ok = h >= GATE_H
            gates.setdefault(task, {})[seed] = dict(
                h=h, passed=bool(ok), l_online=lo, l_baseline=lb)
            print(f"  {task} s{seed}: online {lo:.4f}  baseline {lb:.4f}  "
                  f"h {h:.3f}  -> {'PASS' if ok else 'FAIL'}")
    with open(os.path.join(RESULTS_DIR, "gate.json"), "w") as f:
        json.dump(gates, f, indent=2)
    print(f"wrote {os.path.join(RESULTS_DIR, 'gate.json')}")


def report(data):
    for task, arms in data.items():
        print("=" * 78)
        print(f"task {task}")
        seeds = sorted(arms.get("online", {}))
        print(f"{'arm':<12s} {'per-seed final loss':>34s}  {'median':>8s}")
        for arm in ARMS:
            if arm not in arms:
                continue
            have = [(s, arms[arm][s]["loss"]) for s in seeds
                    if s in arms[arm]]
            if not have:
                continue
            per = " ".join(f"s{s}:{v:.4f}" for s, v in have)
            print(f"{arm:<12s} {per:>34s}  "
                  f"{np.median([v for _, v in have]):8.4f}")
        if "online" not in arms or "baseline" not in arms:
            print("  (gate arms incomplete — run baseline+online first)")
            continue
        # R_gap per paired seed
        print("R_gap = (L_online - L_arm)/(L_online - L_baseline), paired:")
        rgap = {}
        for arm in ARMS:
            if arm in ("online", "baseline") or arm not in arms:
                continue
            rs = {}
            for s in seeds:
                if s in arms[arm] and s in arms["online"] and \
                        s in arms["baseline"]:
                    lo = arms["online"][s]["loss"]
                    lb = arms["baseline"][s]["loss"]
                    la = arms[arm][s]["loss"]
                    if lo > lb:
                        rs[s] = (lo - la) / (lo - lb)
            if rs:
                rgap[arm] = rs
                print(f"  {arm:<12s} "
                      + " ".join(f"s{s}:{v:+.2f}" for s, v in rs.items())
                      + f"   median {np.median(list(rs.values())):+.2f}")
        # ---- registered bars ----
        def paired_wins(a, b):
            return sum(arms[a][s]["loss"] < arms[b][s]["loss"]
                       for s in seeds if s in arms.get(a, {})
                       and s in arms.get(b, {}))
        n_pair = sum(1 for s in seeds
                     if s in arms.get("routeA", {}) and s in arms["online"])
        if n_pair:
            barA = paired_wins("routeA", "online") == n_pair
            print(f"BAR A (routeA beats online on all {n_pair} paired "
                  f"seeds): {'PASS' if barA else 'FAIL'}")
        if "routeA" in rgap and "scalarLive" in rgap:
            common = [s for s in rgap["routeA"] if s in rgap["scalarLive"]]
            if common:
                medA = float(np.median([rgap["routeA"][s]
                                        for s in common]))
                medS = float(np.median([rgap["scalarLive"][s]
                                        for s in common]))
                print(f"BAR B (median R_gap routeA {medA:+.2f} > "
                      f"scalarLive {medS:+.2f}): "
                      f"{'PASS' if medA > medS else 'FAIL'}")
        if "routeA" in rgap and "frozenPhase" in rgap:
            common = [s for s in rgap["routeA"] if s in rgap["frozenPhase"]]
            if common:
                ret = []
                for s in common:
                    lo = arms["online"][s]["loss"]
                    la = arms["routeA"][s]["loss"]
                    lf = arms["frozenPhase"][s]["loss"]
                    if lo > la:
                        ret.append((lo - lf) / (lo - la))
                if ret:
                    mret = float(np.median(ret))
                    print(f"BAR C (frozenPhase retains >= 50% of live "
                          f"routeA closure): median {mret:.2f} -> "
                          f"{'PASS' if mret >= 0.5 else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    data = collect()
    if args.gate:
        gate(data)
    else:
        report(data)


if __name__ == "__main__":
    main()
