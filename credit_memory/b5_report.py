"""B5 aggregation/report: reads results/credit_memory/b5/*.json (written
by credit_memory/b5_train.py or scripts/b5_pilot.sh /
scripts/b5_full_matrix.sh) and prints per-arm medians, per-seed detail,
and the B5G success/failure classification.

Run:  python -m credit_memory.b5_report --clip 0 --arms online,b4_causal
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b5")


def load(arm, clip):
    # match by JSON content, not filename formatting (bash writes "clip0",
    # argparse floats format as "0.0" -- avoid brittleness either way)
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR,
                                          f"b5_{arm}_clip*_s*.json")))
    out = []
    for path in paths:
        run = json.load(open(path))
        if run["clip"] == clip:
            out.append(run)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=float, required=True)
    p.add_argument("--arms", type=str, default="online,b4_arch,b4_causal,bptt")
    args = p.parse_args()
    arms = args.arms.split(",")

    print("=" * 90)
    print(f"B5 report: clip={args.clip}, arms={arms}")
    print("=" * 90)

    all_runs = {arm: load(arm, args.clip) for arm in arms}
    for arm, runs in all_runs.items():
        if not runs:
            print(f"{arm}: NO RUNS FOUND")
            continue
        finals = [r["final_loss"] for r in runs if r["finite"]]
        bests = [r["best_loss"] for r in runs if r["finite"]]
        medlate = [r["median_late_loss"] for r in runs if r["finite"]]
        n_finite = sum(1 for r in runs if r["finite"])
        cos_by_step = {}
        for r in runs:
            for d in r["diagnostics"]:
                cos_by_step.setdefault(d["step"], []).append(
                    d["cos_train_vs_bptt"])
        print(f"\n[{arm}] n_seeds={len(runs)}  finite={n_finite}/{len(runs)}")
        if finals:
            print(f"  final_loss   median={np.median(finals):.4f}  "
                  f"per-seed={[round(v, 4) for v in finals]}")
            print(f"  best_loss    median={np.median(bests):.4f}")
            print(f"  median_late  median={np.median(medlate):.4f}")
        for step in sorted(cos_by_step):
            vals = cos_by_step[step]
            print(f"  cos_vs_bptt @ step {step}: median={np.median(vals):.3f}"
                  f"  per-seed={[round(v, 3) for v in vals]}")

    # A0 vs A2 head-to-head + B5G classification (only if both present)
    if "online" in all_runs and "b4_causal" in all_runs:
        a0, a2 = all_runs["online"], all_runs["b4_causal"]
        seeds_common = sorted(set(r["seed"] for r in a0)
                              & set(r["seed"] for r in a2))
        if seeds_common:
            print("\n" + "-" * 90)
            print("A0 (online) vs A2 (b4_causal) head-to-head, "
                  f"clip={args.clip}, {len(seeds_common)} paired seeds")
            wins, losses_cmp = 0, 0
            for s in seeds_common:
                r0 = next(r for r in a0 if r["seed"] == s)
                r2 = next(r for r in a2 if r["seed"] == s)
                if not (r0["finite"] and r2["finite"]):
                    print(f"  seed {s}: NON-FINITE (online={r0['finite']} "
                          f"b4_causal={r2['finite']})")
                    continue
                better = r2["final_loss"] < r0["final_loss"]
                wins += better
                losses_cmp += not better
                cos0 = r0["diagnostics"][-1]["cos_train_vs_bptt"] \
                    if r0["diagnostics"] else float("nan")
                cos2 = r2["diagnostics"][-1]["cos_train_vs_bptt"] \
                    if r2["diagnostics"] else float("nan")
                print(f"  seed {s}: online final={r0['final_loss']:.4f}  "
                      f"b4_causal final={r2['final_loss']:.4f}  "
                      f"{'A2 WINS' if better else 'A0 wins'}   "
                      f"(final-step cos: online={cos0:.3f} "
                      f"b4_causal={cos2:.3f})")
            print(f"  A2 wins {wins}/{wins + losses_cmp} on final task loss")

            finite_pairs = [(next(r for r in a0 if r["seed"] == s),
                            next(r for r in a2 if r["seed"] == s))
                           for s in seeds_common]
            finite_pairs = [(r0, r2) for r0, r2 in finite_pairs
                            if r0["finite"] and r2["finite"]]
            if finite_pairs:
                loss_improves = sum(r2["final_loss"] < r0["final_loss"]
                                    for r0, r2 in finite_pairs)
                cos_improves = sum(
                    (r2["diagnostics"][-1]["cos_train_vs_bptt"]
                     if r2["diagnostics"] else 0)
                    > (r0["diagnostics"][-1]["cos_train_vs_bptt"]
                       if r0["diagnostics"] else 0)
                    for r0, r2 in finite_pairs)
                n = len(finite_pairs)
                print(f"\n  B5G classification signal: loss improves on "
                      f"{loss_improves}/{n} seeds, cosine improves on "
                      f"{cos_improves}/{n} seeds "
                      f"(interpret per PHASE_B5.md's B5G rubric; this "
                      f"script reports the raw signal, not the final "
                      f"classification)")


if __name__ == "__main__":
    main()
