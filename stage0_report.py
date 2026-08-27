"""Report and gate the paired S5 optimizer/credit Stage 0 matrix.

The four required cells are BPTT (``baseline``) and ``online`` at
``clip=0`` and ``clip=1.0``. The predeclared clipped-regime credit-gap gate
requires at least three paired seeds, a positive Online->BPTT loss gap on
every seed, and median relative headroom >= 0.2 (the repository's existing
benchmark headroom bar). This report never launches correction arms.

Run:  python stage0_report.py --task smnist
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from train_bench import GATE_H

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "results", "bench")
CLIPS = {"clip0": 0.0, "clip1": 1.0}
ARMS = ("baseline", "online")
MIN_SEEDS = 3


def _dist(values):
    x = np.asarray(values, dtype=float)
    return dict(n=int(x.size), values=x.tolist(),
                median=float(np.median(x)), mean=float(np.mean(x)),
                sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                min=float(np.min(x)), max=float(np.max(x)))


def _path(results_dir, task, arm, clip_tag, seed):
    return os.path.join(
        results_dir,
        f"metrics_{task}_{arm}_stage0_{clip_tag}_s{seed}.json")


def _discover_seeds(results_dir, task):
    paths = glob.glob(os.path.join(
        results_dir, f"metrics_{task}_online_stage0_clip1_s*.json"))
    seeds = []
    for path in paths:
        tail = os.path.basename(path).rsplit("_s", 1)[-1]
        seeds.append(int(tail.removesuffix(".json")))
    return sorted(set(seeds))


def _matched_config(cell):
    c = cell["config"]
    keys = ("task", "d_model", "state_size", "n_layers", "dropout",
            "batch_size", "lr", "epochs", "train_samples", "test_samples",
            "seed", "standardized", "scan_impl", "downsample", "seq_len",
            "seq2seq", "copy_k", "copy_alpha")
    return {key: c.get(key) for key in keys}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["smnist", "psmnist", "copy"],
                        required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args(argv)
    seeds = (sorted(set(args.seeds)) if args.seeds
             else _discover_seeds(args.results_dir, args.task))
    if not seeds:
        raise SystemExit("no Stage 0 clipped-online metrics found")

    cells = {}
    missing = []
    for clip_tag, clip in CLIPS.items():
        for arm in ARMS:
            for seed in seeds:
                path = _path(args.results_dir, args.task, arm, clip_tag, seed)
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                row = json.load(open(path))
                if float(row["config"]["clip"]) != clip:
                    raise AssertionError(f"clip mismatch in {path}")
                cells[(clip_tag, arm, seed)] = row
    if missing:
        print("INCOMPLETE Stage 0 matrix; missing:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(2)

    # All four cells for a seed must differ only in the intended learning
    # rule / clip axes, not data, model, or budget.
    for seed in seeds:
        ref = _matched_config(cells[("clip0", "baseline", seed)])
        for clip_tag in CLIPS:
            for arm in ARMS:
                got = _matched_config(cells[(clip_tag, arm, seed)])
                if got != ref:
                    raise AssertionError(
                        f"unmatched config for {clip_tag}/{arm}/s{seed}")

    regimes = {}
    for clip_tag, clip in CLIPS.items():
        per_seed = {}
        gaps, headroom = [], []
        for seed in seeds:
            base = cells[(clip_tag, "baseline", seed)]
            online = cells[(clip_tag, "online", seed)]
            lb, lo = base["final_test_loss"], online["final_test_loss"]
            gap = lo - lb
            h = gap / lo
            gaps.append(gap)
            headroom.append(h)
            per_seed[str(seed)] = dict(
                baseline=dict(loss=lb, accuracy=base["final_test_acc"]),
                online=dict(loss=lo, accuracy=online["final_test_acc"]),
                online_minus_bptt_loss=gap,
                relative_headroom=h,
                p_clip=dict(
                    baseline=base["instrumentation"]["p_clip"],
                    online=online["instrumentation"]["p_clip"]),
                chi=dict(
                    baseline=base["instrumentation"]["chi"],
                    online=online["instrumentation"]["chi"]),
                peak_device_memory_bytes=dict(
                    baseline=base["instrumentation"]["peak_device_memory_bytes"],
                    online=online["instrumentation"]["peak_device_memory_bytes"]),
                steps_per_sec=dict(
                    baseline=base["instrumentation"]["steps_per_sec"],
                    online=online["instrumentation"]["steps_per_sec"]),
                finite=dict(baseline=base["finite"], online=online["finite"]),
                audit=dict(baseline=base["audit"], online=online["audit"]),
            )
            assert base["audit"]["bptt_calls"] == base["total_steps"]
            assert online["audit"]["bptt_calls"] == 0
            assert online["audit"]["exact_grad_calls"] == 0
            assert online["audit"]["exact_lambda_calls"] == 0
        regimes[clip_tag] = dict(
            clip=clip,
            per_seed=per_seed,
            online_minus_bptt_loss=_dist(gaps),
            relative_headroom=_dist(headroom),
            positive_gap_seeds=int(np.sum(np.asarray(gaps) > 0)),
        )

    finite_all = all(row["finite"] for row in cells.values())
    clipped = regimes["clip1"]
    gap_exists = bool(
        len(seeds) >= MIN_SEEDS
        and clipped["positive_gap_seeds"] == len(seeds)
        and clipped["relative_headroom"]["median"] >= GATE_H
    )
    decision = dict(
        minimum_paired_seeds=MIN_SEEDS,
        required_positive_gap_on_every_seed=True,
        median_relative_headroom_bar=GATE_H,
        finite_all=finite_all,
        clipped_online_to_bptt_gap_exists=gap_exists,
        proceed_to_small_correction_pilot=bool(gap_exists and finite_all),
        large_sweep_authorized=False,
        reading=("PROCEED only to the small correction pilot"
                 if gap_exists and finite_all else
                 "STOP: do not interpret RoutePC as credit repair on this config"),
    )
    report = dict(task=args.task, seeds=seeds, regimes=regimes,
                  decision=decision)
    out = os.path.join(args.results_dir, f"stage0_{args.task}_report.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Stage 0: {args.task}, paired seeds {seeds}")
    for clip_tag in CLIPS:
        row = regimes[clip_tag]
        print(f"  {clip_tag}: median online-BPTT loss gap "
              f"{row['online_minus_bptt_loss']['median']:+.4f}; median h "
              f"{row['relative_headroom']['median']:+.3f}; positive "
              f"{row['positive_gap_seeds']}/{len(seeds)}")
    print(f"  decision: {decision['reading']}")
    print(f"  large sweep authorized: NO")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
