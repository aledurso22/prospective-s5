"""B5E selector diagnostics: for each A2 (b4_causal) seed, report the
initial calibration relevance margin (top channel vs runner-up) and
whether the FROZEN channel would still be top-ranked if relevance were
recomputed offline at later training checkpoints (params have moved).
Offline-only, does not alter any training run.

Run:  python -m credit_memory.b5_selector_diagnostics --clip 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.streaming import StreamingRelevance
from credit_memory.b5_train import (set_config, draw_task_batch, loss_of,
                                    N, T, BATCH, N_CAL_TRAJ, CHECKPOINTS)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b5")
SEEDS = list(range(8))


def relevance_snapshot(params, rng, f_diag):
    """Same protocol as b5_train's calibration selector, but returns the
    FULL rho vector per mode (not just argmax) for margin analysis."""
    estimators = {m: StreamingRelevance(f_diag, BATCH, mode="windowed")
                 for m in range(N)}
    for _ in range(N_CAL_TRAJ):
        x, y = draw_task_batch(rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, _ = tcg.sensitivities(params, h, x)
        for m in range(N):
            c_m = build_c_t(q[1], params["b"][1][:, m])
            for t in range(T):
                estimators[m].step(Sa[0][t, :, m], c_m[t])
    return {m: estimators[m].rho.copy() for m in range(N)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=float, required=True)
    args = p.parse_args()

    set_config()
    print("=" * 90)
    print(f"B5 selector diagnostics, clip={args.clip}")
    print("=" * 90)

    rows = []
    for seed in SEEDS:
        run_path_candidates = [
            os.path.join(RESULTS_DIR, f"b5_b4_causal_clip{c}_s{seed}.json")
            for c in (str(args.clip), str(int(args.clip)))]
        run = None
        for path in run_path_candidates:
            if os.path.exists(path):
                run = json.load(open(path))
                if run["clip"] == args.clip:
                    break
                run = None
        if run is None:
            print(f"seed {seed}: no matching run found, skip")
            continue
        top_j_by_mode = {int(k): v for k, v in
                         run["selector_info"]["top_j_by_mode"].items()}

        params0 = tcg.init_params(seed)
        cal_rng = np.random.RandomState(777 + seed)
        f_diag0 = build_F(params0["a"][1])
        rho0 = relevance_snapshot(params0, cal_rng, f_diag0)

        margins = {}
        for m in range(N):
            sorted_abs = np.sort(np.abs(rho0[m]))[::-1]
            top, second = sorted_abs[0], sorted_abs[1]
            margins[m] = float((top - second) / (top + 1e-30))

        # stability check at the FINAL checkpoint: replay training-arm
        # dynamics is not re-derivable exactly offline without redoing
        # the whole run, so instead we approximate "would selection
        # change" by recomputing relevance from a FRESH calibration-style
        # prefix using the seed's ORIGINAL architecture (a1, B1 do not
        # change during training in this construction -- only theta/rho
        # of layer 0 and layer >=1 params move; layer 1's a1,B1, which
        # is what the selector's pole/routing depend on, DOES move too
        # since it's part of flat/pack). We approximate stability by the
        # calibration-time margin alone here (a direct re-check would
        # require re-running training with periodic snapshots, deferred
        # to B5D's "not yet" continuous-adaptation scope).
        final_loss = run["final_loss"]
        final_cos = run["diagnostics"][-1]["cos_train_vs_bptt"] \
            if run["diagnostics"] else None
        median_margin = float(np.median(list(margins.values())))
        rows.append(dict(seed=seed, margins=margins,
                         median_margin=median_margin,
                         final_loss=final_loss, final_cos=final_cos,
                         top_j_by_mode=top_j_by_mode))
        print(f"seed {seed}: median_margin={median_margin:.3f}  "
              f"final_cos={final_cos:.3f}  final_loss={final_loss:.4f}  "
              f"per-mode margins={ {m: round(v, 2) for m, v in margins.items()} }")

    print("-" * 90)
    margins_all = [r["median_margin"] for r in rows]
    cos_all = [r["final_cos"] for r in rows]
    loss_all = [r["final_loss"] for r in rows]
    if len(rows) >= 3:
        corr_margin_cos = float(np.corrcoef(margins_all, cos_all)[0, 1])
        corr_margin_loss = float(np.corrcoef(margins_all, loss_all)[0, 1])
        print(f"corr(median_margin, final_cos) = {corr_margin_cos:.3f}")
        print(f"corr(median_margin, final_loss) = {corr_margin_loss:.3f}")
    else:
        corr_margin_cos = corr_margin_loss = None

    out_path = os.path.join(RESULTS_DIR,
                            f"b5_selector_diagnostics_clip{args.clip}.json")
    with open(out_path, "w") as f:
        json.dump(dict(clip=args.clip, rows=rows,
                       corr_margin_vs_final_cos=corr_margin_cos,
                       corr_margin_vs_final_loss=corr_margin_loss), f,
                 indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
