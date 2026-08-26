"""G7 — failure-mode audit / taxonomy.

Uses the trajectory banks (results/geometry_audit/traj_{arm}_s{seed}.npz)
from G0/G1. For every (arm, seed) that FAILED (final loss > online on
that paired seed) and for the successful controls, align trajectories
around the bad-basin entry (registered rule: first K with trailing-25
mean loss >= online_final(seed); if none, the seed is a control).

Signatures recorded per run (window = entry-250 .. entry+50, or the
analogous window for controls centered at the same quantile of
training):
  gnorm per mode     — modal online-gradient collapse?
  rnorm per mode     — meta-residual explosion / mode dominance?
  phi, |d phi|       — phase freezing vs winding/jumps?
  |w|                — runaway/collapse?
  clip fire, preclip — clipping transition?
  ex_cos, ex_eps     — causal/exact teacher alignment collapse
                       (offline diagnostic, from the 50-step probes)?

REGISTERED CLASSIFIER (fixed before running; thresholds relative to
each run's own early statistics, steps 50..200, to be seed-adaptive):
  TEACHER_COLLAPSE : ex_cos at the checkpoint nearest entry < 0.3 while
                     the early-median ex_cos > 0.6
  RESIDUAL_SPIKE   : max rnorm in window > 10x early median
  GRAD_COLLAPSE    : some mode's gnorm < 0.05x its early median AND that
                     mode holds > 20% of the total rnorm share
  PHASE_FREEZE     : median |d phi| in window < 0.2x early median AND
                     median |w| in window > 3x early median
  PHASE_WINDING    : cumulative |d phi| over the 250 pre-entry steps
                     > 5x the arm's successful-seed median
  CLIP_TRANSITION  : fire rate < 0.5 early, > 0.95 in window
  NONE             : none of the above

A run may carry multiple labels; the primary label is the first in the
order above. Output: per-arm label counts, per-seed table, and the
label matrix used for the audit narrative.

Run:  python -m controls.g7_failure_audit
"""
from __future__ import annotations

import glob
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")
WIN_PRE, WIN_POST = 250, 50


def online_finals():
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    return {int(s): v for s, v in c15["finals"]["online"].items()}


def entry_step(losses, online_final, win=25):
    tr = np.convolve(losses, np.ones(win) / win, mode="valid")
    idx = np.nonzero(tr >= online_final)[0]
    return int(idx[0] + win) if len(idx) else None


def signatures(traj, e):
    losses = traj["losses"]
    gnorm = traj["gnorm"]
    rnorm = traj["rnorm"]
    w = traj["w"]
    phi = np.unwrap(np.angle(w), axis=0)
    dphi = np.abs(np.diff(phi, axis=0))
    rho = np.abs(w)
    early = slice(50, 200)
    if e is None:                       # control: same-relative window
        e = int(0.6 * len(losses))
    win = slice(max(e - WIN_PRE, 1), min(e + WIN_POST, len(losses)))
    sig = {}
    sig["rho_med_win"] = float(np.median(rho[win]))
    sig["rho_med_early"] = float(np.median(rho[early]))
    sig["dphi_med_win"] = float(np.median(dphi[win]))
    sig["dphi_med_early"] = float(np.median(dphi[early]))
    sig["dphi_cum_pre"] = float(dphi[max(e - WIN_PRE, 0):e].sum())
    sig["rnorm_max_win"] = float(rnorm[win].max())
    sig["rnorm_med_early"] = float(np.median(rnorm[early]))
    g_ratio = (gnorm[win].min(axis=0)
               / (np.median(gnorm[early], axis=0) + 1e-30))
    j_min = np.unravel_index(np.argmin(g_ratio), g_ratio.shape)
    share = (rnorm[win][:, j_min[0], j_min[1]].mean()
             / (rnorm[win].mean() + 1e-30))
    sig["gnorm_min_ratio"] = float(g_ratio.min())
    sig["gnorm_min_share"] = float(share / rnorm.shape[-1])
    sig["clip_early"] = float(traj["clip_fire"][early].mean())
    sig["clip_win"] = float(traj["clip_fire"][win].mean())
    if traj["ex_cos"].size:
        k = min(max((e - 1) // 50 - 1, 0), len(traj["ex_cos"]) - 1)
        sig["ex_cos_entry"] = float(np.nanmean(traj["ex_cos"][k]))
        sig["ex_cos_early"] = float(np.nanmean(traj["ex_cos"][1:4]))
    else:
        sig["ex_cos_entry"] = sig["ex_cos_early"] = float("nan")
    return sig


LABEL_ORDER = ["TEACHER_COLLAPSE", "RESIDUAL_SPIKE", "GRAD_COLLAPSE",
               "PHASE_FREEZE", "PHASE_WINDING", "CLIP_TRANSITION"]


def classify(sig, wind_ref):
    labels = []
    if (not np.isnan(sig["ex_cos_entry"])
            and sig["ex_cos_entry"] < 0.3 and sig["ex_cos_early"] > 0.6):
        labels.append("TEACHER_COLLAPSE")
    if sig["rnorm_max_win"] > 10 * (sig["rnorm_med_early"] + 1e-30):
        labels.append("RESIDUAL_SPIKE")
    if sig["gnorm_min_ratio"] < 0.05 and sig["gnorm_min_share"] > 0.2:
        labels.append("GRAD_COLLAPSE")
    if (sig["dphi_med_win"] < 0.2 * sig["dphi_med_early"]
            and sig["rho_med_win"] > 3 * sig["rho_med_early"]):
        labels.append("PHASE_FREEZE")
    if sig["dphi_cum_pre"] > 5 * wind_ref:
        labels.append("PHASE_WINDING")
    if sig["clip_early"] < 0.5 and sig["clip_win"] > 0.95:
        labels.append("CLIP_TRANSITION")
    return labels or ["NONE"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    onl = online_finals()
    banks = sorted(glob.glob(os.path.join(OUT, "traj_*_s*.npz")))
    runs = []
    for b in banks:
        name = os.path.basename(b)[:-4]           # traj_{arm}_s{seed}
        arm = name[len("traj_"):name.rindex("_s")]
        seed = int(name[name.rindex("_s") + 2:])
        traj = np.load(b)
        final = float(traj["losses"][-100:].mean())
        fail = final > onl.get(seed, np.inf) if seed in onl else None
        e = entry_step(traj["losses"], onl.get(seed, np.inf)) \
            if seed in onl else None
        sig = signatures(traj, e)
        runs.append(dict(arm=arm, seed=seed, final=final, fail=fail,
                         entry=e, sig=sig))

    # winding reference: median cumulative pre-entry |dphi| over
    # SUCCESSFUL runs of each arm
    wind_ref = {}
    for arm in sorted({r["arm"] for r in runs}):
        succ = [r["sig"]["dphi_cum_pre"] for r in runs
                if r["arm"] == arm and r["fail"] is False]
        wind_ref[arm] = float(np.median(succ)) if succ else np.inf

    for r in runs:
        r["labels"] = classify(r["sig"], wind_ref[r["arm"]])

    print("-" * 78)
    for r in runs:
        if r["fail"]:
            s = r["sig"]
            print(f"{r['arm']:<11s} s{r['seed']:>2d} entry {r['entry']}  "
                  f"labels {r['labels']}  "
                  f"(rho {s['rho_med_early']:.1f}->{s['rho_med_win']:.1f}"
                  f", dphi {s['dphi_med_early']:.1e}->"
                  f"{s['dphi_med_win']:.1e}, excos "
                  f"{s['ex_cos_early']:.2f}->{s['ex_cos_entry']:.2f})")
    print("-" * 78)
    counts = {}
    for r in runs:
        if r["fail"]:
            for lab in r["labels"][:1]:
                counts.setdefault(r["arm"], {}).setdefault(lab, 0)
                counts[r["arm"]][lab] += 1
    print("primary-label counts per arm:")
    for arm, c in sorted(counts.items()):
        print(f"  {arm:<11s} {c}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, wind_ref=wind_ref, runs=runs, counts=counts)
    with open(os.path.join(OUT, "g7_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g7_summary.json")


if __name__ == "__main__":
    main()
