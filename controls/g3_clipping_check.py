"""G3 — clipping mechanism check (gauge-claim quantification).

C2 found the causal real-w effect is a quasi-static positive RELATIVE
modal gain structure. The open question: does that gain act through the
global-norm clip (and Adam), or independently of it? Also the gauge
claim for the polar arms: only if clipping fires on essentially all
relevant updates is common positive gradient scaling effectively removed
before Adam (supporting gauge language); otherwise common radial scale
is merely "weakly identified / approximately redundant".

Arms: online, real (per-mode real), pcphase, pc0 — each at
  (a) CLIP = 1.0  (frozen protocol; bitwise gates vs stored finals), and
  (b) CLIP = 1e30 (made nonbinding; same code path, no LR retuning —
      registered: do NOT retune learning rates for no-clip arms).
5 paired seeds. Logs clip activation frequency and pre-clip norm
statistics in BOTH regimes.

Registered questions:
  Q1: does real positive modal gain lose its closed-loop benefit when
      clipping is removed?
  Q2: is phase-only comparatively insensitive to clipping?
  Q3: is the full-complex radial channel primarily acting through the
      clip/Adam interaction?
Plus the fire-rate measurement for the gauge claim (pooled and per arm).

Run:  python -m controls.g3_clipping_check
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from toyrig import route_a as cvm
from controls.geometry_traj import setup, train_arm

SEEDS = [0, 1, 2, 3, 4]
ARMS = ["online", "real", "pcphase", "pc0"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def stored_finals():
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c1 = json.load(open(os.path.join(ROOT, "results",
                                     "c1_phase_only_routepc",
                                     "summary.json")))
    fc = json.load(open(os.path.join(ROOT, "results",
                                     "route_pc_factorial",
                                     "summary.json")))
    return {
        "online": {s: rp_ref["finals"]["online"][str(s)] for s in SEEDS},
        "pc0": {s: rp_ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS},
        "pcphase": {s: c1["finals"]["pcphase"][str(s)] for s in SEEDS},
        "real": {s: fc["finals"]["per-mode-real"][str(s)] for s in SEEDS},
    }


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    stored = stored_finals()
    audit0 = dict(rp.BPTT_CALLS)
    med = lambda d: float(np.median([d[s] for s in SEEDS]))
    doc = {"seeds": SEEDS, "arms": {}}

    for arm in ARMS:
        row = {"clip": {}, "noclip": {}}
        for regime, clip in (("clip", cvm.CLIP), ("noclip", 1e30)):
            finals, fire, pre = {}, {}, {}
            for seed in SEEDS:
                out, traj = train_arm(arm, seed, clip=clip)
                finals[seed] = out["final_loss"]
                fire[seed] = float(traj["clip_fire"].mean())
                pre[seed] = dict(
                    p50=float(np.percentile(traj["preclip"], 50)),
                    p90=float(np.percentile(traj["preclip"], 90)),
                    max=float(traj["preclip"].max()))
                print(f"{arm} {regime} s{seed}: final "
                      f"{out['final_loss']:.4f}  fire {fire[seed]:.3f}",
                      flush=True)
            row[regime] = dict(finals={str(s): finals[s] for s in SEEDS},
                               median=med(finals),
                               fire_rate={str(s): fire[s] for s in SEEDS},
                               preclip={str(s): pre[s] for s in SEEDS})
        # gate the clip regime bitwise
        for seed in SEEDS:
            assert row["clip"]["finals"][str(seed)] == stored[arm][seed], \
                f"gate failed {arm} s{seed}"
        row["gate_clip_bitwise"] = True
        doc["arms"][arm] = row
        print("-" * 70)
        print(f"[{arm}] median clip {row['clip']['median']:.4f} vs "
              f"noclip {row['noclip']['median']:.4f}  "
              f"(fire rate {np.mean(list(row['clip']['fire_rate'].values())):.3f})")

    # online reference for "benefit" reading
    on_c = doc["arms"]["online"]["clip"]["median"]
    on_n = doc["arms"]["online"]["noclip"]["median"]
    print("=" * 70)
    print(f"online reference: clip {on_c:.4f} / noclip {on_n:.4f}")
    for arm in ["real", "pcphase", "pc0"]:
        a = doc["arms"][arm]
        ben_c = (on_c - a["clip"]["median"]) / on_c
        ben_n = (on_n - a["noclip"]["median"]) / on_n
        print(f"[{arm}] relative benefit vs online: clip {ben_c:+.3f}  "
              f"noclip {ben_n:+.3f}")

    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    doc["git"] = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()
    doc["probe_calls"] = audit
    with open(os.path.join(OUT, "g3_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g3_summary.json")


if __name__ == "__main__":
    main()
