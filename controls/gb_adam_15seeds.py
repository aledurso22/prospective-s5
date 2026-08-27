"""GB — Cartesian PC0 + Adam MetaOpt, 15 seeds + gain-structure stats
(refinement B).

Extends pc0_adam to the full 15 paired seeds (0..4 stored from G1,
gated against g1_summary; 5..14 new runs). Beyond loss/failures,
reports whether the bounded radius still contains meaningful RELATIVE
modal gain or has effectively collapsed toward phase-only:

  sd_j(log |w_{l,j}|)   per layer/seed (final-100 window): relative
                        gain spread across modes — ~0 would mean the
                        geometry is phase-only in disguise;
  max/min |w|           per layer (final): relative gain range.

Run:  python -m controls.gb_adam_15seeds
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    g1 = json.load(open(os.path.join(OUT, "g1_summary.json")))
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    onl = {int(s): v for s, v in c15["finals"]["online"].items()}
    pc0 = {int(s): v for s, v in c15["finals"]["pc0"].items()}

    audit0 = dict(rp.BPTT_CALLS)
    finals = {int(s): v for s, v in
              g1["arms"]["pc0_adam"]["finals5"].items()}
    trajs = {}
    for seed in range(5):
        trajs[seed] = np.load(os.path.join(OUT,
                                           f"traj_pc0_adam_s{seed}.npz"))
    for seed in range(5, 15):
        out, traj = train_arm("pc0_adam", seed)
        finals[seed] = out["final_loss"]
        trajs[seed] = traj
        np.savez(os.path.join(OUT, f"traj_pc0_adam_s{seed}.npz"), **traj)
        print(f"pc0_adam s{seed}: final {out['final_loss']:.4f}",
              flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}

    med = lambda d, seeds: float(np.median([d[s] for s in seeds]))
    all15 = list(range(15))
    fails = [s for s in all15 if finals[s] > onl[s]]
    beats_o = sum(finals[s] < onl[s] for s in all15)
    beats_c = sum(finals[s] < pc0[s] for s in all15)
    rat = np.median([finals[s] / onl[s] for s in all15])
    print("-" * 78)
    print(f"pc0_adam 15-seed: median {med(finals, all15):.4f}  "
          f"(online {med(onl, all15):.4f} / PC0 {med(pc0, all15):.4f})  "
          f"beats online {beats_o}/15  beats PC0 {beats_c}/15  "
          f"ratio med {rat:.3f}  fails {fails}")

    # gain structure: does bounded radius keep RELATIVE modal gain?
    print("gain structure per layer (pooled over seeds):")
    stats = {}
    for l in range(4):
        sds, ranges, rhos = [], [], []
        for seed in all15:
            w = trajs[seed]["w"][-100:, l, :]
            logr = np.log(np.maximum(np.abs(w), 1e-30))
            sds.append(float(logr.std()))
            ranges.append(float(np.abs(w).max()
                                / max(np.abs(w).min(), 1e-30)))
            rhos.append(float(np.median(np.abs(w))))
        stats[l] = dict(sd_log=float(np.median(sds)),
                        maxmin=float(np.median(ranges)),
                        rho_med=float(np.median(rhos)))
        print(f"  L{l}: sd(log|w|) {stats[l]['sd_log']:.3f}  "
              f"max/min {stats[l]['maxmin']:.2f}  "
              f"median |w| {stats[l]['rho_med']:.2f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, finals={str(s): finals[s] for s in all15},
               median=med(finals, all15), beats_online=beats_o,
               beats_pc0=beats_c, ratio_median=float(rat), fails=fails,
               gain_stats={str(l): stats[l] for l in stats},
               probe_calls=audit)
    with open(os.path.join(OUT, "gb_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote gb_summary.json")


if __name__ == "__main__":
    main()
