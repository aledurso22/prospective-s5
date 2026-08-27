"""GAA — RoutePC-AA: action-aware residual + Adam MetaOpt (2x2 missing
cell, priority 3).

Residual x MetaOpt factorial:

                    SGD MetaOpt     Adam MetaOpt
  schematic     |   PC0             pc0_adam
  action-aware  |   E2 (e2action)   THIS ARM (aae2adam)

The residual is e2action's (clip+Adam-direction Jacobian transform of
the realized online teacher); the w update is pc0_adam's per-component
Adam (one fixed LR = LR_M, no sweep). Registered prediction: AA may not
dramatically beat E2/pc0_adam on median if both modifications partly
repair the same pathology; improved failure rate and stable w-geometry
count as the main possible advantage.

Five seeds first; extend to 15 iff sane (all finite AND median <=
median(online)). Audits: step-1 forward identical to PC0 (bitwise);
BPTT/exact calls 0. Report: finals/median, paired ratios, wins,
failures, |w| and sd_j(log|w_j|) per layer.

Run:  python -m controls.gaa_action_adam
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm

SEEDS5 = [0, 1, 2, 3, 4]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    onl = {int(s): v for s, v in c15["finals"]["online"].items()}
    pc0 = {int(s): v for s, v in c15["finals"]["pc0"].items()}

    # step-1 audit
    import controls.geometry_traj as gt
    orig = gt.STEPS
    gt.STEPS = 1
    _, t_pc0 = gt.train_arm("pc0", 0)
    _, t_aa = gt.train_arm("aae2adam", 0)
    gt.STEPS = orig
    assert np.array_equal(t_pc0["w"][-1], t_aa["w"][-1])
    assert t_pc0["losses"][-1] == t_aa["losses"][-1]
    print("AUDIT step-1 forward identical to PC0: PASS")

    audit0 = dict(rp.BPTT_CALLS)
    finals, stats = {}, {}
    for seed in SEEDS5:
        out, traj = train_arm("aae2adam", seed, extra=True,
                              exact_probes=False)
        finals[seed] = out["final_loss"]
        stats[seed] = dict(
            finite=out["finite"],
            rho_max=float(np.abs(traj["w"]).max()),
            sd_log=[float(np.log(np.maximum(
                np.abs(traj["w"][-100:, l, :]), 1e-30)).std())
                    for l in range(4)])
        np.savez(os.path.join(OUT, f"traj_aae2adam_s{seed}.npz"), **traj)
        print(f"aae2adam s{seed}: final {out['final_loss']:.4f}  "
              f"finite {out['finite']}  rho max "
              f"{stats[seed]['rho_max']:.2f}", flush=True)
    med = lambda d, seeds: float(np.median([d[s] for s in seeds]))
    m5 = med(finals, SEEDS5)
    sane = all(stats[s]["finite"] for s in SEEDS5) \
        and m5 <= med(onl, SEEDS5)
    print(f"[aae2adam] 5-seed median {m5:.4f}  sane {bool(sane)}")
    if sane:
        for seed in range(5, 15):
            out, traj = train_arm("aae2adam", seed, extra=True,
                                  exact_probes=False)
            finals[seed] = out["final_loss"]
            np.savez(os.path.join(OUT, f"traj_aae2adam_s{seed}.npz"),
                     **traj)
            print(f"aae2adam s{seed}: final {out['final_loss']:.4f}",
                  flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    all15 = sorted(finals)
    m15 = med(finals, all15)
    rat_o = [finals[s] / onl[s] for s in all15]
    fails = [s for s in all15 if finals[s] > onl[s]]
    wins = sum(finals[s] < onl[s] for s in all15)
    beats_c = sum(finals[s] < pc0[s] for s in all15)
    print("-" * 78)
    print(f"aae2adam ({len(all15)} seeds): median {m15:.4f}  "
          f"(online {med(onl, all15):.4f} / PC0 {med(pc0, all15):.4f})  "
          f"ratio med {np.median(rat_o):.3f}  wins {wins}/{len(all15)}  "
          f"beats PC0 {beats_c}/{len(all15)}  fails {fails}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, finals={str(s): finals[s] for s in all15},
               median=m15, ratios_online=rat_o, wins_online=wins,
               beats_pc0=beats_c, fails=fails, sane=bool(sane),
               stats={str(s): stats.get(s) for s in all15
                      if s in stats},
               probe_calls=audit)
    with open(os.path.join(OUT, "gaa_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote gaa_summary.json")


if __name__ == "__main__":
    main()
