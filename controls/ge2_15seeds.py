"""GE2 — action-aware RoutePC (E2) on the full 15 paired seeds + the
action-space radial-null verification (priority 1 + 2).

E2 (clip+Adam-Jacobian residual) is extended to seeds 0..14 with the
R_A probe; the obsolete 5-seed 1.5x PC0 gate is NOT applied. Seeds 0..4
are gated bitwise against the stored GD finals.

Reports (per the directive):
  median loss, paired ratios (vs online, vs PC0), wins vs online,
  failure count;
  |w| (max, final median) and sd_j(log |w_j|) per layer (does the
  action-aware arm retain relative modal gain?);
  R_A = ||dA/d radial w|| / ||dA/d tangential w|| (global and per
  layer, over training) — the actual clip+Adam action's sensitivity to
  radial vs tangential geometry directions;
  radial residual share (exists, ~0.41-0.48) CONTRASTED with R_A —
  "radial residual exists" != "radial action sensitivity is null".

BPTT/exact calls: 0 (probes disabled). Run: python -m controls.ge2_15seeds
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
    gd = json.load(open(os.path.join(OUT, "gd_summary.json")))
    stored5 = {int(s): v for s, v in
               gd["arms"]["e2action"]["finals5"].items()}
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    onl = {int(s): v for s, v in c15["finals"]["online"].items()}
    pc0 = {int(s): v for s, v in c15["finals"]["pc0"].items()}

    audit0 = dict(rp.BPTT_CALLS)
    finals, stats = {}, {}
    for seed in range(15):
        out, traj = train_arm("e2action", seed, extra=True,
                              exact_probes=False, ra_probe=True)
        finals[seed] = out["final_loss"]
        if seed in stored5:
            assert finals[seed] == stored5[seed], \
                f"gate failed e2action s{seed}"
        rho = np.abs(traj["w"])
        sd_log = [float(np.log(np.maximum(np.abs(traj["w"][-100:, l, :]),
                                          1e-30)).std())
                        for l in range(4)]
        share = (traj["res_rad"][1:]
                 / (traj["res_rad"][1:] + traj["res_tan"][1:] + 1e-30))
        stats[seed] = dict(
            rho_max=float(rho.max()),
            rho_final_med=float(np.median(rho[-100:])),
            sd_log_per_layer=sd_log,
            rad_share_med=float(np.median(share)),
            ra_glob_med=float(np.median(traj["ra_glob"])),
            ra_glob_early=float(np.median(traj["ra_glob"][:6])),
            ra_glob_late=float(np.median(traj["ra_glob"][-6:])),
            ra_layer_med=[float(np.median(traj["ra_layer"][:, l]))
                          for l in range(4)])
        np.savez(os.path.join(OUT, f"traj_e2action_s{seed}.npz"), **traj)
        s = stats[seed]
        print(f"e2action s{seed}: final {finals[seed]:.4f}  rho max "
              f"{s['rho_max']:.2f}  R_A med {s['ra_glob_med']:.3f} "
              f"(early {s['ra_glob_early']:.3f} late "
              f"{s['ra_glob_late']:.3f})", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    all15 = list(range(15))
    med = lambda d: float(np.median([d[s] for s in all15]))
    rat_o = [finals[s] / onl[s] for s in all15]
    rat_c = [finals[s] / pc0[s] for s in all15]
    fails = [s for s in all15 if finals[s] > onl[s]]
    wins = sum(finals[s] < onl[s] for s in all15)
    beats_c = sum(finals[s] < pc0[s] for s in all15)
    print("-" * 78)
    print(f"e2action 15-seed: median {med(finals):.4f}  "
          f"(online {med(onl):.4f} / PC0 {med(pc0):.4f})")
    print(f"  paired ratios vs online: {['%.2f' % r for r in rat_o]}  "
          f"median {np.median(rat_o):.3f}")
    print(f"  paired ratios vs PC0   : {['%.2f' % r for r in rat_c]}  "
          f"median {np.median(rat_c):.3f}")
    print(f"  wins vs online {wins}/15  beats PC0 {beats_c}/15  "
          f"fails {fails}")
    print("gain structure per layer (pooled sd(log|w|), rho final med):")
    for l in range(4):
        print(f"  L{l}: sd(log|w|) "
              f"{np.median([stats[s]['sd_log_per_layer'][l] for s in all15]):.3f}"
              f"  R_A med "
              f"{np.median([stats[s]['ra_layer_med'][l] for s in all15]):.3f}")
    print(f"R_A global: median "
          f"{np.median([stats[s]['ra_glob_med'] for s in all15]):.3f}  "
          f"early {np.median([stats[s]['ra_glob_early'] for s in all15]):.3f}"
          f"  late "
          f"{np.median([stats[s]['ra_glob_late'] for s in all15]):.3f}")
    print(f"radial residual share (exists): median "
          f"{np.median([stats[s]['rad_share_med'] for s in all15]):.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, finals={str(s): finals[s] for s in all15},
               median=med(finals), ratios_online=rat_o, ratios_pc0=rat_c,
               wins_online=wins, beats_pc0=beats_c, fails=fails,
               stats={str(s): stats[s] for s in all15},
               probe_calls=audit)
    with open(os.path.join(OUT, "ge2_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote ge2_summary.json")


if __name__ == "__main__":
    main()
