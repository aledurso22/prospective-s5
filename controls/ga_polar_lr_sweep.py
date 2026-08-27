"""GA — free log-polar LR control (refinement A).

The G1 NaNs at the original meta-LR do NOT establish intrinsic polar
instability: direct (alpha, phi)-SGD removes the rho^-2 metric factor
that Cartesian SGD has, so the original meta-LR is too hot here by
~rho^2. Registered sweep:

    eta_polar / eta_w,original in {1e-1, 1e-2, 1e-3, 1e-4}

on the 5 frozen seeds, free log-polar, everything else PC0-identical.
No asymmetric tuning beyond this sweep.

REGISTERED SELECTION (fixed before running):
  eligible(rate)   = all 5 seeds finite;
  selected rate    = the eligible rate with the best median (if any);
  competitive      = median <= 1.5 x median(PC0) AND beats online >= 4/5.
  If competitive: extend the SELECTED rate to 15 seeds, then test
  gauge-fixed polar at the SAME selected rate (5 seeds -> 15 if it also
  passes the competitive rule).

Run:  python -m controls.ga_polar_lr_sweep
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm

SEEDS5 = [0, 1, 2, 3, 4]
RATES = [1e-1, 1e-2, 1e-3, 1e-4]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def stored():
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    f = {"online": {}, "pc0": {}}
    for s in range(5):
        f["online"][s] = rp_ref["finals"]["online"][str(s)]
        f["pc0"][s] = rp_ref["finals"]["pc_b0.0"][str(s)]
    for s in range(5, 15):
        f["online"][s] = c15["finals"]["online"][str(s)]
        f["pc0"][s] = c15["finals"]["pc0"][str(s)]
    return f


def run(arm, seeds, lr_scale):
    finals, finite = {}, {}
    for seed in seeds:
        out, traj = train_arm(arm, seed, lr_scale=lr_scale)
        finals[seed] = out["final_loss"]
        finite[seed] = out["finite"]
        rho = np.abs(traj["w"])
        print(f"  {arm} lr={lr_scale:g} s{seed}: final "
              f"{out['final_loss']:.4f}  finite {out['finite']}  "
              f"rho max {np.nanmax(rho):.2f}", flush=True)
        tag = f"traj_{arm}_lr{lr_scale:g}_s{seed}".replace(".", "p")
        np.savez(os.path.join(OUT, f"{tag}.npz"), **traj)
    return finals, finite


def med(f, seeds):
    return float(np.nanmedian([f[s] for s in seeds]))


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    f = stored()
    audit0 = dict(rp.BPTT_CALLS)
    doc = {"rates": {}, "selection": None}
    pc0_med5 = med(f["pc0"], SEEDS5)
    on_med5 = med(f["online"], SEEDS5)

    eligible = []
    for rate in RATES:
        print("=" * 70)
        finals, finite = run("polar", SEEDS5, rate)
        all_fin = all(finite[s] for s in SEEDS5)
        m = med(finals, SEEDS5) if all_fin else float("nan")
        beats = sum(finals[s] < f["online"][s] for s in SEEDS5) \
            if all_fin else 0
        comp = bool(all_fin and m <= 1.5 * pc0_med5 and beats >= 4)
        print(f"[polar lr={rate:g}] median {m:.4f}  finite-all {all_fin}"
              f"  beats online {beats}/5  competitive {comp}")
        doc["rates"][f"{rate:g}"] = dict(
            finals={str(s): finals[s] for s in SEEDS5},
            finite_all=all_fin, median=m, beats_online=beats,
            competitive=comp)
        if all_fin:
            eligible.append((m, rate, comp))

    if eligible:
        eligible.sort()
        m_best, rate_best, comp = eligible[0]
        print(f"selected rate: {rate_best:g} (median {m_best:.4f}, "
              f"competitive {comp})")
        doc["selection"] = dict(rate=rate_best, median5=m_best,
                                competitive=comp)
        if comp:
            finals15, _ = run("polar", range(5, 15), rate_best)
            finals15.update(doc["rates"][f"{rate_best:g}"]["finals"])
            finals15 = {int(k): v for k, v in finals15.items()}
            m15 = med(finals15, list(range(15)))
            fails = [s for s in range(15)
                     if finals15[s] > f["online"][s]]
            beats15 = sum(finals15[s] < f["online"][s] for s in range(15))
            print(f"[polar lr={rate_best:g}] 15-seed median {m15:.4f}  "
                  f"beats online {beats15}/15  fails {fails}")
            doc["selection"]["finals15"] = {str(s): finals15[s]
                                            for s in range(15)}
            doc["selection"]["median15"] = m15
            doc["selection"]["fails15"] = fails
            # gauge-fixed at the same selected rate
            print("=" * 70)
            fg, fin_g = run("polar_gauge", SEEDS5, rate_best)
            all_fin_g = all(fin_g[s] for s in SEEDS5)
            mg = med(fg, SEEDS5) if all_fin_g else float("nan")
            beats_g = sum(fg[s] < f["online"][s] for s in SEEDS5) \
                if all_fin_g else 0
            comp_g = bool(all_fin_g and mg <= 1.5 * pc0_med5
                          and beats_g >= 4)
            print(f"[polar_gauge lr={rate_best:g}] median {mg:.4f}  "
                  f"beats online {beats_g}/5  competitive {comp_g}")
            doc["gauge"] = dict(
                rate=rate_best,
                finals5={str(s): fg[s] for s in SEEDS5},
                median5=mg, beats_online=beats_g, competitive=comp_g)
            if comp_g:
                fg15, _ = run("polar_gauge", range(5, 15), rate_best)
                fg15.update({int(k): v
                             for k, v in doc["gauge"]["finals5"].items()})
                mg15 = med(fg15, list(range(15)))
                fails_g = [s for s in range(15)
                           if fg15[s] > f["online"][s]]
                print(f"[polar_gauge lr={rate_best:g}] 15-seed median "
                      f"{mg15:.4f}  fails {fails_g}")
                doc["gauge"]["finals15"] = {str(s): fg15[s]
                                            for s in range(15)}
                doc["gauge"]["median15"] = mg15
                doc["gauge"]["fails15"] = fails_g
    else:
        print("no eligible rate (all NaN somewhere) — free log-polar "
              "remains unstable at every registered rate")

    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    doc["probe_calls"] = audit
    doc["git"] = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()
    with open(os.path.join(OUT, "ga_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote ga_summary.json")


if __name__ == "__main__":
    main()
