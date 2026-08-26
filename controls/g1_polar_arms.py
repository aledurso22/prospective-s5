"""G1 — nested coordinate/gauge/optimizer arms (refined program).

Three new selectable candidates against A = phase-only (existing),
Online and PC0 (stored), on the frozen toy protocol:

  polar        B — FREE log-polar w = exp(alpha + i phi); (alpha, phi)
               optimized directly through the same causal PC0
               meta-residual (chain map FD-gated, geometry_traj).
               Isolates the coordinate-conditioning correction: the
               eta_eff = eta/rho^2 self-annealing is removed by
               construction (alpha and phi get the full meta LR).
  polar_gauge  C — polar + layerwise gauge fix: alpha's meta-gradient
               demeaned per layer (sum_j alpha~_j = 0; prod_j |w_j| = 1
               per layer). Removes the common radial direction while
               preserving relative modal gain. NOT assumed to help;
               compared against free polar directly.
  pc0_adam     D — Cartesian PC0 with Adam MetaOpt for w (b1 .9,
               b2 .999, eps 1e-8, one fixed LR = LR_M = 1e-3; no sweep).
               Tests whether the failure mode is Cartesian+SGD rather
               than the parameterization itself.

REGISTERED TWO-STAGE RULE (fixed before running):
  sane(arm)        iff all 5 seeds finite AND median <= median(online).
  competitive(arm) iff median <= 1.5 x median(PC0) AND beats online on
                   >= 4/5 paired seeds.
  competitive arms auto-extend to seeds 5..14 (merged with the stored
  5-seed rows) for the failure-rate comparison.

Do not call common radial scale a true gauge symmetry: see the clipping
fire-rate quantification (g0 summary + g3); until then it is "weakly
identified / approximately redundant".

Reporting (addendum statistical standard): per-seed finals, marginal
medians, paired ratios (arm/online, arm/PC0), failures (ratio > 1),
paired wins; plus rho/alpha/phi dynamics: max and final-median |w|
modulus (runaway-radius check), mean |d alpha| and |d phi| per step.

Run:  python -m controls.g1_polar_arms
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm, polar_chain_fd_gate

SEEDS5 = [0, 1, 2, 3, 4]
ARMS = ["polar", "polar_gauge", "pc0_adam"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def stored():
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c1 = json.load(open(os.path.join(ROOT, "results",
                                     "c1_phase_only_routepc",
                                     "summary.json")))
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    f = {"online": {}, "pc0": {}, "pcphase": {}}
    for s in range(5):
        f["online"][s] = rp_ref["finals"]["online"][str(s)]
        f["pc0"][s] = rp_ref["finals"]["pc_b0.0"][str(s)]
        f["pcphase"][s] = c1["finals"]["pcphase"][str(s)]
    for s in range(5, 15):
        f["online"][s] = c15["finals"]["online"][str(s)]
        f["pc0"][s] = c15["finals"]["pc0"][str(s)]
        f["pcphase"][s] = c15["finals"]["pcphase"][str(s)]
    return f


def run_arm(arm, seeds, f):
    finals, dyn = {}, {}
    for seed in seeds:
        print(f"{arm} s{seed}...", flush=True)
        out, traj = train_arm(arm, seed)
        finals[seed] = out["final_loss"]
        rho = np.abs(traj["w"])
        logrho = np.log(np.maximum(rho, 1e-30))
        dphi = np.abs(np.diff(np.unwrap(np.angle(traj["w"]), axis=0),
                              axis=0))
        dyn[seed] = dict(
            rho_max=float(rho.max()), rho_min=float(rho.min()),
            rho_final_med=float(np.median(rho[-100:])),
            dalpha_step=float(np.abs(np.diff(logrho, axis=0)).mean()),
            dphi_step=float(dphi.mean()),
            finite=out["finite"])
        print(f"  final {out['final_loss']:.4f}  "
              f"rho max {rho.max():.2f}  final med "
              f"{np.median(rho[-100:]):.2f}", flush=True)
        np.savez(os.path.join(OUT, f"traj_{arm}_s{seed}.npz"), **traj)
    return finals, dyn


def report(arm, finals, f, seeds):
    med = lambda d: float(np.median([d[s] for s in seeds]))
    beats_o = sum(finals[s] < f["online"][s] for s in seeds)
    rat_o = np.median([finals[s] / f["online"][s] for s in seeds])
    rat_c = np.median([finals[s] / f["pc0"][s] for s in seeds])
    fails = [s for s in seeds if finals[s] > f["online"][s]]
    beats_c = sum(finals[s] < f["pc0"][s] for s in seeds)
    print(f"  {arm:<11s} median {med(finals):.4f} (online "
          f"{med(f['online']):.4f} / PC0 {med(f['pc0']):.4f})  "
          f"beats online {beats_o}/{len(seeds)}  beats PC0 "
          f"{beats_c}/{len(seeds)}  ratio med vs online {rat_o:.3f}  "
          f"vs PC0 {rat_c:.3f}  fails {fails}")
    return dict(median=med(finals), beats_online=beats_o,
                beats_pc0=beats_c, ratio_online=float(rat_o),
                ratio_pc0=float(rat_c), failures=fails)


def main() -> None:
    assert polar_chain_fd_gate(), "polar chain FD gate FAILED — stop"
    setup()
    os.makedirs(OUT, exist_ok=True)
    f = stored()
    audit0 = dict(rp.BPTT_CALLS)
    med = lambda d, seeds: float(np.median([d[s] for s in seeds]))
    doc = {"seeds5": SEEDS5, "arms": {}, "fd_gate": True}

    for arm in ARMS:
        print("=" * 70)
        finals5, dyn5 = run_arm(arm, SEEDS5, f)
        all_finite = all(dyn5[s]["finite"] for s in SEEDS5)
        sane = all_finite and med(finals5, SEEDS5) <= med(f["online"],
                                                         SEEDS5)
        comp = (med(finals5, SEEDS5) <= 1.5 * med(f["pc0"], SEEDS5)
                and sum(finals5[s] < f["online"][s]
                        for s in SEEDS5) >= 4)
        print(f"[{arm}] 5-seed: median {med(finals5, SEEDS5):.4f}  "
              f"sane {'YES' if sane else 'NO'}  "
              f"competitive {'YES' if comp else 'NO'}")
        row = {"finals5": {str(s): finals5[s] for s in SEEDS5},
               "dyn5": {str(s): dyn5[s] for s in SEEDS5},
               "sane": bool(sane), "competitive": bool(comp),
               "report5": report(arm, finals5, f, SEEDS5)}
        if comp:
            finals15 = dict(finals5)
            f_new, dyn_new = run_arm(arm, range(5, 15), f)
            finals15.update(f_new)
            dyn5.update(dyn_new)
            row["finals15"] = {str(s): finals15[s] for s in range(15)}
            row["dyn15"] = {str(s): dyn5[s] for s in range(15)}
            row["report15"] = report(arm, finals15, f, list(range(15)))
        doc["arms"][arm] = row

    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    doc["probe_calls"] = audit
    doc["git"] = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()
    with open(os.path.join(OUT, "g1_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g1_summary.json")


if __name__ == "__main__":
    main()
