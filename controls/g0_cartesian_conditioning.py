"""G0 — Cartesian self-annealing diagnostic (refined geometry program).

Hypothesis under test (sharpened): with w = rho e^{i phi} = e^{alpha+i
phi}, the Euclidean metric inherited from Cartesian complex w is
ds^2 = rho^2 (d alpha^2 + d phi^2), so locally-Cartesian SGD on w is
equivalent to log-polar descent with effective coordinate LR
eta_eff = eta/rho^2 for BOTH coordinates:

    large |w| may self-anneal the ENTIRE Cartesian-SGD meta-learner.

This script (a) regenerates the PC0 and pcPhase trajectory banks with
full logging (15 seeds each; bitwise gates vs ALL stored finals), then
(b) measures, per seed and layer: log rho, |d alpha|, |d phi|,
rho^2|d alpha|, rho^2|d phi|, loss, and the global clipping fire rate.

Registered reading rules (fixed before running):
  * SUPPORTIVE: failures show rho growth BEFORE bad-basin entry, AND
    |d alpha|, |d phi| decay roughly like rho^-2 (log-log slope in
    [-2.5, -1.5]), AND rho^2-rescaled quantities are visibly more
    stationary (lower CV) than raw ones.
  * FALSIFIED (causal direction): rho grows only AFTER divergence/entry.
  * pcphase (rho = 1 always) cannot suffer this pathology; its four
    failures (seeds 3, 6, 9, 13) must have a DIFFERENT precursor —
    checked via d phi statistics before entry.

Bad-basin entry (registered rule): first step K where the trailing-25
mean loss >= L_online_final(seed) on that paired seed (the failure
criterion itself). If no such K exists the seed never "fails" by this
rule (entry = None).

Bank: results/geometry_audit/traj_{pc0,pcphase}_s{seed}.npz
Summary: results/geometry_audit/g0_summary.json

Run:  python -m controls.g0_cartesian_conditioning
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from toyrig.train_cell import STEPS
from controls.geometry_traj import setup, train_arm

SEEDS = list(range(15))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def stored_finals():
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c1 = json.load(open(os.path.join(ROOT, "results",
                                     "c1_phase_only_routepc",
                                     "summary.json")))
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    f = {"pc0": {}, "pcphase": {}, "online": {}}
    for s in range(5):
        f["pc0"][s] = rp_ref["finals"]["pc_b0.0"][str(s)]
        f["pcphase"][s] = c1["finals"]["pcphase"][str(s)]
        f["online"][s] = rp_ref["finals"]["online"][str(s)]
    for s in range(5, 15):
        f["pc0"][s] = c15["finals"]["pc0"][str(s)]
        f["pcphase"][s] = c15["finals"]["pcphase"][str(s)]
        f["online"][s] = c15["finals"]["online"][str(s)]
    return f


def entry_step(losses, online_final, win=25):
    """Point of no return: first K such that the trailing-25 mean loss
    stays >= online_final for ALL steps >= K. (Naive first-crossing is
    meaningless: the initial loss already exceeds online_final on every
    seed.) Returns None for runs that end below online_final."""
    tr = np.convolve(losses, np.ones(win) / win, mode="valid")
    above = tr >= online_final
    if not above[-1]:
        return None
    # find the LAST step below threshold; entry is right after it
    below = np.nonzero(~above)[0]
    if not len(below):
        return win
    return int(below[-1] + 1 + win)


def analyze(arm, seed, traj, online_final):
    w = traj["w"]                                  # (STEPS, L, N)
    rho = np.abs(w)
    phi = np.angle(w)
    logrho = np.log(np.maximum(rho, 1e-30))
    dalpha = np.abs(np.diff(logrho, axis=0))
    dphi = np.abs(np.diff(np.unwrap(phi, axis=0), axis=0))
    row = dict(seed=seed, arm=arm)
    row["entry"] = entry_step(traj["losses"], online_final)
    row["clip_fire_rate"] = float(traj["clip_fire"].mean())
    # rho growth before entry vs after (pooled modes/layers)
    e = row["entry"] if row["entry"] else STEPS
    pre = slice(0, max(e - 25, 1))
    row["rho_med_early"] = float(np.median(rho[:200]))
    row["rho_med_pre_entry"] = float(np.median(rho[pre]))
    row["rho_med_final"] = float(np.median(rho[-100:]))
    # log-log slope of |d| vs rho (pooled, steps 50+)
    sl = slice(50, STEPS - 1)
    for name, d in (("dalpha", dalpha), ("dphi", dphi)):
        dd = d[sl].ravel()
        rr = rho[1:][sl].ravel()
        m = dd > 0
        rho_spread = float(np.log(rr).max() - np.log(rr).min())
        if m.sum() > 100 and rho_spread > 1e-3:
            slope = np.polyfit(np.log(rr[m]), np.log(dd[m]), 1)[0]
        else:
            slope = float("nan")
        row[f"slope_{name}"] = float(slope)
        # stationarity: CV of raw vs rescaled (per-seed pooled modes)
        rs = dd * rr ** 2
        row[f"cv_{name}"] = float(dd.std() / (dd.mean() + 1e-30))
        row[f"cv_r2_{name}"] = float(rs.std() / (rs.mean() + 1e-30))
    return row


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    stored = stored_finals()

    # ---- bank generation + bitwise gates (skipped with --analyze-only
    # when the bank already exists; analysis always reruns) ----
    import sys
    analyze_only = "--analyze-only" in sys.argv and all(
        os.path.exists(os.path.join(OUT, f"traj_{a}_s{s}.npz"))
        for a in ("pc0", "pcphase") for s in SEEDS)
    audit0 = dict(rp.BPTT_CALLS)
    gates = []
    if not analyze_only:
        for arm in ["pc0", "pcphase"]:
            for seed in SEEDS:
                print(f"{arm} s{seed}...", flush=True)
                out, traj = train_arm(arm, seed)
                d = abs(out["final_loss"] - stored[arm][seed])
                gates.append(d)
                print(f"  final {out['final_loss']:.4f}  "
                      f"stored {stored[arm][seed]:.4f}  "
                      f"{'==' if d == 0.0 else 'DIFF %.2e' % d}",
                      flush=True)
                assert d == 0.0, f"gate failed {arm} s{seed}"
                np.savez(os.path.join(OUT, f"traj_{arm}_s{seed}.npz"),
                         **traj)
        print(f"BANK GATES: max |dfinal| {max(gates):.2e} == 0 (PASS)")
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    print(f"exact-teacher probe calls (diagnostic only): {audit}")

    # ---- diagnostic ----
    rows = []
    for arm in ["pc0", "pcphase"]:
        for seed in SEEDS:
            traj = np.load(os.path.join(OUT, f"traj_{arm}_s{seed}.npz"))
            rows.append(analyze(arm, seed, traj, stored["online"][seed]))

    print("-" * 78)
    hdr = (f"{'arm':<8s}{'seed':>4s}{'entry':>7s}{'clip%':>7s}"
           f"{'rho_e':>8s}{'rho_pre':>8s}{'rho_fin':>8s}"
           f"{'sl_alpha':>9s}{'sl_phi':>9s}{'CVa':>6s}{'CVr2a':>7s}"
           f"{'CVp':>6s}{'CVr2p':>7s}")
    print(hdr)
    for r in rows:
        print(f"{r['arm']:<8s}{r['seed']:>4d}"
              f"{str(r['entry']):>7s}{r['clip_fire_rate']:>7.3f}"
              f"{r['rho_med_early']:>8.2f}{r['rho_med_pre_entry']:>8.2f}"
              f"{r['rho_med_final']:>8.2f}{r['slope_dalpha']:>9.2f}"
              f"{r['slope_dphi']:>9.2f}{r['cv_dalpha']:>6.2f}"
              f"{r['cv_r2_dalpha']:>7.2f}{r['cv_dphi']:>6.2f}"
              f"{r['cv_r2_dphi']:>7.2f}")

    fails_pc0 = [3, 6, 7, 8, 12, 13]
    fails_pp = [3, 6, 9, 13]
    pc0_rows = [r for r in rows if r["arm"] == "pc0"]
    pp_rows = [r for r in rows if r["arm"] == "pcphase"]
    fr = [r for r in pc0_rows if r["seed"] in fails_pc0]
    sr = [r for r in pc0_rows if r["seed"] not in fails_pc0]
    med = lambda rs, k: float(np.nanmedian([r[k] for r in rs]))
    summ = dict(
        pc0_fail=dict(
            rho_pre=med(fr, "rho_med_pre_entry"),
            rho_fin=med(fr, "rho_med_final"),
            slope_dalpha=med(fr, "slope_dalpha"),
            slope_dphi=med(fr, "slope_dphi"),
            cv_dalpha=med(fr, "cv_dalpha"),
            cv_r2_dalpha=med(fr, "cv_r2_dalpha"),
            cv_dphi=med(fr, "cv_dphi"),
            cv_r2_dphi=med(fr, "cv_r2_dphi")),
        pc0_success=dict(
            rho_pre=med(sr, "rho_med_pre_entry"),
            rho_fin=med(sr, "rho_med_final"),
            slope_dalpha=med(sr, "slope_dalpha"),
            slope_dphi=med(sr, "slope_dphi"),
            cv_dalpha=med(sr, "cv_dalpha"),
            cv_r2_dalpha=med(sr, "cv_r2_dalpha"),
            cv_dphi=med(sr, "cv_dphi"),
            cv_r2_dphi=med(sr, "cv_r2_dphi")),
        clip_fire_pooled=float(np.mean([r["clip_fire_rate"]
                                        for r in rows])))
    print("-" * 78)
    print("failure seeds (PC0):", json.dumps(summ["pc0_fail"], indent=1))
    print("success seeds (PC0):", json.dumps(summ["pc0_success"],
                                            indent=1))
    print(f"clip fire rate pooled (all arms/seeds): "
          f"{summ['clip_fire_pooled']:.3f}")
    # rho-growth timing vs entry: falsification check
    timing = []
    for r in pc0_rows:
        if r["entry"]:
            timing.append((r["seed"], r["rho_med_early"],
                           r["rho_med_pre_entry"], r["rho_med_final"]))
    print("seed / rho early(0-200) / rho pre-entry / rho final:")
    for t in timing:
        print(f"  s{t[0]:>2d}: {t[1]:.2f} -> {t[2]:.2f} -> {t[3]:.2f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS,
               gates_max=(max(gates) if gates else 0.0),
               probe_calls=audit, rows=rows, summaries=summ,
               fails=dict(pc0=fails_pc0, pcphase=fails_pp))
    with open(os.path.join(OUT, "g0_summary.json"), "w") as f:
        json.dump(doc, f, indent=2, default=float)
    print("wrote g0_summary.json")


if __name__ == "__main__":
    main()
