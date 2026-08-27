"""GE-SIG — failure signature on bounded-radius arms (priority 4).

Radial runaway is NOT necessary for failure (pc0_adam fails on
[3, 9, 10] with bounded radius; pcPhase on [3, 6, 9, 13] with |w| = 1
exactly). The leading remaining hypothesis: causal/exact residual
alignment collapses (or changes sign) before failure and stays bad.

Analysis only, over the stored trajectory banks (pc0_adam, pc0,
pcphase; exact-teacher probes logged every 50 steps): per seed, compare
failures vs successes on

  cos(r_causal, r_exact),  ||eps||,  ||r_causal||,  |d phi|,  loss

in early / mid / late windows and around the point of no return
(registered rule from g0: first K with trailing-25 mean >= online final
for all steps >= K). "Sign change" test: fraction of checkpoints with
cos < 0, and the last time cos exceeded 0.6, per seed — do failures sit
systematically lower (or negative) on cos and never recover?

Run:  python -m controls.ge_failure_signature
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def entry_step(losses, online_final, win=25):
    tr = np.convolve(losses, np.ones(win) / win, mode="valid")
    above = tr >= online_final
    if not above[-1]:
        return None
    below = np.nonzero(~above)[0]
    return win if not len(below) else int(below[-1] + 1 + win)


def windows(n, e):
    third = n // 3
    return dict(early=slice(0, third), mid=slice(third, 2 * third),
                late=slice(2 * third, n),
                entry=slice(max(e - 150, 0) if e else 0,
                            min(e + 100, n) if e else 0))


def main() -> None:
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    onl = {int(s): v for s, v in c15["finals"]["online"].items()}
    arms = ["pc0_adam", "pc0", "pcphase"]
    fail_seeds = {"pc0_adam": [3, 9, 10], "pc0": [3, 6, 7, 8, 12, 13],
                  "pcphase": [3, 6, 9, 13]}
    rows = []
    for arm in arms:
        files = sorted(glob.glob(os.path.join(
            OUT, f"traj_{arm}_s*.npz")),
            key=lambda p: int(p.rsplit("_s", 1)[1][:-4]))
        for f in files:
            seed = int(f.rsplit("_s", 1)[1][:-4])
            traj = np.load(f)
            losses = traj["losses"]
            n = len(losses)
            e = entry_step(losses, onl[seed])
            excos = traj["ex_cos"].mean(axis=1) if traj["ex_cos"].size \
                else np.full(n // 50, np.nan)
            exeps = traj["ex_eps"].mean(axis=1) if traj["ex_eps"].size \
                else np.full(n // 50, np.nan)
            ck = np.arange(len(excos)) * 50 + 50
            w_ = windows(n, e)
            ckw = windows(len(excos),
                          (e // 50 if e else None))
            row = dict(arm=arm, seed=seed,
                       fail=seed in fail_seeds[arm], entry=e)
            for name, sl in ckw.items():
                row[f"cos_{name}"] = float(np.nanmean(excos[sl]))
                row[f"eps_{name}"] = float(np.nanmean(exeps[sl]))
            for name, sl in w_.items():
                row[f"rn_{name}"] = float(traj["rnorm"][sl].mean())
                row[f"loss_{name}"] = float(losses[sl].mean())
            dphi = np.abs(np.diff(np.unwrap(np.angle(traj["w"]),
                                            axis=0), axis=0))
            row["dphi_med"] = float(np.median(dphi))
            row["cos_neg_frac"] = float(np.mean(excos < 0))
            row["cos_last_above_0.6"] = float(
                ck[np.nonzero(excos > 0.6)[0][-1]]
                if np.any(excos > 0.6) else 0.0)
            rows.append(row)

    print("-" * 86)
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        fr = [r for r in rs if r["fail"]]
        sr = [r for r in rs if not r["fail"]]
        print(f"[{arm}] failures {[r['seed'] for r in fr]} vs successes "
              f"{[r['seed'] for r in sr]}")
        for k in ["cos_early", "cos_mid", "cos_late", "cos_entry",
                  "eps_late", "rn_late", "cos_neg_frac",
                  "cos_last_above_0.6", "loss_late"]:
            mf = np.nanmedian([r[k] for r in fr]) if fr else float("nan")
            ms = np.nanmedian([r[k] for r in sr]) if sr else float("nan")
            print(f"   {k:<20s} fail {mf:+.4f}   success {ms:+.4f}")
    git = os.popen("git rev-parse HEAD").read().strip()
    doc = dict(git=git, rows=rows, fail_seeds=fail_seeds)
    with open(os.path.join(OUT, "ge_sig_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote ge_sig_summary.json")


if __name__ == "__main__":
    main()
