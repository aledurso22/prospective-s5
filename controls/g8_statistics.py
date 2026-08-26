"""G8 — statistical cleanup of all primary small-scale comparisons.

15-seed data: online / PC0 / pcPhase (stored) + every G1 arm with a
15-seed row (polar / polar_gauge / pc0_adam as available). p-values are
reported to AVOID unsupported superiority claims, not as the main story.

Per pair (arm A vs B, 15 paired seeds):
  per-seed losses; medians; geometric means; median paired ratio;
  median paired log-loss difference log(L_A/L_B); exact two-sided sign
  test; Wilcoxon signed-rank on log loss; failure count L_A > L_online.

C3 extras (matched-headroom budgets, 5 seeds): at every budget K:
  sign test for BPTT+w vs BPTT; sign test for interaction I_i > 0;
  per-seed interaction values; Spearman correlation between
  (L_online - L_PC0) and (L_BPTT+w - L_BPTT) — does stronger credit
  repair accompany stronger miscalibration of exact credit?

Run:  python -m controls.g8_statistics
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")
SEEDS15 = list(range(15))


def load_finals():
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    F = {k: {int(s): v for s, v in c15["finals"][k].items()}
         for k in ("online", "pc0", "pcphase")}
    g1_path = os.path.join(OUT, "g1_summary.json")
    if os.path.exists(g1_path):
        g1 = json.load(open(g1_path))
        for arm, row in g1["arms"].items():
            if "finals15" in row:
                F[arm] = {int(s): v for s, v in row["finals15"].items()}
    return F


def pair_stats(name, A, B, F):
    a = np.array([F[A][s] for s in SEEDS15])
    b = np.array([F[B][s] for s in SEEDS15])
    ratios = a / b
    logdiff = np.log(ratios)
    wins = int(np.sum(a < b))
    sign_p = stats.binomtest(wins, len(a), 0.5,
                             alternative="two-sided").pvalue
    try:
        wil = stats.wilcoxon(logdiff).pvalue
    except ValueError:
        wil = float("nan")
    return dict(
        pair=f"{name}",
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        geomean_a=float(np.exp(np.mean(np.log(a)))),
        geomean_b=float(np.exp(np.mean(np.log(b)))),
        ratio_median=float(np.median(ratios)),
        logdiff_median=float(np.median(logdiff)),
        wins=f"{wins}/{len(a)}", sign_p=float(sign_p),
        wilcoxon_log_p=float(wil))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    F = load_finals()
    arms = [a for a in F if a != "online"]
    rows = []
    print("=" * 78)
    print("15-seed finals:")
    hdr = f"{'arm':<12s}" + "".join(f"{s:>8d}" for s in SEEDS15)
    print(hdr)
    for arm in F:
        print(f"{arm:<12s}" + "".join(f"{F[arm][s]:>8.3f}"
                                      for s in SEEDS15))
    for arm in arms:
        for base in ["online", "pc0", "pcphase"]:
            if base == arm:
                continue
            rows.append(pair_stats(f"{arm} vs {base}", arm, base, F))
    if "polar" in F and "polar_gauge" in F:
        rows.append(pair_stats("polar_gauge vs polar", "polar_gauge",
                               "polar", F))
    print("-" * 78)
    print(f"{'pair':<26s}{'medA':>8s}{'medB':>8s}{'geoA':>8s}{'geoB':>8s}"
          f"{'rat_med':>8s}{'log_med':>9s}{'wins':>6s}{'sign_p':>9s}"
          f"{'wil_p':>9s}")
    for r in rows:
        print(f"{r['pair']:<26s}{r['median_a']:>8.4f}{r['median_b']:>8.4f}"
              f"{r['geomean_a']:>8.4f}{r['geomean_b']:>8.4f}"
              f"{r['ratio_median']:>8.3f}{r['logdiff_median']:>9.3f}"
              f"{r['wins']:>6s}{r['sign_p']:>9.4f}{r['wilcoxon_log_p']:>9.4f}")
    print("failures (L_arm > L_online per seed):")
    for arm in arms:
        fails = [s for s in SEEDS15 if F[arm][s] > F["online"][s]]
        print(f"  {arm:<12s} {len(fails)}/15 {fails}")

    # ---------------- C3 extras ----------------
    c3 = json.load(open(os.path.join(ROOT, "results",
                                     "c3_matched_budget_bptt_w",
                                     "summary.json")))
    c3rows = []
    print("-" * 78)
    print("C3 matched budgets:")
    for tag, row in c3["rows"].items():
        if row is None:
            continue
        b = np.array([row["bptt"][str(s)] for s in range(5)])
        w = np.array([row["bptt_w"][str(s)] for s in range(5)])
        I = np.array([row["interaction"][str(s)] for s in range(5)])
        repair = np.array([row["online"][str(s)] - row["pc0"][str(s)]
                           for s in range(5)])
        miscal = w - b
        sign_bw = stats.binomtest(int(np.sum(w < b)), 5, 0.5,
                                  alternative="two-sided").pvalue
        sign_i = stats.binomtest(int(np.sum(I > 0)), 5, 0.5,
                                 alternative="two-sided").pvalue
        spear = stats.spearmanr(repair, miscal).statistic
        c3rows.append(dict(budget=tag, K=row["K"],
                           bptt_w_better=int(np.sum(w < b)),
                           sign_p_bptt=float(sign_bw),
                           I_positive=int(np.sum(I > 0)),
                           sign_p_interaction=float(sign_i),
                           spearman=float(spear),
                           interaction=[float(x) for x in I]))
        print(f"  K={row['K']:>4d} ({tag}): BPTT+w better "
              f"{int(np.sum(w < b))}/5 (sign p {sign_bw:.4f})  "
              f"I>0 {int(np.sum(I > 0))}/5 (sign p {sign_i:.4f})  "
              f"Spearman(repair, miscalibration) {spear:+.2f}  "
              f"I per seed {['%+.4f' % x for x in I]}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, finals={a: {str(s): F[a][s] for s in SEEDS15}
                                for a in F},
               pairs=rows, failures={a: [s for s in SEEDS15
                                         if F[a][s] > F["online"][s]]
                                     for a in arms},
               c3=c3rows)
    with open(os.path.join(OUT, "g8_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g8_summary.json")


if __name__ == "__main__":
    main()
