"""eps directional autocorrelation pass (Part 2 of prospective_kappa),
with per-layer splits. The 30-run kappa sweep completed and its κ=0
gate passed bitwise (see results/prospective_kappa_run.log); this pass
runs ONLY the eps-series measurement (deterministic, identical to the
registered Part 2) and assembles the final summary.

Reports lag-1 directional autocorrelation of eps_n = r_causal - r_exact,
decomposed radial/tangential by the current geometry:
    per layer, pooled lower layers (0..L-2), pooled all, per seed;
    full series and late half; plus norm/energy fractions per component.

Run:  python eps_perlayer.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import numpy as np

from diagnostics.prospective_kappa import setup, eps_series, SEEDS, RESULTS_DIR


def ac1_of(s):
    sd = np.std(s)
    if sd < 1e-12:
        return 0.0
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def eps_ac1_perlayer(eps):
    """per (layer, component): series of eps projected on e_r / e_phi."""
    er = {l: [] for l in range(4)}
    ep = {l: [] for l in range(4)}
    for (step, w, rB, rC) in eps:
        for l in range(4):
            u = w[l].real
            v = w[l].imag
            nrm = np.abs(w[l]) + 1e-12
            d = np.asarray([rc - rb for rc, rb in zip(rC[l], rB[l])])
            du, dv = d.real, d.imag
            er[l].append((u * du + v * dv) / nrm)
            ep[l].append((-v * du + u * dv) / nrm)

    def stats(series_l):
        # series_l: (T', N) real per layer
        half = series_l.shape[0] // 2
        per_mode = [ac1_of(series_l[:, j]) for j in range(series_l.shape[1])]
        per_mode_late = [ac1_of(series_l[half:, j])
                         for j in range(series_l.shape[1])]
        energy = float(np.mean(series_l ** 2))
        return dict(ac1=float(np.median(per_mode)),
                    ac1_late=float(np.median(per_mode_late)),
                    energy=energy)

    out = {}
    for comp, series in [("radial", er), ("tangential", ep)]:
        layers = {}
        for l in range(4):
            layers[l] = stats(np.asarray(series[l]))
        pooled_low = np.concatenate([np.asarray(series[l])
                                     for l in range(3)], axis=1)
        pooled_all = np.concatenate([np.asarray(series[l])
                                     for l in range(4)], axis=1)
        out[comp] = dict(per_layer={l: layers[l] for l in range(4)},
                         pooled_lower=stats(pooled_low),
                         pooled_all=stats(pooled_all))
    return out


def parse_kappa_finals():
    finals = {}
    cur = None
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "prospective_kappa_run.log")) as f:
        for line in f:
            m = re.match(r"kappa=([\d.]+) s(\d)\.\.\.", line)
            if m:
                cur = (float(m.group(1)), int(m.group(2)))
                continue
            m = re.match(r"\s*final ([\d.]+) finite", line)
            if m and cur is not None:
                k, s = cur
                finals.setdefault(k, {})[s] = float(m.group(1))
                cur = None
    return finals


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    finals = parse_kappa_finals()
    med = {k: float(np.median([finals[k][s] for s in SEEDS]))
           for k in finals}

    ac_rows = {}
    for seed in SEEDS:
        print(f"eps series s{seed}...", flush=True)
        eps = eps_series(seed)
        ac_rows[seed] = eps_ac1_perlayer(eps)
        pl = ac_rows[seed]["tangential"]["pooled_lower"]
        pr = ac_rows[seed]["radial"]["pooled_lower"]
        print(f"  pooled lower: rho_r(1) {pr['ac1']:+.3f}  "
              f"rho_phi(1) {pl['ac1']:+.3f}", flush=True)

    # ---- aggregate ----
    agg = {}
    for comp in ["radial", "tangential"]:
        agg[comp] = dict(
            per_layer={l: float(np.median(
                [ac_rows[s][comp]["per_layer"][l]["ac1"]
                 for s in SEEDS])) for l in range(4)},
            per_layer_late={l: float(np.median(
                [ac_rows[s][comp]["per_layer"][l]["ac1_late"]
                 for s in SEEDS])) for l in range(4)},
            pooled_lower=float(np.median(
                [ac_rows[s][comp]["pooled_lower"]["ac1"]
                 for s in SEEDS])),
            pooled_lower_late=float(np.median(
                [ac_rows[s][comp]["pooled_lower"]["ac1_late"]
                 for s in SEEDS])),
            pooled_all=float(np.median(
                [ac_rows[s][comp]["pooled_all"]["ac1"] for s in SEEDS])))
        # energy fraction of the tangential component (pooled lower)
    efrac = {}
    for seed in SEEDS:
        er_e = sum(ac_rows[seed]["radial"]["per_layer"][l]["energy"]
                   for l in range(3))
        ep_e = sum(ac_rows[seed]["tangential"]["per_layer"][l]["energy"]
                   for l in range(3))
        efrac[seed] = ep_e / (er_e + ep_e + 1e-30)
    print("-" * 78)
    print("kappa finals (parsed from the completed sweep):")
    for k in sorted(finals):
        print(f"  k={k!r:<5} {['%.4f' % finals[k][s] for s in SEEDS]}  "
              f"med {med[k]:.4f}")
    print(f"eps directional ac1, pooled lower layers (full / late):")
    for comp in ["radial", "tangential"]:
        print(f"  {comp:<11s} {agg[comp]['pooled_lower']:+.3f} / "
              f"{agg[comp]['pooled_lower_late']:+.3f}   per-layer: "
              f"{[round(agg[comp]['per_layer'][l], 3) for l in range(4)]}")
    print(f"tangential energy fraction (lower layers): "
          f"{np.median(list(efrac.values())):.2f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               kappa_finals={str(k): {str(s): finals[k][s]
                                      for s in finals[k]}
                             for k in finals},
               kappa_medians={str(k): med[k] for k in finals},
               k0_gate="bitwise (see run log)",
               eps=dict(per_seed=ac_rows, aggregate=agg,
                        tangential_energy_fraction=efrac))
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(conv(doc), f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
