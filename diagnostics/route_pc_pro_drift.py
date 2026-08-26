"""Drift-vs-noise audit — the registered pre-measurement, done on the
COMPLEX VECTOR residual series (the magnitude-series version in the main
run is confounded: mean(|dr|) is trivially positive and its
autocorrelation reflects amplitude decay, not drift-direction
persistence).

For kappa = 0 (the unmodified residual stream), stationary and moving
regimes, 5 seeds: record r^_n (complex, concatenated over layers/modes)
and d r^_n = r^_n - r^_{n-1} per step, then measure on the late window:

  * systematic vector drift: ||E[dr]|| against the dimension-aware noise
    floor RMS(dr - E dr) * sqrt(D_late / N)  -> t_vec =
    ||E dr|| sqrt(N) / (RMS(fluct) sqrt(dim));
  * complex lag-1 autocorrelation of dr^ (direction persistence):
    |sum dr_n conj(dr_{n-1})| / sum |dr|^2;
  * the same two statistics for r^ itself (base autocorrelation the
    prospective term would exploit).

Run:  python route_pc_pro_drift.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from diagnostics import route_pc_pro as pp
from diagnostics.route_pc_pro import SEEDS, RESULTS_DIR


def stats(R, D):
    half = R.shape[0] // 2
    R, D = R[half:], D[half:]
    N, dim = D.shape

    def vec_t(X):
        mu = X.mean(axis=0)
        fl = X - mu
        rms = float(np.sqrt(np.mean(np.abs(fl) ** 2)))
        return float(np.linalg.norm(mu) * np.sqrt(len(X))
                     / (rms * np.sqrt(dim) + 1e-30)), \
            float(np.linalg.norm(mu) / (np.sqrt(np.mean(
                np.abs(X) ** 2)) + 1e-30))

    def ac1(X):
        num = np.abs(np.sum(X[1:] * np.conj(X[:-1])))
        den = np.sum(np.abs(X) ** 2)
        return float(num / (den + 1e-30))

    t_dr, rel_dr = vec_t(D)
    t_r, rel_r = vec_t(R)
    return dict(t_vec_dr=t_dr, rel_drift_dr=rel_dr, ac1_dr=ac1(D),
                t_vec_r=t_r, rel_drift_r=rel_r, ac1_r=ac1(R),
                n_late=int(N))


def main() -> None:
    pp.setup()
    out = {}
    for moving in [False, True]:
        reg = "moving" if moving else "stationary"
        rows = []
        for seed in SEEDS:
            print(f"{reg} s{seed}: recording complex residual series...",
                  flush=True)
            o = pp.train_pro(seed, 0.0, moving=moving, record_complex=True)
            R = np.asarray(o["series"]["rhat_c"])
            D = np.asarray(o["series"]["dr_c"])
            s = stats(R, D)
            s["seed"] = seed
            rows.append(s)
            print(f"  t_vec(dr) {s['t_vec_dr']:.2f}  ac1(dr) "
                  f"{s['ac1_dr']:+.3f}  t_vec(r) {s['t_vec_r']:.2f}  "
                  f"ac1(r) {s['ac1_r']:+.3f}", flush=True)
        out[reg] = rows
        print(f"  {reg} medians: t_vec(dr) "
              f"{np.median([r['t_vec_dr'] for r in rows]):.2f}  "
              f"ac1(dr) {np.median([r['ac1_dr'] for r in rows]):+.3f}  "
              f"ac1(r) {np.median([r['ac1_r'] for r in rows]):+.3f}")
    with open(os.path.join(RESULTS_DIR, "drift_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote drift_audit.json")


if __name__ == "__main__":
    main()
