"""B3A -- verify the lagged eligibility-readout cross-correlation
decomposition of the exact causal-dual gradient, and report how quickly
it converges with lag.

x_t = F x_{t-1} + d u_t  =>  x_t = sum_k F^k d u_{t-k}
G = c_t^dagger x_t summed over t = sum_k r_k @ (F^k d),
r_k[p] := sum_t conj(c_t[p]) u_{t-k}

No BPTT used (u=Sa0[:, :, m], c_t built from q1, both forward-only). This
is a pure convention/decomposition check plus a decay-rate report.

Run:  python -m credit_memory.phase_b3a_lag_decomposition
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import lagged_r_k, lag_decomposition_gradient

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

N, T, BATCH = 6, 60, 8
SEEDS = list(range(5))


def main() -> None:
    print("=" * 78)
    print(f"Phase B3A: lagged cross-correlation decomposition, N={N}, "
          f"T={T}, BATCH={BATCH}")
    print("=" * 78)

    rows = []
    decay_curves = []
    with set_l2_config(N, T, BATCH):
        for seed in SEEDS:
            params = tcg.init_params(seed)
            rng = np.random.RandomState(80000 + seed)
            x, r = draw_trajectory(params, rng, T, BATCH)
            out = compute_teacher(params, x, r)
            a1, B1 = out["a1"], out["B1"]
            f_diag = build_F(a1)

            for m in range(N):
                c_t = build_c_t(out["q1"], B1[:, m])          # (T,B,2N)
                u_t = out["Sa0"][:, :, m]                      # (T,B)
                rk = lagged_r_k(c_t, u_t, K=T - 1)
                partial = lag_decomposition_gradient(f_diag, rk)
                G_full = partial[-1]
                G_causal_m = out["G_causal"][m]
                rel_err = float(abs(G_full - G_causal_m)
                               / (abs(G_causal_m) + 1e-300))

                frac = np.abs(partial) / (abs(G_causal_m) + 1e-300)
                # lag K needed for partial sum within 1%/5%/10% of the
                # FINAL (exact) value, relative to |G|
                errs = np.abs(partial - G_full) / (abs(G_full) + 1e-300)
                def lag_for(tol):
                    idx = np.where(errs < tol)[0]
                    return int(idx[0]) if len(idx) else -1
                lags = dict(tol_01=lag_for(0.01), tol_05=lag_for(0.05),
                           tol_10=lag_for(0.10))
                rows.append(dict(seed=seed, mode=m, rel_err=rel_err,
                                 lag_for_tolerance=lags))
                decay_curves.append(errs[:20].tolist())

            print(f"seed {seed}: median lag for 5% tol = "
                  f"{np.median([row['lag_for_tolerance']['tol_05'] for row in rows if row['seed'] == seed]):.1f}")

    all_pass = all(row["rel_err"] < 1e-8 for row in rows)
    lag5 = [row["lag_for_tolerance"]["tol_05"] for row in rows]
    lag10 = [row["lag_for_tolerance"]["tol_10"] for row in rows]
    print("-" * 78)
    print(f"ALL decomposition checks < 1e-8: {all_pass}")
    print(f"median lag for <5% error: {np.median(lag5):.1f}  "
          f"<10% error: {np.median(lag10):.1f}  (out of T-1={T - 1})")
    mean_decay = np.mean(decay_curves, axis=0)
    print("mean relative-error decay by lag K (first 10):",
          [round(v, 4) for v in mean_decay[:10]])

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS),
              rows=rows, all_pass=bool(all_pass),
              median_lag_for_5pct_tol=float(np.median(lag5)),
              median_lag_for_10pct_tol=float(np.median(lag10)),
              mean_relative_error_decay_by_lag=mean_decay.tolist())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b3a_lag_decomposition_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
