"""B4A -- exact identity check (convention check only).

Verifies, per natural pole/conjugate-pole channel lambda_j (j=0..2N-1,
the entries of F's diagonal), that

  rho_state[j] = sum_t c_t[j] x_t[j]      (x_t[j] = lambda_j x_{t-1}[j] + u_t)
  rho_lag[j]   = sum_k lambda_j^k (sum_t c_t[j] u_{t-k})

agree to floating-point accuracy, plus the frequency-domain equivalent
used by B3's R3. Convention note: this repo's established Hermitian
pairing (verified throughout Phase A/B2/B3 to reproduce BPTT) uses
conj(c_t), not plain c_t as in the handoff's schematic notation; both
rho_state and rho_lag below use conj(c_t) consistently with
credit_memory/lagcorr.py, so the identity check is convention-consistent
with everything upstream. No equations in lagcorr.py/hankel.py/teacher.py
are modified.

Run:  python -m credit_memory.phase_b4a_identity_check
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import (lagged_r_k, lag_decomposition_gradient,
                                   per_coordinate_contribution,
                                   freq_domain_g_p)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

N, T, BATCH = 6, 60, 8
SEEDS = list(range(5))


def main() -> None:
    print("=" * 78)
    print(f"Phase B4A: per-channel identity check (state vs lag vs freq), "
          f"N={N}, T={T}, BATCH={BATCH}")
    print("=" * 78)

    rows = []
    with set_l2_config(N, T, BATCH):
        for seed in SEEDS:
            params = tcg.init_params(seed)
            rng = np.random.RandomState(90000 + seed)
            x, r = draw_trajectory(params, rng, T, BATCH)
            out = compute_teacher(params, x, r)
            a1, B1 = out["a1"], out["B1"]
            f_diag = build_F(a1)
            d = np.ones(2 * N, np.complex128)

            for m in range(N):
                c_t = build_c_t(out["q1"], B1[:, m])           # (T,B,2N)
                u_t = out["Sa0"][:, :, m]                        # (T,B)

                # rho_state: direct per-channel pole filter + accumulate
                rho_state, _ = per_coordinate_contribution(f_diag, d,
                                                            c_t, u_t)
                # rho_lag: lag-decomposition re-derivation (independent
                # code path, credit_memory/lagcorr.py's r_k machinery)
                rk = lagged_r_k(c_t, u_t, K=T - 1)
                # here we want the PER-CHANNEL partial sums directly,
                # not the scalar G; lag_decomposition_gradient sums the
                # (r_k @ f_diag^k) SCALAR contraction, so recompute the
                # per-channel (elementwise, no contraction) version:
                fk = np.ones(2 * N, np.complex128)
                rho_lag = np.zeros(2 * N, np.complex128)
                for k in range(T):
                    rho_lag += rk[k] * fk
                    fk = fk * f_diag

                rho_freq = freq_domain_g_p(f_diag, d, u_t, c_t)

                err_lag = float(np.linalg.norm(rho_lag - rho_state)
                                / (np.linalg.norm(rho_state) + 1e-300))
                err_freq = float(np.linalg.norm(rho_freq - rho_state)
                                 / (np.linalg.norm(rho_state) + 1e-300))
                rows.append(dict(seed=seed, mode=m,
                                 state_vs_lag_rel_err=err_lag,
                                 state_vs_freq_rel_err=err_freq))
        for seed in SEEDS:
            errs = [row["state_vs_lag_rel_err"] for row in rows
                   if row["seed"] == seed]
            print(f"seed {seed}: max state-vs-lag rel_err = {max(errs):.3e}")

    max_lag_err = max(row["state_vs_lag_rel_err"] for row in rows)
    freq_errs = [row["state_vs_freq_rel_err"] for row in rows]
    all_lag_pass = all(row["state_vs_lag_rel_err"] < 1e-8 for row in rows)
    print("-" * 78)
    print(f"ALL state-vs-lag identity checks < 1e-8: {all_lag_pass}  "
          f"(max {max_lag_err:.2e})")
    print(f"state-vs-frequency-domain median rel disagreement: "
          f"{np.median(freq_errs):.4f}  (expected nontrivial, per B3's "
          f"circular-convolution caveat)")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS),
              rows=rows, all_state_vs_lag_pass=bool(all_lag_pass),
              max_state_vs_lag_rel_err=float(max_lag_err),
              median_state_vs_freq_rel_disagreement=float(np.median(
                  freq_errs)))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_b4a_identity_check_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
