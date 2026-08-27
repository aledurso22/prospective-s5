"""B3D (principal-angle part) -- for each relevance rule's r=1 causal
subspace direction, compute the principal angle (= plain vector cosine
for 1-dim subspaces) to the exact-gradient-oracle's own implied
direction.

The oracle (B2D's L3, r=1: a free complex pole + free q1-readout, fit
against BPTT) has no direction in the original (P,Q) coordinate system a
priori -- its channel is defined by its own pole, not tied to any c_t.
To compare it to R0/R1/R2's genuine rank-1 subspaces, its channel's
BEST-FIT embedding direction v_oracle is recovered by ordinary least
squares regression of the true state x_t onto the oracle's own scalar
channel z_t, using calibration data:
  v_oracle = (sum_t x_t conj(z_t)) / (sum_t |z_t|^2)

The r=1 ladder table itself (cos/rel_err/norm_ratio/frac_gap_recovered
for R0-R3) is already reported in phase_b3c_relevance_summary.json; this
script adds only the principal-angle diagnostic on top, at r=1.

Run:  python -m credit_memory.phase_b3d_principal_angle
"""
from __future__ import annotations

import json
import os
import subprocess

import jax.numpy as jnp
import numpy as np

from credit_memory.hankel import (build_F, build_c_t, analytic_Wc,
                                  estimate_S, solve_Wo, balanced_transform)
from credit_memory.lagcorr import (lagged_r_k, per_coordinate_contribution,
                                   cross_gramian, top_r_eigen_reduction)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)
from credit_memory.phase_b2d_three_levels import fit_oracle, oracle_gradient

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def cos1(v1, v2):
    return float(np.abs(np.vdot(v2, v1))
                / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-300))


def oracle_direction(raw, m, cal_rows, f_diag, d):
    """Regress the true state x_t[:,m] onto the oracle's own r=1 channel
    z_t, using pooled calibration data."""
    num = np.zeros(2 * N, np.complex128)
    den = 0.0 + 0.0j
    for row in cal_rows:
        u_t = row["Sa0"][:, :, m]
        Tn, Bn = u_t.shape
        beta = np.asarray(oracle_beta(raw))[m, 0]
        z = np.zeros((Tn, Bn), np.complex128)
        prev = np.zeros(Bn, np.complex128)
        for t in range(Tn):
            prev = beta * prev + u_t[t]
            z[t] = prev
        xi = np.zeros((Tn, Bn, 2 * N), np.complex128)
        prevx = np.zeros((Bn, 2 * N), np.complex128)
        for t in range(Tn):
            prevx = f_diag[None, :] * prevx + u_t[t][:, None]
            xi[t] = prevx
        x_t = d[None, None, :] * xi
        num += np.einsum("tbp,tb->p", x_t, np.conj(z))
        den += np.sum(np.abs(z) ** 2)
    return num / (den + 1e-30)


def oracle_beta(raw):
    from credit_memory.phase_b2d_three_levels import sig
    rho = raw[:, 0:1]
    theta = raw[:, 1:2]
    return sig(rho) * jnp.exp(1j * theta)


def main() -> None:
    print("=" * 78)
    print("Phase B3D: principal angle, causal r=1 subspaces vs oracle")
    print("=" * 78)

    rows = []
    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)
        Wc = analytic_Wc(f_diag)
        d = np.ones(2 * N, np.complex128)

        q1_cal_pooled = np.concatenate(
            [row["q1"].reshape(-1, N) for row in cal_rows], axis=0)

        for m in range(N):
            raw_oracle, _ = fit_oracle(1, cal_rows, seed_key=3000 + seed
                                       * 10 + m)
            v_oracle = oracle_direction(raw_oracle, m, cal_rows, f_diag, d)

            S = estimate_S(q1_cal_pooled, B1[:, m])
            Wo = solve_Wo(f_diag, S)
            T_bal, Tinv_bal, _ = balanced_transform(Wc, Wo)
            v_r0 = T_bal[:, 0]

            c_t_pool = np.concatenate(
                [build_c_t(row["q1"], B1[:, m]) for row in cal_rows], axis=0)
            u_t_pool = np.concatenate(
                [row["Sa0"][:, :, m] for row in cal_rows], axis=0)
            g_p, _ = per_coordinate_contribution(f_diag, d, c_t_pool,
                                                 u_t_pool)
            top_p = int(np.argmax(np.abs(g_p)))
            v_r1 = np.zeros(2 * N, np.complex128); v_r1[top_p] = 1.0

            rk = lagged_r_k(c_t_pool, u_t_pool, K=c_t_pool.shape[0] - 1)
            M_cross = cross_gramian(f_diag, d, rk)
            _, _, V_r, _ = top_r_eigen_reduction(M_cross, np.diag(f_diag),
                                                 d, 1)
            v_r2 = V_r[:, 0]

            rows.append(dict(
                seed=seed, mode=m,
                cos_R0_vs_oracle=cos1(v_r0, v_oracle),
                cos_R1_vs_oracle=cos1(v_r1, v_oracle),
                cos_R2_vs_oracle=cos1(v_r2, v_oracle),
                top_coordinate_R1=top_p,
                is_top_coordinate_upper_or_lower=(
                    "P (upper mode j=%d)" % top_p if top_p < N else
                    "Q (upper mode j=%d)" % (top_p - N))))
        print(f"seed {seed}: median cos R1-vs-oracle="
              f"{np.median([r['cos_R1_vs_oracle'] for r in rows if r['seed'] == seed]):.3f}")

    print("-" * 78)
    for key in ("cos_R0_vs_oracle", "cos_R1_vs_oracle", "cos_R2_vs_oracle"):
        vals = [r[key] for r in rows]
        print(f"{key}: median={np.median(vals):.4f}  "
              f"mean={np.mean(vals):.4f}  min={np.min(vals):.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ),
              rows=rows,
              aggregate={key: dict(median=float(np.median(
                  [r[key] for r in rows])),
                  mean=float(np.mean([r[key] for r in rows])),
                  min=float(np.min([r[key] for r in rows])))
                  for key in ("cos_R0_vs_oracle", "cos_R1_vs_oracle",
                             "cos_R2_vs_oracle")})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b3d_principal_angle_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
