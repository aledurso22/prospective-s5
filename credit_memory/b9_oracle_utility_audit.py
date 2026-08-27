"""B9: state-lifecycle audit (Part 1/2, code-analysis only, no new
computation needed there -- see PHASE_B9.md) plus an OFFLINE ORACLE
DIAGNOSTIC (Part 3/4). Does NOT change the training algorithm, does NOT
implement prediction-correction/resurrection, does NOT launch S5.

For each candidate upper-layer channel j (j=0..2N-1, matching every
prior phase's {a1[k], conj(a1[k])} convention) and each lower mode m,
computes the EXACT per-candidate contribution to the full causal
gradient:

  gamma_j[m] := conj(c_t[j]) . x_t[j] summed over t   (credit_memory/
                                                        lagcorr.py's own
                                                        per_coordinate_
                                                        contribution,
                                                        UNMODIFIED --
                                                        sum_j gamma_j[m]
                                                        == G_causal[m]
                                                        exactly, already
                                                        verified through
                                                        B3/B4/B7)

For the CURRENT active set S (here: the rank-1 selected channel, S =
{top_j}), computes the oracle marginal utility

  U_j(S) = ||G-G_S||^2 - ||G-G_S-gamma_j||^2

and verifies it against the algebraic identity

  U_j(S) = 2 Re[ conj(G-G_S) . gamma_j ] - ||gamma_j||^2

Also logs, per candidate/mode: |lambda_j| (pole magnitude), |q_j|
(upper-mode naive error magnitude), ||B_j,:|| (routing-weight row norm),
lower-eligibility energy E_m, and lag-1 eligibility autocorrelation --
all of these are ALREADY available causally (no BPTT, no full P/Q
teacher) without any dormant-state machinery.

Same 8 seeds, N/T/BATCH, calibration/test split as B3/B4 (imported
directly for a self-contained, directly-comparable artifact).

Run:  python -m credit_memory.b9_oracle_utility_audit
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.streaming import StreamingRelevance
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


def lag1_autocorr(u_t):
    """u_t: (T, BATCH) complex -> scalar lag-1 autocorrelation, pooled
    over batch, of the (real,imag)-stacked signal (a cheap, standard
    persistence diagnostic; not used by any training arm)."""
    flat = np.concatenate([u_t.real.ravel(), u_t.imag.ravel()])
    x0 = np.concatenate([u_t[:-1].real.ravel(), u_t[:-1].imag.ravel()])
    x1 = np.concatenate([u_t[1:].real.ravel(), u_t[1:].imag.ravel()])
    if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def main() -> None:
    print("=" * 90)
    print(f"Phase B9: oracle marginal-utility audit, {len(SEEDS)} seeds")
    print("=" * 90)

    rows = []
    identity_errs = []
    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)
        d = np.ones(2 * N, np.complex128)

        q1_cal_pooled = np.concatenate(
            [row["q1"].reshape(-1, N) for row in cal_rows], axis=0)

        for m in range(N):
            c_t_pool = np.concatenate(
                [build_c_t(row["q1"], B1[:, m]) for row in cal_rows], axis=0)
            u_t_pool = np.concatenate(
                [row["Sa0"][:, :, m] for row in cal_rows], axis=0)

            # exact per-candidate contribution gamma_j[m]; sum_j == G_causal[m]
            gamma, _ = per_coordinate_contribution(f_diag, d, c_t_pool,
                                                    u_t_pool)
            G = np.sum(gamma)   # exact full-bank gradient for this mode

            # existing (calibration-based) |rho_j| ranking, reused verbatim
            est = StreamingRelevance(f_diag, BATCH, mode="windowed")
            for row in cal_rows:
                u_traj = row["Sa0"][:, :, m]
                c_traj = build_c_t(row["q1"], B1[:, m])
                for t in range(u_traj.shape[0]):
                    est.step(u_traj[t], c_traj[t])
            rho = est.rho
            top_j = int(np.argmax(np.abs(rho)))

            G_S = gamma[top_j]     # active set S = {top_j}
            resid = G - G_S

            U = np.zeros(2 * N)
            U_identity = np.zeros(2 * N)
            for j in range(2 * N):
                gj = gamma[j]
                U[j] = np.abs(resid) ** 2 - np.abs(resid - gj) ** 2
                U_identity[j] = 2 * np.real(np.conj(resid) * gj) \
                    - np.abs(gj) ** 2
            id_err = float(np.max(np.abs(U - U_identity)))
            identity_errs.append(id_err)

            # rank correlation: |rho_j| ranking vs oracle U_j(S) ranking
            spearman = stats.spearmanr(np.abs(rho), U).statistic
            top1_agree = bool(np.argmax(U) == top_j)

            # supplementary: U_j(S) is degenerate/self-referential for
            # j == top_j (S already contains it); also report the
            # correlation restricted to the 2N-1 candidates NOT already
            # in S, which is the more meaningful "would a different
            # choice have been better" comparison.
            mask = np.ones(2 * N, dtype=bool)
            mask[top_j] = False
            spearman_excl = stats.spearmanr(np.abs(rho)[mask],
                                            U[mask]).statistic

            # Part 4: cheap, already-available diagnostics
            j_orig = np.arange(N) % N   # upper mode index for each of 2N
                                        # candidates (P-branch 0..N-1,
                                        # Q-branch N..2N-1 share the SAME
                                        # upper-mode index)
            abs_lambda = np.abs(f_diag)                       # (2N,)
            q1_upper = q1_cal_pooled                           # (n, N)
            abs_q = np.array([np.sqrt(np.mean(np.abs(
                q1_upper[:, j % N]) ** 2)) for j in range(2 * N)])
            B_row_norm = np.array([np.linalg.norm(B1[j % N, :])
                                   for j in range(2 * N)])
            E_m = float(np.sum(np.abs(u_t_pool) ** 2))
            ac1 = lag1_autocorr(u_t_pool)

            rows.append(dict(
                seed=seed, mode=m, top_j=top_j,
                U_j=U.tolist(), U_identity_check_max_err=id_err,
                abs_rho=np.abs(rho).tolist(),
                spearman_rho_vs_U=float(spearman)
                if not np.isnan(spearman) else None,
                spearman_rho_vs_U_excl_top=float(spearman_excl)
                if not np.isnan(spearman_excl) else None,
                top1_agree_rho_vs_oracle=top1_agree,
                abs_lambda_per_candidate=abs_lambda.tolist(),
                abs_q_per_candidate=abs_q.tolist(),
                B_row_norm_per_candidate=B_row_norm.tolist(),
                lower_eligibility_energy_E_m=E_m,
                lag1_autocorr_Sa0=ac1))
        print(f"seed {seed}: median spearman(|rho|,U)="
              f"{np.median([r['spearman_rho_vs_U'] for r in rows if r['seed'] == seed and r['spearman_rho_vs_U'] is not None]):.3f}"
              f"  top1 agree rate="
              f"{np.mean([r['top1_agree_rho_vs_oracle'] for r in rows if r['seed'] == seed]):.2f}")

    print("-" * 90)
    max_identity_err = max(identity_errs)
    print(f"Identity U_j(S) = 2Re[conj(resid).gamma_j] - |gamma_j|^2: "
         f"max abs err across all rows = {max_identity_err:.2e} "
         f"({'PASS' if max_identity_err < 1e-9 else 'FAIL'})")

    spearmans = [r["spearman_rho_vs_U"] for r in rows
                if r["spearman_rho_vs_U"] is not None]
    spearmans_excl = [r["spearman_rho_vs_U_excl_top"] for r in rows
                      if r["spearman_rho_vs_U_excl_top"] is not None]
    top1_rate = np.mean([r["top1_agree_rho_vs_oracle"] for r in rows])
    print(f"median Spearman(|rho_j|, oracle U_j) over {len(spearmans)} "
         f"(seed,mode) rows: {np.median(spearmans):.3f}")
    print(f"median Spearman excluding the already-selected top_j "
         f"(2N-1 candidates): {np.median(spearmans_excl):.3f}")
    print(f"|rho|-argmax matches oracle-argmax on {top1_rate * 100:.1f}% "
         f"of (seed,mode) rows")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                         n_cal_traj=N_CAL_TRAJ),
              rows=rows, max_identity_err=max_identity_err,
              identity_check_pass=bool(max_identity_err < 1e-9),
              median_spearman=float(np.median(spearmans)),
              median_spearman_excl_top=float(np.median(spearmans_excl)),
              top1_agreement_rate=float(top1_rate))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_oracle_utility_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
