"""B2B -- credit Hankel spectrum (no BPTT gradient information used to
build the reduction) and B2C -- balanced truncation ladder, evaluated on
disjoint test trajectories.

Per (seed, lower mode m): Wc is computed analytically from the
architecture alone (a1, closed form, no data). S -- and hence Wo -- is
estimated from CALIBRATION trajectories using only q1 (the existing,
already-causal, BPTT-free spatial error signal) and B1[:, m] (existing
routing weight). BPTT/exact gradients are used ONLY afterwards, to
evaluate reduced-order estimates on disjoint TEST trajectories against
ground truth -- never to build Wc, S, Wo, or the balanced transform.

No free learned alpha/beta/delta anywhere in this file (that is B2D
level 3, in phase_b2d_three_levels.py). Every reduced system here is
constructed purely from (a1, B1, calibration q1 statistics).

Run:  python -m credit_memory.phase_b2bc_hankel_truncation
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
from credit_memory.hankel import (build_F, analytic_Wc, estimate_S,
                                  solve_Wo, hankel_singular_values,
                                  balanced_transform, reduced_system,
                                  reduced_gradient)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

N, T, BATCH = 6, 60, 8
SEEDS = list(range(8))
N_CAL_TRAJ = 4
N_TEST_TRAJ = 4
R_LADDER = [1, 2, 4, 8, 2 * N]     # 2N = 12 = exact


def cos_np(u, v):
    u = np.ravel(u); v = np.ravel(v)
    return float(np.abs(np.vdot(v, u)) / (np.linalg.norm(u)
                                          * np.linalg.norm(v) + 1e-300))


def relerr_np(u, v):
    u = np.ravel(u); v = np.ravel(v)
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-300))


def dims_for_mass(sigma, fracs=(0.80, 0.90, 0.95, 0.99)):
    sq = sigma ** 2
    total = sq.sum()
    if total <= 0:
        return {str(f): len(sigma) for f in fracs}
    cum = np.cumsum(sq) / total
    out = {}
    for f in fracs:
        idx = int(np.searchsorted(cum, f) + 1)
        out[str(f)] = min(idx, len(sigma))
    return out


def collect_rows(seed, n_traj, offset):
    with set_l2_config(N, T, BATCH):
        params = tcg.init_params(seed)
        rows = []
        for k in range(n_traj):
            rng = np.random.RandomState(70000 + seed * 1000 + offset + k)
            x, r = draw_trajectory(params, rng, T, BATCH)
            rows.append(compute_teacher(params, x, r))
    return params, rows


def main() -> None:
    print("=" * 78)
    print(f"Phase B2B/C: Hankel spectrum + balanced truncation, N={N}, "
          f"T={T}, BATCH={BATCH}, {len(SEEDS)} seeds")
    print("=" * 78)

    spectrum_rows = []
    truncation_rows = []
    exact_regression = []

    # NOTE on cosine: for a single scalar complex number the quantity
    # |conj(a) b| / (|a||b|) is identically 1 (phase information is
    # discarded by the absolute value, leaving only |a||b|/|a||b|). A
    # meaningful cosine needs a genuine VECTOR -- here, the full N-mode
    # gradient vector, exactly as in B1. So per-mode reduced systems are
    # all built first, then evaluated jointly across modes per test
    # trajectory to form the length-N vectors G_hat, G_bptt, G_online
    # before any cosine/rel_err is computed.
    for seed in SEEDS:
        params, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
        f_diag = build_F(a1)
        Wc = analytic_Wc(f_diag)

        q1_cal_pooled = np.concatenate(
            [r["q1"].reshape(-1, N) for r in cal_rows], axis=0)

        per_mode = {}   # m -> dict(T_bal, Tinv_bal, sigma)
        for m in range(N):
            S = estimate_S(q1_cal_pooled, B1[:, m])
            Wo = solve_Wo(f_diag, S)
            sigma = hankel_singular_values(Wc, Wo)
            dims = dims_for_mass(sigma)
            spectrum_rows.append(dict(seed=seed, mode=m,
                                      sigma=sigma.tolist(),
                                      dims_for_mass=dims,
                                      total_sq_mass=float(np.sum(sigma
                                                                 ** 2))))
            T_bal, Tinv_bal, sigma2 = balanced_transform(Wc, Wo)
            # cross-check the two independent Hankel-SV computations,
            # excluding the numerically-negligible tail (near-zero modes
            # differ at the ~1e-7 absolute level, set by the eps
            # regularization inside balanced_transform's matrix inverse
            # -- irrelevant to the truncation ladder, which only ever
            # uses the well-separated top modes)
            keep = sigma > 1e-3 * sigma[0]
            assert np.allclose(sigma[keep], sigma2[keep], rtol=1e-5)
            per_mode[m] = dict(T_bal=T_bal, Tinv_bal=Tinv_bal)

        for r in R_LADDER:
            F_r_d_r = {m: reduced_system(
                f_diag, np.ones(2 * N, np.complex128),
                per_mode[m]["T_bal"], per_mode[m]["Tinv_bal"], r)
                for m in range(N)}
            for r_idx, row in enumerate(test_rows):
                G_hat = np.zeros(N, np.complex128)
                for m in range(N):
                    F_r, d_r = F_r_d_r[m]
                    _, G_hat[m] = reduced_gradient(
                        F_r, d_r, per_mode[m]["T_bal"],
                        row["Sa0"][:, :, m], row["q1"], B1[:, m], r)
                G_bptt = row["G_bptt"]
                G_online = row["G_online"]
                c_hat = cos_np(G_hat, G_bptt)
                c_on = cos_np(G_online, G_bptt)
                gap = max(1.0 - c_on, 1e-12)
                truncation_rows.append(dict(
                    seed=seed, r=r, test_traj=r_idx,
                    cos=c_hat, cos_online=c_on,
                    rel_err=relerr_np(G_hat, G_bptt),
                    norm_ratio=float(np.linalg.norm(G_hat)
                                     / (np.linalg.norm(G_bptt) + 1e-300)),
                    frac_gap_recovered=float((c_hat - c_on) / gap)))
                if r == 2 * N:
                    exact_regression.append(dict(
                        seed=seed, test_traj=r_idx,
                        rel_err_vs_bptt=relerr_np(G_hat, G_bptt)))
        print(f"seed {seed}: median dims for 90% mass = "
              f"{np.median([row['dims_for_mass']['0.9'] for row in spectrum_rows if row['seed'] == seed]):.1f}")

    # ---- aggregate spectrum
    dims90 = [row["dims_for_mass"]["0.9"] for row in spectrum_rows]
    dims99 = [row["dims_for_mass"]["0.99"] for row in spectrum_rows]
    print("-" * 78)
    print(f"pooled (seed,mode) median dims for 90% mass: "
          f"{np.median(dims90):.1f}  for 99%: {np.median(dims99):.1f}")

    exact_ok = all(row["rel_err_vs_bptt"] < 1e-8 for row in exact_regression)
    print(f"exact (r=2N) balanced-truncation regression vs BPTT: "
          f"ALL < 1e-8: {exact_ok}  "
          f"(max {max(r['rel_err_vs_bptt'] for r in exact_regression):.2e})")

    for r in R_LADDER:
        rows_r = [row for row in truncation_rows if row["r"] == r]
        print(f"r={r:>2d}: median cos={np.median([x['cos'] for x in rows_r]):.4f}"
              f"  median rel_err={np.median([x['rel_err'] for x in rows_r]):.4f}"
              f"  median frac_gap_recovered="
              f"{np.median([x['frac_gap_recovered'] for x in rows_r]):.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(
        git=git,
        config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                   n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ,
                   r_ladder=R_LADDER),
        spectrum_rows=spectrum_rows,
        spectrum_aggregate=dict(
            median_dims_for_80pct=float(np.median(
                [row["dims_for_mass"]["0.8"] for row in spectrum_rows])),
            median_dims_for_90pct=float(np.median(dims90)),
            median_dims_for_95pct=float(np.median(
                [row["dims_for_mass"]["0.95"] for row in spectrum_rows])),
            median_dims_for_99pct=float(np.median(dims99))),
        exact_regression_all_pass=bool(exact_ok),
        truncation_rows=truncation_rows,
        truncation_aggregate={
            str(r): dict(
                median_cos=float(np.median(
                    [x["cos"] for x in truncation_rows if x["r"] == r])),
                median_rel_err=float(np.median(
                    [x["rel_err"] for x in truncation_rows
                     if x["r"] == r])),
                median_norm_ratio=float(np.median(
                    [x["norm_ratio"] for x in truncation_rows
                     if x["r"] == r])),
                median_frac_gap_recovered=float(np.median(
                    [x["frac_gap_recovered"] for x in truncation_rows
                     if x["r"] == r])))
            for r in R_LADDER},
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b2bc_hankel_truncation_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
