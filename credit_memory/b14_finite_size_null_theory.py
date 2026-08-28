"""Phase B14 -- finite-size/random-subspace null theory: is the observed
r_tc~2 actually anomalous, or the ordinary finite-dimensional spectrum
of the overlap between two small (Haar-)random temporal subspaces?
Theory/mechanism only: no new training algorithm, no S5.

After whitening, C_tc = Q_V^dagger Q_U (Q_V, Q_U orthonormal bases for
V's/U's temporal subspaces). For independent Haar-random p- and
q-dimensional subspaces of a T-dimensional ambient space, the squared
singular values of Q_V^dagger Q_U are squared canonical correlations
following the classical Jacobi/MANOVA random-subspace ensemble --
finite-dimensional Monte Carlo is used as the authoritative baseline
here (per the task's own instruction), not an asymptotic formula.

Run:  python -m credit_memory.b14_finite_size_null_theory
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F
from credit_memory.b9_2_shared_pool import best_pool_exact
from credit_memory.b10_tangent_adjoint_theory import (
    low_rank_trunc, decision_metrics, effective_ranks, direct_routed,
    build_factors, algebraic_rank)
from credit_memory.b10_1_temporal_coupling import svd_compact
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b14")

K = 4
N_NULL_DRAWS = 300


# ===========================================================================
# Core: Haar-random subspace overlap generator and spectral metrics
# ===========================================================================
def random_orthonormal_basis(T_ambient, p, rng):
    X = rng.randn(T_ambient, p) + 1j * rng.randn(T_ambient, p)
    Qm, _ = np.linalg.qr(X)
    return Qm


def haar_null_draw(T_ambient, p, q, rng):
    Qv = random_orthonormal_basis(T_ambient, p, rng)
    Qu = random_orthonormal_basis(T_ambient, q, rng)
    C = np.conj(Qv).T @ Qu
    return np.linalg.svd(C, compute_uv=False)


def spectral_metrics(sv):
    sv = np.asarray(sv)
    sq = sv ** 2
    total = sq.sum()
    cum = np.cumsum(sq) / (total + 1e-300)
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    r95 = int(np.searchsorted(cum, 0.95) + 1)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    stable_rank = float(total / (sv[0] ** 2 + 1e-300)) if len(sv) else 0.0
    p_norm = sq / (total + 1e-300)
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-300)))
    entropy_rank = float(np.exp(entropy))
    return dict(sv=sv.tolist(), top1_frac=float(sq[0] / (total + 1e-300)),
               top2_frac=float(sq[:2].sum() / (total + 1e-300)),
               r90=r90, r95=r95, r99=r99, stable_rank=stable_rank,
               entropy_rank=entropy_rank,
               sigma2_over_sigma1=float(sv[1] / sv[0]) if len(sv) > 1 else None,
               sigma3_over_sigma2=float(sv[2] / sv[1]) if len(sv) > 2 else None,
               frobenius_norm=float(np.sqrt(total)),
               largest_canonical_corr=float(sv[0]))


def compare_to_null(real_sv, null_sv_matrix):
    """null_sv_matrix: (n_draws, n_components). Returns percentile of
    each metric plus per-rank sequential comparison for r_excess."""
    real_metrics = spectral_metrics(real_sv)
    null_metrics_list = [spectral_metrics(null_sv_matrix[d])
                         for d in range(null_sv_matrix.shape[0])]

    def null_stats(key):
        vals = np.array([m[key] for m in null_metrics_list if m[key] is not None])
        real_val = real_metrics[key]
        if real_val is None or len(vals) == 0:
            return None
        pctl = float(np.mean(vals <= real_val))
        return dict(real=real_val, null_median=float(np.median(vals)),
                   null_p5=float(np.percentile(vals, 5)),
                   null_p95=float(np.percentile(vals, 95)),
                   null_p1=float(np.percentile(vals, 1)),
                   null_p99=float(np.percentile(vals, 99)),
                   percentile=pctl)

    comparisons = {key: null_stats(key) for key in
                  ("top1_frac", "top2_frac", "r90", "r95", "r99",
                   "stable_rank", "entropy_rank", "sigma2_over_sigma1",
                   "sigma3_over_sigma2", "frobenius_norm", "largest_canonical_corr")}

    # sequential r_excess: how many ranked real singular values exceed
    # the null's per-rank 95%/99% quantile
    n_comp = null_sv_matrix.shape[1]
    r_excess_95, r_excess_99 = 0, 0
    per_rank = []
    for i in range(min(n_comp, len(real_sv))):
        null_i = null_sv_matrix[:, i]
        q95, q99 = np.percentile(null_i, 95), np.percentile(null_i, 99)
        exceeds95 = real_sv[i] > q95
        exceeds99 = real_sv[i] > q99
        if exceeds95 and i == r_excess_95:
            r_excess_95 += 1
        if exceeds99 and i == r_excess_99:
            r_excess_99 += 1
        per_rank.append(dict(rank=i, real=float(real_sv[i]),
                            null_q95=float(q95), null_q99=float(q99),
                            exceeds95=bool(exceeds95), exceeds99=bool(exceeds99)))
    return dict(real_metrics=real_metrics, comparisons=comparisons,
               r_excess_95=r_excess_95, r_excess_99=r_excess_99,
               per_rank=per_rank)


def shape_r_excess(comparisons):
    """Complementary to the magnitude-based sequential r_excess: counts
    how many CONCENTRATION metrics (r90 low-tail, top2_frac high-tail)
    are outside the null's 95%/99% interval -- distinguishes "are
    individual correlations unusually large" (r_excess_95/99 above)
    from "is the SHAPE of the spectrum unusually concentrated"."""
    out = {}
    r90c = comparisons["r90"]
    out["r90_below_null_p5"] = bool(r90c["real"] < r90c["null_p5"])
    out["r90_below_null_p1"] = bool(r90c["real"] < r90c["null_p1"])
    t2c = comparisons["top2_frac"]
    out["top2_above_null_p95"] = bool(t2c["real"] > t2c["null_p95"])
    out["top2_above_null_p99"] = bool(t2c["real"] > t2c["null_p99"])
    return out


# ===========================================================================
# PART A: dimension/rank bookkeeping (reported directly in main())
# ===========================================================================
def part_a_dims(U, V0):
    Lu, Su, Ruh = svd_compact(U)
    Lv, Sv, Rvh = svd_compact(V0)
    return dict(T_ambient=U.shape[0], M=2 * N, N_lower=N,
               algebraic_rank_U=algebraic_rank(U), algebraic_rank_V=algebraic_rank(V0),
               er_U=effective_ranks(U), er_V=effective_ranks(V0),
               p_dim_V=Rvh.shape[0], q_dim_U=Lu.shape[1])


# ===========================================================================
# PART E: weighted (K_tc-style) null -- real Sigma_U, Sigma_V, random
# temporal orientation.
# ===========================================================================
def part_e_weighted_null(U, V0, n_draws=N_NULL_DRAWS, seed=0):
    Lu, Su, Ruh = svd_compact(U)
    Lv, Sv, Rvh = svd_compact(V0)
    K_tc_real = np.diag(Sv) @ (Rvh @ Lu) @ np.diag(Su)
    sv_real = np.linalg.svd(K_tc_real, compute_uv=False)

    rng = np.random.RandomState(seed)
    T_ambient = U.shape[0]
    p, q = Rvh.shape[0], Lu.shape[1]
    null_sv = np.zeros((n_draws, min(p, q)))
    for d in range(n_draws):
        Qv = random_orthonormal_basis(T_ambient, p, rng)
        Qu = random_orthonormal_basis(T_ambient, q, rng)
        C_null_mat = np.conj(Qv).T @ Qu
        K_null = np.diag(Sv) @ C_null_mat @ np.diag(Su)
        null_sv[d] = np.linalg.svd(K_null, compute_uv=False)
    cmp = compare_to_null(sv_real, null_sv)
    return cmp


# ===========================================================================
# PART F: small-matrix random geometry map (pure math, no real model)
# ===========================================================================
def part_f_geometry_map(p_list, q_list, T_list, n_draws=100, seed=0):
    rng = np.random.RandomState(seed)
    out = {}
    for T_ in T_list:
        out[str(T_)] = {}
        for p_ in p_list:
            for q_ in q_list:
                svs = np.array([haar_null_draw(T_, p_, q_, rng) for _ in range(n_draws)])
                med_metrics = [spectral_metrics(svs[d]) for d in range(n_draws)]
                out[str(T_)][f"{p_}_{q_}"] = dict(
                    median_r90=float(np.median([m["r90"] for m in med_metrics])),
                    median_top2_frac=float(np.median([m["top2_frac"] for m in med_metrics])),
                    median_stable_rank=float(np.median([m["stable_rank"] for m in med_metrics])),
                    median_entropy_rank=float(np.median([m["entropy_rank"] for m in med_metrics])))
    return out


# ===========================================================================
# PART G/H: width and T scaling -- the decisive structural experiment.
# Reuses build_factors/direct_routed at DIFFERENT N (width) or T,
# always via the module-level N/T patched in phase_b2bc_hankel_truncation
# and credit_memory.teacher's set_l2_config context.
# ===========================================================================
def collect_rows_width(seed, N_, T_, BATCH_, n_traj):
    from toyrig import ssm_rig as tcg
    from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
    with set_l2_config(N_, T_, BATCH_):
        params = tcg.init_params(seed)
        rows = []
        for k in range(n_traj):
            rng = np.random.RandomState(70000 + seed * 1000 + k)
            x, r = draw_trajectory(params, rng, T_, BATCH_)
            rows.append(compute_teacher(params, x, r))
    return params, rows


def build_factors_width(rows, a1, N_):
    from credit_memory.b10_tangent_adjoint_theory import adjoint_filter
    U_blocks = {m: [] for m in range(N_)}
    V0_blocks = {j: [] for j in range(N_)}
    for row in rows:
        Sa0, q1 = row["Sa0"], row["q1"]
        for m in range(N_):
            U_blocks[m].append(Sa0[:, :, m].reshape(-1))
        for j in range(N_):
            p_P = adjoint_filter(a1[j], q1[:, :, j])
            V0_blocks[j].append(np.conj(p_P).reshape(-1))
    U = np.stack([np.concatenate(U_blocks[m]) for m in range(N_)], axis=1)
    V0 = np.stack([np.concatenate(V0_blocks[j]) for j in range(N_)], axis=0)
    return U, V0


def direct_routed_width(rows, f_diag, B1, N_):
    from credit_memory.hankel import build_c_t
    from credit_memory.b10_tangent_adjoint_theory import forward_filter
    n2 = 2 * N_
    rho = np.zeros((n2, N_), np.complex128)
    for m in range(N_):
        for row in rows:
            c_full = build_c_t(row["q1"], row["B1"][:, m])
            u_t = row["Sa0"][:, :, m]
            for j in range(n2):
                x_j = forward_filter(f_diag[j], u_t)
                rho[j, m] += np.sum(np.conj(c_full[:, :, j]) * x_j)
    return rho


def K_epsilon_curve(S, N_, eps_fracs=(0.05, 0.10)):
    F_full = float(sum(S[:, m].max() for m in range(N_)))
    out = {}
    n2 = S.shape[0]
    for eps in eps_fracs:
        eps_abs = eps * F_full
        mk = n2
        for Kc in range(1, n2 + 1):
            Pc = best_pool_exact(S, Kc)
            achieved = float(sum(S[list(Pc), m].max() for m in range(N_)))
            if F_full - achieved <= eps_abs:
                mk = Kc
                break
        out[str(eps)] = mk
    return out


def part_g_width_scaling(N_list, seeds, T_=T, BATCH_=BATCH, n_traj=N_CAL_TRAJ,
                         n_null_draws=100):
    out = {}
    for N_ in N_list:
        per_seed = []
        for seed in seeds:
            params, rows = collect_rows_width(seed, N_, T_, BATCH_, n_traj)
            a1, B1 = rows[0]["a1"], rows[0]["B1"]
            f_diag = build_F(a1)
            U, V0 = build_factors_width(rows, a1, N_)
            Lu, Su, Ruh = svd_compact(U)
            Lv, Sv, Rvh = svd_compact(V0)
            C_tc = Rvh @ Lu
            sv_C = np.linalg.svd(C_tc, compute_uv=False)
            K_tc = np.diag(Sv) @ C_tc @ np.diag(Su)
            sv_K = np.linalg.svd(K_tc, compute_uv=False)

            T_ambient = U.shape[0]
            p, q = Rvh.shape[0], Lu.shape[1]
            rng = np.random.RandomState(9000 + seed)
            null_sv_C = np.array([haar_null_draw(T_ambient, p, q, rng)
                                  for _ in range(n_null_draws)])
            cmp_C = compare_to_null(sv_C, null_sv_C)

            null_sv_K = np.zeros((n_null_draws, min(p, q)))
            for d in range(n_null_draws):
                Qv = random_orthonormal_basis(T_ambient, p, rng)
                Qu = random_orthonormal_basis(T_ambient, q, rng)
                C_null_mat = np.conj(Qv).T @ Qu
                Kn = np.diag(Sv) @ C_null_mat @ np.diag(Su)
                null_sv_K[d] = np.linalg.svd(Kn, compute_uv=False)
            cmp_K = compare_to_null(sv_K, null_sv_K)

            R = direct_routed_width(rows, f_diag, B1, N_)
            S = np.abs(R)
            keps = K_epsilon_curve(S, N_)

            per_seed.append(dict(
                algebraic_rank_U=algebraic_rank(U), algebraic_rank_V=algebraic_rank(V0),
                er_U90=effective_ranks(U)["0.9"], er_V90=effective_ranks(V0)["0.9"],
                r_tc_raw=effective_ranks(C_tc)["0.9"],
                r_tc_K=effective_ranks(K_tc)["0.9"],
                r_excess_95_C=cmp_C["r_excess_95"], r_excess_95_K=cmp_K["r_excess_95"],
                null_r90_C=cmp_C["comparisons"]["r90"]["null_median"],
                null_r90_K=cmp_K["comparisons"]["r90"]["null_median"],
                K_epsilon=keps, K_epsilon_over_M=keps["0.05"] / (2 * N_)))
        out[str(N_)] = per_seed
    return out


def main() -> None:
    print("=" * 90)
    print(f"Phase B14: finite-size/random-subspace null theory, {len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(a=[], b_C=[], e_K=[])
    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1 = rows[0]["a1"]
        U, V0_P, V0_Q = build_factors(rows, a1)
        per_seed["a"].append(part_a_dims(U, V0_P))

        Lu, Su, Ruh = svd_compact(U)
        Lv, Sv, Rvh = svd_compact(V0_P)
        C_tc = Rvh @ Lu
        sv_C = np.linalg.svd(C_tc, compute_uv=False)
        T_ambient = U.shape[0]
        p, q = Rvh.shape[0], Lu.shape[1]
        rng = np.random.RandomState(5000 + seed)
        null_sv_C = np.array([haar_null_draw(T_ambient, p, q, rng)
                              for _ in range(N_NULL_DRAWS)])
        cmp_C = compare_to_null(sv_C, null_sv_C)
        cmp_C["shape_flags"] = shape_r_excess(cmp_C["comparisons"])
        per_seed["b_C"].append(cmp_C)

        per_seed["e_K"].append(part_e_weighted_null(U, V0_P, n_draws=N_NULL_DRAWS, seed=6000 + seed))

        print(f"seed {seed}: C_tc r90={cmp_C['real_metrics']['r90']} "
             f"(null median {cmp_C['comparisons']['r90']['null_median']}, "
             f"top2_frac pctl={cmp_C['comparisons']['top2_frac']['percentile']:.2f})  "
             f"K_tc r90={per_seed['e_K'][-1]['real_metrics']['r90']} "
             f"(null median {per_seed['e_K'][-1]['comparisons']['r90']['null_median']}, "
             f"pctl={per_seed['e_K'][-1]['comparisons']['r90']['percentile']:.2f})")

    def med(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p_ in path:
                v = v[p_]
            vals.append(v)
        return float(np.median(vals))

    b_summary = dict(
        median_real_r90=med(per_seed["b_C"], "real_metrics", "r90"),
        median_null_r90=med(per_seed["b_C"], "comparisons", "r90", "null_median"),
        median_top2_percentile=med(per_seed["b_C"], "comparisons", "top2_frac", "percentile"),
        median_r90_percentile=med(per_seed["b_C"], "comparisons", "r90", "percentile"),
        frac_top2_above_p95=float(np.mean([r["shape_flags"]["top2_above_null_p95"]
                                           for r in per_seed["b_C"]])),
        frac_r90_below_p5=float(np.mean([r["shape_flags"]["r90_below_null_p5"]
                                         for r in per_seed["b_C"]])),
        max_r_excess_95=max(r["r_excess_95"] for r in per_seed["b_C"]))

    e_summary = dict(
        median_real_r90=med(per_seed["e_K"], "real_metrics", "r90"),
        median_null_r90=med(per_seed["e_K"], "comparisons", "r90", "null_median"),
        median_r90_percentile=med(per_seed["e_K"], "comparisons", "r90", "percentile"),
        max_r_excess_95=max(r["r_excess_95"] for r in per_seed["e_K"]))

    f_res = part_f_geometry_map([2, 4, 6, 8], [2, 4, 6, 8], [60 * 8 * 4], n_draws=50)

    print("-" * 90)
    print("PART A (dims, seed 0):", json.dumps(per_seed["a"][0], indent=1))
    print("PART B (unweighted C_tc vs Haar null) summary:", json.dumps(b_summary, indent=1))
    print("PART E (weighted K_tc vs Haar-oriented null) summary:", json.dumps(e_summary, indent=1))
    print("PART F (p,q geometry map, T=1920):", json.dumps(f_res["1920"], indent=1))

    print("\nPART G: width scaling (decisive experiment)")
    g_res = part_g_width_scaling([6, 12, 24, 48], SEEDS[:3], n_null_draws=50)
    g_summary = {}
    for N_, seeds_data in g_res.items():
        g_summary[N_] = dict(
            median_r_tc_K=float(np.median([s["r_tc_K"] for s in seeds_data])),
            median_r_tc_raw=float(np.median([s["r_tc_raw"] for s in seeds_data])),
            median_K_eps5=float(np.median([s["K_epsilon"]["0.05"] for s in seeds_data])),
            median_K_eps_over_M=float(np.median([s["K_epsilon_over_M"] for s in seeds_data])),
            median_null_r90_K=float(np.median([s["null_r90_K"] for s in seeds_data])))
        print(f"  N={N_}: {g_summary[N_]}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, n_null_draws=N_NULL_DRAWS),
              part_a=per_seed["a"], part_b_summary=b_summary, part_e_summary=e_summary,
              part_f_geometry_map=f_res, part_g_summary=g_summary, part_g_raw=g_res,
              per_seed_b=per_seed["b_C"], per_seed_e=per_seed["e_K"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b14_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
