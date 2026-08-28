"""Phase B12 -- structural-spectral explanation for the low-dimensional
credit-Hankel spectrum, testing the hypothesis that r_tc is governed by
overlap between marginal temporal spectra (eligibility vs adjoint-
teaching) and the fixed SSM pole architecture, rather than by learned,
trajectory-specific U<->V coordination (which B11's null models already
falsified). Theory/mechanism only: no new training algorithm, no S5,
no new persistent training arm.

Run:  python -m credit_memory.b12_structural_spectral_theory
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F, build_c_t
from credit_memory.b9_2_shared_pool import best_pool_exact, pool_most_frequent
from credit_memory.b10_tangent_adjoint_theory import (
    low_rank_trunc, decision_metrics, effective_ranks, direct_routed,
    build_factors, adjoint_filter)
from credit_memory.b10_1_temporal_coupling import svd_compact
from credit_memory.b11_shared_private_communication import (
    compute_cross_spectrum, pole_transfer_function, part_c2_frequency_coherence)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b12")

K = 4


# ===========================================================================
# PART B: temporal whitening (the decisive test)
# ===========================================================================
def whiten(M, axis):
    """axis='cols': M is (TB,N), whiten its column space's temporal
    energy (flatten singular values to 1, keep directions). axis='rows':
    M is (N,TB), same idea on the row/temporal side."""
    Lm, Sm, Rmh = svd_compact(M)
    return Lm @ Rmh


def part_b_whitening(U, V0, K=K, s_true_abs=None, B1_col=None):
    VU = V0 @ U
    U_w = whiten(U, "cols")
    V0_w = whiten(V0, "rows")

    variants = dict(original=VU, U_whitened=V0 @ U_w,
                    V_whitened=V0_w @ U, both_whitened=V0_w @ U_w)
    out = {}
    for name, M in variants.items():
        sv = np.linalg.svd(M, compute_uv=False)
        er = effective_ranks(M)
        entry = dict(sv=sv.tolist(), effective_rank=er)
        if s_true_abs is not None and B1_col is not None:
            s_hat = np.abs(0.5 * B1_col * M)
            entry["decision"] = decision_metrics(s_true_abs, s_hat, K)
        out[name] = entry
    return out


# ===========================================================================
# PART C: PSD-matched independent null (Fourier phase-randomization
# surrogate, joint-phase-per-realization so each SIDE's own multivariate
# PSD/cross-spectrum is preserved, but the two sides are independently
# randomized so any true cross-relationship is destroyed).
# ===========================================================================
def phase_randomize(X, rng):
    """X: (T,BATCH,N) complex. Same random phase per (batch,freq)
    applied to all N channels jointly -- preserves X's own multivariate
    spectral structure, is independent of any other signal's phases."""
    Tn, Bn, Nn = X.shape
    Xf = np.fft.fft(X, axis=0)
    out = np.zeros_like(Xf)
    for b in range(Bn):
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, Tn))
        out[:, b, :] = Xf[:, b, :] * phases[:, None]
    return np.fft.ifft(out, axis=0)


def part_c_psd_matched_null(rows, a1, n_draws=10, seed=0):
    rng = np.random.RandomState(seed)
    U_true, V0_true, _ = build_factors(rows, a1)
    sv_true = np.linalg.svd(V0_true @ U_true, compute_uv=False)

    draws = []
    for _ in range(n_draws):
        syn_rows = []
        for row in rows:
            new_row = dict(row)
            new_row["Sa0"] = phase_randomize(row["Sa0"], rng)
            new_row["q1"] = phase_randomize(row["q1"], rng)
            syn_rows.append(new_row)
        U_syn, V0_syn, _ = build_factors(syn_rows, a1)
        sv = np.linalg.svd(V0_syn @ U_syn, compute_uv=False)
        draws.append(dict(sv=sv.tolist(), er=effective_ranks(V0_syn @ U_syn),
                          top1_frac=float(sv[0] ** 2 / np.sum(sv ** 2)),
                          top2_frac=float(np.sum(sv[:2] ** 2) / np.sum(sv ** 2))))
    return dict(sv_true=sv_true.tolist(), er_true=effective_ranks(V0_true @ U_true),
               null_draws=draws,
               median_null_er90=float(np.median([d["er"]["0.9"] for d in draws])),
               median_null_top1_frac=float(np.median([d["top1_frac"] for d in draws])),
               median_null_top2_frac=float(np.median([d["top2_frac"] for d in draws])))


# ===========================================================================
# PART D: pole-architecture ablation. U is NOT filtered by poles at all
# (U = raw Sa0); only V0 (adjoint-filtered) depends on the pole set.
# So a pole ablation means: hold Sa0/q1/B1 fixed (the real, task-driven
# signals), swap in a SYNTHETIC pole set only for building V0, and see
# how the rank/spectrum of V0_syn @ U changes.
# ===========================================================================
def build_V0_with_poles(rows, a1_syn):
    V0P_blocks = {j: [] for j in range(N)}
    for row in rows:
        q1 = row["q1"]
        for j in range(N):
            p_P = adjoint_filter(a1_syn[j], q1[:, :, j])
            V0P_blocks[j].append(np.conj(p_P).reshape(-1))
    return np.stack([np.concatenate(V0P_blocks[j]) for j in range(N)], axis=0)


def pole_variants(a1_true, rng):
    mag_true, phase_true = np.abs(a1_true), np.angle(a1_true)
    variants = {}
    variants["true"] = a1_true
    variants["D1_radius_fixed_phase_random"] = mag_true * np.exp(
        1j * rng.uniform(-np.pi, np.pi, N))
    variants["D2_phase_fixed_radius_random"] = rng.uniform(
        mag_true.min(), mag_true.max(), N) * np.exp(1j * phase_true)
    for r0 in (0.90, 0.95, 0.99):
        variants[f"D3_flat_radius_{r0}"] = r0 * np.exp(1j * phase_true)
    variants["D4_narrow_spread"] = np.linspace(0.94, 0.96, N) * np.exp(1j * phase_true)
    variants["D4_medium_spread"] = np.linspace(0.90, 0.995, N) * np.exp(1j * phase_true)
    variants["D4_broad_spread"] = np.linspace(0.5, 0.999, N) * np.exp(1j * phase_true)
    return variants


def part_d_pole_ablation(rows, a1, U, VU_true, seed=0, K=K):
    rng = np.random.RandomState(1000 + seed)
    variants = pole_variants(a1, rng)
    out = {}
    for name, a1_syn in variants.items():
        V0_syn = build_V0_with_poles(rows, a1_syn)
        VU_syn = V0_syn @ U
        sv = np.linalg.svd(VU_syn, compute_uv=False)
        er = effective_ranks(VU_syn)
        sp = stats.spearmanr(np.abs(VU_syn).ravel(), np.abs(VU_true).ravel()).statistic
        out[name] = dict(sv=sv.tolist(), effective_rank=er,
                         spearman_vs_true=float(sp) if not np.isnan(sp) else None)
    return out


# ===========================================================================
# PART F: task-complexity x architecture. r_task independent input
# channels, each independently delayed, summed to a scalar target --
# a controlled way to vary the task's OWN intrinsic temporal rank
# without touching the SSM architecture (N, pole bank) at all.
# ===========================================================================
def make_multi_delay_task(rng, T_, BATCH_, r_task, delays):
    x = rng.randn(T_, BATCH_, r_task)
    y = np.zeros((T_, BATCH_))
    for k, d in enumerate(delays):
        if d > 0:
            y[d:] += x[:-d, :, k]
        else:
            y += x[:, :, k]
    return x, y


def collect_rows_multi_delay(seed, n_traj, r_task, delays, N_=N, T_=T, BATCH_=BATCH):
    from toyrig import ssm_rig as tcg
    from credit_memory.teacher import compute_teacher, set_l2_config
    old_M_IN = tcg.M_IN
    tcg.M_IN = r_task
    try:
        with set_l2_config(N_, T_, BATCH_):
            params = tcg.init_params(seed)
            rows = []
            for k in range(n_traj):
                rng = np.random.RandomState(80000 + seed * 1000 + k)
                x, y = make_multi_delay_task(rng, T_, BATCH_, r_task, delays)
                h, yhat = tcg.forward(params, x)
                r = yhat - y
                rows.append(compute_teacher(params, x, r))
    finally:
        tcg.M_IN = old_M_IN
    return params, rows


def part_f_task_complexity(seed, r_task_list=(1, 2, 4, 8), K=K):
    out = {}
    for r_task in r_task_list:
        delays = [5 + 5 * k for k in range(r_task)]     # distinct, spread delays
        params, rows = collect_rows_multi_delay(seed, N_CAL_TRAJ, r_task, delays)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)
        U, V0_P, V0_Q = build_factors(rows, a1)
        VU = V0_P @ U
        sv = np.linalg.svd(VU, compute_uv=False)
        er = effective_ranks(VU)
        R = direct_routed(rows, f_diag, B1)
        S = np.abs(R)
        F_full = float(sum(S[:, m].max() for m in range(N)))
        min_K = {}
        for eps in (0.05, 0.10):
            eps_abs = eps * F_full
            mk = 2 * N
            for Kc in range(1, 2 * N + 1):
                Pc = best_pool_exact(S, Kc)
                achieved = float(sum(S[list(Pc), m].max() for m in range(N)))
                if F_full - achieved <= eps_abs:
                    mk = Kc
                    break
            min_K[str(eps)] = mk
        out[str(r_task)] = dict(delays=delays, sv=sv.tolist(), effective_rank=er,
                               K_epsilon=min_K)
    return out


# ===========================================================================
# PART E: finite-horizon analytic pole Gramian (architecture only, no
# real task signal): P_T = sum_{k=0}^{T-1} F^k d d^dagger (F^dagger)^k,
# (P_T)_{ij} = d_i conj(d_j) [1-(lambda_i conj(lambda_j))^T]
#              / [1 - lambda_i conj(lambda_j)]
# ===========================================================================
def analytic_gramian(f_diag, d, T_):
    n2 = len(f_diag)
    P = np.zeros((n2, n2), np.complex128)
    for i in range(n2):
        for j in range(n2):
            li, lj = f_diag[i], np.conj(f_diag[j])
            denom = 1 - li * lj
            if abs(denom) < 1e-12:
                P[i, j] = d[i] * np.conj(d[j]) * T_
            else:
                P[i, j] = d[i] * np.conj(d[j]) * (1 - (li * lj) ** T_) / denom
    return P


def direct_gramian(f_diag, d, T_):
    n2 = len(f_diag)
    P = np.zeros((n2, n2), np.complex128)
    Fk_d = d.copy()
    for k in range(T_):
        P += np.outer(Fk_d, np.conj(Fk_d))
        Fk_d = f_diag * Fk_d
    return P


def part_e_analytic_gramian(f_diag, T_=T):
    d = np.ones(len(f_diag), np.complex128)
    P_formula = analytic_gramian(f_diag, d, T_)
    P_direct = direct_gramian(f_diag, d, T_)
    rel_err = float(np.linalg.norm(P_formula - P_direct) / (np.linalg.norm(P_direct) + 1e-300))
    er = effective_ranks(P_formula)
    return dict(rel_err=rel_err, effective_rank=er,
               eigvals=np.sort(np.abs(np.linalg.eigvalsh(
                   (P_formula + np.conj(P_formula).T) / 2)))[::-1].tolist())


def main() -> None:
    print("=" * 90)
    print(f"Phase B12: structural-spectral theory audit, {len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(b=[], c=[], d=[])
    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)
        U, V0_P, V0_Q = build_factors(rows, a1)
        R = direct_routed(rows, f_diag, B1)
        s_true = np.abs(R[:N, :])

        per_seed["b"].append(part_b_whitening(U, V0_P, s_true_abs=s_true, B1_col=B1))
        per_seed["c"].append(part_c_psd_matched_null(rows, a1, n_draws=5, seed=seed))
        VU_true = V0_P @ U
        per_seed["d"].append(part_d_pole_ablation(rows, a1, U, VU_true, seed=seed))

        print(f"seed {seed}: orig_er90={per_seed['b'][-1]['original']['effective_rank']['0.9']}  "
             f"both_whitened_er90={per_seed['b'][-1]['both_whitened']['effective_rank']['0.9']}  "
             f"null_er90={per_seed['c'][-1]['median_null_er90']}  "
             f"D_medium_spread_er90={per_seed['d'][-1]['D4_medium_spread']['effective_rank']['0.9']}")

    f_seeds = SEEDS[:3]
    f_results = {seed: part_f_task_complexity(seed) for seed in f_seeds}

    e_result = part_e_analytic_gramian(build_F(collect_rows(0, 1, offset=0)[1][0]["a1"]))

    def med(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    b_summary = {name: dict(median_er90=med(per_seed["b"], name, "effective_rank", "0.9"))
                for name in ("original", "U_whitened", "V_whitened", "both_whitened")}
    b_decision = {name: dict(
        median_winner=med(per_seed["b"], name, "decision", "winner_preserved"),
        median_regret=med(per_seed["b"], name, "decision", "pool_regret"))
        for name in ("original", "U_whitened", "V_whitened", "both_whitened")}

    c_summary = dict(median_true_er90=med(per_seed["c"], "er_true", "0.9"),
                     median_null_er90=med(per_seed["c"], "median_null_er90"),
                     median_null_top2_frac=med(per_seed["c"], "median_null_top2_frac"))

    d_names = list(per_seed["d"][0].keys())
    d_summary = {name: dict(median_er90=med(per_seed["d"], name, "effective_rank", "0.9"),
                            median_spearman_vs_true=med(per_seed["d"], name, "spearman_vs_true"))
                for name in d_names}

    f_summary = {}
    for r_task in (1, 2, 4, 8):
        ers = [f_results[s][str(r_task)]["effective_rank"]["0.9"] for s in f_seeds]
        keps = [f_results[s][str(r_task)]["K_epsilon"]["0.05"] for s in f_seeds]
        f_summary[str(r_task)] = dict(median_er90=float(np.median(ers)),
                                      median_K_eps5=float(np.median(keps)))

    print("-" * 90)
    print("PART B (whitening) effective rank:", json.dumps(b_summary, indent=1))
    print("PART B decision quality:", json.dumps(b_decision, indent=1))
    print("PART C (PSD-matched null):", json.dumps(c_summary, indent=1))
    print("PART D (pole ablation):", json.dumps(d_summary, indent=1))
    print("PART E (analytic Gramian) rel_err:", e_result["rel_err"],
         "effective_rank:", e_result["effective_rank"])
    print("PART F (task complexity):", json.dumps(f_summary, indent=1))

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, f_seeds=f_seeds),
              part_b_summary=b_summary, part_b_decision=b_decision,
              part_c_summary=c_summary, part_d_summary=d_summary,
              part_e_result=e_result, part_f_summary=f_summary,
              per_seed=per_seed, f_results=f_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b12_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
