"""Phase B15 -- is K-pool scaling failure a COORDINATE-SPARSITY failure
(the dominant credit subspace becomes incoherent/delocalized in the
physical pole-coordinate basis as width grows) while a DENSE
low-dimensional reduced-order credit realization remains viable?
Theory/mechanism only: no new persistent training algorithm, no S5
benchmark suite.

x_t = F x_{t-1} + d u_t, F=diag(f_diag), d=ones(M) (per lower mode m,
u_t=Sa0[:,:,m] -- the SAME driving signal feeds every candidate,
matching the established credit-system convention throughout B9-B14).

Run:  python -m credit_memory.b15_dense_rom_theory
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F, build_c_t
from credit_memory.b9_2_shared_pool import best_pool_exact
from credit_memory.b10_tangent_adjoint_theory import (
    low_rank_trunc, decision_metrics, effective_ranks, direct_routed,
    build_factors, forward_filter, algebraic_rank, qr_pivot_indices,
    cur_reconstruct)
from credit_memory.b10_1_temporal_coupling import svd_compact
from credit_memory.b14_finite_size_null_theory import (
    random_orthonormal_basis, collect_rows_width, build_factors_width,
    direct_routed_width, K_epsilon_curve)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b15")

K = 4
WIDTHS = [6, 12, 24, 48]


# ===========================================================================
# PART A/B/J: coordinate coherence, leverage, K_sub, and the matched
# Haar-random comparison.
# ===========================================================================
def leverage_scores(Q_r):
    """Q_r: (M, r) orthonormal columns. Returns (M,) leverage scores
    summing to r."""
    return np.sum(np.abs(Q_r) ** 2, axis=1)


def coherence_metrics(Q_r, M):
    r = Q_r.shape[1]
    ell = leverage_scores(Q_r)
    mu = (M / r) * np.max(ell) if r > 0 else 0.0
    p = ell / (ell.sum() + 1e-300)
    entropy = float(-np.sum(p * np.log(p + 1e-300)))
    participation_ratio = float(1.0 / np.sum(p ** 2))
    gini = float(np.sum(np.abs(ell[:, None] - ell[None, :])) / (2 * M * ell.sum() + 1e-300))
    return dict(max_leverage=float(np.max(ell)), coherence_mu=float(mu),
               leverage_entropy=entropy, participation_ratio=participation_ratio,
               gini=gini, leverage=ell.tolist())


def K_sub_from_leverage(ell, r, eps_list=(0.01, 0.05, 0.10, 0.20)):
    order = np.argsort(-ell)
    cum = np.cumsum(ell[order])
    out = {}
    for eps in eps_list:
        target = (1 - eps) * r
        k = int(np.searchsorted(cum, target) + 1)
        out[str(eps)] = min(k, len(ell))
    return out


def part_ab_coherence(R, r_list):
    M = R.shape[0]
    Q, sv, Zh = svd_compact(R)
    out = {}
    for r in r_list:
        r = min(r, Q.shape[1])
        Q_r = Q[:, :r]
        cm = coherence_metrics(Q_r, M)
        ksub = K_sub_from_leverage(np.array(cm["leverage"]), r)
        out[str(r)] = dict(coherence=cm, K_sub=ksub, K_sub_over_M={k: v / M for k, v in ksub.items()})
    return out


def haar_leverage_null(M, r, n_draws, rng):
    """Matched Haar-random r-dim subspace of R^M/C^M -- coherence and
    K_sub null distribution."""
    draws = []
    for _ in range(n_draws):
        Q_r = random_orthonormal_basis(M, r, rng)
        cm = coherence_metrics(Q_r, M)
        ksub = K_sub_from_leverage(np.array(cm["leverage"]), r)
        draws.append(dict(coherence_mu=cm["coherence_mu"], max_leverage=cm["max_leverage"],
                         K_sub_005=ksub["0.05"]))
    return dict(median_coherence_mu=float(np.median([d["coherence_mu"] for d in draws])),
               median_K_sub_005=float(np.median([d["K_sub_005"] for d in draws])),
               median_K_sub_005_over_M=float(np.median([d["K_sub_005"] for d in draws])) / M)


# ===========================================================================
# PART C/D: dense rank-r vs coordinate-K approximation, and the minimum
# dense rank r_grad preserving decision/gradient quality.
# ===========================================================================
def part_cd_rank_vs_coordinate(R, N_, K=K, r_list=None, K_list=None):
    M = R.shape[0]
    s_true = np.abs(R)
    r_list = r_list or [1, 2, 4, min(8, M)]
    K_list = K_list or [1, 2, 4, min(8, M), M]

    dense_curve = {}
    for r in r_list:
        R_r = low_rank_trunc(R, r)
        s_hat = np.abs(R_r)
        fro = float(np.linalg.norm(s_hat - s_true) / (np.linalg.norm(s_true) + 1e-300))
        cos = float(np.sum(np.conj(s_true.ravel()) * s_hat.ravel()).real
                    / (np.linalg.norm(s_true) * np.linalg.norm(s_hat) + 1e-300))
        dm = decision_metrics(s_true, s_hat, min(K, M))
        dense_curve[str(r)] = dict(fro_rel_err=fro, cos=cos, **dm)

    coord_curve = {}
    for Kc in K_list:
        row_idx = qr_pivot_indices(R, min(Kc, M), axis="rows")
        col_idx = list(range(R.shape[1]))
        R_hat = cur_reconstruct(R, row_idx, col_idx) if Kc < M else R
        s_hat = np.abs(R_hat)
        fro = float(np.linalg.norm(s_hat - s_true) / (np.linalg.norm(s_true) + 1e-300))
        cos = float(np.sum(np.conj(s_true.ravel()) * s_hat.ravel()).real
                    / (np.linalg.norm(s_true) * np.linalg.norm(s_hat) + 1e-300))
        dm = decision_metrics(s_true, s_hat, min(K, M))
        coord_curve[str(Kc)] = dict(fro_rel_err=fro, cos=cos, **dm)

    return dict(dense=dense_curve, coordinate=coord_curve)


def r_grad_for_tolerance(R, tolerances=(0.90, 0.95, 0.99)):
    M = R.shape[0]
    s_true = np.abs(R)
    norm_true = np.linalg.norm(s_true)
    out = {}
    for tol in tolerances:
        r_found = M
        for r in range(1, M + 1):
            R_r = low_rank_trunc(R, r)
            cos = float(np.sum(np.conj(s_true.ravel()) * np.abs(R_r).ravel()).real
                        / (norm_true * np.linalg.norm(np.abs(R_r)) + 1e-300))
            if cos >= tol:
                r_found = r
                break
        out[str(tol)] = r_found
    return out


# ===========================================================================
# PART E: F-invariance / closure of the dominant credit basis under the
# pole dynamics operator F=diag(f_diag).
# ===========================================================================
def part_e_closure(R, f_diag, r_list):
    Q, sv, Zh = svd_compact(R)
    M = R.shape[0]
    F = f_diag
    out = {}
    for r in r_list:
        r = min(r, Q.shape[1])
        Phi_r = Q[:, :r]
        F_Phi = F[:, None] * Phi_r        # F is diagonal: elementwise row-scale
        proj = Phi_r @ (np.conj(Phi_r).T @ F_Phi)
        residual = F_Phi - proj
        closure_err = float(np.linalg.norm(residual) / (np.linalg.norm(F_Phi) + 1e-300))
        # principal angles between span(F Phi_r) and span(Phi_r)
        Qf, _ = np.linalg.qr(F_Phi)
        M_overlap = np.conj(Phi_r).T @ Qf
        angles = np.degrees(np.arccos(np.clip(np.linalg.svd(M_overlap, compute_uv=False), -1, 1)))
        A_r = np.conj(Phi_r).T @ (F[:, None] * Phi_r)
        b_r = np.conj(Phi_r).T @ np.ones(M, np.complex128)
        out[str(r)] = dict(closure_rel_err=closure_err, principal_angles_deg=angles.tolist(),
                          A_r=A_r.tolist(), b_r=b_r.tolist())
    return out


# ===========================================================================
# PART F: actual reduced-order rollout on held-out trajectories, vs the
# static best-rank-r projection (the critical gap).
# ===========================================================================
def rollout_reduced(A_r, b_r, u_t):
    """u_t: (T,BATCH). z_t: (r,BATCH). Returns x_hat: (T,BATCH,r) proj
    coefficients (caller multiplies by Phi_r for full state)."""
    Tn, Bn = u_t.shape
    r = len(b_r)
    z = np.zeros((Tn, Bn, r), np.complex128)
    prev = np.zeros((Bn, r), np.complex128)
    for t in range(Tn):
        prev = prev @ A_r.T + np.outer(u_t[t], b_r)
        z[t] = prev
    return z


def part_f_rollout(rows_test, f_diag, B1, Phi_r, A_r, b_r, N_, K=K):
    """Held-out test rows: exact full-bank G[m] (=R_true[:,m].sum())
    vs (a) static best-rank-r projection of the TRAIN-derived R
    (already known), (b) dynamic rollout of A_r/b_r on the TEST rows'
    own Sa0/q1."""
    r = Phi_r.shape[1]
    G_true = np.zeros(N_, np.complex128)
    G_dynamic = np.zeros(N_, np.complex128)
    for row in rows_test:
        for m in range(N_):
            c_t = build_c_t(row["q1"], B1[:, m])            # (T,BATCH,M)
            u_t = row["Sa0"][:, :, m]
            z = rollout_reduced(A_r, b_r, u_t)                # (T,BATCH,r)
            x_hat = z @ Phi_r.T                               # (T,BATCH,M)
            G_dynamic[m] += np.sum(np.conj(c_t) * x_hat)
            x_exact = np.stack([forward_filter(f_diag[j], u_t) for j in range(2 * N_)], axis=-1)
            G_true[m] += np.sum(np.conj(c_t) * x_exact)
    cos = float(np.real(np.vdot(G_true, G_dynamic))
               / (np.linalg.norm(G_true) * np.linalg.norm(G_dynamic) + 1e-300))
    rel_err = float(np.linalg.norm(G_dynamic - G_true) / (np.linalg.norm(G_true) + 1e-300))
    return dict(cos=cos, rel_err=rel_err, r=r)


def collect_test_rows_width(seed, N_, T_, BATCH_, n_traj, offset=9000):
    from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
    from toyrig import ssm_rig as tcg
    with set_l2_config(N_, T_, BATCH_):
        params = tcg.init_params(seed)
        rows = []
        for k in range(n_traj):
            rng = np.random.RandomState(70000 + seed * 1000 + offset + k)
            x, r = draw_trajectory(params, rng, T_, BATCH_)
            rows.append(compute_teacher(params, x, r))
    return rows


def width_sweep(N_list, seeds, T_=T, BATCH_=BATCH, n_traj=N_CAL_TRAJ,
               r_list=(1, 2, 4), n_null_draws=50, K=K):
    out = {}
    for N_ in N_list:
        per_seed = []
        for seed in seeds:
            params, cal_rows = collect_rows_width(seed, N_, T_, BATCH_, n_traj)
            a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]
            f_diag = build_F(a1)
            R = direct_routed_width(cal_rows, f_diag, B1, N_)
            M = R.shape[0]

            r_list_eff = [r for r in r_list if r <= M]
            ab = part_ab_coherence(R, r_list_eff)
            rng = np.random.RandomState(11000 + seed)
            null_by_r = {str(r): haar_leverage_null(M, r, n_null_draws, rng)
                        for r in r_list_eff}

            cd = part_cd_rank_vs_coordinate(R, N_, K=min(K, N_),
                                            r_list=r_list_eff,
                                            K_list=[r for r in r_list_eff] + [M])
            r_grad = r_grad_for_tolerance(R)

            e_res = part_e_closure(R, f_diag, r_list_eff)

            test_rows = collect_test_rows_width(seed, N_, T_, BATCH_, n_traj)
            from credit_memory.b10_1_temporal_coupling import svd_compact
            Q, sv, Zh = svd_compact(R)
            rollout = {}
            for r in r_list_eff:
                Phi_r = Q[:, :r]
                A_r = np.array(e_res[str(r)]["A_r"])
                b_r = np.array(e_res[str(r)]["b_r"])
                rollout[str(r)] = part_f_rollout(test_rows, f_diag, B1, Phi_r, A_r, b_r, N_, K=min(K, N_))

            keps = K_epsilon_curve(np.abs(R), N_)

            per_seed.append(dict(M=M, ab=ab, null=null_by_r, cd=cd, r_grad=r_grad,
                                closure=e_res, rollout=rollout, K_epsilon=keps))
        out[str(N_)] = per_seed
    return out


def main() -> None:
    print("=" * 90)
    print("Phase B15: dense-ROM / coordinate-sparsity theory, width scaling")
    print("=" * 90)

    seeds_by_width = {6: SEEDS[:3], 12: SEEDS[:3], 24: SEEDS[:2], 48: SEEDS[:2]}
    res = {}
    for N_ in WIDTHS:
        res[str(N_)] = width_sweep([N_], seeds_by_width[N_])[str(N_)]
        seeds_data = res[str(N_)]
        med = lambda key_fn: float(np.median([key_fn(s) for s in seeds_data]))
        print(f"N={N_}: median coherence_mu(r=2)="
             f"{med(lambda s: s['ab'].get('2', s['ab'][list(s['ab'].keys())[0]])['coherence']['coherence_mu']):.2f}  "
             f"median K_sub/M(r=2,5%)="
             f"{med(lambda s: s['ab'].get('2', s['ab'][list(s['ab'].keys())[0]])['K_sub_over_M']['0.05']):.2f}  "
             f"median r_grad(0.95)={med(lambda s: s['r_grad']['0.95']):.1f}  "
             f"median K_eps5/M={med(lambda s: s['K_epsilon']['0.05'] / s['M']):.2f}  "
             f"median rollout_cos(r=2)="
             f"{med(lambda s: s['rollout'].get('2', s['rollout'][list(s['rollout'].keys())[0]])['cos']):.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, widths=WIDTHS,
                                    seeds_by_width={str(k): v for k, v in seeds_by_width.items()}),
              results=res)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b15_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
