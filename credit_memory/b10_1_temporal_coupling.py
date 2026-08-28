"""Phase B10.1 -- temporal-coupling / principal-angle theory audit,
following B10's finding that neither U nor V is individually low-rank
but their contraction VU collapses to rank ~1-2. Theory/mechanism
audit only: no new training algorithm, no S5.

From compact SVDs U = L_U Sigma_U R_U^dagger (U: TB x N) and
V0 = L_V Sigma_V R_V^dagger (V0: N x TB, unrouted per-branch factor
from B10):

  V0 U = L_V Sigma_V (R_V^dagger L_U) Sigma_U R_U^dagger
       = L_V K_tc R_U^dagger,     K_tc := Sigma_V (R_V^dagger L_U) Sigma_U

C_tc := R_V^dagger L_U (shape r_V x r_U) measures pure temporal-
subspace overlap: its singular values are cos(principal angles)
between the temporal subspace spanned by V0's columns (R_V's own
columns / R_V^dagger's rows) and the temporal subspace spanned by U's
columns (L_U). Since L_V has orthonormal columns and R_U^dagger has
orthonormal rows, singular_values(V0 U) == singular_values(K_tc)
exactly (isometric pre/post-multiplication does not change singular
values) -- this is Part B's key numerical check.

Run:  python -m credit_memory.b10_1_temporal_coupling
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from credit_memory.hankel import build_F
from credit_memory.b9_2_shared_pool import (
    best_pool_exact, pool_most_frequent, pool_largest_lambda, pool_uniform_coverage)
from credit_memory.b10_tangent_adjoint_theory import (
    build_factors, routed_from_factors, direct_routed, low_rank_trunc,
    decision_metrics, effective_ranks, algebraic_rank, qr_pivot_indices,
    cur_reconstruct, greedy_pool, pool_value)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b10_1")

R_LIST = [1, 2, 3, 4, 5, 6]
K_LIST = [1, 2, 4, 6, 8]


def svd_compact(M):
    return np.linalg.svd(M, full_matrices=False)


def coupling_audit(U, V0):
    """U: (TB,N). V0: (N,TB). Returns full dict of SVD factors, C_tc,
    K_tc, principal angles, and consistency checks."""
    Lu, Su, Ruh = svd_compact(U)      # U = Lu diag(Su) Ruh
    Lv, Sv, Rvh = svd_compact(V0)     # V0 = Lv diag(Sv) Rvh
    C_tc = Rvh @ Lu                    # (r_V, r_U)
    K_tc = np.diag(Sv) @ C_tc @ np.diag(Su)

    cos_theta = np.clip(np.linalg.svd(C_tc, compute_uv=False), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_theta))

    VU = V0 @ U
    sv_VU = np.linalg.svd(VU, compute_uv=False)
    sv_Ktc = np.linalg.svd(K_tc, compute_uv=False)
    n_common = min(len(sv_VU), len(sv_Ktc))
    ktc_vu_match_rel_err = float(
        np.max(np.abs(sv_VU[:n_common] - sv_Ktc[:n_common]))
        / (sv_VU[0] + 1e-300))

    A, Sc, Bh = svd_compact(K_tc)      # canonical coupled modes

    return dict(Lu=Lu, Su=Su, Ruh=Ruh, Lv=Lv, Sv=Sv, Rvh=Rvh,
               C_tc=C_tc, K_tc=K_tc, cos_theta=cos_theta,
               angles_deg=angles_deg, VU=VU, A=A, Sc=Sc, Bh=Bh,
               ktc_vu_match_rel_err=ktc_vu_match_rel_err,
               effective_rank_C_tc=effective_ranks(C_tc),
               effective_rank_K_tc=effective_ranks(K_tc),
               effective_rank_VU=effective_ranks(VU))


def rank_source_diagnosis(res):
    """Part B: does effective rank collapse trace to (A) principal-
    angle/subspace-overlap decay itself, (B) amplitude weighting
    Sigma_U/Sigma_V, or (C) both? Compares effective_rank(C_tc)
    (pure overlap, no amplitude) against effective_rank(K_tc)=
    effective_rank(VU) (amplitude-weighted): if they agree, the
    collapse is already present in the pure-overlap geometry and
    amplitude weighting does not materially add to it (answer A)."""
    erC, erK = res["effective_rank_C_tc"], res["effective_rank_K_tc"]
    agree = all(erC[f] == erK[f] for f in ("0.9", "0.95", "0.99"))
    return dict(effective_rank_C_tc=erC, effective_rank_K_tc=erK,
               agree_pure_overlap_explains_it=agree,
               diagnosis=("A_principal_angle_decay" if agree
                         else "B_or_C_amplitude_contributes"))


# ---------------------------------------------------------------------------
# Part C: canonical temporal credit modes, mapped back to physical time.
#
# VU = L_V K_tc R_U^dagger = L_V (A Sigma_c B^dagger) R_U^dagger
#    = (L_V A) Sigma_c (B^dagger R_U^dagger)
# For canonical mode l (l-th column of A / B):
#   candidate-side loading  (physical pole j):  (L_V A)[:, l]
#   lower-mode-side loading (physical mode m):  (B^dagger R_U^dagger)[l, :]
#   teaching-side temporal profile (time t):    phi_l = A[:,l] @ R_V^dagger   (TB,)
#   eligibility-side temporal profile (time t): psi_l = L_U @ B[:,l]          (TB,)
# (R_V^dagger and L_U are the TIME-indexed SVD factors of V0 and U
# respectively; A, B are K_tc's own left/right singular vectors.)
# ---------------------------------------------------------------------------
def canonical_modes(res, n_traj, T, BATCH, l_max=3):
    A, Sc, Bh = res["A"], res["Sc"], res["Bh"]
    B = np.conj(Bh).T
    Rvh, Lu = res["Rvh"], res["Lu"]

    candidate_loadings = res["Lv"] @ A            # (N, r_c)
    mode_loadings = Bh @ res["Ruh"]                # (r_c, N)

    modes = []
    total_energy = float(np.sum(Sc ** 2))
    for l in range(min(l_max, len(Sc))):
        phi_l = A[:, l] @ Rvh                       # (TB,)
        psi_l = Lu @ B[:, l]                         # (TB,)
        phi_traj = np.abs(phi_l).reshape(n_traj, T, BATCH).mean(axis=(0, 2))
        psi_traj = np.abs(psi_l).reshape(n_traj, T, BATCH).mean(axis=(0, 2))
        # cheap timescale proxy: lag-1 autocorrelation of the mean |profile|
        def lag1(v):
            v = v - v.mean()
            if np.std(v) < 1e-12:
                return 0.0
            return float(np.corrcoef(v[:-1], v[1:])[0, 1])
        modes.append(dict(
            l=l, energy_frac=float(Sc[l] ** 2 / (total_energy + 1e-300)),
            candidate_loading_abs=np.abs(candidate_loadings[:, l]).tolist(),
            mode_loading_abs=np.abs(mode_loadings[l, :]).tolist(),
            phi_profile_lag1_autocorr=lag1(phi_traj),
            psi_profile_lag1_autocorr=lag1(psi_traj),
            phi_profile_peak_time=int(np.argmax(phi_traj)),
            psi_profile_peak_time=int(np.argmax(psi_traj))))
    return modes


# ---------------------------------------------------------------------------
# Part D: rank-r reconstruction via K_tc truncation vs. U-only/V-only/
# direct-SVD(VU) truncation (all from B10), before routing.
# ---------------------------------------------------------------------------
def reconstruct_from_ktc(res, r):
    K_r = low_rank_trunc(res["K_tc"], r)
    return res["Lv"] @ K_r @ res["Ruh"]


def part_d_compare(U, V0, res, K=4):
    VU = res["VU"]
    s_true = np.abs(VU)
    out = {}
    for r in R_LIST:
        R_ktc = reconstruct_from_ktc(res, r)
        R_svdvu = low_rank_trunc(VU, r)
        ktc_vs_svdvu_rel_err = float(np.linalg.norm(R_ktc - R_svdvu)
                                     / (np.linalg.norm(R_svdvu) + 1e-300))
        R_Uonly = V0 @ low_rank_trunc(U, r)
        R_Vonly = low_rank_trunc(V0, r) @ U

        variants = dict(K_tc_trunc=R_ktc, U_only=R_Uonly, V_only=R_Vonly,
                        direct_SVD_VU=R_svdvu)
        out[str(r)] = dict(ktc_vs_svdvu_rel_err=ktc_vs_svdvu_rel_err)
        for name, R_hat in variants.items():
            s_hat = np.abs(R_hat)
            fro_err = float(np.linalg.norm(s_hat - s_true)
                            / (np.linalg.norm(s_true) + 1e-300))
            max_err = float(np.max(np.abs(s_hat - s_true)))
            sp = stats.spearmanr(s_hat.ravel(), s_true.ravel()).statistic
            dm = decision_metrics(s_true, s_hat, K)
            out[str(r)][name] = dict(fro_rel_err=fro_err, max_abs_err=max_err,
                                     spearman=float(sp), **dm)
    return out


# ---------------------------------------------------------------------------
# Part E: routing interaction -- does routing reorganize the dominant
# coupled modes, or just preserve them?
# ---------------------------------------------------------------------------
def subspace_angles(Q1, Q2):
    """Q1,Q2: matrices with orthonormal columns (same ambient dim).
    Returns principal angles in degrees between their column spaces."""
    M = np.conj(Q1).T @ Q2
    s = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def part_e_routing(U, V0_P, resP, B1, K=4, r=2):
    VU = resP["VU"]
    R_routed = 0.5 * B1 * VU
    rho_row_slice = R_routed          # P-branch alone, (N,N)

    R_ktc_r = reconstruct_from_ktc(resP, r)
    R_ktc_r_then_routed = 0.5 * B1 * R_ktc_r

    s_true_routed = np.abs(R_routed)
    s_prerouting_compressed = np.abs(R_ktc_r_then_routed)
    dm_prerouting = decision_metrics(s_true_routed, s_prerouting_compressed, K)

    col_idx = qr_pivot_indices(R_routed, r, axis="cols")
    row_idx = qr_pivot_indices(R_routed, r, axis="rows")
    R_direct_cur = cur_reconstruct(R_routed, row_idx, col_idx)
    dm_direct_routed = decision_metrics(s_true_routed, np.abs(R_direct_cur), K)

    # subspace angles: dominant LEFT (candidate) subspace of unrouted VU
    # vs routed R (both P-branch, same N-dim candidate space)
    Lu_vu, _, _ = svd_compact(VU)
    Lu_routed, _, _ = svd_compact(R_routed)
    left_angles = subspace_angles(Lu_vu[:, :r], Lu_routed[:, :r])

    # dominant RIGHT (lower-mode) subspace, comparable across VU, routed R,
    # and (separately) the final combined rho -- shared N-dim mode axis
    _, _, Rh_vu = svd_compact(VU)
    _, _, Rh_routed = svd_compact(R_routed)
    right_angles_vu_vs_routed = subspace_angles(
        np.conj(Rh_vu[:r, :]).T, np.conj(Rh_routed[:r, :]).T)

    return dict(
        prerouting_compression_then_route=dict(fro_rel_err=float(
            np.linalg.norm(s_prerouting_compressed - s_true_routed)
            / (np.linalg.norm(s_true_routed) + 1e-300)), **dm_prerouting),
        direct_routed_compression=dict(fro_rel_err=float(
            np.linalg.norm(np.abs(R_direct_cur) - s_true_routed)
            / (np.linalg.norm(s_true_routed) + 1e-300)), **dm_direct_routed),
        left_subspace_angles_vu_vs_routed_deg=left_angles.tolist(),
        right_subspace_angles_vu_vs_routed_deg=right_angles_vu_vs_routed.tolist())


def part_e_final_rho_subspace(rho_full, R_P_routed, r=2):
    """Right (lower-mode) subspace comparison across unrouted R0_P,
    routed R_P, and the final combined 2N-candidate rho (all share the
    same N-dim lower-mode axis)."""
    _, _, Rh_routed = svd_compact(R_P_routed)
    _, _, Rh_full = svd_compact(rho_full)
    return subspace_angles(np.conj(Rh_routed[:r, :]).T,
                           np.conj(Rh_full[:r, :]).T).tolist()


# ---------------------------------------------------------------------------
# Part F: candidate loadings on dominant modes of the FINAL routed/
# combined relevance matrix, and the r_temporal -> required-K curve.
# ---------------------------------------------------------------------------
def part_f_loadings_and_curve(rho_full, j_rho_unrestricted, f_diag, B1,
                              eps_fracs=(0.05, 0.10)):
    S = np.abs(rho_full)                      # (2N, N)
    A_S, Sig_S, Bh_S = svd_compact(S)
    n2 = S.shape[0]

    top_loading_candidates = {}
    for l in range(min(3, len(Sig_S))):
        order = np.argsort(-np.abs(A_S[:, l]))[:4].tolist()
        top_loading_candidates[str(l)] = order

    winners_true = set(j_rho_unrestricted.values())
    lambda_order = np.argsort(-np.abs(f_diag))

    curve = {}
    F_full = float(sum(S[:, m].max() for m in range(N)))
    for r in range(1, N + 1):
        S_r = low_rank_trunc(S, r)
        winners_r = set(np.argmax(S_r, axis=0).tolist())
        min_K_by_eps = {}
        for eps_frac in eps_fracs:
            eps_abs = eps_frac * F_full
            min_K = None
            for K in range(1, n2 + 1):
                P_hat = best_pool_exact(S_r, K)
                achieved = float(sum(S[list(P_hat), m].max() for m in range(N)))
                regret = F_full - achieved
                if regret <= eps_abs:
                    min_K = K
                    break
            min_K_by_eps[str(eps_frac)] = min_K
        curve[str(r)] = dict(distinct_winners_at_rank_r=len(winners_r),
                             winners_match_true=len(winners_r & winners_true),
                             min_K_by_eps=min_K_by_eps)

    return dict(top_loading_candidates=top_loading_candidates,
               lambda_rank_order=lambda_order.tolist(),
               r_to_min_K_curve=curve)


# ---------------------------------------------------------------------------
# Part G: conditional covering test in the low-rank embedding space.
# ---------------------------------------------------------------------------
def part_g_covering(rho_full, f_diag, j_rho_unrestricted, rho_mat_abs, K=4, r=2):
    S = np.abs(rho_full)
    A_S, Sig_S, Bh_S = svd_compact(S)
    emb = A_S[:, :r] * np.sqrt(Sig_S[:r])[None, :]    # (2N, r) candidate embeddings

    # greedy cover in embedding space: pick K candidates minimizing max
    # over modes of (best achievable score in embedding - achieved score)
    cover_pool = greedy_pool(S, K)      # reuse the exact-objective greedy
    # embedding-space alternative: k-medoids-ish via farthest-point sampling
    chosen = [int(np.argmax(np.linalg.norm(emb, axis=1)))]
    while len(chosen) < K:
        d = np.min([np.linalg.norm(emb - emb[c], axis=1) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    fps_pool = set(chosen)

    oracle_pool = best_pool_exact(S, K)
    rho_pool = pool_most_frequent(j_rho_unrestricted, rho_mat_abs, K)
    largest_lambda_pool = pool_largest_lambda(f_diag, K)
    rng = np.random.RandomState(0)
    random_pool = set(rng.choice(S.shape[0], size=K, replace=False).tolist())

    F_full = float(sum(S[:, m].max() for m in range(N)))
    def regret(P):
        return F_full - float(sum(S[list(P), m].max() for m in range(N)))

    return {name: dict(pool=sorted(P), regret=regret(P))
           for name, P in dict(oracle=oracle_pool, rho_guided=rho_pool,
                               fps_embedding_cover=fps_pool,
                               greedy_exact=cover_pool,
                               largest_lambda=largest_lambda_pool,
                               random=random_pool).items()}


def main() -> None:
    print("=" * 90)
    print(f"Phase B10.1: temporal-coupling / principal-angle theory audit, "
         f"{len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(a=[], b=[], d=[], e=[], f=[], g=[])

    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)

        U, V0_P, V0_Q = build_factors(rows, a1)
        resP = coupling_audit(U, V0_P)
        resQ = coupling_audit(U, V0_Q)
        diagP = rank_source_diagnosis(resP)
        diagQ = rank_source_diagnosis(resQ)
        per_seed["a"].append(dict(P=dict(cos_theta=resP["cos_theta"].tolist(),
                                        angles_deg=resP["angles_deg"].tolist()),
                                  Q=dict(cos_theta=resQ["cos_theta"].tolist(),
                                        angles_deg=resQ["angles_deg"].tolist())))
        per_seed["b"].append(dict(
            P=dict(ktc_vu_match_rel_err=resP["ktc_vu_match_rel_err"],
                  effective_rank_C_tc=resP["effective_rank_C_tc"],
                  effective_rank_K_tc=resP["effective_rank_K_tc"],
                  diagnosis=diagP["diagnosis"]),
            Q=dict(ktc_vu_match_rel_err=resQ["ktc_vu_match_rel_err"],
                  effective_rank_C_tc=resQ["effective_rank_C_tc"],
                  effective_rank_K_tc=resQ["effective_rank_K_tc"],
                  diagnosis=diagQ["diagnosis"])))

        d_res = part_d_compare(U, V0_P, resP)
        per_seed["d"].append(d_res)

        e_res = part_e_routing(U, V0_P, resP, B1)
        R0_P, R0_Q, R_P, R_Q = routed_from_factors(U, V0_P, V0_Q, B1)
        rho_full = np.concatenate([R_P, R_Q], axis=0)
        e_res["final_rho_right_subspace_angles_deg"] = \
            part_e_final_rho_subspace(rho_full, R_P)
        per_seed["e"].append(e_res)

        rho_mat_abs = np.abs(rho_full)
        j_rho_unrestricted = {m: int(np.argmax(rho_mat_abs[:, m])) for m in range(N)}
        f_res = part_f_loadings_and_curve(rho_full, j_rho_unrestricted, f_diag, B1)
        per_seed["f"].append(f_res)

        g_res = part_g_covering(rho_full, f_diag, j_rho_unrestricted, rho_mat_abs)
        per_seed["g"].append(g_res)

        print(f"seed {seed}: P cos_theta[0:2]={np.round(resP['cos_theta'][:2],3)}  "
             f"diagP={diagP['diagnosis']}  "
             f"D r=2 K_tc winner_preserved={d_res['2']['K_tc_trunc']['winner_preserved']:.2f}  "
             f"E right_angle[0]={e_res['right_subspace_angles_vu_vs_routed_deg'][0]:.1f}deg")

    def med_field(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    a_summary = dict(
        median_cos_theta_0_P=med_field(per_seed["a"], "P", "cos_theta", 0),
        median_cos_theta_1_P=med_field(per_seed["a"], "P", "cos_theta", 1),
        median_cos_theta_0_Q=med_field(per_seed["a"], "Q", "cos_theta", 0),
        median_angle_0_P_deg=med_field(per_seed["a"], "P", "angles_deg", 0),
        median_angle_1_P_deg=med_field(per_seed["a"], "P", "angles_deg", 1))

    b_summary = dict(
        max_ktc_vu_match_rel_err=max(
            per_seed["b"][s][br]["ktc_vu_match_rel_err"]
            for s in range(len(SEEDS)) for br in ("P", "Q")),
        n_seeds_diagnosis_A_P=sum(1 for r in per_seed["b"] if r["P"]["diagnosis"].startswith("A")),
        n_seeds_diagnosis_A_Q=sum(1 for r in per_seed["b"] if r["Q"]["diagnosis"].startswith("A")))

    d_summary = {}
    for r in R_LIST:
        d_summary[str(r)] = {}
        for variant in ("K_tc_trunc", "U_only", "V_only", "direct_SVD_VU"):
            d_summary[str(r)][variant] = dict(
                median_fro_rel_err=med_field(per_seed["d"], str(r), variant, "fro_rel_err"),
                median_winner_preserved=med_field(per_seed["d"], str(r), variant, "winner_preserved"),
                median_pool_regret=med_field(per_seed["d"], str(r), variant, "pool_regret"))
        d_summary[str(r)]["ktc_vs_svdvu_max_rel_err"] = max(
            per_seed["d"][s][str(r)]["ktc_vs_svdvu_rel_err"] for s in range(len(SEEDS)))

    e_summary = dict(
        median_prerouting_then_route_winner_preserved=med_field(
            per_seed["e"], "prerouting_compression_then_route", "winner_preserved"),
        median_direct_routed_winner_preserved=med_field(
            per_seed["e"], "direct_routed_compression", "winner_preserved"),
        median_left_angle_0=med_field(per_seed["e"], "left_subspace_angles_vu_vs_routed_deg", 0),
        median_right_angle_0_vu_vs_routed=med_field(
            per_seed["e"], "right_subspace_angles_vu_vs_routed_deg", 0),
        median_right_angle_0_routed_vs_final=med_field(
            per_seed["e"], "final_rho_right_subspace_angles_deg", 0))

    f_curve = {}
    for r in [str(x) for x in range(1, N + 1)]:
        f_curve[r] = dict(
            median_distinct_winners=med_field(per_seed["f"], "r_to_min_K_curve", r,
                                              "distinct_winners_at_rank_r"),
            median_min_K_eps05=float(np.median([
                d["r_to_min_K_curve"][r]["min_K_by_eps"]["0.05"] or N
                for d in per_seed["f"]])),
            median_min_K_eps10=float(np.median([
                d["r_to_min_K_curve"][r]["min_K_by_eps"]["0.1"] or N
                for d in per_seed["f"]])))

    g_summary = {name: dict(median_regret=float(np.median(
        [per_seed["g"][s][name]["regret"] for s in range(len(SEEDS))])))
        for name in ("oracle", "rho_guided", "fps_embedding_cover",
                    "greedy_exact", "largest_lambda", "random")}

    print("-" * 90)
    print("Part A summary:", a_summary)
    print("Part B summary:", b_summary)
    print("Part D (r=2,4) summary:")
    for r in ("2", "4"):
        print(f"  r={r}:", json.dumps(d_summary[r], indent=1))
    print("Part E summary:", e_summary)
    print("Part F curve:", json.dumps(f_curve, indent=1))
    print("Part G summary:", g_summary)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, r_list=R_LIST, k_list=K_LIST),
              part_a_summary=a_summary, part_b_summary=b_summary,
              part_d_summary=d_summary, part_e_summary=e_summary,
              part_f_curve=f_curve, part_g_summary=g_summary,
              per_seed=per_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b10_1_temporal_coupling_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
