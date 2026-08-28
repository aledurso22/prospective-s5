"""Phase B11 -- shared/private communication-subspace mechanism audit.
Explains WHY K_tc/VU has such a sharply decaying spectrum, building on
B10/B10.1/B10.2. Theory/mechanism only: no new training algorithm, no
S5, no new persistent training arm.

Central hypothesis: U = U_shared + U_private, V = V_shared + V_private,
with VU ~= V_shared U_shared because private/private and mismatched
cross terms are weak, even though U and V individually are only
moderately (not sharply) low-rank.

Construction: from K_tc = A Sigma_c B^dagger (compact SVD), the SHARED
temporal directions are
  Psi := L_U B_r          (U-side, TB x r, orthonormal columns)
  Phi := R_V^dagger^H A_r (V-side, TB x r, orthonormal columns --
         NOTE the Hermitian, not plain, transpose of R_V^dagger; this
         is required for Rvh @ Phi = A_r to hold, i.e. for Psi/Phi to
         be genuine projections of the ORIGINAL U/V0 matrices, not an
         artifact of a wrong transpose convention)
U_shared := Psi Psi^dagger U,  V_shared := V0 Phi Phi^dagger
(projections of the RAW U/V0 onto the shared temporal directions found
by the coupling itself -- NOT each side's own independent top-r SVD,
which is a different, and per B10.1, worse construction).

Run:  python -m credit_memory.b11_shared_private_communication
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
    algebraic_rank)
from credit_memory.b10_1_temporal_coupling import (
    coupling_audit, svd_compact, part_d_compare, R_LIST as B101_R_LIST)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b11")

R_LIST = [1, 2, 3, 4]
K = 4


# ===========================================================================
# PART A: shared/private latent decomposition
# ===========================================================================
def shared_private_basis(res, r):
    """res: coupling_audit(U, V0) output. Returns Psi (TB,r), Phi (TB,r),
    both with orthonormal columns, built from K_tc's own canonical
    directions (NOT U's/V's independent SVD)."""
    A, Bh = res["A"], res["Bh"]
    B = np.conj(Bh).T
    Psi = res["Lu"] @ B[:, :r]
    Phi = np.conj(res["Rvh"]).T @ A[:, :r]
    return Psi, Phi


def shared_private_split(U, V0, Psi, Phi):
    U_shared = Psi @ (np.conj(Psi).T @ U)
    U_private = U - U_shared
    V_shared = (V0 @ Phi) @ np.conj(Phi).T
    V_private = V0 - V_shared
    return U_shared, U_private, V_shared, V_private


def part_a_variance_and_credit(U, V0, res, rho_true_abs_P, B1, K=K):
    """A2/A3/A4/A5 combined per r: variance fractions, R_shared/R_private
    ablations, decision metrics, and the 4-way cross-term decomposition."""
    out = {}
    fro_U2, fro_V2 = float(np.linalg.norm(U) ** 2), float(np.linalg.norm(V0) ** 2)
    VU = res["VU"]
    fro_VU2 = float(np.linalg.norm(VU) ** 2)

    for r in R_LIST:
        Psi, Phi = shared_private_basis(res, r)
        U_s, U_p, V_s, V_p = shared_private_split(U, V0, Psi, Phi)

        var_U_shared = float(np.linalg.norm(U_s) ** 2 / (fro_U2 + 1e-300))
        var_V_shared = float(np.linalg.norm(V_s) ** 2 / (fro_V2 + 1e-300))

        VsUs = V_s @ U_s
        VsUp = V_s @ U_p
        VpUs = V_p @ U_s
        VpUp = V_p @ U_p
        terms = dict(VsUs=VsUs, VsUp=VsUp, VpUs=VpUs, VpUp=VpUp)
        term_norms = {k: float(np.linalg.norm(t)) for k, t in terms.items()}
        recon = VsUs + VsUp + VpUs + VpUp
        recon_err = float(np.linalg.norm(recon - VU) / (np.linalg.norm(VU) + 1e-300))

        VU_energy_frac_shared = float(np.linalg.norm(VsUs) ** 2 / (fro_VU2 + 1e-300))

        def routed_abs(R0):
            return np.abs(0.5 * B1 * R0)

        s_true = rho_true_abs_P     # P-branch only, matches R0=VU here
        dm_shared = decision_metrics(s_true, routed_abs(VsUs), K)
        dm_private = decision_metrics(s_true, routed_abs(VpUp), K)
        dm_full_from_terms = decision_metrics(s_true, routed_abs(recon), K)

        rho_energy_shared_only = float(np.linalg.norm(routed_abs(VsUs)) ** 2
                                       / (np.linalg.norm(s_true) ** 2 + 1e-300))

        out[str(r)] = dict(
            var_frac_U_shared=var_U_shared, var_frac_V_shared=var_V_shared,
            VU_energy_frac_from_shared_shared=VU_energy_frac_shared,
            cross_term_norms=term_norms,
            recon_check_rel_err=recon_err,
            R_shared_decision=dm_shared, R_private_decision=dm_private,
            rho_energy_fraction_from_shared_shared=rho_energy_shared_only)
    return out


# ===========================================================================
# PART B: balanced credit / Hankel analogue
# ===========================================================================
def part_b1_spectral_identity(U, V0):
    We = U @ np.conj(U).T
    Wt = np.conj(V0).T @ V0
    eigvals = np.abs(np.linalg.eigvals(Wt @ We))
    eigvals_sorted = np.sort(eigvals)[::-1][:N]
    sv2 = np.linalg.svd(V0 @ U, compute_uv=False) ** 2
    rel_err = float(np.max(np.abs(eigvals_sorted - sv2)) / (sv2[0] + 1e-300))
    return dict(rel_err=rel_err, sv2=sv2.tolist(), eig_Wt_We=eigvals_sorted.tolist())


def part_b4_energy_vs_credit(res):
    """High eligibility-energy directions vs high balanced-credit
    importance: correlate U's own singular values (energy) against
    how much each U-principal-direction loads onto K_tc's dominant
    mode (|B[:,0]|)."""
    Su = res["Su"]
    B = np.conj(res["Bh"]).T
    dominant_loading = np.abs(B[:, 0])
    sp = stats.spearmanr(Su, dominant_loading).statistic
    Sv = res["Sv"]
    A = res["A"]
    dominant_loading_V = np.abs(A[:, 0])
    sp_V = stats.spearmanr(Sv, dominant_loading_V).statistic
    return dict(spearman_U_energy_vs_dominant_credit_loading=float(sp)
               if not np.isnan(sp) else None,
               spearman_V_energy_vs_dominant_credit_loading=float(sp_V)
               if not np.isnan(sp_V) else None)


# ===========================================================================
# PART C: temporal / cross-spectral coherence between raw eligibility
# (Sa0) and raw adjoint-teaching driving signal (q1) -- pole filtering
# enters separately via H_j(e^{i omega}) in C4, not by pre-filtering
# here.
# ===========================================================================
def compute_cross_spectrum(rows):
    """C[omega, j, m] = mean over (trajectory,batch) realizations of
    conj(Q1_hat[j,omega]) Sa0_hat[m,omega]. freqs: (T,) cycles/sample."""
    Sa0_hat_all, Q1_hat_all = [], []
    for row in rows:
        Sa0_hat = np.fft.fft(row["Sa0"], axis=0)     # (T,BATCH,N)
        Q1_hat = np.fft.fft(row["q1"], axis=0)
        for b in range(row["Sa0"].shape[1]):
            Sa0_hat_all.append(Sa0_hat[:, b, :])
            Q1_hat_all.append(Q1_hat[:, b, :])
    Sa0_hat_all = np.stack(Sa0_hat_all, axis=0)       # (n_real,T,N)
    Q1_hat_all = np.stack(Q1_hat_all, axis=0)
    n_real = Sa0_hat_all.shape[0]
    freqs = np.fft.fftfreq(row["Sa0"].shape[0])
    C = np.einsum("rtj,rtm->tjm", np.conj(Q1_hat_all), Sa0_hat_all) / n_real
    return freqs, C


def part_c2_frequency_coherence(freqs, C, top_fracs=(0.9,)):
    Tn = C.shape[0]
    ranks_by_freq, top_sv_by_freq = [], []
    for t in range(Tn):
        sv = np.linalg.svd(C[t], compute_uv=False)
        er = effective_ranks(C[t], fracs=top_fracs)
        ranks_by_freq.append(er[str(top_fracs[0])])
        top_sv_by_freq.append(float(sv[0]) if len(sv) else 0.0)
    total_energy = np.array([np.sum(np.linalg.svd(C[t], compute_uv=False) ** 2)
                             for t in range(Tn)])
    n_strong_freqs = int(np.sum(total_energy > 0.05 * total_energy.max()))
    dominant_freq_idx = int(np.argmax(total_energy))
    return dict(median_rank_at_strong_freqs=float(np.median(
        [ranks_by_freq[t] for t in range(Tn) if total_energy[t] > 0.05 * total_energy.max()])),
        n_strong_freqs=n_strong_freqs, total_freq_bins=Tn,
        dominant_freq=float(freqs[dominant_freq_idx]),
        energy_frac_dominant_freq=float(total_energy[dominant_freq_idx] / total_energy.sum()))


def part_c3_parseval_check(rows, C):
    """Verify sum_t conj(q1_t[j]) Sa0_t[m], averaged over realizations,
    equals (1/T) sum_omega C[omega,j,m] (Parseval, DFT sum convention)."""
    Tn = rows[0]["Sa0"].shape[0]
    time_domain = np.zeros_like(C[0])
    n_real = 0
    for row in rows:
        for b in range(row["Sa0"].shape[1]):
            time_domain += np.conj(row["q1"][:, b, :]).T @ row["Sa0"][:, b, :]
            n_real += 1
    time_domain /= n_real
    freq_domain = C.sum(axis=0) / Tn
    rel_err = float(np.max(np.abs(time_domain - freq_domain))
                    / (np.max(np.abs(time_domain)) + 1e-300))
    return dict(parseval_rel_err=rel_err)


def pole_transfer_function(lam, freqs):
    """H_j(e^{i omega}) = 1/(1 - lambda e^{-i omega})."""
    omega = 2 * np.pi * freqs
    return 1.0 / (1.0 - lam * np.exp(-1j * omega))


def part_c4_pole_matching(freqs, C, f_diag, rho_abs_row, U_abs_row, j_rho_top, j_pool):
    """spectral_match[j] = sum_omega ||C[omega]||_weight * |H_j(omega)|^2,
    using the DOMINANT left-singular-vector energy of C[omega] (summed
    over lower modes m via the mode weighting in rho_abs_row's own m)
    as W(omega). Correlate against |rho|, ideal utility, and pool
    membership for the SAME lower mode row."""
    Tn = C.shape[0]
    W = np.array([np.sum(np.linalg.svd(C[t], compute_uv=False) ** 2) for t in range(Tn)])
    W = W / (W.sum() + 1e-300)
    match = np.zeros(len(f_diag))
    for j, lam in enumerate(f_diag):
        Hj = pole_transfer_function(lam, freqs)
        match[j] = float(np.sum(W * np.abs(Hj) ** 2))
    sp_rho = stats.spearmanr(match, rho_abs_row).statistic
    sp_U = stats.spearmanr(match, U_abs_row).statistic
    mean_match_pool = float(np.mean(match[list(j_pool)]))
    mean_match_nonpool = float(np.mean(np.delete(match, list(j_pool))))
    return dict(spearman_match_vs_rho=float(sp_rho) if not np.isnan(sp_rho) else None,
               spearman_match_vs_U=float(sp_U) if not np.isnan(sp_U) else None,
               mean_match_in_pool=mean_match_pool,
               mean_match_out_pool=mean_match_nonpool,
               top1_match_is_rho_winner=bool(int(np.argmax(match)) == j_rho_top))


def part_c5_mode_approx_by_K_poles(freqs, C, f_diag, K_list=(1, 2, 4)):
    """Fit the dominant coherence weighting W(omega) using K physical
    |H_j(omega)|^2 profiles (nonneg least squares via plain lstsq +
    clipping, kept simple/diagnostic)."""
    Tn = C.shape[0]
    W = np.array([np.sum(np.linalg.svd(C[t], compute_uv=False) ** 2) for t in range(Tn)])
    W = W / (W.max() + 1e-300)
    basis = np.stack([np.abs(pole_transfer_function(lam, freqs)) ** 2
                      for lam in f_diag], axis=1)     # (T, 2N)
    basis = basis / (basis.max(axis=0, keepdims=True) + 1e-300)
    out = {}
    for K_ in K_list:
        best_err, best_set = np.inf, None
        remaining = list(range(basis.shape[1]))
        chosen = []
        cur_resid = W.copy()
        for _ in range(K_):
            scores = [np.abs(np.dot(basis[:, j], cur_resid)) for j in remaining]
            j_best = remaining[int(np.argmax(scores))]
            chosen.append(j_best)
            remaining.remove(j_best)
            coefs, *_ = np.linalg.lstsq(basis[:, chosen], W, rcond=None)
            cur_resid = W - basis[:, chosen] @ coefs
        rel_err = float(np.linalg.norm(cur_resid) / (np.linalg.norm(W) + 1e-300))
        out[str(K_)] = dict(rel_err=rel_err, chosen=chosen)
    return out


# ===========================================================================
# PART D: null models and finite-size scaling
# ===========================================================================
def null_time_shift(rows, rng):
    """Independently circularly time-shift each trajectory's Sa0 (per
    batch element) relative to q1 -- preserves each side's own marginal
    temporal spectrum, destroys precise cross-interface alignment."""
    new_rows = []
    for row in rows:
        Sa0 = row["Sa0"].copy()
        Tn, Bn = Sa0.shape[0], Sa0.shape[1]
        for b in range(Bn):
            shift = int(rng.randint(1, Tn))
            Sa0[:, b, :] = np.roll(Sa0[:, b, :], shift, axis=0)
        new_row = dict(row)
        new_row["Sa0"] = Sa0
        new_rows.append(new_row)
    return new_rows


def null_cross_seed(rows_u_seed, rows_v_seed):
    """Pair U from one seed's rows with V-driving q1 from another
    seed's rows (same trajectory index, mismatched underlying model)."""
    new_rows = []
    for ru, rv in zip(rows_u_seed, rows_v_seed):
        nr = dict(ru)
        nr["q1"] = rv["q1"]
        nr["B1"] = rv["B1"]
        new_rows.append(nr)
    return new_rows


def part_d1_null_models(rows, a1, rng):
    from credit_memory.b10_tangent_adjoint_theory import build_factors
    U_true, V0_true, _ = build_factors(rows, a1)
    sv_true = np.linalg.svd(V0_true @ U_true, compute_uv=False)

    rows_shift = null_time_shift(rows, rng)
    U_s, V0_s, _ = build_factors(rows_shift, a1)
    sv_shift = np.linalg.svd(V0_s @ U_s, compute_uv=False)

    return dict(sv_true=sv_true.tolist(), sv_time_shift_null=sv_shift.tolist(),
               er_true=effective_ranks(V0_true @ U_true),
               er_shift=effective_ranks(V0_s @ U_s))


def part_d1_cross_seed(seed_u, seed_v, rng_offset=0):
    from credit_memory.b10_tangent_adjoint_theory import build_factors
    _, rows_u = collect_rows(seed_u, N_CAL_TRAJ, offset=0)
    _, rows_v = collect_rows(seed_v, N_CAL_TRAJ, offset=0)
    a1_u = rows_u[0]["a1"]
    rows_cross = null_cross_seed(rows_u, rows_v)
    U, V0_cross, _ = build_factors(rows_cross, a1_u)
    sv_cross = np.linalg.svd(V0_cross @ U, compute_uv=False)
    return dict(sv_cross_seed_null=sv_cross.tolist(),
               er_cross=effective_ranks(V0_cross @ U))


def part_d2_T_scaling(seed, T_list=(30, 60, 120)):
    from credit_memory.teacher import set_l2_config
    from credit_memory.b10_tangent_adjoint_theory import build_factors
    out = {}
    for T_ in T_list:
        with set_l2_config(N, T_, BATCH):
            from toyrig import ssm_rig as tcg
            from credit_memory.teacher import compute_teacher, draw_trajectory
            params = tcg.init_params(seed)
            rows_T = []
            for k in range(N_CAL_TRAJ):
                rng = np.random.RandomState(70000 + seed * 1000 + k)
                x, r = draw_trajectory(params, rng, T_, BATCH)
                rows_T.append(compute_teacher(params, x, r))
        a1 = rows_T[0]["a1"]
        U, V0, _ = build_factors(rows_T, a1)
        VU_norm = V0 @ U / T_
        sv = np.linalg.svd(VU_norm, compute_uv=False)
        out[str(T_)] = dict(sv_normalized=sv.tolist(),
                           er=effective_ranks(VU_norm))
    return out


def main() -> None:
    print("=" * 90)
    print(f"Phase B11: shared/private communication-subspace audit, {len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(a=[], b1=[], b4=[], c2=[], c3=[], c4=[], c5=[])
    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)
        from credit_memory.b10_tangent_adjoint_theory import build_factors
        U, V0_P, V0_Q = build_factors(rows, a1)
        resP = coupling_audit(U, V0_P)
        R = direct_routed(rows, f_diag, B1)
        rho_abs_P = np.abs(R[:N, :])

        per_seed["a"].append(part_a_variance_and_credit(U, V0_P, resP, rho_abs_P, B1))
        per_seed["b1"].append(part_b1_spectral_identity(U, V0_P))
        per_seed["b4"].append(part_b4_energy_vs_credit(resP))

        freqs, C = compute_cross_spectrum(rows)
        per_seed["c2"].append(part_c2_frequency_coherence(freqs, C))
        per_seed["c3"].append(part_c3_parseval_check(rows, C))

        j_rho = {m: int(np.argmax(np.abs(R[:, m]))) for m in range(N)}
        j_pool = pool_most_frequent(j_rho, np.abs(R), 4)
        c4_rows = [part_c4_pole_matching(freqs, C, f_diag, np.abs(R[:, m]),
                                         np.abs(R[:, m]), j_rho[m], j_pool)
                  for m in range(N)]
        per_seed["c4"].append(c4_rows)
        per_seed["c5"].append(part_c5_mode_approx_by_K_poles(freqs, C, f_diag))

        print(f"seed {seed}: B1 rel_err={per_seed['b1'][-1]['rel_err']:.2e}  "
             f"B4 U/V energy-credit spearman={per_seed['b4'][-1]['spearman_U_energy_vs_dominant_credit_loading']:.2f}/"
             f"{per_seed['b4'][-1]['spearman_V_energy_vs_dominant_credit_loading']:.2f}  "
             f"C2 n_strong_freqs={per_seed['c2'][-1]['n_strong_freqs']}  "
             f"A(r=3) R_shared winner={per_seed['a'][-1]['3']['R_shared_decision']['winner_preserved']:.2f}")

    # D: null models + T-scaling, smaller seed subset given cost
    d_seeds = SEEDS[:4]
    d1_results, dcross_results = [], []
    rng_master = np.random.RandomState(4242)
    for seed in d_seeds:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1 = rows[0]["a1"]
        d1_results.append(part_d1_null_models(rows, a1, rng_master))
        other_seed = (seed + 1) % len(SEEDS)
        dcross_results.append(part_d1_cross_seed(seed, other_seed))
    d2_result = part_d2_T_scaling(0)

    def med(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    a_summary = {r: dict(
        median_var_frac_U_shared=med(per_seed["a"], r, "var_frac_U_shared"),
        median_var_frac_V_shared=med(per_seed["a"], r, "var_frac_V_shared"),
        median_VU_energy_frac_shared_shared=med(per_seed["a"], r, "VU_energy_frac_from_shared_shared"),
        median_R_shared_winner=med(per_seed["a"], r, "R_shared_decision", "winner_preserved"),
        median_R_private_winner=med(per_seed["a"], r, "R_private_decision", "winner_preserved"),
        median_R_shared_regret=med(per_seed["a"], r, "R_shared_decision", "pool_regret"),
        median_R_private_regret=med(per_seed["a"], r, "R_private_decision", "pool_regret"))
        for r in [str(x) for x in R_LIST]}

    b1_summary = dict(max_rel_err=max(r["rel_err"] for r in per_seed["b1"]))
    b4_summary = dict(
        median_spearman_U=med(per_seed["b4"], "spearman_U_energy_vs_dominant_credit_loading"),
        median_spearman_V=med(per_seed["b4"], "spearman_V_energy_vs_dominant_credit_loading"))

    c2_summary = dict(median_n_strong_freqs=med(per_seed["c2"], "n_strong_freqs"),
                      median_rank_at_strong_freqs=med(per_seed["c2"], "median_rank_at_strong_freqs"),
                      median_energy_frac_dominant=med(per_seed["c2"], "energy_frac_dominant_freq"))
    c3_summary = dict(max_parseval_rel_err=max(r["parseval_rel_err"] for r in per_seed["c3"]))
    all_c4 = [row for run in per_seed["c4"] for row in run]
    c4_summary = dict(
        median_spearman_match_vs_rho=med(all_c4, "spearman_match_vs_rho"),
        mean_match_in_pool=float(np.mean([r["mean_match_in_pool"] for r in all_c4])),
        mean_match_out_pool=float(np.mean([r["mean_match_out_pool"] for r in all_c4])))
    c5_summary = {k: med(per_seed["c5"], k, "rel_err") for k in ("1", "2", "4")}

    print("-" * 90)
    print("PART A summary:", json.dumps(a_summary, indent=1))
    print("PART B1/B4 summary:", b1_summary, b4_summary)
    print("PART C2/C3/C4/C5 summary:", c2_summary, c3_summary, c4_summary, c5_summary)
    print("PART D1 null (median er):",
         dict(true=med(d1_results, "er_true", "0.9"), shift=med(d1_results, "er_shift", "0.9")))
    print("PART D1 cross-seed (median er 90%):", med(dcross_results, "er_cross", "0.9"))
    print("PART D2 T-scaling:", {k: v["er"] for k, v in d2_result.items()})

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, r_list=R_LIST),
              part_a_summary=a_summary, part_b1_summary=b1_summary,
              part_b4_summary=b4_summary, part_c2_summary=c2_summary,
              part_c3_summary=c3_summary, part_c4_summary=c4_summary,
              part_c5_summary=c5_summary,
              part_d1_null=d1_results, part_d1_cross_seed=dcross_results,
              part_d2_T_scaling=d2_result, per_seed=per_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b11_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
