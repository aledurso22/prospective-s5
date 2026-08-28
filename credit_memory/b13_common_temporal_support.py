"""Phase B13 -- does the invariant rank-2 credit geometry come from a
low-dimensional COMMON TEMPORAL SUPPORT between eligibility (U) and
adjoint-teaching (V) signals, as opposed to marginal amplitude
concentration (ruled out in B12) or specific pole architecture (also
ruled out in B12)? Theory/mechanism only: no new training algorithm,
no S5, no new persistent training arm.

Key identity (B12): after double whitening, U_white=L_U R_U^dagger,
V_white=L_V R_V^dagger, so V_white U_white = L_V (R_V^dagger L_U)
R_U^dagger = L_V C_tc R_U^dagger, and since L_V/R_U^dagger are
isometric, singular_values(V_white U_white) = singular_values(C_tc)
exactly. C_tc = R_V^dagger L_U is the pure temporal-subspace-overlap
matrix (principal-angle cosines).

Run:  python -m credit_memory.b13_common_temporal_support
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
    build_factors, adjoint_filter)
from credit_memory.b10_1_temporal_coupling import svd_compact
from credit_memory.b12_structural_spectral_theory import (
    whiten, phase_randomize, build_V0_with_poles)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, collect_rows)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b13")

K = 4


# ===========================================================================
# PART A: dominant modes of the double-whitened coupling C_tc = R_V^dagger L_U
# ===========================================================================
def part_a_common_modes(U, V0, n_traj, T_, BATCH_, l_max=3):
    Lu, Su, Ruh = svd_compact(U)
    Lv, Sv, Rvh = svd_compact(V0)
    C_tc = Rvh @ Lu
    P, sc, Qh = svd_compact(C_tc)
    Q = np.conj(Qh).T

    modes = []
    for l in range(min(l_max, len(sc))):
        v_profile = P[:, l] @ Rvh          # (TB,) V-side temporal mode
        u_profile = Lu @ Q[:, l]           # (TB,) U-side temporal mode
        v_t = v_profile.reshape(n_traj, T_, BATCH_).mean(axis=(0, 2))
        u_t = u_profile.reshape(n_traj, T_, BATCH_).mean(axis=(0, 2))
        v_fft = np.abs(np.fft.fft(v_t))
        u_fft = np.abs(np.fft.fft(u_t))
        freqs = np.fft.fftfreq(T_)
        v_dom = int(np.argmax(v_fft))
        u_dom = int(np.argmax(u_fft))
        # effective bandwidth: spectral spread (std of freq weighted by power)
        def bandwidth(fft_mag, freqs):
            p = fft_mag ** 2
            p = p / (p.sum() + 1e-300)
            mean_f = np.sum(np.abs(freqs) * p)
            return float(np.sqrt(np.sum((np.abs(freqs) - mean_f) ** 2 * p)))
        modes.append(dict(
            l=l, sigma=float(sc[l]),
            v_dominant_freq=float(freqs[v_dom]), u_dominant_freq=float(freqs[u_dom]),
            v_bandwidth=bandwidth(v_fft, freqs), u_bandwidth=bandwidth(u_fft, freqs),
            v_energy_at_dom=float(v_fft[v_dom] ** 2 / np.sum(v_fft ** 2)),
            u_energy_at_dom=float(u_fft[u_dom] ** 2 / np.sum(u_fft ** 2))))
    # quadrature check: are modes 0,1 the same frequency with a phase shift?
    quad = None
    if len(modes) >= 2:
        same_band = abs(modes[0]["v_dominant_freq"]) - abs(modes[1]["v_dominant_freq"])
        quad = dict(freq_diff_mode0_mode1=float(same_band),
                   likely_quadrature_pair=bool(abs(same_band) < 1e-6
                                               and abs(modes[0]["v_dominant_freq"]) > 1e-6))
    return dict(modes=modes, quadrature_check=quad, sigma_all=sc.tolist())


# ===========================================================================
# PART B: support-destroying nulls (the decisive falsifications)
# ===========================================================================
def random_orthonormal_rows(r, TB, rng):
    """(r, TB) matrix with orthonormal rows, statistically independent
    of any other subspace -- a genuine Haar-random temporal basis."""
    X = rng.randn(TB, r) + 1j * rng.randn(TB, r)
    Qm, _ = np.linalg.qr(X)
    return np.conj(Qm).T


def part_b1_random_rotation(U, V0, rng):
    """V_rot: SAME singular values as V0 (verified), completely random
    (Haar) temporal basis unrelated to U's own directions."""
    Lv, Sv, Rvh = svd_compact(V0)
    TB = V0.shape[1]
    Rvh_rand = random_orthonormal_rows(len(Sv), TB, rng)
    V0_rot = Lv @ np.diag(Sv) @ Rvh_rand
    sv_check = np.linalg.svd(V0_rot, compute_uv=False)
    sv_check_err = float(np.max(np.abs(sv_check - Sv)) / (Sv[0] + 1e-300))
    VU_rot = V0_rot @ U
    return dict(singular_value_preserved_rel_err=sv_check_err,
               sv_rotated=np.linalg.svd(VU_rot, compute_uv=False).tolist(),
               effective_rank=effective_ranks(VU_rot))


def part_b2_frequency_permutation(rows, a1, rng):
    """Permute V's (q1's) frequency-bin content relative to U -- moves
    spectral energy to different bins while preserving each bin's own
    magnitude multiset (hence total power and the marginal amplitude
    histogram), destroying alignment with U's own spectral support."""
    syn_rows = []
    for row in rows:
        q1 = row["q1"]
        Tn, Bn, Nn = q1.shape
        q1f = np.fft.fft(q1, axis=0)
        perm = rng.permutation(Tn)
        q1f_perm = q1f[perm]
        new_row = dict(row)
        new_row["q1"] = np.fft.ifft(q1f_perm, axis=0)
        syn_rows.append(new_row)
    U, V0_perm, _ = build_factors(syn_rows, a1)
    VU_perm = V0_perm @ U
    return dict(sv=np.linalg.svd(VU_perm, compute_uv=False).tolist(),
               effective_rank=effective_ranks(VU_perm))


def band_stop(X, freq_idx_to_zero):
    Xf = np.fft.fft(X, axis=0)
    Xf[freq_idx_to_zero] = 0.0
    return np.fft.ifft(Xf, axis=0)


def part_b3_band_stop(rows, a1, dominant_freq_idx, halfwidth=1):
    Tn = rows[0]["Sa0"].shape[0]
    idx_to_zero = [(dominant_freq_idx + d) % Tn for d in range(-halfwidth, halfwidth + 1)]
    idx_to_zero += [(-dominant_freq_idx + d) % Tn for d in range(-halfwidth, halfwidth + 1)]
    idx_to_zero = sorted(set(idx_to_zero))

    def stopped_rows(stop_u, stop_v):
        out = []
        for row in rows:
            new_row = dict(row)
            if stop_u:
                new_row["Sa0"] = band_stop(row["Sa0"], idx_to_zero)
            if stop_v:
                new_row["q1"] = band_stop(row["q1"], idx_to_zero)
            out.append(new_row)
        return out

    out = {}
    for name, (su, sv) in dict(stop_U=(True, False), stop_V=(False, True),
                               stop_both=(True, True)).items():
        s_rows = stopped_rows(su, sv)
        U_s, V0_s, _ = build_factors(s_rows, a1)
        VU_s = V0_s @ U_s
        out[name] = dict(sv=np.linalg.svd(VU_s, compute_uv=False).tolist(),
                         effective_rank=effective_ranks(VU_s))
    return dict(stopped_bins=idx_to_zero, results=out)


def part_b4_inject_shared_band(rows, a1, new_freq_idx, amplitude_scale=1.0, rng=None):
    """Inject a controlled sinusoidal component at a frequency where
    the true data has little support, on BOTH Sa0 and q1 (all channels
    jointly, matching B12's phase-randomization convention of a shared
    per-realization phase so the injected component IS genuinely
    shared/coherent between the two sides)."""
    if rng is None:
        rng = np.random.RandomState(0)
    Tn = rows[0]["Sa0"].shape[0]
    typical_scale = float(np.mean([np.abs(r["Sa0"]).mean() for r in rows]))
    out_rows = []
    for row in rows:
        Bn = row["Sa0"].shape[1]
        new_row = dict(row)
        Sa0 = row["Sa0"].copy()
        q1 = row["q1"].copy()
        for b in range(Bn):
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(Tn)
            inj = amplitude_scale * typical_scale * np.exp(
                1j * (2 * np.pi * new_freq_idx * t / Tn + phase))
            Sa0[:, b, :] += inj[:, None]
            q1[:, b, :] += inj[:, None]
        new_row["Sa0"] = Sa0
        new_row["q1"] = q1
        out_rows.append(new_row)
    U_inj, V0_inj, _ = build_factors(out_rows, a1)
    VU_inj = V0_inj @ U_inj
    return dict(sv=np.linalg.svd(VU_inj, compute_uv=False).tolist(),
               effective_rank=effective_ranks(VU_inj))


# ===========================================================================
# PART C: genuinely distinct spectral-support task complexity (NOT
# delays, which only change phase, per the task's own falsification
# note: delay by tau -> multiplication by exp(-i omega tau), same
# spectral SUPPORT). r_spectral independent oscillatory input channels
# at DISJOINT frequencies, summed to a scalar target.
# ===========================================================================
def make_multi_freq_task(rng, T_, BATCH_, r_spectral, freqs_cycles):
    t = np.arange(T_)
    x = np.zeros((T_, BATCH_, r_spectral))
    y = np.zeros((T_, BATCH_))
    for k in range(r_spectral):
        f = freqs_cycles[k]
        phases = rng.uniform(0, 2 * np.pi, BATCH_)
        amp = rng.randn(BATCH_)
        sig = np.sin(2 * np.pi * f * t[:, None] / T_ + phases[None, :]) * amp[None, :]
        x[:, :, k] = sig
        y += sig
    return x, y


def collect_rows_multi_freq(seed, n_traj, r_spectral, freqs_cycles, N_=N, T_=T, BATCH_=BATCH):
    from toyrig import ssm_rig as tcg
    from credit_memory.teacher import compute_teacher, set_l2_config
    old_M_IN = tcg.M_IN
    tcg.M_IN = r_spectral
    try:
        with set_l2_config(N_, T_, BATCH_):
            params = tcg.init_params(seed)
            rows = []
            for k in range(n_traj):
                rng = np.random.RandomState(85000 + seed * 1000 + k)
                x, y = make_multi_freq_task(rng, T_, BATCH_, r_spectral, freqs_cycles)
                h, yhat = tcg.forward(params, x)
                r = yhat - y
                rows.append(compute_teacher(params, x, r))
    finally:
        tcg.M_IN = old_M_IN
    return params, rows


def part_c_spectral_complexity(seed, r_spectral_list=(1, 2, 4, 8), K=K):
    all_freqs = [3, 11, 19, 27, 5, 13, 21, 29]     # disjoint bins, T=60
    out = {}
    for r_spec in r_spectral_list:
        freqs = all_freqs[:r_spec]
        params, rows = collect_rows_multi_freq(seed, N_CAL_TRAJ, r_spec, freqs)
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
        out[str(r_spec)] = dict(freqs=freqs, sv=sv.tolist(), effective_rank=er,
                               K_epsilon=min_K)
    return out


# ===========================================================================
# PART F: teaching-dimension (d_teach) variation. spatial_q is exactly
# linear in the residual r (q[L-1]=conj(c)*r, then a linear recursion),
# so d_teach independent residual channels r_1..r_d (same x/Sa0, hence
# same eligibility, but genuinely distinct target spectra) combine as
# q1_combined = sum_k spatial_q(r_k) -- exactly what a genuine combined
# residual sum_k r_k would give, without modifying the core toy code.
# ===========================================================================
def collect_rows_multi_teaching(seed, n_traj, d_teach, freqs_cycles, N_=N, T_=T, BATCH_=BATCH):
    from toyrig import ssm_rig as tcg
    from credit_memory.teacher import compute_teacher, set_l2_config
    with set_l2_config(N_, T_, BATCH_):
        params = tcg.init_params(seed)
        rows = []
        for k in range(n_traj):
            rng = np.random.RandomState(95000 + seed * 1000 + k)
            x = rng.randn(T_, BATCH_)
            h, yhat = tcg.forward(params, x)
            combined_q1, base_row = None, None
            for f in freqs_cycles[:d_teach]:
                t_arr = np.arange(T_)
                phases = rng.uniform(0, 2 * np.pi, BATCH_)
                y_k = np.sin(2 * np.pi * f * t_arr[:, None] / T_ + phases[None, :])
                r_k = yhat - y_k
                row_k = compute_teacher(params, x, r_k)
                if combined_q1 is None:
                    combined_q1 = row_k["q1"].copy()
                    base_row = row_k
                else:
                    combined_q1 = combined_q1 + row_k["q1"]
            base_row = dict(base_row)
            base_row["q1"] = combined_q1
            rows.append(base_row)
    return params, rows


def part_f_teaching_dimension(seed, d_teach_list=(1, 2, 4, 8), K=K):
    all_freqs = [3, 11, 19, 27, 5, 13, 21, 29]
    out = {}
    for d_teach in d_teach_list:
        params, rows = collect_rows_multi_teaching(seed, N_CAL_TRAJ, d_teach, all_freqs)
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
        out[str(d_teach)] = dict(sv=sv.tolist(), effective_rank=er, K_epsilon=min_K)
    return out


def main() -> None:
    print("=" * 90)
    print(f"Phase B13: common temporal support audit, {len(SEEDS)} seeds")
    print("=" * 90)

    per_seed = dict(a=[], b1=[], b2=[], b3=[], b4=[])
    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1 = rows[0]["a1"]
        U, V0_P, V0_Q = build_factors(rows, a1)

        per_seed["a"].append(part_a_common_modes(U, V0_P, N_CAL_TRAJ, T, BATCH))
        rng = np.random.RandomState(2000 + seed)
        per_seed["b1"].append(part_b1_random_rotation(U, V0_P, rng))
        per_seed["b2"].append(part_b2_frequency_permutation(rows, a1, np.random.RandomState(3000 + seed)))
        dom_freq_idx = int(np.argmin(np.abs(
            np.fft.fftfreq(T) - per_seed["a"][-1]["modes"][0]["v_dominant_freq"])))
        per_seed["b3"].append(part_b3_band_stop(rows, a1, dom_freq_idx))
        per_seed["b4"].append(part_b4_inject_shared_band(rows, a1, new_freq_idx=20,
                                                         amplitude_scale=0.01,
                                                         rng=np.random.RandomState(4000 + seed)))

        print(f"seed {seed}: true_er90={effective_ranks(V0_P @ U)['0.9']}  "
             f"B1_rotated_er90={per_seed['b1'][-1]['effective_rank']['0.9']}  "
             f"B2_permuted_er90={per_seed['b2'][-1]['effective_rank']['0.9']}  "
             f"B3_stop_both_er90={per_seed['b3'][-1]['results']['stop_both']['effective_rank']['0.9']}  "
             f"B4_inject_er90={per_seed['b4'][-1]['effective_rank']['0.9']}")

    c_seeds = SEEDS[:3]
    c_results = {seed: part_c_spectral_complexity(seed) for seed in c_seeds}
    f_seeds = SEEDS[:3]
    f_results = {seed: part_f_teaching_dimension(seed) for seed in f_seeds}

    def med(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    b_summary = dict(
        median_b1_er90=med(per_seed["b1"], "effective_rank", "0.9"),
        median_b1_sv_preserved_err=max(r["singular_value_preserved_rel_err"] for r in per_seed["b1"]),
        median_b2_er90=med(per_seed["b2"], "effective_rank", "0.9"),
        median_b3_stopU_er90=med(per_seed["b3"], "results", "stop_U", "effective_rank", "0.9"),
        median_b3_stopV_er90=med(per_seed["b3"], "results", "stop_V", "effective_rank", "0.9"),
        median_b3_stopBoth_er90=med(per_seed["b3"], "results", "stop_both", "effective_rank", "0.9"),
        median_b4_er90=med(per_seed["b4"], "effective_rank", "0.9"))

    c_summary = {}
    for r_spec in (1, 2, 4, 8):
        ers = [c_results[s][str(r_spec)]["effective_rank"]["0.9"] for s in c_seeds]
        keps = [c_results[s][str(r_spec)]["K_epsilon"]["0.05"] for s in c_seeds]
        c_summary[str(r_spec)] = dict(median_er90=float(np.median(ers)),
                                      median_K_eps5=float(np.median(keps)))

    f_summary = {}
    for d_teach in (1, 2, 4, 8):
        ers = [f_results[s][str(d_teach)]["effective_rank"]["0.9"] for s in f_seeds]
        keps = [f_results[s][str(d_teach)]["K_epsilon"]["0.05"] for s in f_seeds]
        f_summary[str(d_teach)] = dict(median_er90=float(np.median(ers)),
                                       median_K_eps5=float(np.median(keps)))

    a_summary = dict(
        median_mode0_v_freq=med(per_seed["a"], "modes", 0, "v_dominant_freq"),
        median_mode1_v_freq=med(per_seed["a"], "modes", 1, "v_dominant_freq"),
        n_quadrature_pairs=sum(1 for r in per_seed["a"]
                               if r["quadrature_check"] and r["quadrature_check"]["likely_quadrature_pair"]))

    print("-" * 90)
    print("PART A summary:", json.dumps(a_summary, indent=1))
    print("PART B summary:", json.dumps(b_summary, indent=1))
    print("PART C summary:", json.dumps(c_summary, indent=1))
    print("PART F summary:", json.dumps(f_summary, indent=1))

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, c_seeds=c_seeds, f_seeds=f_seeds),
              part_a_summary=a_summary, part_b_summary=b_summary,
              part_c_summary=c_summary, part_f_summary=f_summary,
              per_seed=per_seed, c_results=c_results, f_results=f_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b13_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
