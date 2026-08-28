"""Phase B16 -- exact invariant/Krylov credit closure under tied
recurrent dynamics. NOT compression: this phase tests whether an
EXACT (not approximate) reduction in persistent credit-state count is
possible when multiple upper-layer channels are TIED to the same pole,
since tied recursion + tied input implies IDENTICAL trajectories by
uniqueness of the linear recursion -- a purely algebraic fact, not an
approximation.

Run:  python -m credit_memory.b16_tied_pole_exact_closure
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np
from scipy import stats

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.full_causal import full_causal_gradient
from credit_memory.b10_tangent_adjoint_theory import forward_filter
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config
from credit_memory.phase_b2bc_hankel_truncation import N, T, BATCH, SEEDS

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b16")


# ===========================================================================
# PART A: exact grouped P/Q reduction
# ===========================================================================
def make_grouped_poles(N_, G, rng):
    """G distinct complex poles (magnitude from the established
    u0=linspace(0.90,0.995) grid, phase random), each assigned to
    ceil(N_/G) upper-layer channels (channel -> group map g)."""
    mags = np.linspace(0.90, 0.995, G)
    phases = rng.uniform(-np.pi, np.pi, G)
    mu = mags * np.exp(1j * phases)
    g_of_j = np.array([j % G for j in range(N_)])
    a1_grouped = mu[g_of_j]
    return a1_grouped, g_of_j, mu


def grouped_gradient(mu, g_of_j, B1, N_, q1, Sa0):
    """p_t[g,m] = mu_g p_{t-1}[g,m] + u_t^m (ONE state per (g,m), not
    per (j,m)); G_P[m] = 0.5 sum_g sum_t p_t[g,m] [sum_{j:g(j)=g}
    B1[j,m] conj(q1_t[j])] + Q-branch analogue (repo's exact
    conjugation convention, matching full_causal_gradient)."""
    Tn, Bn = Sa0.shape[0], Sa0.shape[1]
    G = len(mu)
    Ga = np.zeros(N_, np.complex128)
    for m in range(N_):
        u_t = Sa0[:, :, m]                     # (T,BATCH)
        # aggregate routing*conj(q1) per group, per t (this is the
        # O(N_upper) pass over B/q1 the task's Part B expects)
        agg_P = np.zeros((Tn, Bn, G), np.complex128)   # sum_j in g: B[j,m] conj(q1_t[j])
        agg_Q = np.zeros((Tn, Bn, G), np.complex128)   # sum_j in g: conj(B[j,m]) q1_t[j]
        for j in range(N_):
            g = g_of_j[j]
            agg_P[:, :, g] += B1[j, m] * np.conj(q1[:, :, j])
            agg_Q[:, :, g] += np.conj(B1[j, m]) * q1[:, :, j]
        p = np.zeros((Tn, Bn, G), np.complex128)
        q_st = np.zeros((Tn, Bn, G), np.complex128)
        prevP = np.zeros((Bn, G), np.complex128)
        prevQ = np.zeros((Bn, G), np.complex128)
        for t in range(Tn):
            prevP = mu[None, :] * prevP + u_t[t][:, None]
            prevQ = np.conj(mu)[None, :] * prevQ + u_t[t][:, None]
            p[t] = prevP
            q_st[t] = prevQ
        Ga[m] = 0.5 * np.sum(p * agg_P) + 0.5 * np.sum(q_st * agg_Q)
    return Ga


def part_a_exact_check(seed, N_, G, T_=T, BATCH_=BATCH):
    rng = np.random.RandomState(2000 + seed)
    with set_l2_config(N_, T_, BATCH_):
        params = tcg.init_params(seed)
        a1_grouped, g_of_j, mu = make_grouped_poles(N_, G, rng)
        params["a"][1] = a1_grouped
        x, r = draw_trajectory(params, np.random.RandomState(3000 + seed), T_, BATCH_)
        row = compute_teacher(params, x, r)

    a1, B1 = row["a1"], row["B1"]
    f_diag = build_F(a1)
    Ga_full, Gb_full = full_causal_gradient(a1, B1, N_, row["q1"], row["Sa0"], row["Sb0"])
    Ga_grouped = grouped_gradient(mu, g_of_j, B1, N_, row["q1"], row["Sa0"])
    Ga_bptt = row["G_bptt"]

    err_full_vs_grouped = float(np.max(np.abs(Ga_full - Ga_grouped))
                               / (np.max(np.abs(Ga_full)) + 1e-300))
    err_grouped_vs_bptt = float(np.max(np.abs(Ga_grouped - Ga_bptt))
                               / (np.max(np.abs(Ga_bptt)) + 1e-300))
    return dict(N=N_, G=G, err_full_vs_grouped=err_full_vs_grouped,
               err_grouped_vs_bptt=err_grouped_vs_bptt,
               full_state_count=2 * N_ * N_, grouped_state_count=2 * G * N_)


# ===========================================================================
# PART B: true compute/memory cost -- grouped readout stays O(N_upper
# N_lower) (one pass over B/q1), NOT O(G N_upper N_lower).
# ===========================================================================
def part_b_cost(seed, N_, G, T_=T, BATCH_=BATCH, n_repeat=3):
    import time
    rng = np.random.RandomState(2000 + seed)
    with set_l2_config(N_, T_, BATCH_):
        params = tcg.init_params(seed)
        a1_grouped, g_of_j, mu = make_grouped_poles(N_, G, rng)
        params["a"][1] = a1_grouped
        x, r = draw_trajectory(params, np.random.RandomState(3000 + seed), T_, BATCH_)
        row = compute_teacher(params, x, r)
    a1, B1 = row["a1"], row["B1"]

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        full_causal_gradient(a1, B1, N_, row["q1"], row["Sa0"], row["Sb0"])
    t_full = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        grouped_gradient(mu, g_of_j, B1, N_, row["q1"], row["Sa0"])
    t_grouped = (time.perf_counter() - t0) / n_repeat

    return dict(N=N_, G=G, t_full_s=t_full, t_grouped_s=t_grouped,
               full_state_count=2 * N_ * N_, grouped_state_count=2 * G * N_)


# ===========================================================================
# PART E: Krylov complexity K(A,B) = span{B, AB, A^2B, ...}
# ===========================================================================
def krylov_dimension(A_diag, B, max_iters=None, tol=1e-9):
    M = len(A_diag)
    max_iters = max_iters if max_iters is not None else M
    cols = [B]
    cur = B.copy()
    for _ in range(max_iters):
        cur = A_diag[:, None] * cur
        cols.append(cur)
    Kmat = np.concatenate(cols, axis=1)
    sv = np.linalg.svd(Kmat, compute_uv=False)
    return int(np.sum(sv > tol * (sv[0] + 1e-300)))


def part_e_krylov(M, rng):
    out = {}

    # E1: M distinct diagonal poles, low-rank (r_in=1) generic B
    A_distinct = (0.5 + 0.4 * rng.rand(M)) * np.exp(1j * rng.uniform(-np.pi, np.pi, M))
    B_lowrank = (rng.randn(M, 1) + 1j * rng.randn(M, 1))
    out["E1_distinct_lowrank_B"] = dict(
        dim=krylov_dimension(A_distinct, B_lowrank), M=M, r_in=1,
        prediction="low-rank B does NOT cap Krylov dim under distinct poles")

    # E2: G tied pole groups, generic B; verify K(A,B) = sum_g rank(Pi_g B)
    for G in (1, 2, 4, M // 2 if M >= 4 else M):
        g_of_j = np.array([j % G for j in range(M)])
        mags = np.linspace(0.5, 0.95, G)
        phases = rng.uniform(-np.pi, np.pi, G)
        mu = mags * np.exp(1j * phases)
        A_grouped = mu[g_of_j]
        B_generic = rng.randn(M, 2) + 1j * rng.randn(M, 2)
        dim_measured = krylov_dimension(A_grouped, B_generic, max_iters=2 * M)
        dim_formula = sum(np.linalg.matrix_rank(B_generic[g_of_j == g])
                          for g in range(G))
        out[f"E2_G{G}"] = dict(dim_measured=dim_measured, dim_formula=int(dim_formula),
                              match=bool(dim_measured == dim_formula), G=G, M=M)

    # E4: B's column space EXACTLY spans a union of eigenspaces (a
    # genuine A-invariant subspace) -- Krylov dim should equal rank(U)
    # regardless of iteration count.
    G4 = 4
    g_of_j = np.array([j % G4 for j in range(M)])
    mags = np.linspace(0.5, 0.95, G4)
    A_g4 = mags[g_of_j] * np.exp(1j * rng.uniform(-np.pi, np.pi, G4))[g_of_j]
    invariant_groups = [0, 1]                       # union of 2 of the 4 eigenspaces
    mask = np.isin(g_of_j, invariant_groups)
    U_dim_expected = int(mask.sum())
    # B_invariant spans the FULL invariant subspace U (rank = |U|), not
    # just a single vector within it -- otherwise Krylov dim is capped
    # by rank(B) itself (already covered by E2's formula), not by the
    # invariance property being tested here.
    B_invariant = np.zeros((M, U_dim_expected), np.complex128)
    B_invariant[mask, :] = (rng.randn(mask.sum(), U_dim_expected)
                           + 1j * rng.randn(mask.sum(), U_dim_expected))
    dim_e4 = krylov_dimension(A_g4, B_invariant, max_iters=3 * M)
    out["E4_invariant_subspace"] = dict(dim_measured=dim_e4, dim_expected=U_dim_expected,
                                        match=bool(dim_e4 == U_dim_expected))
    return out


# ===========================================================================
# PART G: selectivity counterexample -- input-dependent scalar a_t(x_t)
# destroys the tied-pole closure because d(a_t)/d(theta) * h_{t-1}
# introduces a NEW, time-varying spatial direction into the sensitivity.
# ===========================================================================
def selective_scalar_sensitivity(seed, M, Tn, stop_grad_through_a=False):
    """h_t = a_t(x_t) h_{t-1} + x_t, A_t = a_t I (scalar-times-identity).
    x_t is a genuinely M-dimensional exogenous drive (so h_t explores
    the full state space over time, NOT confined to a fixed 1-D
    direction -- required for the selectivity term to have anything
    non-trivial to act on). theta[0] feeds the selective gate a_t;
    theta[1] scales the overall drive x_t. Measures the rank of
    d h_T/d theta_0 across many independent time-window realizations
    (finite-difference against direct simulation, no autodiff library
    needed at this toy scale) -- the "reachable spatial directions"
    proxy for the sensitivity's own state-space footprint."""
    rng = np.random.RandomState(seed)
    theta = np.array([0.3, 1.0])
    n_windows = max(8, M // 4)
    x_all = [rng.randn(Tn, M) * 0.3 for _ in range(n_windows)]
    exo_gate_signal_all = [rng.randn(Tn) * 0.3 for _ in range(n_windows)]

    def run(theta_local, selective, x, exo_gate_signal, a_t_fixed=None):
        h = np.zeros((Tn, M), np.complex128)
        prev = np.zeros(M, np.complex128)
        a_seq = np.zeros(Tn)
        for t in range(Tn):
            if a_t_fixed is not None:
                a_t = a_t_fixed[t]     # stop-gradient: reuse BASE theta's a_t, never
                                       # recomputed from theta_local -- theta_0's path
                                       # through a_t is genuinely severed, not just cast
            else:
                if selective:
                    # gate depends on theta AND on the evolving state h_{t-1}
                    gate_signal = np.real(np.mean(prev)) if t > 0 else 0.0
                else:
                    # gate depends on theta but on a FIXED exogenous signal,
                    # never on h_{t-1} -- the fair "genuine effect, no
                    # feedback" contrast
                    gate_signal = exo_gate_signal[t]
                a_t = 0.5 + 0.45 * np.tanh(theta_local[0] * (0.5 + gate_signal))
            a_seq[t] = a_t
            drive = theta_local[1] * x[t]
            prev = a_t * prev + drive
            h[t] = prev
        return h, a_seq

    def d_dtheta0(selective, x, exo_gate, eps=1e-5):
        base, a_seq_base = run(theta, selective, x, exo_gate)
        th2 = theta.copy()
        th2[0] += eps
        a_fixed = a_seq_base if stop_grad_through_a else None
        bumped, _ = run(th2, selective, x, exo_gate, a_t_fixed=a_fixed)
        return (bumped - base) / eps

    grads_exo = np.stack([d_dtheta0(False, x, g)[-1]
                          for x, g in zip(x_all, exo_gate_signal_all)], axis=0)
    grads_sel = np.stack([d_dtheta0(True, x, g)[-1]
                          for x, g in zip(x_all, exo_gate_signal_all)], axis=0)

    def eff_rank(mat, thresh=0.95):
        if np.linalg.norm(mat) < 1e-12:
            return 0
        sv = np.linalg.svd(mat, compute_uv=False)
        cum = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-300)
        return int(np.searchsorted(cum, thresh) + 1)

    result = dict(rank_exogenous=eff_rank(grads_exo), rank_selective=eff_rank(grads_sel),
                 norm_exogenous=float(np.linalg.norm(grads_exo)),
                 norm_selective=float(np.linalg.norm(grads_sel)), M=M)

    if not stop_grad_through_a:
        # G3: selective gate, but stop-gradient through a_t itself --
        # theta_0's path via (d a_t/d theta_0) h_{t-1} is explicitly cut,
        # an approximate control showing what closure looks like if that
        # specific term is dropped.
        sg = selective_scalar_sensitivity(seed, M, Tn, stop_grad_through_a=True)
        result["rank_selective_stopgrad"] = sg["rank_selective"]
        result["norm_selective_stopgrad"] = sg["norm_selective"]
    return result


def main() -> None:
    print("=" * 90)
    print("Phase B16: exact tied-pole credit closure")
    print("=" * 90)

    print("\nPart A: exact grouped P/Q vs full P/Q vs BPTT")
    a_results = []
    for N_ in (6, 12, 24, 48):
        for G in (1, 2, 4, N_):
            for seed in (0, 1, 2):
                r = part_a_exact_check(seed, N_, G)
                a_results.append(r)
        rs = [r for r in a_results if r["N"] == N_]
        max_err = max(max(r["err_full_vs_grouped"], r["err_grouped_vs_bptt"]) for r in rs)
        print(f"  N={N_}: max error across all G,seeds = {max_err:.2e}")

    print("\nPart B: cost/state-count audit")
    b_results = [part_b_cost(0, N_, 1) for N_ in (6, 12, 24, 48)]
    for r in b_results:
        print(f"  N={r['N']}: full_states={r['full_state_count']} "
             f"grouped_states={r['grouped_state_count']} "
             f"(reduction {r['full_state_count'] / r['grouped_state_count']:.1f}x)  "
             f"t_full={r['t_full_s']:.4f}s t_grouped={r['t_grouped_s']:.4f}s")

    print("\nPart E: Krylov complexity")
    rng = np.random.RandomState(0)
    e_results = {M: part_e_krylov(M, rng) for M in (12, 24, 48)}
    for M, res in e_results.items():
        all_match = all(v.get("match", True) for v in res.values())
        print(f"  M={M}: E1_dim={res['E1_distinct_lowrank_B']['dim']} "
             f"(expect {M}), all formula matches={all_match}")

    print("\nPart G: selectivity counterexample")
    g_results = {M: selective_scalar_sensitivity(0, M, 60) for M in (8, 32, 128)}
    for M, res in g_results.items():
        print(f"  M={M}: rank_exo={res['rank_exogenous']} rank_sel={res['rank_selective']} "
             f"rank_sel_stopgrad={res['rank_selective_stopgrad']} "
             f"norm_sel_stopgrad={res['norm_selective_stopgrad']:.2e}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, part_a=a_results, part_b=b_results,
              part_e={str(k): v for k, v in e_results.items()},
              part_g={str(k): v for k, v in g_results.items()})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b16_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
