"""Phase B10 -- tangent/adjoint factorization theory audit. Mechanistic
/theory verification only: no new training algorithm, no S5.

Convention (repo's actual complex/Hermitian convention, unchanged from
Phase A/B1-B9): for candidate j with pole lambda_j = f_diag[j] (P-branch
j<N: lambda_j=a1[j]; Q-branch j=N+j', j'<N: lambda_j=conj(a1[j'])):

  forward filter   x_{j,t} = lambda_j x_{j,t-1} + u_t,      x_{j,-1}=0
  rho_forward[j,m] = sum_t conj(c_{j,t,m}) x_{j,t,m}

Hermitian-adjoint identity: for the causal, lower-triangular Toeplitz
convolution operator H_lambda (H u)_t = sum_{t'<=t} lambda^{t-t'} u_t',
the adjoint is (H^dagger c)_t = sum_{t'>=t} conj(lambda)^{t'-t} c_t',
computed by the BACKWARD recursion

  p_{j,t} = c_{j,t} + conj(lambda_j) p_{j,t+1},   p_{j,T}=0

so that  <c, H u> = <H^dagger c, u>,  i.e.
  rho_adjoint[j,m] = sum_t conj(p_{j,t,m}) u_{t,m} == rho_forward[j,m].

Routing factorization (exact, from credit_memory/hankel.py::build_c_t):
  P-branch: c_{j,t,m} = 0.5 conj(B1[j,m]) * q1_t[j]
  Q-branch: c_{j,t,m} = 0.5 B1[j',m]       * conj(q1_t[j'])
c's mode-m-dependence factors through a SCALAR (routing weight) times a
mode-INDEPENDENT signal (q1[:,j] or conj(q1[:,j'])). Since the adjoint
recursion is linear in c, this scalar factors straight through:
  v_j(m) = 0.5 conj(B1[j,m]) * v0_j    (P), v0_j := H_j^dagger q1[:,j]
  v_j(m) = 0.5 B1[j',m]       * v0_j    (Q), v0_j := H_j^dagger conj(q1[:,j'])
giving, with V0[j,:] := conj(v0_j)^T, U[:,m] := u^m (both flattened over
(T, BATCH), each of the N_CAL_TRAJ calibration trajectories processed
SEPARATELY with its own forward/backward boundary conditions before
concatenating -- no cross-trajectory leakage, per PHASE_B9.md):
  R0_P = V0_P U,           R_P = 0.5 * B1 * R0_P            (elementwise)
  R0_Q = V0_Q U,           R_Q = 0.5 * conj(B1) * R0_Q      (elementwise)

Run:  python -m credit_memory.b10_tangent_adjoint_theory
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
from math import comb

import numpy as np
from scipy import stats
from scipy.linalg import qr as scipy_qr

from credit_memory.hankel import build_F, build_c_t
from credit_memory.b9_2_shared_pool import best_pool_exact
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np, relerr_np)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b10")

K_LIST = [1, 2, 4, 8]
R_LIST = [1, 2, 4]


# ---------------------------------------------------------------------------
# Part A: forward filter, adjoint (backward) filter, and the matrix identity
# ---------------------------------------------------------------------------
def forward_filter(lam, u_t):
    """u_t: (T,BATCH) -> x: (T,BATCH), x_{-1}=0."""
    Tn, Bn = u_t.shape
    x = np.zeros((Tn, Bn), np.complex128)
    prev = np.zeros(Bn, np.complex128)
    for t in range(Tn):
        prev = lam * prev + u_t[t]
        x[t] = prev
    return x


def adjoint_filter(lam, c_t):
    """c_t: (T,BATCH) -> p: (T,BATCH), p_T=0, backward recursion
    p_t = c_t + conj(lam) p_{t+1}."""
    Tn, Bn = c_t.shape
    p = np.zeros((Tn, Bn), np.complex128)
    nxt = np.zeros(Bn, np.complex128)
    for t in range(Tn - 1, -1, -1):
        nxt = c_t[t] + np.conj(lam) * nxt
        p[t] = nxt
    return p


def part_a_identity_check(rows, f_diag, B1):
    """For every candidate j and lower mode m, verify rho_forward ==
    rho_adjoint, summed correctly across trajectories (no cross-
    trajectory leakage: each row's own forward/backward recursion is
    independently zero-initialized/zero-terminated)."""
    errs = []
    for m in range(N):
        rho_fwd = np.zeros(2 * N, np.complex128)
        rho_adj = np.zeros(2 * N, np.complex128)
        for row in rows:
            c_full = build_c_t(row["q1"], row["B1"][:, m])   # (T,BATCH,2N)
            u_t = row["Sa0"][:, :, m]                          # (T,BATCH)
            for j in range(2 * N):
                lam = f_diag[j]
                c_j = c_full[:, :, j]
                x_j = forward_filter(lam, u_t)
                rho_fwd[j] += np.sum(np.conj(c_j) * x_j)
                p_j = adjoint_filter(lam, c_j)
                rho_adj[j] += np.sum(np.conj(p_j) * u_t)
        abs_err = np.max(np.abs(rho_fwd - rho_adj))
        rel_err = abs_err / (np.max(np.abs(rho_fwd)) + 1e-300)
        errs.append(dict(mode=m, abs_err=float(abs_err), rel_err=float(rel_err)))
    return errs


def part_a_matrix_identity_check(seed=0):
    """<c, H u> = <H^dagger c, u> via explicit (T x T) Toeplitz matrices,
    random c/u, independent of the recursive implementation above."""
    rng = np.random.RandomState(seed)
    lam = 0.6 * np.exp(1j * 1.1)
    Tn = 12
    tt, tp = np.meshgrid(np.arange(Tn), np.arange(Tn), indexing="ij")
    H = np.where(tt >= tp, lam ** (tt - tp), 0.0).astype(np.complex128)
    Hdag = np.conj(H).T
    c = rng.randn(Tn) + 1j * rng.randn(Tn)
    u = rng.randn(Tn) + 1j * rng.randn(Tn)
    lhs = np.vdot(c, H @ u)          # <c, Hu> = conj(c).(Hu)
    rhs = np.vdot(Hdag @ c, u)       # <H^dagger c, u>
    return dict(lhs=complex(lhs), rhs=complex(rhs),
               abs_err=float(abs(lhs - rhs)),
               rel_err=float(abs(lhs - rhs) / (abs(lhs) + 1e-300)))


# ---------------------------------------------------------------------------
# Part B: factor matrices U, V0_P, V0_Q (each column/row = concatenation
# over the N_CAL_TRAJ calibration trajectories, each trajectory's own
# forward/backward recursion independently boundary-conditioned -- no
# cross-trajectory leakage, per PHASE_B9.md).
# ---------------------------------------------------------------------------
def build_factors(rows, a1):
    """Returns U (TB, N), V0_P (N, TB), V0_Q (N, TB), TB = N_CAL_TRAJ*T*BATCH."""
    U_blocks = {m: [] for m in range(N)}
    V0P_blocks = {j: [] for j in range(N)}
    V0Q_blocks = {j: [] for j in range(N)}
    for row in rows:
        Sa0, q1 = row["Sa0"], row["q1"]
        for m in range(N):
            U_blocks[m].append(Sa0[:, :, m].reshape(-1))
        for j in range(N):
            p_P = adjoint_filter(a1[j], q1[:, :, j])
            V0P_blocks[j].append(np.conj(p_P).reshape(-1))
            p_Q = adjoint_filter(np.conj(a1[j]), np.conj(q1[:, :, j]))
            V0Q_blocks[j].append(np.conj(p_Q).reshape(-1))
    U = np.stack([np.concatenate(U_blocks[m]) for m in range(N)], axis=1)
    V0_P = np.stack([np.concatenate(V0P_blocks[j]) for j in range(N)], axis=0)
    V0_Q = np.stack([np.concatenate(V0Q_blocks[j]) for j in range(N)], axis=0)
    return U, V0_P, V0_Q


def routed_from_factors(U, V0_P, V0_Q, B1):
    R0_P = V0_P @ U          # (N,N)
    R0_Q = V0_Q @ U
    R_P = 0.5 * B1 * R0_P            # elementwise, B1[j,m]
    R_Q = 0.5 * np.conj(B1) * R0_Q
    return R0_P, R0_Q, R_P, R_Q


def direct_routed(rows, f_diag, B1):
    """Direct per-candidate rho via the forward filter (bit-identical to
    per_coordinate_contribution's own construction), for comparison
    against the VU factorization."""
    rho = np.zeros((2 * N, N), np.complex128)
    for m in range(N):
        for row in rows:
            c_full = build_c_t(row["q1"], row["B1"][:, m])
            u_t = row["Sa0"][:, :, m]
            for j in range(2 * N):
                x_j = forward_filter(f_diag[j], u_t)
                rho[j, m] += np.sum(np.conj(c_full[:, :, j]) * x_j)
    return rho


# ---------------------------------------------------------------------------
# Part C: effective-rank audit
# ---------------------------------------------------------------------------
def effective_ranks(M, fracs=(0.90, 0.95, 0.99)):
    s = np.linalg.svd(M, compute_uv=False)
    sq = s ** 2
    total = sq.sum()
    if total <= 0:
        return {str(f): int(min(M.shape)) for f in fracs}
    cum = np.cumsum(sq) / total
    return {str(f): int(np.searchsorted(cum, f) + 1) for f in fracs}


def algebraic_rank(M, tol=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * (s[0] + 1e-300)))


def part_c_rank_audit(U, V0_P, V0_Q, B1, R0_P, R0_Q, R_P, R_Q, rho_full):
    mats = dict(U=U, V_P=V0_P, V_Q=V0_Q, V_P_U=R0_P, V_Q_U=R0_Q, B=B1,
               R_P_routed=R_P, R_Q_routed=R_Q, rho_abs=np.abs(rho_full))
    out = {name: dict(effective_rank=effective_ranks(M),
                      algebraic_rank=algebraic_rank(M))
          for name, M in mats.items()}
    out["rank_bound_R0_P_holds"] = bool(
        out["V_P_U"]["algebraic_rank"] <= min(out["U"]["algebraic_rank"],
                                              out["V_P"]["algebraic_rank"]))
    out["rank_bound_R0_Q_holds"] = bool(
        out["V_Q_U"]["algebraic_rank"] <= min(out["U"]["algebraic_rank"],
                                              out["V_Q"]["algebraic_rank"]))
    return out


# ---------------------------------------------------------------------------
# Part D: low-rank U/V truncation tests
# ---------------------------------------------------------------------------
def low_rank_trunc(M, r):
    Um, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, len(s))
    return Um[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]


def decision_metrics(s_true, s_hat, K=4):
    """s_*: (2N,N) real |rho|-like score matrices."""
    winners_true = np.argmax(s_true, axis=0)
    winners_hat = np.argmax(s_hat, axis=0)
    winner_preserved = float(np.mean(winners_true == winners_hat))
    topk_recall = []
    for m in range(N):
        tt = set(np.argsort(-s_true[:, m])[:K].tolist())
        hh = set(np.argsort(-s_hat[:, m])[:K].tolist())
        topk_recall.append(len(tt & hh) / K)
    U_true = s_true       # here s IS the objective per-candidate score
    pool_true = best_pool_exact(U_true, K)
    pool_hat = best_pool_exact(s_hat, K)
    F_true_at_true = sum(U_true[list(pool_true), m].max() for m in range(N))
    F_true_at_hat = sum(U_true[list(pool_hat), m].max() for m in range(N))
    regret = F_true_at_true - F_true_at_hat
    jacc = len(set(pool_true) & set(pool_hat)) / len(set(pool_true) | set(pool_hat))
    return dict(winner_preserved=winner_preserved,
               mean_topk_recall=float(np.mean(topk_recall)),
               pool_jaccard=jacc, pool_regret=regret)


def part_d_low_rank_tests(U, V0_P, V0_Q, B1, rho_true_abs):
    results = {}
    for r in R_LIST:
        U_r = low_rank_trunc(U, r)
        VP_r = low_rank_trunc(V0_P, r)
        VQ_r = low_rank_trunc(V0_Q, r)
        variants = dict(
            D1_compress_U_only=(V0_P, U_r, V0_Q, U_r),
            D2_compress_V_only=(VP_r, U, VQ_r, U),
            D3_compress_both=(VP_r, U_r, VQ_r, U_r))
        results[str(r)] = {}
        for name, (vp, u, vq, u2) in variants.items():
            R0_P_hat, R0_Q_hat = vp @ u, vq @ u2
            R_P_hat = 0.5 * B1 * R0_P_hat
            R_Q_hat = 0.5 * np.conj(B1) * R0_Q_hat
            rho_hat = np.concatenate([R_P_hat, R_Q_hat], axis=0)
            s_hat = np.abs(rho_hat)
            fro_err = float(np.linalg.norm(s_hat - rho_true_abs)
                            / (np.linalg.norm(rho_true_abs) + 1e-300))
            max_err = float(np.max(np.abs(s_hat - rho_true_abs)))
            sp = stats.spearmanr(s_hat.ravel(), rho_true_abs.ravel()).statistic
            dm = decision_metrics(rho_true_abs, s_hat)
            results[str(r)][name] = dict(fro_rel_err=fro_err, max_abs_err=max_err,
                                         spearman=float(sp), **dm)
    return results


# ---------------------------------------------------------------------------
# Part E: decision-preservation margin test
# ---------------------------------------------------------------------------
def part_e_margin_test(s_true, s_hat):
    order = np.argsort(-s_true, axis=0)
    best, second = order[0, :], order[1, :]
    delta = s_true[best, np.arange(N)] - s_true[second, np.arange(N)]
    eps = np.max(np.abs(s_hat - s_true), axis=0)     # per-mode max error
    condition_holds = delta > 2 * eps
    winners_true = np.argmax(s_true, axis=0)
    winners_hat = np.argmax(s_hat, axis=0)
    preserved = winners_true == winners_hat
    violations = int(np.sum(condition_holds & ~preserved))
    return dict(n_modes=N, n_condition_holds=int(condition_holds.sum()),
               n_violations=violations,
               frac_preserved_overall=float(np.mean(preserved)),
               frac_certified_by_margin=float(np.mean(condition_holds)),
               delta=delta.tolist(), eps=eps.tolist())


# ---------------------------------------------------------------------------
# Part F: pool-objective perturbation bounds + greedy pool selection
# ---------------------------------------------------------------------------
def greedy_pool(score_mat, K):
    n2 = score_mat.shape[0]
    P, remaining = [], set(range(n2))
    for _ in range(K):
        best_j, best_val = None, -np.inf
        for j in remaining:
            trial = P + [j]
            val = float(sum(score_mat[trial, m].max() for m in range(N)))
            if val > best_val:
                best_val, best_j = val, j
        P.append(best_j)
        remaining.discard(best_j)
    return set(P)


def pool_value(score_mat, P):
    return float(sum(score_mat[list(P), m].max() for m in range(N)))


def part_f_pool_perturbation(s_true, s_hat, K=4):
    eps = float(np.max(np.abs(s_hat - s_true)))
    exact_opt = best_pool_exact(s_true, K)
    greedy_exact = greedy_pool(s_true, K)
    recon_opt = best_pool_exact(s_hat, K)
    greedy_recon = greedy_pool(s_hat, K)

    F_true = lambda P: pool_value(s_true, P)
    F_hat = lambda P: pool_value(s_hat, P)

    bound1_checks = []
    for P in (exact_opt, greedy_exact, recon_opt, greedy_recon):
        diff = abs(F_hat(P) - F_true(P))
        bound1_checks.append(dict(diff=diff, bound=N * eps, holds=bool(diff <= N * eps + 1e-9)))

    gap_opt_vs_reconopt = F_true(exact_opt) - F_true(recon_opt)
    bound2_holds = bool(gap_opt_vs_reconopt <= 2 * N * eps + 1e-9)

    return dict(epsilon_inf=eps,
               F_exact_opt=F_true(exact_opt), F_greedy_exact=F_true(greedy_exact),
               F_true_at_recon_opt=F_true(recon_opt),
               F_true_at_greedy_recon=F_true(greedy_recon),
               bound1_checks=bound1_checks,
               gap_opt_vs_reconopt=gap_opt_vs_reconopt,
               bound2_value=2 * N * eps, bound2_holds=bound2_holds,
               pools=dict(exact_opt=sorted(exact_opt), greedy_exact=sorted(greedy_exact),
                         recon_opt=sorted(recon_opt), greedy_recon=sorted(greedy_recon)))


# ---------------------------------------------------------------------------
# Part G: routing-rank hypothesis (Hadamard-product rank bound)
# ---------------------------------------------------------------------------
def part_g_routing_rank(B1, R0, R_routed):
    rB = algebraic_rank(B1)
    rR0 = algebraic_rank(R0)
    rRouted = algebraic_rank(R_routed)
    bound = rB * rR0
    erB = effective_ranks(B1)
    erR0 = effective_ranks(R0)
    erRouted = effective_ranks(R_routed)
    return dict(rank_B=rB, rank_R0=rR0, rank_routed=rRouted,
               hadamard_bound=bound, bound_holds=bool(rRouted <= bound),
               looseness=bound - rRouted,
               effective_rank_B=erB, effective_rank_R0=erR0,
               effective_rank_routed=erRouted)


# ---------------------------------------------------------------------------
# Part H: CUR / skeleton diagnostic (oracle row/column access)
# ---------------------------------------------------------------------------
def qr_pivot_indices(M, r, axis):
    Mat = M if axis == "cols" else M.T
    _, _, piv = scipy_qr(Mat, mode="economic", pivoting=True)
    return piv[:r].tolist()


def cur_reconstruct(M, row_idx, col_idx):
    C = M[:, col_idx]
    Rr = M[row_idx, :]
    W = M[np.ix_(row_idx, col_idx)]
    W_pinv = np.linalg.pinv(W)
    return C @ W_pinv @ Rr


def part_h_cur_diagnostic(rho_full, rng, K=4):
    s_true = np.abs(rho_full)
    n2 = rho_full.shape[0]
    out = {}
    for r in R_LIST:
        r_eff = min(r, N, n2)
        methods = {}
        col_idx_qr = qr_pivot_indices(rho_full, r_eff, axis="cols")
        row_idx_qr = qr_pivot_indices(rho_full, r_eff, axis="rows")
        methods["qr_pivot"] = (row_idx_qr, col_idx_qr)
        col_idx_rand = sorted(rng.choice(N, size=r_eff, replace=False).tolist())
        row_idx_rand = sorted(rng.choice(n2, size=r_eff, replace=False).tolist())
        methods["random"] = (row_idx_rand, col_idx_rand)
        col_norms = np.argsort(-np.linalg.norm(rho_full, axis=0))[:r_eff].tolist()
        row_norms = np.argsort(-np.linalg.norm(rho_full, axis=1))[:r_eff].tolist()
        methods["oracle_norm_pivot"] = (row_norms, col_norms)

        out[str(r)] = {}
        for name, (row_idx, col_idx) in methods.items():
            try:
                rho_hat = cur_reconstruct(rho_full, row_idx, col_idx)
                s_hat = np.abs(rho_hat)
                fro_err = float(np.linalg.norm(s_hat - s_true)
                                / (np.linalg.norm(s_true) + 1e-300))
                dm = decision_metrics(s_true, s_hat, K)
                out[str(r)][name] = dict(fro_rel_err=fro_err, **dm)
            except np.linalg.LinAlgError:
                out[str(r)][name] = dict(fro_rel_err=None)
    return out


# ---------------------------------------------------------------------------
# Part I: operator sanity check (not an algorithmic claim)
# ---------------------------------------------------------------------------
def part_i_operator_sanity():
    rng = np.random.RandomState(0)
    Tn = 200
    lam = 0.9
    u = rng.randn(Tn)
    x = np.zeros(Tn)
    prev = 0.0
    for t in range(Tn):
        prev = lam * prev + u[t]
        x[t] = prev
    x_prev = np.concatenate([[0.0], x[:-1]])
    u_recovered = x - lam * x_prev
    inverse_err = float(np.max(np.abs(u_recovered - u)))

    # low-pass y_t = lam y_{t-1} + (1-lam) u_t; recover u; compare to
    # continuous-time derivative approximation u ~= y + tau dy/dt under
    # lam = exp(-dt/tau), as dt -> 0
    tau = 1.0
    dts = [0.5, 0.1, 0.02, 0.004]
    limit_errs = []
    for dt in dts:
        lam_dt = np.exp(-dt / tau)
        Tn2 = int(20 * tau / dt)
        tt = np.arange(Tn2) * dt
        u2 = np.sin(tt)                      # smooth test signal
        y = np.zeros(Tn2)
        prev = 0.0
        for t in range(Tn2):
            prev = lam_dt * prev + (1 - lam_dt) * u2[t]
            y[t] = prev
        y_prev = np.concatenate([[0.0], y[:-1]])
        u_rec = (y - lam_dt * y_prev) / (1 - lam_dt)
        dydt = np.gradient(y, dt)
        u_ct_approx = y + tau * dydt
        mid = slice(Tn2 // 4, 3 * Tn2 // 4)   # avoid transient/edge effects
        err = float(np.mean(np.abs(u_ct_approx[mid] - u_rec[mid])))
        limit_errs.append(dict(dt=dt, mean_abs_err=err))
    return dict(discrete_inverse_max_err=inverse_err,
               continuous_time_limit_errs=limit_errs,
               limit_shrinks=bool(limit_errs[-1]["mean_abs_err"]
                                  < limit_errs[0]["mean_abs_err"]))


def main() -> None:
    print("=" * 90)
    print(f"Phase B10: tangent/adjoint factorization theory audit, {len(SEEDS)} seeds")
    print("=" * 90)

    mat_id = part_a_matrix_identity_check()
    print(f"Part A matrix identity <c,Hu>=<H^dag c,u>: rel_err={mat_id['rel_err']:.2e}")

    per_seed = dict(a=[], factor_err=[], c=[], d=[], e=[], f=[], g=[], h=[])
    rng_h = np.random.RandomState(999)

    for seed in SEEDS:
        _, rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        a1, B1 = rows[0]["a1"], rows[0]["B1"]
        f_diag = build_F(a1)

        a_errs = part_a_identity_check(rows, f_diag, B1)
        per_seed["a"].append(a_errs)

        U, V0_P, V0_Q = build_factors(rows, a1)
        R0_P, R0_Q, R_P, R_Q = routed_from_factors(U, V0_P, V0_Q, B1)
        rho_factored = np.concatenate([R_P, R_Q], axis=0)
        rho_direct = direct_routed(rows, f_diag, B1)
        factor_err = float(np.max(np.abs(rho_factored - rho_direct))
                           / (np.max(np.abs(rho_direct)) + 1e-300))
        per_seed["factor_err"].append(factor_err)

        s_true = np.abs(rho_factored)
        c_res = part_c_rank_audit(U, V0_P, V0_Q, B1, R0_P, R0_Q, R_P, R_Q, rho_factored)
        per_seed["c"].append(c_res)

        d_res = part_d_low_rank_tests(U, V0_P, V0_Q, B1, s_true)
        per_seed["d"].append(d_res)

        r_e = 4
        U_r = low_rank_trunc(U, r_e)
        VP_r = low_rank_trunc(V0_P, r_e)
        VQ_r = low_rank_trunc(V0_Q, r_e)
        R_P_hat = 0.5 * B1 * (VP_r @ U_r)
        R_Q_hat = 0.5 * np.conj(B1) * (VQ_r @ U_r)
        s_hat_r4 = np.abs(np.concatenate([R_P_hat, R_Q_hat], axis=0))
        e_res = part_e_margin_test(s_true, s_hat_r4)
        per_seed["e"].append(e_res)
        f_res = part_f_pool_perturbation(s_true, s_hat_r4)
        per_seed["f"].append(f_res)

        g_res = part_g_routing_rank(B1, R0_P, R_P)
        per_seed["g"].append(g_res)

        h_res = part_h_cur_diagnostic(rho_factored, rng_h)
        per_seed["h"].append(h_res)

        print(f"seed {seed}: identity_max_rel_err={max(e['rel_err'] for e in a_errs):.2e}  "
             f"factor_err={factor_err:.2e}  "
             f"rho_eff_rank90={c_res['rho_abs']['effective_rank']['0.9']}  "
             f"E_violations={e_res['n_violations']}  "
             f"F_bound2_holds={f_res['bound2_holds']}")

    i_res = part_i_operator_sanity()

    def med_field(lst, *path):
        vals = []
        for d in lst:
            v = d
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    identity_max_rel_err = max(e["rel_err"] for run in per_seed["a"] for e in run)
    factor_max_rel_err = max(per_seed["factor_err"])

    rank_table = {name: dict(
        median_er90=med_field(per_seed["c"], name, "effective_rank", "0.9"),
        median_er95=med_field(per_seed["c"], name, "effective_rank", "0.95"),
        median_er99=med_field(per_seed["c"], name, "effective_rank", "0.99"))
        for name in ("U", "V_P", "V_Q", "V_P_U", "V_Q_U", "B",
                    "R_P_routed", "R_Q_routed", "rho_abs")}

    d_summary = {}
    for r in R_LIST:
        d_summary[str(r)] = {}
        for variant in ("D1_compress_U_only", "D2_compress_V_only", "D3_compress_both"):
            d_summary[str(r)][variant] = dict(
                median_fro_rel_err=med_field(per_seed["d"], str(r), variant, "fro_rel_err"),
                median_winner_preserved=med_field(per_seed["d"], str(r), variant, "winner_preserved"),
                median_topk_recall=med_field(per_seed["d"], str(r), variant, "mean_topk_recall"),
                median_pool_regret=med_field(per_seed["d"], str(r), variant, "pool_regret"))

    e_summary = dict(
        total_violations=sum(r["n_violations"] for r in per_seed["e"]),
        median_frac_preserved=med_field(per_seed["e"], "frac_preserved_overall"),
        median_frac_certified=med_field(per_seed["e"], "frac_certified_by_margin"))

    f_summary = dict(
        frac_bound2_holds=float(np.mean([r["bound2_holds"] for r in per_seed["f"]])),
        frac_bound1_holds=float(np.mean([c["holds"] for r in per_seed["f"]
                                        for c in r["bound1_checks"]])),
        median_gap_opt_vs_reconopt=med_field(per_seed["f"], "gap_opt_vs_reconopt"),
        median_bound2_value=med_field(per_seed["f"], "bound2_value"))

    g_summary = dict(
        median_effective_rank_B_90=med_field(per_seed["g"], "effective_rank_B", "0.9"),
        median_effective_rank_R0_90=med_field(per_seed["g"], "effective_rank_R0", "0.9"),
        median_effective_rank_routed_90=med_field(per_seed["g"], "effective_rank_routed", "0.9"),
        frac_algebraic_bound_holds=float(np.mean([r["bound_holds"] for r in per_seed["g"]])))

    h_summary = {}
    for r in R_LIST:
        h_summary[str(r)] = {}
        for method in ("qr_pivot", "random", "oracle_norm_pivot"):
            vals = [per_seed["h"][s][str(r)][method] for s in range(len(SEEDS))]
            h_summary[str(r)][method] = dict(
                median_fro_rel_err=float(np.median(
                    [v["fro_rel_err"] for v in vals if v.get("fro_rel_err") is not None])),
                median_winner_preserved=float(np.median(
                    [v["winner_preserved"] for v in vals if "winner_preserved" in v])),
                median_pool_jaccard=float(np.median(
                    [v["pool_jaccard"] for v in vals if "pool_jaccard" in v])))

    print("-" * 90)
    print("Part C rank table (median effective rank, 90/95/99% energy):")
    for name, r in rank_table.items():
        print(f"  {name:14s} {r['median_er90']:.1f} / {r['median_er95']:.1f} / {r['median_er99']:.1f}")
    print("Part D (r=4) summary:", json.dumps(d_summary["4"], indent=1))
    print("Part E:", e_summary)
    print("Part F:", f_summary)
    print("Part G:", g_summary)
    print("Part H (r=2) summary:", json.dumps(h_summary["2"], indent=1))
    print("Part I:", i_res)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                                    n_cal_traj=N_CAL_TRAJ, k_list=K_LIST, r_list=R_LIST),
              part_a_matrix_identity=mat_id,
              part_a_identity_max_rel_err=identity_max_rel_err,
              part_b_factor_reconstruction_max_rel_err=factor_max_rel_err,
              part_c_rank_table=rank_table,
              part_d_summary=d_summary,
              part_e_summary=e_summary,
              part_f_summary=f_summary,
              part_g_summary=g_summary,
              part_h_summary=h_summary,
              part_i=i_res,
              per_seed=per_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b10_tangent_adjoint_theory_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            complex(o).real if isinstance(o, np.complexfloating) else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
