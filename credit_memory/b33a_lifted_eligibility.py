"""Phase B33a -- exact LIFTED eligibility recurrence / proof-of-principle.
NOT a trainable RNN. J_t and G_t here are GIVEN directly as scalar
combinations of a fixed structural involution K -- they are NOT
derived as D_h F_theta / D_theta F_theta for any actual recurrent
parameterization. This tests a strictly more general theory claim than
B29-B32: exact compression does not require rank(S_t) < r. The
eligibility matrix S_t can be FULL ordinary matrix rank (r=64) while
its DYNAMIC (time-varying) content lives in a fixed 2-dimensional
lifted subspace span{I,K} of Hom(R^64,R^64).

Construction (abstract, given):
  r = P = 64
  K: dense, well-conditioned involution (K^2=I), built as K=Q D Q^T
     for a random dense orthogonal Q and D=diag(+-1) with both signs
     present.
  Per-step scalars alpha_t,beta_t (stable: eigenvalues of
     [[alpha,beta],[beta,alpha]] = alpha+-beta, kept < 1 in magnitude)
     and gamma_t,delta_t (direct-term coefficients).
  J_t = alpha_t I + beta_t K,   G_t = gamma_t I + delta_t K
  S_{t+1} = J_t S_t + G_t,  S_0 = 0   (FULL, r x r = 64x64, exact)

Because K^2=I, span{I,K} is exactly invariant under left-multiplication
by any alpha*I+beta*K, so S_t = a_t I + b_t K with the closed 2-d
recursion:
  a_{t+1} = alpha_t a_t + beta_t b_t + gamma_t
  b_{t+1} = beta_t a_t + alpha_t b_t + delta_t

Two exact paths compared: (A) full S_t in R^{64x64}; (B) the 2-d
lifted recursion, reconstructed as S_hat_t = a_t I + b_t K.

Run: python -m credit_memory.b33a_lifted_eligibility
"""
from __future__ import annotations

import numpy as np

R_DIM = 64


def make_K(seed):
    """K = Q D Q^T, D=diag(+-1) both signs present, Q dense orthogonal."""
    rng = np.random.RandomState(seed)
    half = R_DIM // 2
    D = np.concatenate([np.ones(half), -np.ones(R_DIM - half)])
    M = rng.randn(R_DIM, R_DIM)
    Q, _ = np.linalg.qr(M)
    K = Q @ np.diag(D) @ Q.T
    return K


def make_coeffs(seed, T, stable=True):
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(0.30, 0.60, size=T)
    beta = rng.uniform(-0.20, 0.20, size=T)
    if not stable:
        beta = rng.uniform(-0.9, 0.9, size=T)
    gamma = rng.randn(T) * 0.3
    delta = rng.randn(T) * 0.3
    return alpha, beta, gamma, delta


def full_recurrence(K, alpha, beta, gamma, delta):
    T = alpha.shape[0]
    S = np.zeros((R_DIM, R_DIM))
    S_traj = np.zeros((T, R_DIM, R_DIM))
    for t in range(T):
        J_t = alpha[t] * np.eye(R_DIM) + beta[t] * K
        G_t = gamma[t] * np.eye(R_DIM) + delta[t] * K
        S = J_t @ S + G_t
        S_traj[t] = S
    return S_traj


def lifted_recurrence(alpha, beta, gamma, delta):
    T = alpha.shape[0]
    a, b = 0.0, 0.0
    a_traj = np.zeros(T)
    b_traj = np.zeros(T)
    for t in range(T):
        a_next = alpha[t] * a + beta[t] * b + gamma[t]
        b_next = beta[t] * a + alpha[t] * b + delta[t]
        a, b = a_next, b_next
        a_traj[t] = a
        b_traj[t] = b
    return a_traj, b_traj


def reconstruct(a_traj, b_traj, K):
    T = a_traj.shape[0]
    S_hat = np.zeros((T, R_DIM, R_DIM))
    I = np.eye(R_DIM)
    for t in range(T):
        S_hat[t] = a_traj[t] * I + b_traj[t] * K
    return S_hat


def run_correctness_suite():
    print("=" * 78)
    print(f"B33a correctness suite: r=P={R_DIM}, abstract lifted eligibility (eps=0)")
    print("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 100, 1000]
    worst_recon = 0.0
    worst_query = 0.0
    for T in lengths:
        for seed in seeds:
            K = make_K(seed + 1000)
            k2_err = float(np.max(np.abs(K @ K - np.eye(R_DIM))))
            alpha, beta, gamma, delta = make_coeffs(seed, T)

            S_traj = full_recurrence(K, alpha, beta, gamma, delta)
            a_traj, b_traj = lifted_recurrence(alpha, beta, gamma, delta)
            S_hat_traj = reconstruct(a_traj, b_traj, K)

            recon_err = float(np.max(np.abs(S_traj - S_hat_traj)))
            worst_recon = max(worst_recon, recon_err)

            rng = np.random.RandomState(seed + 5000)
            n_query_checks = min(T, 10)
            query_errs = []
            for _ in range(n_query_checks):
                t_idx = rng.randint(0, T)
                q = rng.randn(R_DIM)
                lhs = q @ S_traj[t_idx]
                rhs = q @ S_hat_traj[t_idx]
                query_errs.append(float(np.max(np.abs(lhs - rhs))))
            worst_query = max(worst_query, max(query_errs))

            ranks = []
            rank_check_indices = list(range(min(T, 50))) if T <= 200 else list(range(0, T, max(1, T // 50)))
            for t_idx in rank_check_indices:
                sv = np.linalg.svd(S_traj[t_idx], compute_uv=False)
                rank_t = int(np.sum(sv > 1e-9 * sv[0])) if sv[0] > 1e-12 else 0
                ranks.append(rank_t)
            frac_full_rank = float(np.mean([r_ == R_DIM for r_ in ranks])) if ranks else float("nan")

            print(f"  T={T:5d} seed={seed}  K^2=I err={k2_err:.2e}  recon max|d|={recon_err:.3e}  "
                  f"query max|d|={max(query_errs):.3e}  frac_full_rank(sampled)={frac_full_rank:.2f}")
    print("-" * 78)
    print(f"WORST reconstruction error: {worst_recon:.3e}")
    print(f"WORST query error: {worst_query:.3e}")
    all_pass = worst_recon < 1e-8 and worst_query < 1e-8
    print(f"ALL < 1e-8: {all_pass}")

    print()
    print("Storage accounting (extremely explicit, per instruction):")
    print(f"  Persistent TIME-VARYING dynamic credit storage:")
    print(f"    full: r*P = {R_DIM}*{R_DIM} = {R_DIM*R_DIM} float64 scalars ({R_DIM*R_DIM*8} bytes)")
    print(f"    reduced (lifted coefficients a_t,b_t): 2 float64 scalars (16 bytes)")
    print(f"    ratio on the DYNAMIC/time-varying axis: {R_DIM*R_DIM/2:.0f}x")
    print(f"  Static/structural storage (computed ONCE, held fixed, NOT time-varying):")
    print(f"    K stored densely: r^2 = {R_DIM*R_DIM} float64 scalars ({R_DIM*R_DIM*8} bytes)")
    print(f"    K is NOT part of any forward model in this abstract test -- it exists")
    print(f"    solely to enable the lifted-credit trick, so it is NOT free overhead;")
    print(f"    it must be counted if nothing else already requires storing it.")
    print(f"  TOTAL footprint if only the CURRENT step's persistent state is kept:")
    print(f"    full: {R_DIM*R_DIM} floats (current S_t only)")
    print(f"    lifted: {R_DIM*R_DIM} (K, static, one-time) + 2 (current a_t,b_t) = {R_DIM*R_DIM+2} floats")
    print(f"  => the lifted approach's TOTAL memory footprint is NOT smaller here")
    print(f"     ({R_DIM*R_DIM+2} vs {R_DIM*R_DIM}) once K is honestly counted -- the clean, scientifically")
    print(f"     supportable claim is specifically about PERSISTENT DYNAMIC credit")
    print(f"     storage ({R_DIM*R_DIM} -> 2, a {R_DIM*R_DIM/2:.0f}x reduction on that axis alone),")
    print(f"     NOT a blanket '{R_DIM*R_DIM/2:.0f}x total memory reduction' claim.")
    return dict(all_pass=all_pass, worst_recon=worst_recon, worst_query=worst_query)


def run_falsification_suite():
    print()
    print("=" * 78)
    print("Falsification: break lifted closure with J_t^eps = alpha_t I + beta_t K + eps*R")
    print("=" * 78)
    eps_list = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    lengths = [5, 20, 100, 500]
    seed = 0
    K = make_K(seed + 1000)

    rng_R = np.random.RandomState(9999)
    M = rng_R.randn(R_DIM, R_DIM)
    R_generic = (M + M.T) / 2.0  # symmetric, generic, NOT in span{I,K} generically
    # confirm R_generic is NOT in span{I,K} (residual after projecting out I,K components)
    I = np.eye(R_DIM)
    basis = [I.ravel() / np.linalg.norm(I.ravel()), K.ravel() / np.linalg.norm(K.ravel())]
    basis, _ = np.linalg.qr(np.stack(basis, axis=1))
    r_flat = R_generic.ravel()
    proj = basis @ (basis.T @ r_flat)
    resid_outside = np.linalg.norm(r_flat - proj) / np.linalg.norm(r_flat)
    print(f"  R_generic fraction outside span{{I,K}}: {resid_outside:.4f} (expect close to 1.0, i.e. mostly outside)")

    for eps in eps_list:
        for T in lengths:
            alpha, beta, gamma, delta = make_coeffs(seed, T)
            # FULL recurrence, exact, with the eps*R term included
            S = np.zeros((R_DIM, R_DIM))
            S_traj = np.zeros((T, R_DIM, R_DIM))
            for t in range(T):
                J_t = alpha[t] * np.eye(R_DIM) + beta[t] * K + eps * R_generic
                G_t = gamma[t] * np.eye(R_DIM) + delta[t] * K
                S = J_t @ S + G_t
                S_traj[t] = S

            # FORCED two-coefficient recurrence -- deliberately still assumes
            # closure to span{I,K}, ignoring the eps*R term entirely (same
            # convention as B29/B32's falsification: old reduced path frozen).
            a_traj, b_traj = lifted_recurrence(alpha, beta, gamma, delta)
            S_hat_traj = reconstruct(a_traj, b_traj, K)
            recon_err = float(np.max(np.abs(S_traj - S_hat_traj)))
            rel_recon = recon_err / (float(np.max(np.abs(S_traj))) + 1e-12)

            # lifted/Krylov span dimension of {S_1,...,S_T}, vectorized
            flat = S_traj.reshape(T, -1)
            sv = np.linalg.svd(flat, compute_uv=False)
            span_dim = int(np.sum(sv > 1e-9 * sv[0])) if sv[0] > 1e-12 else 0

            print(f"  eps={eps:.0e}  T={T:4d}  span_dim{{S_1..S_T}}={span_dim:3d}/{min(T,R_DIM*R_DIM)}  "
                  f"forced_recon_max|d|={recon_err:.3e}  rel={rel_recon:.3e}")


def main():
    corr = run_correctness_suite()
    run_falsification_suite()
    print()
    print("=" * 78)
    print(f"B33a CORRECTNESS SUITE PASS (<1e-8 everywhere): {corr['all_pass']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
