"""Phase B30a -- general flag-SSM reduced-RTRL correctness/scaling test,
r=64, d=4. Standalone, independent of B28/B29. No training, no runtime
optimization, no LRU comparison. Per instruction: stop after this
correctness + memory-accounting suite.

Architecture (T = V (+) U, dim V = d = 4, dim U = 60, r = d+dim(U) = 64):
  u_{t+1} = R_U u_t + D_U x_t                          (U: pure linear,
                                                          NO theta, NO v)
  v_{t+1} = R_V v_t + K u_t
            + B_V Phi_theta(C_V v_t + C_U u_t, x_t)     (V: nonlinear,
                                                          theta lives here)
theta = Phi's own MLP weights (W1,b1,W2,b2) ONLY -- R_U,D_U,R_V,K,B_V,
C_V,C_U are all FIXED, untrained structural matrices (matching B29's
convention: only the nonlinear-feedback path is trainable).

Why the flag holds here (no eigenvector argument needed, unlike B29):
u_{t+1} never reads v_t or theta, so D_theta u_t = 0 for ALL t by
induction on t (u_0 is theta-independent). Therefore the FULL
sensitivity S_t = D_theta s_t (s_t=(u_t,v_t)) is IDENTICALLY confined
to the V-block for every t -- a consequence of the one-directional
(U->V, never V->U) block-triangular coupling, not of any special
eigenvector alignment. The general reduced-module algorithm below
never hand-derives a closed-form scalar recursion (cf. B29): it
restricts the SAME jax.jacobian-based RTRL machinery to V's own
dynamics, treating u_t as a precomputed (theta-independent) VALUE.

Three independent gradient paths:
  1. BPTT: jax.grad through jax.lax.scan over the whole sequence.
  2. Full exact RTRL: S_{t+1}=J_t S_t+G_t, J_t/G_t = FULL 64x64 / 64xP_c
     jacobians (autodiff) of the combined (u,v) step.
  3. Reduced exact RTRL: E_{t+1}=J_v,t E_t+G_v,t, J_v,t/G_v,t = 4x4 /
     4xP_c jacobians (autodiff) of v_step ALONE, with u_t injected as a
     precomputed value (its own theta-free forward simulation) -- the
     GENERAL restricted-operator algorithm, not a hand-derived formula.

Run: python -m credit_memory.b30a_flag_r64d4_test
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from credit_memory.b25_nonlinear_credit import krylov_subspace

D_U_DIM = 60
D_V_DIM = 4
R_DIM = D_U_DIM + D_V_DIM  # 64
C_DIM = 8          # shared readout dim (C_V, C_U both map into this)
K_OUT = 8          # Phi's output dim (B_V maps this into V's 4 dims)
H_DIM = 560        # Phi hidden width -> P_c = 18*560+8 = 10088
X_DIM = 1

P_MAT = jnp.concatenate([jnp.zeros((D_U_DIM, D_V_DIM)), jnp.eye(D_V_DIM)], axis=0)  # (64,4)


def make_stable_dense(n, rng, radius):
    M = rng.randn(n, n) / np.sqrt(n)
    eig = np.max(np.abs(np.linalg.eigvals(M)))
    return M * (radius / eig)


def make_consts(seed=12345):
    """Fixed (non-trained) structural matrices -- ONE draw, shared across
    all (data-seed, T) settings below, exactly as R was fixed in B29."""
    rng = np.random.RandomState(seed)
    R_U = jnp.array(make_stable_dense(D_U_DIM, rng, radius=0.85))
    D_U = jnp.array(rng.randn(D_U_DIM) * 0.5)
    R_V = jnp.array(make_stable_dense(D_V_DIM, rng, radius=0.80))
    K = jnp.array(rng.randn(D_V_DIM, D_U_DIM) * (0.15 / np.sqrt(D_U_DIM)))
    B_V = jnp.array(rng.randn(D_V_DIM, K_OUT) * (0.5 / np.sqrt(K_OUT)))
    C_V = jnp.array(rng.randn(C_DIM, D_V_DIM) * (0.5 / np.sqrt(D_V_DIM)))
    C_U = jnp.array(rng.randn(C_DIM, D_U_DIM) * (0.5 / np.sqrt(D_U_DIM)))
    return dict(R_U=R_U, D_U=D_U, R_V=R_V, K=K, B_V=B_V, C_V=C_V, C_U=C_U)


def make_theta(seed):
    rng = np.random.RandomState(seed)
    in_dim = C_DIM + X_DIM
    scale1 = 1.0 / np.sqrt(in_dim)
    scale2 = 1.0 / np.sqrt(H_DIM)
    theta = dict(
        W1=jnp.array(rng.randn(H_DIM, in_dim) * scale1),
        b1=jnp.array(rng.randn(H_DIM) * 0.05),
        W2=jnp.array(rng.randn(K_OUT, H_DIM) * scale2),
        b2=jnp.array(rng.randn(K_OUT) * 0.05),
    )
    return theta


def phi_forward(z, x, theta):
    xin = jnp.concatenate([z, jnp.atleast_1d(x)])
    hact = jnp.tanh(theta["W1"] @ xin + theta["b1"])
    return theta["W2"] @ hact + theta["b2"]


def u_step(u, x, c):
    return c["R_U"] @ u + c["D_U"] * x


def v_step(v, u_val, x, theta, c):
    z = c["C_V"] @ v + c["C_U"] @ u_val
    phi_out = phi_forward(z, x, theta)
    return c["R_V"] @ v + c["K"] @ u_val + c["B_V"] @ phi_out


def full_step_state(s, x, theta, c):
    u, v = s[:D_U_DIM], s[D_U_DIM:]
    v_next = v_step(v, u, x, theta, c)
    u_next = u_step(u, x, c)
    return jnp.concatenate([u_next, v_next])


def rollout_full_states(u0, v0, xs, theta, c):
    def step(carry, x):
        u, v = carry
        v_next = v_step(v, u, x, theta, c)
        u_next = u_step(u, x, c)
        return (u_next, v_next), jnp.concatenate([u_next, v_next])
    _, Ss = jax.lax.scan(step, (u0, v0), xs)
    return Ss  # (T,64)


def ell(y, phase):
    return jnp.sin(y + phase) + 0.5 * y ** 2


def dell_dy(y, phase):
    return jnp.cos(y + phase) + y


def loss_bptt(theta, u0, v0, xs, qs, phases, c):
    Ss = rollout_full_states(u0, v0, xs, theta, c)
    ys = jnp.einsum("ti,ti->t", Ss, qs)
    return jnp.sum(ell(ys, phases))


grad_bptt_fn = jax.jit(jax.grad(loss_bptt, argnums=0))


def full_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel, theta_flat, P_c):
    T = xs.shape[0]
    s = jnp.concatenate([u0, v0])
    S = jnp.zeros((R_DIM, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    S_traj = []
    for t in range(T):
        x = xs[t]
        J_t = jax.jacobian(lambda ss: full_step_state(ss, x, theta, c))(s)
        G_t = jax.jacobian(lambda th_flat: full_step_state(s, x, unravel(th_flat), c))(theta_flat)
        S = J_t @ S + G_t
        S_traj.append(S)
        s_next = full_step_state(s, x, theta, c)
        y = qs[t] @ s_next
        dl_ds = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_ds @ S
        s = s_next
    return g_total, jnp.stack(S_traj)


def reduced_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel, theta_flat, P_c):
    T = xs.shape[0]
    # Precompute the FULL u-trajectory once -- theta-free, exact.
    u_traj = [u0]
    u = u0
    for t in range(T):
        u = u_step(u, xs[t], c)
        u_traj.append(u)

    v = v0
    E = jnp.zeros((D_V_DIM, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    E_traj = []
    for t in range(T):
        x = xs[t]
        u_val = u_traj[t]
        J_v = jax.jacobian(lambda vv: v_step(vv, u_val, x, theta, c))(v)
        G_v = jax.jacobian(lambda th_flat: v_step(v, u_val, x, unravel(th_flat), c))(theta_flat)
        E = J_v @ E + G_v
        E_traj.append(E)
        v_next = v_step(v, u_val, x, theta, c)
        u_next = u_traj[t + 1]
        s_next = jnp.concatenate([u_next, v_next])
        y = qs[t] @ s_next
        dl_dv = dell_dy(y, phases[t]) * qs[t][D_U_DIM:]
        g_total = g_total + dl_dv @ E
        v = v_next
    return g_total, jnp.stack(E_traj)


def reconstruct_S_from_E(E_traj):
    T = E_traj.shape[0]
    P_c = E_traj.shape[-1]
    S_recon = jnp.zeros((T, R_DIM, P_c), dtype=jnp.float64)
    S_recon = S_recon.at[:, D_U_DIM:, :].set(E_traj)
    return S_recon


def make_setting(seed, T, c):
    rng = np.random.RandomState(seed)
    u0 = jnp.array(rng.randn(D_U_DIM) * 0.2)
    v0 = jnp.array(rng.randn(D_V_DIM) * 0.2)
    xs = jnp.array(rng.randn(T) * 0.5)
    qs = jnp.array(rng.randn(T, R_DIM))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    theta = make_theta(seed + 9999)
    return theta, u0, v0, xs, qs, phases


def run_correctness_suite():
    c = make_consts()
    theta0 = make_theta(0)
    theta_flat0, unravel = ravel_pytree(theta0)
    P_c = theta_flat0.shape[0]
    print("=" * 78)
    print(f"B30a correctness suite: r=64 (U=60,V=4), P_c={P_c}")
    print("=" * 78)

    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 50]
    eps_num = 1e-12
    worst_full_rel, worst_reduced_rel, worst_S_recon = 0.0, 0.0, 0.0
    t_start = time.time()
    for T in lengths:
        for seed in seeds:
            theta, u0, v0, xs, qs, phases = make_setting(seed, T, c)
            theta_flat, unravel_s = ravel_pytree(theta)

            g_b = grad_bptt_fn(theta, u0, v0, xs, qs, phases, c)
            g_b_flat, _ = ravel_pytree(g_b)

            g_f, S_full = full_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel_s, theta_flat, P_c)
            g_r, E_traj = reduced_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel_s, theta_flat, P_c)
            S_recon = reconstruct_S_from_E(E_traj)

            max_abs_full = float(jnp.max(jnp.abs(g_f - g_b_flat)))
            rel_full = float(jnp.linalg.norm(g_f - g_b_flat) / (jnp.linalg.norm(g_b_flat) + eps_num))
            max_abs_reduced = float(jnp.max(jnp.abs(g_r - g_b_flat)))
            rel_reduced = float(jnp.linalg.norm(g_r - g_b_flat) / (jnp.linalg.norm(g_b_flat) + eps_num))
            max_abs_S = float(jnp.max(jnp.abs(S_recon - S_full)))
            max_abs_U_rows_full = float(jnp.max(jnp.abs(S_full[:, :D_U_DIM, :])))

            worst_full_rel = max(worst_full_rel, rel_full)
            worst_reduced_rel = max(worst_reduced_rel, rel_reduced)
            worst_S_recon = max(worst_S_recon, max_abs_S)

            print(f"  T={T:3d} seed={seed}  full: max|d|={max_abs_full:.3e} rel={rel_full:.3e}  "
                  f"reduced: max|d|={max_abs_reduced:.3e} rel={rel_reduced:.3e}  "
                  f"S_recon max|d|={max_abs_S:.3e}  full_S_Urows_max|.|={max_abs_U_rows_full:.3e}")
    elapsed = time.time() - t_start
    print("-" * 78)
    print(f"WORST: full_rel={worst_full_rel:.3e}  reduced_rel={worst_reduced_rel:.3e}  "
          f"S_recon={worst_S_recon:.3e}   (elapsed {elapsed:.1f}s)")
    all_pass = worst_full_rel < 1e-8 and worst_reduced_rel < 1e-8 and worst_S_recon < 1e-8
    print(f"ALL < 1e-8: {all_pass}")

    bytes_full = R_DIM * P_c * 8
    bytes_reduced = D_V_DIM * P_c * 8
    print()
    print("Persistent sensitivity storage:")
    print(f"  Full:    r*P_c = {R_DIM}*{P_c} = {R_DIM*P_c} float64 scalars = {bytes_full} bytes ({bytes_full/1e6:.3f} MB)")
    print(f"  Reduced: d*P_c = {D_V_DIM}*{P_c} = {D_V_DIM*P_c} float64 scalars = {bytes_reduced} bytes ({bytes_reduced/1e6:.3f} MB)")
    print(f"  Reduction ratio: {(R_DIM*P_c)/(D_V_DIM*P_c):.2f}x  (target 64/4=16x)")
    return dict(all_pass=all_pass, worst_full_rel=worst_full_rel, worst_reduced_rel=worst_reduced_rel,
                worst_S_recon=worst_S_recon, P_c=P_c, elapsed=elapsed)


def run_structural_diagnostics():
    print()
    print("=" * 78)
    print("Structural diagnostics")
    print("=" * 78)
    c = make_consts()
    theta = make_theta(0)

    R_U_np, D_U_np = np.asarray(c["R_U"]), np.asarray(c["D_U"])
    R_V_np, K_np = np.asarray(c["R_V"]), np.asarray(c["K"])
    A_lin = np.zeros((R_DIM, R_DIM))
    A_lin[:D_U_DIM, :D_U_DIM] = R_U_np
    A_lin[D_U_DIM:, D_U_DIM:] = R_V_np
    A_lin[D_U_DIM:, :D_U_DIM] = K_np
    B_full = np.zeros(R_DIM)
    B_full[:D_U_DIM] = D_U_np

    K_reach = krylov_subspace(A_lin, B_full[:, None])
    print(f"1. Forward reachable rank from x (linear skeleton [[R_U,0],[K,R_V]], seeded at [D_U;0]): "
          f"{K_reach.shape[1]} / {R_DIM}")

    # invariant check ||(I-PP^+) J_t P|| over sampled states
    rng = np.random.RandomState(777)
    max_leak = 0.0
    P_np = np.asarray(P_MAT)
    proj_perp = np.eye(R_DIM) - P_np @ P_np.T
    for _ in range(8):
        u_s = jnp.array(rng.randn(D_U_DIM) * 0.3)
        v_s = jnp.array(rng.randn(D_V_DIM) * 0.3)
        x_s = float(rng.randn() * 0.5)
        s_s = jnp.concatenate([u_s, v_s])
        J_t = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta, c))(s_s))
        leak = np.linalg.norm(proj_perp @ J_t @ P_np)
        max_leak = max(max_leak, leak)
    print(f"2. max_t ||(I-PP^+) J_t P|| over 8 sampled states: {max_leak:.3e}  (expect ~0)")

    # nonzero commutator: Q = nonlinear-correction contribution to J_t at one representative point
    u_rep = jnp.array(rng.randn(D_U_DIM) * 0.3)
    v_rep = jnp.array(rng.randn(D_V_DIM) * 0.3)
    x_rep = float(rng.randn() * 0.5)
    s_rep = jnp.concatenate([u_rep, v_rep])
    J_rep = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_rep, theta, c))(s_rep))
    Q = J_rep - A_lin
    comm = A_lin @ Q - Q @ A_lin
    comm_norm = np.linalg.norm(comm)
    print(f"3. ||[A_lin, Q]||_F = {comm_norm:.6e}  (Q = J_t - A_lin at a representative state; nonzero => noncommutative)")

    # evidence of genuine (not fixed-direct-sum) coupling
    K_norm = np.linalg.norm(K_np)
    C_U_norm = np.linalg.norm(np.asarray(c["C_U"]))

    def v_after_T(u0, v0, xs, theta, c, T=10):
        v = v0
        u = u0
        for t in range(T):
            v_next = v_step(v, u, xs[t], theta, c)
            u = u_step(u, xs[t], c)
            v = v_next
        return v

    xs_probe = jnp.array(rng.randn(10) * 0.5)
    dv_du0 = jax.jacobian(lambda uu0: v_after_T(uu0, v_rep, xs_probe, theta, c))(u_rep)
    dv_du0_norm = float(jnp.linalg.norm(dv_du0))
    print(f"4. ||K||_F={K_norm:.4f}  ||C_U||_F={C_U_norm:.4f}  "
          f"||d(v_10)/d(u_0)||_F={dv_du0_norm:.4e}  (nonzero => genuine U->V coupling, not a fixed direct sum)")

    # 5. Common-invariant-complement diagnostic. J_t = [[A_U,0],[B_t,A_V,t]]
    # (A_U = R_U, time-invariant, since u_next never reads v or theta).
    # Any invariant COMPLEMENT to V would be a graph {(u,Lu)}, requiring one
    # L (4x60) satisfying L A_U - A_V,t L = B_t simultaneously for every
    # sampled t. Stack the vectorized linear systems across many sampled
    # states and solve least squares for a single common L; report the
    # normalized residual. Near-zero => the family MAY split under a
    # coordinate change (do not claim indecomposable then). Clearly
    # nonzero and stable across more samples => no common invariant
    # complement for this SAMPLED family (a finite-sample finding, not a
    # universal proof).
    n_samples_list = [8, 16, 32]
    residuals_by_n = {}
    I60 = np.eye(D_U_DIM)
    I4 = np.eye(D_V_DIM)
    A_U_np = R_U_np  # time-invariant top-left block
    rng_inv = np.random.RandomState(2468)
    all_M, all_b = [], []
    sample_log = []
    for n_target in n_samples_list:
        while len(sample_log) < n_target:
            u_s = jnp.array(rng_inv.randn(D_U_DIM) * 0.3)
            v_s = jnp.array(rng_inv.randn(D_V_DIM) * 0.3)
            x_s = float(rng_inv.randn() * 0.5)
            s_s = jnp.concatenate([u_s, v_s])
            J_t = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta, c))(s_s))
            A_V_t = J_t[D_U_DIM:, D_U_DIM:]
            B_t = J_t[D_U_DIM:, :D_U_DIM]
            # vec(L A_U - A_V,t L) = [(A_U^T kron I4) - (I60 kron A_V,t)] vec(L)
            M_t = np.kron(A_U_np.T, I4) - np.kron(I60, A_V_t)
            b_t = B_t.reshape(-1, order="F")  # vec (column-major) of B_t
            all_M.append(M_t)
            all_b.append(b_t)
            sample_log.append(1)
        M_stack = np.concatenate(all_M, axis=0)
        b_stack = np.concatenate(all_b, axis=0)
        vecL, *_ = np.linalg.lstsq(M_stack, b_stack, rcond=None)
        resid = M_stack @ vecL - b_stack
        norm_resid = np.linalg.norm(resid) / (np.linalg.norm(b_stack) + 1e-12)
        residuals_by_n[n_target] = float(norm_resid)
    print(f"5. Common-invariant-complement diagnostic (single L solving "
          f"L A_U - A_V,t L = B_t across n sampled states, least squares):")
    for n_target in n_samples_list:
        print(f"     n_samples={n_target:3d}  normalized residual = {residuals_by_n[n_target]:.4e}")
    complement_conclusion = (
        "residual stable and clearly nonzero across n -- no common invariant "
        "complement found for this sampled Jacobian family (finite-sample "
        "finding, not a universal proof)"
        if min(residuals_by_n.values()) > 1e-6 else
        "residual near numerical zero -- a common invariant complement may "
        "exist; do NOT claim indecomposability from this diagnostic"
    )
    print(f"     => {complement_conclusion}")

    return dict(reach_rank=int(K_reach.shape[1]), max_leak=float(max_leak), commutator_norm=float(comm_norm),
                K_norm=float(K_norm), C_U_norm=float(C_U_norm), dv_du0_norm=dv_du0_norm,
                complement_residuals=residuals_by_n)


def main():
    corr = run_correctness_suite()
    struct = run_structural_diagnostics()
    print()
    print("=" * 78)
    print(f"B30a CORRECTNESS SUITE PASS (<1e-8 everywhere): {corr['all_pass']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
