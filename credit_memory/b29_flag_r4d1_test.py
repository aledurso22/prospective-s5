"""Phase B29 -- reduced exact-sensitivity theorem test, explicit r=4,
d=1 flag construction. Standalone controlled correctness + falsification
suite. NOT connected to Autoencode/B28 -- no training, no environment,
no benchmark. Per instruction: stop after this suite.

Architecture (given, fixed):
  R = [[0.7,0.2,0,0],[0,0.7,0.2,0],[0,0,0.7,0.2],[0,0,0,0.7]]
  h_{t+1} = R h_t + e4 x_t + e1 tanh(w.h_t + beta*x_t + b)
  theta = (w1,w2,w3,w4,beta,b)  (6 trainable scalars; R, e1, e4 fixed)
  z_t = w.h_t + beta*x_t + b

Claim under test: for this family, exact sensitivity S_t = D_theta h_t
(in R^{4x6}) is confined to V=span(e1), i.e. S_t = e1 E_t for E_t in
R^{1x6}, because R's first COLUMN is R e1 = 0.7 e1 (R is upper
bidiagonal, so e1 is an eigenvector of R with eigenvalue 0.7) -- this
is exactly what makes span(e1) invariant under J_t = R + phi'(z_t) e1 w^T.

Three independent gradient paths:
  1. BPTT: jax.grad through the whole unrolled sequence (autodiff).
  2. Full exact RTRL: S_{t+1} = J_t S_t + G_t, with J_t/G_t computed via
     PER-STEP jax.jacobian (autodiff), not hand-derived formulas --
     independent of both BPTT's single reverse pass and the reduced
     path's hand-coded closed form.
  3. Reduced exact RTRL: E_{t+1} = (0.7 + phi'(z_t) w1) E_t +
     phi'(z_t) [h_t^T, x_t, 1], a hand-derived closed-form scalar-row
     recursion, maintaining only 1x6 = 6 floats instead of 4x6 = 24.

Run: python -m credit_memory.b29_flag_r4d1_test
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b25_nonlinear_credit import algebra_closure, krylov_subspace

R_BASE = jnp.array([
    [0.7, 0.2, 0.0, 0.0],
    [0.0, 0.7, 0.2, 0.0],
    [0.0, 0.0, 0.7, 0.2],
    [0.0, 0.0, 0.0, 0.7],
], dtype=jnp.float64)

E1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
E4 = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)


def unpack(theta):
    return theta[:4], theta[4], theta[5]


def z_of(h, x, theta):
    w, beta, b = unpack(theta)
    return w @ h + beta * x + b


def forward_step(h, x, theta, R):
    w, beta, b = unpack(theta)
    z = w @ h + beta * x + b
    return R @ h + E4 * x + E1 * jnp.tanh(z)


def ell(y, phase):
    return jnp.sin(y + phase) + 0.5 * y ** 2


def dell_dy(y, phase):
    return jnp.cos(y + phase) + y


# ---------------------------------------------------------------------
# Path 1: BPTT (autodiff through the whole unrolled sequence).
# ---------------------------------------------------------------------
def rollout_states(h0, xs, theta, R):
    def step(h, x):
        h_next = forward_step(h, x, theta, R)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs  # (T,4), states h_1..h_T


def loss_bptt(theta, h0, xs, qs, phases, R):
    Hs = rollout_states(h0, xs, theta, R)
    ys = jnp.einsum("ti,ti->t", Hs, qs)
    return jnp.sum(ell(ys, phases))


grad_bptt_fn = jax.jit(jax.grad(loss_bptt, argnums=0), static_argnums=())


# ---------------------------------------------------------------------
# Path 2: full exact RTRL. J_t, G_t via per-step jax.jacobian (autodiff),
# NOT the hand-derived closed forms -- a genuinely independent path.
# ---------------------------------------------------------------------
def full_rtrl_grad(theta, h0, xs, qs, phases, R):
    T = xs.shape[0]
    S = jnp.zeros((4, 6), dtype=jnp.float64)
    h = h0
    g_total = jnp.zeros(6, dtype=jnp.float64)
    S_traj = []
    for t in range(T):
        x = xs[t]
        J_t = jax.jacobian(lambda hh: forward_step(hh, x, theta, R))(h)
        G_t = jax.jacobian(lambda th: forward_step(h, x, th, R))(theta)
        S = J_t @ S + G_t
        S_traj.append(S)
        h_next = forward_step(h, x, theta, R)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total, jnp.stack(S_traj)


# ---------------------------------------------------------------------
# Path 3: reduced exact RTRL -- hand-derived closed-form scalar-row
# recursion, deliberately hard-codes the coefficient "0.7" (R_BASE's
# eigenvalue on e1) -- used AS-IS even in the falsification sweep.
# ---------------------------------------------------------------------
def reduced_rtrl_grad(theta, h0, xs, qs, phases):
    w, beta, b = unpack(theta)
    w1 = w[0]
    T = xs.shape[0]
    E = jnp.zeros(6, dtype=jnp.float64)
    h = h0
    g_total = jnp.zeros(6, dtype=jnp.float64)
    E_traj = []
    for t in range(T):
        x = xs[t]
        z = z_of(h, x, theta)
        phip = 1.0 - jnp.tanh(z) ** 2
        direct = jnp.concatenate([h, jnp.array([x, 1.0])])
        E = (0.7 + phip * w1) * E + phip * direct
        E_traj.append(E)
        h_next = forward_step(h, x, theta, R_BASE)
        y = qs[t] @ h_next
        dl_dh0 = dell_dy(y, phases[t]) * qs[t][0]  # only e1-component of dl/dh matters
        g_total = g_total + dl_dh0 * E
        h = h_next
    return g_total, jnp.stack(E_traj)


def reconstruct_S_from_E(E_traj):
    T = E_traj.shape[0]
    S_recon = jnp.zeros((T, 4, 6), dtype=jnp.float64)
    S_recon = S_recon.at[:, 0, :].set(E_traj)
    return S_recon


# ---------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------
def make_setting(seed, T):
    rng = np.random.RandomState(seed)
    theta = jnp.array(np.concatenate([
        rng.randn(4) * 0.5,      # w
        rng.randn(1) * 0.3,      # beta
        rng.randn(1) * 0.3,      # b
    ]))
    h0 = jnp.array(rng.randn(4) * 0.3)
    xs = jnp.array(rng.randn(T) * 0.5)
    qs = jnp.array(rng.randn(T, 4))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    return theta, h0, xs, qs, phases


def run_correctness_suite():
    print("=" * 78)
    print("Correctness suite: R = R_BASE (exact flag holds)")
    print("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 100]
    eps_num = 1e-12
    worst_full_rel, worst_reduced_rel, worst_S_recon = 0.0, 0.0, 0.0
    rows = []
    for T in lengths:
        for seed in seeds:
            theta, h0, xs, qs, phases = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, xs, qs, phases, R_BASE)
            g_f, S_full = full_rtrl_grad(theta, h0, xs, qs, phases, R_BASE)
            g_r, E_traj = reduced_rtrl_grad(theta, h0, xs, qs, phases)
            S_recon = reconstruct_S_from_E(E_traj)

            max_abs_full = float(jnp.max(jnp.abs(g_f - g_b)))
            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + eps_num))
            max_abs_reduced = float(jnp.max(jnp.abs(g_r - g_b)))
            rel_reduced = float(jnp.linalg.norm(g_r - g_b) / (jnp.linalg.norm(g_b) + eps_num))
            max_abs_S = float(jnp.max(jnp.abs(S_recon - S_full)))

            worst_full_rel = max(worst_full_rel, rel_full)
            worst_reduced_rel = max(worst_reduced_rel, rel_reduced)
            worst_S_recon = max(worst_S_recon, max_abs_S)

            rows.append((T, seed, max_abs_full, rel_full, max_abs_reduced, rel_reduced, max_abs_S))
            print(f"  T={T:4d} seed={seed}  "
                  f"full: max|d|={max_abs_full:.3e} rel={rel_full:.3e}  "
                  f"reduced: max|d|={max_abs_reduced:.3e} rel={rel_reduced:.3e}  "
                  f"S_recon max|d|={max_abs_S:.3e}")
    print("-" * 78)
    print(f"WORST over all settings: full_rel={worst_full_rel:.3e}  "
          f"reduced_rel={worst_reduced_rel:.3e}  S_recon={worst_S_recon:.3e}")
    all_pass = worst_full_rel < 1e-10 and worst_reduced_rel < 1e-10 and worst_S_recon < 1e-10
    print(f"ALL < 1e-10: {all_pass}")
    print()
    print("Persistent sensitivity storage:")
    print("  Full:    4 x 6 = 24 float64 scalars")
    print("  Reduced: 1 x 6 =  6 float64 scalars")
    print("  Exact temporal sensitivity saving: 24/6 = 4x")
    return rows, all_pass


def run_falsification_suite():
    print()
    print("=" * 78)
    print("Falsification suite: R_eps = R_BASE + eps * e2 e1^T (breaks the flag)")
    print("=" * 78)
    eps_list = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    lengths = [1, 5, 20, 100]
    seed = 0
    E2 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    results = []
    for eps in eps_list:
        R_eps = R_BASE + eps * jnp.outer(E2, E1)
        for T in lengths:
            theta, h0, xs, qs, phases = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, xs, qs, phases, R_eps)
            g_f, S_full = full_rtrl_grad(theta, h0, xs, qs, phases, R_eps)
            # reduced path deliberately still uses the OLD d=1 recursion,
            # unaware of R_eps (its forward-dynamics call inside
            # reduced_rtrl_grad still uses R_BASE, but g_b/g_f above use
            # R_eps -- this mismatch IS the point: the reduced path's own
            # internal rollout also silently uses the wrong dynamics).
            g_r, E_traj = reduced_rtrl_grad_eps(theta, h0, xs, qs, phases, R_eps)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            rel_reduced = float(jnp.linalg.norm(g_r - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            max_abs_reduced = float(jnp.max(jnp.abs(g_r - g_b)))
            results.append((eps, T, rel_full, rel_reduced, max_abs_reduced))
            print(f"  eps={eps:.0e}  T={T:4d}  full_rel={rel_full:.3e}  "
                  f"reduced_rel={rel_reduced:.3e}  reduced_max_abs={max_abs_reduced:.3e}")
    return results


def reduced_rtrl_grad_eps(theta, h0, xs, qs, phases, R_true):
    """Same OLD d=1 reduced recursion (hard-coded 0.7 coefficient), but
    now the actual forward dynamics driving h_t use R_true (=R_eps) --
    i.e. the reduced path is evaluated on trajectories generated by the
    TRUE (leaky) system, using its stale invariant-subspace assumption."""
    w, beta, b = unpack(theta)
    w1 = w[0]
    T = xs.shape[0]
    E = jnp.zeros(6, dtype=jnp.float64)
    h = h0
    g_total = jnp.zeros(6, dtype=jnp.float64)
    E_traj = []
    for t in range(T):
        x = xs[t]
        z = z_of(h, x, theta)
        phip = 1.0 - jnp.tanh(z) ** 2
        direct = jnp.concatenate([h, jnp.array([x, 1.0])])
        E = (0.7 + phip * w1) * E + phip * direct
        E_traj.append(E)
        h_next = forward_step(h, x, theta, R_true)
        y = qs[t] @ h_next
        dl_dh0 = dell_dy(y, phases[t]) * qs[t][0]
        g_total = g_total + dl_dh0 * E
        h = h_next
    return g_total, jnp.stack(E_traj)


def run_structural_diagnostics():
    print()
    print("=" * 78)
    print("Structural diagnostics at eps=0 (R = R_BASE)")
    print("=" * 78)
    R_np = np.asarray(R_BASE)
    e1_np, e4_np = np.asarray(E1), np.asarray(E4)

    # 1. controllability / forward reachability from input e4
    K_e4 = krylov_subspace(R_np, e4_np[:, None])
    print(f"1. Controllability/reachability rank from e4: {K_e4.shape[1]} / 4"
          f"  (Krylov subspace of R seeded at e4)")

    # 2. dimension of generated algebra for a representative theta
    theta0, *_ = make_setting(seed=0, T=1)
    w0 = np.asarray(theta0[:4])
    Q = np.outer(e1_np, w0)  # e1 w^T, the rank-1 correction direction in J_t
    alg_basis = algebra_closure([R_np, Q])
    print(f"2. Generated algebra dim(span({{R, e1 w^T}}) closed under mult) "
          f"for representative w={w0.round(4).tolist()}: {len(alg_basis)} / 16")

    # 3. nonzero commutator
    comm = R_np @ Q - Q @ R_np
    comm_norm = np.linalg.norm(comm)
    print(f"3. ||[R, e1 w^T]||_F = {comm_norm:.6e}  (nonzero => noncommutative)")

    # 4. commutant dimension (matrices commuting with BOTH R and Q)
    I4 = np.eye(4)
    ad_R = np.kron(I4, R_np) - np.kron(R_np.T, I4)  # vec(RX-XR)
    ad_Q = np.kron(I4, Q) - np.kron(Q.T, I4)        # vec(QX-XQ)
    M = np.concatenate([ad_R, ad_Q], axis=0)        # (32,16)
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    commutant_dim = 16 - rank_M
    print(f"4. Commutant dimension (of {{R, e1 w^T}}): {commutant_dim} / 16"
          f"  (rank of stacked commutator operator = {rank_M})")

    # Also: Krylov subspace seeded at e1 under R alone -- should be
    # EXACTLY span(e1) (dim 1) at eps=0, confirming the flag claim
    # structurally (not just via the gradient-matching numbers above).
    K_e1 = krylov_subspace(R_np, e1_np[:, None])
    print(f"   (sanity) Krylov subspace of R seeded at e1: dim={K_e1.shape[1]} "
          f"(expect 1 = span(e1) at eps=0)")
    return dict(controllability_rank=int(K_e4.shape[1]), algebra_dim=len(alg_basis),
                commutator_norm=float(comm_norm), commutant_dim=int(commutant_dim),
                krylov_e1_dim=int(K_e1.shape[1]))


def main():
    rows, all_pass = run_correctness_suite()
    falsification = run_falsification_suite()
    structural = run_structural_diagnostics()
    print()
    print("=" * 78)
    print(f"CORRECTNESS SUITE PASS (<1e-10 everywhere): {all_pass}")
    print("=" * 78)


if __name__ == "__main__":
    main()
