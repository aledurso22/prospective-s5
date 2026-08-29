"""Phase B32a -- minimal r=2, d=1 MOVING invariant-bundle construction.
Generalizes B29's fixed-flag theorem: the exact-compression condition
is not a fixed invariant subspace V, but a causally moving,
source-compatible bundle E_t with J_t E_t + im(G_t) subseteq E_{t+1}.
B29/B30/B31's fixed flag is the special case E_t=V for all t. This
phase tests the minimal nontrivial case where E_t itself MOVES, while
the AMBIENT Jacobian is fully dense with no common fixed invariant
line at all.

Construction (given):
  A = [[0.8,0.3],[0,0.5]]           (z-space recurrence matrix; e1 is
                                      an eigenvector, A e1 = 0.8 e1 --
                                      the SAME fact underlying B29,
                                      just about to be viewed through a
                                      rotating ambient frame)
  P_{2k}=I,  P_{2k+1} = (1/sqrt2)[[1,-1],[1,1]]   (a 45-degree rotation)
  z_{t+1} = A z_t + e1 (theta.phi_t)      (theta enters LINEARLY here;
                                            phi_t is a per-step input
                                            feature vector, P_c-dim)
  h_t = P_t z_t                           (AMBIENT/observed state)

Consequently J_t^h = P_{t+1} A P_t^{-1} is fully dense and ALTERNATES
between two different matrices (no common fixed invariant line: P_odd
is a genuine rotation with complex eigenvalues, so it alone has no
real invariant line). Yet the exact sensitivity D_theta h_t is
confined to the MOVING 1-dim bundle E_t=span(P_t e1), because in the
UNDERLYING z-coordinates it is still confined to span(e1) (B29's exact
mechanism), and h_t=P_t z_t just rotates that line along with time.

Four independent exact gradient paths:
  1. Ambient BPTT (jax.grad through jax.lax.scan on the h-recurrence).
  2. Full ambient RTRL: S_{t+1}=J_t^h S_t + G_t^h, r x P_c, via
     per-step autodiff jacobians of the AMBIENT step (no z-space
     shortcut used).
  3. Prescribed moving-bundle reduced RTRL: E_t (1 x P_c), obtained by
     projecting the SAME autodiff J_t^h/G_t^h onto the KNOWN moving
     basis vectors b_t=P_t e1, b_{t+1}=P_{t+1} e1 -- a general
     restricted-operator algorithm (not a hand-derived formula),
     generalized to a time-varying basis instead of a fixed one.
  4. General dynamic-QR/rank-factorization RTRL: maintains the full
     ambient S_t but re-factorizes it via SVD at every step and
     TRUNCATES to the numerically discovered rank, with NO knowledge
     of P_t or e1 at all -- demonstrating the reduction is discoverable
     by a generic algorithm, not merely prescribed.

Run: python -m credit_memory.b32a_moving_bundle_r2d1
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from credit_memory.b25_nonlinear_credit import algebra_closure

R_DIM = 2
P_C = 6

A_MAT = jnp.array([[0.8, 0.3], [0.0, 0.5]], dtype=jnp.float64)
E1 = jnp.array([1.0, 0.0], dtype=jnp.float64)
E2 = jnp.array([0.0, 1.0], dtype=jnp.float64)
I2 = jnp.eye(2, dtype=jnp.float64)
ROT = (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, -1.0], [1.0, 1.0]], dtype=jnp.float64)


def P_of(t):
    return I2 if (t % 2 == 0) else ROT


def make_P_sequence(T):
    return jnp.stack([P_of(t) for t in range(T + 1)])  # (T+1,2,2)


def z_step(z, phi, theta, eps):
    """eps=0: the theorem's exact construction. eps>0: falsification --
    a PERSISTENT source component (eps*e2*(theta.phi)) injected every
    step, OUTSIDE the moving bundle span(e1) in z-coordinates."""
    scal = theta @ phi
    return A_MAT @ z + E1 * scal + eps * E2 * scal


def h_step(h, phi, theta, P_t, P_next, eps):
    z = P_t.T @ h
    z_next = z_step(z, phi, theta, eps)
    return P_next @ z_next


def rollout_h(h0, phis, theta, Ps, eps):
    def step(h, inp):
        phi, P_t, P_next = inp
        h_next = h_step(h, phi, theta, P_t, P_next, eps)
        return h_next, h_next
    xs = (phis, Ps[:-1], Ps[1:])
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs  # (T,2)


def ell(y, phase):
    return jnp.sin(y + phase) + 0.5 * y ** 2


def dell_dy(y, phase):
    return jnp.cos(y + phase) + y


def loss_bptt(theta, h0, phis, qs, phases, Ps, eps):
    Hs = rollout_h(h0, phis, theta, Ps, eps)
    ys = jnp.einsum("ti,ti->t", Hs, qs)
    return jnp.sum(ell(ys, phases))


grad_bptt_fn = jax.jit(jax.grad(loss_bptt, argnums=0), static_argnums=())


def full_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps):
    T = phis.shape[0]
    h = h0
    S = jnp.zeros((R_DIM, P_C), dtype=jnp.float64)
    g_total = jnp.zeros(P_C, dtype=jnp.float64)
    S_traj = []
    for t in range(T):
        phi = phis[t]
        P_t, P_next = Ps[t], Ps[t + 1]
        J_t = jax.jacobian(lambda hh: h_step(hh, phi, theta, P_t, P_next, eps))(h)
        G_t = jax.jacobian(lambda th: h_step(h, phi, th, P_t, P_next, eps))(theta)
        S = J_t @ S + G_t
        S_traj.append(S)
        h_next = h_step(h, phi, theta, P_t, P_next, eps)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total, jnp.stack(S_traj)


def bundle_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps):
    """Prescribed moving-bundle reduced RTRL: E_t (1,P_c), obtained by
    projecting the SAME autodiff J_t^h/G_t^h onto the KNOWN basis
    vectors b_t=P_t e1. Deliberately uses ONLY the eps=0 bundle
    assumption (b_t=P_t e1) even when eps>0 is passed in for the
    forward dynamics -- this is the falsification path."""
    T = phis.shape[0]
    h = h0
    E = jnp.zeros(P_C, dtype=jnp.float64)
    g_total = jnp.zeros(P_C, dtype=jnp.float64)
    E_traj = []
    for t in range(T):
        phi = phis[t]
        P_t, P_next = Ps[t], Ps[t + 1]
        b_t = P_t @ E1
        b_next = P_next @ E1
        J_t = jax.jacobian(lambda hh: h_step(hh, phi, theta, P_t, P_next, eps))(h)
        G_t = jax.jacobian(lambda th: h_step(h, phi, th, P_t, P_next, eps))(theta)
        alpha_t = b_next @ (J_t @ b_t)
        g_row_t = b_next @ G_t
        E = alpha_t * E + g_row_t
        E_traj.append(E)
        h_next = h_step(h, phi, theta, P_t, P_next, eps)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]  # (2,), full ambient dL/dh_next
        # dL/dtheta contribution: dl_dh . S_recon = dl_dh . (b_next (outer) E) = (dl_dh.b_next) * E
        g_total = g_total + (dl_dh @ b_next) * E
        h = h_next
    return g_total, jnp.stack(E_traj)


def dynamic_qr_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps, rtol=1e-9):
    """General dynamic-QR/rank-factorization RTRL: NO knowledge of P_t
    or e1 at all. Maintains the full ambient S_t, but at every step
    re-factorizes via SVD and TRUNCATES to the numerically discovered
    rank before propagating -- discovers (not assumes) low rank."""
    T = phis.shape[0]
    h = h0
    S = jnp.zeros((R_DIM, P_C), dtype=jnp.float64)
    g_total = jnp.zeros(P_C, dtype=jnp.float64)
    ranks = []
    sv_traj = []
    for t in range(T):
        phi = phis[t]
        P_t, P_next = Ps[t], Ps[t + 1]
        J_t = jax.jacobian(lambda hh: h_step(hh, phi, theta, P_t, P_next, eps))(h)
        G_t = jax.jacobian(lambda th: h_step(h, phi, th, P_t, P_next, eps))(theta)
        S_candidate = J_t @ S + G_t
        U, s, Vt = jnp.linalg.svd(S_candidate, full_matrices=False)
        s_np = np.asarray(s)
        smax = s_np[0] if s_np[0] > 0 else 1.0
        rank_t = int(np.sum(s_np > rtol * smax))
        rank_t = max(rank_t, 1) if smax > 1e-15 else 0
        ranks.append(rank_t)
        sv_traj.append(s_np.tolist())
        S_trunc = (U[:, :rank_t] * s[:rank_t]) @ Vt[:rank_t, :] if rank_t > 0 else jnp.zeros_like(S_candidate)
        S = S_trunc
        h_next = h_step(h, phi, theta, P_t, P_next, eps)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total, ranks, sv_traj


def make_setting(seed, T):
    rng = np.random.RandomState(seed)
    theta = jnp.array(rng.randn(P_C) * 0.5)
    h0 = jnp.array(rng.randn(2) * 0.3)
    phis = jnp.array(rng.randn(T, P_C) * 0.5)
    qs = jnp.array(rng.randn(T, 2))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    Ps = make_P_sequence(T)
    return theta, h0, phis, qs, phases, Ps


def run_correctness_suite():
    print("=" * 78)
    print("B32a correctness suite: r=2, d=1 MOVING bundle, eps=0 (theorem holds)")
    print("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 100]
    eps_num = 1e-12
    worst = dict(full=0.0, bundle=0.0, qr=0.0, S_recon=0.0)
    rank_report = {}
    for T in lengths:
        for seed in seeds:
            theta, h0, phis, qs, phases, Ps = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, phis, qs, phases, Ps, 0.0)
            g_f, S_full = full_rtrl_grad(theta, h0, phis, qs, phases, Ps, 0.0)
            g_e, E_traj = bundle_rtrl_grad(theta, h0, phis, qs, phases, Ps, 0.0)
            g_q, ranks, sv_traj = dynamic_qr_rtrl_grad(theta, h0, phis, qs, phases, Ps, 0.0)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + eps_num))
            rel_bundle = float(jnp.linalg.norm(g_e - g_b) / (jnp.linalg.norm(g_b) + eps_num))
            rel_qr = float(jnp.linalg.norm(g_q - g_b) / (jnp.linalg.norm(g_b) + eps_num))

            # E_traj[t] represents D_theta h_{t+1} (the NEXT state), so it
            # embeds via b_{t+1}=P_{t+1} e1, NOT b_t -- matches how E is
            # updated (appended AFTER the alpha_t/g_row_t step) inside
            # bundle_rtrl_grad.
            b_traj = jnp.stack([P_of(t + 1) @ E1 for t in range(T)])  # (T,2)
            S_recon = jnp.einsum("ti,tj->tij", b_traj, E_traj)
            max_abs_S = float(jnp.max(jnp.abs(S_recon - S_full)))

            worst["full"] = max(worst["full"], rel_full)
            worst["bundle"] = max(worst["bundle"], rel_bundle)
            worst["qr"] = max(worst["qr"], rel_qr)
            worst["S_recon"] = max(worst["S_recon"], max_abs_S)
            rank_report[(T, seed)] = ranks

            print(f"  T={T:3d} seed={seed}  full_rel={rel_full:.3e}  bundle_rel={rel_bundle:.3e}  "
                  f"qr_rel={rel_qr:.3e}  S_recon max|d|={max_abs_S:.3e}  "
                  f"qr_ranks(unique)={sorted(set(ranks))}")
    print("-" * 78)
    print(f"WORST: full_rel={worst['full']:.3e}  bundle_rel={worst['bundle']:.3e}  "
          f"qr_rel={worst['qr']:.3e}  S_recon={worst['S_recon']:.3e}")
    all_pass = all(v < 1e-8 for v in worst.values())
    print(f"ALL < 1e-8: {all_pass}")

    # long-sequence rank check
    theta, h0, phis, qs, phases, Ps = make_setting(0, 100)
    _, ranks, sv_traj = dynamic_qr_rtrl_grad(theta, h0, phis, qs, phases, Ps, 0.0)
    print(f"Sensitivity rank over T=100 steps (dynamic QR, no prior knowledge): "
          f"unique ranks observed = {sorted(set(ranks))}  (expect {{1}})")
    second_sv = [sv[1] if len(sv) > 1 else 0.0 for sv in sv_traj]
    print(f"  max 2nd singular value over all 100 steps: {max(second_sv):.3e}  (expect ~machine eps)")

    print()
    print(f"Persistent sensitivity storage: full=2x{P_C}={2*P_C} floats, reduced=1x{P_C}={P_C} floats, "
          f"ratio={2*P_C/P_C:.1f}x")
    return dict(all_pass=all_pass, worst=worst)


def run_falsification_suite():
    print()
    print("=" * 78)
    print("Falsification: persistent source leak OUTSIDE the moving bundle (eps*e2)")
    print("=" * 78)
    eps_list = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    lengths = [1, 5, 20, 100]
    seed = 0
    for eps in eps_list:
        for T in lengths:
            theta, h0, phis, qs, phases, Ps = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, phis, qs, phases, Ps, eps)
            g_f, S_full = full_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps)
            g_e, E_traj = bundle_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps)
            g_q, ranks, sv_traj = dynamic_qr_rtrl_grad(theta, h0, phis, qs, phases, Ps, eps)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            rel_bundle = float(jnp.linalg.norm(g_e - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            rel_qr = float(jnp.linalg.norm(g_q - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            unique_ranks = sorted(set(ranks))
            print(f"  eps={eps:.0e}  T={T:3d}  full_rel={rel_full:.3e}  bundle_rel={rel_bundle:.3e}  "
                  f"qr_rel={rel_qr:.3e}  qr_ranks(unique)={unique_ranks}")


def run_structural_diagnostics():
    print()
    print("=" * 78)
    print("Structural diagnostics (eps=0)")
    print("=" * 78)
    A_np = np.asarray(A_MAT)
    ROT_np = np.asarray(ROT)
    I2_np = np.eye(2)

    J_even_to_odd = ROT_np @ A_np @ I2_np.T   # P_{t+1}=ROT, P_t=I
    J_odd_to_even = I2_np @ A_np @ ROT_np.T   # P_{t+1}=I, P_t=ROT

    print(f"1. Ambient Jacobians dense:")
    print(f"   J(even->odd) =\n{J_even_to_odd}")
    print(f"   J(odd->even) =\n{J_odd_to_even}")
    dense = np.all(np.abs(J_even_to_odd) > 1e-12) and np.all(np.abs(J_odd_to_even) > 1e-12)
    print(f"   both fully dense (no zero entries): {dense}")

    alg_basis = algebra_closure([J_even_to_odd, J_odd_to_even])
    print(f"2. Static generated algebra dim(span{{J_e2o,J_o2e}} closed under mult) = {len(alg_basis)} / 4 "
          f"(expect 4 = M_2)")

    rot_eigs = np.linalg.eigvals(ROT_np)
    print(f"3. No common invariant real line: eigenvalues of ROT = {rot_eigs} "
          f"(complex => ROT alone has no real invariant line, so no common one for {{A,ROT}} either)")

    # commutant of {J_even_to_odd, J_odd_to_even}
    I2v = np.eye(2)
    def ad(M):
        return np.kron(I2v, M) - np.kron(M.T, I2v)
    M_stack = np.concatenate([ad(J_even_to_odd), ad(J_odd_to_even)], axis=0)
    rank_M = np.linalg.matrix_rank(M_stack, tol=1e-9)
    commutant_dim = 4 - rank_M
    print(f"   commutant dimension = {commutant_dim} / 4 (expect 1 = scalar multiples of I)")

    # moving-bundle residual: J_t (P_t e1) vs span(P_{t+1} e1)
    e1_np = np.asarray(E1)
    b0 = I2_np @ e1_np  # t=0 (even), b_0
    b1 = ROT_np @ e1_np  # t=1 (odd), b_1
    Jb0 = J_even_to_odd @ b0
    # component of Jb0 orthogonal to b1
    resid_bundle = Jb0 - (b1 @ Jb0) * b1  # b1 is unit norm
    print(f"4. Moving-bundle residual ||J_t b_t - proj_{{b_next}}(J_t b_t)|| = {np.linalg.norm(resid_bundle):.3e} "
          f"(expect ~0)")

    # source-alignment residual: G_t^h column space vs span(b_next)
    theta0, h0, phis, qs, phases, Ps = make_setting(0, 5)
    phi0 = phis[0]
    G0 = np.asarray(jax.jacobian(lambda th: h_step(h0, phi0, th, Ps[0], Ps[1], 0.0))(theta0))
    # component of each column orthogonal to b1
    proj = np.outer(b1, b1) @ G0
    resid_source = np.linalg.norm(G0 - proj)
    print(f"5. Source-alignment residual ||G_t - proj_{{b_next}}(G_t)|| = {resid_source:.3e} (expect ~0)")

    return dict(dense=bool(dense), algebra_dim=len(alg_basis), commutant_dim=int(commutant_dim),
                resid_bundle=float(np.linalg.norm(resid_bundle)), resid_source=float(resid_source))


def main():
    corr = run_correctness_suite()
    run_falsification_suite()
    struct = run_structural_diagnostics()
    print()
    print("=" * 78)
    print(f"B32a CORRECTNESS SUITE PASS (<1e-8 everywhere): {corr['all_pass']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
