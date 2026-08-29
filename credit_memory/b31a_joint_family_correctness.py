"""Phase B31a -- correctness gate for JOINTLY training all V-valued
recurrent-Jacobian families (R_V, K, B_V, C_V, C_U, Phi_theta) on the
same r=64, d=4 flag architecture as B30a/B30b, while R_U, D_U stay
fixed. Standalone; no training here (that is B31b).

Architecture (only R_U, D_U remain FIXED/untrained; everything else
that touches v_{t+1} is now part of theta):
  u_{t+1} = R_U u_t + D_U x_t                        (fixed, untrained)
  v_{t+1} = R_V v_t + K u_t + B_V Phi_theta(C_V v_t + C_U u_t, x_t)
theta = (R_V, K, B_V, C_V, C_U, Phi's W1,b1,W2,b2). No D_V family
exists in this implementation (v_{t+1} has no direct x_t term), so
P_{D_V}=0 -- not fabricated.

Why every one of these families is STRUCTURALLY (not just
numerically) V-valued: each appears ONLY inside the v_{t+1} equation
-- none of R_V,K,B_V,C_V,C_U,theta appear anywhere in u_{t+1}'s
formula (which reads only R_U,D_U,x_t). Therefore the direct term
G_t=d(s_{t+1})/d(family)|_{s_t fixed} has IDENTICALLY ZERO U-rows for
every one of these families, for EVERY value they take -- this is a
fact about the functional form of u_{t+1}, not about specific
parameter values, so it survives arbitrary optimizer updates to these
families (as long as u_{t+1}'s own formula is never touched, i.e. as
long as R_U/D_U stay fixed and untrained, per instruction).

Run: python -m credit_memory.b31a_joint_family_correctness
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

D_U_DIM = 60
D_V_DIM = 4
R_DIM = D_U_DIM + D_V_DIM
C_DIM = 8
K_OUT = 8
H_DIM = 560
X_DIM = 1

P_MAT = jnp.concatenate([jnp.zeros((D_U_DIM, D_V_DIM)), jnp.eye(D_V_DIM)], axis=0)

FAMILY_SHAPES = dict(
    R_V=(D_V_DIM, D_V_DIM),
    K=(D_V_DIM, D_U_DIM),
    B_V=(D_V_DIM, K_OUT),
    C_V=(C_DIM, D_V_DIM),
    C_U=(C_DIM, D_U_DIM),
    W1=(H_DIM, C_DIM + X_DIM),
    b1=(H_DIM,),
    W2=(K_OUT, H_DIM),
    b2=(K_OUT,),
)
PHI_KEYS = ("W1", "b1", "W2", "b2")
TRAINABLE_KEYS = ("R_V", "K", "B_V", "C_V", "C_U") + PHI_KEYS


def make_stable_dense(n, rng, radius):
    M = rng.randn(n, n) / np.sqrt(n)
    eig = np.max(np.abs(np.linalg.eigvals(M)))
    return M * (radius / eig)


def make_fixed_consts(seed=12345):
    """Only R_U, D_U remain fixed/untrained in B31."""
    rng = np.random.RandomState(seed)
    R_U = jnp.array(make_stable_dense(D_U_DIM, rng, radius=0.85))
    D_U = jnp.array(rng.randn(D_U_DIM) * 0.5)
    return dict(R_U=R_U, D_U=D_U)


def make_theta(seed, r_v_radius=0.80):
    rng = np.random.RandomState(seed)
    theta = dict(
        R_V=jnp.array(make_stable_dense(D_V_DIM, rng, radius=r_v_radius)),
        K=jnp.array(rng.randn(D_V_DIM, D_U_DIM) * (0.15 / np.sqrt(D_U_DIM))),
        B_V=jnp.array(rng.randn(D_V_DIM, K_OUT) * (0.5 / np.sqrt(K_OUT))),
        C_V=jnp.array(rng.randn(C_DIM, D_V_DIM) * (0.5 / np.sqrt(D_V_DIM))),
        C_U=jnp.array(rng.randn(C_DIM, D_U_DIM) * (0.5 / np.sqrt(D_U_DIM))),
        W1=jnp.array(rng.randn(H_DIM, C_DIM + X_DIM) * (1.0 / np.sqrt(C_DIM + X_DIM))),
        b1=jnp.array(rng.randn(H_DIM) * 0.05),
        W2=jnp.array(rng.randn(K_OUT, H_DIM) * (1.0 / np.sqrt(H_DIM))),
        b2=jnp.array(rng.randn(K_OUT) * 0.05),
    )
    return theta


def phi_forward(z, x, theta):
    xin = jnp.concatenate([z, jnp.atleast_1d(x)])
    hact = jnp.tanh(theta["W1"] @ xin + theta["b1"])
    return theta["W2"] @ hact + theta["b2"]


def u_step(u, x, c):
    return c["R_U"] @ u + c["D_U"] * x


def v_step(v, u_val, x, theta):
    z = theta["C_V"] @ v + theta["C_U"] @ u_val
    phi_out = phi_forward(z, x, theta)
    return theta["R_V"] @ v + theta["K"] @ u_val + theta["B_V"] @ phi_out


def full_step_state(s, x, theta, c):
    u, v = s[:D_U_DIM], s[D_U_DIM:]
    v_next = v_step(v, u, x, theta)
    u_next = u_step(u, x, c)
    return jnp.concatenate([u_next, v_next])


def rollout_full_states(u0, v0, xs, theta, c):
    def step(carry, x):
        u, v = carry
        v_next = v_step(v, u, x, theta)
        u_next = u_step(u, x, c)
        return (u_next, v_next), jnp.concatenate([u_next, v_next])
    _, Ss = jax.lax.scan(step, (u0, v0), xs)
    return Ss


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
        J_v = jax.jacobian(lambda vv: v_step(vv, u_val, x, theta))(v)
        G_v = jax.jacobian(lambda th_flat: v_step(v, u_val, x, unravel(th_flat)))(theta_flat)
        E = J_v @ E + G_v
        E_traj.append(E)
        v_next = v_step(v, u_val, x, theta)
        u_next = u_traj[t + 1]
        s_next = jnp.concatenate([u_next, v_next])
        y = qs[t] @ s_next
        dl_dv = dell_dy(y, phases[t]) * qs[t][D_U_DIM:]
        g_total = g_total + dl_dv @ E
        v = v_next
    return g_total, jnp.stack(E_traj)


def reconstruct_S_from_E(E_traj):
    T, _, P_c = E_traj.shape
    S_recon = jnp.zeros((T, R_DIM, P_c), dtype=jnp.float64)
    return S_recon.at[:, D_U_DIM:, :].set(E_traj)


def make_setting(seed, T, c):
    rng = np.random.RandomState(seed)
    theta = make_theta(seed + 9999)
    u0 = jnp.array(rng.randn(D_U_DIM) * 0.2)
    v0 = jnp.array(rng.randn(D_V_DIM) * 0.2)
    xs = jnp.array(rng.randn(T) * 0.5)
    qs = jnp.array(rng.randn(T, R_DIM))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    return theta, u0, v0, xs, qs, phases


def report_family_table():
    print("=" * 78)
    print("Per-family accounting")
    print("=" * 78)
    total = 0
    for fam in TRAINABLE_KEYS:
        shape = FAMILY_SHAPES[fam]
        p_f = int(np.prod(shape))
        total += p_f
        v_valued = "YES (structural -- never appears in u_{t+1})"
        print(f"  {fam:4s}  shape={shape}  P_f={p_f:5d}  full=64x{p_f}={64*p_f:6d}  "
              f"reduced=4x{p_f}={4*p_f:5d}  V-valued: {v_valued}")
    print(f"  D_V: not present in this implementation -- P_DV=0 (not fabricated)")
    print(f"  TOTAL P_c = {total}")
    print(f"  M_full = 64*P_c = {64*total}  ({64*total*8/1e6:.3f} MB)")
    print(f"  M_reduced = 4*P_c = {4*total}  ({4*total*8/1e6:.3f} MB)")
    print(f"  ratio = {64*total/(4*total):.2f}x")
    return total


def run_correctness_suite():
    c = make_fixed_consts()
    theta0 = make_theta(0)
    theta_flat0, unravel = ravel_pytree(theta0)
    P_c = theta_flat0.shape[0]
    print()
    print("=" * 78)
    print(f"B31a correctness suite: joint theta over {TRAINABLE_KEYS}, P_c={P_c}")
    print("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 50]
    eps_num = 1e-12
    worst_full_rel = worst_reduced_rel = worst_S_recon = worst_U_leak = worst_per_family_U = 0.0

    for T in lengths:
        for seed in seeds:
            theta, u0, v0, xs, qs, phases = make_setting(seed, T, c)
            theta_flat, unravel_s = ravel_pytree(theta)

            g_b = grad_bptt_fn(theta, u0, v0, xs, qs, phases, c)
            g_b_flat, _ = ravel_pytree(g_b)

            g_f, S_full = full_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel_s, theta_flat, P_c)
            g_r, E_traj = reduced_rtrl_grad(theta, u0, v0, xs, qs, phases, c, unravel_s, theta_flat, P_c)
            S_recon = reconstruct_S_from_E(E_traj)

            rel_full = float(jnp.linalg.norm(g_f - g_b_flat) / (jnp.linalg.norm(g_b_flat) + eps_num))
            rel_reduced = float(jnp.linalg.norm(g_r - g_b_flat) / (jnp.linalg.norm(g_b_flat) + eps_num))
            max_abs_S = float(jnp.max(jnp.abs(S_recon - S_full)))
            max_abs_U_rows = float(jnp.max(jnp.abs(S_full[:, :D_U_DIM, :])))

            worst_full_rel = max(worst_full_rel, rel_full)
            worst_reduced_rel = max(worst_reduced_rel, rel_reduced)
            worst_S_recon = max(worst_S_recon, max_abs_S)
            worst_U_leak = max(worst_U_leak, max_abs_U_rows)

            print(f"  T={T:3d} seed={seed}  full_rel={rel_full:.3e}  reduced_rel={rel_reduced:.3e}  "
                  f"S_recon max|d|={max_abs_S:.3e}  full_S_Urows max|.|={max_abs_U_rows:.3e}")

    print("-" * 78)
    print(f"WORST: full_rel={worst_full_rel:.3e}  reduced_rel={worst_reduced_rel:.3e}  "
          f"S_recon={worst_S_recon:.3e}  U_leak={worst_U_leak:.3e}")
    all_pass = worst_full_rel < 1e-8 and worst_reduced_rel < 1e-8 and worst_S_recon < 1e-8 and worst_U_leak < 1e-8
    print(f"ALL < 1e-8: {all_pass}")

    # per-family U-rows check (numerical confirmation of the analytical claim)
    print()
    print("Per-family direct-source U-rows check (should all be exactly 0):")
    theta, u0, v0, xs, qs, phases = make_setting(0, 10, c)
    theta_flat, unravel_s = ravel_pytree(theta)
    s = jnp.concatenate([u0, v0])
    worst_family_leak = 0.0
    for t in range(3):
        x = xs[t]
        for fam in TRAINABLE_KEYS:
            def f_fam(p_fam, fam=fam, s=s, x=x, theta=theta):
                th2 = dict(theta)
                th2[fam] = p_fam
                return full_step_state(s, x, th2, c)
            G_fam = jax.jacobian(f_fam)(theta[fam])  # (64, *shape)
            leak = float(jnp.max(jnp.abs(G_fam[:D_U_DIM])))
            worst_family_leak = max(worst_family_leak, leak)
        s = full_step_state(s, x, theta, c)
    print(f"  worst per-family U-rows leak over sampled (t,family): {worst_family_leak:.3e}  (expect exactly 0)")

    return dict(all_pass=all_pass, worst_full_rel=worst_full_rel, worst_reduced_rel=worst_reduced_rel,
                worst_S_recon=worst_S_recon, worst_U_leak=worst_U_leak, worst_family_leak=worst_family_leak, P_c=P_c)


def run_invariance_after_perturbation():
    print()
    print("=" * 78)
    print("Invariance survives arbitrary trainable-parameter values (not just init)")
    print("=" * 78)
    c = make_fixed_consts()
    P_np = np.asarray(P_MAT)
    proj_perp = np.eye(R_DIM) - P_np @ P_np.T
    rng = np.random.RandomState(4242)
    max_leak = 0.0
    n_checks = 12
    for i in range(n_checks):
        # simulate a "post-optimizer-step" theta: fresh random draw, NOT
        # just the initialization -- large, varied perturbations.
        theta_pert = make_theta(seed=5000 + i, r_v_radius=float(rng.uniform(0.1, 0.95)))
        u_s = jnp.array(rng.randn(D_U_DIM) * 0.3)
        v_s = jnp.array(rng.randn(D_V_DIM) * 0.3)
        x_s = float(rng.randn() * 0.5)
        s_s = jnp.concatenate([u_s, v_s])
        J_t = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta_pert, c))(s_s))
        leak = np.linalg.norm(proj_perp @ J_t @ P_np)
        max_leak = max(max_leak, leak)
    print(f"  max ||(I-PP^+) J_t P|| over {n_checks} INDEPENDENTLY-DRAWN (post-update-like) theta values: "
          f"{max_leak:.3e}  (expect exactly 0 -- structural, not incidental to init)")
    return dict(max_leak_after_perturbation=float(max_leak), n_checks=n_checks)


def main():
    total_pc = report_family_table()
    corr = run_correctness_suite()
    inv = run_invariance_after_perturbation()
    print()
    print("=" * 78)
    print(f"B31a CORRECTNESS SUITE PASS (<1e-8 everywhere, incl. per-family U-rows and post-perturbation invariance): "
          f"{corr['all_pass'] and corr['worst_family_leak'] < 1e-10 and inv['max_leak_after_perturbation'] < 1e-10}")
    print("=" * 78)


if __name__ == "__main__":
    main()
