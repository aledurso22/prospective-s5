"""Phase B33b-min -- ACTUAL trainable recurrent model with full-rank
sensitivity and 2-scalar exact eligibility. Unlike B33a (explicitly
abstract, J_t/G_t given directly), here J_t=D_h F_theta and
G_t=D_theta F_theta are obtained via genuine autodiff on a real
recurrent parameterization F_theta -- this IS legitimate RTRL.

Model: r=P=64. theta in R^64 is the persistent trainable parameter
vector (same space as h). K=I-2vv^T is a Householder involution
(K^2=I), applied ONLY implicitly (Kx=x-2v(v.x), O(r) cost, O(r)
storage) -- never materialized as a dense 64x64 matrix except for
diagnostics.

  h_{t+1} = (alpha_t I + beta_t K) h_t + (gamma_t I + delta_t K) theta + c_t

alpha_t,beta_t,gamma_t,delta_t,c_t are CAUSAL and EXOGENOUS wrt theta
and h: generated from an exogenous input x_t by a FIXED, FROZEN
(never-trained) generator network, using a stable eigenvalue
parameterization (rho*tanh) so both eigenspaces of K stay within
(-rho,rho), rho=0.95.

Run: python -m credit_memory.b33b_min_trainable
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

R_DIM = 64
X_DIM = 4
GEN_HIDDEN = 32
RHO = 0.95


def make_v(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(R_DIM)
    v = v / np.linalg.norm(v)
    return jnp.array(v)


def make_gen_params(seed):
    """Fixed, FROZEN coefficient generator -- never trained. Maps an
    exogenous input x_t to (u_plus,u_minus,gamma,delta,c) causally and
    independently of theta/h."""
    rng = np.random.RandomState(seed)
    scale1 = 1.0 / np.sqrt(X_DIM)
    scale2 = 1.0 / np.sqrt(GEN_HIDDEN)
    return dict(
        W1=jnp.array(rng.randn(GEN_HIDDEN, X_DIM) * scale1),
        b1=jnp.array(rng.randn(GEN_HIDDEN) * 0.1),
        W_scalars=jnp.array(rng.randn(4, GEN_HIDDEN) * scale2),
        b_scalars=jnp.array(rng.randn(4) * 0.1),
        W_c=jnp.array(rng.randn(R_DIM, GEN_HIDDEN) * scale2),
        b_c=jnp.array(rng.randn(R_DIM) * 0.1),
    )


def gen_forward(x_t, gen_params):
    """Returns (alpha_t, beta_t, gamma_t, delta_t, c_t) -- all exogenous
    (function of x_t and the FIXED gen_params only)."""
    hid = jnp.tanh(gen_params["W1"] @ x_t + gen_params["b1"])
    scalars = gen_params["W_scalars"] @ hid + gen_params["b_scalars"]
    u_plus, u_minus, gamma_raw, delta_raw = scalars
    lam_plus = RHO * jnp.tanh(u_plus)
    lam_minus = RHO * jnp.tanh(u_minus)
    alpha_t = (lam_plus + lam_minus) / 2.0
    beta_t = (lam_plus - lam_minus) / 2.0
    gamma_t = 0.3 * jnp.tanh(gamma_raw)
    delta_t = 0.3 * jnp.tanh(delta_raw)
    c_raw = gen_params["W_c"] @ hid + gen_params["b_c"]
    c_t = 0.1 * jnp.tanh(c_raw)
    return alpha_t, beta_t, gamma_t, delta_t, c_t


def apply_K(x, v):
    return x - 2.0 * v * (v @ x)


ZERO_R = jnp.zeros((R_DIM, R_DIM), dtype=jnp.float64)


def h_step(h, theta, x_t, v, gen_params, eps=0.0, R_mat=None):
    """R_mat=None (or omitted) means no out-of-algebra perturbation
    (Parts A/B). eps is a plain Python float (kept static, never
    traced); R_mat defaults to an all-zero matrix so the extra term is
    always computed unconditionally (no traced-value branching)."""
    if R_mat is None:
        R_mat = ZERO_R
    alpha_t, beta_t, gamma_t, delta_t, c_t = gen_forward(x_t, gen_params)
    Kh = apply_K(h, v)
    Ktheta = apply_K(theta, v)
    h_next = alpha_t * h + beta_t * Kh + gamma_t * theta + delta_t * Ktheta + c_t
    h_next = h_next + eps * (R_mat @ h)
    return h_next


def rollout_h(h0, theta, xs, v, gen_params, eps=0.0, R_mat=None):
    def step(h, x_t):
        h_next = h_step(h, theta, x_t, v, gen_params, eps, R_mat)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs  # (T,64)


def ell(y, phase):
    return jnp.sin(y + phase) + 0.5 * y ** 2


def dell_dy(y, phase):
    return jnp.cos(y + phase) + y


def loss_bptt(theta, h0, xs, qs, phases, v, gen_params, eps, R_mat):
    Hs = rollout_h(h0, theta, xs, v, gen_params, eps, R_mat)
    ys = jnp.einsum("ti,ti->t", Hs, qs)
    return jnp.sum(ell(ys, phases))


grad_bptt_fn = jax.jit(jax.grad(loss_bptt, argnums=0), static_argnums=())


def full_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params, eps, R_mat):
    T = xs.shape[0]
    h = h0
    S = jnp.zeros((R_DIM, R_DIM), dtype=jnp.float64)
    g_total = jnp.zeros(R_DIM, dtype=jnp.float64)
    S_traj = []
    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda hh: h_step(hh, theta, x_t, v, gen_params, eps, R_mat))(h)
        G_t = jax.jacobian(lambda th: h_step(h, th, x_t, v, gen_params, eps, R_mat))(theta)
        S = J_t @ S + G_t
        S_traj.append(S)
        h_next = h_step(h, theta, x_t, v, gen_params, eps, R_mat)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total, jnp.stack(S_traj)


def lifted_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params):
    """Deliberately uses ONLY the eps=0 two-scalar closure, even when
    called on eps>0 trajectories elsewhere (falsification)."""
    T = xs.shape[0]
    h = h0
    a, b = 0.0, 0.0
    g_total = jnp.zeros(R_DIM, dtype=jnp.float64)
    ab_traj = []
    for t in range(T):
        x_t = xs[t]
        alpha_t, beta_t, gamma_t, delta_t, c_t = gen_forward(x_t, gen_params)
        a_next = alpha_t * a + beta_t * b + gamma_t
        b_next = beta_t * a + alpha_t * b + delta_t
        a, b = a_next, b_next
        ab_traj.append((float(a), float(b)))
        h_next = h_step(h, theta, x_t, v, gen_params, 0.0, None)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + a * dl_dh + b * apply_K(dl_dh, v)
        h = h_next
    return g_total, ab_traj


def reconstruct_S(ab_traj, v):
    K = np.eye(R_DIM) - 2.0 * np.outer(np.asarray(v), np.asarray(v))
    I = np.eye(R_DIM)
    return np.stack([a * I + b * K for (a, b) in ab_traj])


def make_setting(seed, T):
    rng = np.random.RandomState(seed)
    theta = jnp.array(rng.randn(R_DIM) * 0.3)
    h0 = jnp.array(rng.randn(R_DIM) * 0.2)
    xs = jnp.array(rng.randn(T, X_DIM) * 0.7)
    qs = jnp.array(rng.randn(T, R_DIM))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    return theta, h0, xs, qs, phases


def run_correctness_suite():
    print("=" * 78)
    print(f"B33b-min correctness suite (Part A): r=P={R_DIM}, actual trainable model")
    print("=" * 78)
    v = make_v(42)
    gen_params = make_gen_params(43)
    seeds = [0, 1, 2, 3, 4]
    lengths = [1, 5, 20, 100, 1000]
    eps_num = 1e-12
    worst = dict(full=0.0, lifted=0.0, S_recon=0.0, query=0.0)
    rank_fracs = []

    for T in lengths:
        for seed in seeds:
            theta, h0, xs, qs, phases = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, xs, qs, phases, v, gen_params, 0.0, None)
            g_f, S_traj = full_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params, 0.0, None)
            g_l, ab_traj = lifted_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + eps_num))
            rel_lifted = float(jnp.linalg.norm(g_l - g_b) / (jnp.linalg.norm(g_b) + eps_num))

            S_np = np.asarray(S_traj)
            S_hat = reconstruct_S(ab_traj, v)
            recon_err = float(np.max(np.abs(S_np - S_hat)))

            rng_q = np.random.RandomState(seed + 7000)
            n_q = min(T, 8)
            q_errs = []
            for _ in range(n_q):
                t_idx = rng_q.randint(0, T)
                q = rng_q.randn(R_DIM)
                a_t, b_t = ab_traj[t_idx]
                lhs = S_np[t_idx].T @ q
                rhs = a_t * q + b_t * np.asarray(apply_K(jnp.array(q), v))
                q_errs.append(float(np.max(np.abs(lhs - rhs))))

            worst["full"] = max(worst["full"], rel_full)
            worst["lifted"] = max(worst["lifted"], rel_lifted)
            worst["S_recon"] = max(worst["S_recon"], recon_err)
            worst["query"] = max(worst["query"], max(q_errs))

            rank_idx = list(range(min(T, 30))) if T <= 200 else list(range(0, T, max(1, T // 30)))
            ranks = []
            for t_idx in rank_idx:
                sv = np.linalg.svd(S_np[t_idx], compute_uv=False)
                ranks.append(int(np.sum(sv > 1e-9 * sv[0])) if sv[0] > 1e-12 else 0)
            frac_full = float(np.mean([r_ == R_DIM for r_ in ranks])) if ranks else float("nan")
            rank_fracs.append(frac_full)

            print(f"  T={T:5d} seed={seed}  full_rel={rel_full:.3e}  lifted_rel={rel_lifted:.3e}  "
                  f"S_recon max|d|={recon_err:.3e}  query max|d|={max(q_errs):.3e}  "
                  f"frac_rank64(sampled)={frac_full:.2f}")

    print("-" * 78)
    print(f"WORST: full_rel={worst['full']:.3e}  lifted_rel={worst['lifted']:.3e}  "
          f"S_recon={worst['S_recon']:.3e}  query={worst['query']:.3e}")
    print(f"Mean frac_rank64 across all settings: {np.mean(rank_fracs):.4f}")
    all_pass = all(v_ < 1e-8 for v_ in worst.values())
    print(f"ALL < 1e-8: {all_pass}")

    print()
    print("Storage accounting:")
    print(f"  Forward/model state: h_t={R_DIM} floats, theta={R_DIM} floats, "
          f"Householder v={R_DIM} floats")
    gen_flat, _ = ravel_pytree(gen_params)
    print(f"  Fixed (frozen, never trained) coefficient-generator params: {gen_flat.shape[0]} floats")
    print(f"  Additional PERSISTENT DYNAMIC CREDIT state:")
    print(f"    full RTRL: {R_DIM}*{R_DIM} = {R_DIM*R_DIM} dynamic sensitivity floats")
    print(f"    lifted rule: 2 dynamic eligibility floats (a_t,b_t)")
    print(f"    ratio: {R_DIM*R_DIM/2:.0f}x -- this IS the legitimate claim (4096 -> 2 in")
    print(f"    ADDITIONAL persistent dynamic credit storage for this model). K/v is part")
    print(f"    of the forward model needed by BOTH paths to run at all -- not charged")
    print(f"    uniquely to the reduced learner, and this is NOT a claim about total")
    print(f"    learner memory (h_t, theta, and the generator's own params are shared).")
    return dict(all_pass=all_pass, worst=worst, mean_rank_frac=float(np.mean(rank_fracs)),
                v=v, gen_params=gen_params)


# ---------------------------------------------------------------------
# Part B -- teacher/student system identification. Teacher and student
# share K/v, the FIXED coefficient generator, W, and the input sequence
# -- they differ ONLY in theta (the persistent trainable parameter).
# ---------------------------------------------------------------------
def make_W(seed, y_dim=8):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(y_dim, R_DIM) * (1.0 / np.sqrt(R_DIM)))


def make_sequence(seed, T):
    rng = np.random.RandomState(seed)
    h0 = jnp.array(rng.randn(R_DIM) * 0.2)
    xs = jnp.array(rng.randn(T, X_DIM) * 0.7)
    return h0, xs


def teacher_targets(h0, theta_star, xs, v, gen_params, W):
    Hs = rollout_h(h0, theta_star, xs, v, gen_params)
    return Hs @ W.T  # (T, y_dim)


def loss_mse(theta, h0, xs, targets, v, gen_params, W):
    Hs = rollout_h(h0, theta, xs, v, gen_params)
    Ys = Hs @ W.T
    return jnp.mean(0.5 * jnp.sum((Ys - targets) ** 2, axis=1))


grad_bptt_mse_fn = jax.jit(jax.grad(loss_mse, argnums=0))
loss_mse_fn = jax.jit(loss_mse)


def full_rtrl_grad_mse(theta, h0, xs, targets, v, gen_params, W):
    T = xs.shape[0]
    h = h0
    S = jnp.zeros((R_DIM, R_DIM), dtype=jnp.float64)
    g_total = jnp.zeros(R_DIM, dtype=jnp.float64)
    loss_total = 0.0
    S_traj = []
    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda hh: h_step(hh, theta, x_t, v, gen_params))(h)
        G_t = jax.jacobian(lambda th: h_step(h, th, x_t, v, gen_params))(theta)
        S = J_t @ S + G_t
        S_traj.append(S)
        h_next = h_step(h, theta, x_t, v, gen_params)
        y = W @ h_next
        diff = y - targets[t]
        loss_total = loss_total + 0.5 * jnp.sum(diff ** 2)
        dl_dh = W.T @ diff
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total / T, loss_total / T, S_traj


def lifted_rtrl_grad_mse(theta, h0, xs, targets, v, gen_params, W):
    T = xs.shape[0]
    h = h0
    a, b = 0.0, 0.0
    g_total = jnp.zeros(R_DIM, dtype=jnp.float64)
    loss_total = 0.0
    for t in range(T):
        x_t = xs[t]
        alpha_t, beta_t, gamma_t, delta_t, c_t = gen_forward(x_t, gen_params)
        a_next = alpha_t * a + beta_t * b + gamma_t
        b_next = beta_t * a + alpha_t * b + delta_t
        a, b = a_next, b_next
        h_next = h_step(h, theta, x_t, v, gen_params)
        y = W @ h_next
        diff = y - targets[t]
        loss_total = loss_total + 0.5 * jnp.sum(diff ** 2)
        dl_dh = W.T @ diff
        g_total = g_total + a * dl_dh + b * apply_K(dl_dh, v)
        h = h_next
    return g_total / T, loss_total / T


def sgd_step(theta, grad, lr):
    return theta - lr * grad


def run_training(v, gen_params, T=128, n_train=100, n_val=20, lr=3.0,
                  seed_teacher=777, seed_init=555, seed_readout=999,
                  seq_seed_offset=10_000, val_seed_offset=90_000):
    print()
    print("=" * 78)
    print("B33b-min Part B: teacher/student system identification")
    print("=" * 78)
    W = make_W(seed_readout)
    rng_star = np.random.RandomState(seed_teacher)
    theta_star = jnp.array(rng_star.randn(R_DIM) * 0.3)
    rng_init = np.random.RandomState(seed_init)
    theta_init = jnp.array(rng_init.randn(R_DIM) * 0.3)

    theta_full, theta_lifted, theta_bptt = theta_init, theta_init, theta_init
    train_losses = dict(full=[], lifted=[], bptt=[])
    grad_diag = []
    rank_samples = []

    t_start = time.time()
    for step in range(n_train):
        h0, xs = make_sequence(seq_seed_offset + step, T)
        targets = teacher_targets(h0, theta_star, xs, v, gen_params, W)

        g_full, loss_full, S_traj = full_rtrl_grad_mse(theta_full, h0, xs, targets, v, gen_params, W)
        g_lifted, loss_lifted = lifted_rtrl_grad_mse(theta_lifted, h0, xs, targets, v, gen_params, W)
        g_bptt = grad_bptt_mse_fn(theta_bptt, h0, xs, targets, v, gen_params, W)
        loss_bptt_val = float(loss_mse_fn(theta_bptt, h0, xs, targets, v, gen_params, W))

        train_losses["full"].append(float(loss_full))
        train_losses["lifted"].append(float(loss_lifted))
        train_losses["bptt"].append(loss_bptt_val)

        if step < 5:
            grad_diag.append(dict(
                step=step,
                diff_full_lifted=float(jnp.linalg.norm(g_full - g_lifted)),
                diff_lifted_bptt=float(jnp.linalg.norm(g_lifted - g_bptt)),
                diff_full_bptt=float(jnp.linalg.norm(g_full - g_bptt)),
            ))

        if step % 20 == 0:
            sv0 = np.linalg.svd(np.asarray(S_traj[-1]), compute_uv=False)
            rank_last = int(np.sum(sv0 > 1e-9 * sv0[0])) if sv0[0] > 1e-12 else 0
            rank_samples.append(rank_last)

        theta_full = sgd_step(theta_full, g_full, lr)
        theta_lifted = sgd_step(theta_lifted, g_lifted, lr)
        theta_bptt = sgd_step(theta_bptt, g_bptt, lr)

        if step < 5:
            grad_diag[-1]["param_diff_full_lifted"] = float(jnp.linalg.norm(theta_full - theta_lifted))
            grad_diag[-1]["param_diff_lifted_bptt"] = float(jnp.linalg.norm(theta_lifted - theta_bptt))

        if step % 10 == 0 or step == n_train - 1:
            print(f"  step={step:4d}  loss_full={loss_full:.6e}  loss_lifted={loss_lifted:.6e}  "
                  f"loss_bptt={loss_bptt_val:.6e}", flush=True)
    elapsed = time.time() - t_start

    val_losses = dict(full=[], lifted=[], bptt=[])
    for i in range(n_val):
        h0, xs = make_sequence(val_seed_offset + i, T)
        targets = teacher_targets(h0, theta_star, xs, v, gen_params, W)
        val_losses["full"].append(float(loss_mse_fn(theta_full, h0, xs, targets, v, gen_params, W)))
        val_losses["lifted"].append(float(loss_mse_fn(theta_lifted, h0, xs, targets, v, gen_params, W)))
        val_losses["bptt"].append(float(loss_mse_fn(theta_bptt, h0, xs, targets, v, gen_params, W)))

    dist_to_teacher = dict(
        full=float(jnp.linalg.norm(theta_full - theta_star)),
        lifted=float(jnp.linalg.norm(theta_lifted - theta_star)),
        bptt=float(jnp.linalg.norm(theta_bptt - theta_star)),
        init=float(jnp.linalg.norm(theta_init - theta_star)),
    )

    print("-" * 78)
    print("First-updates gradient/parameter agreement:")
    for row in grad_diag:
        print(f"  step={row['step']}  ||g_full-g_lifted||={row['diff_full_lifted']:.3e}  "
              f"||g_lifted-g_bptt||={row['diff_lifted_bptt']:.3e}  "
              f"||theta_full-theta_lifted||={row['param_diff_full_lifted']:.3e}  "
              f"||theta_lifted-theta_bptt||={row['param_diff_lifted_bptt']:.3e}")
    print("-" * 78)
    tl = train_losses
    print(f"Initial train loss: {tl['full'][0]:.6e}")
    print(f"Final train loss: full={tl['full'][-1]:.6e}  lifted={tl['lifted'][-1]:.6e}  bptt={tl['bptt'][-1]:.6e}")
    vl = val_losses
    print(f"Mean validation loss ({n_val} held-out sequences): "
          f"full={np.mean(vl['full']):.6e}  lifted={np.mean(vl['lifted']):.6e}  bptt={np.mean(vl['bptt']):.6e}")
    print(f"Distance to teacher theta*: init={dist_to_teacher['init']:.4f}  "
          f"full={dist_to_teacher['full']:.4f}  lifted={dist_to_teacher['lifted']:.4f}  "
          f"bptt={dist_to_teacher['bptt']:.4f}")
    print(f"Sampled rank(S_t) during training (every 20 steps, last step of each sequence): "
          f"{rank_samples}  (all should be {R_DIM})")
    print(f"T={T}  n_train={n_train}  elapsed={elapsed:.1f}s")
    print(f"Additional persistent dynamic credit storage: full={R_DIM*R_DIM} floats, "
          f"lifted=2 floats, ratio={R_DIM*R_DIM/2:.0f}x")

    return dict(train_losses=train_losses, val_losses=val_losses, grad_diag=grad_diag,
                dist_to_teacher=dist_to_teacher, rank_samples=rank_samples, T=T,
                n_train=n_train, n_val=n_val, lr=lr, elapsed=elapsed)


# ---------------------------------------------------------------------
# Part C -- closure falsification. Full RTRL gets the true (eps-
# perturbed) dynamics; the lifted rule deliberately keeps using the OLD
# eps=0 two-scalar closure (same convention as B29/B32a/B33a).
# ---------------------------------------------------------------------
def run_falsification(v, gen_params):
    print()
    print("=" * 78)
    print("B33b-min Part C: closure falsification (moderate eps only, avoid B33a-style blowup)")
    print("=" * 78)
    eps_list = [0.0, 1e-6, 1e-4, 1e-3, 1e-2]
    lengths = [5, 20, 100, 500]
    seed = 0

    rng_R = np.random.RandomState(31415)
    M = rng_R.randn(R_DIM, R_DIM)
    R_generic = jnp.array((M + M.T) / 2.0)

    I = np.eye(R_DIM)
    K_np = np.eye(R_DIM) - 2.0 * np.outer(np.asarray(v), np.asarray(v))
    basis = [I.ravel() / np.linalg.norm(I.ravel()), K_np.ravel() / np.linalg.norm(K_np.ravel())]
    basis, _ = np.linalg.qr(np.stack(basis, axis=1))
    r_flat = np.asarray(R_generic).ravel()
    proj = basis @ (basis.T @ r_flat)
    frac_outside = np.linalg.norm(r_flat - proj) / np.linalg.norm(r_flat)
    print(f"  R_generic fraction outside span{{I,K}}: {frac_outside:.4f}")

    for eps in eps_list:
        for T in lengths:
            theta, h0, xs, qs, phases = make_setting(seed, T)
            g_b = grad_bptt_fn(theta, h0, xs, qs, phases, v, gen_params, eps, R_generic)
            g_f, S_traj = full_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params, eps, R_generic)
            g_l, ab_traj = lifted_rtrl_grad(theta, h0, xs, qs, phases, v, gen_params)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            rel_lifted = float(jnp.linalg.norm(g_l - g_b) / (jnp.linalg.norm(g_b) + 1e-12))

            S_np = np.asarray(S_traj)
            S_hat = reconstruct_S(ab_traj, v)
            recon_err = float(np.max(np.abs(S_np - S_hat)))
            max_abs_val = float(np.max(np.abs(S_np)))

            flat = S_np.reshape(T, -1)
            sv = np.linalg.svd(flat, compute_uv=False)
            span_dim = int(np.sum(sv > 1e-9 * sv[0])) if sv[0] > 1e-12 else 0

            print(f"  eps={eps:.0e}  T={T:4d}  full_rel={rel_full:.3e}  lifted_rel={rel_lifted:.3e}  "
                  f"recon_max|d|={recon_err:.3e}  span_dim={span_dim:3d}/{min(T,R_DIM*R_DIM)}  "
                  f"max|S_t|={max_abs_val:.3e}")


def main():
    corr = run_correctness_suite()
    train_result = run_training(corr["v"], corr["gen_params"])
    run_falsification(corr["v"], corr["gen_params"])
    print()
    print("=" * 78)
    print(f"B33b-min Part A CORRECTNESS PASS (<1e-8 everywhere): {corr['all_pass']}")
    print("=" * 78)
    with open("/tmp/b33b_min_result.json", "w") as f:
        json.dump(dict(part_a=dict(all_pass=corr["all_pass"], worst=corr["worst"],
                                    mean_rank_frac=corr["mean_rank_frac"]),
                       part_b=dict(train_losses=train_result["train_losses"],
                                   val_losses=train_result["val_losses"],
                                   grad_diag=train_result["grad_diag"],
                                   dist_to_teacher=train_result["dist_to_teacher"],
                                   rank_samples=train_result["rank_samples"],
                                   T=train_result["T"], n_train=train_result["n_train"],
                                   elapsed=train_result["elapsed"])), f, indent=2)


if __name__ == "__main__":
    main()
