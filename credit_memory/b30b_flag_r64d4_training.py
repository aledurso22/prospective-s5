"""Phase B30b -- supervised learning equivalence on the r=64, d=4 flag
SSM. Shows the 16x-smaller exact reduced-RTRL credit state is not only
gradient-correct (B30a) but can actually train the same nonlinear
recurrent model, matching full-RTRL and BPTT training trajectories.

Reuses the EXACT B30a architecture/consts (`credit_memory.
b30a_flag_r64d4_test`, frozen, unmodified) -- structural matrices
(R_U,D_U,R_V,K,B_V,C_V,C_U) fixed and SHARED between teacher and
student; only the dominant compressed family theta (Phi's MLP,
P_c=10,088) is trained. Teacher/student system-identification setup:
teacher draws a frozen Phi_theta*, generates y_t=W.s_t along random
input sequences; student (same architecture, different theta init) is
trained to match y_t via multi-time MSE.

Three independent training paths (A: full RTRL, B: reduced RTRL,
C: BPTT), updated ONLY at sequence boundaries (one full-sequence
gradient per optimizer step), same plain-SGD optimizer/LR/init/data
order across all three, so gradients and optimization trajectories are
directly comparable -- NOT a per-step-online-update vs sequence-end
comparison.

Run: python -m credit_memory.b30b_flag_r64d4_training
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from credit_memory.b30a_flag_r64d4_test import (
    make_consts, make_theta, u_step, v_step, full_step_state,
    rollout_full_states, D_U_DIM, D_V_DIM, R_DIM,
)


def ell_mse(y, target):
    return (y - target) ** 2


def dell_dy_mse(y, target):
    return 2.0 * (y - target)


def teacher_rollout(u0, v0, xs, theta_star, c, W):
    Ss = rollout_full_states(u0, v0, xs, theta_star, c)  # (T,64)
    ys = Ss @ W
    return ys


def loss_bptt_mse(theta, u0, v0, xs, targets, c, W):
    Ss = rollout_full_states(u0, v0, xs, theta, c)
    ys = Ss @ W
    return jnp.mean(ell_mse(ys, targets))


grad_bptt_mse_fn = jax.jit(jax.grad(loss_bptt_mse, argnums=0))
loss_bptt_mse_fn = jax.jit(loss_bptt_mse)


def full_rtrl_grad_mse(theta, u0, v0, xs, targets, c, unravel, theta_flat, P_c, W):
    T = xs.shape[0]
    s = jnp.concatenate([u0, v0])
    S = jnp.zeros((R_DIM, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    loss_total = 0.0
    for t in range(T):
        x = xs[t]
        J_t = jax.jacobian(lambda ss: full_step_state(ss, x, theta, c))(s)
        G_t = jax.jacobian(lambda th_flat: full_step_state(s, x, unravel(th_flat), c))(theta_flat)
        S = J_t @ S + G_t
        s_next = full_step_state(s, x, theta, c)
        y = s_next @ W
        loss_total = loss_total + ell_mse(y, targets[t])
        dl_ds = dell_dy_mse(y, targets[t]) * W
        g_total = g_total + dl_ds @ S
        s = s_next
    return g_total / T, loss_total / T


def reduced_rtrl_grad_mse(theta, u0, v0, xs, targets, c, unravel, theta_flat, P_c, W):
    T = xs.shape[0]
    u_traj = [u0]
    u = u0
    for t in range(T):
        u = u_step(u, xs[t], c)
        u_traj.append(u)
    v = v0
    E = jnp.zeros((D_V_DIM, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    loss_total = 0.0
    W_v = W[D_U_DIM:]
    for t in range(T):
        x = xs[t]
        u_val = u_traj[t]
        J_v = jax.jacobian(lambda vv: v_step(vv, u_val, x, theta, c))(v)
        G_v = jax.jacobian(lambda th_flat: v_step(v, u_val, x, unravel(th_flat), c))(theta_flat)
        E = J_v @ E + G_v
        v_next = v_step(v, u_val, x, theta, c)
        u_next = u_traj[t + 1]
        s_next = jnp.concatenate([u_next, v_next])
        y = s_next @ W
        loss_total = loss_total + ell_mse(y, targets[t])
        dl_dv = dell_dy_mse(y, targets[t]) * W_v
        g_total = g_total + dl_dv @ E
        v = v_next
    return g_total / T, loss_total / T


def make_readout(seed):
    rng = np.random.RandomState(seed)
    W_u = jnp.array(rng.randn(D_U_DIM) * (0.3 / np.sqrt(D_U_DIM)))
    W_v = jnp.array(rng.randn(D_V_DIM) * 1.0)
    return jnp.concatenate([W_u, W_v])


def make_sequence(seed, T):
    rng = np.random.RandomState(seed)
    u0 = jnp.array(rng.randn(D_U_DIM) * 0.2)
    v0 = jnp.array(rng.randn(D_V_DIM) * 0.2)
    xs = jnp.array(rng.randn(T) * 0.5)
    return u0, v0, xs


def sgd_step(theta, grad, lr):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, theta, grad)


def run_training(T=128, n_train=100, n_val=20, lr=0.05,
                 seed_teacher=777, seed_init=555, seed_readout=999,
                 seq_seed_offset=10_000, val_seed_offset=90_000):
    c = make_consts()  # default seed=12345, SAME as B30a
    theta_star = make_theta(seed_teacher)
    theta_star_flat, unravel_star = ravel_pytree(theta_star)
    P_c = theta_star_flat.shape[0]
    W = make_readout(seed_readout)

    theta_init = make_theta(seed_init)
    theta_full = theta_init
    theta_reduced = theta_init
    theta_bptt = theta_init

    train_losses = dict(full=[], reduced=[], bptt=[])
    grad_diag = []

    t_start = time.time()
    for step in range(n_train):
        seq_seed = seq_seed_offset + step
        u0, v0, xs = make_sequence(seq_seed, T)
        ys_star = teacher_rollout(u0, v0, xs, theta_star, c, W)

        theta_flat_full, unravel_f = ravel_pytree(theta_full)
        g_full, loss_full = full_rtrl_grad_mse(theta_full, u0, v0, xs, ys_star, c, unravel_f, theta_flat_full, P_c, W)

        theta_flat_red, unravel_r = ravel_pytree(theta_reduced)
        g_reduced, loss_reduced = reduced_rtrl_grad_mse(theta_reduced, u0, v0, xs, ys_star, c, unravel_r, theta_flat_red, P_c, W)

        g_bptt = grad_bptt_mse_fn(theta_bptt, u0, v0, xs, ys_star, c, W)
        loss_bptt_val = float(loss_bptt_mse_fn(theta_bptt, u0, v0, xs, ys_star, c, W))
        g_bptt_flat, _ = ravel_pytree(g_bptt)

        train_losses["full"].append(float(loss_full))
        train_losses["reduced"].append(float(loss_reduced))
        train_losses["bptt"].append(loss_bptt_val)

        if step < 5:
            diff_fr = float(jnp.linalg.norm(g_full - g_reduced))
            diff_rb = float(jnp.linalg.norm(g_reduced - g_bptt_flat))
            diff_fb = float(jnp.linalg.norm(g_full - g_bptt_flat))
            grad_diag.append(dict(step=step, diff_full_reduced=diff_fr, diff_reduced_bptt=diff_rb,
                                   diff_full_bptt=diff_fb))

        theta_full = sgd_step(theta_full, unravel_f(g_full), lr)
        theta_reduced = sgd_step(theta_reduced, unravel_r(g_reduced), lr)
        theta_bptt = sgd_step(theta_bptt, g_bptt, lr)

        if step < 5:
            pf, _ = ravel_pytree(theta_full)
            pr, _ = ravel_pytree(theta_reduced)
            pb, _ = ravel_pytree(theta_bptt)
            grad_diag[-1]["param_diff_full_reduced"] = float(jnp.linalg.norm(pf - pr))
            grad_diag[-1]["param_diff_reduced_bptt"] = float(jnp.linalg.norm(pr - pb))
            grad_diag[-1]["param_diff_full_bptt"] = float(jnp.linalg.norm(pf - pb))

        if step % 10 == 0 or step == n_train - 1:
            print(f"  step={step:4d}  loss_full={loss_full:.6e}  loss_reduced={loss_reduced:.6e}  "
                  f"loss_bptt={loss_bptt_val:.6e}", flush=True)
    elapsed = time.time() - t_start

    val_losses = dict(full=[], reduced=[], bptt=[])
    for i in range(n_val):
        u0, v0, xs = make_sequence(val_seed_offset + i, T)
        ys_star = teacher_rollout(u0, v0, xs, theta_star, c, W)
        val_losses["full"].append(float(loss_bptt_mse_fn(theta_full, u0, v0, xs, ys_star, c, W)))
        val_losses["reduced"].append(float(loss_bptt_mse_fn(theta_reduced, u0, v0, xs, ys_star, c, W)))
        val_losses["bptt"].append(float(loss_bptt_mse_fn(theta_bptt, u0, v0, xs, ys_star, c, W)))

    return dict(train_losses=train_losses, val_losses=val_losses, grad_diag=grad_diag,
                P_c=int(P_c), T=T, n_train=n_train, n_val=n_val, lr=lr, elapsed=elapsed)


def main():
    print("=" * 78)
    print("B30b supervised training equivalence, r=64 d=4 flag SSM")
    print("=" * 78)
    result = run_training()
    print("-" * 78)
    print("First-updates gradient/parameter agreement:")
    for row in result["grad_diag"]:
        print(f"  step={row['step']}  ||g_full-g_reduced||={row['diff_full_reduced']:.3e}  "
              f"||g_reduced-g_bptt||={row['diff_reduced_bptt']:.3e}  "
              f"||g_full-g_bptt||={row['diff_full_bptt']:.3e}  "
              f"||theta_full-theta_reduced||={row['param_diff_full_reduced']:.3e}  "
              f"||theta_reduced-theta_bptt||={row['param_diff_reduced_bptt']:.3e}")
    print("-" * 78)
    tl = result["train_losses"]
    print(f"Final train loss: full={tl['full'][-1]:.6e}  reduced={tl['reduced'][-1]:.6e}  bptt={tl['bptt'][-1]:.6e}")
    vl = result["val_losses"]
    print(f"Mean validation loss ({result['n_val']} held-out sequences): "
          f"full={np.mean(vl['full']):.6e}  reduced={np.mean(vl['reduced']):.6e}  bptt={np.mean(vl['bptt']):.6e}")
    print(f"P_c={result['P_c']}  T={result['T']}  n_train={result['n_train']}  elapsed={result['elapsed']:.1f}s")
    bytes_full = R_DIM * result["P_c"] * 8
    bytes_reduced = D_V_DIM * result["P_c"] * 8
    print(f"Persistent credit: full={R_DIM}*{result['P_c']}={R_DIM*result['P_c']} floats ({bytes_full/1e6:.3f} MB), "
          f"reduced={D_V_DIM}*{result['P_c']}={D_V_DIM*result['P_c']} floats ({bytes_reduced/1e6:.3f} MB), "
          f"ratio={(R_DIM*result['P_c'])/(D_V_DIM*result['P_c']):.2f}x")
    with open("/tmp/b30b_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
