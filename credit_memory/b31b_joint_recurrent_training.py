"""Phase B31b -- actual training: teacher and student differ in the
RECURRENT dynamics itself (R_V, K, B_V, C_V, C_U), not just Phi. Shows
the student must learn genuine recurrent dynamics, not merely fit a
nonlinear readout on a matched temporal backbone, while exact reduced
credit (16x smaller) still trains identically to full RTRL/BPTT.

Reuses B31a's architecture/consts unmodified via import. Only R_U, D_U
stay fixed; teacher and student each draw their OWN independent
R_V,K,B_V,C_V,C_U,Phi (P_c=10,888).

Run: python -m credit_memory.b31b_joint_recurrent_training
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from credit_memory.b25_nonlinear_credit import krylov_subspace
from credit_memory.b31a_joint_family_correctness import (
    make_fixed_consts, make_theta, u_step, v_step, full_step_state,
    rollout_full_states, TRAINABLE_KEYS, D_U_DIM, D_V_DIM, R_DIM, P_MAT,
)


def ell_mse(y, target):
    return (y - target) ** 2


def dell_dy_mse(y, target):
    return 2.0 * (y - target)


def teacher_rollout(u0, v0, xs, theta_star, c, W):
    Ss = rollout_full_states(u0, v0, xs, theta_star, c)
    return Ss @ W


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
        J_v = jax.jacobian(lambda vv: v_step(vv, u_val, x, theta))(v)
        G_v = jax.jacobian(lambda th_flat: v_step(v, u_val, x, unravel(th_flat)))(theta_flat)
        E = J_v @ E + G_v
        v_next = v_step(v, u_val, x, theta)
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


RHO_MAX = 0.95


def project_stable_R_V(theta, rho_max=RHO_MAX):
    """R_V is now TRAINABLE (unlike B30) with no structural spectral-
    radius guarantee -- the same recurring instability diagnosed and
    fixed in B28's "ours" architecture and B18 (see memory
    b18-init-instability.md: new block-recurrent architectures need
    explicit stability projection). Confirmed empirically here: raw SGD
    on the joint (R_V,K,B_V,C_V,C_U,Phi) family diverged to NaN within
    ~2 sequence-boundary steps at T=128, lr=0.003. Rescales R_V toward
    rho_max whenever its spectral radius exceeds it; a no-op otherwise.
    Applied to R_V ONLY -- K,B_V,C_V,C_U,Phi are not dynamical-stability
    generators themselves and are left untouched."""
    R_V = theta["R_V"]
    eigval_mag = jnp.max(jnp.abs(jnp.linalg.eigvals(R_V)))
    scale = jnp.where(eigval_mag > rho_max, rho_max / eigval_mag, 1.0)
    new_theta = dict(theta)
    new_theta["R_V"] = R_V * scale
    return new_theta


# ---------------------------------------------------------------------
# Structural diagnostics, usable both before (theta_init) and after
# (theta_final) training.
# ---------------------------------------------------------------------
def linear_skeleton(theta, c):
    R_U_np, D_U_np = np.asarray(c["R_U"]), np.asarray(c["D_U"])
    R_V_np, K_np = np.asarray(theta["R_V"]), np.asarray(theta["K"])
    A_lin = np.zeros((R_DIM, R_DIM))
    A_lin[:D_U_DIM, :D_U_DIM] = R_U_np
    A_lin[D_U_DIM:, D_U_DIM:] = R_V_np
    A_lin[D_U_DIM:, :D_U_DIM] = K_np
    B_full = np.zeros(R_DIM)
    B_full[:D_U_DIM] = D_U_np
    return A_lin, B_full


def structural_report(theta, c, sample_states, label):
    A_lin, B_full = linear_skeleton(theta, c)
    reach_rank = krylov_subspace(A_lin, B_full[:, None]).shape[1]

    P_np = np.asarray(P_MAT)
    proj_perp = np.eye(R_DIM) - P_np @ P_np.T
    max_leak = 0.0
    for (u_s, v_s, x_s) in sample_states:
        s_s = jnp.concatenate([u_s, v_s])
        J_t = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta, c))(s_s))
        leak = np.linalg.norm(proj_perp @ J_t @ P_np)
        max_leak = max(max_leak, leak)

    u_rep, v_rep, x_rep = sample_states[0]
    s_rep = jnp.concatenate([u_rep, v_rep])
    J_rep = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_rep, theta, c))(s_rep))
    Q = J_rep - A_lin
    comm_norm = np.linalg.norm(A_lin @ Q - Q @ A_lin)

    print(f"  [{label}] reach_rank={reach_rank}/{R_DIM}  max_invariant_leak={max_leak:.3e}  "
          f"commutator_norm={comm_norm:.6e}")
    return dict(reach_rank=int(reach_rank), max_leak=float(max_leak), commutator_norm=float(comm_norm))


def jacobian_change(theta_init, theta_final, c, sample_states):
    diffs, init_norms = [], []
    for (u_s, v_s, x_s) in sample_states:
        s_s = jnp.concatenate([u_s, v_s])
        J_init = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta_init, c))(s_s))
        J_final = np.asarray(jax.jacobian(lambda ss: full_step_state(ss, x_s, theta_final, c))(s_s))
        diffs.append(np.linalg.norm(J_final - J_init))
        init_norms.append(np.linalg.norm(J_init))
    mean_abs = float(np.mean(diffs))
    mean_rel = float(np.mean(diffs) / (np.mean(init_norms) + 1e-12))
    return dict(mean_abs_jacobian_change=mean_abs, mean_rel_jacobian_change=mean_rel)


def family_norm_report(theta, label):
    out = {}
    for fam in ("R_V", "K", "B_V", "C_V", "C_U"):
        out[fam] = float(jnp.linalg.norm(theta[fam]))
    print(f"  [{label}] family norms: " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def run_training(T=128, n_train=100, n_val=20, lr=0.001,
                 seed_teacher=777, seed_init=555, seed_readout=999,
                 seq_seed_offset=10_000, val_seed_offset=90_000):
    c = make_fixed_consts()
    theta_star = make_theta(seed_teacher)
    theta_star_flat, _ = ravel_pytree(theta_star)
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
        u0, v0, xs = make_sequence(seq_seed_offset + step, T)
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

        theta_full = project_stable_R_V(sgd_step(theta_full, unravel_f(g_full), lr))
        theta_reduced = project_stable_R_V(sgd_step(theta_reduced, unravel_r(g_reduced), lr))
        theta_bptt = project_stable_R_V(sgd_step(theta_bptt, g_bptt, lr))

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

    # distance to teacher, per family, for all three final students
    dist_to_teacher = {}
    for name, theta_x in (("full", theta_full), ("reduced", theta_reduced), ("bptt", theta_bptt)):
        d = {}
        for fam in ("R_V", "K", "B_V", "C_V", "C_U"):
            d[fam] = float(jnp.linalg.norm(theta_x[fam] - theta_star[fam]))
        dist_to_teacher[name] = d

    # sampled states for structural / Jacobian-change diagnostics
    rng = np.random.RandomState(2024)
    sample_states = []
    for _ in range(12):
        u_s = jnp.array(rng.randn(D_U_DIM) * 0.3)
        v_s = jnp.array(rng.randn(D_V_DIM) * 0.3)
        x_s = float(rng.randn() * 0.5)
        sample_states.append((u_s, v_s, x_s))

    print()
    print("Structural diagnostics BEFORE training (theta_init):")
    struct_before = structural_report(theta_init, c, sample_states, "before")
    family_norm_report(theta_init, "before")

    print("Structural diagnostics AFTER training:")
    struct_after = {}
    for name, theta_x in (("full", theta_full), ("reduced", theta_reduced), ("bptt", theta_bptt)):
        struct_after[name] = structural_report(theta_x, c, sample_states, f"after-{name}")
        family_norm_report(theta_x, f"after-{name}")

    jac_change = {}
    for name, theta_x in (("full", theta_full), ("reduced", theta_reduced), ("bptt", theta_bptt)):
        jac_change[name] = jacobian_change(theta_init, theta_x, c, sample_states)
        print(f"  [{name}] Jacobian change from init: mean_abs={jac_change[name]['mean_abs_jacobian_change']:.4e}  "
              f"mean_rel={jac_change[name]['mean_rel_jacobian_change']:.4e}")

    return dict(train_losses=train_losses, val_losses=val_losses, grad_diag=grad_diag,
                P_c=int(P_c), T=T, n_train=n_train, n_val=n_val, lr=lr, elapsed=elapsed,
                dist_to_teacher=dist_to_teacher, struct_before=struct_before, struct_after=struct_after,
                jac_change=jac_change)


def main():
    print("=" * 78)
    print("B31b joint recurrent-dynamics training equivalence, r=64 d=4 flag SSM")
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
    print("Distance to teacher params, per family:")
    for name, d in result["dist_to_teacher"].items():
        print(f"  [{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))
    print(f"P_c={result['P_c']}  T={result['T']}  n_train={result['n_train']}  elapsed={result['elapsed']:.1f}s")
    bytes_full = R_DIM * result["P_c"] * 8
    bytes_reduced = D_V_DIM * result["P_c"] * 8
    print(f"Persistent credit: full={R_DIM}*{result['P_c']}={R_DIM*result['P_c']} floats ({bytes_full/1e6:.3f} MB), "
          f"reduced={D_V_DIM}*{result['P_c']}={D_V_DIM*result['P_c']} floats ({bytes_reduced/1e6:.3f} MB), "
          f"ratio={(R_DIM*result['P_c'])/(D_V_DIM*result['P_c']):.2f}x")
    with open("/tmp/b31b_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
