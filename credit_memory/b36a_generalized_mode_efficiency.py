"""B36a -- deconfounded generalized-mode representation efficiency.
Pure BPTT / fixed-shared-parameter training (no per-sample continual
updates, no Hessian transport, no K-sweeps, no moving-weight
diagnostics -- leaving the B35 continual-learning branch entirely).

Three d=2, two-real-state architectures, matched exactly on real
recurrent state dimension, trainable recurrent parameter count, and
readout/input parameterization (r=2n for all three; n=number of
factors):
  RealDiagonal: h_{i,t+1} = lambda_i*h_{i,t} + b_i*u_t, r independent
    scalar modes, no coupling between coordinates.
  ComplexLocal: u=a+ib, M_C=[[a,-b],[b,a]] (semisimple).
  DualLocal:    u=a+b*eps, eps^2=0, M_D=[[a,0],[b,a]] (nonsemisimple).

Exact persistent eligibility count under online RTRL is IDENTICAL for
all three at matched n: RealDiagonal needs one scalar trace per
independent coordinate (r=2n total, verified below); Complex/Dual need
2 real coordinates per factor (2n total, established in B35a-j). No
discrepancy to report -- matched by construction.

Run: python -m credit_memory.b36a_generalized_mode_efficiency
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, DIVERGENCE_LOSS_CEIL

H = 64
N_TRAIN = 150
LR_GRID = (0.01, 0.03, 0.1)
TUNING_SEEDS = (900, 901)
EVAL_SEEDS = tuple(range(1, 11))   # 10 fresh evaluation seeds
N_FACTORS_GRID = (1, 2, 4, 8)
U_SCALE = 0.5
CLIP_NORM = 5.0

# Teacher constants (reused from B35b1/B35j for continuity).
LAM_T, MU_T = 0.85, 0.30
B_T_GEN = jnp.array([1.0, 0.7])
C_T_GEN = jnp.array([0.6, 1.0])
J_T = jnp.array([[LAM_T, MU_T], [0.0, LAM_T]])

RHO_T, OMEGA_T = 0.80, 0.6
B_T_OSC = jnp.array([1.0, 0.7])
C_T_OSC = jnp.array([0.6, 1.0])
R_T = RHO_T * jnp.array([[jnp.cos(OMEGA_T), -jnp.sin(OMEGA_T)], [jnp.sin(OMEGA_T), jnp.cos(OMEGA_T)]])


def teacher_rollout(A, b, c, h0, us):
    def step(h, u):
        h_next = A @ h + b * u
        return h_next, c @ h_next
    _, ys = jax.lax.scan(step, h0, us)
    return ys


TEACHERS = dict(
    generalized_mode=dict(A=J_T, b=B_T_GEN, c=C_T_GEN),
    oscillatory=dict(A=R_T, b=B_T_OSC, c=C_T_OSC),
)


def verify_generalized_mode(T=40):
    us = jnp.zeros(T).at[0].set(1.0)
    ys = np.asarray(teacher_rollout(J_T, B_T_GEN, C_T_GEN, jnp.zeros(2), us))
    t_idx = np.arange(T)
    basis = np.stack([LAM_T ** t_idx, t_idx * LAM_T ** t_idx], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, ys, rcond=None)
    fit = basis @ coeffs
    return float(coeffs[1]), float(np.max(np.abs(fit - ys)))


# =======================================================================
# Architecture step functions -- all take (h, theta, u) -> h_next, all
# theta of length r=2n.
# =======================================================================
def real_diagonal_step(h, theta, u, n):
    r = 2 * n
    lam = theta[:r]
    b_in = theta[r:2 * r]
    return lam * h + b_in * u


def complex_step(h, theta, u, n):
    r = 2 * n
    ab = theta[:r].reshape(n, 2)
    b_in = theta[r:2 * r]
    H_ = h.reshape(n, 2)
    a, b = ab[:, 0], ab[:, 1]
    h0, h1 = H_[:, 0], H_[:, 1]
    out0 = a * h0 - b * h1
    out1 = b * h0 + a * h1
    return jnp.stack([out0, out1], axis=1).reshape(r) + b_in * u


def dual_step(h, theta, u, n):
    r = 2 * n
    ab = theta[:r].reshape(n, 2)
    b_in = theta[r:2 * r]
    H_ = h.reshape(n, 2)
    a, b = ab[:, 0], ab[:, 1]
    h0, h1 = H_[:, 0], H_[:, 1]
    out0 = a * h0
    out1 = b * h0 + a * h1
    return jnp.stack([out0, out1], axis=1).reshape(r) + b_in * u


STEP_FNS = dict(RealDiagonal=real_diagonal_step, ComplexLocal=complex_step, DualLocal=dual_step)


def make_params(arch, n, seed, rho_max=0.95):
    r = 2 * n
    rng = np.random.RandomState(seed)
    if arch == "RealDiagonal":
        lam = rng.uniform(-rho_max, rho_max, size=r)
    else:
        base = rng.uniform(-rho_max, rho_max, size=n)
        tail = rng.randn(n) * 0.3
        tail = np.clip(tail, -rho_max, rho_max)
        lam = np.stack([base, tail], axis=1).reshape(r)
    b_in = rng.randn(r) / np.sqrt(r)
    c_out = rng.randn(r) / np.sqrt(r)
    return jnp.array(np.concatenate([lam, b_in, c_out]))


def unpack(theta, r):
    return theta[:r], theta[r:2 * r], theta[2 * r:3 * r]


def rollout_student(arch, n, theta, us):
    r = 2 * n
    rec_theta, b_in, c_out = unpack(theta, r)
    step_fn = STEP_FNS[arch]
    full_theta = jnp.concatenate([rec_theta, b_in])

    def step(h, u):
        h_next = step_fn(h, full_theta, u, n)
        return h_next, c_out @ h_next
    h0 = jnp.zeros(r, dtype=jnp.float64)
    _, ys = jax.lax.scan(step, h0, us)
    return ys


# =======================================================================
# RTRL vs BPTT verification (fixed theta, no online updates).
# =======================================================================
def verify_rtrl_vs_bptt(arch, n, seed=0, T=20):
    r = 2 * n
    theta = make_params(arch, n, seed)
    rec_theta, b_in, c_out = unpack(theta, r)
    step_fn = STEP_FNS[arch]
    full_theta = jnp.concatenate([rec_theta, b_in])
    rng = np.random.RandomState(seed + 1)
    us = jnp.array(rng.randn(T) * U_SCALE)
    qs = jnp.array(rng.randn(T, r) * 0.5)

    def loss_bptt(full_theta):
        def step(h, inputs):
            u, q = inputs
            h_next = step_fn(h, full_theta, u, n)
            return h_next, jnp.dot(q, h_next)
        h0 = jnp.zeros(r, dtype=jnp.float64)
        _, ys = jax.lax.scan(step, h0, (us, qs))
        return jnp.sum(jnp.sin(ys) + 0.5 * ys ** 2)

    g_bptt = jax.grad(loss_bptt)(full_theta)

    # forward RTRL: exact full (r, 2r) sensitivity via autodiff Jacobians, propagated online.
    h = jnp.zeros(r, dtype=jnp.float64)
    S = jnp.zeros((r, 2 * r), dtype=jnp.float64)
    g_rtrl = jnp.zeros(2 * r, dtype=jnp.float64)
    for t in range(T):
        u_t = us[t]
        J_t = jax.jacobian(lambda hh: step_fn(hh, full_theta, u_t, n))(h)
        G_t = jax.jacobian(lambda th: step_fn(h, th, u_t, n))(full_theta)
        S = J_t @ S + G_t
        h_next = step_fn(h, full_theta, u_t, n)
        y = qs[t] @ h_next
        dl_dh = (jnp.cos(y) + y) * qs[t]
        g_rtrl = g_rtrl + dl_dh @ S
        h = h_next
    rel_err = float(jnp.linalg.norm(g_rtrl - g_bptt) / (jnp.linalg.norm(g_bptt) + 1e-12))
    return rel_err


def verify_reduced_credit(arch, n, seed=0, T=20):
    """Verify the REDUCED (2n-scalar) exact RTRL matches BPTT -- confirming
    the persistent eligibility count claim (2n for all three architectures)."""
    r = 2 * n
    theta = make_params(arch, n, seed)
    rec_theta, b_in, c_out = unpack(theta, r)
    full_theta = jnp.concatenate([rec_theta, b_in])
    step_fn = STEP_FNS[arch]
    rng = np.random.RandomState(seed + 1)
    us = jnp.array(rng.randn(T) * U_SCALE)
    qs = jnp.array(rng.randn(T, r) * 0.5)

    def loss_bptt(full_theta):
        def step(h, inputs):
            u, q = inputs
            h_next = step_fn(h, full_theta, u, n)
            return h_next, jnp.dot(q, h_next)
        h0 = jnp.zeros(r, dtype=jnp.float64)
        _, ys = jax.lax.scan(step, h0, (us, qs))
        return jnp.sum(jnp.sin(ys) + 0.5 * ys ** 2)
    g_bptt_full = jax.grad(loss_bptt)(full_theta)
    g_bptt_rec = g_bptt_full[:r]   # gradient w.r.t. recurrent theta only

    if arch == "RealDiagonal":
        # 1 scalar eligibility per independent coordinate: s_{t+1}=lambda_i*s_t+h_i,t
        h = jnp.zeros(r, dtype=jnp.float64)
        s = jnp.zeros(r, dtype=jnp.float64)
        g_red = jnp.zeros(r, dtype=jnp.float64)
        lam = rec_theta
        for t in range(T):
            u_t = us[t]
            h_next = lam * h + b_in * u_t
            s_next = lam * s + h
            y = qs[t] @ h_next
            dl_dh = (jnp.cos(y) + y) * qs[t]
            g_red = g_red + dl_dh * s_next   # diagonal: d(h_i)/d(lambda_i) = s_i, no cross terms
            h, s = h_next, s_next
    else:
        mult_fn = (lambda u, v, n=n: __import__("credit_memory.b35j_complex_vs_dual", fromlist=["x"])
                   .complex_mult_blockwise(u, v, n)) if arch == "ComplexLocal" else \
                  (lambda u, v, n=n: __import__("credit_memory.b35a_product_local_algebra", fromlist=["x"])
                   .alg_mult_blockwise(u, v, n, 2))
        tmult_fn = (lambda u, q, n=n: __import__("credit_memory.b35j_complex_vs_dual", fromlist=["x"])
                    .complex_transpose_mult(u, q, n)) if arch == "ComplexLocal" else \
                   (lambda u, q, n=n: __import__("credit_memory.b35a_product_local_algebra", fromlist=["x"])
                    .transpose_mult_blockwise(u, q, n, 2))
        h = jnp.zeros(r, dtype=jnp.float64)
        s = jnp.zeros(r, dtype=jnp.float64)
        g_red = jnp.zeros(r, dtype=jnp.float64)
        for t in range(T):
            u_t = us[t]
            h_next = mult_fn(rec_theta, h) + b_in * u_t
            s_next = mult_fn(rec_theta, s) + h
            y = qs[t] @ h_next
            dl_dh = (jnp.cos(y) + y) * qs[t]
            g_red = g_red + tmult_fn(s_next, dl_dh)
            h, s = h_next, s_next

    rel_err = float(jnp.linalg.norm(g_red - g_bptt_rec) / (jnp.linalg.norm(g_bptt_rec) + 1e-12))
    return rel_err


# =======================================================================
# BPTT training (fixed-length H sequences, random input, Adam).
# =======================================================================
def make_train_sequence(seed, T, teacher_dim=2):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(T) * U_SCALE)


def clip_vec(g, max_norm=CLIP_NORM):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return g * scale


def train_one(arch, n, teacher_name, lr, seed, n_train=N_TRAIN, H_len=H):
    teacher = TEACHERS[teacher_name]
    theta = make_params(arch, n, seed)

    def loss_fn(theta, us, targets):
        ys = rollout_student(arch, n, theta, us)
        return jnp.mean(0.5 * (ys - targets) ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    r = 2 * n
    opt_state = adam_init(theta)
    diverged = False
    for step in range(n_train):
        us = make_train_sequence(20_000 + step, H_len)
        targets = teacher_rollout(teacher["A"], teacher["b"], teacher["c"], jnp.zeros(2), us)
        loss_val = float(loss_jit(theta, us, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged = True
            break
        g = clip_vec(grad_fn(theta, us, targets))
        theta, opt_state = adam_step(theta, g, opt_state, lr)
        rec_theta, b_in, c_out = unpack(theta, r)
        if arch == "RealDiagonal":
            rec_theta = jnp.clip(rec_theta, -0.95, 0.95)
        else:
            RT = rec_theta.reshape(n, 2)
            base = jnp.clip(RT[:, 0], -0.95, 0.95)
            tail_norm = jnp.abs(RT[:, 1])
            tail_scale = jnp.where(tail_norm > 1.0, 1.0 / (tail_norm + 1e-12), 1.0)
            RT = jnp.stack([base, RT[:, 1] * tail_scale], axis=1)
            rec_theta = RT.reshape(r)
        theta = jnp.concatenate([rec_theta, b_in, c_out])
    return theta, diverged


# =======================================================================
# Extrapolation evaluation: impulse response NMSE at H, 2H, 4H.
# =======================================================================
def eval_impulse_nmse(arch, n, teacher_name, theta, horizon):
    teacher = TEACHERS[teacher_name]
    us = jnp.zeros(horizon).at[0].set(1.0)
    y_teacher = teacher_rollout(teacher["A"], teacher["b"], teacher["c"], jnp.zeros(2), us)
    y_student = rollout_student(arch, n, theta, us)
    mse = float(jnp.mean((y_student - y_teacher) ** 2))
    var = float(jnp.mean(y_teacher ** 2))
    return mse / (var + 1e-12)


def select_lr(arch, n, teacher_name, lr_grid=LR_GRID, tuning_seeds=TUNING_SEEDS):
    scores = {}
    for lr in lr_grid:
        vals = []
        for seed in tuning_seeds:
            theta, diverged = train_one(arch, n, teacher_name, lr, seed)
            if diverged:
                vals.append(float("inf"))
            else:
                vals.append(eval_impulse_nmse(arch, n, teacher_name, theta, H))
        scores[lr] = float(np.mean(vals))
    return min(scores, key=scores.get), scores


def bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs)
    boots = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(diffs))


# =======================================================================
# Main experiment.
# =======================================================================
def run_capacity_sweep():
    print("=" * 78)
    print("VALIDITY: RTRL (full) and reduced-credit RTRL vs BPTT, all archs")
    print("=" * 78)
    all_ok = True
    for arch in ("RealDiagonal", "ComplexLocal", "DualLocal"):
        for n in (1, 2, 4, 8):
            e1 = verify_rtrl_vs_bptt(arch, n)
            e2 = verify_reduced_credit(arch, n)
            ok = e1 < 1e-10 and e2 < 1e-10
            all_ok &= ok
            print(f"  {arch:14s} n={n}  full_RTRL_rel_err={e1:.3e}  reduced_credit_rel_err={e2:.3e}  PASS={ok}")
    print(f"ALL VALIDITY PASS: {all_ok}")
    c1, fit_err = verify_generalized_mode()
    print(f"Teacher A generalized-mode check: c1={c1:.4f} (nonzero), fit_err={fit_err:.2e}")
    if not all_ok:
        print("STOPPING.")
        return None

    results = {}
    for teacher_name in ("generalized_mode", "oscillatory"):
        print(f"\n{'='*78}\nTEACHER: {teacher_name}\n{'='*78}")
        for arch in ("RealDiagonal", "ComplexLocal", "DualLocal"):
            for n in N_FACTORS_GRID:
                best_lr, lr_scores = select_lr(arch, n, teacher_name)
                nmse_H, nmse_2H, nmse_4H = [], [], []
                for seed in EVAL_SEEDS:
                    theta, diverged = train_one(arch, n, teacher_name, best_lr, seed)
                    if diverged:
                        nmse_H.append(np.nan); nmse_2H.append(np.nan); nmse_4H.append(np.nan)
                        continue
                    nmse_H.append(eval_impulse_nmse(arch, n, teacher_name, theta, H))
                    nmse_2H.append(eval_impulse_nmse(arch, n, teacher_name, theta, 2 * H))
                    nmse_4H.append(eval_impulse_nmse(arch, n, teacher_name, theta, 4 * H))
                key = (teacher_name, arch, n)
                results[key] = dict(best_lr=best_lr, nmse_H=nmse_H, nmse_2H=nmse_2H, nmse_4H=nmse_4H)
                print(f"  {arch:14s} n={n}  best_lr={best_lr}  "
                      f"median NMSE(H)={np.nanmedian(nmse_H):.4e}  "
                      f"NMSE(2H)={np.nanmedian(nmse_2H):.4e}  NMSE(4H)={np.nanmedian(nmse_4H):.4e}  "
                      f"n_nan={np.sum(np.isnan(nmse_H))}")

    with open("/tmp/b36a_results.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, f, indent=2, default=str)
    print("\nSaved to /tmp/b36a_results.json")
    return results


if __name__ == "__main__":
    run_capacity_sweep()
