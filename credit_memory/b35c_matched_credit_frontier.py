"""B35c -- final robustness benchmark: is the B35b matched-credit
advantage a stable frontier across budgets, or a single-budget result?

B35b (commit 5d96970, PHASE_B35B.md) is FROZEN and untouched here.
Architecture is frozen too -- RegularBlock/GenericBlock/RTU are reused
exactly as validated; only sizes vary per a PREDECLARED rule.

Predeclared sizing rule (fixed BEFORE running/seeing results), d=4
(the practical local size used throughout B35a/B35b):
  RegularBlock: p=d=4, Q=C/p        -> r=Q*d=C,   P=Q*p=C,   credit=Q*p=C
  GenericBlock: p=d=4, Q=C/(d*p)    -> r=Q*d=C/4, P=Q*p=C/4, credit=Q*d*p=C
  RTU:          hidden=C/8          -> r=2*hidden=C/4, P=4*hidden=C/2, credit=8*hidden=C
Budgets C in {32,64,128,256} all divide evenly under this rule.

Run: python -m credit_memory.b35c_matched_credit_frontier
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import jordan_rollout, RHO_BASE, RHO_NIL
from credit_memory.b35b2_generic_vs_regular import (
    make_regular_student, make_generic_student, make_dense_linear_teacher,
    generic_module_rtrl_grad, make_generic_params,
)
from credit_memory.p2a_expressivity_credit_frontier import (
    rtu_make_params, rtu_step, adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL,
    make_teacher_C_multipole,
)

D_LOCAL = 4
BUDGETS = (32, 64, 128, 256)
SEEDS = (0, 1, 2)
LR_GRID = (0.01, 0.03, 0.1)
N_TRAIN, N_VAL, N_TEST, T_TRAIN = 100, 15, 15, 64


# ---------------------------------------------------------------------
# Predeclared per-budget architecture configs.
# ---------------------------------------------------------------------
def regular_config(C, d=D_LOCAL):
    p = d
    Q = C // p
    return dict(Q=Q, d=d, p=p, r=Q * d, P=Q * p, credit=Q * p, generic_dense_pred=Q * d * p)


def generic_config(C, d=D_LOCAL):
    p = d
    Q = C // (d * p)
    return dict(Q=Q, d=d, p=p, r=Q * d, P=Q * p, credit=Q * d * p, generic_dense_pred=Q * d * p)


def rtu_config(C):
    h = C // 8
    return dict(hidden=h, r=2 * h, P=4 * h, credit=8 * h, generic_dense_pred=(2 * h) * (4 * h))


# ---------------------------------------------------------------------
# RTU student (reuses the validated RTU cell; new scalar-readout wrapper).
# ---------------------------------------------------------------------
def make_rtu_student(hidden_dim):
    def make_params(seed):
        rng = np.random.RandomState(seed + 555)
        p = rtu_make_params(seed, hidden_dim)
        C_out = jnp.array(rng.randn(2 * hidden_dim) * (1.0 / np.sqrt(2 * hidden_dim)))
        return dict(rtu=p, C_out=C_out)

    def rollout_y(params, xs):
        def step(h, x_t):
            h_next = rtu_step(h, params["rtu"], x_t, hidden_dim)
            return h_next, h_next
        h0 = jnp.zeros(2 * hidden_dim, dtype=jnp.float64)
        _, Hs = jax.lax.scan(step, h0, xs)
        return Hs @ params["C_out"]

    def project(params):
        return params

    return rollout_y, make_params, project


# ---------------------------------------------------------------------
# Task C: RTU-multipole teacher, scalar-output wrapper (reused, cheap).
# ---------------------------------------------------------------------
_MULTIPOLE_TEACHER = make_teacher_C_multipole(seed=779, hidden_dim=16)


def multipole_scalar_rollout(h0, xs):
    Hs = _MULTIPOLE_TEACHER.rollout_fn(h0, _MULTIPOLE_TEACHER.params, xs)
    return Hs @ _MULTIPOLE_TEACHER.W[0]


TASKS = [
    ("A_generalized_mode", jordan_rollout, 2),
    ("B_neutral_dense", make_dense_linear_teacher(seed=999, state_dim=8)[0], 8),
    ("C_multipole", multipole_scalar_rollout, _MULTIPOLE_TEACHER.state_dim),
]


# ---------------------------------------------------------------------
# Generic training loop (own copy, not editing frozen b35b2) -- also
# retains per-seed test NMSE values for median/spread/CI reporting.
# ---------------------------------------------------------------------
def make_xs(seed, T):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(T) * 0.4)


def train_one_run(rollout_y, make_params, project, teacher_rollout, teacher_state_dim, lr, seed_init,
                   n_train=N_TRAIN, T=T_TRAIN):
    def loss_fn(params, xs, targets):
        ys = rollout_y(params, xs)
        return jnp.mean(0.5 * (ys - targets) ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_params(seed_init)
    params = project(params)
    opt_state = adam_init(params)
    diverged, diverged_at = False, None
    t0 = time.time()
    for step in range(n_train):
        xs = make_xs(20_000 + step, T)
        targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
        loss_val = float(loss_jit(params, xs, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        g = clip_grad(grad_fn(params, xs, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project(params)
    train_time = time.time() - t0
    if diverged:
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init,
                    train_time=train_time)

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            xs = make_xs(offset + i, T)
            targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
            losses.append(float(loss_jit(params, xs, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_var = eval_split(95_000, N_VAL)
    test_losses, test_var = eval_split(200_000, N_TEST)
    if not all(np.isfinite(val_losses)) or not all(np.isfinite(test_losses)):
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=n_train, lr=lr, seed=seed_init,
                    train_time=train_time)
    nmse = float(np.mean(val_losses) / (np.mean(val_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_var) + 1e-12))
    return dict(nmse=nmse, test_nmse=test_nmse, diverged=False, diverged_at=None, lr=lr, seed=seed_init,
                train_time=train_time)


def train_with_grid(rollout_y, make_params, project, teacher_rollout, teacher_state_dim,
                     lr_grid=LR_GRID, seeds=SEEDS):
    all_runs = []
    t0 = time.time()
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run(rollout_y, make_params, project, teacher_rollout, teacher_state_dim,
                                 lr, seed_init=1000 + seed)
            res["lr_tag"] = lr
            all_runs.append(res)
    total_time = time.time() - t0
    finite = [r for r in all_runs if not r["diverged"]]
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    if not finite:
        return dict(status="all_diverged", n_diverged=n_diverged, n_total=len(all_runs), total_time=total_time)
    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))
    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite if r["lr_tag"] == best_lr]
    test_vals = [r["test_nmse"] for r in best_runs]
    return dict(status="ok", n_diverged=n_diverged, n_total=len(all_runs), best_lr=best_lr,
                test_nmse_values=test_vals, test_nmse_median=float(np.median(test_vals)),
                test_nmse_mean=float(np.mean(test_vals)), test_nmse_std=float(np.std(test_vals)),
                total_time=total_time)


# ---------------------------------------------------------------------
# Per-architecture eligibility-update timing (measured once per config,
# not per task -- an architectural property).
# ---------------------------------------------------------------------
def measure_regular_step_time(Q, d):
    from credit_memory.b35b2_generic_vs_regular import regular_reduced_grad
    theta = project_local_tails(jnp.array(np.random.RandomState(0).randn(Q * d) * 0.2), Q, d,
                                 rho_nil=RHO_NIL, rho_base=RHO_BASE)
    b_in = jnp.array(np.random.RandomState(1).randn(Q * d) * 0.3)
    h0 = jnp.zeros(Q * d, dtype=jnp.float64)
    xs = jnp.array(np.random.RandomState(2).randn(5) * 0.4)
    qs = jnp.zeros((5, Q * d))

    @jax.jit
    def one_step(h, s, x_t):
        h_next = alg_mult_blockwise(theta, h, Q, d) + b_in * x_t
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        return h_next, s_next

    h, s = h0, jnp.zeros(Q * d, dtype=jnp.float64)
    h, s = one_step(h, s, xs[0])
    jax.block_until_ready((h, s))
    t0 = time.time()
    for _ in range(20):
        h, s = one_step(h, s, xs[0])
    jax.block_until_ready((h, s))
    return (time.time() - t0) / 20


def measure_generic_step_time(Q, d, p):
    theta, basis = make_generic_params(seed=1, Q=Q, d=d, p=p)
    b_in = jnp.array(np.random.RandomState(1).randn(Q * d) * 0.3)
    from credit_memory.b35b2_generic_vs_regular import _generic_rtrl_batched
    H = jnp.zeros((Q, d), dtype=jnp.float64)
    S = jnp.zeros((Q, d, p), dtype=jnp.float64)
    Theta = theta.reshape(Q, p)
    B_in = b_in.reshape(Q, d)

    @jax.jit
    def one_step(H, S, x_t):
        return _generic_rtrl_batched(H, S, Theta, basis, x_t, B_in)

    H, S = one_step(H, S, 0.1)
    jax.block_until_ready((H, S))
    t0 = time.time()
    for _ in range(20):
        H, S = one_step(H, S, 0.1)
    jax.block_until_ready((H, S))
    return (time.time() - t0) / 20


def measure_rtu_step_time(hidden_dim):
    from credit_memory.b28_rtu_faithful import rtu_streaming_init, rtu_streaming_step
    rng = np.random.RandomState(0)
    params = rtu_make_params(0, hidden_dim)
    stream_state = dict(real=np.zeros(hidden_dim), imag=np.zeros(hidden_dim),
                         S=rtu_streaming_init(hidden_dim, 1))
    u_t = jnp.array(rng.randn(1) * 0.5)
    rtu_streaming_step(params, stream_state, u_t)
    t0 = time.time()
    for _ in range(20):
        rtu_streaming_step(params, stream_state, u_t)
    return (time.time() - t0) / 20


# ---------------------------------------------------------------------
# Main frontier sweep.
# ---------------------------------------------------------------------
def run_frontier():
    print("=" * 78)
    print(f"B35c matched-credit frontier: C in {BUDGETS}, d={D_LOCAL} (predeclared)")
    print("=" * 78)
    results = []
    for C in BUDGETS:
        rc = regular_config(C)
        gc = generic_config(C)
        tc = rtu_config(C)
        print(f"\n--- C={C} ---")
        print(f"  RegularBlock: Q={rc['Q']} r={rc['r']} P={rc['P']} credit={rc['credit']}  "
              f"generic_dense_equiv={rc['generic_dense_pred']}")
        print(f"  GenericBlock: Q={gc['Q']} r={gc['r']} P={gc['P']} credit={gc['credit']}  "
              f"generic_dense_equiv={gc['generic_dense_pred']}")
        print(f"  RTU:          hidden={tc['hidden']} r={tc['r']} P={tc['P']} credit={tc['credit']}  "
              f"generic_dense_equiv={tc['generic_dense_pred']}")

        reg_step_t = measure_regular_step_time(rc["Q"], rc["d"])
        gen_step_t = measure_generic_step_time(gc["Q"], gc["d"], gc["p"])
        rtu_step_t = measure_rtu_step_time(tc["hidden"])
        print(f"  eligibility-step time: Regular={reg_step_t*1e6:.2f}us  Generic={gen_step_t*1e6:.2f}us  "
              f"RTU={rtu_step_t*1e6:.2f}us")

        reg_rollout, reg_make_params, reg_project = make_regular_student(rc["Q"], rc["d"])
        gen_rollout, gen_make_params, gen_project = make_generic_student(100 + C, gc["Q"], gc["d"], gc["p"])
        rtu_rollout, rtu_make_p, rtu_project = make_rtu_student(tc["hidden"])

        archs = dict(RegularBlock=(reg_rollout, reg_make_params, reg_project, rc),
                     GenericBlock=(gen_rollout, gen_make_params, gen_project, gc),
                     RTU=(rtu_rollout, rtu_make_p, rtu_project, tc))

        for task_name, teacher_rollout, tdim in TASKS:
            for arch_name, (rollout_y, make_params, project, cfg) in archs.items():
                res = train_with_grid(rollout_y, make_params, project, teacher_rollout, tdim)
                row = dict(C=C, arch=arch_name, task=task_name, r=cfg["r"], P=cfg["P"], credit=cfg["credit"],
                           generic_dense_pred=cfg["generic_dense_pred"],
                           elig_step_us=dict(RegularBlock=reg_step_t, GenericBlock=gen_step_t,
                                              RTU=rtu_step_t)[arch_name] * 1e6,
                           **res)
                results.append(row)
                if res["status"] == "all_diverged":
                    print(f"    [{task_name}] {arch_name:14s} C={C:4d}  ALL DIVERGED ({res['n_diverged']}/{res['n_total']})")
                else:
                    print(f"    [{task_name}] {arch_name:14s} C={C:4d}  best_lr={res['best_lr']}  "
                          f"test_NMSE median={res['test_nmse_median']:.4e} mean={res['test_nmse_mean']:.4e} "
                          f"std={res['test_nmse_std']:.4e}  diverged={res['n_diverged']}/{res['n_total']}  "
                          f"train_time={res['total_time']:.1f}s")

    with open("/tmp/b35c_frontier_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /tmp/b35c_frontier_results.json")
    return results


if __name__ == "__main__":
    run_frontier()
