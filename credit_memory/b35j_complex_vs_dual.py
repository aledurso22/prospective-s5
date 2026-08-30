"""B35j -- direct ComplexLocal vs DualLocal comparison. Does not
redesign ProductLocal, does not introduce Hessian transport.

FACTORIZATION CHECK (per instruction, reported not silently bypassed):
B35d's actual RegularBlock uses D_LOCAL=4 (credit_memory/
b35c_matched_credit_frontier.py), NOT d=2. "DualLocal" here is
therefore a FRESH, smaller d=2 instance of the SAME jet-algebra
machinery (alg_mult_blockwise, unmodified, at d=2 instead of 4) --
this exactly reproduces M_D(a,b)=[[a,0],[b,a]] (verified below), not a
redesign. "ComplexLocal" is a new, parallel implementation with the
identical per-factor (Q factors, 2 real coordinates each, block-
diagonal) structure but the complex-multiplication rule instead.
Capacity match: Q=64 factors, d=2 => r=P=128 real coordinates for
both, matching B35d's primary C=128 scale.

Run: python -m credit_memory.b35j_complex_vs_dual
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, transpose_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_NIL
from credit_memory.b35d_streaming_sysid import (
    make_A, teacher_step, B_TEACHER, T_TOTAL, T_CHANGE, EVAL_SEEDS, clip_vec, CLIP_NORM,
    compute_metrics, X_DIM_TEACHER,
)
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, DIVERGENCE_LOSS_CEIL

D = 2
Q_FACTORS = 64
R_STATE = Q_FACTORS * D   # 128, matches B35d's primary C=128 scale
LR_GRID = (0.01, 0.02, 0.05)
TUNING_SEEDS = (100, 101)
SELECTION_WINDOW = (200, 800)
K_VALUES = (1, 5, 20, 100)
RHO_MAX = 0.95   # stability bound, same philosophy/magnitude as DualLocal's RHO_BASE


# =======================================================================
# ComplexLocal: parallel per-factor implementation, complex multiplication.
# =======================================================================
def complex_mult_blockwise(u, v, Q=Q_FACTORS):
    U, V = u.reshape(Q, 2), v.reshape(Q, 2)
    a_u, b_u = U[:, 0], U[:, 1]
    a_v, b_v = V[:, 0], V[:, 1]
    a_w = a_u * a_v - b_u * b_v
    b_w = a_u * b_v + a_v * b_u
    return jnp.stack([a_w, b_w], axis=1).reshape(Q * 2)


def complex_transpose_mult(u, q, Q=Q_FACTORS):
    U, Qv = u.reshape(Q, 2), q.reshape(Q, 2)
    a, b = U[:, 0], U[:, 1]
    q0, q1 = Qv[:, 0], Qv[:, 1]
    out0 = a * q0 + b * q1
    out1 = -b * q0 + a * q1
    return jnp.stack([out0, out1], axis=1).reshape(Q * 2)


def project_complex(theta, Q=Q_FACTORS, rho_max=RHO_MAX):
    T = theta.reshape(Q, 2)
    modulus = jnp.sqrt(jnp.sum(T ** 2, axis=1))
    scale = jnp.where(modulus > rho_max, rho_max / (modulus + 1e-12), 1.0)
    return (T * scale[:, None]).reshape(Q * 2)


def project_dual(theta, Q=Q_FACTORS, d=D):
    return project_local_tails(theta, Q, d, rho_nil=RHO_NIL, rho_base=RHO_MAX)


ARCH = dict(
    DualLocal=dict(mult=lambda u, v: alg_mult_blockwise(u, v, Q_FACTORS, D),
                   tmult=lambda u, q: transpose_mult_blockwise(u, q, Q_FACTORS, D),
                   project=project_dual),
    ComplexLocal=dict(mult=complex_mult_blockwise, tmult=complex_transpose_mult, project=project_complex),
)


# =======================================================================
# Verification: reduced RTRL vs full real-coordinate RTRL vs BPTT,
# fixed theta, for BOTH algebras.
# =======================================================================
def make_rollout_and_bptt(mult_fn, B_in, theta_shape_check=None):
    def rollout(h0, theta, xs, us):
        def step(h, inputs):
            x_t, u_t = inputs
            model_in = jnp.concatenate([x_t, jnp.array([u_t])])
            h_next = mult_fn(theta, h) + B_in @ model_in
            return h_next, h_next
        _, Hs = jax.lax.scan(step, h0, (xs, us))
        return Hs
    return rollout


def verify_reduced_vs_full_vs_bptt(name, mult_fn, tmult_fn, T=15, seed=0):
    rng = np.random.RandomState(seed)
    theta = jnp.array(rng.randn(R_STATE) * 0.2)
    B_in = jnp.array(rng.randn(R_STATE, X_DIM_TEACHER + 1) / np.sqrt(X_DIM_TEACHER + 1))
    h0 = jnp.array(rng.randn(R_STATE) * 0.1)
    xs = jnp.array(rng.randn(T, X_DIM_TEACHER) * 0.4)
    us = jnp.array(rng.randn(T) * 0.4)
    qs = jnp.array(rng.randn(T, R_STATE) * 0.5)

    def loss_bptt(theta):
        def step(h, inputs):
            x_t, u_t, q_t = inputs
            model_in = jnp.concatenate([x_t, jnp.array([u_t])])
            h_next = mult_fn(theta, h) + B_in @ model_in
            return h_next, jnp.dot(q_t, h_next)
        _, ys = jax.lax.scan(step, h0, (xs, us, qs))
        return jnp.sum(jnp.sin(ys) + 0.5 * ys ** 2)

    g_bptt = jax.grad(loss_bptt)(theta)

    # full real-coordinate RTRL (autodiff jacobian per step, independent check)
    h = h0
    S = jnp.zeros((R_STATE, R_STATE), dtype=jnp.float64)
    g_full = jnp.zeros(R_STATE, dtype=jnp.float64)
    for t in range(T):
        x_t, u_t = xs[t], us[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        J_t = jax.jacobian(lambda hh: mult_fn(theta, hh) + B_in @ model_in)(h)
        G_t = jax.jacobian(lambda th: mult_fn(th, h) + B_in @ model_in)(theta)
        S = J_t @ S + G_t
        h_next = mult_fn(theta, h) + B_in @ model_in
        y = qs[t] @ h_next
        dl_dh = (jnp.cos(y) + y) * qs[t]
        g_full = g_full + dl_dh @ S
        h = h_next
    rel_full = float(jnp.linalg.norm(g_full - g_bptt) / (jnp.linalg.norm(g_bptt) + 1e-12))

    # reduced (2-real-coordinate-per-factor) RTRL
    h = h0
    s = jnp.zeros(R_STATE, dtype=jnp.float64)
    g_reduced = jnp.zeros(R_STATE, dtype=jnp.float64)
    for t in range(T):
        x_t, u_t = xs[t], us[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        h_next = mult_fn(theta, h) + B_in @ model_in
        s_next = mult_fn(theta, s) + h
        y = qs[t] @ h_next
        dl_dh = (jnp.cos(y) + y) * qs[t]
        g_reduced = g_reduced + tmult_fn(s_next, dl_dh)
        h, s = h_next, s_next
    rel_reduced = float(jnp.linalg.norm(g_reduced - g_bptt) / (jnp.linalg.norm(g_bptt) + 1e-12))

    print(f"  [{name}] reduced-vs-BPTT rel_err={rel_reduced:.3e}   full-vs-BPTT rel_err={rel_full:.3e}")
    return rel_reduced < 1e-8 and rel_full < 1e-8


# =======================================================================
# Teachers: B35d's own mixed teacher (reused unmodified), plus two
# minimal 2-dim positive controls (same rho/omega/lambda/mu constants,
# same regime-change convention, standalone single-block systems).
# =======================================================================
def teacher_step_mixed(x, u, t):
    return teacher_step(x, u, t)   # reused unmodified from b35d


R_OSC_PRE, OMEGA_OSC, R_OSC_POST = 0.80, 0.6, 0.50
LAM_GEN_PRE, MU_GEN, LAM_GEN_POST = 0.85, 0.30, 0.55
B_OSC = jnp.array([0.8, 0.6])
B_GEN = jnp.array([0.8, 0.6])


def make_osc_matrix(rho, omega):
    return rho * jnp.array([[jnp.cos(omega), -jnp.sin(omega)], [jnp.sin(omega), jnp.cos(omega)]])


def make_gen_matrix(lam, mu):
    return jnp.array([[lam, mu], [0.0, lam]])


OSC_A_PRE, OSC_A_POST = make_osc_matrix(R_OSC_PRE, OMEGA_OSC), make_osc_matrix(R_OSC_POST, OMEGA_OSC)
GEN_A_PRE, GEN_A_POST = make_gen_matrix(LAM_GEN_PRE, MU_GEN), make_gen_matrix(LAM_GEN_POST, MU_GEN)


def teacher_step_osc(x, u, t):
    A = jnp.where(t < T_CHANGE, OSC_A_PRE, OSC_A_POST)
    return A @ x + B_OSC * u


def teacher_step_gen(x, u, t):
    A = jnp.where(t < T_CHANGE, GEN_A_PRE, GEN_A_POST)
    return A @ x + B_GEN * u


TEACHERS = dict(
    mixed=dict(step=teacher_step_mixed, dim=X_DIM_TEACHER),
    oscillatory=dict(step=teacher_step_osc, dim=2),
    generalized_mode=dict(step=teacher_step_gen, dim=2),
)


def make_trajectory(seed, teacher_step_fn, teacher_dim, T_total=T_TOTAL, u_scale=0.3):
    rng = np.random.RandomState(seed)
    us = jnp.array(rng.randn(T_total) * u_scale)

    def step(x, inputs):
        u, t = inputs
        x_next = teacher_step_fn(x, u, t)
        return x_next, x_next

    x0 = jnp.zeros(teacher_dim, dtype=jnp.float64)
    t_idx = jnp.arange(T_total)
    _, xs_next_all = jax.lax.scan(step, x0, (us, t_idx))
    xs_t = jnp.concatenate([x0[None, :], xs_next_all[:-1]], axis=0)
    return xs_t, us, xs_next_all


# =======================================================================
# Block-update training loop (K=1 recovers continual regime A). Carry
# eligibility across block boundaries (only variant implemented, per
# scope). One optimizer update per block, on the ACCUMULATED gradient.
# =======================================================================
def make_block_step_fn(mult_fn, tmult_fn, project_fn, B_in, teacher_dim, lr, K):
    def block_step(carry, block_inputs):
        h, s, theta, C_out, opt_th, opt_co = carry
        xs_blk, us_blk, xs_next_blk = block_inputs   # each (K, ...)

        def inner(inner_carry, inputs):
            h, s, g_acc, g_cout_acc = inner_carry
            x_t, u_t, x_next = inputs
            model_in = jnp.concatenate([x_t, jnp.array([u_t])])
            h_next = mult_fn(theta, h) + B_in @ model_in
            s_next = mult_fn(theta, s) + h
            x_hat = C_out @ h_next
            diff = x_hat - x_next
            loss = 0.5 * jnp.sum(diff ** 2)
            dl_dh = C_out.T @ diff
            g_acc = g_acc + tmult_fn(s_next, dl_dh)
            g_cout_acc = g_cout_acc + jnp.outer(diff, h_next)
            return (h_next, s_next, g_acc, g_cout_acc), (loss, x_hat, jnp.all(jnp.isfinite(h_next)))

        g0 = jnp.zeros_like(theta)
        gc0 = jnp.zeros_like(C_out)
        (h_new, s_new, g_acc, g_cout_acc), (losses, x_hats, finites) = jax.lax.scan(
            inner, (h, s, g0, gc0), (xs_blk, us_blk, xs_next_blk))

        g_theta = clip_vec(g_acc)
        g_cout = clip_vec(g_cout_acc.reshape(-1)).reshape(C_out.shape)
        theta_after_opt, opt_th_new = adam_step(theta, g_theta, opt_th, lr)
        theta_new = project_fn(theta_after_opt)
        C_out_new, opt_co_new = adam_step(C_out, g_cout, opt_co, lr)

        new_carry = (h_new, s_new, theta_new, C_out_new, opt_th_new, opt_co_new)
        return new_carry, (losses, x_hats, finites, jnp.linalg.norm(theta_new - theta), jnp.linalg.norm(g_theta))
    return block_step


def run_training(arch_name, teacher_name, seed, lr, K, T_total=T_TOTAL):
    teacher = TEACHERS[teacher_name]
    xs_t, us, xs_next = make_trajectory(seed, teacher["step"], teacher["dim"], T_total=T_total)
    rng = np.random.RandomState(1000 + seed)
    theta = ARCH[arch_name]["project"](jnp.array(rng.randn(R_STATE) * 0.2))
    B_in = jnp.array(rng.randn(R_STATE, teacher["dim"] + 1) / np.sqrt(teacher["dim"] + 1))
    C_out = jnp.array(rng.randn(teacher["dim"], R_STATE) * (1.0 / np.sqrt(R_STATE)))
    h0 = jnp.zeros(R_STATE, dtype=jnp.float64)
    s0 = jnp.zeros(R_STATE, dtype=jnp.float64)
    opt_th, opt_co = adam_init(theta), adam_init(C_out)

    n_blocks = T_total // K
    xs_b = xs_t[:n_blocks * K].reshape(n_blocks, K, teacher["dim"])
    us_b = us[:n_blocks * K].reshape(n_blocks, K)
    xn_b = xs_next[:n_blocks * K].reshape(n_blocks, K, teacher["dim"])

    step_fn = make_block_step_fn(ARCH[arch_name]["mult"], ARCH[arch_name]["tmult"], ARCH[arch_name]["project"],
                                  B_in, teacher["dim"], lr, K)
    scan_fn = jax.jit(lambda carry, inputs: jax.lax.scan(step_fn, carry, inputs))
    carry0 = (h0, s0, theta, C_out, opt_th, opt_co)
    _, (losses, x_hats, finites, delta_norms, grad_norms) = scan_fn(carry0, (xs_b, us_b, xn_b))

    losses = losses.reshape(-1)
    x_hats = x_hats.reshape(-1, teacher["dim"])
    finites = finites.reshape(-1)
    return dict(losses=losses, x_hats=x_hats, finites=finites, xs_next=xs_next[:n_blocks * K],
                delta_norms=delta_norms, grad_norms=grad_norms)


# =======================================================================
# LR selection (once per architecture, K=1, mixed teacher, tuning seeds).
# =======================================================================
def select_lr(arch_name, lr_grid=LR_GRID, tuning_seeds=TUNING_SEEDS):
    scores = {}
    for lr in lr_grid:
        seed_scores = []
        for seed in tuning_seeds:
            out = run_training(arch_name, "mixed", seed, lr, K=1, T_total=800)
            a, b = SELECTION_WINDOW
            finite = np.asarray(out["finites"])
            losses = np.asarray(out["losses"])
            seg = losses[a:b][finite[a:b]]
            seed_scores.append(np.mean(seg) if seg.size else float("inf"))
        scores[lr] = float(np.mean(seed_scores))
    return min(scores, key=scores.get), scores


# =======================================================================
# Carried-vs-frozen gradient mismatch (B35e-style), K=1 only, both algebras.
# =======================================================================
def carried_vs_frozen_check(arch_name, lr, seed=EVAL_SEEDS[0], checkpoints=(50, 150, 300)):
    mult_fn, tmult_fn = ARCH[arch_name]["mult"], ARCH[arch_name]["tmult"]
    project_fn = ARCH[arch_name]["project"]
    teacher = TEACHERS["mixed"]
    xs_t, us, xs_next = make_trajectory(seed, teacher["step"], teacher["dim"], T_total=max(checkpoints) + 5)
    rng = np.random.RandomState(1000 + seed)
    theta = project_fn(jnp.array(rng.randn(R_STATE) * 0.2))
    B_in = jnp.array(rng.randn(R_STATE, teacher["dim"] + 1) / np.sqrt(teacher["dim"] + 1))
    C_out = jnp.array(rng.randn(teacher["dim"], R_STATE) * (1.0 / np.sqrt(R_STATE)))
    h0 = jnp.zeros(R_STATE, dtype=jnp.float64)
    h, s = h0, jnp.zeros(R_STATE, dtype=jnp.float64)
    opt_th, opt_co = adam_init(theta), adam_init(C_out)
    theta_hist, h_hist, s_hist = [theta], [h], [s]

    for k in range(max(checkpoints)):
        x_k, u_k, x_next_k = xs_t[k], us[k], xs_next[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        h_next = mult_fn(theta, h) + B_in @ model_in
        s_next = mult_fn(theta, s) + h
        diff = C_out @ h_next - x_next_k
        dl_dh = C_out.T @ diff
        g_theta = clip_vec(tmult_fn(s_next, dl_dh))
        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        theta_after, opt_th = adam_step(theta, g_theta, opt_th, lr)
        theta = project_fn(theta_after)
        C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)
        h, s = h_next, s_next
        theta_hist.append(theta); h_hist.append(h); s_hist.append(s)

    results = {}
    for t in checkpoints:
        theta_t, h_t, s_t = theta_hist[t], h_hist[t], s_hist[t]
        target_t = xs_next[t - 1]
        q_t = C_out.T @ (C_out @ h_t - target_t)
        g_carried = tmult_fn(s_t, q_t)

        def rollout(th):
            def step(hh, inputs):
                x_k, u_k = inputs
                model_in = jnp.concatenate([x_k, jnp.array([u_k])])
                return mult_fn(th, hh) + B_in @ model_in, None
            hf, _ = jax.lax.scan(step, h0, (xs_t[:t], us[:t]))
            return hf

        def loss_frozen(th):
            diff = C_out @ rollout(th) - target_t
            return 0.5 * jnp.sum(diff ** 2)

        g_frozen = jax.grad(loss_frozen)(theta_t)
        eps = float(jnp.linalg.norm(g_carried - g_frozen) / (jnp.linalg.norm(g_frozen) + 1e-8))
        cos = float(jnp.dot(g_carried, g_frozen) / (jnp.linalg.norm(g_carried) * jnp.linalg.norm(g_frozen) + 1e-8))
        results[t] = dict(eps=eps, cos=cos)
    return results


# =======================================================================
# Main experiment.
# =======================================================================
def summarize_run(name, teacher_name, arch_name, K, seeds=EVAL_SEEDS, lr=None):
    pre_vals, post_vals, all_losses, all_delta, all_grad, n_div = [], [], [], [], [], 0
    for seed in seeds:
        out = run_training(arch_name, teacher_name, seed, lr, K)
        finite = np.asarray(out["finites"])
        m = compute_metrics(jnp.array(out["losses"]), jnp.array(out["x_hats"]), jnp.array(out["finites"]),
                             out["xs_next"])
        if m["pre_nmse"] is not None:
            pre_vals.append(m["pre_nmse"])
        if m["post_nmse"] is not None:
            post_vals.append(m["post_nmse"])
        if m["diverged"]:
            n_div += 1
        all_losses.append(np.asarray(out["losses"])[finite])
        all_delta.append(np.asarray(out["delta_norms"]))
        all_grad.append(np.asarray(out["grad_norms"]))
    loss_all = np.concatenate(all_losses) if all_losses else np.array([np.nan])
    delta_all = np.concatenate(all_delta) if all_delta else np.array([np.nan])
    grad_all = np.concatenate(all_grad) if all_grad else np.array([np.nan])
    row = dict(name=name, teacher=teacher_name, arch=arch_name, K=K, lr=lr,
               pre_median=float(np.median(pre_vals)) if pre_vals else None,
               post_median=float(np.median(post_vals)) if post_vals else None,
               loss_std=float(np.std(loss_all)), n_diverged=n_div, n_total=len(seeds),
               delta_median=float(np.median(delta_all)), grad_median=float(np.median(grad_all)))
    print(f"  [{name}] teacher={teacher_name} K={K:4d} lr={lr}  pre_med={row['pre_median']}  "
          f"post_med={row['post_median']}  loss_std={row['loss_std']:.4e}  diverged={n_div}/{len(seeds)}  "
          f"delta_med={row['delta_median']:.4e}  grad_med={row['grad_median']:.4e}")
    return row


def run_experiment():
    print("=" * 78)
    print("VALIDITY: reduced-vs-full-vs-BPTT, both algebras")
    print("=" * 78)
    ok1 = verify_reduced_vs_full_vs_bptt("DualLocal", ARCH["DualLocal"]["mult"], ARCH["DualLocal"]["tmult"])
    ok2 = verify_reduced_vs_full_vs_bptt("ComplexLocal", ARCH["ComplexLocal"]["mult"], ARCH["ComplexLocal"]["tmult"])
    print(f"ALL PASS: {ok1 and ok2}")
    if not (ok1 and ok2):
        print("STOPPING.")
        return None

    print("\n" + "=" * 78)
    print("LR selection (K=1, mixed teacher, tuning seeds)")
    print("=" * 78)
    lrs = {}
    for arch_name in ("DualLocal", "ComplexLocal"):
        best_lr, scores = select_lr(arch_name)
        lrs[arch_name] = best_lr
        print(f"  [{arch_name}] scores={scores} -> best_lr={best_lr}")

    print("\n" + "=" * 78)
    print("K sweep (mixed teacher)")
    print("=" * 78)
    results = []
    for arch_name in ("DualLocal", "ComplexLocal"):
        for K in K_VALUES:
            results.append(summarize_run(arch_name, "mixed", arch_name, K, lr=lrs[arch_name]))

    print("\n" + "=" * 78)
    print("Diagnostic teachers (K=1 and K=100)")
    print("=" * 78)
    for teacher_name in ("oscillatory", "generalized_mode"):
        for arch_name in ("DualLocal", "ComplexLocal"):
            for K in (1, 100):
                results.append(summarize_run(arch_name, teacher_name, arch_name, K, lr=lrs[arch_name]))

    print("\n" + "=" * 78)
    print("Carried-vs-frozen mismatch (K=1, mixed teacher)")
    print("=" * 78)
    for arch_name in ("DualLocal", "ComplexLocal"):
        res = carried_vs_frozen_check(arch_name, lrs[arch_name])
        for t, r in res.items():
            print(f"  [{arch_name}] t={t}: eps_frozen={r['eps']:.4f}  cos={r['cos']:.4f}")

    return results, lrs


if __name__ == "__main__":
    run_experiment()
