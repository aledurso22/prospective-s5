"""B34->B positive-control stability audit (triggered by View 2's medium
tier: B34->B diverged 6/6 at r=500, using the SAME GRAD_CLIP_NORM=10 and
LR grid carried over unchanged from View 1's r=64 setting). Before
touching any optimizer knob, this establishes (1) whether the resized
B34 student can EXACTLY represent teacher B's dynamics at all (r=64,
200, 500, 800), (2) where scale/stability breaks down as r grows, using
ONLY teacher B / B34's own positive control -- never cross-teacher
performance -- to pick anything.

Run: python -m credit_memory.p2a_b34_stability_audit
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.p2a_expressivity_credit_frontier import (
    make_rollout, make_sequence, adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL,
    T_SEQ, make_teacher_B_jet, jet_make_gen_params,
)
from credit_memory.p2a_view2_matched_credit import (
    make_jet_reduced_step, make_student_params, make_loss_decoupled,
    train_student_one_run_v2, jet_make_theta, make_jet_step, jet_param_count,
)
from credit_memory.b34a_jet_algebra_correctness import gen_forward as jet_gen_forward

R_LEVELS = (64, 200, 500, 800)


# ---------------------------------------------------------------------
# STEP 1: exact containment check.
# ---------------------------------------------------------------------
def run_containment_check():
    print("=" * 78)
    print("STEP 1: exact containment -- can the resized B34 student represent teacher B exactly?")
    print("=" * 78)
    for r in R_LEVELS:
        gen_seed = 3000 + r
        gen_params = jet_make_gen_params(seed=gen_seed, r=r)
        teacher, gen_params_t = make_teacher_B_jet(seed=778, r=r, gen_seed=gen_seed)
        assert jnp.max(jnp.abs(gen_params["W1"] - gen_params_t["W1"])) == 0.0, "gen_params mismatch"

        step = make_jet_step(gen_params, r)
        rollout = make_rollout(step)

        h0_t, xs = make_sequence(20_000, T_SEQ, r)

        # (a) SAME h0 as teacher, theta = teacher's theta_star exactly ->
        # must reproduce the teacher's own trajectory to machine precision
        # (pure functional-form containment: same code path, same params).
        Hs_a = rollout(h0_t, teacher.params, xs)
        Hs_teacher = teacher.rollout_fn(h0_t, teacher.params, xs)
        max_diff_a = float(jnp.max(jnp.abs(Hs_a - Hs_teacher)))

        # (b) student's ACTUAL protocol: h0=0, theta = teacher's theta_star
        # (the best a student could possibly do without any training at
        # all, on the exact right parameters) -- isolates the effect of
        # the zero-init/mismatched-h0 choice made for View 2's decoupled
        # readout, independent of any learning difficulty.
        h0_zero = jnp.zeros(r, dtype=jnp.float64)
        Hs_b = rollout(h0_zero, teacher.params, xs)
        max_abs_b = float(jnp.max(jnp.abs(Hs_b)))
        max_abs_teacher = float(jnp.max(jnp.abs(Hs_teacher)))
        finite_b = bool(jnp.all(jnp.isfinite(Hs_b)))
        # gap between the zero-start trajectory and the teacher's own
        # trajectory (same xs, same theta, different h0) at the END of
        # the sequence -- does the mismatch persist/grow or wash out?
        gap_last = float(jnp.max(jnp.abs(Hs_b[-1] - Hs_teacher[-1]))) if finite_b else float("nan")
        gap_first = float(jnp.max(jnp.abs(Hs_b[0] - Hs_teacher[0]))) if finite_b else float("nan")

        print(f"  r={r:4d}  (a) same-h0 max|diff|={max_diff_a:.3e} (expect ~0)   "
              f"(b) zero-h0,theta=theta*: max|Hs|={max_abs_b:.3e} finite={finite_b}  "
              f"teacher max|Hs|={max_abs_teacher:.3e}  gap(t=0)={gap_first:.3e}  gap(t=T-1)={gap_last:.3e}")


# ---------------------------------------------------------------------
# STEP 2: scale/stability diagnostic, B34->B only, at init and over a
# few early (untrained) training steps.
# ---------------------------------------------------------------------
def diagnose_rollout(step_fn, reduced_step_fn, theta, gen_params, h0, xs, r, W, targets, tag):
    T = xs.shape[0]
    h = h0
    s = jnp.zeros(r, dtype=jnp.float64)
    h_norms, y_norms, exo_norms, s_norms, g_norms = [], [], [], [], []
    n_nonfinite = 0
    for t in range(T):
        x_t = jnp.stack([xs[t], 0.0, 0.0, 0.0])
        a_t, b_t, kappa_t, c_t = jet_gen_forward(x_t, gen_params, r)
        exo_norms.append(float(jnp.linalg.norm(jnp.concatenate([a_t, b_t, kappa_t, c_t]))))
        h_next, s_next, g_contrib = reduced_step_fn(h, s, theta, x_t)
        if not bool(jnp.all(jnp.isfinite(h_next))):
            n_nonfinite += 1
            h_norms.append(float("nan")); y_norms.append(float("nan"))
            s_norms.append(float("nan")); g_norms.append(float("nan"))
            h, s = h_next, s_next
            continue
        h_norms.append(float(jnp.linalg.norm(h_next)))
        s_norms.append(float(jnp.linalg.norm(s_next)))
        g_norms.append(float(jnp.linalg.norm(g_contrib)))
        h, s = h_next, s_next
    h_norms = np.array(h_norms); s_norms = np.array(s_norms)
    finite_mask = np.isfinite(h_norms)
    theta_np = np.asarray(theta)
    scalar_frac = float(theta_np[0] ** 2 / (np.sum(theta_np ** 2) + 1e-18))
    h_final = np.asarray(h)
    h_scalar_frac = float(h_final[0] ** 2 / (np.sum(h_final ** 2) + 1e-18)) if np.all(np.isfinite(h_final)) else float("nan")
    print(f"    [{tag}] r={r:4d}  max|h_t|={np.nanmax(h_norms):.3e}  median|h_t|={np.nanmedian(h_norms[finite_mask]) if finite_mask.any() else float('nan'):.3e}  "
          f"max|s_t|={np.nanmax(s_norms):.3e}  n_nonfinite_steps={n_nonfinite}/{T}  "
          f"theta_scalar_frac={scalar_frac:.4f}  h_final_scalar_frac={h_scalar_frac:.4f}  "
          f"exo_norm(median)={np.median(exo_norms):.3e}")
    return dict(h_norms=h_norms, s_norms=s_norms, n_nonfinite=n_nonfinite)


def run_scale_diagnostic():
    print("\n" + "=" * 78)
    print("STEP 2: scale/stability diagnostic (B34->B only), at init and after a few training steps")
    print("=" * 78)
    for r in R_LEVELS:
        gen_seed = 3000 + r
        gen_params = jet_make_gen_params(seed=gen_seed, r=r)
        teacher, _ = make_teacher_B_jet(seed=778, r=r, gen_seed=gen_seed)
        reduced_step = make_jet_reduced_step(gen_params, r)
        step = make_jet_step(gen_params, r)
        rollout = make_rollout(step)
        loss_fn = make_loss_decoupled(rollout)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

        h0_t, xs = make_sequence(20_000, T_SEQ, r)
        targets = teacher.targets(h0_t, xs)

        print(f"\n  --- r={r} ---")
        # (a) at the student's OWN random init (theta ~ N(0,0.2), the
        # actual jet_make_theta convention used throughout).
        theta_init = jet_make_theta(1000, r)
        diagnose_rollout(step, reduced_step, theta_init, gen_params, jnp.zeros(r, dtype=jnp.float64),
                          xs, r, teacher.W, targets, "random init")

        # (b) at the teacher's OWN theta_star (a reference "well-behaved"
        # point, since the teacher itself is stable by construction --
        # b34a's own correctness suite tracks max|h_t| for stability).
        diagnose_rollout(step, reduced_step, teacher.params, gen_params, jnp.zeros(r, dtype=jnp.float64),
                          xs, r, teacher.W, targets, "teacher theta* (zero h0)")

        # (c) raw (PRE-CLIP) gradient norm over a few early Adam steps
        # from the student's own random init -- the actual View-2 training
        # loop, but instrumented to report the gradient BEFORE clip_grad.
        params = make_student_params(lambda seed: jet_make_theta(seed, r), r, 1000)
        opt_state = adam_init(params)
        h0_student = jnp.zeros(r, dtype=jnp.float64)
        raw_grad_norms = []
        for step_i in range(8):
            h0_i, xs_i = make_sequence(20_000 + step_i, T_SEQ, r)
            targets_i = teacher.targets(h0_i, xs_i)
            g = grad_fn(params, h0_student, xs_i, targets_i)
            flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)])
            raw_norm = float(jnp.linalg.norm(flat))
            raw_grad_norms.append(raw_norm)
            g_clipped = clip_grad(g)
            params, opt_state = adam_step(params, g_clipped, opt_state, lr=0.01)
        print(f"    [raw grad norm, first 8 Adam steps, lr=0.01] {['%.3e' % v for v in raw_grad_norms]}")


if __name__ == "__main__":
    run_containment_check()
    run_scale_diagnostic()
