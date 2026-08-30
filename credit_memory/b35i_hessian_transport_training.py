"""B35i -- train the genuinely online Hessian-transported ProductLocal
learner. Architecture, optimizer, projection, teacher, data stream, and
the B35d protocol (LR, EVAL_SEEDS, T_TOTAL, T_CHANGE, windows) are all
UNCHANGED; only the state/eligibility TRANSPORT applied after each
parameter update differs across variants.

Indexing convention (carefully separated from B35g/B35h): at step t,
carry=(h,s,r,theta,C_out,opt_th,opt_co) enters with h=h_t, s=s_t, r=r_t,
theta=theta_t (the parameter ACTUALLY used this step).
  h_next = alg_mult(theta,h,Q,d)+B_in@in_t        (=h_{t+1})
  s_next = alg_mult(theta,s,Q,d)+h                 (=s_{t+1}, uses OLD h)
  r_next = alg_mult(theta,r,Q,d)+2*s                (=r_{t+1}, uses OLD s)
  ... compute loss/gradient from h_next, s_next ...
  theta_after_opt = adam_step(theta, g_theta, ...)
  theta_new = project(theta_after_opt)
  Delta_t := theta_new - theta   (the TOTAL realized parameter change --
    optimizer step AND any projection correction combined, so h/s are
    re-centered around the parameter ACTUALLY used going forward, not
    a pre-projection candidate that never actually gets used)
  A0: s^+ = s_next,                         h^+ = h_next
  A1: s^+ = s_next + alg_mult(r_next,Delta_t),  h^+ = h_next  (untransported, deliberate ablation)
  A2: s^+ = s_next + alg_mult(r_next,Delta_t),
      h^+ = h_next + alg_mult(s_next,Delta_t) + 0.5*alg_mult(r_next, alg_mult(Delta_t,Delta_t))
  r is carried forward as r_next UNCHANGED (no correction to r itself,
  per instruction); next step's carry is (h^+, s^+, r_next, theta_new,...).

Run: python -m credit_memory.b35i_hessian_transport_training
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, transpose_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.b35c_matched_credit_frontier import regular_config
from credit_memory.b35d_streaming_sysid import (
    init_regular_streaming, make_teacher_trajectory, clip_vec, CLIP_NORM,
    T_TOTAL, T_CHANGE, EVAL_SEEDS, compute_metrics,
)
from credit_memory.b35h_hessian_transport_diagnostic import make_rollout_fn, make_M
from credit_memory.b35h_online_2P_carry import reduced_s_and_r
from credit_memory.b35e_staleness_diagnostic import record_regular_trajectory
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, DIVERGENCE_LOSS_CEIL

C_BUDGET = 128
FROZEN_LR = {128: 0.01, 64: 0.02}   # from PHASE_B35D.md section 5, unchanged


def make_step_fn(Q, d, B_in, lr_theta, lr_cout, variant):
    def step(carry, inputs):
        h, s, r, theta, C_out, opt_th, opt_co = carry
        x_t, u_t, x_next = inputs
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])

        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        r_next = alg_mult_blockwise(theta, r, Q, d) + 2 * s

        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * jnp.sum(diff ** 2)
        dl_dh = C_out.T @ diff
        g_theta = clip_vec(transpose_mult_blockwise(s_next, dl_dh, Q, d))
        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)

        theta_after_opt, opt_th_new = adam_step(theta, g_theta, opt_th, lr_theta)
        theta_new = project_local_tails(theta_after_opt, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
        Delta_t = theta_new - theta   # TOTAL realized change (optimizer + projection)

        if variant == "A0":
            s_plus, h_plus = s_next, h_next
        elif variant == "A1":
            s_plus = s_next + alg_mult_blockwise(r_next, Delta_t, Q, d)
            h_plus = h_next
        elif variant == "A2":
            s_plus = s_next + alg_mult_blockwise(r_next, Delta_t, Q, d)
            Delta_sq = alg_mult_blockwise(Delta_t, Delta_t, Q, d)
            h_plus = h_next + alg_mult_blockwise(s_next, Delta_t, Q, d) + \
                0.5 * alg_mult_blockwise(r_next, Delta_sq, Q, d)
        else:
            raise ValueError(variant)

        C_out_new, opt_co_new = adam_step(C_out, g_cout, opt_co, lr_cout)

        finite = jnp.all(jnp.isfinite(h_plus)) & jnp.isfinite(loss)
        delta_norm = jnp.linalg.norm(Delta_t)
        s_corr_norm = jnp.linalg.norm(s_plus - s_next)
        h_corr_norm = jnp.linalg.norm(h_plus - h_next)
        new_carry = (h_plus, s_plus, r_next, theta_new, C_out_new, opt_th_new, opt_co_new)
        return new_carry, (loss, x_hat, finite, delta_norm, s_corr_norm, h_corr_norm,
                            jnp.linalg.norm(s_next), jnp.linalg.norm(h_next))
    return step


def init_carry(Q, d, seed):
    carry2, B_in = init_regular_streaming(Q, d, seed=1000 + seed)
    h, s, theta, C_out, opt_th, opt_co = carry2
    r = jnp.zeros(Q * d, dtype=jnp.float64)
    return (h, s, r, theta, C_out, opt_th, opt_co), B_in


def run_scanned(C, seed, variant, T_total=T_TOTAL):
    rc = regular_config(C)
    Q, d = rc["Q"], rc["d"]
    lr = FROZEN_LR[C]
    carry, B_in = init_carry(Q, d, seed)
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_total)
    step_fn = make_step_fn(Q, d, B_in, lr, lr, variant)
    scan_fn = jax.jit(lambda carry, inputs: jax.lax.scan(step_fn, carry, inputs))
    final_carry, (losses, x_hats, finites, delta_norms, s_corr_norms, h_corr_norms,
                  s_norms, h_norms) = scan_fn(carry, (xs_t, us, xs_next))
    return dict(losses=losses, x_hats=x_hats, finites=finites, delta_norms=delta_norms,
                s_corr_norms=s_corr_norms, h_corr_norms=h_corr_norms, s_norms=s_norms,
                h_norms=h_norms, xs_next=xs_next)


# =======================================================================
# Unit test: verify O(||Delta||^2) / O(||Delta||^3) scaling for the A2
# transport formula against brute-force replay, using the REAL
# optimizer direction from one representative early step.
# =======================================================================
def unit_test_scaling(C=C_BUDGET, seed=EVAL_SEEDS[0], step_index=20):
    """Verifies the TAYLOR-EXPANSION property of the transport formula:
    S_t^replay(theta_t+Delta) vs S_t(theta_t)+H_t[Delta], using the
    PROPER fixed-theta_t replay reference (matching B35h exactly) --
    NOT the carried/moving-parameter trace (which is contaminated by
    prior staleness and is a different object; comparing against it
    would be a category error, not a test of the transport formula).
    Delta is the REAL optimizer update extracted from one actual
    continual-training step at this checkpoint (A0 trajectory)."""
    print("=" * 78)
    print(f"Unit test: A2 transport scaling vs brute-force replay (C={C}, seed={seed}, step={step_index})")
    print("=" * 78)
    lr = FROZEN_LR[C]
    rec = record_regular_trajectory(C=C, seed=seed, lr=lr, update_interval=1, T_record=step_index + 5)
    Q, d, B_in, h0 = rec["Q"], rec["d"], rec["B_in"], rec["h0"]
    theta_t = rec["theta_hist"][step_index]        # REALIZED moving-parameter value at this checkpoint
    xs_seg, us_seg = rec["xs_t"][:step_index], rec["us"][:step_index]

    # FIXED-THETA replay reference at theta_t (the correct S_t(theta_t), h_t(theta_t))
    h_ref, s_ref, r_ref = reduced_s_and_r(theta_t, xs_seg, us_seg, h0, Q, d, B_in)

    # extract a REAL Delta_t: one more training step from this SAME realized
    # trajectory's own (carried) state, exactly as the training loop does.
    h_c, s_c, r_c = rec["h_hist"][step_index], rec["s_hist"][step_index], None
    # r isn't tracked by record_regular_trajectory; recompute it via the SAME
    # moving-path recursion (not needed for Delta_t itself, only s/h/theta are).
    x_t, u_t, x_next = rec["xs_t"][step_index], rec["us"][step_index], rec["target_hist"][step_index]
    model_in = jnp.concatenate([x_t, jnp.array([u_t])])
    h_next_c = alg_mult_blockwise(theta_t, h_c, Q, d) + B_in @ model_in
    s_next_c = alg_mult_blockwise(theta_t, s_c, Q, d) + h_c
    C_out_t = rec["cout_hist"][step_index]
    diff_c = C_out_t @ h_next_c - x_next
    dl_dh_c = C_out_t.T @ diff_c
    g_theta_c = clip_vec(transpose_mult_blockwise(s_next_c, dl_dh_c, Q, d))
    opt_th_dummy = adam_init(theta_t)
    theta_after_opt, _ = adam_step(theta_t, g_theta_c, opt_th_dummy, lr)
    theta_new_real = project_local_tails(theta_after_opt, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
    Delta_real = theta_new_real - theta_t

    # replay function: FULL history from h_0 through step_index inputs.
    rollout = make_rollout_fn(xs_seg, us_seg, h0, Q, d, B_in)

    etas = (0.25, 0.5, 1.0, 2.0, 4.0)   # scaled around the REAL |Delta_real| (eta=1 => actual step)
    S_norm, h_err_norm = [], []
    for eta in etas:
        Delta_scaled = eta * Delta_real
        theta_pert = theta_t + Delta_scaled
        S_replay = jax.jacobian(rollout)(theta_pert)
        h_replay = rollout(theta_pert)

        Delta_sq = alg_mult_blockwise(Delta_scaled, Delta_scaled, Q, d)
        s_plus = s_ref + alg_mult_blockwise(r_ref, Delta_scaled, Q, d)
        h_plus = h_ref + alg_mult_blockwise(s_ref, Delta_scaled, Q, d) + \
            0.5 * alg_mult_blockwise(r_ref, Delta_sq, Q, d)
        S_plus_matrix = make_M(s_plus, Q, d)
        S_norm.append(float(jnp.linalg.norm(S_replay - S_plus_matrix)))
        h_err_norm.append(float(jnp.linalg.norm(h_replay - h_plus)))

    def slope(etas, norms):
        e, n = np.abs(np.array(etas)), np.array(norms)
        mask = n > 1e-15
        return np.polyfit(np.log(e[mask]), np.log(n[mask]), 1)[0] if mask.sum() > 1 else float("nan")

    print(f"  |Delta_real| = {float(jnp.linalg.norm(Delta_real)):.4e}")
    for eta, sN, hN in zip(etas, S_norm, h_err_norm):
        print(f"    eta={eta:.2f}  ||S_replay-S+||={sN:.4e}  ||h_replay-h+||={hN:.4e}")
    print(f"  slope ||S_replay-S+|| vs eta: {slope(etas,S_norm):.3f}  (predict 2)")
    print(f"  slope ||h_replay-h+|| vs eta: {slope(etas,h_err_norm):.3f}  (predict 3)")
    return slope(etas, S_norm), slope(etas, h_err_norm)


def report_variant(name, C, seeds=EVAL_SEEDS):
    print(f"\n--- {name}, C={C} ---")
    all_metrics = []
    all_delta, all_scorr, all_hcorr, all_snorm, all_hnorm, all_losses = [], [], [], [], [], []
    for seed in seeds:
        out = run_scanned(C, seed, name)
        m = compute_metrics(out["losses"], out["x_hats"], out["finites"], out["xs_next"])
        all_metrics.append(m)
        finite_mask = np.asarray(out["finites"])
        all_delta.append(np.asarray(out["delta_norms"])[finite_mask])
        all_scorr.append(np.asarray(out["s_corr_norms"])[finite_mask])
        all_hcorr.append(np.asarray(out["h_corr_norms"])[finite_mask])
        all_snorm.append(np.asarray(out["s_norms"])[finite_mask])
        all_hnorm.append(np.asarray(out["h_norms"])[finite_mask])
        all_losses.append(np.asarray(out["losses"])[finite_mask])
        print(f"    seed={seed}: pre_nmse={m['pre_nmse']}  post_nmse={m['post_nmse']}  "
              f"diverged={m['diverged']}  n_nonfinite={m['n_nonfinite']}  "
              f"steps_to_recover={m['steps_to_recover']}")
    pre_vals = [m["pre_nmse"] for m in all_metrics if m["pre_nmse"] is not None]
    post_vals = [m["post_nmse"] for m in all_metrics if m["post_nmse"] is not None]
    delta_all = np.concatenate(all_delta) if all_delta else np.array([])
    scorr_all = np.concatenate(all_scorr) if all_scorr else np.array([])
    hcorr_all = np.concatenate(all_hcorr) if all_hcorr else np.array([])
    snorm_all = np.concatenate(all_snorm) if all_snorm else np.array([])
    hnorm_all = np.concatenate(all_hnorm) if all_hnorm else np.array([])
    loss_all = np.concatenate(all_losses) if all_losses else np.array([])
    loss_std = float(np.std(loss_all)) if loss_all.size else float("nan")
    print(f"    ACROSS SEEDS: pre_nmse median={np.median(pre_vals):.4e}  post_nmse median={np.median(post_vals):.4e}  "
          f"loss_volatility(std)={loss_std:.4e}")
    if delta_all.size:
        print(f"    update-norm ||Delta_t||: median={np.median(delta_all):.4e}  p90={np.percentile(delta_all,90):.4e}")
    if scorr_all.size:
        rel_s = scorr_all / (snorm_all + 1e-12)
        rel_h = hcorr_all / (hnorm_all + 1e-12)
        print(f"    ||correction_s||/||s||: median={np.median(rel_s):.4e}  p90={np.percentile(rel_s,90):.4e}")
        print(f"    ||correction_h||/||h||: median={np.median(rel_h):.4e}  p90={np.percentile(rel_h,90):.4e}")
    n_div = sum(1 for m in all_metrics if m["diverged"])
    return dict(pre_median=float(np.median(pre_vals)) if pre_vals else None,
                post_median=float(np.median(post_vals)) if post_vals else None,
                loss_std=loss_std, n_diverged=n_div)


if __name__ == "__main__":
    unit_test_scaling()
    print()
    summary = {}
    for variant in ("A0", "A1", "A2"):
        summary[variant] = report_variant(variant, C_BUDGET)
    print("\n" + "=" * 78)
    print("STAGE A SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k}: pre_median={v['pre_median']}  post_median={v['post_median']}")
