"""B35g -- optimizer/projection interaction diagnostic. Architecture,
credit recurrence (s_{t+1}=alg_mult(theta,s_t)+h_t), B35d's result, and
B35f's interpretation are all UNCHANGED. Compares three TRAINING-RULE
variants for RegularBlock's theta update only (C_out keeps Adam
throughout, isolating the test to theta):

  A. baseline: Adam + existing hard per-factor projection (B35d as-is,
     frozen LR).
  B. Adam + projection-aware moment reset: whenever a factor's
     projection correction is nonzero (materially projected), reset
     that factor's Adam first/second moments to zero immediately after
     the projection.
  C. momentum-free normalized SGD (theta -= lr * g/||g||) + the SAME
     projection, LR selected only on dedicated tuning seeds (matching
     B35d's own selection protocol).

Falsifiable prediction: if Adam/projection state mismatch drives B35d's
instability, B or C should substantially reduce RegularBlock's loss
oscillation despite the architecture/Jordan dynamics/eligibility/
stability boundary being identical. If neither helps, reject the
mechanism.

Run: python -m credit_memory.b35g_optimizer_projection_diagnostic
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, transpose_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.b35c_matched_credit_frontier import regular_config
from credit_memory.b35d_streaming_sysid import (
    init_regular_streaming, make_teacher_trajectory, clip_vec, CLIP_NORM,
    T_TOTAL, EVAL_SEEDS, compute_metrics,
)
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, DIVERGENCE_LOSS_CEIL

FROZEN_LR_ADAM = dict(regular={128: 0.01, 64: 0.02})
SGD_LR_GRID = (0.01, 0.02, 0.05)
TUNING_SEEDS = (100, 101)
SELECTION_WINDOW = (200, 800)
PROJ_MATERIAL_THRESHOLD = 1e-9


def reset_adam_moments_for_factors(opt_state, factor_mask, Q, d):
    m = opt_state["m"].reshape(Q, d)
    v = opt_state["v"].reshape(Q, d)
    m = jnp.where(factor_mask[:, None], 0.0, m)
    v = jnp.where(factor_mask[:, None], 0.0, v)
    return dict(m=m.reshape(-1), v=v.reshape(-1), t=opt_state["t"])


def run_regular_variant(C, seed, lr, variant, T_total=T_TOTAL):
    rc = regular_config(C)
    Q, d = rc["Q"], rc["d"]
    carry, B_in = init_regular_streaming(Q, d, seed=1000 + seed)
    h, s, theta, C_out, opt_th, opt_co = carry
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_total)

    records = []
    losses, x_hats, finites = [], [], []
    consecutive_run = 0
    max_consecutive_run = 0
    for t in range(T_total):
        x_t, u_t, x_next = xs_t[t], us[t], xs_next[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * float(jnp.sum(diff ** 2))
        finite = bool(jnp.all(jnp.isfinite(h_next))) and np.isfinite(loss) and loss <= DIVERGENCE_LOSS_CEIL
        losses.append(loss); x_hats.append(np.asarray(x_hat)); finites.append(finite)
        if not finite:
            break
        dl_dh = C_out.T @ diff
        g_theta = clip_vec(transpose_mult_blockwise(s_next, dl_dh, Q, d))

        theta_before = theta
        if variant in ("adam", "adam_reset"):
            theta_after_opt, opt_th = adam_step(theta, g_theta, opt_th, lr)
        elif variant == "norm_sgd":
            g_norm = jnp.linalg.norm(g_theta)
            theta_after_opt = theta - lr * g_theta / (g_norm + 1e-12)
        else:
            raise ValueError(variant)
        delta_optimizer = theta_after_opt - theta_before

        theta_new = project_local_tails(theta_after_opt, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
        c_t = theta_new - theta_after_opt

        proj_corr_norm = float(jnp.linalg.norm(c_t))
        delta_opt_norm = float(jnp.linalg.norm(delta_optimizer))
        correction_ratio = proj_corr_norm / (delta_opt_norm + 1e-12)
        cos_sim = (float(jnp.dot(delta_optimizer, c_t)) / (delta_opt_norm * proj_corr_norm + 1e-12)
                   if proj_corr_norm > 0 else None)

        c_resh = np.asarray(c_t).reshape(Q, d)
        factor_mask = jnp.array(np.linalg.norm(c_resh, axis=1) > PROJ_MATERIAL_THRESHOLD)
        any_projected = bool(jnp.any(factor_mask))
        if any_projected:
            consecutive_run += 1
            max_consecutive_run = max(max_consecutive_run, consecutive_run)
        else:
            consecutive_run = 0

        if variant == "adam_reset" and any_projected:
            opt_th = reset_adam_moments_for_factors(opt_th, factor_mask, Q, d)

        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)

        records.append(dict(t=t, loss=loss, any_projected=any_projected, proj_corr_norm=proj_corr_norm,
                             delta_opt_norm=delta_opt_norm, correction_ratio=correction_ratio,
                             cos_sim=cos_sim, consecutive_run=consecutive_run))
        h, s, theta = h_next, s_next, theta_new

    metrics = compute_metrics(jnp.array(losses), jnp.array(x_hats), jnp.array(finites), xs_next)
    return records, max_consecutive_run, metrics


def select_lr_norm_sgd(C, lr_grid=SGD_LR_GRID, tuning_seeds=TUNING_SEEDS):
    scores = {}
    for lr in lr_grid:
        seed_scores = []
        for seed in tuning_seeds:
            recs, _, _ = run_regular_variant(C, seed, lr, "norm_sgd")
            a, b = SELECTION_WINDOW
            seg = [r["loss"] for r in recs[a:b]]
            seed_scores.append(np.mean(seg) if seg else float("inf"))
        scores[lr] = float(np.mean(seed_scores))
    return min(scores, key=scores.get), scores


def summarize_variant(name, all_records, all_metrics, all_max_runs):
    print(f"\n--- {name} ---")
    for m, seed in zip(all_metrics, EVAL_SEEDS):
        print(f"    seed={seed}: pre_nmse={m['pre_nmse']}  post_nmse={m['post_nmse']}  "
              f"diverged={m['diverged']}  n_nonfinite={m['n_nonfinite']}  "
              f"n_steps_completed={m['n_steps_completed']}  cum_loss={m['cum_loss']:.3e}")
    pre_vals = [m["pre_nmse"] for m in all_metrics if m["pre_nmse"] is not None]
    post_vals = [m["post_nmse"] for m in all_metrics if m["post_nmse"] is not None]
    print(f"    ACROSS SEEDS: pre_nmse median={np.median(pre_vals):.4e} mean={np.mean(pre_vals):.4e} "
          f"std={np.std(pre_vals):.4e}  |  post_nmse median={np.median(post_vals):.4e} "
          f"mean={np.mean(post_vals):.4e} std={np.std(post_vals):.4e}")

    all_recs_flat = [r for recs in all_records for r in recs]
    n_proj = sum(1 for r in all_recs_flat if r["any_projected"])
    print(f"    projection-event frequency: {n_proj}/{len(all_recs_flat)} ({100*n_proj/len(all_recs_flat):.3f}%)")
    print(f"    max consecutive-projection run (chattering) across seeds: {max(all_max_runs)}")

    ratios = [r["correction_ratio"] for r in all_recs_flat if r["any_projected"]]
    cosines = [r["cos_sim"] for r in all_recs_flat if r["cos_sim"] is not None]
    delta_opt_norms = [r["delta_opt_norm"] for r in all_recs_flat]
    if ratios:
        print(f"    correction_ratio (||c_t||/||delta_opt||) when projected: median={np.median(ratios):.3f}  "
              f"p90={np.percentile(ratios,90):.3f}  p99={np.percentile(ratios,99):.3f}  max={np.max(ratios):.3f}")
    if cosines:
        print(f"    cos(delta_optimizer, c_t) when projected: median={np.median(cosines):.4f}  "
              f"mean={np.mean(cosines):.4f}  frac_negative={np.mean(np.array(cosines)<0):.4f}")
    print(f"    optimizer-step norm (||delta_optimizer||): median={np.median(delta_opt_norms):.4e}  "
          f"p90={np.percentile(delta_opt_norms,90):.4e}  p99={np.percentile(delta_opt_norms,99):.4e}")

    losses = np.array([r["loss"] for r in all_recs_flat])
    print(f"    loss volatility (std of per-step loss across all seeds/steps): {np.std(losses):.4e}")

    # does large correction_ratio precede/coincide with loss spikes? correlate
    # correction_ratio (0 when not projected) against loss at the SAME step
    # and at t+1 (does a correction event predict the NEXT step's loss).
    ratio_all = np.array([r["correction_ratio"] if r["any_projected"] else 0.0 for r in all_recs_flat])
    corr_same = np.corrcoef(ratio_all, losses)[0, 1]
    if len(ratio_all) > 1:
        corr_next = np.corrcoef(ratio_all[:-1], losses[1:])[0, 1]
    else:
        corr_next = float("nan")
    print(f"    corr(correction_ratio_t, loss_t) = {corr_same:.4f}   corr(correction_ratio_t, loss_{{t+1}}) = {corr_next:.4f}")
    return dict(n_proj=n_proj, n_total=len(all_recs_flat), loss_std=float(np.std(losses)),
                pre_median=float(np.median(pre_vals)) if pre_vals else None,
                post_median=float(np.median(post_vals)) if post_vals else None)


def run_audit(C=128):
    print("=" * 78)
    print(f"B35g optimizer/projection interaction diagnostic, C={C}, EVAL_SEEDS={EVAL_SEEDS}")
    print("=" * 78)

    lr_adam = FROZEN_LR_ADAM["regular"][C]
    print(f"\nBaseline/reset-variant LR (frozen from B35d): {lr_adam}")
    best_sgd_lr, sgd_scores = select_lr_norm_sgd(C)
    print(f"norm_sgd LR selected on TUNING_SEEDS={TUNING_SEEDS}: {sgd_scores} -> best_lr={best_sgd_lr}")

    summary = {}
    for variant_name, variant_key, lr in [("A: Adam baseline", "adam", lr_adam),
                                            ("B: Adam + projection-aware moment reset", "adam_reset", lr_adam),
                                            ("C: momentum-free normalized SGD", "norm_sgd", best_sgd_lr)]:
        all_records, all_metrics, all_max_runs = [], [], []
        for seed in EVAL_SEEDS:
            recs, max_run, metrics = run_regular_variant(C, seed, lr, variant_key)
            all_records.append(recs); all_metrics.append(metrics); all_max_runs.append(max_run)
        summary[variant_name] = summarize_variant(variant_name, all_records, all_metrics, all_max_runs)

    print("\n" + "=" * 78)
    print("SUMMARY comparison")
    print("=" * 78)
    for name, s in summary.items():
        print(f"  {name}: pre_nmse_median={s['pre_median']}  post_nmse_median={s['post_median']}  "
              f"loss_std={s['loss_std']:.4e}  proj_freq={s['n_proj']}/{s['n_total']}")
    return summary


if __name__ == "__main__":
    run_audit(C=128)
