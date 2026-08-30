"""B35f -- RegularBlock continual-optimization mechanism audit.
Diagnostic only: does not alter B35d/B35e, the architectures, optimizer,
LR, clipping, or projections. Reuses the FROZEN B35d evaluation setup
(same LRs as selected there, same eval seeds) and instruments the exact
same per-step update equations purely to RECORD quantities B35d's own
loop discards.

Hypothesis under test: RegularBlock's generalized/Jordan dynamics
produce larger transient eligibility/gradient amplification under
single-sample learning than GenericBlock.

Run: python -m credit_memory.b35f_optimization_mechanism_audit
"""
from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b34a_jet_algebra_correctness import make_M as make_M_local
from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, transpose_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.b35c_matched_credit_frontier import regular_config, generic_config
from credit_memory.b35d_streaming_sysid import (
    init_regular_streaming, init_generic_streaming, make_teacher_trajectory,
    clip_vec, project_generic_theta, CLIP_NORM, T_TOTAL, EVAL_SEEDS,
)
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step

# FROZEN B35d-selected LRs (not re-selected here) -- from PHASE_B35D.md section 5.
FROZEN_LR = dict(regular={128: 0.01, 64: 0.02}, generic={128: 0.05, 64: 0.01})
GAMMA_H = 50
GAMMA_EVERY = 150   # sparse checkpoints for the (relatively) expensive Gamma_H diagnostic


def gamma_H_regular(theta, Q, d, H=GAMMA_H):
    Theta = np.asarray(theta).reshape(Q, d)
    gammas, rhos = np.zeros(Q), np.zeros(Q)
    for q in range(Q):
        Mq = np.asarray(make_M_local(jnp.array(Theta[q]), d))
        rhos[q] = np.max(np.abs(np.linalg.eigvals(Mq)))
        Mk = np.eye(d)
        local_max = 0.0
        for k in range(1, H + 1):
            Mk = Mk @ Mq
            local_max = max(local_max, np.linalg.norm(Mk, ord=2))
        gammas[q] = local_max
    return gammas, rhos


def gamma_H_generic(theta, basis, Q, d, p, H=GAMMA_H):
    Theta = np.asarray(theta).reshape(Q, p)
    basis_np = np.asarray(basis)
    gammas, rhos = np.zeros(Q), np.zeros(Q)
    for q in range(Q):
        Aq = np.einsum("k,kij->ij", Theta[q], basis_np[q])
        rhos[q] = np.max(np.abs(np.linalg.eigvals(Aq)))
        Mk = np.eye(d)
        local_max = 0.0
        for k in range(1, H + 1):
            Mk = Mk @ Aq
            local_max = max(local_max, np.linalg.norm(Mk, ord=2))
        gammas[q] = local_max
    return gammas, rhos


def instrumented_regular_run(C, seed, T_total=T_TOTAL):
    lr = FROZEN_LR["regular"][C]
    rc = regular_config(C)
    Q, d = rc["Q"], rc["d"]
    carry, B_in = init_regular_streaming(Q, d, seed=1000 + seed)
    h, s, theta, C_out, opt_th, opt_co = carry
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_total)

    records, per_factor, gamma_records = [], [], []
    for t in range(T_total):
        x_t, u_t, x_next = xs_t[t], us[t], xs_next[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * float(jnp.sum(diff ** 2))
        dl_dh = C_out.T @ diff

        g_raw = transpose_mult_blockwise(s_next, dl_dh, Q, d)
        raw_norm = float(jnp.linalg.norm(g_raw))
        clipped = raw_norm > CLIP_NORM
        clip_factor = min(1.0, CLIP_NORM / (raw_norm + 1e-12))
        g_theta = clip_vec(g_raw)

        theta_before = theta
        theta_after_adam, opt_th = adam_step(theta, g_theta, opt_th, lr)
        adam_delta = float(jnp.linalg.norm(theta_after_adam - theta_before))
        adam_relative = adam_delta / (float(jnp.linalg.norm(theta_before)) + 1e-12)

        theta_new = project_local_tails(theta_after_adam, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
        proj_correction = float(jnp.linalg.norm(theta_new - theta_after_adam))

        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)

        Theta_resh = np.asarray(theta).reshape(Q, d)
        base = np.abs(Theta_resh[:, 0])
        tail_norm = np.sum(np.abs(Theta_resh[:, 1:]), axis=1) if d > 1 else np.zeros(Q)
        elig_pf = np.linalg.norm(np.asarray(s_next).reshape(Q, d), axis=1)
        grad_pf = np.linalg.norm(np.asarray(g_raw).reshape(Q, d), axis=1)

        records.append(dict(t=t, loss=loss, h_norm=float(jnp.linalg.norm(h_next)),
                             elig_norm=float(jnp.linalg.norm(s_next)), unclipped_grad_norm=raw_norm,
                             clipped=clipped, clip_factor=clip_factor, adam_relative=adam_relative,
                             proj_correction=proj_correction))
        per_factor.append(dict(base=base, tail_norm=tail_norm, elig_norm=elig_pf, grad_norm=grad_pf))
        if t % GAMMA_EVERY == 0:
            gammas, rhos = gamma_H_regular(theta, Q, d)
            gamma_records.append(dict(t=t, gammas=gammas, rhos=rhos))

        h, s, theta = h_next, s_next, theta_new
    return records, per_factor, gamma_records


def instrumented_generic_run(C, seed, T_total=T_TOTAL):
    lr = FROZEN_LR["generic"][C]
    gc = generic_config(C)
    Q, d, p = gc["Q"], gc["d"], gc["p"]
    carry, basis, B_in = init_generic_streaming(Q, d, p, seed=1000 + seed)
    h, S, theta, C_out, opt_th, opt_co = carry
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_total)

    records, gamma_records = [], []
    for t in range(T_total):
        x_t, u_t, x_next = xs_t[t], us[t], xs_next[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        H_ = h.reshape(Q, d)
        Theta = theta.reshape(Q, p)
        A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
        H_next = jnp.einsum("qij,qj->qi", A_batch, H_) + (B_in @ model_in).reshape(Q, d)
        G_batch = jnp.einsum("qkij,qj->qik", basis, H_)
        S_next = jnp.einsum("qij,qjk->qik", A_batch, S) + G_batch
        h_next = H_next.reshape(Q * d)
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * float(jnp.sum(diff ** 2))
        dl_dh = (C_out.T @ diff).reshape(Q, d)

        g_raw = jnp.einsum("qd,qdk->qk", dl_dh, S_next).reshape(-1)
        raw_norm = float(jnp.linalg.norm(g_raw))
        clipped = raw_norm > CLIP_NORM
        clip_factor = min(1.0, CLIP_NORM / (raw_norm + 1e-12))
        g_theta = clip_vec(g_raw)

        theta_before = theta
        theta_after_adam, opt_th = adam_step(theta, g_theta, opt_th, lr)
        adam_delta = float(jnp.linalg.norm(theta_after_adam - theta_before))
        adam_relative = adam_delta / (float(jnp.linalg.norm(theta_before)) + 1e-12)

        theta_new = project_generic_theta(theta_after_adam, basis, Q, d, p)
        proj_correction = float(jnp.linalg.norm(theta_new - theta_after_adam))

        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)

        records.append(dict(t=t, loss=loss, h_norm=float(jnp.linalg.norm(h_next)),
                             elig_norm=float(jnp.linalg.norm(S_next.reshape(-1))), unclipped_grad_norm=raw_norm,
                             clipped=clipped, clip_factor=clip_factor, adam_relative=adam_relative,
                             proj_correction=proj_correction))
        if t % GAMMA_EVERY == 0:
            gammas, rhos = gamma_H_generic(theta, basis, Q, d, p)
            gamma_records.append(dict(t=t, gammas=gammas, rhos=rhos))

        h, S, theta = h_next, S_next, theta_new
    return records, gamma_records


def pct_report(name, arr):
    arr = np.asarray(arr)
    print(f"    {name}: median={np.median(arr):.4e}  p90={np.percentile(arr,90):.4e}  "
          f"p99={np.percentile(arr,99):.4e}  max={np.max(arr):.4e}")


def run_audit(C=128):
    print("=" * 78)
    print(f"B35f optimization-mechanism audit, C={C}, FROZEN LR: regular={FROZEN_LR['regular'][C]} "
          f"generic={FROZEN_LR['generic'][C]}, EVAL_SEEDS={EVAL_SEEDS}")
    print("=" * 78)

    reg_all, gen_all = [], []
    reg_pf_all, reg_gamma_all, gen_gamma_all = [], [], []
    for seed in EVAL_SEEDS:
        r_rec, r_pf, r_gamma = instrumented_regular_run(C, seed)
        g_rec, g_gamma = instrumented_generic_run(C, seed)
        reg_all.extend(r_rec); reg_pf_all.extend(r_pf); reg_gamma_all.extend(r_gamma)
        gen_all.extend(g_rec); gen_gamma_all.extend(g_gamma)

    for name, recs in [("RegularBlock", reg_all), ("GenericBlock", gen_all)]:
        print(f"\n--- {name}, distributions across all seeds x steps (n={len(recs)}) ---")
        pct_report("loss", [r["loss"] for r in recs])
        pct_report("h_norm", [r["h_norm"] for r in recs])
        pct_report("elig_norm", [r["elig_norm"] for r in recs])
        pct_report("unclipped_grad_norm", [r["unclipped_grad_norm"] for r in recs])
        n_clipped = sum(1 for r in recs if r["clipped"])
        print(f"    clipping activated: {n_clipped}/{len(recs)} ({100*n_clipped/len(recs):.2f}%)  "
              f"mean_clip_factor_when_active="
              f"{np.mean([r['clip_factor'] for r in recs if r['clipped']]) if n_clipped else float('nan'):.4f}")
        pct_report("adam_relative_step", [r["adam_relative"] for r in recs])
        pct_report("proj_correction", [r["proj_correction"] for r in recs])

    print(f"\n--- RegularBlock per-factor (n={len(reg_pf_all)} step-records x Q factors each) ---")
    all_base = np.concatenate([pf["base"] for pf in reg_pf_all])
    all_tail = np.concatenate([pf["tail_norm"] for pf in reg_pf_all])
    all_elig_pf = np.concatenate([pf["elig_norm"] for pf in reg_pf_all])
    all_grad_pf = np.concatenate([pf["grad_norm"] for pf in reg_pf_all])
    pct_report("base |lambda_q|", all_base)
    pct_report("nilpotent-tail L1 norm", all_tail)
    pct_report("per-factor eligibility norm", all_elig_pf)
    pct_report("per-factor gradient norm", all_grad_pf)

    print(f"\n--- Gamma_H={GAMMA_H} transient-gain diagnostic (sparse checkpoints, per-factor) ---")
    for name, gamma_all in [("RegularBlock", reg_gamma_all), ("GenericBlock", gen_gamma_all)]:
        all_gammas = np.concatenate([g["gammas"] for g in gamma_all])
        all_rhos = np.concatenate([g["rhos"] for g in gamma_all])
        print(f"  {name}:")
        pct_report(f"  rho(J) (spectral radius)", all_rhos)
        pct_report(f"  Gamma_H={GAMMA_H} (finite-horizon transient gain)", all_gammas)
        frac_exceed = np.mean(all_gammas > 1.0)
        print(f"    fraction of (checkpoint,factor) with Gamma_H > 1 despite rho(J)<1 possibly: "
              f"{frac_exceed:.4f}  (rho<1 nearly always by construction of base-clip; "
              f"Gamma_H>1 shows nonnormal/Jordan transient amplification beyond what rho alone predicts)")

    # Correlation: does regular's loss correlate with elig_norm / grad_norm / clipping / proj_correction?
    print("\n--- Correlations with per-step loss (RegularBlock) ---")
    losses = np.array([r["loss"] for r in reg_all])
    for key in ("elig_norm", "unclipped_grad_norm", "proj_correction", "adam_relative"):
        vals = np.array([r[key] for r in reg_all])
        corr = np.corrcoef(losses, vals)[0, 1]
        print(f"    corr(loss, {key}) = {corr:.4f}")
    clip_flags = np.array([1.0 if r["clipped"] else 0.0 for r in reg_all])
    print(f"    corr(loss, clipped_flag) = {np.corrcoef(losses, clip_flags)[0,1]:.4f}")

    return dict(reg_all=reg_all, gen_all=gen_all, reg_pf_all=reg_pf_all,
                reg_gamma_all=reg_gamma_all, gen_gamma_all=gen_gamma_all)


if __name__ == "__main__":
    results_128 = run_audit(C=128)
    results_64 = run_audit(C=64)
