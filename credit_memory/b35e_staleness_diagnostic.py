"""B35e -- moving-parameter trace diagnostic ONLY. Does not modify
B35d, does not redesign RegularBlock, does not optimize performance.
Tests directly, on short recorded B35d-style continual-update
trajectories, the theory-audit prediction that the carried trace is
the derivative w.r.t. a COMMON perturbation of the realized (moving)
parameter path -- not the frozen-current counterfactual gradient.

Notation (checkpoint t = immediately after t realized online steps):
  theta_0..theta_{t-1}: REALIZED historical parameter values actually
    used at each step (theta_k active during step k).
  h_t, s_t: realized state / carried reduced eligibility AT checkpoint t
    (these are byte-identical to what B35d's own loop produces -- this
    script reimplements the same update equations, importing the same
    primitives, purely to RECORD history B35d's own loop discards).
  theta_t: the CURRENT parameter at the checkpoint (after t updates).
  C_out_t: FROZEN at its checkpoint value for this whole diagnostic
    (isolates the theta-staleness question; C_out's own possible
    staleness is a separate, non-tested mechanism).

  A. g_t^carried = E_t^T q_t = transpose_mult(s_t, q_t, Q, d),
     q_t = dl_t/dh_t at h_t against the target used when h_t was
     produced in the original run.
  B. g_t^path: replay theta_0..theta_{t-1} exactly as realized, but
     with a SHARED additive perturbation alpha added to every one of
     them; autodiff d l_t(alpha)/d alpha at alpha=0.
  C. g_t^frozen: replay the SAME input prefix from h_0, but using
     theta_t (current, single fixed value) at every historical step;
     ordinary single-fixed-parameter gradient (BPTT/exact RTRL, already
     validated machinery).

Run: python -m credit_memory.b35e_staleness_diagnostic
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import (
    alg_mult_blockwise, transpose_mult_blockwise, project_local_tails,
)
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.b35b2_generic_vs_regular import make_generic_params
from credit_memory.b35c_matched_credit_frontier import regular_config, generic_config
from credit_memory.b35d_streaming_sysid import (
    init_regular_streaming, init_generic_streaming, make_teacher_trajectory,
    clip_vec, project_generic_theta, X_DIM_TEACHER,
)
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step

C_BUDGET = 64
CHECKPOINTS = (50, 150, 300)
LR_GRID = (0.005, 0.02, 0.05)
UPDATE_INTERVAL_GRID = (1, 5)
T_RECORD = max(CHECKPOINTS) + 5
SEED = 11   # one representative B35d evaluation seed


# =======================================================================
# Record a REALIZED continual-update trajectory (RegularBlock), same
# update equations as b35d_streaming_sysid.make_regular_streaming_scan,
# reimplemented only to keep FULL history (theta_k, h_k, s_k) that the
# scanned version discards.
# =======================================================================
def record_regular_trajectory(C=C_BUDGET, seed=SEED, lr=0.02, update_interval=1, T_record=T_RECORD):
    rc = regular_config(C)
    Q, d = rc["Q"], rc["d"]
    carry, B_in = init_regular_streaming(Q, d, seed=1000 + seed)
    h, s, theta, C_out, opt_th, opt_co = carry
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_record + 2)

    theta_hist, h_hist, s_hist, cout_hist, target_hist = [theta], [h], [s], [C_out], []
    for k in range(T_record):
        x_k, u_k, x_next_k = xs_t[k], us[k], xs_next[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        x_hat = C_out @ h_next
        diff = x_hat - x_next_k
        dl_dh = C_out.T @ diff
        g_theta = clip_vec(transpose_mult_blockwise(s_next, dl_dh, Q, d))
        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        if k % update_interval == 0:
            theta_new, opt_th = adam_step(theta, g_theta, opt_th, lr)
            theta_new = project_local_tails(theta_new, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
            C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)
        else:
            theta_new = theta
        h, s, theta = h_next, s_next, theta_new
        target_hist.append(x_next_k)
        theta_hist.append(theta); h_hist.append(h); s_hist.append(s); cout_hist.append(C_out)

    return dict(theta_hist=theta_hist, h_hist=h_hist, s_hist=s_hist, cout_hist=cout_hist,
                target_hist=target_hist, xs_t=xs_t, us=us, B_in=B_in, Q=Q, d=d,
                h0=carry[0])


def regular_checkpoint_diagnostics(rec, t, eps=1e-8):
    Q, d, B_in, h0 = rec["Q"], rec["d"], rec["B_in"], rec["h0"]
    h_t, s_t, theta_t, C_out_t = rec["h_hist"][t], rec["s_hist"][t], rec["theta_hist"][t], rec["cout_hist"][t]
    target_t = rec["target_hist"][t - 1]
    q_t = C_out_t.T @ (C_out_t @ h_t - target_t)

    # A. carried
    g_carried = transpose_mult_blockwise(s_t, q_t, Q, d)

    # replay inputs [0..t-1]
    xs_seg = rec["xs_t"][:t]
    us_seg = rec["us"][:t]
    theta_seg = jnp.stack(rec["theta_hist"][:t])   # theta_0..theta_{t-1}

    def rollout_path(alpha):
        def step(h, inputs):
            theta_k, x_k, u_k = inputs
            model_in = jnp.concatenate([x_k, jnp.array([u_k])])
            h_next = alg_mult_blockwise(theta_k + alpha, h, Q, d) + B_in @ model_in
            return h_next, None
        h_final, _ = jax.lax.scan(step, h0, (theta_seg, xs_seg, us_seg))
        return h_final

    def loss_path(alpha):
        h_final = rollout_path(alpha)
        diff = C_out_t @ h_final - target_t
        return 0.5 * jnp.sum(diff ** 2)

    g_path = jax.grad(loss_path)(jnp.zeros(Q * d, dtype=jnp.float64))

    def rollout_frozen(theta_fixed):
        def step(h, inputs):
            x_k, u_k = inputs
            model_in = jnp.concatenate([x_k, jnp.array([u_k])])
            h_next = alg_mult_blockwise(theta_fixed, h, Q, d) + B_in @ model_in
            return h_next, None
        h_final, _ = jax.lax.scan(step, h0, (xs_seg, us_seg))
        return h_final

    def loss_frozen(theta_fixed):
        h_final = rollout_frozen(theta_fixed)
        diff = C_out_t @ h_final - target_t
        return 0.5 * jnp.sum(diff ** 2)

    g_frozen = jax.grad(loss_frozen)(theta_t)

    rel_carried_path = float(jnp.linalg.norm(g_carried - g_path) / (jnp.linalg.norm(g_path) + eps))
    eps_t = float(jnp.linalg.norm(g_carried - g_frozen) / (jnp.linalg.norm(g_frozen) + eps))
    cos_t = float(jnp.dot(g_carried, g_frozen) / (jnp.linalg.norm(g_carried) * jnp.linalg.norm(g_frozen) + eps))
    return dict(t=t, rel_carried_path=rel_carried_path, eps_frozen=eps_t, cos_frozen=cos_t,
                norm_carried=float(jnp.linalg.norm(g_carried)), norm_frozen=float(jnp.linalg.norm(g_frozen)))


# =======================================================================
# GenericBlock analogue.
# =======================================================================
def record_generic_trajectory(C=C_BUDGET, seed=SEED, lr=0.02, update_interval=1, T_record=T_RECORD):
    gc = generic_config(C)
    Q, d, p = gc["Q"], gc["d"], gc["p"]
    carry, basis, B_in = init_generic_streaming(Q, d, p, seed=1000 + seed)
    h, S, theta, C_out, opt_th, opt_co = carry
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed, T_total=T_record + 2)

    theta_hist, h_hist, cout_hist, target_hist = [theta], [h], [C_out], []
    S_hist = [S]
    for k in range(T_record):
        x_k, u_k, x_next_k = xs_t[k], us[k], xs_next[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        H = h.reshape(Q, d)
        Theta = theta.reshape(Q, p)
        A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
        H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
        G_batch = jnp.einsum("qkij,qj->qik", basis, H)
        S_next = jnp.einsum("qij,qjk->qik", A_batch, S) + G_batch
        h_next = H_next.reshape(Q * d)
        x_hat = C_out @ h_next
        diff = x_hat - x_next_k
        dl_dh = (C_out.T @ diff).reshape(Q, d)
        g_theta = clip_vec(jnp.einsum("qd,qdk->qk", dl_dh, S_next).reshape(-1))
        g_cout = clip_vec(jnp.outer(diff, h_next).reshape(-1)).reshape(C_out.shape)
        if k % update_interval == 0:
            theta_new, opt_th = adam_step(theta, g_theta, opt_th, lr)
            theta_new = project_generic_theta(theta_new, basis, Q, d, p)
            C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr)
        else:
            theta_new = theta
        h, S, theta = h_next, S_next, theta_new
        target_hist.append(x_next_k)
        theta_hist.append(theta); h_hist.append(h); S_hist.append(S); cout_hist.append(C_out)

    return dict(theta_hist=theta_hist, h_hist=h_hist, S_hist=S_hist, cout_hist=cout_hist,
                target_hist=target_hist, xs_t=xs_t, us=us, B_in=B_in, basis=basis, Q=Q, d=d, p=p,
                h0=carry[0])


def generic_checkpoint_diagnostics(rec, t, eps=1e-8):
    Q, d, p, B_in, basis, h0 = rec["Q"], rec["d"], rec["p"], rec["B_in"], rec["basis"], rec["h0"]
    h_t, S_t, theta_t, C_out_t = rec["h_hist"][t], rec["S_hist"][t], rec["theta_hist"][t], rec["cout_hist"][t]
    target_t = rec["target_hist"][t - 1]
    q_t = C_out_t.T @ (C_out_t @ h_t - target_t)

    g_carried = jnp.einsum("qd,qdk->qk", q_t.reshape(Q, d), S_t).reshape(-1)

    xs_seg = rec["xs_t"][:t]
    us_seg = rec["us"][:t]
    theta_seg = jnp.stack(rec["theta_hist"][:t])   # (t, Q*p)

    def rollout_path(alpha):
        def step(h, inputs):
            theta_k, x_k, u_k = inputs
            model_in = jnp.concatenate([x_k, jnp.array([u_k])])
            Theta = (theta_k + alpha).reshape(Q, p)
            A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
            H = h.reshape(Q, d)
            H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
            return H_next.reshape(Q * d), None
        h_final, _ = jax.lax.scan(step, h0, (theta_seg, xs_seg, us_seg))
        return h_final

    def loss_path(alpha):
        diff = C_out_t @ rollout_path(alpha) - target_t
        return 0.5 * jnp.sum(diff ** 2)

    g_path = jax.grad(loss_path)(jnp.zeros(Q * p, dtype=jnp.float64))

    def rollout_frozen(theta_fixed):
        Theta = theta_fixed.reshape(Q, p)
        A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)

        def step(h, inputs):
            x_k, u_k = inputs
            model_in = jnp.concatenate([x_k, jnp.array([u_k])])
            H = h.reshape(Q, d)
            H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
            return H_next.reshape(Q * d), None
        h_final, _ = jax.lax.scan(step, h0, (xs_seg, us_seg))
        return h_final

    def loss_frozen(theta_fixed):
        diff = C_out_t @ rollout_frozen(theta_fixed) - target_t
        return 0.5 * jnp.sum(diff ** 2)

    g_frozen = jax.grad(loss_frozen)(theta_t)

    rel_carried_path = float(jnp.linalg.norm(g_carried - g_path) / (jnp.linalg.norm(g_path) + eps))
    eps_t = float(jnp.linalg.norm(g_carried - g_frozen) / (jnp.linalg.norm(g_frozen) + eps))
    cos_t = float(jnp.dot(g_carried, g_frozen) / (jnp.linalg.norm(g_carried) * jnp.linalg.norm(g_frozen) + eps))
    return dict(t=t, rel_carried_path=rel_carried_path, eps_frozen=eps_t, cos_frozen=cos_t,
                norm_carried=float(jnp.linalg.norm(g_carried)), norm_frozen=float(jnp.linalg.norm(g_frozen)))


def loss_volatility(rec, t, window=20):
    """Trailing loss volatility (std of per-step loss over a window
    ending at checkpoint t) -- to correlate against eps_frozen."""
    h_seg = rec["h_hist"][max(0, t - window):t + 1]
    cout_seg = rec["cout_hist"][max(0, t - window):t + 1]
    targets = rec["target_hist"][max(0, t - window - 1):t]
    losses = []
    for h, c, tgt in zip(h_seg[1:], cout_seg[1:], targets):
        diff = c @ h - tgt
        losses.append(0.5 * float(jnp.sum(diff ** 2)))
    return float(np.std(losses)) if losses else float("nan")


def run_diagnostic():
    print("=" * 78)
    print(f"B35e staleness diagnostic -- RegularBlock, C={C_BUDGET}, seed={SEED}")
    print("=" * 78)
    rows = []
    for lr in LR_GRID:
        for interval in UPDATE_INTERVAL_GRID:
            rec = record_regular_trajectory(lr=lr, update_interval=interval)
            print(f"\n  --- lr={lr} update_interval={interval} ---")
            for t in CHECKPOINTS:
                diag = regular_checkpoint_diagnostics(rec, t)
                vol = loss_volatility(rec, t)
                diag.update(lr=lr, interval=interval, loss_volatility=vol)
                rows.append(diag)
                print(f"    t={t:4d}  carried-vs-path rel_err={diag['rel_carried_path']:.3e}  "
                      f"carried-vs-frozen eps={diag['eps_frozen']:.4f}  cos={diag['cos_frozen']:.4f}  "
                      f"|g_carried|={diag['norm_carried']:.3e}  |g_frozen|={diag['norm_frozen']:.3e}  "
                      f"loss_volatility(20-step)={vol:.4e}")

    print("\n" + "=" * 78)
    print(f"B35e staleness diagnostic -- GenericBlock, C={C_BUDGET}, seed={SEED}")
    print("=" * 78)
    rows_g = []
    for lr in LR_GRID:
        for interval in UPDATE_INTERVAL_GRID:
            rec = record_generic_trajectory(lr=lr, update_interval=interval)
            print(f"\n  --- lr={lr} update_interval={interval} ---")
            for t in CHECKPOINTS:
                diag = generic_checkpoint_diagnostics(rec, t)
                rows_g.append(diag)
                print(f"    t={t:4d}  carried-vs-path rel_err={diag['rel_carried_path']:.3e}  "
                      f"carried-vs-frozen eps={diag['eps_frozen']:.4f}  cos={diag['cos_frozen']:.4f}")

    # Summary: does eps_frozen grow with lr? shrink with larger interval?
    print("\n" + "=" * 78)
    print("SUMMARY (RegularBlock): mean eps_frozen by (lr, interval)")
    print("=" * 78)
    for lr in LR_GRID:
        for interval in UPDATE_INTERVAL_GRID:
            vals = [r["eps_frozen"] for r in rows if r["lr"] == lr and r["interval"] == interval]
            print(f"  lr={lr:.3f} interval={interval}: mean eps_frozen={np.mean(vals):.4f}  "
                  f"mean rel_carried_path={np.mean([r['rel_carried_path'] for r in rows if r['lr']==lr and r['interval']==interval]):.3e}")

    # Correlation between eps_frozen and loss volatility
    eps_vals = np.array([r["eps_frozen"] for r in rows])
    vol_vals = np.array([r["loss_volatility"] for r in rows])
    valid = np.isfinite(vol_vals)
    if valid.sum() > 2:
        corr = np.corrcoef(eps_vals[valid], vol_vals[valid])[0, 1]
        print(f"\nCorrelation(eps_frozen, loss_volatility) across all (lr,interval,t) rows: {corr:.4f}")

    return rows, rows_g


if __name__ == "__main__":
    run_diagnostic()
