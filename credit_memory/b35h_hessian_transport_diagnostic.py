"""B35h -- ProductLocal Hessian-transport viability diagnostic.
Diagnostic only: no training, no architecture/optimizer/projection/
credit-budget change, no effect on B35d's results. Tests whether
ProductLocal's block-local structure extends to give an O(P)
Hessian-vector-product correction for fixed-parameter sensitivity
retargeting.

Derivation (verified numerically below, not just asserted): for the
linear regular-block recurrence h_{k+1}=alg_mult(theta,h_k,Q,d)+b_k
(b_k theta-independent), the established first-order sensitivity
recursion is s_{k+1}=alg_mult(theta,s_k,Q,d)+h_k (S_k=M_{s_k}). Writing
w_k := d(s_k)/dtheta[Delta] (the directional derivative of the
sensitivity trace itself), differentiating that recursion once more
gives, by the SAME associativity/commutativity of the algebra used
throughout this whole project:
  w_{k+1} = alg_mult(theta, w_k, Q, d) + 2*alg_mult(s_k, Delta, Q, d),
  w_0 = 0,
and H_t[Delta] = M_{w_t} exactly -- an O(P) companion recursion to
s_t's own (same per-factor alg_mult cost), NOT an O(P^2) object.

Run: python -m credit_memory.b35h_hessian_transport_diagnostic
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise
from credit_memory.b35c_matched_credit_frontier import regular_config
from credit_memory.b35d_streaming_sysid import init_regular_streaming, make_teacher_trajectory
from credit_memory.b35e_staleness_diagnostic import record_regular_trajectory

C_BUDGET = 64
CHECKPOINTS = (50, 150, 300)
ETAS = (-0.1, -0.03, -0.01, 0.01, 0.03, 0.1)
SEED = 11


# =======================================================================
# Core replay / reduced-sensitivity / Hessian-vector-product functions.
# =======================================================================
def make_rollout_fn(xs_seg, us_seg, h0, Q, d, B_in):
    def rollout(theta):
        def step(h, inputs):
            x_k, u_k = inputs
            model_in = jnp.concatenate([x_k, jnp.array([u_k])])
            h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
            return h_next, None
        h_final, _ = jax.lax.scan(step, h0, (xs_seg, us_seg))
        return h_final
    return rollout


def reduced_s_and_h(theta, xs_seg, us_seg, h0, Q, d, B_in):
    h, s = h0, jnp.zeros(Q * d, dtype=jnp.float64)
    for k in range(xs_seg.shape[0]):
        x_k, u_k = xs_seg[k], us_seg[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        h, s = h_next, s_next
    return h, s


def reduced_w_hvp(theta, xs_seg, us_seg, h0, Delta, Q, d, B_in):
    """O(P) companion recursion for the Hessian-vector product H_t[Delta]=M_w."""
    h, s, w = h0, jnp.zeros(Q * d, dtype=jnp.float64), jnp.zeros(Q * d, dtype=jnp.float64)
    for k in range(xs_seg.shape[0]):
        x_k, u_k = xs_seg[k], us_seg[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        w_next = alg_mult_blockwise(theta, w, Q, d) + 2 * alg_mult_blockwise(s, Delta, Q, d)
        h, s, w = h_next, s_next, w_next
    return w


def make_M(u, Q, d):
    from credit_memory.b34a_jet_algebra_correctness import make_M as make_M_local
    U = u.reshape(Q, d)
    blocks = [make_M_local(U[q], d) for q in range(Q)]
    from jax.scipy.linalg import block_diag
    return block_diag(*blocks)


# =======================================================================
# Verification: my O(P) w-recursion vs brute-force nested-jacobian Hessian.
# =======================================================================
def verify_hessian_recursion(Q, d, xs_seg, us_seg, h0, B_in, theta_t, Delta, seed_tag=""):
    rollout = make_rollout_fn(xs_seg, us_seg, h0, Q, d, B_in)
    _, hvp_full = jax.jvp(jax.jacobian(rollout), (theta_t,), (Delta,))   # (r,P) brute-force HVP
    w_t = reduced_w_hvp(theta_t, xs_seg, us_seg, h0, Delta, Q, d, B_in)
    H_reduced = make_M(w_t, Q, d)
    rel_err = float(jnp.linalg.norm(H_reduced - hvp_full) / (jnp.linalg.norm(hvp_full) + 1e-12))
    print(f"  {seed_tag} O(P) Hessian-recursion vs brute-force JVP-of-jacobian: rel_err={rel_err:.3e}")
    return rel_err, w_t, hvp_full


# =======================================================================
# E0/E1 scaling test + no/diag/full correction comparison.
# =======================================================================
def run_checkpoint_test(rec, t, v, etas=ETAS):
    """Full (r,P)-matrix comparisons (Frobenius norm), matching the
    original spec literally: E_0/E_1 compare S_t^replay(theta_t+eta v)
    against S_t(theta_t) [+ eta*H_t[v]] as (r,P) sensitivity matrices,
    not a further v-projected scalar."""
    Q, d, B_in, h0 = rec["Q"], rec["d"], rec["B_in"], rec["h0"]
    theta_t = rec["theta_hist"][t]
    xs_seg, us_seg = rec["xs_t"][:t], rec["us"][:t]

    rollout = make_rollout_fn(xs_seg, us_seg, h0, Q, d, B_in)
    h_ref, s_ref = reduced_s_and_h(theta_t, xs_seg, us_seg, h0, Q, d, B_in)
    S_t = make_M(s_ref, Q, d)   # S_t(theta_t): fixed-parameter reference sensitivity, (r,P)

    # full brute-force Hessian (r,P,P) -- ground truth for verification and diagonal extraction
    Hfull = jax.jacobian(jax.jacobian(rollout))(theta_t)
    w_v = reduced_w_hvp(theta_t, xs_seg, us_seg, h0, v, Q, d, B_in)
    H_v_full = make_M(w_v, Q, d)                          # (r,P) = H_t[v], O(P) block-local recursion
    H_v_bruteforce = jnp.einsum("ijk,k->ij", Hfull, v)    # (r,P) ground truth H_t[v]
    rel_err_H = float(jnp.linalg.norm(H_v_full - H_v_bruteforce) / (jnp.linalg.norm(H_v_bruteforce) + 1e-12))

    # diagonal-Hessian approximation of H_t[v]: keep only Hfull[i,j,j] (drop
    # cross-parameter j!=k terms), giving H_diag[i,k] = Hfull[i,k,k]*v[k].
    H_diag_diag = jnp.einsum("ijj->ij", Hfull)   # (r,P), Hfull[i,j,j]
    H_v_diag = H_diag_diag * v[None, :]          # (r,P), same shape as H_v_full

    rows = []
    for eta in etas:
        theta_pert = theta_t + eta * v
        S_replay = jax.jacobian(rollout)(theta_pert)   # (r,P) exact fixed-parameter replay sensitivity
        h_replay = rollout(theta_pert)

        E0 = S_replay - S_t                            # (r,P) zeroth-order (no correction)
        E1_full = E0 - eta * H_v_full                  # (r,P) full block-local correction
        E1_diag = E0 - eta * H_v_diag                  # (r,P) diagonal-only correction

        # state transport (contract S_t, H_t[v] with v once more -- these ARE
        # single vectors of interest for the r-dim state update itself).
        Sv_ref = S_t @ v
        h_no_corr = h_ref + eta * Sv_ref
        h_full_corr = h_ref + eta * Sv_ref + 0.5 * eta ** 2 * (H_v_full @ v)
        state_err_no_corr = float(jnp.linalg.norm(h_replay - h_no_corr))
        state_err_full_corr = float(jnp.linalg.norm(h_replay - h_full_corr))

        rows.append(dict(eta=eta, E0_norm=float(jnp.linalg.norm(E0)), E1_full_norm=float(jnp.linalg.norm(E1_full)),
                          E1_diag_norm=float(jnp.linalg.norm(E1_diag)),
                          state_err_no_corr=state_err_no_corr, state_err_full_corr=state_err_full_corr))
    return rows, rel_err_H


def loglog_slope(etas, norms):
    etas = np.abs(np.asarray(etas))
    norms = np.asarray(norms)
    mask = (norms > 1e-14) & (etas > 0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(etas[mask]), np.log(norms[mask]), 1)
    return slope


def run_diagnostic():
    print("=" * 78)
    print(f"B35h Hessian-transport diagnostic, C={C_BUDGET}, seed={SEED}")
    print("=" * 78)
    rec = record_regular_trajectory(C=C_BUDGET, seed=SEED, lr=0.02, update_interval=1, T_record=max(CHECKPOINTS) + 5)
    Q, d = rec["Q"], rec["d"]
    rng = np.random.RandomState(0)
    v = jnp.array(rng.randn(Q * d))
    v = v / jnp.linalg.norm(v)

    print("\n--- Verification: O(P) Hessian recursion vs brute-force JVP-of-jacobian ---")
    for t in CHECKPOINTS:
        xs_seg, us_seg = rec["xs_t"][:t], rec["us"][:t]
        theta_t = rec["theta_hist"][t]
        verify_hessian_recursion(Q, d, xs_seg, us_seg, rec["h0"], rec["B_in"], theta_t, v, seed_tag=f"t={t}:")

    print("\n--- E0/E1 log-log scaling + no/diag/full correction comparison ---")
    all_rows = {}
    for t in CHECKPOINTS:
        rows, rel_err_H = run_checkpoint_test(rec, t, v)
        all_rows[t] = rows
        print(f"\n  t={t}  (H_v reduced-vs-bruteforce rel_err={rel_err_H:.3e})")
        for r in rows:
            print(f"    eta={r['eta']:+.3f}  E0={r['E0_norm']:.4e}  E1_full={r['E1_full_norm']:.4e}  "
                  f"E1_diag={r['E1_diag_norm']:.4e}  state_err_no_corr={r['state_err_no_corr']:.4e}  "
                  f"state_err_full_corr={r['state_err_full_corr']:.4e}")
        etas_ = [r["eta"] for r in rows]
        slope_E0 = loglog_slope(etas_, [r["E0_norm"] for r in rows])
        slope_E1_full = loglog_slope(etas_, [r["E1_full_norm"] for r in rows])
        slope_E1_diag = loglog_slope(etas_, [r["E1_diag_norm"] for r in rows])
        slope_state_no = loglog_slope(etas_, [r["state_err_no_corr"] for r in rows])
        slope_state_full = loglog_slope(etas_, [r["state_err_full_corr"] for r in rows])
        frac_removed_full = 1 - np.mean([r["E1_full_norm"] / (r["E0_norm"] + 1e-12) for r in rows])
        frac_removed_diag = 1 - np.mean([r["E1_diag_norm"] / (r["E0_norm"] + 1e-12) for r in rows])
        print(f"    log-log slopes: E0={slope_E0:.3f} (predict 1)  E1_full={slope_E1_full:.3f} (predict 2)  "
              f"E1_diag={slope_E1_diag:.3f}  state_no_corr={slope_state_no:.3f} (predict 1)  "
              f"state_full_corr={slope_state_full:.3f} (predict 2)")
        print(f"    mean fraction of E0 mismatch removed: full={frac_removed_full:.4f}  diag={frac_removed_diag:.4f}")

    print("\n--- O(P) scaling: persistent scalar count and per-step update cost ---")
    for C in (64, 128, 256):
        rc = regular_config(C)
        Qc, dc = rc["Q"], rc["d"]
        P = Qc * dc
        theta = jnp.zeros(P, dtype=jnp.float64)
        s = jnp.zeros(P, dtype=jnp.float64)
        w = jnp.zeros(P, dtype=jnp.float64)
        Delta = jnp.ones(P, dtype=jnp.float64) * 0.01
        h = jnp.zeros(P, dtype=jnp.float64)

        @jax.jit
        def combined_step(h, s, w, theta, Delta):
            h_next = alg_mult_blockwise(theta, h, Qc, dc)
            s_next = alg_mult_blockwise(theta, s, Qc, dc) + h
            w_next = alg_mult_blockwise(theta, w, Qc, dc) + 2 * alg_mult_blockwise(s, Delta, Qc, dc)
            return h_next, s_next, w_next

        h, s, w = combined_step(h, s, w, theta, Delta)
        jax.block_until_ready((h, s, w))
        t0 = time.time()
        for _ in range(50):
            h, s, w = combined_step(h, s, w, theta, Delta)
        jax.block_until_ready((h, s, w))
        step_time = (time.time() - t0) / 50
        print(f"    C={C}: P={P}  persistent scalars (s_t + w_t) = 2*{P}={2*P}  "
              f"combined (s+H) update step time={step_time*1e6:.2f}us")

    return all_rows


if __name__ == "__main__":
    run_diagnostic()
