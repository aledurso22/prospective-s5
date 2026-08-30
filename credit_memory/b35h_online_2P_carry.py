"""B35h-online -- verify the direction-independent 2P Hessian carry.
Diagnostic only: no training, no tuning.

B35h's w_t recursion required a FIXED direction Delta known from t=0,
which cannot be the persistent online object in a real continual
learner (the optimizer's direction changes every step and is only
known after the current gradient). This script derives and verifies
that w_t always factors as w_t = alg_mult(r_t, Delta, Q, d) for a
DIRECTION-INDEPENDENT trace r_t satisfying its own recursion with no
Delta dependence at all:

Derivation (by induction, using ONLY associativity/bilinearity of the
commutative algebra product already established throughout this
project -- w_{k+1}=alg_mult(theta,w_k)+2*alg_mult(s_k,Delta), assume
w_k=alg_mult(r_k,Delta):
  w_{k+1} = alg_mult(theta, alg_mult(r_k,Delta)) + alg_mult(2*s_k, Delta)
          = alg_mult(alg_mult(theta,r_k) + 2*s_k, Delta)   [associativity + bilinearity]
  => r_{k+1} = alg_mult(theta, r_k, Q, d) + 2*s_k,  r_0 = 0.
And the quadratic contraction follows from ONE more associativity step:
  H_t[Delta,Delta] = alg_mult(w_t,Delta) = alg_mult(alg_mult(r_t,Delta),Delta)
                    = alg_mult(r_t, alg_mult(Delta,Delta))  [associativity]
                    = alg_mult(r_t, Delta*Delta).

Run: python -m credit_memory.b35h_online_2P_carry
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise
from credit_memory.b35c_matched_credit_frontier import regular_config
from credit_memory.b35e_staleness_diagnostic import record_regular_trajectory
from credit_memory.b35h_hessian_transport_diagnostic import make_rollout_fn, make_M

C_BUDGET = 64
CHECKPOINTS = (50, 150, 300)
N_DIRECTIONS = 5
SEED = 11


def reduced_s_and_r(theta, xs_seg, us_seg, h0, Q, d, B_in):
    """Propagate s_t (first-order sensitivity) and r_t (direction-
    independent second trace) TOGETHER, with r_t's own recursion never
    referencing any direction Delta."""
    h = h0
    s = jnp.zeros(Q * d, dtype=jnp.float64)
    r = jnp.zeros(Q * d, dtype=jnp.float64)
    for k in range(xs_seg.shape[0]):
        x_k, u_k = xs_seg[k], us_seg[k]
        model_in = jnp.concatenate([x_k, jnp.array([u_k])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        r_next = alg_mult_blockwise(theta, r, Q, d) + 2 * s     # NO Delta anywhere
        h, s, r = h_next, s_next, r_next
    return h, s, r


def run_checkpoint_verification(rec, t, n_directions=N_DIRECTIONS, seed=0):
    Q, d, B_in, h0 = rec["Q"], rec["d"], rec["B_in"], rec["h0"]
    theta_t = rec["theta_hist"][t]
    xs_seg, us_seg = rec["xs_t"][:t], rec["us"][:t]
    rollout = make_rollout_fn(xs_seg, us_seg, h0, Q, d, B_in)

    # r_t propagated ONCE, before any direction is chosen.
    h_ref, s_ref, r_t = reduced_s_and_r(theta_t, xs_seg, us_seg, h0, Q, d, B_in)

    # brute-force ground truth Hessian, computed ONCE per checkpoint.
    Hfull = jax.jacobian(jax.jacobian(rollout))(theta_t)   # (r,P,P)

    rng = np.random.RandomState(seed)
    results = []
    for _ in range(n_directions):
        Delta = jnp.array(rng.randn(Q * d))
        Delta = Delta / jnp.linalg.norm(Delta)

        # H_t[Delta] from r_t, drawn AFTER r_t was already propagated.
        w_from_r = alg_mult_blockwise(r_t, Delta, Q, d)
        H_from_r = make_M(w_from_r, Q, d)
        H_bruteforce = jnp.einsum("ijk,k->ij", Hfull, Delta)
        rel_err_H = float(jnp.linalg.norm(H_from_r - H_bruteforce) / (jnp.linalg.norm(H_bruteforce) + 1e-12))

        # quadratic contraction H_t[Delta,Delta] = alg_mult(r_t, Delta*Delta)
        Delta_sq = alg_mult_blockwise(Delta, Delta, Q, d)
        quad_from_r = alg_mult_blockwise(r_t, Delta_sq, Q, d)
        quad_bruteforce = jnp.einsum("ijk,j,k->i", Hfull, Delta, Delta)
        rel_err_quad = float(jnp.linalg.norm(quad_from_r - quad_bruteforce) /
                              (jnp.linalg.norm(quad_bruteforce) + 1e-12))

        results.append(dict(rel_err_H=rel_err_H, rel_err_quad=rel_err_quad))
    return results, r_t


def run_audit():
    print("=" * 78)
    print(f"B35h-online: direction-independent 2P Hessian carry, C={C_BUDGET}, seed={SEED}")
    print("=" * 78)
    rec = record_regular_trajectory(C=C_BUDGET, seed=SEED, lr=0.02, update_interval=1,
                                     T_record=max(CHECKPOINTS) + 5)
    Q, d = rec["Q"], rec["d"]
    P = Q * d

    worst_H, worst_quad = 0.0, 0.0
    for t in CHECKPOINTS:
        results, r_t = run_checkpoint_verification(rec, t)
        print(f"\n  t={t}  (r_t propagated once, {N_DIRECTIONS} directions drawn AFTER)")
        for i, res in enumerate(results):
            print(f"    direction {i}: H_t[Delta] rel_err (vs brute-force)={res['rel_err_H']:.3e}   "
                  f"H_t[Delta,Delta] rel_err (vs brute-force)={res['rel_err_quad']:.3e}")
            worst_H = max(worst_H, res["rel_err_H"])
            worst_quad = max(worst_quad, res["rel_err_quad"])
        print(f"    |r_t| persistent scalars = {int(r_t.shape[0])}  (= P = {P})")

    print("\n" + "=" * 78)
    print(f"WORST across all checkpoints/directions: H_t[Delta] rel_err={worst_H:.3e}  "
          f"H_t[Delta,Delta] rel_err={worst_quad:.3e}")
    print(f"Persistent scalars: s_t (P={P}) + r_t (P={P}) = 2P = {2*P}, "
          f"independent of sequence length t and independent of any later-chosen direction "
          f"(r_t's recursion never references Delta).")
    all_pass = worst_H < 1e-8 and worst_quad < 1e-8
    print(f"\nALL IDENTITIES VERIFIED TO MACHINE PRECISION: {all_pass}")
    return all_pass


if __name__ == "__main__":
    run_audit()
