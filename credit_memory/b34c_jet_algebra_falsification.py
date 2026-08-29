"""Phase B34c -- lean closure falsification for the jet-algebra exact
RTRL construction (B34a/B34b). ONE clean test: introduce a stable
generic operator R outside the regular multiplication algebra
(h_{t+1} = phi(y_t) + eps*R h_t), verify full RTRL/BPTT stay exact
while the forced r-scalar algebra recurrence breaks for eps>0. Small,
lean sweep only -- not an extensive tuning exercise.

Run: python -m credit_memory.b34c_jet_algebra_falsification
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b34a_jet_algebra_correctness import (
    make_gen_params, make_grad_bptt_fn, full_rtrl_grad, reduced_algebra_grad,
    reconstruct_S, make_setting,
)

R = 64


def run_falsification(r=R, eps_list=(0.0, 1e-4, 1e-2), lengths=(20, 100), seed=0):
    print("=" * 78)
    print(f"B34c lean closure falsification: r={r}, J_t^eps = J_t (via h_next) + eps*R h_t")
    print("=" * 78)
    gen_params = make_gen_params(seed=1000 + r, r=r)
    grad_bptt_fn = make_grad_bptt_fn()

    rng_R = np.random.RandomState(24680)
    M = rng_R.randn(r, r)
    R_generic = jnp.array((M + M.T) / 2.0)  # fixed, generic, NOT of the form M_u

    for eps in eps_list:
        for T in lengths:
            theta, h0, xs, qs, phases = make_setting(seed, T, r)
            g_b = grad_bptt_fn(theta, h0, xs, qs, phases, gen_params, r, eps, R_generic)
            g_f, S_traj = full_rtrl_grad(theta, h0, xs, qs, phases, gen_params, r, eps, R_generic)
            # forced reduced path: deliberately still uses the OLD eps=0
            # closure, unaware of R_generic (same B29/B32a/B34a convention).
            g_red, s_traj = reduced_algebra_grad(theta, h0, xs, qs, phases, gen_params, r)

            rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
            rel_reduced = float(jnp.linalg.norm(g_red - g_b) / (jnp.linalg.norm(g_b) + 1e-12))

            S_np = np.asarray(S_traj)
            S_hat = np.asarray(reconstruct_S(s_traj, r))
            recon_err = float(np.max(np.abs(S_np - S_hat)))

            flat = S_np.reshape(T, -1)
            sv = np.linalg.svd(flat, compute_uv=False)
            span_dim = int(np.sum(sv > 1e-9 * sv[0])) if sv[0] > 1e-12 else 0

            print(f"  eps={eps:.0e}  T={T:4d}  full_rel={rel_full:.3e}  reduced_rel={rel_reduced:.3e}  "
                  f"recon_max|d|={recon_err:.3e}  span_dim{{S_1..S_T}}={span_dim:3d}/{min(T, r*r)}")


if __name__ == "__main__":
    run_falsification()
