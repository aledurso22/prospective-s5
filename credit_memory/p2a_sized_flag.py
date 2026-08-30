"""Phase 2A View 2 -- size-parameterized bounded-interface/flag
architecture. b31a's actual math (v_step, u_step, phi_forward) is
reused UNMODIFIED; only the outer glue that b31a hardcodes as module
constants (D_U_DIM=60, D_V_DIM=4, C_DIM=8, K_OUT=8, H_DIM=560) is
generalized into explicit arguments, so Flag can be instantiated at
different sizes to hit different matched-credit budgets. b31a.py
itself is untouched (still frozen, still imported unmodified for its
one canonical r=64 instance used as View 1's/View 2's Teacher D).

Structural form (identical to b31a, just re-sized):
  u_{t+1} = R_U u_t + D_U x_t                        (fixed, untrained)
  v_{t+1} = R_V v_t + K u_t + B_V Phi_theta(C_V v_t + C_U u_t, x_t)
theta = (R_V, K, B_V, C_V, C_U, W1, b1, W2, b2). d_v, c_dim, k_out are
also shrunk from b31a's (4, 8, 8) defaults in some tiers to make small
credit budgets reachable -- Flag's credit floor at (d_v, c, k)=(4,8,8)
is ~3232, far above what RTU/B34 need for a "small" tier; shrinking to
(2,2,2) lowers the floor to ~28, letting all three architectures share
a common small/medium/large budget range.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b31a_joint_family_correctness import v_step, u_step, phi_forward, make_stable_dense


def sized_flag_P(d_u, h_dim, d_v, c_dim, k_out):
    return d_v * d_v + d_v * d_u + d_v * k_out + c_dim * d_v + c_dim * d_u + \
        h_dim * (c_dim + 1) + h_dim + k_out * h_dim + k_out


def sized_flag_credit_scalars(d_u, h_dim, d_v, c_dim, k_out):
    P_c = sized_flag_P(d_u, h_dim, d_v, c_dim, k_out)
    r = d_u + d_v
    return dict(reduced=int(d_v * P_c), full=int(r * P_c), ratio=r / d_v, P_c=int(P_c), r=int(r))


def make_sized_flag_consts(seed, d_u):
    rng = np.random.RandomState(seed)
    R_U = jnp.array(make_stable_dense(d_u, rng, radius=0.85))
    D_U = jnp.array(rng.randn(d_u) * 0.5)
    return dict(R_U=R_U, D_U=D_U)


def make_sized_flag_theta(seed, d_u, h_dim, d_v, c_dim, k_out, r_v_radius=0.80):
    rng = np.random.RandomState(seed)
    theta = dict(
        R_V=jnp.array(make_stable_dense(d_v, rng, radius=r_v_radius)),
        K=jnp.array(rng.randn(d_v, d_u) * (0.15 / np.sqrt(d_u))),
        B_V=jnp.array(rng.randn(d_v, k_out) * (0.5 / np.sqrt(k_out))),
        C_V=jnp.array(rng.randn(c_dim, d_v) * (0.5 / np.sqrt(d_v))),
        C_U=jnp.array(rng.randn(c_dim, d_u) * (0.5 / np.sqrt(d_u))),
        W1=jnp.array(rng.randn(h_dim, c_dim + 1) * (1.0 / np.sqrt(c_dim + 1))),
        b1=jnp.array(rng.randn(h_dim) * 0.05),
        W2=jnp.array(rng.randn(k_out, h_dim) * (1.0 / np.sqrt(h_dim))),
        b2=jnp.array(rng.randn(k_out) * 0.05),
    )
    return theta


def make_sized_flag_step(consts, d_u):
    """Same functional form as b31a.full_step_state, calling b31a's
    UNMODIFIED v_step/u_step; only the split index (b31a hardcodes the
    module constant D_U_DIM=60) is generalized to d_u."""
    def step(s, theta, x):
        u, v = s[:d_u], s[d_u:]
        v_next = v_step(v, u, x, theta)
        u_next = u_step(u, x, consts)
        return jnp.concatenate([u_next, v_next])
    return step


def sized_flag_param_count(theta):
    return int(sum(np.prod(v.shape) for v in theta.values()))


# ---------------------------------------------------------------------
# Self-check: at (d_u, d_v, c, k, h) = (60, 4, 8, 8, 560) -- b31a's own
# fixed sizes -- this generalized step must agree EXACTLY with b31a's
# own full_step_state (same theta, same consts, same input).
# ---------------------------------------------------------------------
def run_equivalence_check():
    from credit_memory.b31a_joint_family_correctness import (
        make_fixed_consts, make_theta as b31a_make_theta, full_step_state as b31a_full_step_state,
        D_U_DIM, D_V_DIM, C_DIM, K_OUT, H_DIM,
    )
    consts = make_fixed_consts(seed=12345)
    theta = b31a_make_theta(seed=0)
    sized_consts = dict(R_U=consts["R_U"], D_U=consts["D_U"])
    step_sized = make_sized_flag_step(sized_consts, D_U_DIM)

    rng = np.random.RandomState(0)
    s = jnp.array(rng.randn(D_U_DIM + D_V_DIM) * 0.2)
    worst = 0.0
    for i in range(20):
        x = float(rng.randn() * 0.5)
        out_b31a = b31a_full_step_state(s, x, theta, consts)
        out_sized = step_sized(s, theta, x)
        d = float(jnp.max(jnp.abs(out_b31a - out_sized)))
        worst = max(worst, d)
        s = out_b31a
    P_sized = sized_flag_P(D_U_DIM, H_DIM, D_V_DIM, C_DIM, K_OUT)
    from credit_memory.b31a_joint_family_correctness import TRAINABLE_KEYS, FAMILY_SHAPES
    P_b31a = sum(int(np.prod(FAMILY_SHAPES[k])) for k in TRAINABLE_KEYS)
    print(f"Equivalence check (d_u={D_U_DIM},d_v={D_V_DIM},c={C_DIM},k={K_OUT},h={H_DIM}): "
          f"max|diff over 20 steps|={worst:.3e}  P_sized={P_sized}  P_b31a={P_b31a}  "
          f"P_match={P_sized == P_b31a}")
    return worst < 1e-12 and P_sized == P_b31a


if __name__ == "__main__":
    ok = run_equivalence_check()
    print(f"PASS: {ok}")
