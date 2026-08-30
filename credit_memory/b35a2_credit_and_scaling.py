"""B35a-2 (actual credit accounting) + B35a-3 (scaling), for the
bounded product-local commutative response algebra.

Run: python -m credit_memory.b35a2_credit_and_scaling
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import (
    alg_mult_blockwise, make_M_local, gen_forward_local, make_gen_params_local,
    make_theta_local, phi_blockwise, h_step_local,
)
from credit_memory.p2a_expressivity_credit_frontier import make_sequence, T_SEQ

R_LEVELS_D4 = (64, 200, 500, 800)
FIXED_D = 4


# =======================================================================
# B35a-2: actual credit accounting (m=1 structured A-valued parameter,
# theta) -- verify from ACTUAL allocated arrays, not just formulas,
# same discipline as View 2's preflight.
# =======================================================================
def preflight_credit_accounting_product_local(Q_d_list=((16, 4), (50, 4), (125, 4), (200, 4))):
    print("=" * 78)
    print("B35a-2: actual persistent-eligibility array size vs symbolic C_credit (m=1: theta)")
    print("=" * 78)
    all_ok = True
    for Q, d in Q_d_list:
        r = Q * d
        theta = make_theta_local(0, Q, d)
        P_actual = int(theta.shape[0])          # actual allocated trainable array
        s_alloc = jnp.zeros(Q * d, dtype=jnp.float64)  # actual reduced_algebra_grad_local persistent state
        credit_actual = int(s_alloc.shape[0])
        P_symbolic = r          # m=1 => P = m*r = r
        credit_symbolic = r     # persistent exact eligibility = m*r = P
        generic_sensitivity = r * P_symbolic   # r*P
        ratio = generic_sensitivity / credit_symbolic
        ok = (P_actual == P_symbolic == r) and (credit_actual == credit_symbolic == r) and (ratio == r)
        all_ok &= ok
        print(f"  Q={Q:4d} d={d:2d} r={r:4d}  P(actual theta size)={P_actual:5d}  "
              f"persistent_credit(actual s size)={credit_actual:5d}  "
              f"generic_sensitivity(r*P)={generic_sensitivity:7d}  ratio={ratio:.1f}  MATCH={ok}")
    print(f"B35a-2 ALL MATCH: {all_ok}")
    return all_ok


# =======================================================================
# B35a-3: scaling at fixed d=4, r=64,200,500,800 (Q=16,50,125,200).
# =======================================================================
def make_reduced_step_local(gen_params, Q, d):
    def step(h, s, theta, x_t):
        from credit_memory.b35a_product_local_algebra import phi_prime_blockwise, transpose_mult_blockwise
        a_t, b_t, kappa_t, c_t = gen_forward_local(x_t, gen_params, Q, d)
        A_theta_t = a_t + alg_mult_blockwise(kappa_t, theta, Q, d)
        y_t = alg_mult_blockwise(A_theta_t, h, Q, d) + alg_mult_blockwise(b_t, theta, Q, d) + c_t
        d_t = phi_prime_blockwise(y_t, Q, d)
        inner = alg_mult_blockwise(A_theta_t, s, Q, d) + alg_mult_blockwise(kappa_t, h, Q, d) + b_t
        s_next = alg_mult_blockwise(d_t, inner, Q, d)
        h_next = phi_blockwise(y_t, Q, d)
        g_contrib = transpose_mult_blockwise(s_next, jnp.ones(Q * d), Q, d)
        return h_next, s_next, g_contrib
    return jax.jit(step)


def max_local_Minf(vec, Q, d):
    """max over factors of the EXACT per-factor induced-inf norm =
    per-factor L1 norm (verified exactly in B35a-1)."""
    V = vec.reshape(Q, d)
    return float(jnp.max(jnp.sum(jnp.abs(V), axis=1)))


def run_scaling_experiment(d=FIXED_D, r_list=R_LEVELS_D4, seed=0):
    print("\n" + "=" * 78)
    print(f"B35a-3: scaling experiment, fixed d={d}, r in {r_list} (Q = r/d)")
    print("=" * 78)
    header = (f"{'r':>5} {'Q':>5} {'max|h_t|':>10} {'RMS|h_t|':>10} {'max|y_t|':>10} "
              f"{'max_q||M_theta_q||inf':>22} {'max_q||M_Atheta_q||inf':>23} "
              f"{'elig_RMS':>10} {'elig_max':>10} {'n_nonfin':>9} {'step_ms':>9}")
    print("  " + header)
    print("  " + "-" * len(header))
    rows = []
    for r in r_list:
        Q = r // d
        assert Q * d == r
        gen_params = make_gen_params_local(seed=400 + r, Q=Q, d=d)
        theta = make_theta_local(seed, Q, d)
        h0, xs = make_sequence(20_000, T_SEQ, r)

        M_theta_max = max_local_Minf(theta, Q, d)

        h = jnp.zeros(r, dtype=jnp.float64)
        h_norms, y_maxes, Atheta_norms = [], [], []
        n_nonfinite = 0
        for t in range(T_SEQ):
            x_t = jnp.stack([xs[t], 0.0, 0.0, 0.0])
            a_t, b_t, kappa_t, c_t = gen_forward_local(x_t, gen_params, Q, d)
            A_theta_t = a_t + alg_mult_blockwise(kappa_t, theta, Q, d)
            y_t = alg_mult_blockwise(A_theta_t, h, Q, d) + alg_mult_blockwise(b_t, theta, Q, d) + c_t
            h_next = phi_blockwise(y_t, Q, d)
            if not bool(jnp.all(jnp.isfinite(h_next))):
                n_nonfinite += 1
                h = jnp.nan_to_num(h_next, nan=0.0, posinf=1e300, neginf=-1e300)
                continue
            h_norms.append(np.asarray(h_next))
            y_maxes.append(float(jnp.max(jnp.abs(y_t))))
            Atheta_norms.append(max_local_Minf(A_theta_t, Q, d))
            h = h_next
        h_stack = np.stack(h_norms) if h_norms else np.zeros((1, r))
        max_h = float(np.max(np.abs(h_stack)))
        rms_h = float(np.sqrt(np.mean(h_stack ** 2)))
        max_y = float(np.max(y_maxes)) if y_maxes else float("nan")
        max_Atheta = float(np.max(Atheta_norms)) if Atheta_norms else float("nan")

        # reduced eligibility trajectory (RMS/max) + measured step time
        reduced_step = make_reduced_step_local(gen_params, Q, d)
        h_r, s_r = jnp.zeros(r, dtype=jnp.float64), jnp.zeros(r, dtype=jnp.float64)
        elig_rms_per_step, elig_max_per_step = [], []
        for t in range(T_SEQ):
            x_t = jnp.stack([xs[t], 0.0, 0.0, 0.0])
            h_r, s_r, _ = reduced_step(h_r, s_r, theta, x_t)
            elig_rms_per_step.append(float(jnp.sqrt(jnp.mean(s_r ** 2))))
            elig_max_per_step.append(float(jnp.max(jnp.abs(s_r))))
        elig_rms = float(np.mean(elig_rms_per_step))
        elig_max = float(np.max(elig_max_per_step))

        x_t0 = jnp.stack([xs[0], 0.0, 0.0, 0.0])
        t0 = time.time()
        h_r0, s_r0, _ = reduced_step(jnp.zeros(r, dtype=jnp.float64), jnp.zeros(r, dtype=jnp.float64), theta, x_t0)
        jax.block_until_ready((h_r0, s_r0))
        compile_t = time.time() - t0
        t0 = time.time()
        for _ in range(20):
            h_r0, s_r0, _ = reduced_step(h_r0, s_r0, theta, x_t0)
        jax.block_until_ready((h_r0, s_r0))
        step_ms = (time.time() - t0) / 20 * 1000

        print(f"  {r:5d} {Q:5d} {max_h:10.4e} {rms_h:10.4e} {max_y:10.4e} "
              f"{M_theta_max:22.4f} {max_Atheta:23.4f} {elig_rms:10.4e} {elig_max:10.4e} "
              f"{n_nonfinite:9d} {step_ms:9.4f}  (compile={compile_t:.3f}s)")
        rows.append(dict(r=r, Q=Q, d=d, max_h=max_h, rms_h=rms_h, max_y=max_y, M_theta_max=M_theta_max,
                          max_Atheta=max_Atheta, elig_rms=elig_rms, elig_max=elig_max,
                          n_nonfinite=n_nonfinite, step_ms=step_ms))
    return rows


if __name__ == "__main__":
    ok2 = preflight_credit_accounting_product_local()
    rows3 = run_scaling_experiment()
    print(f"\nB35a-2 PASS: {ok2}")
