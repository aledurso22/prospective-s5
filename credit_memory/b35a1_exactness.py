"""B35a-1 -- algebra/exactness verification for the bounded product-
local commutative response algebra (b35a_product_local_algebra.py).

Run: python -m credit_memory.b35a1_exactness
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import (
    alg_mult_blockwise, make_M_full_blockdiag, h_step_local, gen_forward_local,
    make_gen_params_local, make_theta_local, make_setting_local, make_grad_bptt_fn_local,
    reduced_algebra_grad_local, full_rtrl_grad_local, rollout_h_local,
)


def verify_blockwise_mult_vs_explicit(seeds=(0, 1, 2), Q_d_list=((4, 1), (4, 2), (4, 4), (2, 8), (1, 8))):
    print("=" * 78)
    print("1a. blockwise multiplication vs explicit regular-representation matrices, M_a@M_b == M_{a*b}")
    print("=" * 78)
    worst_mult = worst_matmul = 0.0
    for Q, d in Q_d_list:
        r = Q * d
        for seed in seeds:
            rng = np.random.RandomState(seed)
            a = jnp.array(rng.randn(r) * 0.3)
            b = jnp.array(rng.randn(r) * 0.3)
            ab_direct = alg_mult_blockwise(a, b, Q, d)
            M_a = make_M_full_blockdiag(a, Q, d)
            M_b = make_M_full_blockdiag(b, Q, d)
            ab_via_M = M_a @ b
            err_mult = float(jnp.max(jnp.abs(ab_direct - ab_via_M)))

            M_ab = make_M_full_blockdiag(ab_direct, Q, d)
            err_matmul = float(jnp.max(jnp.abs(M_a @ M_b - M_ab)))
            worst_mult = max(worst_mult, err_mult)
            worst_matmul = max(worst_matmul, err_matmul)
            print(f"  Q={Q:3d} d={d:2d} r={r:4d} seed={seed}  "
                  f"|alg_mult - M_a@b|={err_mult:.2e}  |M_a@M_b - M_{{a*b}}|={err_matmul:.2e}")
    print(f"WORST: mult={worst_mult:.2e}  matmul={worst_matmul:.2e}   PASS={worst_mult<1e-10 and worst_matmul<1e-10}")
    return worst_mult < 1e-10 and worst_matmul < 1e-10


def verify_J_and_G_are_blockdiag(seeds=(0, 1, 2), Q_d_list=((4, 1), (4, 2), (4, 4), (2, 8))):
    print("\n" + "=" * 78)
    print("1b. J_t = M_{u_t} (u_t = A_theta_t) and G_t = M_{g_t} (g_t = d_t*(kappa_t*h+b_t)), block-diagonal")
    print("=" * 78)
    worst_J = worst_G = worst_blockdiag = 0.0
    for Q, d in Q_d_list:
        r = Q * d
        for seed in seeds:
            gen_params = make_gen_params_local(seed=100 + r, Q=Q, d=d)
            theta = make_theta_local(seed, Q, d)
            rng = np.random.RandomState(seed + 500)
            h = jnp.array(rng.randn(r) * 0.15)
            x_t = jnp.array(rng.randn(4) * 0.5)

            J_t = jax.jacobian(lambda hh: h_step_local(hh, theta, x_t, gen_params, Q, d))(h)
            G_t = jax.jacobian(lambda th: h_step_local(h, th, x_t, gen_params, Q, d))(theta)

            a_t, b_t, kappa_t, c_t = gen_forward_local(x_t, gen_params, Q, d)
            A_theta_t = a_t + alg_mult_blockwise(kappa_t, theta, Q, d)
            y_t = alg_mult_blockwise(A_theta_t, h, Q, d) + alg_mult_blockwise(b_t, theta, Q, d) + c_t
            from credit_memory.b35a_product_local_algebra import phi_prime_blockwise
            d_t = phi_prime_blockwise(y_t, Q, d)
            g_t = alg_mult_blockwise(d_t, alg_mult_blockwise(kappa_t, h, Q, d) + b_t, Q, d)
            u_t = alg_mult_blockwise(d_t, A_theta_t, Q, d)

            M_u = make_M_full_blockdiag(u_t, Q, d)
            M_g = make_M_full_blockdiag(g_t, Q, d)
            err_J = float(jnp.max(jnp.abs(J_t - M_u)))
            err_G = float(jnp.max(jnp.abs(G_t - M_g)))

            # block-diagonal structural check: off-block entries of J_t,G_t are exactly 0
            mask = np.ones((r, r), dtype=bool)
            for q in range(Q):
                mask[q * d:(q + 1) * d, q * d:(q + 1) * d] = False
            off_block_J = float(jnp.max(jnp.abs(J_t)) if Q == 1 else jnp.max(jnp.abs(jnp.asarray(J_t)[mask])))
            off_block_G = float(jnp.max(jnp.abs(G_t)) if Q == 1 else jnp.max(jnp.abs(jnp.asarray(G_t)[mask])))
            worst_blockdiag = max(worst_blockdiag, off_block_J, off_block_G)

            worst_J = max(worst_J, err_J)
            worst_G = max(worst_G, err_G)
            print(f"  Q={Q:3d} d={d:2d} r={r:4d} seed={seed}  |J_t-M_u|={err_J:.2e}  |G_t-M_g|={err_G:.2e}  "
                  f"off-block max|.|={max(off_block_J, off_block_G):.2e}")
    ok = worst_J < 1e-8 and worst_G < 1e-8 and worst_blockdiag < 1e-10
    print(f"WORST: J={worst_J:.2e}  G={worst_G:.2e}  off-block={worst_blockdiag:.2e}   PASS={ok}")
    return ok


def verify_reduced_full_bptt(seeds=(0, 1, 2), lengths=(1, 5, 20), Q_d_list=((4, 1), (4, 2), (4, 4), (2, 8), (1, 16))):
    print("\n" + "=" * 78)
    print("1c. reduced RTRL == full RTRL == BPTT, to machine precision")
    print("=" * 78)
    grad_bptt_fn = make_grad_bptt_fn_local()
    worst_full = worst_reduced = 0.0
    for Q, d in Q_d_list:
        r = Q * d
        gen_params = make_gen_params_local(seed=200 + r, Q=Q, d=d)
        for T in lengths:
            for seed in seeds:
                theta, h0, xs, qs, phases = make_setting_local(seed, T, Q, d)
                g_b = grad_bptt_fn(theta, h0, xs, qs, phases, gen_params, Q, d)
                g_f, _ = full_rtrl_grad_local(theta, h0, xs, qs, phases, gen_params, Q, d)
                g_r, _ = reduced_algebra_grad_local(theta, h0, xs, qs, phases, gen_params, Q, d)
                rel_full = float(jnp.linalg.norm(g_f - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
                rel_reduced = float(jnp.linalg.norm(g_r - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
                worst_full = max(worst_full, rel_full)
                worst_reduced = max(worst_reduced, rel_reduced)
        print(f"  Q={Q:3d} d={d:2d} r={r:4d}  worst_full_rel={worst_full:.2e}  worst_reduced_rel={worst_reduced:.2e}")
    ok = worst_full < 1e-8 and worst_reduced < 1e-8
    print(f"WORST OVERALL: full={worst_full:.2e}  reduced={worst_reduced:.2e}   PASS={ok}")
    return ok


def verify_optimizer_trajectory(Q=16, d=4, T=32, n_steps=10, lr=0.01, seed=0):
    print("\n" + "=" * 78)
    print(f"1d. reduced-RTRL vs BPTT optimizer trajectory, {n_steps} updates, Q={Q} d={d} r={Q*d}")
    print("=" * 78)
    r = Q * d
    gen_params = make_gen_params_local(seed=300 + r, Q=Q, d=d)
    theta_red = make_theta_local(seed, Q, d)
    theta_bptt = make_theta_local(seed, Q, d)
    grad_bptt_fn = make_grad_bptt_fn_local()

    from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, clip_grad
    opt_red = adam_init(theta_red)
    opt_bptt = adam_init(theta_bptt)

    grad_discs, param_discs = [], []
    for step in range(n_steps):
        _, h0, xs, qs, phases = make_setting_local(1000 + step, T, Q, d)

        g_red, _ = reduced_algebra_grad_local(theta_red, h0, xs, qs, phases, gen_params, Q, d)
        g_bptt = grad_bptt_fn(theta_bptt, h0, xs, qs, phases, gen_params, Q, d)
        grad_discs.append(float(jnp.linalg.norm(g_red - g_bptt)))

        theta_red, opt_red = adam_step(theta_red, clip_grad(g_red), opt_red, lr)
        theta_bptt, opt_bptt = adam_step(theta_bptt, clip_grad(g_bptt), opt_bptt, lr)
        from credit_memory.b35a_product_local_algebra import project_local_tails
        theta_red = project_local_tails(theta_red, Q, d)
        theta_bptt = project_local_tails(theta_bptt, Q, d)
        param_discs.append(float(jnp.linalg.norm(theta_red - theta_bptt)))

    print(f"  grad discrepancies (10 steps): {['%.2e' % v for v in grad_discs]}")
    print(f"  param discrepancies (10 steps): {['%.2e' % v for v in param_discs]}")
    ok = max(grad_discs) < 1e-6 and max(param_discs) < 1e-5
    print(f"PASS={ok}")
    return ok


if __name__ == "__main__":
    r1 = verify_blockwise_mult_vs_explicit()
    r2 = verify_J_and_G_are_blockdiag()
    r3 = verify_reduced_full_bptt()
    r4 = verify_optimizer_trajectory()
    print("\n" + "=" * 78)
    print(f"B35a-1 ALL PASS: {r1 and r2 and r3 and r4}")
    print("=" * 78)
