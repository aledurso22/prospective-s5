"""B37a isolation of the SMALLEST FAILING EXAMPLE (exact_jordan, r=8, seed=0).

Computes EXACT ground truth via rational forward-mode AD of the forward model,
then measures the error of ALL THREE float64 paths against it. This determines
whether the >1e-10 discrepancies are a derivation error or float64 roundoff,
and which of the three float64 paths is least accurate.
"""
from __future__ import annotations

from fractions import Fraction as F
import numpy as np
import jax.numpy as jnp

from credit_memory.b37a_universal_quotient import (
    make_case, reduced_sensitivities, full_rtrl_sensitivities,
    autodiff_sensitivities, M_INPUTS,
)
from credit_memory.b37a_exact_rational_check import (
    forward_dual, conv, divmod_monic, alg_mult, mult_matrix, matvec_from_cols, to_F,
)

FAMILY, R, SEED = "exact_jordan", 8, 0
T_EXACT = 40


def main():
    theta, a, B, z0, xs, info = make_case(FAMILY, R, SEED)
    r, m = R, M_INPUTS
    print("=" * 92)
    print(f"SMALLEST FAILING EXAMPLE: family={FAMILY} r={r} seed={SEED}")
    print(f"  note={info.get('note')}   rho(M_u)={info['rho_Mu']:.6f}   cond(M_u)={info['cond_Mu']:.3e}")
    zs = np.asarray(jnp.stack([z0]))
    print("=" * 92)

    # ---- float64 paths ----
    S_th_red, _, _ = reduced_sensitivities(theta, a, B, z0, xs, r, m)
    S_th_rtrl, _, _ = full_rtrl_sensitivities(theta, a, B, z0, xs, r, m)
    S_th_ad, _, _ = autodiff_sensitivities(theta, a, B, z0, xs, r)

    # ---- exact ground truth, contracted along a fixed direction ----
    rng = np.random.RandomState(12345)
    direction_np = rng.randn(r)
    d_F = to_F(direction_np)
    theta_F, a_F, z0_F = to_F(theta), to_F(a), to_F(z0)
    B_F = [to_F(np.asarray(B)[:, j]) for j in range(m)]
    xs_F = [[F(float(v)) for v in np.asarray(xs)[t]] for t in range(T_EXACT)]

    exact_dz = forward_dual(theta_F, a_F, B_F, z0_F, xs_F, r, m, T_EXACT, "theta", d_F)

    d_j = jnp.array(direction_np)
    print(f"{'t':>4s} {'||exact||':>12s} {'reduced':>12s} {'full-RTRL':>12s} {'autodiff':>12s}"
          "   (relative error of each float64 path vs EXACT)")
    worst = {"reduced": 0.0, "rtrl": 0.0, "ad": 0.0}
    for t in list(range(0, T_EXACT, 4)) + [T_EXACT - 1]:
        ex = np.array([float(v) for v in exact_dz[t]])
        nrm = np.linalg.norm(ex)
        errs = {}
        for tag, S in (("reduced", S_th_red), ("rtrl", S_th_rtrl), ("ad", S_th_ad)):
            got = np.asarray(S[t] @ d_j)
            errs[tag] = float(np.linalg.norm(got - ex) / (1.0 + nrm))
            worst[tag] = max(worst[tag], errs[tag])
        print(f"{t:4d} {nrm:12.4e} {errs['reduced']:12.3e} {errs['rtrl']:12.3e} {errs['ad']:12.3e}")

    print("-" * 92)
    print("WORST relative error vs EXACT rational ground truth over all sampled t:")
    for tag, label in (("reduced", "REDUCED (this work)"), ("rtrl", "full RTRL (r x r matrix)"),
                       ("ad", "BPTT/autodiff")):
        print(f"   {label:28s} {worst[tag]:.3e}")
    ranking = sorted(worst, key=worst.get)
    print(f"\nAccuracy ranking (best -> worst): {' < '.join(ranking)}")
    print("=" * 92)


if __name__ == "__main__":
    main()
