"""B34 dimension-normalization audit (forward-only, no training, no
cross-family performance). Follow-up to p2a_b34_stability_audit.py's
finding that teacher B's own (untrained) forward dynamics explode as r
grows (max|h_t| ~1.4e16 at r=800 using the teacher's OWN theta*).

Scientific question: can DIMENSION-INDEPENDENT control of the algebra
multiplication norm stabilize the single long jet as r grows, and does
it need to be ||.||_1-based (1/r tail scaling) rather than 1/sqrt(r)?

Part 1: verify, numerically, the two algebra identities the scaling
argument rests on:
  (a) ||M_a||_inf = ||a||_1 EXACTLY, for the lower-triangular Toeplitz
      regular-representation matrix M_a used throughout B29-B34
      (make_M). This is an exact identity here (not just a bound): row
      r-1 of M_a contains every coefficient of a exactly once, so its
      abs-row-sum equals ||a||_1, and every other row is a strict
      partial sum <= ||a||_1 -- so the max row sum is ||a||_1 exactly.
  (b) ||a*b||_1 <= ||a||_1 ||b||_1 (submultiplicativity under truncated
      convolution -- a direct consequence of triangle inequality on
      the untruncated convolution sum).

Part 2: forward-only scaling experiment at r=64,200,500,800, four tail
conditions (A current / B sqrt(64/r) / C 64/r / D direct L1
renormalization to a fixed target), applied ONLY to the nilpotent tail
(index>=1) of theta and of kappa_t -- the scalar/semisimple component
(index 0) is left at its native, r-independent scale (an O(1) base
multiplier/timescale, not to be shrunk). No optimizer, no gradients, no
cross-family teacher/student comparison.

Run: python -m credit_memory.p2a_b34_dimension_norm_audit
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b34a_jet_algebra_correctness import (
    make_M, alg_mult, phi, make_gen_params as jet_make_gen_params, gen_forward as jet_gen_forward,
    X_DIM as JET_X_DIM,
)
from credit_memory.p2a_expressivity_credit_frontier import make_sequence, T_SEQ
from credit_memory.p2a_view2_matched_credit import jet_make_theta

R_LEVELS = (64, 200, 500, 800)
REF_R = 64


# =======================================================================
# PART 1: verify the two algebra-norm identities/inequalities.
# =======================================================================
def verify_algebra_norm_facts(seed=0, r_list=(4, 16, 64, 200)):
    print("=" * 78)
    print("PART 1: verify ||M_a||_inf = ||a||_1 (exact) and ||a*b||_1 <= ||a||_1||b||_1")
    print("=" * 78)
    rng = np.random.RandomState(seed)
    all_ok = True
    for r in r_list:
        a = jnp.array(rng.randn(r) * 0.3)
        b = jnp.array(rng.randn(r) * 0.3)
        M_a = make_M(a, r)
        row_sums = jnp.sum(jnp.abs(M_a), axis=1)
        col_sums = jnp.sum(jnp.abs(M_a), axis=0)
        inf_norm = float(jnp.max(row_sums))   # induced-infinity (max abs row sum)
        one_norm_induced = float(jnp.max(col_sums))  # induced-1 (max abs col sum)
        l1_a = float(jnp.sum(jnp.abs(a)))
        match_inf = abs(inf_norm - l1_a) < 1e-9
        match_one = abs(one_norm_induced - l1_a) < 1e-9

        ab = alg_mult(a, b, r)
        l1_ab = float(jnp.sum(jnp.abs(ab)))
        l1_a_l1_b = l1_a * float(jnp.sum(jnp.abs(b)))
        submult_ok = l1_ab <= l1_a_l1_b + 1e-9

        ok = match_inf and match_one and submult_ok
        all_ok &= ok
        print(f"  r={r:4d}  ||a||_1={l1_a:.4f}  ||M_a||_inf={inf_norm:.4f} (match={match_inf})  "
              f"||M_a||_1(induced)={one_norm_induced:.4f} (match={match_one})  "
              f"||a*b||_1={l1_ab:.4f} <= ||a||_1||b||_1={l1_a_l1_b:.4f}  (submult_ok={submult_ok})")
    print(f"ALL IDENTITIES/INEQUALITIES VERIFIED: {all_ok}")
    return all_ok


# =======================================================================
# PART 2: forward-only dimension-normalization experiment.
# =======================================================================
def tail_scale(vec, r, mode, target_l1=None, ref_r=REF_R):
    """vec: (r,) array; index 0 (scalar/semisimple component) is left
    UNTOUCHED at its native scale. Tail (index>=1) is rescaled per mode:
    'A' none, 'B' sqrt(ref_r/r), 'C' ref_r/r, 'D' renormalize tail L1
    norm to target_l1 exactly."""
    a0 = vec[0:1]
    tail = vec[1:]
    if mode == "A":
        new_tail = tail
    elif mode == "B":
        new_tail = tail * jnp.sqrt(ref_r / r)
    elif mode == "C":
        new_tail = tail * (ref_r / r)
    elif mode == "D":
        l1 = jnp.sum(jnp.abs(tail)) + 1e-12
        new_tail = tail * (target_l1 / l1)
    else:
        raise ValueError(mode)
    return jnp.concatenate([a0, new_tail])


def make_scaled_gen_forward(gen_params, r, mode, target_l1_kappa):
    def scaled_gen_forward(x_t):
        a_t, b_t, kappa_t, c_t = jet_gen_forward(x_t, gen_params, r)
        kappa_scaled = tail_scale(kappa_t, r, mode, target_l1_kappa)
        return a_t, b_t, kappa_scaled, c_t
    return scaled_gen_forward


def rollout_forward_only(theta, scaled_gen_forward_fn, xs, r):
    """Forward-only rollout (no training, no readout, no gradients),
    tracking the diagnostics requested: h_t, y_t, and the composite
    per-step multiplicative operator A_theta_t = a_t + kappa_t*theta
    (the algebra element whose regular-rep matrix actually multiplies
    h_t in y_t = M_{A_theta_t} h_t + ...)."""
    T = xs.shape[0]
    h = jnp.zeros(r, dtype=jnp.float64)
    h_norms, y_norms, Atheta_l1s, Atheta_Minf = [], [], [], []
    n_nonfinite = 0
    for t in range(T):
        x_t = jnp.stack([xs[t], 0.0, 0.0, 0.0])
        a_t, b_t, kappa_t, c_t = scaled_gen_forward_fn(x_t)
        A_theta_t = a_t + alg_mult(kappa_t, theta, r)
        y_t = alg_mult(A_theta_t, h, r) + alg_mult(b_t, theta, r) + c_t
        h_next = phi(y_t, r)
        finite = bool(jnp.all(jnp.isfinite(h_next)) and jnp.all(jnp.isfinite(y_t)))
        if not finite:
            n_nonfinite += 1
            h_norms.append(float("nan")); y_norms.append(float("nan"))
            Atheta_l1s.append(float("nan")); Atheta_Minf.append(float("nan"))
            h = jnp.nan_to_num(h_next, nan=0.0, posinf=1e300, neginf=-1e300)
            continue
        h_norms.append(float(jnp.linalg.norm(h_next)))
        y_norms.append(float(jnp.linalg.norm(y_t)))
        Atheta_l1s.append(float(jnp.sum(jnp.abs(A_theta_t))))
        Atheta_Minf.append(float(jnp.max(jnp.sum(jnp.abs(make_M(A_theta_t, r)), axis=1))))
        h = h_next
    return dict(h_norms=np.array(h_norms), y_norms=np.array(y_norms),
                Atheta_l1s=np.array(Atheta_l1s), Atheta_Minf=np.array(Atheta_Minf),
                n_nonfinite=n_nonfinite, T=T)


def run_dimension_normalization_experiment():
    print("\n" + "=" * 78)
    print("PART 2: forward-only dimension-normalization experiment (no training, no cross-family use)")
    print("=" * 78)

    # Reference (r=64, condition A / current) tail L1 norms, used as the
    # fixed target for condition D at every other r.
    theta_ref = jet_make_theta(1000, REF_R)
    ref_theta_tail_l1 = float(jnp.sum(jnp.abs(theta_ref[1:])))
    gen_params_ref = jet_make_gen_params(seed=3000 + REF_R, r=REF_R)
    h0_ref, xs_ref = make_sequence(20_000, T_SEQ, REF_R)
    kappa_tail_l1s_ref = []
    for t in range(T_SEQ):
        x_t = jnp.stack([xs_ref[t], 0.0, 0.0, 0.0])
        _, _, kappa_t, _ = jet_gen_forward(x_t, gen_params_ref, REF_R)
        kappa_tail_l1s_ref.append(float(jnp.sum(jnp.abs(kappa_t[1:]))))
    ref_kappa_tail_l1 = float(np.median(kappa_tail_l1s_ref))
    print(f"  Reference (r={REF_R}, condition A) target L1 norms for condition D: "
          f"theta_tail_l1={ref_theta_tail_l1:.4f}  kappa_tail_l1(median over T)={ref_kappa_tail_l1:.4f}")

    theta_seed_pairs = [("random_init", 1000), ("teacher_construction", 778)]
    conditions = ["A", "B", "C", "D"]

    header = (f"{'r':>5} {'cond':>5} {'source':>20} {'th_tail_l1':>11} {'M_theta_inf':>12} "
              f"{'kappa_tail_l1(med)':>19} {'max_Atheta_l1':>14} {'max_Atheta_Minf':>16} "
              f"{'max|h_t|':>12} {'max|y_t|':>12} {'n_nonfinite':>12}")
    print("\n  " + header)
    print("  " + "-" * len(header))

    rows = []
    for r in R_LEVELS:
        gen_params = jet_make_gen_params(seed=3000 + r, r=r)
        h0, xs = make_sequence(20_000, T_SEQ, r)
        for source_name, theta_seed in theta_seed_pairs:
            theta_native = jet_make_theta(theta_seed, r)
            for mode in conditions:
                target_l1 = ref_kappa_tail_l1 if mode == "D" else None
                theta_scaled = tail_scale(theta_native, r, mode,
                                           target_l1=ref_theta_tail_l1 if mode == "D" else None)
                scaled_gen_fwd = make_scaled_gen_forward(gen_params, r, mode, target_l1)

                th_tail_l1 = float(jnp.sum(jnp.abs(theta_scaled[1:])))
                M_theta_inf = float(jnp.max(jnp.sum(jnp.abs(make_M(theta_scaled, r)), axis=1)))
                # representative kappa tail L1 (median over T, post-scaling)
                kappa_tail_l1s = []
                for t in range(0, T_SEQ, 8):
                    x_t = jnp.stack([xs[t], 0.0, 0.0, 0.0])
                    _, _, kappa_t_s, _ = scaled_gen_fwd(x_t)
                    kappa_tail_l1s.append(float(jnp.sum(jnp.abs(kappa_t_s[1:]))))
                kappa_tail_l1_med = float(np.median(kappa_tail_l1s))

                res = rollout_forward_only(theta_scaled, scaled_gen_fwd, xs, r)
                max_h = float(np.nanmax(res["h_norms"])) if np.any(np.isfinite(res["h_norms"])) else float("inf")
                max_y = float(np.nanmax(res["y_norms"])) if np.any(np.isfinite(res["y_norms"])) else float("inf")
                max_Atheta_l1 = float(np.nanmax(res["Atheta_l1s"])) if np.any(np.isfinite(res["Atheta_l1s"])) else float("inf")
                max_Atheta_Minf = float(np.nanmax(res["Atheta_Minf"])) if np.any(np.isfinite(res["Atheta_Minf"])) else float("inf")

                print(f"  {r:5d} {mode:>5} {source_name:>20} {th_tail_l1:11.4f} {M_theta_inf:12.4f} "
                      f"{kappa_tail_l1_med:19.4f} {max_Atheta_l1:14.4e} {max_Atheta_Minf:16.4e} "
                      f"{max_h:12.4e} {max_y:12.4e} {res['n_nonfinite']:12d}")
                rows.append(dict(r=r, mode=mode, source=source_name, th_tail_l1=th_tail_l1,
                                  M_theta_inf=M_theta_inf, kappa_tail_l1_med=kappa_tail_l1_med,
                                  max_Atheta_l1=max_Atheta_l1, max_Atheta_Minf=max_Atheta_Minf,
                                  max_h=max_h, max_y=max_y, n_nonfinite=res["n_nonfinite"]))
    return rows


if __name__ == "__main__":
    ok = verify_algebra_norm_facts()
    rows = run_dimension_normalization_experiment()
    print(f"\nPart 1 identities verified: {ok}")
    print("Part 2 complete -- forward-only, no optimization, no cross-family use. See table above.")
    print("\nIMPORTANT CAVEAT (init-only scaling): this experiment controls the algebra/operator norm "
          "ONLY at construction time. If theta remains trainable under this scaling, gradient updates "
          "can subsequently grow ||theta_tail||_1 (or the corresponding kappa-analog, if ever made "
          "trainable) back out of a safe range. A durable fix -- if the single long jet is kept at all -- "
          "would need to constrain/project the algebra operator norm DURING training, not only at init, "
          "analogous to the R_V spectral-radius projection already used for the bounded-interface flag.")
