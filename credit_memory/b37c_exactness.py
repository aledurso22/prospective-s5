"""B37c section 3 -- exact reduced eligibilities, verified before any training.

Three independent paths compared to machine precision:
  (1) reduced traces  e^u_{t+1} = u e^u_t + z_t,  e^b_{t+1} = u e^b_t + x_t 1
      with grad_u = sum_t M_{e^u_t}^T q_t,  grad_b = sum_t M_{e^b_t}^T q_t
  (2) full RTRL carrying dense sensitivity matrices S_t in R^{r x r}
  (3) autodiff/BPTT through the whole rollout
Also verifies M_{e_t} == S_t entrywise, the algebra homomorphism M_a M_b = M_{a*b},
and M_a v = flat(a * v).
"""
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b37c_productlocal_native import (
    spec_from_blocks, real_dim, alg_mult, alg_add, alg_scale, alg_zero, alg_one,
    make_M, flat, unflat, rollout, generic_init, to_jax)
from credit_memory.b37b_quotient_trainability import make_teacher, FAMILIES


def reduced_grads(params, xs, ys, spec):
    """Path (1): O((m+1)r) persistent state."""
    u, b, C = params["u"], params["b"], params["C"]
    r = real_dim(spec)
    z = alg_zero(spec, np_=np); z = tuple(np.zeros_like(x) for x in z)
    eu = tuple(np.zeros_like(x) for x in z)
    eb = tuple(np.zeros_like(x) for x in z)
    one = [np.zeros((d, 2 if k == "C" else 1)) for k, d in spec]
    for o in one: o[0, 0] = 1.0
    one = tuple(one)
    ju = tuple(jnp.asarray(x) for x in u)
    gu, gb = np.zeros(r), np.zeros(r)
    gC = np.zeros_like(np.asarray(C))
    T = len(xs)
    for t in range(T):
        zt = z                                          # PRE-update state
        eu_t, eb_t = eu, eb
        # state update
        z = tuple(np.asarray(v) for v in alg_add(
            alg_mult(ju, tuple(jnp.asarray(v) for v in zt), spec),
            alg_scale(tuple(jnp.asarray(v) for v in b), float(xs[t]))))
        # trace update (uses PRE-update z_t and x_t, exactly as specified)
        eu = tuple(np.asarray(v) for v in alg_add(
            alg_mult(ju, tuple(jnp.asarray(v) for v in eu_t), spec),
            tuple(jnp.asarray(v) for v in zt)))
        eb = tuple(np.asarray(v) for v in alg_add(
            alg_mult(ju, tuple(jnp.asarray(v) for v in eb_t), spec),
            alg_scale(tuple(jnp.asarray(v) for v in one), float(xs[t]))))
        zf = np.asarray(flat(tuple(jnp.asarray(v) for v in z)))
        yt = float((np.asarray(C) @ zf)[0])
        dldy = 2.0 * (yt - float(ys[t])) / T
        qt = dldy * np.asarray(C)[0]                    # dl_t / dz_{t+1}
        gu += make_M(eu, spec).T @ qt
        gb += make_M(eb, spec).T @ qt
        gC += dldy * zf.reshape(1, -1)
    return gu, gb, gC, eu, eb


def full_rtrl_grads(params, xs, ys, spec):
    """Path (2): dense r x r sensitivity matrices."""
    u, b, C = params["u"], params["b"], params["C"]
    r = real_dim(spec)
    Mu = make_M(u, spec)
    z = np.zeros(r)
    Su, Sb = np.zeros((r, r)), np.zeros((r, r))
    gu, gb = np.zeros(r), np.zeros(r)
    gC = np.zeros_like(np.asarray(C))
    ju = tuple(jnp.asarray(x) for x in u)
    T = len(xs)
    for t in range(T):
        zt = z.copy()
        Su_t, Sb_t = Su.copy(), Sb.copy()
        z = Mu @ zt + np.asarray(flat(tuple(jnp.asarray(v) for v in b))) * float(xs[t])
        Su = Mu @ Su_t + make_M(unflat(jnp.asarray(zt), spec), spec)
        Sb = Mu @ Sb_t + np.eye(r) * float(xs[t])
        yt = float((np.asarray(C) @ z)[0])
        dldy = 2.0 * (yt - float(ys[t])) / T
        qt = dldy * np.asarray(C)[0]
        gu += Su.T @ qt
        gb += Sb.T @ qt
        gC += dldy * z.reshape(1, -1)
    return gu, gb, gC, Su, Sb


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / (1 + np.linalg.norm(b)))


print("=" * 104)
print("ALGEBRA IDENTITIES (fixed structural constants)")
print("=" * 104)
worst = dict(hom=0.0, act=0.0)
for f in FAMILIES:
    for r in (4, 8):
        sp = spec_from_blocks(make_teacher(f, r, 0)["blocks"])
        rng = np.random.RandomState(3)
        A = tuple(rng.randn(d, 2 if k == "C" else 1) for k, d in sp)
        B = tuple(rng.randn(d, 2 if k == "C" else 1) for k, d in sp)
        AB = alg_mult(tuple(jnp.asarray(x) for x in A), tuple(jnp.asarray(x) for x in B), sp)
        worst["hom"] = max(worst["hom"], rel(make_M(A, sp) @ make_M(B, sp),
                                            make_M(tuple(np.asarray(x) for x in AB), sp)))
        worst["act"] = max(worst["act"], rel(make_M(A, sp) @ np.asarray(flat(
            tuple(jnp.asarray(x) for x in B))), np.asarray(flat(AB))))
print(f"  max rel err  M_a M_b = M_(a*b) : {worst['hom']:.3e}")
print(f"  max rel err  M_a v   = a * v   : {worst['act']:.3e}")

print()
print("=" * 104)
print("EXACT REDUCED ELIGIBILITIES vs FULL RTRL vs AUTODIFF   (T=25, m=1)")
print("=" * 104)
print(f"{'family':21s} {'r':>2s} {'red vs AD (u)':>14s} {'red vs AD (b)':>14s} {'red vs AD (C)':>14s} "
      f"{'red vs full':>12s} {'M_e == S':>10s}")
allmax = 0.0
for f in FAMILIES:
    for r in (4, 8):
        t = make_teacher(f, r, 0)
        sp = spec_from_blocks(t["blocks"])
        p = generic_init(sp, 0)
        rng = np.random.RandomState(5)
        xs, ys = rng.randn(25) * 0.5, rng.randn(25) * 0.5
        gu, gb, gC, eu_f, eb_f = reduced_grads(p, xs, ys, sp)
        fu, fb, fC, Su, Sb = full_rtrl_grads(p, xs, ys, sp)
        jp = to_jax(p)
        loss = lambda q: jnp.mean((rollout(q, jnp.asarray(xs), sp) - jnp.asarray(ys)) ** 2)
        g = jax.grad(loss)(jp)
        au = np.concatenate([np.asarray(x).ravel() for x in g["u"]])
        ab = np.concatenate([np.asarray(x).ravel() for x in g["b"]])
        e1, e2 = rel(gu, au), rel(gb, ab)
        e3, e4 = rel(gC, np.asarray(g["C"])), max(rel(gu, fu), rel(gb, fb))
        # the reduction itself: dense full-RTRL S_t must equal M_{e_t} of the r-vector trace
        e5 = max(rel(Su, make_M(eu_f, sp)), rel(Sb, make_M(eb_f, sp)))
        allmax = max(allmax, e1, e2, e3, e4, e5)
        print(f"{f:21s} {r:2d} {e1:14.3e} {e2:14.3e} {e3:14.3e} {e4:12.3e} {e5:10.3e}")
print(f"\n  worst relative error across all paths and families: {allmax:.3e}"
      f"   (preregistered threshold 1e-10: {'PASS' if allmax < 1e-10 else 'FAIL'})")

print()
print("=" * 104)
print("STORAGE AND ARITHMETIC SCALING (measured, not optimized)")
print("=" * 104)
print(f"{'family':21s} {'r':>2s} {'pi':>26s} {'P_dyn=(m+1)r':>13s} {'full RTRL':>10s} {'ratio':>6s} "
      f"{'mult flops':>11s} {'dense r^2':>10s}")
for f in FAMILIES:
    for r in (4, 8):
        sp = spec_from_blocks(make_teacher(f, r, 0)["blocks"])
        rr, m = real_dim(sp), 1
        red, full = (m + 1) * rr, (m + 1) * rr * rr
        fl = sum((d * (d + 1) // 2) * (4 if k == "C" else 1) for k, d in sp)
        print(f"{f:21s} {r:2d} {''.join(f'{k}{d}' for k, d in sp):>26s} {red:13d} {full:10d} "
              f"{full//red:5d}x {fl:11d} {rr*rr:10d}")
