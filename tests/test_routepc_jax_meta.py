"""Validation gates for the Route-A meta machinery in the JAX pipeline
(train_bench.py). Run BEFORE any cluster job.

  1. rotation   ssm_online_rot backward == conj(w) scaling of the plain
                online blocks (and == the tcg numpy rig's online gradient
                scaled by hand), w=1 reproduces ssm_online bit-for-bit.
  2. meta-grad  the nested autodiff meta-gradient (grad of the one-step-
                lookahead teacher loss w.r.t. w, through the custom-VJP
                backward) matches central finite differences in float64.
  3. fallback   an OnlineS5SSM model applied WITHOUT the meta collection
                is bitwise identical to meta = 1+0j.
  4. teacher    remap_for_teacher(params) through the baseline model
                reproduces the online model's forward.

Run:  python check_routeA_meta.py
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from toyrig import ssm_rig as tcg          # noqa: E402
from ssm.online_s5.scan import ssm_online, ssm_online_rot   # noqa: E402
from ssm.model import build_model            # noqa: E402
from train_bench import remap_for_teacher, make_meta        # noqa: E402


def gate_rotation() -> bool:
    """conj(w) scaling vs the tcg rig's online gradient (scaled by hand)."""
    N, T, B = 4, 48, 3
    rng = np.random.RandomState(7)
    a = np.linspace(0.9, 0.98, N) * np.exp(1j * rng.uniform(-1, 1, N))
    Bm = (rng.randn(N, 1) + 1j * rng.randn(N, 1)) / np.sqrt(2)
    c = (rng.randn(N) + 1j * rng.randn(N)) / np.sqrt(2)
    x = rng.randn(T, B)
    y = rng.randn(T, B)
    w = (0.8 * np.exp(1j * rng.uniform(-1, 1, N))).astype(np.complex128)

    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 1, N, T, 1, 1, B
    params = dict(rho=[np.zeros(N)], theta=[np.zeros(N)],
                  b=[Bm.copy()], c=c.copy())
    params["a"] = [a.copy()]
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    G = tcg.assemble(params, h, x, r, q, Sa, Sb)
    ga_np = np.conj(w) * G["a"][0]
    gb_np = np.conj(w) * G["b"][0][:, 0]

    ga_jx = np.zeros(N, complex)
    gb_jx = np.zeros(N, complex)
    for b in range(B):
        xb = jnp.asarray(x[:, b])
        yb = y[:, b]
        yhat_b, vjp_fn = jax.vjp(ssm_online_rot,
                                 jnp.asarray(a.real), jnp.asarray(a.imag),
                                 jnp.asarray(Bm[:, 0].real),
                                 jnp.asarray(Bm[:, 0].imag),
                                 jnp.asarray(c.real), jnp.asarray(c.imag),
                                 0.0, xb,
                                 jnp.asarray(w.real), jnp.asarray(w.imag))
        dy = jnp.asarray(np.asarray(yhat_b) - yb)
        ct = vjp_fn(dy)
        ga_jx += np.asarray(ct[0]) - 1j * np.asarray(ct[1])
        gb_jx += np.asarray(ct[2]) - 1j * np.asarray(ct[3])

    # w = 1 must be the plain online rule. This is a TRAINING-mode claim,
    # so run it in float32 (x64 only perturbs ssm_online's hardcoded
    # complex64 cast vs the rot variant's promotion).
    jax.config.update("jax_enable_x64", False)
    y1, v1 = jax.vjp(ssm_online, jnp.asarray(a.real), jnp.asarray(a.imag),
                     jnp.asarray(Bm[:, 0].real), jnp.asarray(Bm[:, 0].imag),
                     jnp.asarray(c.real), jnp.asarray(c.imag), 0.0,
                     jnp.asarray(x[:, 0].astype(np.float32)))
    y2, v2 = jax.vjp(ssm_online_rot, jnp.asarray(a.real), jnp.asarray(a.imag),
                     jnp.asarray(Bm[:, 0].real), jnp.asarray(Bm[:, 0].imag),
                     jnp.asarray(c.real), jnp.asarray(c.imag), 0.0,
                     jnp.asarray(x[:, 0].astype(np.float32)),
                     jnp.ones(N, jnp.float32), jnp.zeros(N, jnp.float32))
    dy0 = jnp.asarray((np.asarray(y1) - y[:, 0]).astype(np.float32))
    ct1, ct2 = v1(dy0), v2(dy0)
    bit = max(float(jnp.max(jnp.abs(u - v))) for u, v in zip(ct1, ct2))
    jax.config.update("jax_enable_x64", True)

    def rel(u, v):
        return float(np.max(np.abs(u - v)) / (np.max(np.abs(u)) + 1e-12))

    ok = (max(rel(ga_np, ga_jx), rel(gb_np, gb_jx)) < 1e-3
          and bit == 0.0)
    print(f"[1] rotation: rel diff vs rig (a) {rel(ga_np, ga_jx):.2e}  "
          f"(B) {rel(gb_np, gb_jx):.2e}   w=1 bitwise (float32): {bit}")
    return ok


def _tiny_models():
    H, N, L, T = 4, 4, 1, 32
    online = build_model("online", d_model=H, state_size=N, n_layers=L,
                         n_classes=3, dropout_rate=0.0, seq2seq=False)
    teacher = build_model("baseline", d_model=H, state_size=N, n_layers=L,
                          n_classes=3, dropout_rate=0.0, seq2seq=False)
    key = jax.random.PRNGKey(0)
    params = online.init({"params": key, "dropout": key},
                         jnp.ones((1, T)), train=False)["params"]
    params = jax.tree_util.tree_map(
        lambda z: z.astype(np.float64), params)
    return online, teacher, params, H, N, T


def gate_meta_grad() -> bool:
    """Nested-autodiff meta-gradient vs central finite differences (x64)."""
    online, teacher, params, H, N, T = _tiny_models()
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.randn(8, T))
    y = jnp.asarray(rng.randint(0, 3, 8))
    lr = 1.0     # scales the meta-gradient to O(1e-5), far above the
                 # float64 FD noise floor; the gate tests the autodiff
                 # machinery, not the lr value

    meta0 = make_meta(params, H, N)
    meta0 = jax.tree_util.tree_map(
        lambda z: jnp.asarray(z, jnp.float64), meta0)

    def look(m):
        def loss_p(p):
            logits = online.apply({"params": p, "meta": m}, x, train=False)
            ce = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.take_along_axis(
                ce, y[:, None], axis=1))
        g = jax.grad(loss_p)(params)
        p_next = jax.tree_util.tree_map(
            lambda pp, gg: pp - lr * gg, params, g)
        logits = teacher.apply({"params": remap_for_teacher(p_next)}, x,
                               train=False)
        ce = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.take_along_axis(ce, y[:, None], axis=1))

    gw = jax.grad(look)(meta0)
    gw_max = float(max(jnp.max(jnp.abs(z)) for z in
                       jax.tree_util.tree_leaves(gw)))

    # central FD on a few entries of each leaf
    keys = sorted(meta0.keys())
    eps = 1e-6
    rng2 = np.random.RandomState(2)
    worst = 0.0
    for k in keys:
        for leaf in ["w_re", "w_im"]:
            W = np.asarray(meta0[k][leaf])
            idxs = [(int(a), int(b)) for a, b in
                    zip(rng2.randint(0, H, 4), rng2.randint(0, N, 4))]
            for idx in idxs:
                def perturb(delta):
                    W2 = W.copy()
                    W2[idx] += delta
                    return {kk: {lf: jnp.asarray(
                                     W2 if (kk == k and lf == leaf)
                                     else np.asarray(meta0[kk][lf]))
                                 for lf in ["w_re", "w_im"]}
                            for kk in keys}
                fp, fm = look(perturb(eps)), look(perturb(-eps))
                fd = float((fp - fm) / (2 * eps))
                ad = float(np.asarray(gw[k][leaf])[idx])
                # referenced to the gradient NORM (per-entry relative error
                # on near-zero entries measures FD rounding noise, not the
                # autodiff chain)
                worst = max(worst, abs(fd - ad) / gw_max)
                print(f"    dlook/d{leaf}[{k}]{idx}: autodiff {ad:+.8e}  "
                      f"FD {fd:+.8e}")
    print(f"[2] meta-grad FD gate: worst norm-referenced diff {worst:.2e}  "
          f"(bar 1e-3)")
    return worst < 1e-3


def gate_fallback() -> bool:
    """No meta collection == meta = 1+0j, bitwise (params grads)."""
    online, _, params, H, N, T = _tiny_models()
    rng = np.random.RandomState(3)
    x = jnp.asarray(rng.randn(4, T))
    meta1 = make_meta(params, H, N)

    def loss_p(p, m):
        return jnp.sum(online.apply({"params": p, "meta": m}, x,
                                    train=False) ** 2)

    g0 = jax.grad(loss_p)(params, {})
    g1 = jax.grad(loss_p)(params, meta1)
    d = max(float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree_util.tree_leaves(g0),
                            jax.tree_util.tree_leaves(g1)))
    print(f"[3] fallback: max |dgrad| no-meta vs w=1: {d}")
    return d == 0.0


def gate_teacher() -> bool:
    """remap_for_teacher makes the baseline model reproduce the online
    forward. (1e-5: the baseline scan internals downcast to complex64,
    and the two scan implementations reorder float ops.)"""
    online, teacher, params, H, N, T = _tiny_models()
    rng = np.random.RandomState(4)
    x = jnp.asarray(rng.randn(4, T))
    yo = online.apply({"params": params}, x, train=False)
    yt = teacher.apply({"params": remap_for_teacher(params)}, x,
                       train=False)
    d = float(jnp.max(jnp.abs(yo - yt)))
    print(f"[4] teacher remap: max |dlogit| {d:.2e}")
    return d < 1e-5


def main() -> None:
    ok = [gate_rotation(), gate_fallback(), gate_teacher(), gate_meta_grad()]
    print("-" * 70)
    print("ALL GATES PASS" if all(ok) else "GATE FAILURE — fix before "
          "the cluster grid")
    assert all(ok)


if __name__ == "__main__":
    main()
