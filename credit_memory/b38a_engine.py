"""B38a -- end-to-end training with exact causal ProductLocal credit.

Architecture is B37c's, unchanged and imported (alg_mult, alg_add, alg_one,
flat, unflat, make_M, spec_from_blocks, generic_init, ...). No global quotient
q, no companion matrix, no CRT transform, no parameter-dependent basis.

Index convention: B37c's  y_t = C_out z_{t+1}  is kept (the B37b/c teachers are
generated as y_t = C_* h_{t+1}), so the spec's y_t = C_out z_t is the same
equation under a relabelling of z. The trace recursions are unaffected:
    e^u_{t+1}   = u e^u_t + z_t
    e^{b_c}_{t+1} = u e^{b_c}_t + x_{c,t} 1_A

Gradients use a NATIVE adjoint multiplication: M_e^T is never materialized.
  real factor:    (M_e^T q)_l = sum_{m} e_m q_{m+l}
  complex factor: (M_e^T q)_l = sum_{m} conj(e_m) q_{m+l}
so the whole gradient path is O(P) work and O(P) memory per step, with no tape.
"""
from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b37c_productlocal_native import (      # frozen B37c architecture
    spec_from_blocks, real_dim, kdim, alg_mult, alg_add, alg_scale, alg_zero,
    alg_one, flat, unflat, make_M, spectral_radius, generic_init, exact_init,
    to_jax)


def transpose_mult(e, q, spec):
    """Adjoint of multiplication-by-e, computed natively (no r x r matrix)."""
    out = []
    for (k, d), ej, qj in zip(spec, e, q):
        if k == "R":
            c = jnp.stack([jnp.sum(ej[:d - l, 0] * qj[l:, 0]) for l in range(d)])[:, None]
        else:
            er, ei, qr, qi = ej[:, 0], ej[:, 1], qj[:, 0], qj[:, 1]
            cr = jnp.stack([jnp.sum(er[:d - l] * qr[l:] + ei[:d - l] * qi[l:]) for l in range(d)])
            ci = jnp.stack([jnp.sum(er[:d - l] * qi[l:] - ei[:d - l] * qr[l:]) for l in range(d)])
            c = jnp.stack([cr, ci], axis=1)
        out.append(c)
    return tuple(out)


# ----------------------------------------------------------- chunk forward
def chunk_forward(params, z0, xs, spec):
    """z carried in, states out. Used by both arms."""
    u, b, C = params["u"], params["b"], params["C"]

    def step(z, x):
        z = alg_add(alg_mult(u, z, spec), alg_scale(b, x))
        return z, flat(z)
    zT, zs = jax.lax.scan(step, z0, xs)
    return zT, zs


def chunk_loss(params, z0, xs, ys, spec, denom):
    """Chunk loss with the incoming hidden state treated as fixed."""
    z0 = jax.tree.map(jax.lax.stop_gradient, z0)
    _, zs = chunk_forward(params, z0, xs, spec)
    pred = zs @ params["C"][0]
    return jnp.sum((pred - ys) ** 2) / denom


# ------------------------------------------- ARM B: forward causal RTRL grad
def rtrl_chunk_grad(params, z0, xs, ys, spec, denom):
    """Exact chunk gradient from forward eligibility states only.
    Eligibility is RESET at the chunk boundary (e_0 = 0); z0 enters fixed.
    No autodiff, no tape: a single forward scan carrying (z, e^u, e^b, grads)."""
    u, b, C = params["u"], params["b"], params["C"]
    one = alg_one(spec)
    r = real_dim(spec)

    def step(carry, inp):
        z, eu, eb, gu, gb, gC = carry
        x, ystar = inp
        zt = z                                              # PRE-update state
        znew = alg_add(alg_mult(u, zt, spec), alg_scale(b, x))
        eu = alg_add(alg_mult(u, eu, spec), zt)
        eb = alg_add(alg_mult(u, eb, spec), alg_scale(one, x))
        zf = flat(znew)
        dldy = 2.0 * (jnp.dot(C[0], zf) - ystar) / denom
        q = unflat(dldy * C[0], spec)                       # dl_t / dz_{t+1}
        gu = alg_add(gu, transpose_mult(eu, q, spec))
        gb = alg_add(gb, transpose_mult(eb, q, spec))
        gC = gC + (dldy * zf).reshape(1, r)
        return (znew, eu, eb, gu, gb, gC), None

    z_ = alg_zero(spec)
    init = (z0, z_, z_, z_, z_, jnp.zeros((1, r)))
    (zT, _, _, gu, gb, gC), _ = jax.lax.scan(step, init, (xs, ys))
    return dict(u=gu, b=gb, C=gC), zT


def rtrl_batch_grad(params, z0b, xsb, ysb, spec, denom):
    gs, zT = jax.vmap(lambda z, x, y: rtrl_chunk_grad(params, z, x, y, spec, denom)
                      )(z0b, xsb, ysb)
    return jax.tree.map(lambda a: jnp.sum(a, axis=0), gs), zT


# ---------------------------------------------- ARM A: matched TBPTT grad
def tbptt_batch_grad(params, z0b, xsb, ysb, spec, denom):
    def loss(p):
        return jnp.sum(jax.vmap(lambda z, x, y: chunk_loss(p, z, x, y, spec, denom)
                                )(z0b, xsb, ysb))
    g = jax.grad(loss)(params)
    zT, _ = jax.vmap(lambda z, x: chunk_forward(params, z, x, spec))(z0b, xsb)
    return g, zT


# ------------------------------------------------------------- Adam (shared)
def adam_step(params, m, v, g, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = jax.tree.map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, g)
    v = jax.tree.map(lambda vi, gi: b2 * vi + (1 - b2) * gi ** 2, v, g)
    mh = jax.tree.map(lambda x: x / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda x: x / (1 - b2 ** t), v)
    params = jax.tree.map(lambda p, a, c: p - lr * a / (jnp.sqrt(c) + eps), params, mh, vh)
    return params, m, v, t


@functools.lru_cache(maxsize=None)
def build_update(spec, L, arm):
    """One TBPTT/RTRL chunk update. Arms A and B share EVERYTHING except which
    routine produces the gradient."""
    gradf = tbptt_batch_grad if arm == "A" else rtrl_batch_grad

    def upd(params, m, v, t, z0b, xsb, ysb, lr, denom):
        g, zT = gradf(params, z0b, xsb, ysb, spec, denom)
        params, m, v, t = adam_step(params, m, v, g, t, lr)
        return params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT), g
    return jax.jit(upd)


@functools.lru_cache(maxsize=None)
def build_both(spec, L):
    """Both gradients at the SAME point, for the in-training identity check."""
    def both(params, z0b, xsb, ysb, denom):
        ga, _ = tbptt_batch_grad(params, z0b, xsb, ysb, spec, denom)
        gb, _ = rtrl_batch_grad(params, z0b, xsb, ysb, spec, denom)
        return ga, gb
    return jax.jit(both)


@functools.lru_cache(maxsize=None)
def build_eval(spec):
    def ev(params, xsb, ysb):
        z0 = alg_zero(spec)
        pred = jax.vmap(lambda x: chunk_forward(params, z0, x, spec)[1] @ params["C"][0])(xsb)
        return jnp.mean((pred - ysb) ** 2)
    return jax.jit(ev)


# ------------------------------------------ ARM C: every-token online updates
@functools.lru_cache(maxsize=None)
def build_online_scan(spec):
    """ARM C -- genuine streaming: theta_t -> theta_{t+1} after EVERY observed
    loss, with the eligibility CARRIED across parameter updates (never reset).

    This is NOT a numerical reproduction of BPTT: once theta changes every step
    there is no single fixed parameter vector that generated the history. The
    carried trace is the exact sensitivity under the corresponding fixed /
    path-shift interpretation; here it is used as an online learning signal and
    Arm C is judged as an online learning algorithm, not as an identity. The
    optimizer itself is never differentiated through."""
    r = real_dim(spec)
    one = alg_one(spec)

    def run(params, xs, ys, lr):
        def step(carry, inp):
            params, m, v, t, z, eu, eb = carry
            x, ystar = inp
            u, b, C = params["u"], params["b"], params["C"]
            zt = z
            znew = alg_add(alg_mult(u, zt, spec), alg_scale(b, x))
            eu = alg_add(alg_mult(u, eu, spec), zt)
            eb = alg_add(alg_mult(u, eb, spec), alg_scale(one, x))
            zf = flat(znew)
            err = jnp.dot(C[0], zf) - ystar
            dldy = 2.0 * err
            q = unflat(dldy * C[0], spec)
            g = dict(u=transpose_mult(eu, q, spec), b=transpose_mult(eb, q, spec),
                     C=(dldy * zf).reshape(1, r))
            params, m, v, t = adam_step(params, m, v, g, t, lr)
            gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g)))
            return (params, m, v, t, znew, eu, eb), (err ** 2, gn)
        m0 = jax.tree.map(jnp.zeros_like, params)
        v0 = jax.tree.map(jnp.zeros_like, params)
        z_ = alg_zero(spec)
        carry = (params, m0, v0, jnp.array(0.0), z_, z_, z_)
        (params, _, _, _, _, _, _), (sq, gn) = jax.lax.scan(step, carry, (xs, ys))
        return params, sq, gn
    return jax.jit(run)
