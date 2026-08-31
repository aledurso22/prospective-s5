"""B38b section 6 -- training on a selectivity-dependent task.

TASK (gated accumulator with input-driven reset):
    z_{t+1} = (1 - g_t) * lam * z_t + v_t,     y_t = z_{t+1}
    x_t = (v_t, g_t, noise, noise),  g_t ~ Bernoulli(p)
The decay a_t = (1-g_t)*lam is INPUT-DEPENDENT, so no input-independent LTI can
express it. A non-selective ablation of the SAME architecture is trained as a
control to confirm the task actually requires selectivity.

Arms: A = matched TBPTT (autodiff), B = exact source-local causal RTRL,
C = every-token online with the eligibility carried across parameter updates.
"""
from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b38b_selective import (
    Q_BOTT, M_LOC, init_L, fixed_R, nparams, tile_step_L, flat_tile_L, step_L)

LAM, P_GATE = 0.9, 0.15
M_IN = 4
LR_GRID = (3e-3, 1e-2, 3e-2)
EPOCHS, EP_BLOCK = 200, 10
N_TRAIN, N_VAL, N_TEST, T_SEQ = 16, 8, 8, 256


def make_data(n, T, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, T)
    g = (rng.rand(n, T) < P_GATE).astype(float)
    noise = rng.randn(n, T, 2) * 0.5
    xs = np.concatenate([v[:, :, None], g[:, :, None], noise], axis=2)
    ys = np.zeros((n, T))
    z = np.zeros(n)
    for t in range(T):
        z = (1 - g[:, t]) * LAM * z + v[:, t]
        ys[:, t] = z
    return jnp.asarray(xs), jnp.asarray(ys)


def flat_all(params, J, d):
    return jnp.stack([flat_tile_L(params, j, d, Q_BOTT) for j in range(J)])


def unflat_all(th, J, d):
    q = Q_BOTT
    o = 0
    out = {}
    out["W"] = th[:, o:o + q * M_LOC].reshape(J, q, M_LOC); o += q * M_LOC
    out["p"] = th[:, o:o + q]; o += q
    out["Atil"] = th[:, o:o + d]; o += d
    out["uD"] = th[:, o:o + d * q].reshape(J, d, q); o += d * q
    out["cD"] = th[:, o:o + d]; o += d
    out["uB"] = th[:, o:o + d * q].reshape(J, d, q); o += d * q
    out["cB"] = th[:, o:o + d]
    return out


def chunk_fwd(params, R, h0, xs, J, d, ablate=False):
    def st(h, x):
        hn, _ = step_L(params, R * 0.0 if ablate else R, h, x, d)
        return hn, params["C"][0] @ hn.reshape(-1)
    hT, ys = jax.lax.scan(st, h0, xs)
    return hT, ys


def chunk_loss(params, R, h0, xs, ys, J, d, denom, ablate=False):
    h0 = jax.lax.stop_gradient(h0)
    _, pred = chunk_fwd(params, R, h0, xs, J, d, ablate)
    return jnp.sum((pred - ys) ** 2) / denom


def tbptt_grad(params, R, h0b, xsb, ysb, J, d, denom, ablate=False):
    def L(p):
        return jnp.sum(jax.vmap(lambda h, x, y: chunk_loss(
            p, R, h, x, y, J, d, denom, ablate))(h0b, xsb, ysb))
    g = jax.grad(L)(params)
    hT = jax.vmap(lambda h, x: chunk_fwd(params, R, h, x, J, d, ablate)[0])(h0b, xsb)
    return g, hT


def rtrl_grad_one(params, R, h0, xs, ys, J, d, denom, ablate=False):
    """Exact source-local RTRL: per-tile E in R^{d x P_tau}, O(1) per tile."""
    q = Q_BOTT
    th = flat_all(params, J, d)
    Ptau = th.shape[1]
    C2 = params["C"][0].reshape(J, d)
    Rz = R * 0.0 if ablate else R

    def step(carry, inp):
        h, E, gt, gC = carry
        x, ystar = inp
        xi = x[0]
        xl = jnp.einsum("jkm,m->jk", Rz, x)
        G = jax.vmap(lambda t, hh, xx: jax.jacrev(
            lambda tt: tile_step_L(tt, jax.lax.stop_gradient(hh), xx, xi, d, q)[0])(t)
        )(th, h, xl)
        hn, a = jax.vmap(lambda t, hh, xx: tile_step_L(t, hh, xx, xi, d, q))(th, h, xl)
        E = a[:, :, None] * E + G                       # J_t is exactly diagonal
        hf = hn.reshape(-1)
        dldy = 2.0 * (jnp.dot(params["C"][0], hf) - ystar) / denom
        qt = dldy * C2
        gt = gt + jnp.einsum("jdp,jd->jp", E, qt)
        gC = gC + (dldy * hf).reshape(1, -1)
        return (hn, E, gt, gC), None
    init = (h0, jnp.zeros((J, d, Ptau)), jnp.zeros((J, Ptau)),
            jnp.zeros_like(params["C"]))
    (hT, _, gt, gC), _ = jax.lax.scan(step, init, (xs, ys))
    return gt, gC, hT


def rtrl_grad(params, R, h0b, xsb, ysb, J, d, denom, ablate=False):
    gt, gC, hT = jax.vmap(lambda h, x, y: rtrl_grad_one(
        params, R, h, x, y, J, d, denom, ablate))(h0b, xsb, ysb)
    g = unflat_all(jnp.sum(gt, axis=0), J, d)
    g["C"] = jnp.sum(gC, axis=0)
    return g, hT


def adam(params, m, v, g, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b ** 2, v, g)
    mh = jax.tree.map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda a: a / (1 - b2 ** t), v)
    return jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps),
                        params, mh, vh), m, v, t


@functools.lru_cache(maxsize=None)
def build_epochs(J, d, L, arm, n_ep, B, ablate, check=False):
    gf = tbptt_grad if arm == "A" else rtrl_grad
    denom = float(B * L)

    def relerr(a, b):
        fa = jnp.concatenate([x.ravel() for x in jax.tree.leaves(a)])
        fb = jnp.concatenate([x.ravel() for x in jax.tree.leaves(b)])
        return jnp.linalg.norm(fa - fb) / (1.0 + jnp.linalg.norm(fb))

    def run(params, m, v, t, R, xs, ys, lr):
        nch = xs.shape[1] // L
        xc = xs[:, :nch * L].reshape(B, nch, L, M_IN).transpose(1, 0, 2, 3)
        yc = ys[:, :nch * L].reshape(B, nch, L).transpose(1, 0, 2)

        def chunk(carry, cc):
            params, m, v, t, h = carry
            x, y = cc
            if check:
                ga, _ = tbptt_grad(params, R, h, x, y, J, d, denom, ablate)
                gb, hT = rtrl_grad(params, R, h, x, y, J, d, denom, ablate)
                e = jnp.stack([relerr(gb[k], ga[k]) for k in
                               ("Atil", "uD", "cD", "uB", "cB", "W", "p", "C")])
                g = gb
            else:
                g, hT = gf(params, R, h, x, y, J, d, denom, ablate)
                e = jnp.zeros(8)
            params, m, v, t = adam(params, m, v, g, t, lr)
            return (params, m, v, t, jax.lax.stop_gradient(hT)), e

        def epoch(carry, _):
            params, m, v, t = carry
            (params, m, v, t, _), e = jax.lax.scan(
                chunk, (params, m, v, t, jnp.zeros((B, J, d))), (xc, yc))
            return (params, m, v, t), jnp.max(e, axis=0)
        (params, m, v, t), e = jax.lax.scan(epoch, (params, m, v, t), None, length=n_ep)
        return params, m, v, t, jnp.max(e, axis=0)
    return jax.jit(run)


@functools.lru_cache(maxsize=None)
def build_eval(J, d, ablate):
    def ev(params, R, xsb, ysb):
        pred = jax.vmap(lambda x: chunk_fwd(params, R, jnp.zeros((J, d)), x, J, d,
                                            ablate)[1])(xsb)
        return jnp.mean((pred - ysb) ** 2)
    return jax.jit(ev)


def train(J, d, seed, arm, L, lr, ablate=False, check=False, epochs=EPOCHS):
    R = fixed_R(J, M_IN)
    p = init_L(J, d, M_IN, seed)
    xtr, ytr = make_data(N_TRAIN, T_SEQ, 6000 + seed)
    xva, yva = make_data(N_VAL, T_SEQ, 7000 + seed)
    xte, yte = make_data(N_TEST, T_SEQ, 8000 + seed)
    ev = build_eval(J, d, ablate)
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    t = jnp.array(0.0)
    run = build_epochs(J, d, L, arm, EP_BLOCK, N_TRAIN, ablate, check)
    best_val, best = float(ev(p, R, xva, yva)), p
    worst = np.zeros(8)
    curve = []
    for _ in range(epochs // EP_BLOCK):
        p, m, v, t, e = run(p, m, v, t, R, xtr, ytr, lr)
        worst = np.maximum(worst, np.asarray(e))
        vl = float(ev(p, R, xva, yva))
        curve.append(vl)
        if np.isfinite(vl) and vl < best_val:
            best_val, best = vl, p
    yn = float(jnp.mean(yte ** 2))
    return dict(test_nmse=float(ev(best, R, xte, yte)) / yn, val_loss=best_val,
                curve=curve, params=best, worst_err=worst.tolist(),
                diverged=bool(not np.isfinite(best_val)))
