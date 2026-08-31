"""B38b -- minimal source-local selective SSM (Mamba-1-like, not vanilla Mamba-1).

Recurrence (both arms identical in form):
    h_{t+1,j} = a_{t,j} h_{t,j} + b_{t,j} xi_t,      a_{t,j} = exp(Delta_{t,j} A_j)
with Delta_{t,j}, b_{t,j} INPUT-DEPENDENT. State partitioned into J tiles of
bounded size d_tile = O(1). Nothing else is added: no Mamba-3 rotations, no
trapezoidal integration, no RoPE, no multi-head value states, no deep stacks,
no language modelling, no universal quotient q, no ProductLocal Jordan factors.
Local propagation is scalar-diagonal, which is enough to isolate source locality.

WHY THE JACOBIAN IS NOT THE ISSUE. Delta and b depend on x_t only, never on h,
so J_t = d h_{t+1}/d h_t = diag(a_t) is EXACTLY diagonal in both arms. Hence for
a parameter phi the sensitivity column S_t[:, phi] stays supported on exactly
the channels phi influences:
    e_{t+1}[j, phi] = a_{t,j} e_t[j, phi] + (da_{t,j}/dphi) h_{t,j} + (db_{t,j}/dphi) xi_t
so   dim M_phi = |{j : phi influences j}|   and   sum_phi dim M_phi = sum_phi |supp phi|.
A diagonal J_t therefore does NOT by itself give O(P): what decides it is the
FAN-OUT of each trainable selector parameter.

Arm S (shared selectors, negative control): a shared input projection and a
shared scalar Delta, Mamba-1 style -- each of those parameters influences all N
channels, so its module has dimension N.
Arm L (source-local): every trainable selector parameter influences exactly one
tile (c = 1). Tiles read the input through FIXED, NON-TRAINABLE random
projections R_tau (structural constants needing no credit, like B37c's
multiplication tables), so locality restricts trainable fan-out without blinding
a tile to the input.

The instantaneous G_t is a per-step, per-tile Jacobian of the one-step map with
h held fixed. It carries NO tape across time -- that is ordinary RTRL.
"""
from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Q_BOTT = 2          # selector bottleneck width
M_LOC = 2           # local input slice seen by each tile (Arm L)


def softplus(z):
    return jnp.logaddexp(z, 0.0)


# --------------------------------------------------------------- parameters
def init_L(J, d, m, seed):
    rng = np.random.RandomState(500 + seed)
    q = Q_BOTT
    N = J * d
    return dict(W=jnp.asarray(rng.randn(J, q, M_LOC) * 0.5),
                p=jnp.asarray(rng.randn(J, q) * 0.1),
                Atil=jnp.asarray(rng.randn(J, d) * 0.5),
                uD=jnp.asarray(rng.randn(J, d, q) * 0.5),
                cD=jnp.asarray(rng.randn(J, d) * 0.1),
                uB=jnp.asarray(rng.randn(J, d, q) * 0.5),
                cB=jnp.asarray(rng.randn(J, d) * 0.1),
                C=jnp.asarray(rng.randn(1, N) / np.sqrt(N)))


def fixed_R(J, m, seed=0):
    """Structural constant, NOT trainable: each tile's fixed view of the input."""
    rng = np.random.RandomState(9100 + seed)
    return jnp.asarray(rng.randn(J, M_LOC, m) / np.sqrt(m))


def init_S(J, d, m, seed):
    rng = np.random.RandomState(500 + seed)
    q = Q_BOTT
    N = J * d
    return dict(W=jnp.asarray(rng.randn(q, m) * 0.5),          # SHARED -> all N
                p=jnp.asarray(rng.randn(q) * 0.1),             # SHARED -> all N
                uD=jnp.asarray(rng.randn(q) * 0.5),            # SHARED scalar Delta
                cD=jnp.asarray(rng.randn() * 0.1),             # SHARED
                Atil=jnp.asarray(rng.randn(J, d) * 0.5),       # per channel
                uB=jnp.asarray(rng.randn(J, d, q) * 0.5),      # per channel
                cB=jnp.asarray(rng.randn(J, d) * 0.1),         # per channel
                C=jnp.asarray(rng.randn(1, N) / np.sqrt(N)))


SHARED_KEYS_S = ("W", "p", "uD", "cD")          # fan-out = all N channels
LOCAL_KEYS_S = ("Atil", "uB", "cB")             # fan-out = 1 channel
TILE_KEYS_L = ("W", "p", "Atil", "uD", "cD", "uB", "cB")   # fan-out = 1 tile


def nparams(params):
    return int(sum(np.asarray(v).size for v in params.values()))


# ------------------------------------------------------------------ forward
def tile_step_L(th, h, xl, xi, d, q):
    """One tile's update. th is the tile's FLAT trainable parameter vector."""
    o = 0
    W = th[o:o + q * M_LOC].reshape(q, M_LOC); o += q * M_LOC
    p = th[o:o + q]; o += q
    Atil = th[o:o + d]; o += d
    uD = th[o:o + d * q].reshape(d, q); o += d * q
    cD = th[o:o + d]; o += d
    uB = th[o:o + d * q].reshape(d, q); o += d * q
    cB = th[o:o + d]
    g = jnp.tanh(W @ xl + p)
    A = -softplus(Atil)
    Dl = softplus(uD @ g + cD)
    a = jnp.exp(Dl * A)
    b = uB @ g + cB
    return a * h + b * xi, a


def flat_tile_L(params, j, d, q):
    return jnp.concatenate([params["W"][j].reshape(-1), params["p"][j],
                            params["Atil"][j], params["uD"][j].reshape(-1),
                            params["cD"][j], params["uB"][j].reshape(-1),
                            params["cB"][j]])


def step_L(params, R, h, x, d):
    q = Q_BOTT
    J = R.shape[0]
    xi = x[0]
    xl = jnp.einsum("jkm,m->jk", R, x)
    th = jax.vmap(lambda j: flat_tile_L(params, j, d, q))(jnp.arange(J))
    hn, a = jax.vmap(lambda t, hh, xx: tile_step_L(t, hh, xx, xi, d, q))(th, h, xl)
    return hn, a


def step_S(params, h, x, d):
    xi = x[0]
    g = jnp.tanh(params["W"] @ x + params["p"])
    Dl = softplus(jnp.dot(params["uD"], g) + params["cD"])     # SHARED scalar
    A = -softplus(params["Atil"])
    a = jnp.exp(Dl * A)
    b = jnp.einsum("jdq,q->jd", params["uB"], g) + params["cB"]
    return a * h + b * xi, a


def rollout(params, xs, J, d, arm, R=None):
    h0 = jnp.zeros((J, d))

    def st(h, x):
        hn, _ = step_L(params, R, h, x, d) if arm == "L" else step_S(params, h, x, d)
        return hn, params["C"][0] @ hn.reshape(-1)
    _, ys = jax.lax.scan(st, h0, xs)
    return ys


def loss_fn(params, xs, ys, J, d, arm, R=None):
    return jnp.mean((rollout(params, xs, J, d, arm, R) - ys) ** 2)
