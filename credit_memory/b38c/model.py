"""B38c -- single-layer source-local selective byte-level LM.

Recurrence (per tile tau, bounded d_tile):
    g_{t,tau} = tanh(E_tau[byte_t])                       tile-LOCAL embedding
    s_{t,tau} = uD_tau g_{t,tau} + cD_tau
    Delta     = softplus(s),   A_tau = -softplus(Atil_tau)
    a_{t,tau} = exp(Delta * A_tau)                        in (0,1) by construction
    b_{t,tau} = uB_tau g_{t,tau} + cB_tau
    h_{t+1,tau} = a_{t,tau} * h_{t,tau} + b_{t,tau}
Readout (memoryless, does NOT feed back):  logits_t = W_out h_{t+1} + c.

Every recurrently influential trainable parameter (E, Atil, uD, cD, uB, cB)
affects exactly ONE tile. There is no trainable dense input projection shared
across tiles. No attention, no second recurrent layer, no Mamba-3
rotations/trapezoidal state, no universal quotient coordinates.

Since Delta, b depend on the byte only (never on h), J_t = diag(a_t) is exactly
diagonal, so dim M_phi = |supp phi| and the eligibility decomposes per tile.

ANALYTIC INSTANTANEOUS SOURCE DERIVATIVES (no jacrev in the hot path; B38b
showed per-tile jacrev dominated runtime). With sg = sigmoid and
    alpha_k := h_k * A_k * a_k * sg(s_k)          [uses the PRE-update h]
    dG/dcD_k    = alpha_k
    dG/duD_{ki} = alpha_k g_i
    dG/dAtil_k  = -h_k * Delta_k * a_k * sg(Atil_k)
    dG/dcB_k    = 1
    dG/duB_{ki} = g_i
    dG/dE_{v,i} = [alpha_k uD_{ki} + uB_{ki}] (1 - g_i^2) * 1[byte_t = v]
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

V_BYTE = 256
REC_KEYS = ("E", "Atil", "uD", "cD", "uB", "cB")      # recurrently influential
OUT_KEYS = ("Wout", "cout")


def softplus(z):
    return jnp.logaddexp(z, 0.0)


def sigmoid(z):
    return jax.nn.sigmoid(z)


# --------------------------------------------------------------- init
def init_local(J, d, q, seed, V=V_BYTE, dtype=jnp.float32):
    r = np.random.RandomState(1000 + seed)
    N = J * d
    p = dict(E=r.randn(J, V, q) * 0.5,
             Atil=r.randn(J, d) * 0.5,
             uD=r.randn(J, d, q) * 0.5,
             cD=r.randn(J, d) * 0.1,
             uB=r.randn(J, d, q) * 0.5,
             cB=r.randn(J, d) * 0.1,
             Wout=r.randn(V, N) / np.sqrt(N),
             cout=np.zeros(V))
    return {k: jnp.asarray(v, dtype=dtype) for k, v in p.items()}


def init_shared(J, d, q, seed, V=V_BYTE, dtype=jnp.float32):
    """Arm C control: ONE shared embedding feeding every tile through trainable
    per-tile mixing, so each embedding parameter affects ALL tiles. Embedding
    width q_s = J*q keeps the embedding parameter count matched to Arm L."""
    r = np.random.RandomState(1000 + seed)
    N, qs = J * d, J * q
    p = dict(Emb=r.randn(V, qs) * 0.5,
             Wmix=r.randn(J, q, qs) / np.sqrt(qs),
             Atil=r.randn(J, d) * 0.5,
             uD=r.randn(J, d, q) * 0.5,
             cD=r.randn(J, d) * 0.1,
             uB=r.randn(J, d, q) * 0.5,
             cB=r.randn(J, d) * 0.1,
             Wout=r.randn(V, N) / np.sqrt(N),
             cout=np.zeros(V))
    return {k: jnp.asarray(v, dtype=dtype) for k, v in p.items()}


def nparams(p):
    return int(sum(np.asarray(v).size for v in p.values()))


# --------------------------------------------------------------- forward
def tile_feats(p, bt, arm):
    """g_{t,tau}: (B, J, q)."""
    if arm == "C":
        e = p["Emb"][bt]                                    # (B, qs)
        return jnp.tanh(jnp.einsum("jqs,bs->bjq", p["Wmix"], e))
    return jnp.tanh(jnp.take(p["E"], bt, axis=1).transpose(1, 0, 2))


def cell(p, h, bt, arm):
    """One step. h: (B,J,d). Returns h', and the intermediates the analytic
    source derivatives need."""
    g = tile_feats(p, bt, arm)
    s = jnp.einsum("jkq,bjq->bjk", p["uD"], g) + p["cD"]
    Dl = softplus(s)
    A = -softplus(p["Atil"])
    a = jnp.exp(Dl * A)
    bb = jnp.einsum("jkq,bjq->bjk", p["uB"], g) + p["cB"]
    return a * h + bb, (g, s, Dl, A, a)


def forward(p, bytes_in, J, d, arm, h0=None):
    B = bytes_in.shape[0]
    h0 = jnp.zeros((B, J, d), p["Wout"].dtype) if h0 is None else h0

    def st(h, bt):
        hn, _ = cell(p, h, bt, arm)
        return hn, hn.reshape(B, -1) @ p["Wout"].T + p["cout"]
    hT, logits = jax.lax.scan(st, h0, bytes_in.T)
    return hT, logits.transpose(1, 0, 2)                    # (B,T,V)


def ce_loss(p, bytes_in, targets, J, d, arm, h0=None):
    _, lg = forward(p, bytes_in, J, d, arm, h0)
    lp = jax.nn.log_softmax(lg, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, targets[:, :, None], -1))


def bits_per_byte(nats):
    return float(nats) / float(np.log(2.0))


# ------------------------------------------- analytic instantaneous G_t
def source_grads(p, h, g, s, Dl, A, a, bt, V):
    """All dG/dphi at one step, fully vectorized over (B, J, d). h is PRE-update."""
    sg_s = sigmoid(s)
    alpha = h * A * a * sg_s                                 # (B,J,d)
    G = {}
    G["cD"] = alpha
    G["uD"] = alpha[..., None] * g[:, :, None, :]            # (B,J,d,q)
    G["Atil"] = -h * Dl * a * sigmoid(p["Atil"])
    G["cB"] = jnp.ones_like(alpha)
    G["uB"] = jnp.broadcast_to(g[:, :, None, :], G["uD"].shape)
    dg = 1.0 - g ** 2                                        # (B,J,q)
    coef = (alpha[..., None] * p["uD"] + p["uB"]) * dg[:, :, None, :]   # (B,J,d,q)
    G["E_row"] = coef                                        # placed at row bt
    return G, alpha


# ------------------------------------------------ reduced source-local RTRL
def rtrl_chunk(p, h0, bytes_in, targets, J, d, V, denom):
    """Exact source-local RTRL over one chunk. Eligibility reset at entry, h0
    entering fixed. Pure forward scan: no tape, no per-tile jacrev, no Python
    loop over tiles or parameters."""
    B, T = bytes_in.shape
    q = p["uD"].shape[2]
    dt = p["Wout"].dtype
    z = lambda *sh: jnp.zeros(sh, dt)
    init = (h0, z(B, J, d, V, q), z(B, J, d), z(B, J, d, q), z(B, J, d),
            z(B, J, d, q), z(B, J, d),
            {k: jnp.zeros_like(p[k]) for k in REC_KEYS + OUT_KEYS})

    def step(carry, inp):
        h, eE, eA, euD, ecD, euB, ecB, gacc = carry
        bt, tg = inp
        hn, (g, s, Dl, A, a) = cell(p, h, bt, "L")
        G, _ = source_grads(p, h, g, s, Dl, A, a, bt, V)
        oh = jax.nn.one_hot(bt, V, dtype=dt)                  # (B,V)
        eE = a[:, :, :, None, None] * eE + \
            G["E_row"][:, :, :, None, :] * oh[:, None, None, :, None]
        eA = a * eA + G["Atil"]
        euD = a[..., None] * euD + G["uD"]
        ecD = a * ecD + G["cD"]
        euB = a[..., None] * euB + G["uB"]
        ecB = a * ecB + G["cB"]
        hf = hn.reshape(B, -1)
        lg = hf @ p["Wout"].T + p["cout"]
        pr = jax.nn.softmax(lg)
        dl = (pr - jax.nn.one_hot(tg, V, dtype=dt)) / denom   # (B,V)
        qt = (dl @ p["Wout"]).reshape(B, J, d)                # dL/dh_{t+1}
        gacc = dict(gacc)
        gacc["E"] = gacc["E"] + jnp.einsum("bjk,bjkvq->jvq", qt, eE)
        gacc["Atil"] = gacc["Atil"] + jnp.einsum("bjk,bjk->jk", qt, eA)
        gacc["uD"] = gacc["uD"] + jnp.einsum("bjk,bjkq->jkq", qt, euD)
        gacc["cD"] = gacc["cD"] + jnp.einsum("bjk,bjk->jk", qt, ecD)
        gacc["uB"] = gacc["uB"] + jnp.einsum("bjk,bjkq->jkq", qt, euB)
        gacc["cB"] = gacc["cB"] + jnp.einsum("bjk,bjk->jk", qt, ecB)
        gacc["Wout"] = gacc["Wout"] + dl.T @ hf
        gacc["cout"] = gacc["cout"] + dl.sum(0)
        return (hn, eE, eA, euD, ecD, euB, ecB, gacc), None
    (hT, eE, eA, euD, ecD, euB, ecB, gacc), _ = jax.lax.scan(
        step, init, (bytes_in.T, targets.T))
    return gacc, hT


def elig_bytes(B, J, d, q, V, itemsize=4):
    """Persistent eligibility state only -- NOT input buffers."""
    per = B * J * d * (V * q + 1 + q + 1 + q + 1)
    return int(per * itemsize), dict(E=B * J * d * V * q, other=B * J * d * (3 + 2 * q))
