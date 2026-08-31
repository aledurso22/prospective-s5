"""B37c -- native ProductLocal parameterization.

    A_pi = prod_{j=1}^{J}  K_j[eps_j]/(eps_j^{d_j}),   K_j in {R, C}

is the model's ACTUAL coordinate system. The local bases 1, eps_j, ..., eps_j^{d_j-1}
and their multiplication tables are fixed structural constants. There is no global
polynomial q, no companion matrix, no CRT transform H(q), no root extraction, no
Vandermonde/Hermite system, and no basis change is differentiated through.

Trainable: u in A_pi, b_c in A_pi per input channel, dense C_out.
    z_{t+1} = u z_t + sum_c b_c x_{c,t},    y_t = C_out z_{t+1}
(the y-from-z_{t+1} convention is B37b's, kept for comparability).

Exact reduced eligibilities (verified separately in b37c_exactness.py):
    e^u_{t+1}   = u e^u_t + z_t,          e^u_0   = 0
    e^{b_c}_{t+1} = u e^{b_c}_t + x_{c,t} 1_A,  e^{b_c}_0 = 0
    grad_u l_t = M_{e^u_t}^T q_t,   grad_{b_c} l_t = M_{e^{b_c}_t}^T q_t,
    grad_C l_t = (dl/dy_t) z_t^T,   q_t = dl_t/dz_t
Persistent eligibility storage: (m+1) r reals.

Structural scope (explicitly NOT a universality claim): for a FIXED pi the model
covers one real-Jordan stratum plus its degenerations, not all M_r(R).
"""
from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b37b_quotient_trainability import (        # frozen B37b, imported only
    FAMILIES, R_VALUES, EVAL_SEEDS, LR_GRID, N_STEPS, CHUNK, DIVERGE,
    make_teacher, get_data)


# ---------------------------------------------------------------- structure
def spec_from_blocks(blocks):
    """pi from the teacher's real-Jordan BLOCK TYPES only (no eigenvalues used)."""
    spec, i = [], 0
    while i < len(blocks):
        lam, n = blocks[i]
        if abs(np.imag(complex(lam))) > 1e-14:
            assert abs(complex(blocks[i + 1][0]) - np.conj(complex(lam))) < 1e-12
            assert blocks[i + 1][1] == n
            spec.append(("C", n)); i += 2
        else:
            spec.append(("R", n)); i += 1
    return tuple(spec)


def real_dim(spec):
    return sum(d * (2 if k == "C" else 1) for k, d in spec)


def kdim(k):
    return 2 if k == "C" else 1


# ------------------------------------------------- algebra (fixed constants)
def alg_mult(a, b, spec):
    """(a b)_n = sum_{l=0}^{n} a_l b_{n-l}, per factor, truncated at d_j.
    Complex factors are realified directly into real coordinates."""
    out = []
    for (k, d), aj, bj in zip(spec, a, b):
        if k == "R":
            c = jnp.stack([jnp.sum(aj[:n + 1, 0] * bj[n::-1, 0]) for n in range(d)])[:, None]
        else:
            ar, ai, br, bi = aj[:, 0], aj[:, 1], bj[:, 0], bj[:, 1]
            cr = jnp.stack([jnp.sum(ar[:n + 1] * br[n::-1] - ai[:n + 1] * bi[n::-1])
                            for n in range(d)])
            ci = jnp.stack([jnp.sum(ar[:n + 1] * bi[n::-1] + ai[:n + 1] * br[n::-1])
                            for n in range(d)])
            c = jnp.stack([cr, ci], axis=1)
        out.append(c)
    return tuple(out)


def alg_add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def alg_scale(a, s):
    return tuple(x * s for x in a)


def alg_zero(spec, np_=jnp):
    return tuple(np_.zeros((d, kdim(k))) for k, d in spec)


def alg_one(spec, np_=jnp):
    out = []
    for k, d in spec:
        z = np_.zeros((d, kdim(k)))
        z = z.at[0, 0].set(1.0) if np_ is jnp else z
        if np_ is not jnp:
            z[0, 0] = 1.0
        out.append(z)
    return tuple(out)


def flat(a):
    return jnp.concatenate([x.reshape(-1) for x in a])


def unflat(v, spec):
    out, o = [], 0
    for k, d in spec:
        n = d * kdim(k)
        out.append(v[o:o + n].reshape(d, kdim(k))); o += n
    return tuple(out)


def make_M(a, spec):
    """Block-diagonal regular representation M_a (real coords). Each block is a
    lower-triangular Toeplitz built from the local multiplication table; complex
    factors use realified 2x2 scalar blocks."""
    a = [np.asarray(x) for x in a]
    r = real_dim(spec)
    M = np.zeros((r, r))
    o = 0
    for (k, d), aj in zip(spec, a):
        w = kdim(k)
        for n in range(d):
            for l in range(n + 1):
                c = aj[n - l]
                if k == "R":
                    M[o + n, o + l] = c[0]
                else:
                    M[o + 2 * n, o + 2 * l] = c[0];     M[o + 2 * n, o + 2 * l + 1] = -c[1]
                    M[o + 2 * n + 1, o + 2 * l] = c[1]; M[o + 2 * n + 1, o + 2 * l + 1] = c[0]
        o += d * w
    return M


def spectral_radius(u, spec):
    """rho(M_u) = max_j |u_{j,0}|: M_u is block lower-triangular Toeplitz, so
    stability depends ONLY on the constant terms. (Verified against eigvals.)"""
    vals = []
    for (k, d), uj in zip(spec, u):
        uj = np.asarray(uj)
        vals.append(abs(uj[0, 0]) if k == "R" else float(np.hypot(uj[0, 0], uj[0, 1])))
    return float(max(vals))


# ---------------------------------------------------------------- the model
def rollout(params, xs, spec):
    u, b, C = params["u"], params["b"], params["C"]

    def step(z, x):
        z = alg_add(alg_mult(u, z, spec), alg_scale(b, x))
        return z, C @ flat(z)
    _, ys = jax.lax.scan(step, alg_zero(spec), xs)
    return ys[:, 0]


def rollout_states(params, xs, spec):
    u, b, C = params["u"], params["b"], params["C"]

    def step(z, x):
        z = alg_add(alg_mult(u, z, spec), alg_scale(b, x))
        return z, flat(z)
    _, zs = jax.lax.scan(step, alg_zero(spec), xs)
    return zs


def batched_loss(params, xs, ys, spec):
    pred = jax.vmap(lambda x: rollout(params, x, spec))(xs)
    return jnp.mean((pred - ys) ** 2)


# ------------------------------------------------------------ initializations
def generic_init(spec, seed):
    """PRIMARY arm. Random stable init. Uses pi (block TYPES) only -- no teacher
    eigenvalues, eigenvectors, or similarity."""
    rng = np.random.RandomState(9000 + seed)
    u = []
    for k, d in spec:
        z = np.zeros((d, kdim(k)))
        if k == "R":
            z[0, 0] = rng.uniform(-0.9, 0.9)
        else:
            mod, ang = rng.uniform(0.3, 0.9), rng.uniform(0.4, 2.6)
            z[0, 0], z[0, 1] = mod * np.cos(ang), mod * np.sin(ang)
        if d > 1:
            z[1:] = rng.randn(d - 1, kdim(k)) * 0.3
        u.append(z)
    r = real_dim(spec)
    b = [rng.randn(d, kdim(k)) / np.sqrt(r) for k, d in spec]
    return dict(u=tuple(u), b=tuple(b), C=rng.randn(1, r) / np.sqrt(r))


def _phi(spec, blocks):
    """Complex Phi with M_u Phi = Phi J for the exact init (sanity arm only)."""
    r = real_dim(spec)
    cols, o, bi = [], 0, 0
    for k, d in spec:
        rev = np.eye(d)[:, ::-1]
        if k == "R":
            P = np.zeros((r, d), dtype=complex)
            P[o:o + d, :] = rev
            cols.append(P); o += d; bi += 1
        else:
            for sign in (-1j, +1j):
                P = np.zeros((r, d), dtype=complex)
                for l in range(d):
                    P[o + 2 * l, l] = 1.0
                    P[o + 2 * l + 1, l] = sign
                cols.append(P @ rev)
            o += 2 * d; bi += 2
    return np.concatenate(cols, axis=1)


def exact_init(teacher, spec):
    """SANITY ARM ONLY (uses teacher eigenvalues). u_j = lambda_j + eps_j makes
    M_{u_j} exactly the real Jordan block, so this verifies that pi is
    structurally capable of representing the teacher's Jordan type."""
    blocks, r = teacher["blocks"], real_dim(spec)
    u, i = [], 0
    for k, d in spec:
        lam = complex(blocks[i][0])
        z = np.zeros((d, kdim(k)))
        z[0, 0] = lam.real
        if k == "C":
            z[0, 1] = lam.imag
        if d > 1:
            z[1, 0] = 1.0
        u.append(z); i += 2 if k == "C" else 1
    M = make_M(u, spec)
    Phi = _phi(spec, blocks)
    J = np.zeros((r, r), dtype=complex)
    o = 0
    for lam, n in blocks:
        J[o:o + n, o:o + n] = lam * np.eye(n) + (np.diag(np.ones(n - 1), 1) if n > 1 else 0)
        o += n
    res_MJ = float(np.linalg.norm(M @ Phi - Phi @ J) / (1 + np.linalg.norm(M)))
    Tf = teacher["S"] @ np.linalg.inv(Phi)
    best = None
    rng = np.random.RandomState(0)
    for c1, c2 in [(1, 0), (0, 1), (1, 1), (1, -1)] + [tuple(rng.randn(2)) for _ in range(20)]:
        Tr = c1 * Tf.real + c2 * Tf.imag
        c = np.linalg.cond(Tr)
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, Tr)
    condT, T = best
    res = float(np.linalg.norm(teacher["A"] @ T - T @ M) /
                (1 + np.linalg.norm(teacher["A"]) * np.linalg.norm(T)))
    bvec = np.linalg.solve(T, teacher["Bs"])
    return (dict(u=tuple(u), b=tuple(np.asarray(x) for x in unflat(jnp.array(bvec), spec)),
                 C=(teacher["Cs"] @ T).reshape(1, r)),
            dict(condT=float(condT), resid_AT_TM=res, resid_MPhi_PhiJ=res_MJ))


def perturb_params(params, eps, seed):
    rng = np.random.RandomState(60000 + seed)
    out = {}
    for key in ("u", "b"):
        out[key] = tuple(np.asarray(x) + eps * (np.abs(np.asarray(x)) + 1.0)
                         * rng.randn(*np.asarray(x).shape) for x in params[key])
    C = np.asarray(params["C"])
    out["C"] = C + eps * (np.abs(C) + 1.0) * rng.randn(*C.shape)
    return out


def to_jax(p):
    return dict(u=tuple(jnp.asarray(x) for x in p["u"]),
                b=tuple(jnp.asarray(x) for x in p["b"]), C=jnp.asarray(p["C"]))


# ------------------------------------------------------------------ training
@functools.lru_cache(maxsize=None)
def _build(spec):
    def loss_of(p, xs, ys):
        return batched_loss(p, xs, ys, spec)

    def run_chunk(params, m, v, t0, xs, ys, lr):
        def one(carry, _):
            params, m, v, t = carry
            loss, g = jax.value_and_grad(loss_of)(params, xs, ys)
            t = t + 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = jax.tree.map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, g)
            v = jax.tree.map(lambda vi, gi: b2 * vi + (1 - b2) * gi ** 2, v, g)
            mh = jax.tree.map(lambda mi: mi / (1 - b1 ** t), m)
            vh = jax.tree.map(lambda vi: vi / (1 - b2 ** t), v)
            params = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps),
                                  params, mh, vh)
            gn = jnp.stack([jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g[k])))
                            for k in ("u", "b", "C")])
            return (params, m, v, t), (loss, gn)
        (params, m, v, t), (losses, gns) = jax.lax.scan(
            one, (params, m, v, t0), None, length=CHUNK)
        return params, m, v, t, losses, gns
    return jax.jit(run_chunk), jax.jit(loss_of)


def markov(params, teacher, spec, K=40):
    M = make_M(params["u"], spec)
    C = np.asarray(params["C"])[0]
    vs = np.asarray(flat(to_jax(params)["b"]))
    A, Bs, Cs = teacher["A"], teacher["Bs"], teacher["Cs"]
    vt, worst, gs_all = Bs.copy(), 0.0, []
    for _ in range(K):
        gs, gt = float(C @ vs), float(Cs @ vt)
        if not np.isfinite(gs):
            return float("inf"), float("inf")
        gs_all.append(gs)
        worst = max(worst, abs(gs - gt) / (1 + abs(gt)))
        vs, vt = M @ vs, A @ vt
    H = np.array([[gs_all[i + j] for j in range(K // 2)] for i in range(K // 2)])
    sv = np.linalg.svd(H, compute_uv=False)
    hank = float(sv[0] / max(sv[-1], 1e-300))
    return float(worst), hank


def diagnostics(params, teacher, spec, xs_te, H=40):
    pj = to_jax(params)
    fl = np.concatenate([np.asarray(x).reshape(-1) for x in params["u"]])
    if not np.all(np.isfinite(fl)):
        return dict(max_z=np.inf, rho=np.inf, gamma_H=np.inf, markov=np.inf,
                    hankel=np.inf, b_norm=np.inf, C_norm=np.inf, condM=np.inf)
    M = make_M(params["u"], spec)
    zs = np.asarray(jax.vmap(lambda x: rollout_states(pj, x, spec))(xs_te))
    mk, hk = markov(params, teacher, spec)
    P, g = np.eye(M.shape[0]), 0.0
    for _ in range(H):
        P = M @ P
        g = max(g, float(np.linalg.norm(P, 2)))
    return dict(max_z=float(np.max(np.abs(zs))) if np.all(np.isfinite(zs)) else np.inf,
                rho=spectral_radius(params["u"], spec), gamma_H=g, markov=mk, hankel=hk,
                b_norm=float(np.linalg.norm(np.concatenate(
                    [np.asarray(x).reshape(-1) for x in params["b"]]))),
                C_norm=float(np.linalg.norm(np.asarray(params["C"]))),
                condM=float(np.linalg.cond(M)))


def train_one(teacher, params0, family, r, spec, lr, seed, n_steps=N_STEPS):
    (xs_tr, ys_tr), (xs_va, ys_va), (xs_te, ys_te) = get_data(teacher, family, r, seed)
    run_chunk, loss_of = _build(spec)
    params = to_jax(params0)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    t = jnp.array(0.0)
    init_val = float(loss_of(params, xs_va, ys_va))
    best_val, best = init_val, params
    gsum, gcount, diverged, div_step = np.zeros(3), 0, False, None
    for c in range(n_steps // CHUNK):
        params, m, v, t, losses, gns = run_chunk(params, m, v, t, xs_tr, ys_tr, lr)
        losses, gns = np.asarray(losses), np.asarray(gns)
        ok = np.isfinite(gns).all(axis=1) & (np.abs(gns) < DIVERGE).all(axis=1)
        if ok.any():
            gsum += gns[ok].sum(axis=0); gcount += int(ok.sum())
        if not np.isfinite(losses).all() or float(np.max(losses)) > DIVERGE:
            diverged, div_step = True, (c + 1) * CHUNK
            break
        vl = float(loss_of(params, xs_va, ys_va))
        if np.isfinite(vl) and vl < best_val:
            best_val, best = vl, params
    ynorm = float(jnp.mean(ys_te ** 2))
    npb = dict(u=tuple(np.asarray(x) for x in best["u"]),
               b=tuple(np.asarray(x) for x in best["b"]), C=np.asarray(best["C"]))
    fv = float(loss_of(params, xs_va, ys_va))
    d = diagnostics(npb, teacher, spec, xs_te)
    g = (gsum / max(gcount, 1)).tolist()
    return dict(test_nmse=float(loss_of(best, xs_te, ys_te)) / (ynorm + 1e-30),
                val_loss=best_val, init_val=init_val,
                final_val=fv if np.isfinite(fv) else float("inf"),
                diverged=bool(diverged), div_step=div_step,
                gnorm_u=g[0], gnorm_b=g[1], gnorm_C=g[2], **d)
