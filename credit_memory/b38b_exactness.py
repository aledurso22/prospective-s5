"""B38b section 4 -- exact-credit verification.

    grad^{local exact RTRL} == grad^{full dense RTRL} == grad^{BPTT}

checked separately for EVERY recurrently influential parameter family:
A (Atil), phi_Delta (uD, cD), phi_B (uB, cB), and the trainable input
projection feeding the selectors (W, p) -- plus the readout C. Verifying only
the local recurrent A parameters would miss the point of the phase.
"""
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b38b_selective import (
    Q_BOTT, M_LOC, init_L, init_S, fixed_R, nparams, tile_step_L, flat_tile_L,
    step_L, step_S, rollout, loss_fn, TILE_KEYS_L)


def unflat_tile(gflat, d, q):
    o = 0
    out = {}
    out["W"] = gflat[o:o + q * M_LOC].reshape(q, M_LOC); o += q * M_LOC
    out["p"] = gflat[o:o + q]; o += q
    out["Atil"] = gflat[o:o + d]; o += d
    out["uD"] = gflat[o:o + d * q].reshape(d, q); o += d * q
    out["cD"] = gflat[o:o + d]; o += d
    out["uB"] = gflat[o:o + d * q].reshape(d, q); o += d * q
    out["cB"] = gflat[o:o + d]
    return out


def reduced_local_rtrl(params, R, xs, ys, J, d):
    """Arm L: per-tile eligibility E_tau in R^{d x P_tau}. O(1) per tile."""
    q = Q_BOTT
    T = xs.shape[0]
    th = np.stack([np.asarray(flat_tile_L(params, j, d, q)) for j in range(J)])
    Ptau = th.shape[1]
    E = np.zeros((J, d, Ptau))
    h = np.zeros((J, d))
    g_tile = np.zeros((J, Ptau))
    gC = np.zeros_like(np.asarray(params["C"]))
    Cm = np.asarray(params["C"])[0].reshape(J, d)
    jac = jax.jit(jax.vmap(lambda t, hh, xx, xi: jax.jacrev(
        lambda tt: tile_step_L(tt, jax.lax.stop_gradient(hh), xx, xi, d, q)[0])(t),
        in_axes=(0, 0, 0, None)))
    afn = jax.jit(jax.vmap(lambda t, hh, xx, xi: tile_step_L(t, hh, xx, xi, d, q),
                           in_axes=(0, 0, 0, None)))
    thj = jnp.asarray(th)
    for t in range(T):
        x = xs[t]
        xi = x[0]
        xl = jnp.einsum("jkm,m->jk", R, x)
        G = np.asarray(jac(thj, jnp.asarray(h), xl, xi))          # (J, d, Ptau)
        hn, a = afn(thj, jnp.asarray(h), xl, xi)
        hn, a = np.asarray(hn), np.asarray(a)
        E = a[:, :, None] * E + G                                  # diagonal J_t
        y = float(np.asarray(params["C"])[0] @ hn.reshape(-1))
        dldy = 2.0 * (y - float(ys[t])) / T
        qt = dldy * Cm                                             # (J, d)
        g_tile += np.einsum("jdp,jd->jp", E, qt)
        gC += dldy * hn.reshape(1, -1)
        h = hn
    out = {k: [] for k in TILE_KEYS_L}
    for j in range(J):
        u = unflat_tile(g_tile[j], d, q)
        for k in TILE_KEYS_L:
            out[k].append(np.asarray(u[k]))
    out = {k: np.stack(v) for k, v in out.items()}
    out["C"] = gC
    return out


def full_dense_rtrl(params, xs, ys, J, d, arm, R=None):
    """Dense S_t in R^{N x P} over ALL recurrent parameters. No approximation."""
    keys = [k for k in params if k != "C"]
    shapes = {k: np.asarray(params[k]).shape for k in keys}
    sizes = {k: int(np.prod(shapes[k])) for k in keys}
    P = sum(sizes.values())
    N = J * d

    def flat(pd):
        return jnp.concatenate([pd[k].reshape(-1) for k in keys])

    def unflat(v):
        o, out = 0, {}
        for k in keys:
            out[k] = v[o:o + sizes[k]].reshape(shapes[k]); o += sizes[k]
        out["C"] = params["C"]
        return out

    def stepf(v, h, x):
        pd = unflat(v)
        hn, a = (step_L(pd, R, h, x, d) if arm == "L" else step_S(pd, h, x, d))
        return hn.reshape(-1), a.reshape(-1)

    v0 = flat({k: params[k] for k in keys})
    jacf = jax.jit(jax.jacrev(lambda v, h, x: stepf(v, jax.lax.stop_gradient(h), x)[0]))
    fwd = jax.jit(lambda v, h, x: stepf(v, h, x))
    S = np.zeros((N, P))
    h = jnp.zeros((J, d))
    g = np.zeros(P)
    gC = np.zeros_like(np.asarray(params["C"]))
    T = xs.shape[0]
    Sup = np.zeros((N, P), dtype=bool)
    for t in range(T):
        G = np.asarray(jacf(v0, h, xs[t]))
        hn, a = fwd(v0, h, xs[t])
        S = np.asarray(a)[:, None] * S + G
        Sup |= np.abs(S) > 1e-30
        y = float(np.asarray(params["C"])[0] @ np.asarray(hn))
        dldy = 2.0 * (y - float(ys[t])) / T
        qt = dldy * np.asarray(params["C"])[0]
        g += S.T @ qt
        gC += dldy * np.asarray(hn).reshape(1, -1)
        h = hn.reshape(J, d)
    out = {}
    o = 0
    for k in keys:
        out[k] = g[o:o + sizes[k]].reshape(shapes[k]); o += sizes[k]
    out["C"] = gC
    return out, Sup, keys, sizes


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / (1 + np.linalg.norm(b)))


if __name__ == "__main__":
    print("=" * 112)
    print("EXACT CREDIT: reduced source-local RTRL  vs  full dense RTRL  vs  BPTT/autodiff")
    print("Arm L, per parameter family (T=30). A=Atil, phi_Delta=(uD,cD), phi_B=(uB,cB),"
          " input proj=(W,p)")
    print("=" * 112)
    hdr = ["Atil (A)", "uD", "cD", "uB", "cB", "W", "p", "C"]
    print(f"{'J':>2s} {'d':>2s} {'m':>2s} {'N':>4s} {'P':>5s} | " +
          " ".join(f"{h:>10s}" for h in hdr))
    worst = 0.0
    for (J, d, m) in [(2, 2, 3), (3, 2, 4), (4, 3, 4), (5, 3, 5), (3, 4, 3)]:
        R = fixed_R(J, m)
        p = init_L(J, d, m, 0)
        rng = np.random.RandomState(7)
        xs = jnp.asarray(rng.randn(30, m) * 0.7)
        ys = jnp.asarray(rng.randn(30) * 0.5)
        gr = reduced_local_rtrl(p, R, xs, ys, J, d)
        gf, _, _, _ = full_dense_rtrl(p, xs, ys, J, d, "L", R)
        gb = jax.grad(lambda pp: loss_fn(pp, xs, ys, J, d, "L", R))(p)
        cells, row = [], []
        for k in ["Atil", "uD", "cD", "uB", "cB", "W", "p", "C"]:
            e = max(rel(gr[k], np.asarray(gb[k])), rel(gf[k], np.asarray(gb[k])))
            worst = max(worst, e)
            cells.append(f"{e:10.2e}")
        print(f"{J:2d} {d:2d} {m:2d} {J*d:4d} {nparams(p):5d} | " + " ".join(cells))
    print(f"\n  worst relative error over all families and sizes: {worst:.3e}"
          f"   (tolerance 1e-10: {'PASS' if worst < 1e-10 else 'FAIL'})")

    print()
    print("=" * 112)
    print("Arm S (negative control): full dense RTRL vs BPTT -- exact, not approximated")
    print("=" * 112)
    ws = 0.0
    for (J, d, m) in [(2, 2, 3), (3, 2, 4), (4, 3, 4)]:
        p = init_S(J, d, m, 0)
        rng = np.random.RandomState(7)
        xs = jnp.asarray(rng.randn(30, m) * 0.7)
        ys = jnp.asarray(rng.randn(30) * 0.5)
        gf, _, _, _ = full_dense_rtrl(p, xs, ys, J, d, "S")
        gb = jax.grad(lambda pp: loss_fn(pp, xs, ys, J, d, "S"))(p)
        cells = []
        for k in ["Atil", "uD", "cD", "uB", "cB", "W", "p", "C"]:
            e = rel(gf[k], np.asarray(gb[k])); ws = max(ws, e)
            cells.append(f"{e:10.2e}")
        print(f"{J:2d} {d:2d} {m:2d} {J*d:4d} {nparams(p):5d} | " + " ".join(cells))
    print(f"\n  Arm S dense-RTRL vs BPTT worst: {ws:.3e}")
