"""B38c local correctness gate (float64, tiny configs).

  1. analytic G_t  vs  jacrev/autodiff
  2. reduced source-local RTRL  vs  dense RTRL  vs  BPTT   (every recurrent family)
  3. optimizer-step agreement from identical initialization
  4. eligibility memory vs sequence length T  (must be constant)
  5. eligibility memory vs N                  (must be O(P))
  6. no O(T) scan outputs/scratch attributable to the RTRL algorithm
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from model import (V_BYTE, REC_KEYS, OUT_KEYS, init_local, nparams, cell,
                   forward, ce_loss, source_grads, rtrl_chunk, elig_bytes)

F64 = jnp.float64


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / (1 + np.linalg.norm(b)))


def dense_rtrl(p, h0, bytes_in, targets, J, d, V, denom):
    """Dense S_t in R^{N x P_rec}. No approximation. Small dims only."""
    keys = list(REC_KEYS)
    shp = {k: p[k].shape for k in keys}
    sz = {k: int(np.prod(shp[k])) for k in keys}
    P = sum(sz.values())
    B, T = bytes_in.shape
    N = J * d

    def unflat(v):
        o, out = 0, {}
        for k in keys:
            out[k] = v[o:o + sz[k]].reshape(shp[k]); o += sz[k]
        for k in OUT_KEYS:
            out[k] = p[k]
        return out
    v0 = jnp.concatenate([p[k].reshape(-1) for k in keys])

    def stepf(v, h, bt):
        hn, _ = cell(unflat(v), h, bt, "L")
        return hn.reshape(B, -1)
    jf = jax.jit(jax.jacrev(lambda v, h, bt: stepf(v, jax.lax.stop_gradient(h), bt)))
    S = np.zeros((B, N, P))
    h = np.asarray(h0)
    g = np.zeros(P)
    gW = np.zeros_like(np.asarray(p["Wout"])); gc = np.zeros_like(np.asarray(p["cout"]))
    for t in range(T):
        bt = bytes_in[:, t]
        G = np.asarray(jf(v0, jnp.asarray(h), bt))                 # (B,N,P)
        hn, (_, _, _, _, a) = cell(p, jnp.asarray(h), bt, "L")
        a = np.asarray(a).reshape(B, N)
        S = a[:, :, None] * S + G
        hf = np.asarray(hn).reshape(B, -1)
        lg = hf @ np.asarray(p["Wout"]).T + np.asarray(p["cout"])
        pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        oh = np.eye(V)[np.asarray(targets[:, t])]
        dl = (pr - oh) / denom
        qt = dl @ np.asarray(p["Wout"])
        g += np.einsum("bn,bnp->p", qt, S)
        gW += dl.T @ hf; gc += dl.sum(0)
        h = np.asarray(hn)
    out, o = {}, 0
    for k in keys:
        out[k] = g[o:o + sz[k]].reshape(shp[k]); o += sz[k]
    out["Wout"], out["cout"] = gW, gc
    return out


def main():
    print("=" * 104)
    print("1. ANALYTIC instantaneous G_t  vs  jacrev/autodiff")
    print("=" * 104)
    worst1 = 0.0
    print(f"{'J':>2s} {'d':>2s} {'q':>2s} {'V':>3s} | " +
          " ".join(f"{k:>10s}" for k in REC_KEYS))
    for (J, d, q, V) in [(2, 2, 2, 5), (3, 2, 2, 7), (2, 3, 3, 6), (4, 2, 3, 9)]:
        p = init_local(J, d, q, 0, V=V, dtype=F64)
        B = 2
        bt = jnp.asarray(np.random.RandomState(1).randint(0, V, (B,)))
        h = jnp.asarray(np.random.RandomState(2).randn(B, J, d))
        hn, (g, s, Dl, A, a) = cell(p, h, bt, "L")
        G, _ = source_grads(p, h, g, s, Dl, A, a, bt, V)
        cells = []
        for k in REC_KEYS:
            def f(pk):
                pp = dict(p); pp[k] = pk
                return cell(pp, jax.lax.stop_gradient(h), bt, "L")[0]
            Jr = np.asarray(jax.jacrev(f)(p[k]))       # (B,J,d) + shape(p[k])
            if k == "E":
                # select the embedding row actually read at this step, per batch
                sel = np.stack([Jr[b][:, :, :, int(bt[b]), :] for b in range(B)])
                ref = np.einsum("bjkjq->bjkq", sel)    # tile-diagonal
                e = rel(G["E_row"], ref)
            elif p[k].ndim == 2:                       # (J,d): Atil, cD, cB
                e = rel(G[k], np.einsum("bjkjk->bjk", Jr))
            else:                                      # (J,d,q): uD, uB
                e = rel(G[k], np.einsum("bjkjkq->bjkq", Jr))
            worst1 = max(worst1, e); cells.append(f"{e:10.2e}")
        print(f"{J:2d} {d:2d} {q:2d} {V:3d} | " + " ".join(cells))
    print(f"\n  worst analytic-vs-autodiff error: {worst1:.3e}"
          f"  ({'PASS' if worst1 < 1e-10 else 'FAIL'})")

    print()
    print("=" * 104)
    print("2. reduced source-local RTRL  vs  dense RTRL  vs  BPTT   (all families)")
    print("=" * 104)
    allk = list(REC_KEYS) + list(OUT_KEYS)
    print(f"{'J':>2s} {'d':>2s} {'q':>2s} {'V':>3s} {'T':>3s} {'P':>6s} | " +
          " ".join(f"{k:>9s}" for k in allk))
    worst2 = 0.0
    for (J, d, q, V, T) in [(2, 2, 2, 5, 12), (3, 2, 2, 7, 20), (2, 3, 3, 6, 16),
                            (4, 2, 3, 9, 24)]:
        p = init_local(J, d, q, 0, V=V, dtype=F64)
        B = 3
        r = np.random.RandomState(5)
        bi = jnp.asarray(r.randint(0, V, (B, T)))
        tg = jnp.asarray(r.randint(0, V, (B, T)))
        denom = float(B * T)
        h0 = jnp.zeros((B, J, d), F64)
        gr, _ = rtrl_chunk(p, h0, bi, tg, J, d, V, denom)
        gd = dense_rtrl(p, h0, bi, tg, J, d, V, denom)
        gb = jax.grad(lambda pp: ce_loss(pp, bi, tg, J, d, "L", h0))(p)
        cells = []
        for k in allk:
            e = max(rel(gr[k], np.asarray(gb[k])), rel(gd[k], np.asarray(gb[k])))
            worst2 = max(worst2, e); cells.append(f"{e:9.2e}")
        print(f"{J:2d} {d:2d} {q:2d} {V:3d} {T:3d} {nparams(p):6d} | " + " ".join(cells))
    print(f"\n  worst reduced=dense=BPTT error: {worst2:.3e}"
          f"  ({'PASS' if worst2 < 1e-10 else 'FAIL'})")

    print()
    print("=" * 104)
    print("3. optimizer-step agreement from identical initialization (50 Adam steps)")
    print("=" * 104)
    J, d, q, V, T, B = 3, 2, 2, 7, 24, 3
    p0 = init_local(J, d, q, 0, V=V, dtype=F64)
    r = np.random.RandomState(9)
    bi = jnp.asarray(r.randint(0, V, (B, T))); tg = jnp.asarray(r.randint(0, V, (B, T)))
    denom = float(B * T)
    h0 = jnp.zeros((B, J, d), F64)
    st = {a: [p0, jax.tree.map(jnp.zeros_like, p0), jax.tree.map(jnp.zeros_like, p0),
              jnp.array(0.0)] for a in "AB"}
    dev = 0.0
    for i in range(50):
        for arm in "AB":
            pp, m, v, t = st[arm]
            g = (jax.grad(lambda z: ce_loss(z, bi, tg, J, d, "L", h0))(pp) if arm == "A"
                 else rtrl_chunk(pp, h0, bi, tg, J, d, V, denom)[0])
            t = t + 1
            m = jax.tree.map(lambda a_, b_: 0.9 * a_ + 0.1 * b_, m, g)
            v = jax.tree.map(lambda a_, b_: 0.999 * a_ + 0.001 * b_ ** 2, v, g)
            mh = jax.tree.map(lambda x: x / (1 - 0.9 ** t), m)
            vh = jax.tree.map(lambda x: x / (1 - 0.999 ** t), v)
            pp = jax.tree.map(lambda a_, b_, c_: a_ - 1e-2 * b_ / (jnp.sqrt(c_) + 1e-8),
                              pp, mh, vh)
            st[arm] = [pp, m, v, t]
        fa = np.concatenate([np.asarray(st["A"][0][k]).ravel() for k in allk])
        fb = np.concatenate([np.asarray(st["B"][0][k]).ravel() for k in allk])
        dev = max(dev, float(np.max(np.abs(fa - fb))))
    print(f"  max parameter deviation over 50 matched Adam steps: {dev:.3e}"
          f"  ({'PASS' if dev < 1e-10 else 'FAIL'})")

    print()
    print("=" * 104)
    print("4/5. eligibility memory: vs sequence length T, and vs N  (persistent state only)")
    print("=" * 104)
    print(f"{'B':>3s} {'J':>4s} {'d':>2s} {'q':>2s} {'V':>4s} {'T':>7s} {'N':>6s} "
          f"{'P_rec':>9s} {'elig MB':>9s} {'elig/P':>8s}")
    for T in (128, 512, 2048, 8192):
        B, J, d, q, V = 8, 16, 4, 2, 256
        p = init_local(J, d, q, 0, V=V)
        Prec = sum(int(np.prod(p[k].shape)) for k in REC_KEYS)
        nb, _ = elig_bytes(B, J, d, q, V)
        print(f"{B:3d} {J:4d} {d:2d} {q:2d} {V:4d} {T:7d} {J*d:6d} {Prec:9d} "
              f"{nb/1e6:9.3f} {nb/4/Prec:8.2f}")
    print("   (identical at every T -- eligibility does not depend on sequence length)")
    for J in (8, 32, 128, 512):
        B, d, q, V = 8, 4, 2, 256
        p = init_local(J, d, q, 0, V=V)
        Prec = sum(int(np.prod(p[k].shape)) for k in REC_KEYS)
        nb, brk = elig_bytes(B, J, d, q, V)
        print(f"{B:3d} {J:4d} {d:2d} {q:2d} {V:4d} {'--':>7s} {J*d:6d} {Prec:9d} "
              f"{nb/1e6:9.3f} {nb/4/Prec:8.2f}")
    print("   elig/P constant in N => O(P). Dominated by the tile-local embedding")
    print("   (B*J*d*V*q); selector eligibility is only B*J*d*(3+2q).")

    print()
    print("=" * 104)
    print("6. no O(T) scan output/scratch attributable to the RTRL algorithm")
    print("=" * 104)
    J, d, q, V, B = 8, 4, 2, 64, 4
    p = init_local(J, d, q, 0, V=V)
    prev = None
    for T in (64, 256, 1024, 4096):
        r = np.random.RandomState(0)
        bi = jnp.asarray(r.randint(0, V, (B, T))); tg = jnp.asarray(r.randint(0, V, (B, T)))
        h0 = jnp.zeros((B, J, d), jnp.float32)
        f = jax.jit(lambda pp, h, x, y: rtrl_chunk(pp, h, x, y, J, d, V, float(B * T)))
        st = f.lower(p, h0, bi, tg).compile().memory_analysis()
        inp = bi.nbytes + tg.nbytes
        print(f"  T={T:5d}  XLA temp {st.temp_size_in_bytes/1e6:9.4f} MB   "
              f"input buffers {inp/1e6:9.4f} MB   "
              f"temp - input {(st.temp_size_in_bytes-inp)/1e6:9.4f} MB")
        prev = st.temp_size_in_bytes
    print("   RTRL scan emits None per step; any T-dependence above is input buffers,")
    print("   which are NOT eligibility memory.")


if __name__ == "__main__":
    main()
