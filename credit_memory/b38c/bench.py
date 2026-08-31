"""B38c memory/throughput benchmark: matched BPTT vs exact source-local RTRL on
the IDENTICAL architecture. Runs unchanged on CPU or GPU.

Protocol: compile + warm up, device-synchronize before and after every timed
region, compilation excluded, median of >=5 reps. No kernel optimization.

Two regimes are reported separately and never conflated:
  batched-context : the same resident B x T inputs for both algorithms;
  streaming RTRL  : small input chunks with recurrent state + eligibility
                    carried indefinitely, full sequence never materialized.
Input-buffer memory is reported separately and is NEVER called eligibility memory.
"""
import argparse, json, time
import numpy as np
import jax
import jax.numpy as jnp

from model import (V_BYTE, REC_KEYS, OUT_KEYS, init_local, nparams, ce_loss,
                   forward, rtrl_chunk, elig_bytes)


def sync(x):
    jax.block_until_ready(x)


def peak_device_bytes():
    try:
        st = jax.local_devices()[0].memory_stats()
        return int(st.get("peak_bytes_in_use", -1)) if st else -1
    except Exception:
        return -1


def build(J, d, q, V, algo):
    def upd(p, m, v, h, x, y, lr):
        B, T = x.shape
        if algo == "bptt":
            g = jax.grad(lambda pp: ce_loss(pp, x, y, J, d, "L",
                                            jax.lax.stop_gradient(h)))(p)
            hT, _ = forward(p, x, J, d, "L", h)
        else:
            g, hT = rtrl_chunk(p, h, x, y, J, d, V, float(B * T))
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b ** 2, v, g)
        p = jax.tree.map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + 1e-8), p, m, v)
        return p, m, v, jax.lax.stop_gradient(hT)
    return jax.jit(upd)


def measure(J, d, q, V, B, T, algo, reps=5):
    p = init_local(J, d, q, 0, V=V)
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    h = jnp.zeros((B, J, d), jnp.float32)
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randint(0, V, (B, T)).astype(np.int32))
    y = jnp.asarray(rng.randint(0, V, (B, T)).astype(np.int32))
    fn = build(J, d, q, V, algo)
    args = (p, m, v, h, x, y, 1e-3)
    rec = dict(algo=algo, J=J, d=d, q=q, B=B, T=T, N=J * d,
               P=nparams(p),
               model_bytes=int(sum(np.prod(z.shape) for z in p.values()) * 4),
               opt_bytes=int(2 * sum(np.prod(z.shape) for z in p.values()) * 4),
               input_bytes=int(x.nbytes + y.nbytes),
               elig_bytes=elig_bytes(B, J, d, q, V)[0] if algo == "rtrl" else 0)
    try:
        st = fn.lower(*args).compile().memory_analysis()
        rec["xla_temp_bytes"] = int(st.temp_size_in_bytes)
    except Exception:
        rec["xla_temp_bytes"] = -1
    try:
        out = fn(*args); sync(out)
    except Exception as e:                       # record OOM, do not silently shrink
        rec.update(oom=True, error=str(e)[:200], wall=float("nan"),
                   tokens_per_s=float("nan"), peak_device_bytes=-1)
        return rec
    ts = []
    for _ in range(reps):
        sync(out)
        t0 = time.perf_counter()
        out = fn(*args)
        sync(out)
        ts.append(time.perf_counter() - t0)
    w = float(np.median(ts))
    rec.update(oom=False, wall=w, per_update=w, tokens_per_s=B * T / w,
               peak_device_bytes=peak_device_bytes())
    return rec


def streaming(J, d, q, V, B, chunk, n_chunks, reps=3):
    """Eligibility + state carried across chunks; full sequence never materialized."""
    p = init_local(J, d, q, 0, V=V)
    fn = jax.jit(lambda pp, h, x, y: rtrl_chunk(pp, h, x, y, J, d, V, float(B * chunk)))
    h = jnp.zeros((B, J, d), jnp.float32)
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randint(0, V, (B, chunk)).astype(np.int32))
    y = jnp.asarray(rng.randint(0, V, (B, chunk)).astype(np.int32))
    o = fn(p, h, x, y); sync(o)
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        g, h = fn(p, h, x, y)
    sync(h)
    w = time.perf_counter() - t0
    return dict(mode="streaming", J=J, d=d, q=q, B=B, chunk=chunk,
                n_chunks=n_chunks, tokens=B * chunk * n_chunks,
                wall=w, tokens_per_s=B * chunk * n_chunks / w,
                elig_bytes=elig_bytes(B, J, d, q, V)[0],
                input_bytes=int(x.nbytes + y.nbytes),
                peak_device_bytes=peak_device_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--T", default="128,512,2048,8192")
    ap.add_argument("--N", default="128,512,2048")
    ap.add_argument("--stream-chunks", type=int, default=64)
    a = ap.parse_args()
    Ts = [int(z) for z in a.T.split(",")]
    Ns = [int(z) for z in a.N.split(",")]
    print(f"backend={jax.default_backend()}  devices={jax.devices()}")
    rows = []
    BACKEND = jax.default_backend()
    print("\n== BATCHED-CONTEXT (identical resident B x T inputs for both algorithms) ==")
    hdr = (f"{'algo':>5s} {'N':>6s} {'T':>7s} {'B':>4s} {'tok/s':>11s} {'ms/upd':>9s} "
           f"{'elig MB':>9s} {'model MB':>9s} {'opt MB':>8s} {'input MB':>9s} "
           f"{'XLA temp MB':>12s} {'peak dev MB':>12s} {'oom':>5s}")
    print(hdr)
    for N in Ns:
        J = N // a.d
        for T in Ts:
            for algo in ("bptt", "rtrl"):
                r = measure(J, a.d, a.q, V_BYTE, a.batch, T, algo)
                rows.append(r)
                print(f"{algo:>5s} {r['N']:6d} {T:7d} {a.batch:4d} "
                      f"{r['tokens_per_s']:11.3e} {r['wall']*1e3:9.2f} "
                      f"{r['elig_bytes']/1e6:9.3f} {r['model_bytes']/1e6:9.3f} "
                      f"{r['opt_bytes']/1e6:8.3f} {r['input_bytes']/1e6:9.3f} "
                      f"{r['xla_temp_bytes']/1e6:12.3f} "
                      f"{r['peak_device_bytes']/1e6 if r['peak_device_bytes']>0 else -1:12.1f} "
                      f"{str(r['oom']):>5s}", flush=True)
    print("\n== STREAMING RTRL (state + eligibility carried; sequence never materialized) ==")
    for N in Ns:
        r = streaming(N // a.d, a.d, a.q, V_BYTE, a.batch, 256, a.stream_chunks)
        rows.append(r)
        print(f"  N={r['J']*r['d']:6d} chunk=256 x {r['n_chunks']} = {r['tokens']:9d} tok "
              f"| {r['tokens_per_s']:11.3e} tok/s | elig {r['elig_bytes']/1e6:8.3f} MB "
              f"| chunk input {r['input_bytes']/1e6:7.4f} MB "
              f"| peak dev {r['peak_device_bytes']/1e6 if r['peak_device_bytes']>0 else -1:9.1f} MB")
    json.dump(dict(backend=jax.default_backend(),
                   devices=[str(x) for x in jax.devices()], rows=rows),
              open(a.out, "w"), indent=1, default=float)
    print(f"\nsaved {a.out}")
