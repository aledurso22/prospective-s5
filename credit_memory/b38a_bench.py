"""B38a section 3 -- derivative-memory scaling and steady-state timing.

BACKEND NOTE: this machine exposes a CPU-only JAX backend (no CUDA device), so
every number below is CPU. Peak *GPU* memory cannot be measured here. What IS
measured is XLA's own compiled scratch requirement (temp_size_in_bytes), which
is the backend-reported working-set the executable needs and is the quantity
that becomes device memory on GPU, plus argument/output buffer sizes.

Protocol: compile + warm up, jax.block_until_ready before and after every timed
region, compilation excluded, >=5 timed reps, median reported. No kernel
optimization of any kind.
"""
import json, time
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b37c_productlocal_native import real_dim, alg_zero, generic_init, to_jax
from credit_memory.b38a_engine import tbptt_batch_grad, rtrl_batch_grad, adam_step


def spec_of(r, d=1):
    assert r % d == 0
    return (("R", d),) * (r // d)


def build(spec, arm, L):
    """Inputs arrive ALREADY shaped (nch, B, L) so the compiled executable's
    temp_size reflects the compute working set only, not a data reshape copy --
    otherwise both arms' scratch would be dominated by the B*T input buffer and
    the derivative-memory question would be masked."""
    gradf = tbptt_batch_grad if arm == "A" else rtrl_batch_grad

    def upd(params, m, v, t, z, xc, yc, lr):
        B = xc.shape[1]

        def chunk(carry, cc):
            params, m, v, t, z = carry
            x, y = cc
            g, zT = gradf(params, z, x, y, spec, float(B * L))
            params, m, v, t = adam_step(params, m, v, g, t, lr)
            return (params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT)), None
        (params, m, v, t, z), _ = jax.lax.scan(chunk, (params, m, v, t, z), (xc, yc))
        return params, m, v, t, z
    return jax.jit(upd)


def measure(spec, arm, T, B, L, reps=5):
    r = real_dim(spec)
    rng = np.random.RandomState(0)
    nch = T // L
    xs = jnp.asarray(rng.randn(nch, B, L) * 0.5)
    ys = jnp.asarray(rng.randn(nch, B, L) * 0.5)
    p = to_jax(generic_init(spec, 0))
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    t = jnp.array(0.0)
    z = jax.tree.map(lambda a: jnp.zeros((B,) + a.shape), alg_zero(spec))
    fn = build(spec, arm, L)
    args = (p, m, v, t, z, xs, ys, 1e-3)
    try:
        st = fn.lower(*args).compile().memory_analysis()
        temp, argsz = st.temp_size_in_bytes, st.argument_size_in_bytes
    except Exception:
        temp, argsz = -1, -1
    out = fn(*args); jax.block_until_ready(out)            # warm up, excludes compile
    ts = []
    for _ in range(reps):
        jax.block_until_ready(out)
        s = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - s)
    wall = float(np.median(ts))
    nupd = nch
    P = 3 * r                                   # u (r) + b (r) + C_out (r)
    return dict(arm=arm, T=T, B=B, L=L, r=r, wall=wall, per_update=wall / nupd,
                tokens_per_s=B * T / wall, temp_bytes=int(temp), arg_bytes=int(argsz),
                updates=nupd,
                mem_state=B * r * 8,            # recurrent state
                mem_elig=(B * 2 * r * 8 if arm == "B" else 0),   # (m+1)r per sample, m=1
                mem_opt=2 * P * 8,              # Adam m, v
                mem_data=int(xs.nbytes + ys.nbytes))


rows = []
def line(d, tag):
    rows.append(dict(d, tag=tag))
    print(f"{tag:9s} arm={d['arm']} T={d['T']:6d} B={d['B']:4d} r={d['r']:4d} L={d['L']:6d} "
          f"| {d['wall']*1e3:9.2f} ms {d['per_update']*1e6:9.1f} us/upd "
          f"{d['tokens_per_s']:11.3e} tok/s | XLA temp {d['temp_bytes']/1e6:9.3f} MB "
          f"| state {d['mem_state']/1e3:7.2f} kB elig {d['mem_elig']/1e3:7.2f} kB "
          f"opt {d['mem_opt']/1e3:6.2f} kB", flush=True)

print("=" * 128)
print("A. SEQUENCE-LENGTH SCALING   (B=8, r=8; TBPTT window = full T vs truncated L=128; RTRL over full T)")
print("=" * 128)
sp = spec_of(8)
for T in (128, 512, 2048, 8192, 32768):
    line(measure(sp, "A", T, 8, T), "bptt-full")
    line(measure(sp, "A", T, 8, min(128, T)), "tbptt-128")
    line(measure(sp, "B", T, 8, T), "rtrl-full")

print()
print("=" * 128)
print("B. BATCH SCALING   (T=512, r=8, full-window)")
print("=" * 128)
for B in (1, 8, 32, 128):
    line(measure(sp, "A", 512, B, 512), "bptt-full")
    line(measure(sp, "B", 512, B, 512), "rtrl-full")

print()
print("=" * 128)
print("C. STATE-DIMENSION SCALING   (T=512, B=8, full-window, pi = R1^r)")
print("=" * 128)
for r in (8, 32, 128, 512):
    s = spec_of(r)
    line(measure(s, "A", 512, 8, 512), "bptt-full")
    line(measure(s, "B", 512, 8, 512), "rtrl-full")

json.dump(rows, open("results/b38a/bench.json", "w"), indent=1)
print("\nsaved results/b38a/bench.json")


# ---------------------------------------------------------------------------
# D. Time-major inputs. In sections A-C the batch is the LEADING data axis, so
# vmap-over-scan must transpose each (B, L) input into (L, B), costing O(B*T)
# scratch for BOTH methods. That input-layout cost is shared by any streaming
# method and is not derivative memory. Feeding data already time-major removes
# it and isolates the actual persistent derivative state.
# ---------------------------------------------------------------------------
from credit_memory.b38a_engine import rtrl_chunk_grad, chunk_loss, chunk_forward


def build_tm(spec, arm, L):
    def upd(params, m, v, t, z, xc, yc, lr):
        B = xc.shape[2]

        def chunk(carry, cc):
            params, m, v, t, z = carry
            x, y = cc                                   # (L, B)
            if arm == "B":
                gs, zT = jax.vmap(lambda zz, xx, yy: rtrl_chunk_grad(
                    params, zz, xx, yy, spec, float(B * L)), in_axes=(0, 1, 1))(z, x, y)
                g = jax.tree.map(lambda a: jnp.sum(a, axis=0), gs)
            else:
                def loss(p):
                    return jnp.sum(jax.vmap(lambda zz, xx, yy: chunk_loss(
                        p, zz, xx, yy, spec, float(B * L)), in_axes=(0, 1, 1))(z, x, y))
                g = jax.grad(loss)(params)
                zT, _ = jax.vmap(lambda zz, xx: chunk_forward(params, zz, xx, spec),
                                 in_axes=(0, 1))(z, x)
            params, m, v, t = adam_step(params, m, v, g, t, lr)
            return (params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT)), None
        (params, m, v, t, z), _ = jax.lax.scan(chunk, (params, m, v, t, z), (xc, yc))
        return params, m, v, t, z
    return jax.jit(upd)


def measure_tm(spec, arm, T, B, L, reps=5):
    r = real_dim(spec)
    rng = np.random.RandomState(0)
    nch = T // L
    xs = jnp.asarray(rng.randn(nch, L, B) * 0.5)
    ys = jnp.asarray(rng.randn(nch, L, B) * 0.5)
    p = to_jax(generic_init(spec, 0))
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    z = jax.tree.map(lambda a: jnp.zeros((B,) + a.shape), alg_zero(spec))
    fn = build_tm(spec, arm, L)
    args = (p, m, v, jnp.array(0.0), z, xs, ys, 1e-3)
    try:
        temp = fn.lower(*args).compile().memory_analysis().temp_size_in_bytes
    except Exception:
        temp = -1
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        jax.block_until_ready(out)
        s = time.perf_counter(); out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - s)
    wall = float(np.median(ts))
    P = 3 * r
    return dict(arm=arm, T=T, B=B, L=L, r=r, wall=wall, per_update=wall / nch,
                tokens_per_s=B * T / wall, temp_bytes=int(temp), arg_bytes=0,
                updates=nch, mem_state=B * r * 8,
                mem_elig=(B * 2 * r * 8 if arm == "B" else 0), mem_opt=2 * P * 8,
                mem_data=int(xs.nbytes + ys.nbytes))


print()
print("=" * 128)
print("D. TIME-MAJOR INPUTS -- isolates persistent derivative memory (B=8, r=8, full window)")
print("=" * 128)
for T in (128, 512, 2048, 8192, 32768):
    line(measure_tm(sp, "A", T, 8, T), "bptt-tm")
    line(measure_tm(sp, "B", T, 8, T), "rtrl-tm")
json.dump(rows, open("results/b38a/bench.json", "w"), indent=1)
print("\nsaved results/b38a/bench.json")
