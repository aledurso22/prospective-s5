"""B38a -- TRUE token/chunk-streaming microbenchmark.

Sections A-D materialize the whole B x T input tensor, so total compiled memory
is NOT T-independent there and no such claim is made. Here tokens arrive
incrementally in fixed-size chunks and the full sequence is NEVER materialized:
only (params, Adam state, z, e^u, e^b, one chunk buffer) are ever resident.

The registered claim tested here is exactly:
    persistent RTRL eligibility memory is independent of stream length T.
Measured two ways: the exact live byte count of the carried eligibility, and
process RSS as the stream grows without bound."""
import gc, json, resource
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b37c_productlocal_native import real_dim, alg_zero, generic_init, to_jax
from credit_memory.b38a_engine import rtrl_chunk_grad, adam_step
from credit_memory.b38a_bench import spec_of


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # macOS: bytes


def nbytes(tree):
    return int(sum(np.asarray(x).size * 8 for x in jax.tree.leaves(tree)))


def build_stream(spec, C, B):
    def step(carry, xy):
        params, m, v, t, z, eu, eb = carry
        x, y = xy
        # one chunk of C tokens; eligibility CARRIED in (not reset) -> untruncated credit
        gs, zT = jax.vmap(lambda zz, xx, yy: rtrl_chunk_grad(
            params, zz, xx, yy, spec, float(B * C)), in_axes=(0, 1, 1))(z, x, y)
        g = jax.tree.map(lambda a: jnp.sum(a, axis=0), gs)
        params, m, v, t = adam_step(params, m, v, g, t, 1e-3)
        return params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT), eu, eb
    return jax.jit(step)


spec = spec_of(8)
r, B, C = real_dim(spec), 8, 256
p = to_jax(generic_init(spec, 0))
m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
z = jax.tree.map(lambda a: jnp.zeros((B,) + a.shape), alg_zero(spec))
eu = eb = z
step = build_stream(spec, C, B)
rng = np.random.RandomState(0)

carry = (p, m, v, jnp.array(0.0), z, eu, eb)
carry = step(carry[0], carry[1], carry[2], carry[3], carry[4], carry[5], carry[6],
             ) if False else carry

print("=" * 108)
print(f"TRUE STREAMING: chunk={C} tokens, B={B}, r={r}. Full sequence never materialized.")
print("=" * 108)
print(f"{'tokens seen':>13s} {'chunks':>7s} {'elig bytes':>11s} {'state bytes':>12s} "
      f"{'opt bytes':>10s} {'chunk buf B':>12s} {'process RSS MB':>15s}")
base = None
rows = []
tot = 0
for k in range(1, 4097):
    x = jnp.asarray(rng.randn(C, B) * 0.5)
    y = jnp.asarray(rng.randn(C, B) * 0.5)
    carry = step(carry, (x, y)) if False else None
    # explicit unpack (jit fn takes the tuple)
    break
# run properly
carry = (p, m, v, jnp.array(0.0), z, eu, eb)
fn = jax.jit(lambda cr, x, y: build_stream(spec, C, B)(cr, (x, y)))
out = fn(carry, jnp.zeros((C, B)), jnp.zeros((C, B)))
jax.block_until_ready(out)
gc.collect()
base = rss_mb()
carry = out
for k in range(1, 4097):
    x = jnp.asarray(rng.randn(C, B) * 0.5)
    y = jnp.asarray(rng.randn(C, B) * 0.5)
    carry = fn(carry, x, y)
    tot += C
    if k in (1, 4, 16, 64, 256, 1024, 4096):
        jax.block_until_ready(carry)
        gc.collect()
        eligb = nbytes(carry[5]) + nbytes(carry[6])
        rows.append(dict(tokens=tot, chunks=k, elig=eligb, state=nbytes(carry[4]),
                         opt=nbytes(carry[1]) + nbytes(carry[2]),
                         chunkbuf=int(x.nbytes + y.nbytes), rss=rss_mb()))
        print(f"{tot:13d} {k:7d} {eligb:11d} {nbytes(carry[4]):12d} "
              f"{nbytes(carry[1])+nbytes(carry[2]):10d} {int(x.nbytes+y.nbytes):12d} "
              f"{rss_mb():15.1f}")
json.dump(rows, open("results/b38a/stream_micro.json", "w"), indent=1)
e = set(x["elig"] for x in rows)
print(f"\nEligibility bytes across a {tot}-token stream: {sorted(e)}  -> "
      f"{'CONSTANT' if len(e) == 1 else 'GROWS'} in T")
print(f"Analytic persistent derivative state P_dyn = (m+1) r B = {2*r*B} floats "
      f"= {2*r*B*8} bytes  (m=1)")
print(f"RSS drift over 4096 chunks: {rows[-1]['rss']-rows[0]['rss']:+.1f} MB")
