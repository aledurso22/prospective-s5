"""B38c training: matched BPTT (A) vs exact source-local RTRL (B), plus the
shared-selector BPTT control (C). Same update boundaries for A and B; no
every-token updates in the primary comparison."""
import argparse, json, os, time
import numpy as np
import jax
import jax.numpy as jnp

from model import (V_BYTE, REC_KEYS, OUT_KEYS, init_local, init_shared, nparams,
                   forward, ce_loss, rtrl_chunk, elig_bytes, bits_per_byte)
from data import load_bytes, batches, fixed_eval_set


def adam_step(p, m, v, g, t, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g)))
    sc = jnp.minimum(1.0, clip / (gn + 1e-12))
    g = jax.tree.map(lambda x: x * sc, g)
    t = t + 1
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b ** 2, v, g)
    mh = jax.tree.map(lambda x: x / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda x: x / (1 - b2 ** t), v)
    p = jax.tree.map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + eps), p, mh, vh)
    return p, m, v, t, gn


def make_update(J, d, q, V, arm, algo):
    """algo: 'bptt' (autodiff) or 'rtrl' (exact source-local forward credit).
    Both use identical chunk boundaries and carry h across chunks (stop-grad)."""
    def upd(p, m, v, t, h, x, y, lr):
        B, T = x.shape
        denom = float(B * T)
        if algo == "bptt":
            h = jax.lax.stop_gradient(h)
            loss, g = jax.value_and_grad(
                lambda pp: ce_loss(pp, x, y, J, d, arm, h))(p)
            hT, _ = forward(p, x, J, d, arm, h)
        else:
            g, hT = rtrl_chunk(p, h, x, y, J, d, V, denom)
            loss = ce_loss(p, x, y, J, d, arm, h)
        p, m, v, t, gn = adam_step(p, m, v, g, t, lr)
        return p, m, v, t, jax.lax.stop_gradient(hT), loss, gn
    return jax.jit(upd)


def make_both(J, d, q, V):
    """Both gradients at the same point, for in-training identity checking."""
    def both(p, h, x, y):
        B, T = x.shape
        denom = float(B * T)
        ga = jax.grad(lambda pp: ce_loss(pp, x, y, J, d, "L",
                                         jax.lax.stop_gradient(h)))(p)
        gb, _ = rtrl_chunk(p, h, x, y, J, d, V, denom)
        errs = jnp.stack([
            jnp.linalg.norm((gb[k] - ga[k]).ravel()) / (1 + jnp.linalg.norm(ga[k].ravel()))
            for k in list(REC_KEYS) + list(OUT_KEYS)])
        return errs
    return jax.jit(both)


def evaluate(p, J, d, arm, evalset):
    tot, n = 0.0, 0
    for x, y in evalset:
        tot += float(ce_loss(p, jnp.asarray(x), jnp.asarray(y), J, d, arm))
        n += 1
    return tot / n


def run(cfg):
    J, d, q = cfg["J"], cfg["d_tile"], cfg["q"]
    V, B, T = V_BYTE, cfg["batch"], cfg["chunk"]
    arm = cfg["arm"]                      # 'L' or 'C'
    algo = cfg["algo"]                    # 'bptt' or 'rtrl'
    tr, va, te, src = load_bytes(cfg)
    dtype = jnp.float64 if cfg.get("f64") else jnp.float32
    if cfg.get("f64"):
        jax.config.update("jax_enable_x64", True)
    p = (init_local(J, d, q, cfg["seed"], V=V, dtype=dtype) if arm == "L"
         else init_shared(J, d, q, cfg["seed"], V=V, dtype=dtype))
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    t = jnp.array(0.0)
    upd = make_update(J, d, q, V, arm, algo)
    both = make_both(J, d, q, V) if (arm == "L" and cfg.get("check_grads")) else None
    ev = fixed_eval_set(va, B, T, 1234, cfg.get("n_eval", 4))
    h = jnp.zeros((B, J, d), dtype)
    hist, worst = [], np.zeros(len(REC_KEYS) + len(OUT_KEYS))
    steps = cfg["steps"]
    t0 = time.time()
    ntok = 0
    for i, (x, y) in enumerate(batches(tr, B, T, cfg["seed"] + 77, steps)):
        x, y = jnp.asarray(x), jnp.asarray(y)
        if both is not None and i % max(1, steps // 20) == 0:
            worst = np.maximum(worst, np.asarray(both(p, h, x, y)))
        p, m, v, t, h, loss, gn = upd(p, m, v, t, h, x, y, cfg["lr"])
        ntok += B * T
        if (i + 1) % max(1, steps // 10) == 0:
            vl = evaluate(p, J, d, arm, ev)
            hist.append(dict(step=i + 1, train=float(loss), val=vl,
                             val_bpb=bits_per_byte(vl), gnorm=float(gn)))
            print(f"  step {i+1:6d}  train {float(loss):.4f}  val {vl:.4f}  "
                  f"val_bpb {bits_per_byte(vl):.4f}  |g| {float(gn):.3f}", flush=True)
    wall = time.time() - t0
    vl = evaluate(p, J, d, arm, ev)
    eb, brk = elig_bytes(B, J, d, q, V, 8 if cfg.get("f64") else 4)
    dev = [str(x) for x in jax.devices()]
    out = dict(cfg=cfg, source=src, P=nparams(p),
               backend=jax.default_backend(), devices=dev,
               P_rec=int(sum(np.prod(p[k].shape) for k in REC_KEYS if k in p)),
               val_ce=vl, val_bpb=bits_per_byte(vl), hist=hist,
               wall=wall, tokens_per_s=ntok / wall,
               elig_bytes=eb if algo == "rtrl" else 0,
               grad_check=worst.tolist() if both is not None else None)
    print(f"  FINAL val CE {vl:.4f} nats  |  {bits_per_byte(vl):.4f} bits/byte  "
          f"|  P={nparams(p)}  |  {wall:.1f}s  {ntok/wall:.0f} tok/s "
          f"on {jax.default_backend()} ({', '.join(dev)})")
    if both is not None:
        print(f"  worst in-training RTRL-vs-BPTT gradient error: {worst.max():.3e}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--override", default=None, help="JSON dict of overrides")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.override:
        cfg.update(json.loads(a.override))
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    r = run(cfg)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1, default=float)
