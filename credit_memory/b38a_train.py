"""B38a -- matched trainer. Arms A (TBPTT) and B (forward causal RTRL) share
initialization, data, chunk schedule, Adam state and evaluation exactly; the
ONLY difference is which routine produces each chunk gradient."""
from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b37b_quotient_trainability import make_teacher, make_dataset
from credit_memory.b37c_productlocal_native import (
    spec_from_blocks, real_dim, alg_zero, generic_init, to_jax, make_M,
    spectral_radius, flat)
from credit_memory.b38a_engine import (
    tbptt_batch_grad, rtrl_batch_grad, adam_step, build_eval, chunk_forward)

FAMILIES_ALL = ('random_stable_diag', 'distinct_real', 'complex_conjugate',
                'repeated_poles', 'exact_jordan', 'multi_jordan_shared',
                'nearly_defective', 'nonnormal', 'stiff')
T_SEQ = 256
N_TRAIN, N_VAL, N_TEST = 32, 16, 16
EPOCHS = 200
EP_BLOCK = 10
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)
EVAL_SEEDS = (0, 1, 2)
L_VALUES = (32, 128, 256)
DIVERGE = 1e12

_DATA = {}
_TEACH = {}


def make_teacher_norm(family, r, seed):
    """B37b's teacher with C_* rescaled so the target has unit variance.

    DECLARED PREPROCESSING, applied identically to arms A, B and C. At T=256 the
    high-transient families reach a large steady-state output amplitude
    (exact_jordan mean y^2 = 2.8e6, nonnormal 2.6e7), which would require port
    norms of ~1e3 and confound a training-ALGORITHM comparison with a pure
    target-scale effect. Scaling C_* is an exact output relabelling: it changes
    neither the teacher's dynamics, its Jordan structure, nor its conditioning,
    and B37b's make_teacher is imported unmodified."""
    key = (family, r, seed)
    if key not in _TEACH:
        t = dict(make_teacher(family, r, seed))
        xs = np.random.RandomState(4242).randn(8, T_SEQ) * 0.5
        A, Bs, Cs = t["A"], t["Bs"], t["Cs"]
        ys = []
        for i in range(8):
            h = np.zeros(A.shape[0])
            for x in xs[i]:
                h = A @ h + Bs * x
                ys.append(Cs @ h)
        sig = float(np.std(ys))
        t["Cs"] = Cs / sig
        t["y_scale"] = sig
        _TEACH[key] = t
    return _TEACH[key]


def data_for(teacher, family, r, seed):
    key = (family, r, seed)
    if key not in _DATA:
        _DATA[key] = (make_dataset(teacher, N_TRAIN, T_SEQ, 21000 + seed),
                      make_dataset(teacher, N_VAL, T_SEQ, 31000 + seed),
                      make_dataset(teacher, N_TEST, T_SEQ, 41000 + seed))
    return _DATA[key]


def batch_z0(spec, B):
    return jax.tree.map(lambda a: jnp.zeros((B,) + a.shape), alg_zero(spec))


@functools.lru_cache(maxsize=None)
def build_epochs(spec, L, arm, n_ep, B):
    """n_ep epochs, each scanning all chunks in order, carrying z across chunks
    (stop-gradient at every boundary) and resetting eligibility per chunk."""
    gradf = tbptt_batch_grad if arm == "A" else rtrl_batch_grad
    denom = float(B * L)

    def one_chunk(carry, chunk):
        params, m, v, t, z = carry
        xs, ys = chunk
        g, zT = gradf(params, z, xs, ys, spec, denom)
        params, m, v, t = adam_step(params, m, v, g, t, lr_holder[0])
        return (params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT)), 0.0

    def run(params, m, v, t, xs, ys, lr):
        nch = xs.shape[1] // L
        xc = xs[:, :nch * L].reshape(B, nch, L).transpose(1, 0, 2)
        yc = ys[:, :nch * L].reshape(B, nch, L).transpose(1, 0, 2)

        def chunk(carry, cc):
            params, m, v, t, z = carry
            x, y = cc
            g, zT = gradf(params, z, x, y, spec, denom)
            params, m, v, t = adam_step(params, m, v, g, t, lr)
            return (params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT)), None

        def epoch(carry, _):
            params, m, v, t = carry
            (params, m, v, t, _), _ = jax.lax.scan(
                chunk, (params, m, v, t, batch_z0(spec, B)), (xc, yc))
            return (params, m, v, t), None
        (params, m, v, t), _ = jax.lax.scan(epoch, (params, m, v, t), None, length=n_ep)
        return params, m, v, t
    return jax.jit(run)


lr_holder = [0.0]


def train(family, r, seed, arm, L, lr, epochs=EPOCHS, params0=None):
    teacher = make_teacher_norm(family, r, seed)
    spec = spec_from_blocks(teacher["blocks"])
    (xtr, ytr), (xva, yva), (xte, yte) = data_for(teacher, family, r, seed)
    ev = build_eval(spec)
    params = to_jax(generic_init(spec, seed)) if params0 is None else params0
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    t = jnp.array(0.0)
    runner = build_epochs(spec, L, arm, EP_BLOCK, N_TRAIN)
    best_val, best = float(ev(params, xva, yva)), params
    curve, diverged = [], False
    for blk in range(epochs // EP_BLOCK):
        params, m, v, t = runner(params, m, v, t, xtr, ytr, lr)
        vl = float(ev(params, xva, yva))
        curve.append(vl)
        if not np.isfinite(vl) or vl > DIVERGE:
            diverged = True
            break
        if vl < best_val:
            best_val, best = vl, params
    ynorm = float(jnp.mean(yte ** 2))
    return dict(test_nmse=float(ev(best, xte, yte)) / (ynorm + 1e-30),
                val_loss=best_val, curve=curve, diverged=bool(diverged),
                params=best, spec=spec, teacher=teacher)


def markov_err(params, teacher, spec, K=40):
    M = make_M(tuple(np.asarray(x) for x in params["u"]), spec)
    C = np.asarray(params["C"])[0]
    vs = np.asarray(flat(params["b"]))
    A, Bs, Cs = teacher["A"], teacher["Bs"], teacher["Cs"]
    vt, worst = Bs.copy(), 0.0
    for _ in range(K):
        gs, gt = float(C @ vs), float(Cs @ vt)
        if not np.isfinite(gs):
            return float("inf")
        worst = max(worst, abs(gs - gt) / (1 + abs(gt)))
        vs, vt = M @ vs, A @ vt
    return float(worst)


@functools.lru_cache(maxsize=None)
def build_epochs_check(spec, L, n_ep, B):
    """Identical to build_epochs(arm='B') but ALSO computes the matched TBPTT
    gradient at the same point every chunk and emits the per-block relative
    errors. The RTRL gradient is the one that drives the update, so this is the
    identity verified during actual training, not on an isolated toy case."""
    denom = float(B * L)

    def relerr(a, b):
        fa = jnp.concatenate([x.ravel() for x in jax.tree.leaves(a)])
        fb = jnp.concatenate([x.ravel() for x in jax.tree.leaves(b)])
        return jnp.linalg.norm(fa - fb) / (1.0 + jnp.linalg.norm(fb))

    def run(params, m, v, t, xs, ys, lr):
        nch = xs.shape[1] // L
        xc = xs[:, :nch * L].reshape(B, nch, L).transpose(1, 0, 2)
        yc = ys[:, :nch * L].reshape(B, nch, L).transpose(1, 0, 2)

        def chunk(carry, cc):
            params, m, v, t, z = carry
            x, y = cc
            ga, _ = tbptt_batch_grad(params, z, x, y, spec, denom)
            gb, zT = rtrl_batch_grad(params, z, x, y, spec, denom)
            errs = jnp.stack([relerr(gb["u"], ga["u"]), relerr(gb["b"], ga["b"]),
                              relerr(gb["C"], ga["C"])])
            params, m, v, t = adam_step(params, m, v, gb, t, lr)   # RTRL drives it
            return (params, m, v, t, jax.tree.map(jax.lax.stop_gradient, zT)), errs

        def epoch(carry, _):
            params, m, v, t = carry
            (params, m, v, t, _), e = jax.lax.scan(
                chunk, (params, m, v, t, batch_z0(spec, B)), (xc, yc))
            return (params, m, v, t), jnp.max(e, axis=0)
        (params, m, v, t), e = jax.lax.scan(epoch, (params, m, v, t), None, length=n_ep)
        return params, m, v, t, jnp.max(e, axis=0)
    return jax.jit(run)
