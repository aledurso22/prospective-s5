"""Benchmark trainer — all arms x tasks, paired seeds/streams.

The cluster benchmark entry point. One run = one (task, arm, seed); the
SLURM grid (scripts/bench.sbatch + bench_grid.sh) assembles the matrix and
orders gates before mechanism arms.

Arms (--arm):
  baseline    exact-BPTT S5 (train.py semantics)
  online      streaming online gradient (custom-VJP rule; w = 1)
  tbptt       baseline forward, backward truncated to --tbptt-window
  routeA      live meta-learned per-(layer, channel, mode) complex w;
              teacher = exact-BPTT loss at the one-step-lookahead params
              (same batch, same dropout mask); meta-chain = plain SGD at
              the current schedule value (Adam/clip ignored in the chain —
              the documented cvm simplification, carried over verbatim)
  scalarLive  routeA with w constrained real (magnitude-only control)
  routePC     EXPLORATORY supplement (toy gate: route_pc.py PASS, median
              R_gap 0.90; audit: pc_signal_audit.py SIGNAL PRESENT).
              Correction-only Simonetto (PC0; prediction measured NOT
              load-bearing): the BPTT teacher is replaced by the realized
              online gradient on the newly arrived batch. Zero BPTT calls.
  routePCreal EXPLORATORY control: routePC with w constrained real
              (per-mode gain, phase pinned 0) — the causal arm's
              rotation-vs-gain control (route_pc_factorial.py).
  frozenPhase w = exp(i arg w_routeA) loaded from --w-file, never updated
  frozenMag   w = |w_routeA| loaded from --w-file, never updated

Tasks (--task):
  smnist      sequential MNIST (train.py's loader; --downsample allowed)
  psmnist     smnist + one fixed pixel permutation (seed 12345)
  copy        synthetic copy task: tokens at positions 0..K-1, delimiter,
              recall scored at the last K positions (seq2seq head, masked CE)

Paired protocol: data generation and batch permutations depend only on
(task config, seed) — never on the arm — so every arm of the same seed sees
identical streams. Mechanism arms require the headroom gate
h = (L_online - L_baseline) / L_online >= 0.2 (bench_report.py --gate);
the grid enforces this, the trainer itself runs what it is told.

Metrics -> results/bench/metrics_{task}_{arm}{tag}_s{seed}.json
routeA/scalarLive also save w -> results/bench/w_{task}_{arm}{tag}_s{seed}.npz

Usage:
    python train_bench.py --task smnist --arm routeA --seed 0 \
        --epochs 3 --subset 20000
    python train_bench.py --task copy --arm frozenPhase --seed 0 \
        --w-file results/bench/w_copy_routeA_s0.npz
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict

from train import load_mnist
from ssm.model import build_model, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "bench")

GATE_H = 0.2          # registered headroom bar (bench_report.py enforces)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def gen_copy(n: int, T: int, K: int, A: int, seed: int):
    """Copy task: tokens 1..A at positions 0..K-1, delimiter A+1 at
    T-K-1, blanks elsewhere; targets = the memorized tokens at the last K
    positions, -1 (ignored) at all others."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T), np.int8)
    x[:, :K] = rng.integers(1, A + 1, size=(n, K))
    x[:, T - K - 1] = A + 1
    y = -np.ones((n, T), np.int64)
    y[:, T - K:] = x[:, :K]
    return x, y


def load_task(args):
    """Returns (x_train, y_train, x_test, y_test, meta dict)."""
    if args.task in ("smnist", "psmnist"):
        x_train, y_train, x_test, y_test = load_mnist(
            downsample=args.downsample)
        if args.standardize:
            mu, sigma = x_train.mean(), x_train.std() + 1e-8
            x_train = (x_train - mu) / sigma
            x_test = (x_test - mu) / sigma
        if args.task == "psmnist":
            perm = np.random.default_rng(12345).permutation(x_train.shape[1])
            x_train, x_test = x_train[:, perm], x_test[:, perm]
        meta = dict(seq2seq=False, n_classes=10, one_hot=0)
        return x_train, y_train, x_test, y_test, meta

    # copy
    T, K, A = args.seq_len, args.copy_k, args.copy_alpha
    n_tr = args.subset if args.subset else 20000
    n_te = args.test_subset if args.test_subset else 2000
    x_train, y_train = gen_copy(n_tr, T, K, A, seed=args.seed + 777000)
    x_test, y_test = gen_copy(n_te, T, K, A, seed=args.seed + 888000)
    meta = dict(seq2seq=True, n_classes=A + 1, one_hot=A + 2)
    return x_train, y_train, x_test, y_test, meta


def prep_x(x: np.ndarray, one_hot: int) -> jnp.ndarray:
    if one_hot:
        return jnp.asarray(np.eye(one_hot, dtype=np.float32)[x])
    return jnp.asarray(x)


# ---------------------------------------------------------------------------
# Loss (masked for seq2seq)
# ---------------------------------------------------------------------------

def loss_fn(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    if labels.ndim == 1:
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, labels).mean()
    mask = labels >= 0
    lbl = jnp.clip(labels, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, lbl)
    return (ce * mask).sum() / mask.sum()


def acc_fn(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    pred = jnp.argmax(logits, axis=-1)
    if labels.ndim == 1:
        return jnp.mean(pred == labels)
    mask = labels >= 0
    return (jnp.sum((pred == labels) & mask) / mask.sum())


# ---------------------------------------------------------------------------
# Meta tree (w) helpers
# ---------------------------------------------------------------------------

def make_meta(params_tree, H: int, N: int):
    """Skeleton mirroring only the SSM nodes (dicts holding a Lambda and a
    log_step) — paths discovered from the actual param tree, not hardcoded."""
    def rec(node):
        if hasattr(node, "items"):
            if "Lambda" in node and "log_step" in node:
                return {"w_re": jnp.ones((H, N), jnp.float32),
                        "w_im": jnp.zeros((H, N), jnp.float32)}
            out = {}
            for k, v in node.items():
                sub = rec(v)
                if sub:
                    out[k] = sub
            return out or None
        return None
    return rec(params_tree)


def map_w(meta, fn):
    """Apply fn(w_re, w_im) -> (w_re, w_im) at every SSM node."""
    if hasattr(meta, "items"):
        if "w_re" in meta and "w_im" in meta:
            re, im = fn(np.asarray(meta["w_re"]), np.asarray(meta["w_im"]))
            return {"w_re": jnp.asarray(re, jnp.float32),
                    "w_im": jnp.asarray(im, jnp.float32)}
        return {k: map_w(v, fn) for k, v in meta.items()}
    raise TypeError(f"unexpected meta leaf: {type(meta)}")


def project_phase(re, im):
    r = np.hypot(re, im)
    r = np.where(r < 1e-12, 1.0, r)
    return re / r, im / r


def project_mag(re, im):
    return np.hypot(re, im), np.zeros_like(im)


def zero_wim_jnp(meta):
    """scalarLive projection (jnp-native, jit-traceable): pin w_im = 0."""
    if hasattr(meta, "items"):
        if "w_re" in meta and "w_im" in meta:
            return {"w_re": meta["w_re"],
                    "w_im": jnp.zeros_like(meta["w_im"])}
        return {k: zero_wim_jnp(v) for k, v in meta.items()}
    raise TypeError(f"unexpected meta leaf: {type(meta)}")


def save_meta(meta, path):
    flat = flatten_dict(unfreeze(meta), sep="/")
    np.savez(path, **{k: np.asarray(v) for k, v in flat.items()})


def load_meta(path):
    with np.load(path) as z:
        flat = {k: np.asarray(z[k]) for k in z.files}
    return freeze(unflatten_dict(flat, sep="/"))


def remap_for_teacher(params):
    """Online-model params -> baseline-model param tree.

    The two models differ only in the SSM class name, which flax uses as
    the module name: OnlineS5SSM_i -> S5SSM_i. Everything else is shared.
    """
    out = {}
    for k, v in params.items():
        if k.startswith("OnlineS5SSM_"):
            out["S5SSM_" + k[len("OnlineS5SSM_"):]] = v
        else:
            out[k] = v
    return freeze(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["smnist", "psmnist", "copy"],
                   required=True)
    p.add_argument("--arm", choices=["baseline", "online", "tbptt", "routeA",
                                     "scalarLive", "frozenPhase", "frozenMag",
                                     "routePC", "routePCreal"],
                   required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=20000,
                   help="train samples (0 = full 60k for smnist; 20000 copy)")
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--eval-subset", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-m", type=float, default=1e-3,
                   help="meta (w) learning rate; plain SGD, mirroring cvm")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--state-size", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--downsample", type=int, default=1, choices=[1, 2])
    p.add_argument("--scan", choices=["assoc", "lax"], default="assoc")
    p.add_argument("--tbptt-window", type=int, default=64)
    p.add_argument("--w-file", type=str, default="",
                   help="routeA w .npz for the frozen arms")
    p.add_argument("--seq-len", type=int, default=120, help="copy: T")
    p.add_argument("--copy-k", type=int, default=10, help="copy: recall len")
    p.add_argument("--copy-alpha", type=int, default=8, help="copy: alphabet")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    # ---------------- data ----------------
    x_train, y_train, x_test, y_test, tmeta = load_task(args)
    rng_np = np.random.default_rng(args.seed)
    if args.task in ("smnist", "psmnist"):
        if args.subset and args.subset < len(x_train):
            idx = rng_np.choice(len(x_train), args.subset, replace=False)
            x_train, y_train = x_train[idx], y_train[idx]
        if args.test_subset and args.test_subset < len(x_test):
            idx = rng_np.choice(len(x_test), args.test_subset,
                                replace=False)
            x_test, y_test = x_test[idx], y_test[idx]
    if args.eval_subset and args.eval_subset < len(x_test):
        idx = rng_np.choice(len(x_test), args.eval_subset, replace=False)
        x_ev, y_ev = x_test[idx], y_test[idx]
    else:
        x_ev, y_ev = x_test, y_test
    n_train = len(x_train)
    steps_per_epoch = n_train // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"task {args.task}  arm {args.arm}  seed {args.seed}  "
          f"train {n_train}  test {len(x_test)}  steps {total_steps}")

    # ---------------- models ----------------
    is_meta_arm = args.arm in ("routeA", "scalarLive", "frozenPhase",
                               "frozenMag", "routePC", "routePCreal")
    model_type = {"baseline": "baseline", "online": "online",
                  "tbptt": "tbptt"}.get(args.arm, "online")
    model = build_model(model_type=model_type, d_model=args.d_model,
                        state_size=args.state_size, n_layers=args.n_layers,
                        n_classes=tmeta["n_classes"],
                        dropout_rate=args.dropout, scan_impl=args.scan,
                        seq2seq=tmeta["seq2seq"],
                        tbptt_window=args.tbptt_window)
    key = jax.random.PRNGKey(args.seed)
    init_rng, dropout_rng = jax.random.split(key)
    T_in = args.seq_len if args.task == "copy" else x_train.shape[1]
    d_in = tmeta["one_hot"] if tmeta["one_hot"] else 1
    dummy = jnp.ones((1, T_in, d_in), jnp.float32)
    variables = model.init({"params": init_rng, "dropout": dropout_rng},
                           dummy, train=False)
    params = variables["params"]
    n_params = count_params(params)
    print(f"model: {model_type}  params: {n_params:,}")

    teacher = None
    if args.arm in ("routeA", "scalarLive"):
        teacher = build_model(model_type="baseline", d_model=args.d_model,
                              state_size=args.state_size,
                              n_layers=args.n_layers,
                              n_classes=tmeta["n_classes"],
                              dropout_rate=args.dropout,
                              scan_impl=args.scan,
                              seq2seq=tmeta["seq2seq"])

    # ---------------- meta (w) state ----------------
    meta = None
    meta_tx = None
    meta_state = None
    if is_meta_arm:
        meta = make_meta(params, args.d_model, args.state_size)
        if args.arm in ("frozenPhase", "frozenMag"):
            if not args.w_file:
                raise ValueError(f"--arm {args.arm} requires --w-file")
            loaded = load_meta(args.w_file)
            proj = project_phase if args.arm == "frozenPhase" else project_mag
            meta = map_w(loaded, proj)
            print(f"loaded w from {args.w_file} "
                  f"(projection: {args.arm})")
        else:
            meta_tx = optax.sgd(args.lr_m)
            meta_state = meta_tx.init(meta)

    # ---------------- step functions ----------------
    schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    tx = optax.adam(schedule)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx,
                              dropout_rng=dropout_rng)

    def apply_model(p, m, x, y, drng, train):
        logits = model.apply({"params": p, "meta": m}, x, train=train,
                             rngs={"dropout": drng} if train else None)
        return loss_fn(logits, y), logits

    @partial(jax.jit, donate_argnums=(0,))
    def plain_step(state, meta, x, y):
        drng, new_drng = jax.random.split(state.dropout_rng)
        (loss, logits), grads = jax.value_and_grad(
            apply_model, has_aux=True)(state.params, meta, x, y, drng, True)
        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_rng=new_drng)
        return state, loss, acc_fn(logits, y)

    @partial(jax.jit, donate_argnums=(0, 2))
    def routeA_step(state, meta, meta_state, x, y):
        drng, new_drng = jax.random.split(state.dropout_rng)
        params0 = state.params
        (loss, logits), grads = jax.value_and_grad(
            apply_model, has_aux=True)(params0, meta, x, y, drng, True)
        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_rng=new_drng)
        lr_t = schedule(state.step)

        def look(m):
            # online gradient with geometry m (custom VJP backward) ...
            g = jax.grad(lambda p: apply_model(p, m, x, y, drng,
                                               True)[0])(params0)
            # ... plain-SGD lookahead (Adam/clip ignored in the chain —
            # the documented cvm simplification) ...
            p_next = jax.tree_util.tree_map(
                lambda pp, gg: pp - lr_t * gg, params0, g)
            # ... teacher = exact-BPTT loss at the lookahead params
            logits = teacher.apply(
                {"params": remap_for_teacher(p_next)}, x, train=True,
                rngs={"dropout": drng})
            return loss_fn(logits, y)

        gw = jax.grad(look)(meta)
        upd, meta_state = meta_tx.update(gw, meta_state, meta)
        meta = optax.apply_updates(meta, upd)
        if args.arm == "scalarLive":
            meta = zero_wim_jnp(meta)
        return state, meta, meta_state, loss, acc_fn(logits, y)

    # routePC (exploratory supplement; toy gate PASS in route_pc.py):
    # correction-only Simonetto (beta = 0 — prediction was measured NOT
    # load-bearing). NO BPTT anywhere: the Route-A teacher is replaced by
    # the realized online gradient on the newly arrived batch.
    #
    # At batch n, with (x_prev, y_prev, params_prev, drng_prev) the
    # previous step's realized context and w_pred the geometry that
    # produced the current params:
    #   h_n   = online grad of the current batch, w = 1 (unscaled)
    #   c     = grad_w <h_n, g_prev(w)>|_{w_pred}     (the u/v chain,
    #           autodiff through the custom-VJP backward — same nested-grad
    #           primitive as routeA's look, FD-gated in
    #           check_routeA_meta.py)
    #   w_corr = w_pred + lr_m * lr_t * c      (toy sign convention:
    #           w -= LR_M * (-LR) * chain)
    #   applied update: Adam on the online grad of the current batch
    #           computed with w = w_corr.
    def routePC_corr(m, params_prev, x_prev, y_prev, drng_prev, h_n):
        g_prev = jax.grad(
            lambda p: apply_model(p, m, x_prev, y_prev, drng_prev,
                                  True)[0])(params_prev)
        return sum(jnp.sum(a * b) for a, b in
                   zip(jax.tree_util.tree_leaves(h_n),
                       jax.tree_util.tree_leaves(g_prev)))

    @partial(jax.jit, donate_argnums=(0, 2))
    def routePC_step(state, meta, meta_state, prev, x, y, drng):
        params0 = state.params
        x_prev, y_prev, params_prev, drng_prev = prev
        # (1) h_n: current batch's unscaled online gradient
        (loss, logits), h_n = jax.value_and_grad(
            apply_model, has_aux=True)(params0, {}, x, y, drng, True)
        lr_t = schedule(state.step + 1)
        # (2) delayed correction of the geometry that produced params0
        c = jax.grad(routePC_corr)(meta, params_prev, x_prev, y_prev,
                                   drng_prev, h_n)
        g_meta = jax.tree_util.tree_map(lambda cc: -lr_t * cc, c)
        upd, meta_state = meta_tx.update(g_meta, meta_state, meta)
        meta = optax.apply_updates(meta, upd)     # w_corr (PC0: pred=corr)
        if args.arm == "routePCreal":
            meta = zero_wim_jnp(meta)
        # (3) applied update with the corrected geometry
        grads = jax.grad(lambda p: apply_model(p, meta, x, y, drng,
                                               True)[0])(params0)
        state = state.apply_gradients(grads=grads)
        new_prev = (x, y, params0, drng)
        return state, meta, meta_state, new_prev, loss, acc_fn(logits, y)

    def pc_bootstrap(state, x, y, drng):
        """routePC step 1: no previous geometry exists, so the toy applies
        a plain online update with w = 1 and only records the context."""
        params0 = state.params
        (loss, logits), grads = jax.value_and_grad(
            apply_model, has_aux=True)(params0, {}, x, y, drng, True)
        state = state.apply_gradients(grads=grads)
        return state, loss, acc_fn(logits, y), (x, y, params0, drng)

    @jax.jit
    def eval_step(state, x, y):
        logits = model.apply({"params": state.params, "meta": {}}, x,
                             train=False)
        return loss_fn(logits, y), acc_fn(logits, y)

    def evaluate(xd, yd, batch_size):
        n = xd.shape[0]
        tot_l, tot_c = 0.0, 0.0
        for i in range(0, n, batch_size):
            xb = prep_x(xd[i:i + batch_size], tmeta["one_hot"])
            yb = jnp.asarray(yd[i:i + batch_size])
            l, a = eval_step(state, xb, yb)
            bs = xb.shape[0]
            tot_l += float(l) * bs
            tot_c += float(a) * bs
        return tot_l / n, tot_c / n

    # ---------------- startup self-checks (meta arms) ----------------
    if args.arm in ("routeA", "scalarLive", "routePC", "routePCreal"):
        xb = prep_x(x_train[:8], tmeta["one_hot"])
        yb = jnp.asarray(y_train[:8])
        dr = jax.random.PRNGKey(0)
        l_nometa, g_nometa = jax.value_and_grad(
            lambda p: apply_model(p, {}, xb, yb, dr, True)[0])(state.params)
        l_meta, g_meta = jax.value_and_grad(
            lambda p: apply_model(p, meta, xb, yb, dr, True)[0])(state.params)
        d = max(float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(jax.tree_util.tree_leaves(g_nometa),
                                jax.tree_util.tree_leaves(g_meta)))
        print(f"self-check: w=1 meta == no meta  "
              f"(|dloss| {abs(float(l_nometa) - float(l_meta)):.2e}, "
              f"max |dgrad| {d:.2e})")
        assert d < 1e-6, "meta-path wiring bug: w=1 must equal no meta"
        if teacher is not None:
            lo = teacher.apply({"params": remap_for_teacher(state.params)},
                               xb, train=False)
            lm = model.apply({"params": state.params}, xb, train=False)
            dt = float(jnp.max(jnp.abs(lo - lm)))
            print(f"self-check: teacher forward == online forward "
                  f"(max |dlogit| {dt:.2e})")
            assert dt < 1e-4, "teacher/online forward mismatch"

    # ---------------- training loop ----------------
    history = []
    t_start = time.time()
    step = 0
    done = False
    prev = None
    for epoch in range(args.epochs):
        perm = rng_np.permutation(n_train)
        ep_loss, ep_acc, ep_n = 0.0, 0.0, 0
        t_ep = time.time()
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = prep_x(x_train[idx], tmeta["one_hot"])
            yb = jnp.asarray(y_train[idx])
            if args.arm in ("routeA", "scalarLive"):
                state, meta, meta_state, loss, acc = routeA_step(
                    state, meta, meta_state, xb, yb)
            elif args.arm in ("routePC", "routePCreal"):
                drng, new_drng = jax.random.split(state.dropout_rng)
                state = state.replace(dropout_rng=new_drng)
                if prev is None:
                    state, loss, acc, prev = pc_bootstrap(
                        state, xb, yb, drng)
                else:
                    state, meta, meta_state, prev, loss, acc = routePC_step(
                        state, meta, meta_state, prev, xb, yb, drng)
            elif args.arm in ("frozenPhase", "frozenMag"):
                state, loss, acc = plain_step(state, meta, xb, yb)
            else:
                state, loss, acc = plain_step(state, {}, xb, yb)
            ep_loss += float(loss) * len(idx)
            ep_acc += float(acc) * len(idx)
            ep_n += len(idx)
            step += 1
            if step == 1 or step % 50 == 0:
                dt = time.time() - t_start
                print(f"  step {step:5d}/{total_steps}  loss "
                      f"{float(loss):.4f}  ({step / max(dt, 1e-9):.2f} "
                      f"steps/s)", flush=True)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        test_loss, test_acc = evaluate(x_ev, y_ev, 512)
        ep_time = time.time() - t_ep
        history.append(dict(epoch=epoch + 1,
                            train_loss=ep_loss / max(ep_n, 1),
                            train_acc=ep_acc / max(ep_n, 1),
                            test_loss=test_loss, test_acc=test_acc,
                            eval_samples=len(x_ev), epoch_time_sec=ep_time))
        print(f"epoch {epoch + 1}/{args.epochs}  train loss "
              f"{ep_loss / max(ep_n, 1):.4f}  test acc ({len(x_ev)}) "
              f"{test_acc:.4f}  ({ep_time:.1f}s)", flush=True)
        if done:
            break

    final_loss, final_acc = evaluate(x_test, y_test, 512)
    wall = time.time() - t_start
    print(f"FINAL  task={args.task} arm={args.arm} seed={args.seed}  "
          f"test acc={final_acc:.4f}  test loss={final_loss:.4f}  "
          f"wall={wall:.1f}s  ({step / wall:.2f} steps/s)")

    w_path = ""
    if args.arm in ("routeA", "scalarLive", "routePC", "routePCreal"):
        w_path = os.path.join(
            RESULTS_DIR,
            f"w_{args.task}_{args.arm}{args.tag}_s{args.seed}.npz")
        save_meta(meta, w_path)
        print(f"wrote {w_path}")

    config = dict(task=args.task, arm=args.arm, model_type=model_type,
                  d_model=args.d_model, state_size=args.state_size,
                  n_layers=args.n_layers, dropout=args.dropout,
                  batch_size=args.batch_size, lr=args.lr, lr_m=args.lr_m,
                  schedule="cosine->0", optimizer="adam",
                  meta_optimizer=("sgd" if meta_tx else None),
                  epochs=args.epochs, train_samples=n_train,
                  test_samples=len(x_test), seed=args.seed,
                  standardized=bool(args.standardize),
                  scan_impl=args.scan, downsample=args.downsample,
                  seq_len=int(T_in), seq2seq=tmeta["seq2seq"],
                  tbptt_window=(args.tbptt_window if args.arm == "tbptt"
                                else None),
                  w_file=(args.w_file if args.arm in ("frozenPhase",
                                                      "frozenMag")
                          else None),
                  copy_k=(args.copy_k if args.task == "copy" else None),
                  copy_alpha=(args.copy_alpha if args.task == "copy"
                              else None))
    metrics = dict(config=config, params=n_params, history=history,
                   final_test_acc=final_acc, final_test_loss=final_loss,
                   final_eval_samples=len(x_test),
                   per_epoch_eval_samples=len(x_ev),
                   wall_time_sec=wall, total_steps=step,
                   steps_per_sec=step / wall,
                   device=str(jax.devices()[0]), w_file=w_path or None)
    out_path = os.path.join(
        RESULTS_DIR,
        f"metrics_{args.task}_{args.arm}{args.tag}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
