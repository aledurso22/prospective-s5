"""Sequential MNIST training for baseline S5 and prospective SSM (JAX/flax).

Dataset: sequential MNIST — each 28x28 image is flattened row-major into a
784-length scalar sequence of pixel intensities in [0, 1] (optionally
standardized). MNIST IDX files are downloaded directly from
https://storage.googleapis.com/cvdf-datasets/mnist/ and parsed with
numpy + gzip + struct (no torchvision / tensorflow).

Usage:
    python train.py --model baseline    --epochs 3 --subset 20000
    python train.py --model prospective --epochs 3 --subset 20000

Metrics are written to results/metrics_<model>.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import time
import urllib.request

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ssm.model import build_model, count_params

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# MNIST IDX download + parsing (numpy + gzip + struct only)
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _parse_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _parse_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_dir: str = DATA_DIR,
               downsample: int = 1) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """Download (if needed) and parse MNIST; returns flattened sequences.

    Args:
        downsample: 1 keeps 28x28 (T=784); 2 average-pools 2x2 blocks to
            14x14 (T=196) — a sandbox adaptation for 2-core CPU training.

    Returns:
        x_train: (60000, T) float32 in [0, 1]
        y_train: (60000,) int64
        x_test:  (10000, T) float32 in [0, 1]
        y_test:  (10000,) int64
    """
    paths = {}
    for key, url in MNIST_URLS.items():
        dest = os.path.join(data_dir, os.path.basename(url))
        _download(url, dest)
        paths[key] = dest
    x_train = _parse_idx_images(paths["train_images"])
    y_train = _parse_idx_labels(paths["train_labels"])
    x_test = _parse_idx_images(paths["test_images"])
    y_test = _parse_idx_labels(paths["test_labels"])
    if downsample == 2:
        def ds(x):
            n = x.shape[0]
            return x.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(n, -1)
        x_train, x_test = ds(x_train), ds(x_test)
    elif downsample != 1:
        raise ValueError(f"unsupported downsample factor: {downsample}")
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def loss_fn(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@jax.jit(donate_argnums=(0,))
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params):
        logits = state.apply_fn(params, x, train=True,
                                rngs={"dropout": dropout_rng})
        return loss_fn(logits, y), logits

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return state, loss, acc


@jax.jit
def eval_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    logits = state.apply_fn(state.params, x, train=False)
    loss = loss_fn(logits, y)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc


def evaluate(state: TrainState, x: np.ndarray, y: np.ndarray,
             batch_size: int) -> tuple[float, float]:
    n = x.shape[0]
    total_loss, total_correct = 0.0, 0
    for i in range(0, n, batch_size):
        xb = jnp.asarray(x[i:i + batch_size])
        yb = jnp.asarray(y[i:i + batch_size])
        loss, acc = eval_step(state, xb, yb)
        bs = xb.shape[0]
        total_loss += float(loss) * bs
        total_correct += float(acc) * bs
    return total_loss / n, total_correct / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["baseline", "prospective"], required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=20000,
                   help="number of training samples (0 = full 60k)")
    p.add_argument("--test-subset", type=int, default=10000,
                   help="number of test samples for the FINAL eval (0 = full 10k)")
    p.add_argument("--eval-subset", type=int, default=2000,
                   help="per-epoch eval subset size (CPU budget; final eval "
                        "always uses --test-subset)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=96, help="H")
    p.add_argument("--state-size", type=int, default=64, help="N")
    p.add_argument("--n-layers", type=int, default=3, help="L")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--standardize", action="store_true",
                   help="standardize pixel intensities (zero mean, unit var)")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="prospective arm only: partial-prospection strength. "
                        "1.0 (default) = the derivation as written; 0.0 turns "
                        "the prospective term OFF and the layer collapses to "
                        "first-order explicit-Euler S5 — the matched control "
                        "(same params, same code path, same discretization, "
                        "unlike --model baseline which is bilinear).")
    p.add_argument("--stabilized", action="store_true",
                   help="prospective arm only: enable the documented stability "
                        "fallback (A := Delta*Lambda, clamps on Delta and rho). "
                        "DEFAULT IS OFF — the layer then runs the derivation "
                        "verbatim and diverges, which is the point. The "
                        "results/ metrics were produced WITH this flag.")
    p.add_argument("--rho-init", type=float, default=0.1,
                   help="prospective friction rho at init (prospective model "
                        "only). Physical mode ~ (1 - rho): memory horizon "
                        "~ 1/rho_init tokens. Use 1e-3 for seq_len ~ 784.")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the metrics filename, e.g. --tag _rho1e-3 "
                        "writes results/metrics_<model>_rho1e-3.json (lets "
                        "parallel runs of the same model coexist).")
    p.add_argument("--scan", choices=["assoc", "lax"], default="assoc",
                   help="scan implementation: assoc = jax.lax.associative_scan "
                        "(default; parallel-scan proof), lax = sequential "
                        "jax.lax.scan (much faster on XLA-CPU)")
    p.add_argument("--downsample", type=int, default=1, choices=[1, 2],
                   help="2 = average-pool 28x28 -> 14x14 (T=196; sandbox "
                        "adaptation for CPU training)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="if > 0, cap total optimizer steps (speed testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    # ---------------- data ----------------
    x_train, y_train, x_test, y_test = load_mnist(downsample=args.downsample)
    if args.standardize:
        mu, sigma = x_train.mean(), x_train.std() + 1e-8
        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma

    rng_np = np.random.default_rng(args.seed)
    if args.subset and args.subset < len(x_train):
        idx = rng_np.choice(len(x_train), args.subset, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
    if args.test_subset and args.test_subset < len(x_test):
        idx = rng_np.choice(len(x_test), args.test_subset, replace=False)
        x_test, y_test = x_test[idx], y_test[idx]
    # Per-epoch eval runs on a smaller fixed subset (CPU budget); the final
    # reported accuracy is computed once on the full --test-subset.
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
    print(f"train samples: {n_train}  test samples: {len(x_test)}  "
          f"steps/epoch: {steps_per_epoch}  total steps: {total_steps}")

    # ---------------- model ----------------
    model = build_model(model_type=args.model, d_model=args.d_model,
                        state_size=args.state_size, n_layers=args.n_layers,
                        dropout_rate=args.dropout, scan_impl=args.scan,
                        exact=not args.stabilized, gamma=args.gamma,
                        rho_init=args.rho_init)
    key = jax.random.PRNGKey(args.seed)
    init_rng, dropout_rng = jax.random.split(key)
    dummy = jnp.ones((1, x_train.shape[1]), jnp.float32)
    params = model.init({"params": init_rng, "dropout": dropout_rng},
                        dummy, train=False)
    n_params = count_params(params)
    print(f"model: {args.model}  params: {n_params:,}")

    schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    tx = optax.adam(schedule)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx,
                              dropout_rng=dropout_rng)

    config = dict(model=args.model, d_model=args.d_model,
                  state_size=args.state_size, n_layers=args.n_layers,
                  dropout=args.dropout, batch_size=args.batch_size,
                  lr=args.lr, schedule="cosine->0", optimizer="adam",
                  epochs=args.epochs, train_samples=n_train,
                  test_samples=len(x_test), seed=args.seed,
                  standardized=bool(args.standardize),
                  scan_impl=args.scan, downsample=args.downsample,
                  rho_init=(args.rho_init if args.model == "prospective"
                            else None),
                  exact=(not args.stabilized if args.model == "prospective"
                         else None),
                  gamma=(args.gamma if args.model == "prospective" else None),
                  seq_len=int(x_train.shape[1]))

    # ---------------- training loop ----------------
    history = []
    t_start = time.time()
    step = 0
    done = False
    for epoch in range(args.epochs):
        perm = rng_np.permutation(n_train)
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        t_ep = time.time()
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = jnp.asarray(x_train[idx])
            yb = jnp.asarray(y_train[idx])
            state, loss, acc = train_step(state, xb, yb)
            ep_loss += float(loss) * len(idx)
            ep_correct += float(acc) * len(idx)
            ep_n += len(idx)
            step += 1
            if step == 1 or step % 50 == 0:
                dt = time.time() - t_start
                print(f"  step {step:5d}/{total_steps}  loss {float(loss):.4f}  "
                      f"({step / max(dt, 1e-9):.2f} steps/s)")
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        test_loss, test_acc = evaluate(state, x_ev, y_ev, 512)
        ep_time = time.time() - t_ep
        history.append(dict(epoch=epoch + 1,
                            train_loss=ep_loss / max(ep_n, 1),
                            train_acc=ep_correct / max(ep_n, 1),
                            test_loss=test_loss, test_acc=test_acc,
                            eval_samples=len(x_ev),
                            epoch_time_sec=ep_time))
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"train loss {ep_loss / max(ep_n, 1):.4f}  "
              f"train acc {ep_correct / max(ep_n, 1):.4f}  "
              f"test acc ({len(x_ev)} samples) {test_acc:.4f}  ({ep_time:.1f}s)")
        if done:
            break

    # Final eval on the full --test-subset (default: all 10k).
    final_loss, final_test_acc = evaluate(state, x_test, y_test, 512)
    print(f"final eval on {len(x_test)} test samples: acc={final_test_acc:.4f}")
    wall = time.time() - t_start
    print(f"FINAL  model={args.model}  test acc={final_test_acc:.4f}  "
          f"wall={wall:.1f}s  ({step / wall:.2f} steps/s)")

    metrics = dict(config=config, params=n_params, history=history,
                   final_test_acc=final_test_acc, final_test_loss=final_loss,
                   final_eval_samples=len(x_test),
                   per_epoch_eval_samples=len(x_ev),
                   wall_time_sec=wall,
                   total_steps=step, steps_per_sec=step / wall,
                   device=str(jax.devices()[0]))
    out_path = os.path.join(RESULTS_DIR, f"metrics_{args.model}{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
