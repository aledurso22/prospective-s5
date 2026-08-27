"""S5 CCM Phase S1 training entry point.

Implements S0 (online), S1 (full exact causal CCM), and Sref (BPTT).
S2/S3 (rank-1 frozen/reactive) and S4+ (r=2/4/8) are NOT implemented --
see experiments/s5_ccm/ccm_rank1.py's "S2/S3 implementation status" and
README.md's "S4+ status" for exactly what is ready and what remains.
Selecting --arm rank1_frozen/rank1_reactive/rank2/rank4/rank8 raises a
clear error rather than silently running something unverified.

Reuses train_bench.py's data loaders/loss/accuracy/provenance functions
directly (no reimplementation, no risk of silently diverging from the
existing, tested S5 benchmark's data handling).

MUST run experiments/s5_ccm/exactness_check.py successfully (exit 0)
before trusting --arm s1_full_causal on real data -- this script does
NOT re-verify exactness itself (that would require BPTT on every step,
defeating S1's own point); it only enforces that a config/env-controlled
exactness-check artifact exists and passed, if present, and prints a
loud warning if it cannot find one.

Run:
  python -m experiments.s5_ccm.train --config experiments/s5_ccm/configs/l2_smoke.json --arm online
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import train_bench as tb
from experiments.s5_ccm.ccm_core import build_mixed_classifier

RESULTS_DIR_DEFAULT = os.environ.get(
    "S5_CCM_RESULTS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
DATA_ROOT_DEFAULT = os.environ.get(
    "S5_CCM_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "data"))
CHECKPOINT_ROOT_DEFAULT = os.environ.get(
    "S5_CCM_CHECKPOINT_ROOT",
    os.path.join(RESULTS_DIR_DEFAULT, "checkpoints"))

ARM_MODEL_TYPES = {
    "online": ["online", "online"],
    "s1_full_causal": ["online", "baseline"],
    "bptt": ["baseline", "baseline"],
}
NOT_IMPLEMENTED_ARMS = {
    "rank1_frozen": "S2",
    "rank1_reactive": "S3",
    "rank2": "S4", "rank4": "S4", "rank8": "S4",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file; CLI flags override its values")
    p.add_argument("--arm", type=str, default="online",
                   choices=list(ARM_MODEL_TYPES) + list(NOT_IMPLEMENTED_ARMS))
    p.add_argument("--task", type=str, default="smnist",
                   choices=["smnist", "psmnist", "copy"])
    p.add_argument("--n-layers", type=int, default=2,
                   help="L=2 only in this phase; the arm-model_types "
                        "table and exactness construction are L=2-"
                        "specific (see README 'What NOT to do'#7)")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--state-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--scan", choices=["assoc", "lax"], default="assoc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=0,
                   help="if >0, cap total optimizer steps regardless of "
                        "--epochs (used by the smoke config)")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--eval-subset", type=int, default=2000)
    p.add_argument("--downsample", type=int, default=1, choices=[1, 2])
    p.add_argument("--standardize", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=120)
    p.add_argument("--copy-k", type=int, default=10)
    p.add_argument("--copy-alpha", type=int, default=8)
    p.add_argument("--diagnostic-checkpoints", type=str,
                   default="0,200,600,1200",
                   help="comma-separated optimizer steps at which to run "
                        "the offline gradient-cosine-vs-BPTT probe "
                        "(diagnostic only, never used in the update)")
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    p.add_argument("--data-root", type=str, default=DATA_ROOT_DEFAULT)
    p.add_argument("--checkpoint-root", type=str,
                   default=CHECKPOINT_ROOT_DEFAULT)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--resume", type=int, default=1,
                   help="1: skip if the output JSON for this exact "
                        "(arm,task,seed,tag) already exists and is "
                        "finite; 0: always rerun")
    args, _ = p.parse_known_args(argv)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)
        args, _ = p.parse_known_args(argv, namespace=args)
    if args.n_layers != 2:
        raise ValueError("this phase (S1) is L=2 only -- see README "
                         "'What NOT to do'; do not pass --n-layers != 2")
    return args


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    fname = f"s5ccm_{args.arm}_{args.task}_clip{args.clip}_s{args.seed}{tag}.json"
    return os.path.join(args.results_dir, fname)


def build_model(args, n_classes, seq2seq):
    if args.arm in NOT_IMPLEMENTED_ARMS:
        phase = NOT_IMPLEMENTED_ARMS[args.arm]
        raise NotImplementedError(
            f"--arm {args.arm} ({phase}) is not implemented in this "
            f"preparation phase. See experiments/s5_ccm/ccm_rank1.py's "
            f"'S2/S3 implementation status' (S2/S3) or README.md's "
            f"'S4+ status' (S4+) for exactly what is ready and what "
            f"remains -- do not improvise a selector or reduction here.")
    model_types = ARM_MODEL_TYPES[args.arm]
    return build_mixed_classifier(model_types, d_model=args.d_model,
                                  state_size=args.state_size,
                                  n_classes=n_classes,
                                  dropout_rate=args.dropout,
                                  scan_impl=args.scan, seq2seq=seq2seq)


def check_exactness_gate(args):
    """Refuses to proceed with --arm s1_full_causal unless an exactness-
    check artifact exists and passed -- loud failure, not a silent
    continue, per instruction ('the cluster run should fail loudly rather
    than continue')."""
    if args.arm != "s1_full_causal":
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "exactness_check_summary.json")
    if not os.path.exists(path):
        raise SystemExit(
            "REFUSING TO RUN --arm s1_full_causal: no exactness-check "
            f"artifact found at {path}. Run "
            "`python -m experiments.s5_ccm.exactness_check` first (see "
            "README 'Recommended first sbatch sequence' step 1) and "
            "confirm it passes before training with this arm.")
    doc = json.load(open(path))
    if not doc.get("all_ssm0_recurrence_gates_pass", False):
        raise SystemExit(
            f"REFUSING TO RUN --arm s1_full_causal: the exactness check "
            f"at {path} did NOT pass (all_ssm0_recurrence_gates_pass="
            f"{doc.get('all_ssm0_recurrence_gates_pass')}). Diagnose "
            "before training -- do not proceed with an unverified "
            "construction.")
    print(f"exactness gate: OK (verified at git {doc.get('git', '?')})")


def main() -> None:
    args = parse_args()
    check_exactness_gate(args)
    op = out_path(args)
    if args.resume and os.path.exists(op):
        prior = json.load(open(op))
        if prior.get("finite", False):
            print(f"resume: {op} already exists and is finite, skipping "
                 f"(pass --resume 0 to force a rerun)")
            return

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)
    print(f"jax devices: {jax.devices()}")
    print(f"arm={args.arm} task={args.task} d_model={args.d_model} "
         f"state_size={args.state_size} clip={args.clip} seed={args.seed}")

    np.random.seed(args.seed)
    # train_bench.load_task reads args.task/.downsample/.standardize/
    # .subset/.test_subset/.seq_len/.copy_k/.copy_alpha/.seed directly --
    # reused unmodified, no local reimplementation of data handling.
    x_train, y_train, x_test, y_test, meta = tb.load_task(args)
    n_classes, one_hot, seq2seq = (meta["n_classes"], meta["one_hot"],
                                   meta["seq2seq"])
    x_train_j = tb.prep_x(x_train, one_hot)
    y_train_j = jnp.asarray(y_train)
    x_test_j = tb.prep_x(x_test[:args.eval_subset], one_hot)
    y_test_j = jnp.asarray(y_test[:args.eval_subset])

    model = build_model(args, n_classes, seq2seq)
    key = jax.random.PRNGKey(args.seed)
    dummy = x_train_j[:2]
    variables = model.init({"params": key, "dropout": key}, dummy,
                           train=False)
    params = variables["params"]

    opt = optax.chain(
        optax.clip_by_global_norm(args.clip) if args.clip > 0
        else optax.identity(),
        optax.adam(args.lr))
    opt_state = opt.init(params)

    def loss_from_params(p, x, y, dkey):
        logits = model.apply({"params": p}, x, train=True,
                             rngs={"dropout": dkey})
        return tb.loss_fn(logits, y)

    @jax.jit
    def train_step(p, opt_state, x, y, dkey):
        loss, grads = jax.value_and_grad(loss_from_params)(p, x, y, dkey)
        nrm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss, nrm

    n_train = x_train_j.shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = args.steps if args.steps > 0 else (
        steps_per_epoch * args.epochs)
    diag_steps = set(int(s) for s in
                     args.diagnostic_checkpoints.split(",") if s != "")

    rng = np.random.RandomState(args.seed)
    dkey = key
    history = []
    diagnostics = []
    finite = True
    t0 = time.time()
    peak_mem = 0

    for step in range(1, total_steps + 1):
        idx = rng.randint(0, n_train, size=args.batch_size)
        xb, yb = x_train_j[idx], y_train_j[idx]
        dkey, step_key = jax.random.split(dkey)
        params, opt_state, loss, gnorm = train_step(params, opt_state, xb,
                                                     yb, step_key)
        loss_v = float(loss)
        if not np.isfinite(loss_v):
            finite = False
            print(f"step {step}: NON-FINITE loss, stopping")
            break
        if step % max(1, steps_per_epoch // 4) == 0 or step == total_steps:
            logits = model.apply({"params": params}, x_test_j, train=False)
            val_loss = float(tb.loss_fn(logits, y_test_j))
            val_acc = float(tb.acc_fn(logits, y_test_j))
            history.append(dict(step=step, train_loss=loss_v,
                               val_loss=val_loss, val_acc=val_acc,
                               grad_norm=float(gnorm)))
            print(f"step {step}/{total_steps}: train_loss={loss_v:.4f} "
                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if step in diag_steps:
            diagnostics.append(dict(
                step=step, note="gradient-cosine-vs-BPTT probe stub -- "
                "see README 'Metrics for the actual S5 runs': wire "
                "experiments/s5_ccm/exactness_check.py's per-block "
                "comparison against a frozen batch at this checkpoint's "
                "params before the primary L=2 experiment is launched "
                "for real; not computed automatically inside this loop "
                "to avoid adding a BPTT call to every logged step by "
                "default."))
        mem = tb.peak_device_memory()
        if mem:
            peak_mem = max(peak_mem, mem)

    wall = time.time() - t0
    out = dict(arm=args.arm, task=args.task, config=vars(args),
              finite=finite, steps_run=step if "step" in dir() else 0,
              history=history, diagnostics=diagnostics,
              final_train_loss=history[-1]["train_loss"] if history
              else None,
              final_val_loss=history[-1]["val_loss"] if history else None,
              best_val_loss=(min(h["val_loss"] for h in history)
                            if history else None),
              wall_time_sec=wall,
              steps_per_sec=(step / wall if wall > 0 else 0),
              peak_device_memory_bytes=peak_mem,
              provenance=tb.provenance())
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
