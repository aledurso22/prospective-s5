"""REGISTERED EXPERIMENT 2 — coupled-equilibrium bidirectional SSM.

The last open slot. Every bidirectional SSM in the literature runs TWO
INDEPENDENT scans (forward + backward) and fuses them at the readout. A
genuinely COUPLED bidirectional state — s_t depending on both s_{t-1} and
s_{t+1} — is a two-point boundary problem with no scan form; it needs an
iterative solve, which is what the prospective metric provides in O(1)
steps on the stiff spectrum (pesm_s5_spectrum.py). Classical anchor: for
beta=0 the coupled equilibrium IS the linear-Gaussian smoother (RTS
lineage); beta>0 makes it a non-Gaussian implicit layer with no closed
form. This experiment tests whether the coupled model CLASS buys anything
over the decoupled class — not the solver (that question was killed by
registered_stiff_deq.py); the solver here is just the enabler.

ARMS (same block scaffold, same budget; only the temporal structure differs):
  causal    one forward S5 rollout per layer (ssm/baseline_s5 S5SSM,
            complex HiPPO init). Cannot see the future at all.
  decoupled the field's bidirectional SSM: two independent S5SSM scans
            (forward + backward on the flipped input), fused by addition
            at the layer output. ~2x layer params — a conservative bias
            AGAINST the coupled arm, predeclared as such.
  coupled   the whole-sequence equilibrium layer (from
            registered_stiff_deq.py): s* = argmin of the stiff chain
            energy + tanh anchor, newton solver, K=8, amortizer = the
            explicit rollout. Fresh deterministic init per forward.

TASKS:
  classify  sequential MNIST (T=784), mean-pool readout, 10 classes.
            Coupling is not structurally required here — the no-harm cell.
  infill    masked-segment infilling on MNIST rows: timesteps [342, 442)
            zeroed (central ~4 rows), regress the masked pixels (MSE).
            Joint left-right consistency is the point — the discriminator.

REGISTERED GRID: arms {causal, decoupled, coupled} x tasks {classify,
infill} x seeds {0,1,2} = 18 runs. Config: D=64, L=2, N=64, K=8, Adam lr
1e-3 constant, batch 32, 20k train subset; 3000 steps (classify), 1500
steps (infill); final eval on the full 10k test set (classify) / 2048 test
samples (infill). Same init per (arm, seed) pair within each task.

REGISTERED GATES (fixed 2026-08-23, before any run):
  G1 (the claim): on infill, median-over-seeds MSE of coupled <= 0.9x
     median MSE of decoupled, AND coupled < decoupled on >= 2/3 seeds.
  G2 (no harm): on classify, |acc_coupled - acc_decoupled| <= 0.01
     (median over seeds). A win is welcome but not required.
  G3 (task sanity): causal must be the WORST arm on infill on >= 2/3
     seeds — otherwise the task does not test bidirectionality and the
     experiment is inconclusive by design, not negative.
  KILL RULE: no G1 => the coupled-equilibrium bidirectional model class is
     dead and the program closes as a mechanisms/negative-results paper.
     No retuning or re-gridding is licensed by a null.

Usage:
  python registered_bissm.py --smoke
  python registered_bissm.py --arm coupled --task infill --seed 0
  python registered_bissm.py --grid
  python registered_bissm.py --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from train import load_mnist
from ssm.baseline_s5.layer import S5SSM
from archive.solver.registered_stiff_deq import StiffEquilibriumLayer

# ---------------------------------------------------------------------------
# Registered hyperparameters (do not tune post hoc — see docstring)
# ---------------------------------------------------------------------------

D_MODEL = 64
N_LAYERS = 2
N_STATE = 64
K_SOLVER = 8
LR = 1e-3
BATCH = 32
SUBSET = 20000
STEPS = {"classify": 3000, "infill": 1500}
ARMS = ["causal", "decoupled", "coupled"]
TASKS = ["classify", "infill"]
SEEDS = [0, 1, 2]
MASK_T0, MASK_LEN = 342, 100          # central ~4 MNIST rows
EVAL_INFILL = 2048

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "registered_bissm")


# ---------------------------------------------------------------------------
# Layers (the only place the arms differ)
# ---------------------------------------------------------------------------

class BiS5(nn.Module):
    """Decoupled bidirectional S5: two independent scans, additive fusion."""
    state_size: int
    d_model: int

    @nn.compact
    def __call__(self, u):
        y_f = S5SSM(state_size=self.state_size, d_model=self.d_model,
                    name="fwd")(u)
        y_b = S5SSM(state_size=self.state_size, d_model=self.d_model,
                    name="bwd")(jnp.flip(u, 1))
        return y_f + jnp.flip(y_b, 1)


def make_layer(arm: str, d_model: int):
    if arm == "causal":
        return S5SSM(state_size=N_STATE, d_model=d_model)
    if arm == "decoupled":
        return BiS5(state_size=N_STATE, d_model=d_model)
    if arm == "coupled":
        return StiffEquilibriumLayer(d_model, "newton", K_SOLVER)
    raise ValueError(arm)


class Block(nn.Module):
    """Pre-norm: LayerNorm -> temporal layer -> residual; LN -> GLU -> res."""
    d_model: int
    arm: str

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = make_layer(self.arm, self.d_model)(y)
        x = x + y
        z = nn.LayerNorm()(x)
        z = nn.Dense(2 * self.d_model)(z)
        a, gate = jnp.split(z, 2, axis=-1)
        z = nn.Dense(self.d_model)(jax.nn.gelu(a) * jax.nn.sigmoid(gate))
        return x + z


class SeqModel(nn.Module):
    """Shared trunk; head depends on the task."""
    arm: str
    task: str
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS

    @nn.compact
    def __call__(self, x):                       # x: (B, T) real pixels
        x = nn.Dense(self.d_model)(x[..., None])
        for _ in range(self.n_layers):
            x = Block(self.d_model, self.arm)(x)
        if self.task == "classify":
            x = jnp.mean(x, axis=1)
            x = nn.LayerNorm()(x)
            return nn.Dense(10)(x)               # (B, 10) logits
        return nn.Dense(1)(x)[..., 0]            # (B, T) pixel regression


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def loss_acc(arm_task, params, apply_fn, x, y):
    logits_or_pred = apply_fn(params, x)
    if arm_task == "classify":
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_or_pred, y).mean()
        acc = jnp.mean(jnp.argmax(logits_or_pred, -1) == y)
        return loss, acc
    pred = logits_or_pred[:, MASK_T0:MASK_T0 + MASK_LEN]
    mse = jnp.mean((pred - y) ** 2)
    return mse, mse                                # "acc" slot carries MSE


def run_cell(arm: str, task: str, seed: int, smoke: bool = False) -> dict:
    xtr, ytr, xte, yte = load_mnist()
    rng = np.random.default_rng(seed)
    if SUBSET:
        idx = rng.choice(len(xtr), SUBSET, replace=False)
        xtr, ytr = xtr[idx], ytr[idx]

    model = SeqModel(arm=arm, task=task,
                     d_model=32 if smoke else D_MODEL,
                     n_layers=1 if smoke else N_LAYERS)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.asarray(xtr[:4]))
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(LR))

    steps = 60 if smoke else STEPS[task]
    batch = 16 if smoke else BATCH

    @jax.jit
    def train_step(state, x, y):
        def compute_loss(params):
            return loss_acc(task, params, state.apply_fn, x, y)[0]
        loss, grads = jax.value_and_grad(compute_loss)(state.params)
        return state.apply_gradients(grads=grads), loss

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(xtr), size=batch)
        xb = jnp.asarray(xtr[idx])
        if task == "infill":
            yb_full = xb[:, MASK_T0:MASK_T0 + MASK_LEN]
            xb = xb.at[:, MASK_T0:MASK_T0 + MASK_LEN].set(0.0)
            state, loss = train_step(state, xb, jnp.asarray(yb_full))
        else:
            state, loss = train_step(state, xb, jnp.asarray(ytr[idx]))
        losses.append(float(loss))

    wall = time.time() - t0
    losses = np.asarray(losses)
    finite = bool(np.all(np.isfinite(losses)))

    if task == "classify":
        xe, ye = jnp.asarray(xte), jnp.asarray(yte)
        accs, losses_ev = [], []
        for i in range(0, len(xte), 512):
            l, a = loss_acc(task, state.params, state.apply_fn,
                            xe[i:i + 512], ye[i:i + 512])
            accs.append(float(a) * len(xe[i:i + 512]))
            losses_ev.append(float(l) * len(xe[i:i + 512]))
        final_metric = sum(accs) / len(xte)
        final_loss = sum(losses_ev) / len(xte)
    else:
        xe = jnp.asarray(xte[:EVAL_INFILL])
        yb = xe[:, MASK_T0:MASK_T0 + MASK_LEN]
        xe = xe.at[:, MASK_T0:MASK_T0 + MASK_LEN].set(0.0)
        final_loss, final_metric = loss_acc(task, state.params,
                                            state.apply_fn, xe, yb)
        final_metric, final_loss = float(final_metric), float(final_loss)

    return dict(arm=arm, task=task, seed=seed, n_params=n_params,
                wall_time_sec=wall,
                final_metric=final_metric,      # acc (classify) / MSE (infill)
                final_loss=final_loss,
                min_train_loss=float(losses.min()),
                final100_train_loss=float(losses[-100:].mean()),
                finite=finite)


# ---------------------------------------------------------------------------
# Grid driver + registered gate evaluation
# ---------------------------------------------------------------------------

def _path(arm, task, seed):
    return os.path.join(RESULTS_DIR, f"{arm}_{task}_s{seed}.json")


def run_grid():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for task in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                out = run_cell(arm, task, seed)
                with open(_path(arm, task, seed), "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[done] {arm} {task} seed={seed}  "
                      f"metric={out['final_metric']:.4f}", flush=True)


def summarize():
    runs = {}
    for task in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                with open(_path(arm, task, seed)) as f:
                    runs[(arm, task, seed)] = json.load(f)

    print("=" * 78)
    print("REGISTERED GATES — evaluation (experiment 2: coupled BiSSM)")
    print("=" * 78)
    for arm in ARMS:
        for task in TASKS:
            ms = [runs[(arm, task, s)]["final_metric"] for s in SEEDS]
            print(f"  {arm:<10s} {task:<9s}: "
                  f"{['%.4f' % m for m in ms]}  (params "
                  f"{runs[(arm, task, 0)]['n_params']:,})")

    # G1: infill win — coupled median MSE <= 0.9 * decoupled median
    c = [runs[("coupled", "infill", s)]["final_metric"] for s in SEEDS]
    d = [runs[("decoupled", "infill", s)]["final_metric"] for s in SEEDS]
    pair_wins = sum(ci < di for ci, di in zip(c, d))
    ratio = float(np.median(c) / np.median(d))
    g1 = ratio <= 0.9 and pair_wins >= 2
    print(f"\nG1 infill: coupled/decoupled median MSE ratio = {ratio:.3f} "
          f"(need <= 0.90), paired wins {pair_wins}/3  ->  "
          f"{'WIN' if g1 else 'NO WIN'}")

    # G2: no harm on classify
    ca = [runs[("coupled", "classify", s)]["final_metric"] for s in SEEDS]
    da = [runs[("decoupled", "classify", s)]["final_metric"] for s in SEEDS]
    gap = abs(float(np.median(ca)) - float(np.median(da)))
    g2 = gap <= 0.01
    print(f"G2 classify: |median acc gap| coupled vs decoupled = {gap:.4f} "
          f"(need <= 0.01)  ->  {'OK' if g2 else 'HARM'}")
    caa = [runs[("causal", "classify", s)]["final_metric"] for s in SEEDS]
    print(f"   (context: causal classify median {float(np.median(caa)):.4f})")

    # G3: task sanity — causal worst on infill >= 2/3 seeds
    a = [runs[("causal", "infill", s)]["final_metric"] for s in SEEDS]
    worst = sum(ai >= max(ci, di)
                for ai, ci, di in zip(a, c, d))
    g3 = worst >= 2
    print(f"G3 sanity: causal is the worst infill arm on {worst}/3 seeds "
          f"->  {'OK' if g3 else 'INCONCLUSIVE-BY-DESIGN'}")

    print("-" * 78)
    print("VERDICT: " + ("WIN — coupling buys something where joint "
                           "conditioning is required"
                           if g1 else
                           "KILL — coupled-equilibrium model class dead; "
                           "close as mechanisms/negative paper"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    args = ap.parse_args()

    if args.smoke:
        print("SMOKE TEST (not part of the registered grid)")
        for task in TASKS:
            for arm in ARMS:
                out = run_cell(arm, task, 0, smoke=True)
                print(f"  {arm:<10s} {task:<9s} metric "
                      f"{out['final_metric']:.4f}  finite {out['finite']}")
        print("smoke OK")
        return
    if args.grid:
        run_grid()
        return
    if args.summarize:
        summarize()
        return
    if args.arm and args.task and args.seed is not None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = run_cell(args.arm, args.task, args.seed)
        with open(_path(args.arm, args.task, args.seed), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[done] {args.arm} {args.task} seed={args.seed}  "
              f"metric={out['final_metric']:.4f}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
