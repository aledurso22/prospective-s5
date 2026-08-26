"""REGISTERED EXPERIMENT — prospective solver metric for a stiff S5-DEQ.

This is the Phase-1 test the PESM nulls and the prospective-s5 no-gos both
point to. It is designed to be un-killable by the mechanisms that killed the
prospective-DEQ/PID program (Status v3, 18 Aug 2026):

  * NO history, continuation, or warm starts across problems: every forward
    pass solves a FRESH energy from a deterministic amortizer (the explicit
    first-order rollout of the current input). Fixed init => fixed branch;
    branch semantics are settled by construction, not discovered.
  * The energies are near-convex (strongly convex chain + bounded tanh
    anchor), unlike the multistable DEQ-Transformer where branch switching
    was discovered.
  * Solver costs follow the uniform NFE contract of prospective-deq/
    solvers.py: 1 gradient evaluation per step, every arm; the Newton arm's
    tridiagonal solves (3 scans each) are logged as structure, not hidden.

MODEL (identical across arms — only the solver differs):
  Whole-sequence equilibrium layer per channel d (real):
      E(s) = sum_t 1/2 (s_t - lam_d s_{t-1} - b_t)^2 + beta/2 (tanh s_t - u_t)^2
  with b = B x, u = tanh(W_u x), lam = sigmoid(lam_raw) initialized STIFF,
  linspace(0.99, 0.999) over channels (kappa ~ 4e4..4e6 — the regime where
  the task actually needs |lam| -> 1 and the metric has something to fix).
  s* = argmin E, computed by K = 8 unrolled solver steps from the amortizer
  rollout s_t = lam s_{t-1} + b_t. Backprop through the unrolled steps.
  y = C s + D x. Blocks: pre-norm layer + GLU FFN, residual. L = 2, D = 64.

ARMS (solver only):
  newton   prospective: damped Gauss-Newton, one tridiagonal solve per step
           (3 associative scans). gamma = 1.
  gd       Euclidean control: s <- s - eta grad E, eta = 0.25. gamma = 0.
  broyden  the DEQ field's baseline: limited-memory Broyden (m = 5) on
           grad E = 0, Woodbury step (ported from prospective-deq).

TASK: delayed copy. Vocab 12: 1..8 signal symbols, 0 filler, 9 = GO marker.
Layout: [8 signal][G fillers][GO][8 answer slots]; loss = cross-entropy on
the 8 answer positions (predict the signal in order). G in {64, 256}: at
G=256 only the stiffest modes carry the signal (0.99^256 ~ 0.08, 0.999^256 ~
0.77) — stiffness is task-relevant. G=64 is the predeclared control cell.

REGISTERED GRID: arms {newton, gd, broyden} x G {64, 256} x seeds {0,1,2}
= 18 runs. Config: D=64, L=2, K=8, Adam lr 1e-3 constant, batch 32,
1500 steps, eval every 250 (512 fresh samples), final eval 2048.
Same init per (arm, seed) pair.

REGISTERED GATES (fixed 2026-08-23, before any run):
  P1 stability: collapse := any non-finite train loss, OR mean train loss
     of the final 100 steps > 1.5x the running minimum. WIN: newton has
     strictly fewer collapses than gd across the 6 cells.
  P2 accuracy at G=256: WIN iff newton >= gd on >= 2 of 3 seeds AND the
     median paired gap > +0.02. (broyden reported, not gated.)
  P3 solver quality: relative energy residual after the fixed K=8 steps,
     mean over layers, at train step 500, G=256: WIN iff newton <= 0.1 x gd
     on >= 2 of 3 seeds.
  CONTROL: at G=64 the arms must tie (median |paired gap| < 0.05); a
     separation there falsifies the stiffness mechanism story.
  KILL RULE: no P1 win AND no P2 win => the prospective-metric sequence-
     model direction closes as a mechanisms/negative result; no retuning,
     regridding, or re-baselining is licensed by a null.

Usage:
  python registered_stiff_deq.py --smoke          # tiny code check (G=16)
  python registered_stiff_deq.py --arm newton --g 256 --seed 0   # one cell
  python registered_stiff_deq.py --grid           # all 18 registered runs
  python registered_stiff_deq.py --summarize      # evaluate the gates
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

# ---------------------------------------------------------------------------
# Registered hyperparameters (do not tune post hoc — see docstring)
# ---------------------------------------------------------------------------

D_MODEL = 64
N_LAYERS = 2
K_SOLVER = 8
ETA = 0.25
BROYDEN_M = 5
LAM_INIT = (0.99, 0.999)     # stiff on purpose
BETA_INIT = 0.5
LR = 1e-3
BATCH = 32
STEPS = 1500
EVAL_EVERY = 250
EVAL_SAMPLES = 512
FINAL_EVAL_SAMPLES = 2048
ARMS = ["newton", "gd", "broyden"]
G_GRID = [64, 256]
SEEDS = [0, 1, 2]
N_SIGNAL = 8
VOCAB = 12                 # 0 filler, 1..8 signal, 9 = GO marker
GO = 9

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "registered_stiff_deq")


# ---------------------------------------------------------------------------
# Hermitian tridiagonal solve via 3 associative scans
# (ported from pesm_s5_spectrum.py, same branch — copied to avoid that
# module's float64 config side effect; real inputs are the use case here)
# ---------------------------------------------------------------------------

def _mobius_op(Mi, Mj):
    N = jnp.einsum("...ij,...jk->...ik", Mj, Mi)
    n = jnp.max(jnp.abs(N), axis=(-2, -1), keepdims=True)
    return N / jnp.where(n > 0, n, 1.0)


def _aff_op(qi, qj):
    ai, bi = qi
    aj, bj = qj
    return aj * ai, aj * bi + bj


def tridiag_solve(diag, sub, g):
    """Solve H x = g along axis 0: H Hermitian tridiagonal SPD, real diag,
    (possibly complex) sub-diagonal; sub[0] ignored."""
    piv2 = -(jnp.abs(sub) ** 2)
    M = jnp.stack([jnp.stack([diag, piv2], axis=-1),
                   jnp.stack([jnp.ones_like(diag), jnp.zeros_like(diag)],
                             axis=-1)], axis=-2)
    Nm = jax.lax.associative_scan(_mobius_op, M, axis=0)
    d = Nm[..., 0, 0] / Nm[..., 1, 0]
    d_prev = jnp.concatenate([jnp.ones_like(d[:1]), d[:-1]], axis=0)
    _, v = jax.lax.associative_scan(_aff_op, (-sub / d_prev, g), axis=0)
    y = v / d
    w_next = jnp.concatenate(
        [jnp.conj(sub[1:]), jnp.zeros_like(sub[:1])], axis=0)
    _, x_rev = jax.lax.associative_scan(
        _aff_op, (jnp.flip(-w_next / d, axis=0), jnp.flip(y, axis=0)), axis=0)
    return jnp.flip(x_rev, axis=0)


# ---------------------------------------------------------------------------
# Delayed-copy data
# ---------------------------------------------------------------------------

def make_batch(key, batch_size: int, gap: int):
    T = N_SIGNAL + gap + 1 + N_SIGNAL
    sig = jax.random.randint(key, (batch_size, N_SIGNAL), 1, N_SIGNAL + 1)
    tokens = np.zeros((batch_size, T), np.int32)
    tokens[:, :N_SIGNAL] = np.asarray(sig)
    tokens[:, N_SIGNAL + gap] = GO
    return jnp.asarray(tokens), sig


# ---------------------------------------------------------------------------
# Energy and the three solver arms
# ---------------------------------------------------------------------------

def energy_grad(s, b, u, lam, beta):
    """grad E at trajectory s. s, b, u: (T, ...); lam broadcastable."""
    s_prev = jnp.concatenate([jnp.zeros_like(s[:1]), s[:-1]], axis=0)
    r = s - lam * s_prev - b
    r_next = jnp.concatenate([r[1:], jnp.zeros_like(r[:1])], axis=0)
    z = jnp.tanh(s)
    return (r - lam * r_next) + beta * (z - u) * (1 - z ** 2)


def newton_solve(b, u, lam, beta, s0, K):
    """Prospective arm: K damped Gauss-Newton steps (3 scans each)."""
    def step(s, _):
        g = energy_grad(s, b, u, lam, beta)
        w = (1 - jnp.tanh(s) ** 2) ** 2
        dg = 1 + lam ** 2 + beta * w
        dg = dg.at[-1].set(1 + beta * w[-1])       # last row: no r_{t+1}
        sub = jnp.full_like(s, -lam).at[0].set(0.0)
        return s - tridiag_solve(dg, sub, g), None
    s, _ = jax.lax.scan(step, s0, None, length=K)
    return s


def gd_solve(b, u, lam, beta, s0, K):
    """Euclidean control arm: K gradient steps, fixed eta."""
    def step(s, _):
        return s - ETA * energy_grad(s, b, u, lam, beta), None
    s, _ = jax.lax.scan(step, s0, None, length=K)
    return s


def broyden_solve(b, u, lam, beta, s0, K):
    """Field baseline: limited-memory Broyden on grad E = 0, per sample.
    Ported from prospective-deq/solvers.py (Woodbury step, secant update)."""
    def one_sample(b_s, u_s, s0_s):
        # b_s, u_s, s0_s: (T, D); the solver works on the flattened state
        shape = s0_s.shape
        gfun = lambda sf: energy_grad(sf.reshape(shape), b_s, u_s,
                                      lam, beta).reshape(-1)
        Df = s0_s.size

        def body(carry, _):
            s, g_prev, ds, U, V = carry
            g = gfun(s)
            dg = g - g_prev
            B_ds = ds + U.T @ (V @ ds)
            denom = jnp.dot(ds, ds)
            skip = denom <= 1e-24
            u_ = jnp.where(skip, jnp.zeros_like(ds),
                           (dg - B_ds) / jnp.where(skip, 1.0, denom))
            U = jnp.roll(U, -1, axis=0).at[-1].set(u_)
            V = jnp.roll(V, -1, axis=0).at[-1].set(ds)
            Mm = jnp.eye(BROYDEN_M, dtype=g.dtype) + V @ U.T
            p = g - U.T @ jnp.linalg.solve(Mm, V @ g)
            s_new = s - p
            return (s_new, g, s_new - s, U, V), None

        carry0 = (s0_s.reshape(-1), jnp.zeros(Df, s0_s.dtype),
                  jnp.zeros(Df, s0_s.dtype),
                  jnp.zeros((BROYDEN_M, Df), s0_s.dtype),
                  jnp.zeros((BROYDEN_M, Df), s0_s.dtype))
        (sK, _, _, _, _), _ = jax.lax.scan(body, carry0, None, length=K)
        return sK.reshape(s0_s.shape)

    # in/out over the batch axis of (T, B, D) arrays
    return jax.vmap(one_sample, in_axes=(1, 1, 1), out_axes=1)(b, u, s0)


def solve(arm, b, u, lam, beta, s0, K):
    if arm == "newton":
        return newton_solve(b, u, lam, beta, s0, K)
    if arm == "gd":
        return gd_solve(b, u, lam, beta, s0, K)
    if arm == "broyden":
        return broyden_solve(b, u, lam, beta, s0, K)
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StiffEquilibriumLayer(nn.Module):
    """Whole-sequence stiff equilibrium layer (see module docstring)."""
    d_model: int
    solver: str
    K: int

    @nn.compact
    def __call__(self, x):
        Bsz, T, D = x.shape
        lam0 = jnp.linspace(LAM_INIT[0], LAM_INIT[1], D)
        lam_raw = self.param("lam_raw",
                             lambda rng, shape: jnp.log(lam0 / (1 - lam0)),
                             (D,))
        lam = jax.nn.sigmoid(lam_raw)
        beta = jnp.exp(self.param("log_beta", nn.initializers.constant(
            float(np.log(BETA_INIT))), (1,)))

        b = nn.Dense(D, name="B")(x)
        u = jnp.tanh(nn.Dense(D, name="Wu")(x))
        bT = jnp.swapaxes(b, 0, 1)                       # (T, B, D)
        uT = jnp.swapaxes(u, 0, 1)

        # amortizer: explicit first-order rollout (the SSM step), 1 scan
        def roll(s_prev, b_t):
            s = lam * s_prev + b_t
            return s, s
        _, s_roll = jax.lax.scan(roll, jnp.zeros((Bsz, D), x.dtype), bT)
        s_sol = solve(self.solver, bT, uT, lam, beta, s_roll, self.K)

        # P3 probe: relative residual after the K steps (collected only when
        # intermediates are captured, i.e. at eval)
        g0 = jnp.linalg.norm(energy_grad(s_roll, bT, uT, lam, beta),
                             axis=(0, 2))                  # (B,)
        gK = jnp.linalg.norm(energy_grad(s_sol, bT, uT, lam, beta),
                             axis=(0, 2))
        self.sow("intermediates", "rel_res",
                 jnp.median(gK / jnp.maximum(g0, 1e-30)))
        s = jnp.swapaxes(s_sol, 0, 1)
        return nn.Dense(D, name="C")(s) + nn.Dense(D, name="D")(x)


class Block(nn.Module):
    d_model: int
    solver: str
    K: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = StiffEquilibriumLayer(self.d_model, self.solver, self.K)(y)
        x = x + y
        z = nn.LayerNorm()(x)
        z = nn.Dense(2 * self.d_model)(z)
        a, gate = jnp.split(z, 2, axis=-1)
        z = nn.Dense(self.d_model)(jax.nn.gelu(a) * jax.nn.sigmoid(gate))
        return x + z


class CopyModel(nn.Module):
    solver: str
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    K: int = K_SOLVER

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(VOCAB, self.d_model)(tokens)
        for _ in range(self.n_layers):
            x = Block(self.d_model, self.solver, self.K)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(VOCAB)(x)                        # (B, T, V)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(logits, targets):
    ans = logits[:, -N_SIGNAL:, :]
    return optax.softmax_cross_entropy_with_integer_labels(
        ans, targets).mean()


def accuracy(logits, targets):
    ans = logits[:, -N_SIGNAL:, :]
    return float(jnp.mean(jnp.argmax(ans, -1) == targets))


@jax.jit
def train_step(state, tokens, targets):
    def compute_loss(params):
        return loss_fn(state.apply_fn(params, tokens), targets)
    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


def eval_model(state, tokens, targets):
    """Eval with the P3 residual probe captured (mean over layers)."""
    logits, mods = state.apply_fn(state.params, tokens,
                                  capture_intermediates=True)
    res = []

    def _collect(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "rel_res" and isinstance(v, tuple):
                    res.append(float(v[0]))
                else:
                    _collect(v)

    _collect(mods["intermediates"])
    rel_res = float(np.mean(res)) if res else float("nan")
    return logits, rel_res


def run_cell(arm: str, gap: int, seed: int, steps: int = STEPS,
             smoke: bool = False) -> dict:
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    model = CopyModel(solver=arm,
                      d_model=32 if smoke else D_MODEL,
                      n_layers=1 if smoke else N_LAYERS,
                      K=4 if smoke else K_SOLVER)
    init_tokens, _ = make_batch(key, 4, gap)
    params = model.init(key, init_tokens)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(LR))

    history = []
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        bt, bs = make_batch(jax.random.PRNGKey(int(rng.integers(1 << 31))),
                            BATCH, gap)
        state, loss = train_step(state, bt, bs)
        losses.append(float(loss))
        if step % EVAL_EVERY == 0 or step == steps:
            et, es = make_batch(jax.random.PRNGKey(10_000 + seed),
                                EVAL_SAMPLES, gap)
            logits, rel_res = eval_model(state, et, es)
            history.append(dict(step=step,
                                eval_loss=float(loss_fn(logits, es)),
                                eval_acc=accuracy(logits, es),
                                rel_res=rel_res))
    wall = time.time() - t0

    losses = np.asarray(losses)
    finite = bool(np.all(np.isfinite(losses)))
    final_mean = (float(losses[-100:].mean()) if len(losses) >= 100
                  else float(losses.mean()))
    collapse = (not finite) or (final_mean > 1.5 * float(losses.min()))

    et, es = make_batch(jax.random.PRNGKey(20_000 + seed),
                        FINAL_EVAL_SAMPLES, gap)
    logits, _ = eval_model(state, et, es)

    return dict(arm=arm, gap=gap, seed=seed,
                K=(4 if smoke else K_SOLVER), n_params=n_params,
                wall_time_sec=wall, history=history,
                final_acc=accuracy(logits, es),
                final_loss=float(loss_fn(logits, es)),
                final100_train_loss=final_mean,
                min_train_loss=float(losses.min()), finite=finite,
                collapse=collapse)


# ---------------------------------------------------------------------------
# Grid driver + registered gate evaluation
# ---------------------------------------------------------------------------

def _path(arm, gap, seed):
    return os.path.join(RESULTS_DIR, f"{arm}_G{gap}_s{seed}.json")


def run_grid():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for gap in G_GRID:
        for arm in ARMS:
            for seed in SEEDS:
                out = run_cell(arm, gap, seed)
                with open(_path(arm, gap, seed), "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[done] {arm} G={gap} seed={seed}  "
                      f"acc={out['final_acc']:.4f}  collapse={out['collapse']}",
                      flush=True)


def _res_at(runs, arm, gap, seed, step=500):
    for h in runs[(arm, gap, seed)]["history"]:
        if h["step"] == step:
            return h["rel_res"]
    return float("nan")


def summarize():
    runs = {}
    for gap in G_GRID:
        for arm in ARMS:
            for seed in SEEDS:
                with open(_path(arm, gap, seed)) as f:
                    runs[(arm, gap, seed)] = json.load(f)

    print("=" * 78)
    print("REGISTERED GATES — evaluation")
    print("=" * 78)
    coll = {arm: sum(runs[(arm, g, s)]["collapse"] for g in G_GRID
                     for s in SEEDS) for arm in ARMS}
    p1 = coll["newton"] < coll["gd"]
    print(f"P1 stability: collapses newton={coll['newton']} gd={coll['gd']} "
          f"broyden={coll['broyden']}  ->  {'WIN' if p1 else 'NO WIN'}")

    gaps = [runs[("newton", 256, s)]["final_acc"]
            - runs[("gd", 256, s)]["final_acc"] for s in SEEDS]
    wins = sum(g > 0 for g in gaps)
    med = float(np.median(gaps))
    p2 = wins >= 2 and med > 0.02
    print(f"P2 accuracy G=256: paired gaps (newton-gd) = "
          f"{['%+.4f' % g for g in gaps]}  median {med:+.4f}  ->  "
          f"{'WIN' if p2 else 'NO WIN'}")
    for arm in ARMS:
        accs = [runs[(arm, 256, s)]["final_acc"] for s in SEEDS]
        print(f"   {arm:<8s} G=256 acc per seed: "
              f"{['%.4f' % a for a in accs]}")

    p3_wins = 0
    for s in SEEDS:
        rn = _res_at(runs, "newton", 256, s)
        rg = _res_at(runs, "gd", 256, s)
        p3_wins += int(rn <= 0.1 * rg)
        print(f"P3 seed {s}: rel_res@500 newton {rn:.3e} vs gd {rg:.3e}")
    p3 = p3_wins >= 2
    print(f"P3 solver quality: newton <= 0.1x gd on {p3_wins}/3 seeds  ->  "
          f"{'WIN' if p3 else 'NO WIN'}")

    gaps64 = [runs[("newton", 64, s)]["final_acc"]
              - runs[("gd", 64, s)]["final_acc"] for s in SEEDS]
    med64 = float(np.median(np.abs(gaps64)))
    control = med64 < 0.05
    print(f"CONTROL G=64: |paired gaps| median {med64:.4f}  ->  "
          f"{'OK (tie)' if control else 'FALSIFIED (separation)'}")
    print("-" * 78)
    kill = (not p1) and (not p2)
    print("VERDICT: " + ("KILL — close as mechanisms/negative result"
                           if kill else
                           "CONTINUE — prospective metric helps in the "
                           "stiff regime"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--g", type=int, choices=G_GRID)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    args = ap.parse_args()

    if args.smoke:
        print("SMOKE TEST (not part of the registered grid)")
        for arm in ARMS:
            out = run_cell(arm, 16, 0, steps=60, smoke=True)
            print(f"  {arm:<8s} loss min {out['min_train_loss']:.4f}  "
                  f"final acc {out['final_acc']:.4f}  finite {out['finite']}"
                  f"  rel_res@60 {out['history'][-1]['rel_res']:.2e}")
        print("smoke OK")
        return
    if args.grid:
        run_grid()
        return
    if args.summarize:
        summarize()
        return
    if args.arm and args.g is not None and args.seed is not None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = run_cell(args.arm, args.g, args.seed)
        with open(_path(args.arm, args.g, args.seed), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[done] {args.arm} G={args.g} seed={args.seed}  "
              f"acc={out['final_acc']:.4f}  collapse={out['collapse']}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
