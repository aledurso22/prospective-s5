"""Phase 2A -- controlled expressivity/credit frontier.

Scientific question: what recurrent computations can different
exact-credit structures represent at comparable model/credit budgets?
NOT another exactness phase -- B29-B34's correctness work is frozen and
reused here unmodified.

Four architecture families:
1. RTU (independent complex/2x2 modes) -- b28_rtu_faithful.py's cell.
2. B34 jet-algebra (R[eps]/(eps^r)) -- b34a, reused unmodified.
3. Bounded-interface/flag (T=V(+)U) -- b31a's architecture, reused
   unmodified (R_U,D_U fixed; R_V,K,B_V,C_V,C_U,Phi trainable).
4. Dense BPTT oracle -- plain tanh RNN, expressivity reference only.

COMMON EXTERNAL INPUT INTERFACE (fixed after an earlier version wrongly
gave the flag architecture only 1 scalar of external input while others
got 4): every sequence is ONE scalar exogenous input x_t per step. Each
architecture embeds it into whatever its own (unmodified) internals
expect: flag consumes x_t directly (its native convention); RTU/dense
consume it via a (hidden,1) input-weight column; B34's frozen generator
hard-expects a 4-vector, so x_t is embedded as (x_t,0,0,0) -- extra
zero-padding carries no additional information, so no architecture
receives more external information than another.

POSITIVE-CONTROL VALIDITY: teacher and student share the exact same
fixed structural substrate for the three matched pairs (RTU teacher A /
RTU student; B34 teacher B / B34 student -- SAME frozen coefficient
generator, not an independently-redrawn one, so teacher B is guaranteed
to lie in the student's hypothesis class; flag teacher D / flag student
-- SAME FLAG_CONSTS). Only the trainable parameters differ.

OPTIMIZATION: Adam (not raw SGD) with a small per-(architecture,teacher)
LR grid (3 values) and a fixed number of seeds; the run with the best
FINITE validation NMSE is selected and its LR recorded. A diverged run
contributes to the divergence fraction, never to an NMSE average.
Divergence is reported as an "optimization/dynamical instability"
outcome, explicitly NOT interpreted as evidence of representational
impossibility.

Run: python -m credit_memory.p2a_expressivity_credit_frontier
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b28_rtu_faithful import make_rtu_params, rtu_g_phi_norm
from credit_memory.b31a_joint_family_correctness import (
    make_fixed_consts, make_theta as make_flag_theta, full_step_state as flag_full_step_state,
    D_V_DIM as FLAG_V_DIM, R_DIM as FLAG_R_DIM,
)
from credit_memory.b34a_jet_algebra_correctness import (
    make_gen_params as jet_make_gen_params, h_step as jet_h_step,
)

R_STATE = 64          # matched state dimension target, View 1
T_SEQ = 128
N_TRAIN = 80  # doubled from an initial 40 after the RTU positive-control
              # sanity check needed more steps to clearly pass (0.33->0.18
              # NMSE); applied uniformly to ALL architectures/teachers for
              # a fair, non-per-combo-tuned budget, not just RTU.
N_VAL = 10
SEEDS = (0, 1)         # per (arch, teacher, LR) grid cell
LR_GRID_DEFAULT = (0.003, 0.01, 0.03)  # architecture-specific grids set below

RTU_HIDDEN = R_STATE // 2       # 32 -> total (real,imag) state = 64
DENSE_HIDDEN = R_STATE          # 64
Y_DIM = 8

FLAG_CONSTS = make_fixed_consts()  # shared R_U,D_U -- fixed, across student AND teacher D


# ---------------------------------------------------------------------
# Generic BPTT rollout/loss.
# ---------------------------------------------------------------------
def make_rollout(step_fn):
    def rollout(h0, params, xs):
        def body(h, x_t):
            h_next = step_fn(h, params, x_t)
            return h_next, h_next
        _, Hs = jax.lax.scan(body, h0, xs)
        return Hs
    return rollout


def make_loss_mse(rollout_fn):
    def loss(params, h0, xs, targets, W):
        Hs = rollout_fn(h0, params, xs)
        Ys = Hs @ W.T
        return jnp.mean(0.5 * jnp.sum((Ys - targets) ** 2, axis=1))
    return loss


# ---------------------------------------------------------------------
# 1. RTU (independent modes). Input embedding: x_t (scalar) -> (1,)
#    vector via a (hidden,1) input-weight column -- exactly one scalar
#    of external information, matching the shared interface.
# ---------------------------------------------------------------------
def rtu_make_params(seed, hidden_dim=RTU_HIDDEN):
    rng = np.random.RandomState(seed)
    return make_rtu_params(rng, hidden_dim, 1)


def rtu_step(h, params, x_t, hidden_dim=RTU_HIDDEN):
    real, imag = h[:hidden_dim], h[hidden_dim:]
    g, phi_, norm, _ = rtu_g_phi_norm(params["nu_log"], params["theta_log"])
    xv = jnp.atleast_1d(x_t)
    u_real = params["B_real"] @ xv
    u_imag = params["B_imag"] @ xv
    pre_real = g * real - phi_ * imag + norm * u_real
    pre_imag = g * imag + phi_ * real + norm * u_imag
    return jnp.concatenate([jnp.tanh(pre_real), jnp.tanh(pre_imag)])


def rtu_param_count(params):
    return int(sum(np.prod(v.shape) for v in params.values()))


def rtu_credit_scalars(hidden_dim=RTU_HIDDEN):
    """RTU is fully independent per-unit (diagonal role for every
    family). Exact online credit per family: shape (2, hidden_dim,
    *own_per_unit_param_shape). Established B28 accounting, reused."""
    reduced = 4 * hidden_dim + 4 * hidden_dim * 1
    full = (2 * hidden_dim) * (2 * hidden_dim + 2 * hidden_dim * 1)
    return dict(reduced=int(reduced), full=int(full), ratio=full / reduced)


# ---------------------------------------------------------------------
# 2. B34 jet-algebra (reused unmodified from b34a). theta only
#    trainable. Input embedding: x_t -> (x_t,0,0,0) (b34a's frozen
#    generator hard-expects a 4-vector; zero-padding adds no info).
# ---------------------------------------------------------------------
def jet_make_theta(seed, r=R_STATE):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(r) * 0.2)


def make_jet_step(gen_params, r=R_STATE):
    def step(h, theta, x_t):
        x4 = jnp.stack([x_t, 0.0, 0.0, 0.0])
        return jet_h_step(h, theta, x4, gen_params, r)
    return step


def jet_param_count(theta):
    return int(theta.shape[0])


def jet_credit_scalars(r=R_STATE):
    return dict(reduced=r, full=r * r, ratio=float(r))


def jet_max_state_norm(theta, gen_params, h0, xs, r=R_STATE):
    step = make_jet_step(gen_params, r)
    Hs = make_rollout(step)(h0, theta, xs)
    return float(jnp.max(jnp.abs(Hs)))


# ---------------------------------------------------------------------
# 3. Bounded-interface / flag (T=V(+)U), reused unmodified from B31a.
#    Native input IS a scalar already -- no embedding needed.
# ---------------------------------------------------------------------
def flag_make_params(seed):
    return make_flag_theta(seed)


def flag_step(h, params, x_t, consts=FLAG_CONSTS):
    return flag_full_step_state(h, x_t, params, consts)


def flag_param_count(params):
    return int(sum(np.prod(v.shape) for v in params.values()))


def flag_credit_scalars(P_c):
    return dict(reduced=int(FLAG_V_DIM * P_c), full=int(FLAG_R_DIM * P_c),
                ratio=FLAG_R_DIM / FLAG_V_DIM)


RHO_MAX = 0.95


def project_stable_R_V(params):
    """Legitimate ARCHITECTURAL stability constraint (not an
    optimization patch): R_V is trainable with no structural spectral-
    radius guarantee, so its spectral radius is projected to stay
    <=RHO_MAX, exactly as in B31b."""
    if "R_V" not in params:
        return params
    R_V = params["R_V"]
    eigval_mag = jnp.max(jnp.abs(jnp.linalg.eigvals(R_V)))
    scale = jnp.where(eigval_mag > RHO_MAX, RHO_MAX / eigval_mag, 1.0)
    new_params = dict(params)
    new_params["R_V"] = R_V * scale
    return new_params


# ---------------------------------------------------------------------
# 4. Dense BPTT oracle. Input embedding: same (hidden,1) convention as RTU.
# ---------------------------------------------------------------------
def make_stable_dense(n, rng, radius):
    M = rng.randn(n, n) / np.sqrt(n)
    eig = np.max(np.abs(np.linalg.eigvals(M)))
    return M * (radius / eig)


def dense_make_params(seed, hidden=DENSE_HIDDEN):
    rng = np.random.RandomState(seed)
    return dict(
        A=jnp.array(make_stable_dense(hidden, rng, radius=0.9)),
        B=jnp.array(rng.randn(hidden, 1) / 1.0),
        b=jnp.array(rng.randn(hidden) * 0.05),
    )


def dense_step(h, params, x_t):
    xv = jnp.atleast_1d(x_t)
    return jnp.tanh(params["A"] @ h + params["B"] @ xv + params["b"])


def dense_param_count(params):
    return int(sum(np.prod(v.shape) for v in params.values()))


# ---------------------------------------------------------------------
# Teachers -- ONE scalar exogenous input, shared across all architectures.
# ---------------------------------------------------------------------
def make_readout(seed, state_dim, y_dim=Y_DIM):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(y_dim, state_dim) * (1.0 / np.sqrt(state_dim)))


def make_sequence(seed, T, state_dim):
    rng = np.random.RandomState(seed)
    h0 = jnp.array(rng.randn(state_dim) * 0.15)
    xs = jnp.array(rng.randn(T) * 0.6)  # ONE scalar per step
    return h0, xs


class Teacher:
    def __init__(self, name, state_dim, rollout_fn, params, W):
        self.name = name
        self.state_dim = state_dim
        self.rollout_fn = rollout_fn
        self.params = params
        self.W = W

    def targets(self, h0, xs):
        Hs = self.rollout_fn(h0, self.params, xs)
        return Hs @ self.W.T


def make_teacher_A_independent(seed=777):
    """Teacher A: independent-mode positive control for RTU -- a frozen
    RTU instance, SAME architecture the RTU student uses (differ only
    in trainable-parameter values)."""
    params = rtu_make_params(seed, RTU_HIDDEN)
    rollout = make_rollout(lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN))
    W = make_readout(seed + 1, 2 * RTU_HIDDEN)
    return Teacher("A_independent", 2 * RTU_HIDDEN, rollout, params, W)


def make_teacher_B_jet(seed=778, r=R_STATE, gen_seed=None):
    """Teacher B: jet/indecomposable positive control for B34 -- a
    frozen B34 instance. Returns (teacher, gen_params); the B34 STUDENT
    must reuse this EXACT gen_params (not an independently-redrawn one)
    for the positive-control pairing to be valid. Readout deliberately
    zeroes the scalar (index-0) jet coordinate's column so the task
    genuinely requires propagation along several jet coordinates."""
    if gen_seed is None:
        gen_seed = 2000 + r
    gen_params = jet_make_gen_params(seed=gen_seed, r=r)
    theta_star = jet_make_theta(seed, r)
    step = make_jet_step(gen_params, r)
    rollout = make_rollout(step)
    W = make_readout(seed + 1, r)
    W = W.at[:, 0].set(0.0)
    return Teacher("B_jet", r, rollout, theta_star, W), gen_params


def make_teacher_C_multipole(seed=779, hidden_dim=RTU_HIDDEN):
    """Teacher C: distinct-timescale / multi-pole negative control for
    B34 -- an RTU instance with EXPLICITLY clustered (r,theta) pairs (4
    clearly-separated decay/frequency clusters), intentionally NOT
    required to be realizable by any student."""
    rng = np.random.RandomState(seed)
    clusters = [(0.97, 0.10), (0.85, 1.00), (0.65, 2.50), (0.30, 5.00)]
    per_cluster = hidden_dim // len(clusters)
    r_vals, theta_vals = [], []
    for (r_c, th_c) in clusters:
        r_vals += [r_c] * per_cluster
        theta_vals += [th_c] * per_cluster
    while len(r_vals) < hidden_dim:
        r_vals.append(clusters[-1][0]); theta_vals.append(clusters[-1][1])
    r_arr = np.array(r_vals[:hidden_dim])
    theta_arr = np.array(theta_vals[:hidden_dim])
    nu_log = np.log(-np.log(r_arr))
    theta_log = np.log(theta_arr)
    params = dict(
        nu_log=jnp.array(nu_log), theta_log=jnp.array(theta_log),
        B_real=jnp.array(rng.randn(hidden_dim, 1)),
        B_imag=jnp.array(rng.randn(hidden_dim, 1)),
    )
    rollout = make_rollout(lambda h, p, x: rtu_step(h, p, x, hidden_dim))
    W = make_readout(seed + 1, 2 * hidden_dim)
    return Teacher("C_multipole", 2 * hidden_dim, rollout, params, W)


def make_teacher_D_coupled(seed=780):
    """Teacher D: globally coupled bounded-interface teacher -- a frozen
    flag instance sharing FLAG_CONSTS with the student (positive control
    for the flag architecture)."""
    theta_star = make_flag_theta(seed)
    rollout = make_rollout(lambda h, p, x: flag_step(h, p, x, FLAG_CONSTS))
    W = make_readout(seed + 1, FLAG_R_DIM)
    return Teacher("D_coupled", FLAG_R_DIM, rollout, theta_star, W)


# ---------------------------------------------------------------------
# Adam optimizer (uniform across all architectures -- not per-arch tuned).
# ---------------------------------------------------------------------
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=zeros, v=jax.tree_util.tree_map(jnp.zeros_like, params), t=0)


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * (g * g), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return new_params, dict(m=m, v=v, t=t)


GRAD_CLIP_NORM = 10.0


def clip_grad(g, max_norm=GRAD_CLIP_NORM):
    leaves, _ = jax.tree_util.tree_flatten(g)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    norm = jnp.linalg.norm(flat)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda x: x * scale, g)


DIVERGENCE_LOSS_CEIL = 1e4


def train_student_one_run(rollout_fn, make_params_fn, param_count_fn, teacher, lr, seed_init,
                           n_train=N_TRAIN, n_val=N_VAL, T=T_SEQ,
                           seq_seed_offset=20_000, val_seed_offset=95_000):
    """ONE (architecture, teacher, lr, seed) training run. Adam + common
    gradient clipping; architecture-appropriate structural projection
    (R_V spectral cap) where applicable. Divergence stops training
    immediately and is reported explicitly -- NEVER folded into an NMSE
    number."""
    loss_fn = make_loss_mse(rollout_fn)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_params_fn(seed_init)
    is_dict_params = isinstance(params, dict)
    opt_state = adam_init(params)
    train_losses = []
    diverged = False
    diverged_at_step = None
    t0 = time.time()
    for step in range(n_train):
        h0, xs = make_sequence(seq_seed_offset + step, T, teacher.state_dim)
        targets = teacher.targets(h0, xs)
        loss_val = float(loss_jit(params, h0, xs, targets, teacher.W))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged = True
            diverged_at_step = step
            break
        train_losses.append(loss_val)
        g = grad_fn(params, h0, xs, targets, teacher.W)
        g = clip_grad(g)
        params, opt_state = adam_step(params, g, opt_state, lr)
        if is_dict_params:
            params = project_stable_R_V(params)
    elapsed = time.time() - t0

    if diverged:
        return dict(train_losses=train_losses, nmse=None, val_loss=None, elapsed=elapsed,
                    P_c=param_count_fn(params), diverged=True, diverged_at_step=diverged_at_step,
                    lr=lr, seed=seed_init)

    val_losses, val_ref_var = [], []
    for i in range(n_val):
        h0, xs = make_sequence(val_seed_offset + i, T, teacher.state_dim)
        targets = teacher.targets(h0, xs)
        vl = float(loss_jit(params, h0, xs, targets, teacher.W))
        val_losses.append(vl)
        val_ref_var.append(float(jnp.mean(targets ** 2)))
    if not all(np.isfinite(val_losses)):
        return dict(train_losses=train_losses, nmse=None, val_loss=None, elapsed=elapsed,
                    P_c=param_count_fn(params), diverged=True, diverged_at_step=n_train, lr=lr, seed=seed_init)
    nmse = float(np.mean(val_losses) / (np.mean(val_ref_var) + 1e-12))
    return dict(train_losses=train_losses, val_loss=float(np.mean(val_losses)), nmse=nmse,
                elapsed=elapsed, P_c=param_count_fn(params), diverged=False, diverged_at_step=None,
                lr=lr, seed=seed_init)


def train_student_with_grid(rollout_fn, make_params_fn, param_count_fn, teacher, lr_grid, seeds=SEEDS,
                             **kwargs):
    """Small LR grid x fixed seed count; selects the (lr) with the best
    MEAN FINITE validation NMSE. Reports divergence fraction over ALL
    grid x seed runs, never mixes a sentinel into the NMSE numbers."""
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_student_one_run(rollout_fn, make_params_fn, param_count_fn, teacher, lr,
                                         seed_init=1000 + seed, **kwargs)
            res["lr_tag"] = lr
            all_runs.append(res)

    n_total = len(all_runs)
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    finite_runs = [r for r in all_runs if not r["diverged"]]

    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite_runs if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))

    if not by_lr:
        return dict(status="all_diverged", n_total=n_total, n_diverged=n_diverged,
                     best_lr=None, nmse_values=[], nmse_mean=None, nmse_median=None, P_c=all_runs[0]["P_c"])

    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite_runs if r["lr_tag"] == best_lr]
    nmse_values = [r["nmse"] for r in best_runs]
    return dict(status="ok", n_total=n_total, n_diverged=n_diverged, best_lr=best_lr,
                nmse_values=nmse_values, nmse_mean=float(np.mean(nmse_values)),
                nmse_median=float(np.median(nmse_values)), P_c=best_runs[0]["P_c"],
                diverged_at_best_lr=sum(1 for r in all_runs if r["lr_tag"] == best_lr and r["diverged"]))


# ---------------------------------------------------------------------
# One-time exact-online-vs-BPTT verification (unchanged from before).
# ---------------------------------------------------------------------
def verify_full_rtrl_once(step_fn, params, state_dim, seed=0, T=10):
    from jax.flatten_util import ravel_pytree
    theta_flat, unravel = ravel_pytree(params)
    P_c = theta_flat.shape[0]
    rng = np.random.RandomState(seed)
    h0 = jnp.array(rng.randn(state_dim) * 0.15)
    xs = jnp.array(rng.randn(T) * 0.6)
    qs = jnp.array(rng.randn(T, state_dim))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))

    def ell(y, ph):
        return jnp.sin(y + ph) + 0.5 * y ** 2

    def dell_dy(y, ph):
        return jnp.cos(y + ph) + y

    def loss_bptt(th):
        def body(h, inp):
            x_t, q, ph = inp
            h_next = step_fn(h, unravel(th), x_t)
            return h_next, q @ h_next
        _, ys = jax.lax.scan(body, h0, (xs, qs, phases))
        return jnp.sum(ell(ys, phases))

    g_b = jax.grad(loss_bptt)(theta_flat)

    h = h0
    S = jnp.zeros((state_dim, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda hh: step_fn(hh, unravel(theta_flat), x_t))(h)
        G_t = jax.jacobian(lambda thf: step_fn(h, unravel(thf), x_t))(theta_flat)
        S = J_t @ S + G_t
        h_next = step_fn(h, unravel(theta_flat), x_t)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    rel = float(jnp.linalg.norm(g_total - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
    return rel


# ---------------------------------------------------------------------
# Sanity checks -- MUST pass before any cross-family matrix.
# ---------------------------------------------------------------------
def run_sanity_checks():
    print("=" * 78)
    print("Phase 2A sanity checks: positive controls + dense oracle, BEFORE cross-family matrix")
    print("=" * 78)
    teacher_A = make_teacher_A_independent()
    teacher_B, teacher_B_gen = make_teacher_B_jet()
    teacher_C = make_teacher_C_multipole()
    teacher_D = make_teacher_D_coupled()

    lr_grid_rtu = (0.03, 0.1, 0.3)
    lr_grid_jet = (0.01, 0.03, 0.1)
    lr_grid_flag = (0.003, 0.01, 0.03)
    lr_grid_dense = (0.01, 0.03, 0.1)

    checks = {}

    res = train_student_with_grid(make_rollout(lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN)),
                                   lambda seed: rtu_make_params(seed, RTU_HIDDEN), rtu_param_count,
                                   teacher_A, lr_grid_rtu)
    checks["RTU->A"] = res
    print(f"  RTU -> A (positive control):        {res}")

    res = train_student_with_grid(make_rollout(make_jet_step(teacher_B_gen, R_STATE)),
                                   lambda seed: jet_make_theta(seed, R_STATE), jet_param_count,
                                   teacher_B, lr_grid_jet)
    checks["B34->B"] = res
    print(f"  B34 -> B (positive control, SAME gen_params as teacher): {res}")

    res = train_student_with_grid(make_rollout(lambda h, p, x: flag_step(h, p, x, FLAG_CONSTS)),
                                   flag_make_params, flag_param_count, teacher_D, lr_grid_flag)
    checks["Flag->D"] = res
    print(f"  Flag -> D (positive control):        {res}")

    for name, t in [("A", teacher_A), ("B", teacher_B), ("C", teacher_C), ("D", teacher_D)]:
        res = train_student_with_grid(make_rollout(dense_step), dense_make_params, dense_param_count,
                                       t, lr_grid_dense)
        checks[f"Dense->{name}"] = res
        print(f"  Dense -> {name}:                      {res}")

    print()
    print("Positive-control PASS criteria (NMSE << 1, using a permissive 0.3 threshold):")
    all_pass = True
    for key in ("RTU->A", "B34->B", "Flag->D"):
        r = checks[key]
        ok = r["status"] == "ok" and r["nmse_mean"] < 0.3
        all_pass = all_pass and ok
        print(f"  {key}: {'PASS' if ok else 'FAIL'}  (nmse_mean={r.get('nmse_mean')})")
    print(f"ALL POSITIVE CONTROLS PASS: {all_pass}")
    return dict(all_pass=all_pass, checks=checks, teacher_B_gen=teacher_B_gen)


# ---------------------------------------------------------------------
# Cross-family matrix (View 1) -- ONLY run after sanity checks pass.
# ---------------------------------------------------------------------
def run_view1_matrix(teacher_B_gen):
    print()
    print("=" * 78)
    print("Phase 2A View 1: matched state size (r~64), cross-family matrix")
    print("=" * 78)

    teacher_A = make_teacher_A_independent()
    # teacher_B recreated deterministically (same seeds as the sanity check's
    # make_teacher_B_jet()) so it matches teacher_B_gen exactly.
    teacher_B, _ = make_teacher_B_jet()
    teacher_C = make_teacher_C_multipole()
    teacher_D = make_teacher_D_coupled()
    teachers = [teacher_A, teacher_B, teacher_C, teacher_D]

    lr_grids = dict(RTU=(0.03, 0.1, 0.3), B34=(0.01, 0.03, 0.1),
                     BoundedInterfaceFlag=(0.003, 0.01, 0.03), DenseBPTTOracle=(0.01, 0.03, 0.1))

    architectures = dict(
        RTU=dict(rollout=make_rollout(lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN)),
                 make_params=lambda seed: rtu_make_params(seed, RTU_HIDDEN),
                 param_count=rtu_param_count, state_dim=2 * RTU_HIDDEN,
                 credit=rtu_credit_scalars(RTU_HIDDEN)),
        B34=dict(rollout=make_rollout(make_jet_step(teacher_B_gen, R_STATE)),
                 make_params=lambda seed: jet_make_theta(seed, R_STATE),
                 param_count=jet_param_count, state_dim=R_STATE,
                 credit=jet_credit_scalars(R_STATE)),
        BoundedInterfaceFlag=dict(
                 rollout=make_rollout(lambda h, p, x: flag_step(h, p, x, FLAG_CONSTS)),
                 make_params=flag_make_params, param_count=flag_param_count,
                 state_dim=FLAG_R_DIM, credit=None),
        DenseBPTTOracle=dict(rollout=make_rollout(dense_step),
                 make_params=dense_make_params, param_count=dense_param_count,
                 state_dim=DENSE_HIDDEN, credit=dict(reduced=None, full=None, ratio=None)),
    )

    results = []
    for arch_name, arch in architectures.items():
        for teacher in teachers:
            res = train_student_with_grid(arch["rollout"], arch["make_params"], arch["param_count"],
                                           teacher, lr_grids[arch_name])
            credit = arch["credit"]
            if arch_name == "BoundedInterfaceFlag" and credit is None:
                credit = flag_credit_scalars(res["P_c"])
            row = dict(arch=arch_name, teacher=teacher.name, state_dim=arch["state_dim"],
                       credit=credit, **res)
            results.append(row)
            if res["status"] == "all_diverged":
                print(f"  arch={arch_name:22s} teacher={teacher.name:14s}  ALL RUNS DIVERGED "
                      f"({res['n_diverged']}/{res['n_total']}) -- optimization/dynamical instability, "
                      f"NOT evidence of representational impossibility", flush=True)
            else:
                print(f"  arch={arch_name:22s} teacher={teacher.name:14s} state_dim={arch['state_dim']:3d}  "
                      f"P_c={res['P_c']:6d}  best_lr={res['best_lr']}  "
                      f"NMSE(finite)={res['nmse_mean']:.4e} (median {res['nmse_median']:.4e})  "
                      f"diverged={res['n_diverged']}/{res['n_total']}", flush=True)

    with open("/tmp/p2a_view1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print("Saved to /tmp/p2a_view1_results.json")
    return results


if __name__ == "__main__":
    sanity = run_sanity_checks()
    if not sanity["all_pass"]:
        print()
        print("STOPPING: not all positive controls passed. Repair before running the cross-family matrix.")
    else:
        run_view1_matrix(sanity["teacher_B_gen"])
