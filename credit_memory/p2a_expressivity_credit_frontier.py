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
                           seq_seed_offset=20_000, val_seed_offset=95_000, test_seed_offset=200_000):
    """ONE (architecture, teacher, lr, seed) training run. Adam + common
    gradient clipping; architecture-appropriate structural projection
    (R_V spectral cap) where applicable. Divergence stops training
    immediately and is reported explicitly -- NEVER folded into an NMSE
    number.

    Train/validation/test separation (audit fix): val_seed_offset
    sequences are used ONLY for LR selection (train_student_with_grid);
    test_seed_offset sequences are a disjoint, untouched set evaluated
    here but never used for selection -- the reported "final" number in
    the matrix must come from the test set, not the selection set."""
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
        return dict(train_losses=train_losses, nmse=None, test_nmse=None, val_loss=None, elapsed=elapsed,
                    P_c=param_count_fn(params), diverged=True, diverged_at_step=diverged_at_step,
                    lr=lr, seed=seed_init)

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            h0, xs = make_sequence(offset + i, T, teacher.state_dim)
            targets = teacher.targets(h0, xs)
            losses.append(float(loss_jit(params, h0, xs, targets, teacher.W)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_ref_var = eval_split(val_seed_offset, n_val)
    test_losses, test_ref_var = eval_split(test_seed_offset, n_val)
    if not all(np.isfinite(val_losses)) or not all(np.isfinite(test_losses)):
        return dict(train_losses=train_losses, nmse=None, test_nmse=None, val_loss=None, elapsed=elapsed,
                    P_c=param_count_fn(params), diverged=True, diverged_at_step=n_train, lr=lr, seed=seed_init)
    nmse = float(np.mean(val_losses) / (np.mean(val_ref_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_ref_var) + 1e-12))
    return dict(train_losses=train_losses, val_loss=float(np.mean(val_losses)), nmse=nmse,
                test_nmse=test_nmse, elapsed=elapsed, P_c=param_count_fn(params), diverged=False,
                diverged_at_step=None, lr=lr, seed=seed_init)


def train_student_with_grid(rollout_fn, make_params_fn, param_count_fn, teacher, lr_grid, seeds=SEEDS,
                             **kwargs):
    """Small LR grid x fixed seed count. LR is selected using VALIDATION
    NMSE ONLY (by_lr keys off r["nmse"], the validation-set number,
    never r["test_nmse"]). The TEST NMSE reported for the selected LR
    comes from a disjoint set never used in selection. Reports
    divergence fraction over ALL grid x seed runs, never mixes a
    sentinel into either NMSE number."""
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
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))  # VALIDATION nmse -- selection criterion

    if not by_lr:
        return dict(status="all_diverged", n_total=n_total, n_diverged=n_diverged,
                     best_lr=None, val_nmse_values=[], val_nmse_mean=None, val_nmse_median=None,
                     test_nmse_values=[], test_nmse_mean=None, test_nmse_median=None,
                     P_c=all_runs[0]["P_c"])

    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite_runs if r["lr_tag"] == best_lr]
    val_nmse_values = [r["nmse"] for r in best_runs]
    test_nmse_values = [r["test_nmse"] for r in best_runs]
    return dict(status="ok", n_total=n_total, n_diverged=n_diverged, best_lr=best_lr,
                val_nmse_values=val_nmse_values, val_nmse_mean=float(np.mean(val_nmse_values)),
                val_nmse_median=float(np.median(val_nmse_values)),
                test_nmse_values=test_nmse_values, test_nmse_mean=float(np.mean(test_nmse_values)),
                test_nmse_median=float(np.median(test_nmse_values)),
                P_c=best_runs[0]["P_c"],
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
# Audit item 3: positive-control saturation -- same protocol, 5x steps.
# ---------------------------------------------------------------------
def run_saturation_audit(teacher_B_gen, multiplier=5):
    print()
    print("=" * 78)
    print(f"Audit 3: positive-control saturation, {multiplier}x training horizon "
          f"(N_TRAIN={N_TRAIN*multiplier}), same optimizer/clipping/data/init/LR-selection protocol")
    print("=" * 78)
    teacher_A = make_teacher_A_independent()
    teacher_B, _ = make_teacher_B_jet()
    teacher_D = make_teacher_D_coupled()

    configs = [
        ("RTU->A", make_rollout(lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN)),
         lambda seed: rtu_make_params(seed, RTU_HIDDEN), rtu_param_count, teacher_A, (0.03, 0.1, 0.3)),
        ("B34->B", make_rollout(make_jet_step(teacher_B_gen, R_STATE)),
         lambda seed: jet_make_theta(seed, R_STATE), jet_param_count, teacher_B, (0.01, 0.03, 0.1)),
        ("Flag->D", make_rollout(lambda h, p, x: flag_step(h, p, x, FLAG_CONSTS)),
         flag_make_params, flag_param_count, teacher_D, (0.003, 0.01, 0.03)),
    ]

    results = {}
    for name, rollout_fn, make_params_fn, param_count_fn, teacher, lr_grid in configs:
        res = train_student_with_grid(rollout_fn, make_params_fn, param_count_fn, teacher, lr_grid,
                                       n_train=N_TRAIN * multiplier)
        results[name] = res
        # learning curve: sparse checkpoints from the single run at the selected LR/first seed,
        # rerun once more to capture train_losses (train_student_with_grid discards per-run curves)
        curve_res = train_student_one_run(rollout_fn, make_params_fn, param_count_fn, teacher,
                                           res["best_lr"], seed_init=1000 + SEEDS[0],
                                           n_train=N_TRAIN * multiplier)
        tl = curve_res["train_losses"]
        n = len(tl)
        checkpoints = sorted(set([0, n // 10, n // 4, n // 2, 3 * n // 4, n - 1])) if n > 0 else []
        curve_str = ", ".join(f"step{c}={tl[c]:.4e}" for c in checkpoints)
        print(f"  {name}: best_lr={res['best_lr']}  VAL_NMSE={res['val_nmse_mean']:.4e}  "
              f"TEST_NMSE={res['test_nmse_mean']:.4e}  diverged={res['n_diverged']}/{res['n_total']}")
        print(f"    learning curve (seed {SEEDS[0]}, best lr): {curve_str}")
    return results


# ---------------------------------------------------------------------
# Audit item 4: exact-online (full RTRL, autodiff S-propagation) vs
# BPTT -- FULL OPTIMIZER TRAJECTORY comparison (not just one gradient),
# on each architecture's own positive-control teacher.
# ---------------------------------------------------------------------
def full_rtrl_grad_mse_generic(step_fn, params, unravel, theta_flat, P_c, h0, xs, targets, W, state_dim):
    T = xs.shape[0]
    h = h0
    S = jnp.zeros((state_dim, P_c), dtype=jnp.float64)
    g_total = jnp.zeros(P_c, dtype=jnp.float64)
    loss_total = 0.0
    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda hh: step_fn(hh, unravel(theta_flat), x_t))(h)
        G_t = jax.jacobian(lambda thf: step_fn(h, unravel(thf), x_t))(theta_flat)
        S = J_t @ S + G_t
        h_next = step_fn(h, unravel(theta_flat), x_t)
        y = W @ h_next
        diff = y - targets[t]
        loss_total = loss_total + 0.5 * jnp.sum(diff ** 2)
        dl_dh = W.T @ diff
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total / T, loss_total / T


def run_exact_rtrl_vs_bptt_trajectory(step_fn, make_params_fn, teacher, lr, state_dim, seed_init=1000,
                                       n_steps=10, T=T_SEQ):
    """Two identical copies (same init, same Adam state, same data),
    one updated via generic exact full-RTRL gradients (autodiff S
    propagation -- exact, not the BPTT reverse-mode graph), one via
    BPTT, for n_steps Adam updates. Reports max gradient discrepancy
    (step 0) and max parameter discrepancy over the trajectory."""
    from jax.flatten_util import ravel_pytree
    loss_fn = make_loss_mse(make_rollout(step_fn))
    grad_bptt_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    params_rtrl = make_params_fn(seed_init)
    params_bptt = make_params_fn(seed_init)
    theta_flat0, unravel = ravel_pytree(params_rtrl)
    P_c = theta_flat0.shape[0]
    opt_rtrl = adam_init(params_rtrl)
    opt_bptt = adam_init(params_bptt)

    grad_discrepancies, param_discrepancies = [], []
    for step in range(n_steps):
        h0, xs = make_sequence(20_000 + step, T, state_dim)
        targets = teacher.targets(h0, xs)

        theta_flat_rtrl, _ = ravel_pytree(params_rtrl)
        g_rtrl_flat, _ = full_rtrl_grad_mse_generic(step_fn, params_rtrl, unravel, theta_flat_rtrl, P_c,
                                                     h0, xs, targets, teacher.W, state_dim)
        g_rtrl = unravel(g_rtrl_flat)
        g_bptt = grad_bptt_fn(params_bptt, h0, xs, targets, teacher.W)

        g_rtrl_flat_cmp, _ = ravel_pytree(g_rtrl)
        g_bptt_flat_cmp, _ = ravel_pytree(g_bptt)
        grad_disc = float(jnp.linalg.norm(g_rtrl_flat_cmp - g_bptt_flat_cmp))
        grad_discrepancies.append(grad_disc)

        g_rtrl_c = clip_grad(g_rtrl)
        g_bptt_c = clip_grad(g_bptt)
        params_rtrl, opt_rtrl = adam_step(params_rtrl, g_rtrl_c, opt_rtrl, lr)
        params_bptt, opt_bptt = adam_step(params_bptt, g_bptt_c, opt_bptt, lr)
        if isinstance(params_rtrl, dict):
            params_rtrl = project_stable_R_V(params_rtrl)
            params_bptt = project_stable_R_V(params_bptt)

        p_rtrl_flat, _ = ravel_pytree(params_rtrl)
        p_bptt_flat, _ = ravel_pytree(params_bptt)
        param_discrepancies.append(float(jnp.linalg.norm(p_rtrl_flat - p_bptt_flat)))

    return dict(grad_discrepancy_step0=grad_discrepancies[0], max_grad_discrepancy=max(grad_discrepancies),
                max_param_discrepancy=max(param_discrepancies), param_discrepancies=param_discrepancies)


def run_exact_gradient_trajectory_audit(teacher_B_gen):
    print()
    print("=" * 78)
    print("Audit 4: exact-online (full RTRL) vs BPTT, full optimizer trajectory (10 steps)")
    print("=" * 78)
    teacher_A = make_teacher_A_independent()
    teacher_B, _ = make_teacher_B_jet()
    teacher_D = make_teacher_D_coupled()

    res_rtu = run_exact_rtrl_vs_bptt_trajectory(
        lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN), lambda seed: rtu_make_params(seed, RTU_HIDDEN),
        teacher_A, lr=0.1, state_dim=2 * RTU_HIDDEN)
    print(f"  RTU->A:  grad_discrepancy(step0)={res_rtu['grad_discrepancy_step0']:.3e}  "
          f"max_grad_discrepancy={res_rtu['max_grad_discrepancy']:.3e}  "
          f"max_param_discrepancy={res_rtu['max_param_discrepancy']:.3e}")

    res_b34 = run_exact_rtrl_vs_bptt_trajectory(
        make_jet_step(teacher_B_gen, R_STATE), lambda seed: jet_make_theta(seed, R_STATE),
        teacher_B, lr=0.03, state_dim=R_STATE)
    print(f"  B34->B:  grad_discrepancy(step0)={res_b34['grad_discrepancy_step0']:.3e}  "
          f"max_grad_discrepancy={res_b34['max_grad_discrepancy']:.3e}  "
          f"max_param_discrepancy={res_b34['max_param_discrepancy']:.3e}")

    res_flag = run_exact_rtrl_vs_bptt_trajectory(
        lambda h, p, x: flag_step(h, p, x, FLAG_CONSTS), flag_make_params,
        teacher_D, lr=0.003, state_dim=FLAG_R_DIM)
    print(f"  Flag->D: grad_discrepancy(step0)={res_flag['grad_discrepancy_step0']:.3e}  "
          f"max_grad_discrepancy={res_flag['max_grad_discrepancy']:.3e}  "
          f"max_param_discrepancy={res_flag['max_param_discrepancy']:.3e}")

    return dict(RTU=res_rtu, B34=res_b34, Flag=res_flag)


# ---------------------------------------------------------------------
# Audit item 5: structural accounting, kept separate from optimization outcome.
# ---------------------------------------------------------------------
def print_structural_accounting():
    print()
    print("=" * 78)
    print("Audit 5: structural accounting (state dim r, trainable P, exact credit, r*P, ratio)")
    print("=" * 78)
    rows = [
        ("RTU", 2 * RTU_HIDDEN, 128, rtu_credit_scalars(RTU_HIDDEN)),
        ("B34", R_STATE, R_STATE, jet_credit_scalars(R_STATE)),
        ("BoundedInterfaceFlag", FLAG_R_DIM, 10888, flag_credit_scalars(10888)),
    ]
    for name, r, P, credit in rows:
        print(f"  {name:22s} r={r:4d}  P={P:6d}  exact_credit={credit['reduced']:7d}  "
              f"r*P(full)={credit['full']:7d}  ratio={credit['ratio']:.1f}x")
    print(f"  {'DenseBPTTBaseline':22s} r={DENSE_HIDDEN:4d}  P={4224:6d}  "
          f"exact_credit=N/A (not an online learner)  r*P(full)={DENSE_HIDDEN*4224:7d} (hypothetical, "
          f"illustrative only)  ratio=N/A")
    print("  NOTE: View 1 is NOT matched parameter count or matched credit budget -- do not read it that way.")


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
    print("Positive-control PASS criteria (TEST NMSE << 1, permissive 0.3 threshold; "
          "LR was selected on VALIDATION NMSE only, never on this test number):")
    all_pass = True
    for key in ("RTU->A", "B34->B", "Flag->D"):
        r = checks[key]
        ok = r["status"] == "ok" and r["test_nmse_mean"] < 0.3
        all_pass = all_pass and ok
        print(f"  {key}: {'PASS' if ok else 'FAIL'}  (val_nmse_mean={r.get('val_nmse_mean')}, "
              f"test_nmse_mean={r.get('test_nmse_mean')})")
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
                     BoundedInterfaceFlag=(0.003, 0.01, 0.03), DenseBPTTBaseline=(0.01, 0.03, 0.1))

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
        DenseBPTTBaseline=dict(rollout=make_rollout(dense_step),
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
                      f"VAL_NMSE={res['val_nmse_mean']:.4e}  TEST_NMSE={res['test_nmse_mean']:.4e} "
                      f"(median {res['test_nmse_median']:.4e})  "
                      f"diverged={res['n_diverged']}/{res['n_total']}", flush=True)

    with open("/tmp/p2a_view1_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print("Saved to /tmp/p2a_view1_results_v2.json")
    return results


if __name__ == "__main__":
    sanity = run_sanity_checks()
    if not sanity["all_pass"]:
        print()
        print("STOPPING: not all positive controls passed. Repair before running the cross-family matrix.")
    else:
        run_view1_matrix(sanity["teacher_B_gen"])
        run_saturation_audit(sanity["teacher_B_gen"])
        run_exact_gradient_trajectory_audit(sanity["teacher_B_gen"])
        print_structural_accounting()
