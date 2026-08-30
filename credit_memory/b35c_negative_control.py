"""B35c negative control -- a deliberately NONCOMMUTING state-tracking
teacher (S_3 permutation-word composition). ProductLocal's transition
family is exactly commutative (elements of a direct product of
commutative rings, R[eps_q]/(eps_q^d)): ANY two selectable transitions
theta^(0), theta^(1) satisfy M_{theta^(0)}@M_{theta^(1)} = M_{theta^(1)}
@M_{theta^(0)} regardless of trained values. A task whose ground truth
requires genuinely noncommuting composition (order-dependent generator
products) is therefore EXPECTED to expose a real limitation, not a bug
-- establishing the phase boundary rather than hiding it.

Teacher: h_{t+1} = G[x_t] @ h_t, x_t in {0,1,2} selects among 3
S_3-generating 3x3 permutation matrices (identity, a transposition, a
3-cycle) -- verified non-commuting below.

Students (matched state dim r=12, plain BPTT -- this is a qualitative
phase-boundary demonstration, not a credit-accounting claim, so no
custom RTRL is implemented here):
  RegularBlock: 3 LEARNED commutative-algebra elements theta^(k)
    (k=0,1,2), h_{t+1} = alg_mult_blockwise(theta^{x_t}, h, Q, d) --
    genuinely input-selected, but every selectable transition commutes
    with every other, by construction of the algebra.
  GenericBlock: 3 LEARNED unrestricted dense (r,r) matrices A^{(k)} --
    NOT constrained to commute.
  BoundedInterfaceFlag: reused sized-flag architecture (p2a_sized_flag),
    x_t fed as its native scalar input, nonlinear Phi coupling.

Run: python -m credit_memory.b35c_negative_control
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL
from credit_memory.p2a_sized_flag import make_sized_flag_consts, make_sized_flag_theta, make_sized_flag_step

R = 12
D_LOCAL = 4
Q_LOCAL = R // D_LOCAL
N_TRAIN, N_VAL, N_TEST, T_TRAIN = 100, 15, 15, 48
SEEDS = (0, 1, 2)
LR_GRID = (0.003, 0.01, 0.03, 0.1)

G0 = jnp.eye(3, dtype=jnp.float64)
G1 = jnp.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=jnp.float64)   # swap coords 0,1
G2 = jnp.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=jnp.float64)   # 3-cycle
GENS = jnp.stack([G0, G1, G2])


def verify_noncommuting():
    d12 = G1 @ G2 - G2 @ G1
    d01 = G0 @ G1 - G1 @ G0
    print(f"  ||G1@G2 - G2@G1|| = {float(jnp.max(jnp.abs(d12))):.4f}  (expect > 0, genuinely noncommuting)")
    print(f"  ||G0@G1 - G1@G0|| = {float(jnp.max(jnp.abs(d01))):.4f}  (expect 0, identity commutes with everything)")
    return float(jnp.max(jnp.abs(d12))) > 0.5


def make_symbol_seq(seed, T):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randint(0, 3, size=T))


def teacher_rollout(h0, xs_idx):
    def step(h, k):
        h_next = GENS[k] @ h
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs_idx)
    return Hs


# ---------------------------------------------------------------------
# RegularBlock: lookup table of 3 COMMUTATIVE-ALGEBRA elements.
# ---------------------------------------------------------------------
def make_regular_lookup_params(seed, Q=Q_LOCAL, d=D_LOCAL, r=R):
    rng = np.random.RandomState(seed)
    thetas = jnp.stack([project_local_tails(jnp.array(rng.randn(r) * 0.3), Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
                         for _ in range(3)])
    W_out = jnp.array(rng.randn(3, r) * (1.0 / np.sqrt(r)))
    h0 = jnp.array(rng.randn(r) * 0.5)   # TRAINABLE init -- a purely multiplicative
    return dict(thetas=thetas, W_out=W_out, h0=h0)   # recursion started at 0 is a fixed point (zero gradient)


def regular_lookup_rollout(params, xs_idx, Q=Q_LOCAL, d=D_LOCAL):
    def step(h, k):
        h_next = alg_mult_blockwise(params["thetas"][k], h, Q, d)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, params["h0"], xs_idx)
    return Hs @ params["W_out"].T


def project_regular_lookup(params, Q=Q_LOCAL, d=D_LOCAL):
    thetas = jnp.stack([project_local_tails(params["thetas"][k], Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
                         for k in range(3)])
    return dict(params, thetas=thetas)


# ---------------------------------------------------------------------
# GenericBlock: lookup table of 3 UNRESTRICTED dense (r,r) matrices.
# ---------------------------------------------------------------------
def make_generic_lookup_params(seed, r=R):
    rng = np.random.RandomState(seed)
    As = jnp.array(rng.randn(3, r, r) / np.sqrt(r) * 0.8)
    W_out = jnp.array(rng.randn(3, r) * (1.0 / np.sqrt(r)))
    h0 = jnp.array(rng.randn(r) * 0.5)
    return dict(As=As, W_out=W_out, h0=h0)


def generic_lookup_rollout(params, xs_idx, r=R):
    def step(h, k):
        h_next = params["As"][k] @ h
        return h_next, h_next
    _, Hs = jax.lax.scan(step, params["h0"], xs_idx)
    return Hs @ params["W_out"].T


def project_generic_lookup(params, rho_max=0.98):
    eig = jnp.max(jnp.abs(jnp.linalg.eigvals(params["As"])), axis=(1, 2)) if False else \
        jnp.max(jnp.abs(jnp.linalg.eigvals(params["As"])), axis=1)
    scale = jnp.where(eig > rho_max, rho_max / (eig + 1e-12), 1.0)
    As_new = params["As"] * scale[:, None, None]
    return dict(params, As=As_new)


# ---------------------------------------------------------------------
# BoundedInterfaceFlag: reused sized-flag architecture, x_t (as float
# symbol value) fed natively.
# ---------------------------------------------------------------------
FLAG_DU, FLAG_DV, FLAG_C, FLAG_K, FLAG_H = 8, 4, 4, 4, 16
FLAG_CONSTS_NC = make_sized_flag_consts(seed=55, d_u=FLAG_DU)
FLAG_STEP_NC = make_sized_flag_step(FLAG_CONSTS_NC, FLAG_DU)


def make_flag_lookup_params(seed):
    theta = make_sized_flag_theta(seed, FLAG_DU, FLAG_H, FLAG_DV, FLAG_C, FLAG_K)
    r = FLAG_DU + FLAG_DV
    rng = np.random.RandomState(seed + 555)
    W_out = jnp.array(rng.randn(3, r) * (1.0 / np.sqrt(r)))
    h0 = jnp.array(rng.randn(r) * 0.5)
    return dict(theta=theta, W_out=W_out, h0=h0)


def flag_lookup_rollout(params, xs_idx):
    def step(h, k):
        x_t = jnp.asarray(k, dtype=jnp.float64) - 1.0   # zero-centered symbol value
        h_next = FLAG_STEP_NC(h, params["theta"], x_t)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, params["h0"], xs_idx)
    return Hs @ params["W_out"].T


def project_flag_lookup(params):
    from credit_memory.p2a_expressivity_credit_frontier import project_stable_R_V
    return dict(params, theta=project_stable_R_V(params["theta"]))


# ---------------------------------------------------------------------
# Training (BPTT, plain -- qualitative demonstration, not a credit claim).
# ---------------------------------------------------------------------
def train_one_run(rollout_fn, make_params, project, lr, seed_init, n_train=N_TRAIN, T=T_TRAIN):
    def loss_fn(params, xs_idx, targets):
        ys = rollout_fn(params, xs_idx)
        return jnp.mean(0.5 * jnp.sum((ys - targets) ** 2, axis=1))
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_params(seed_init)
    params = project(params)
    opt_state = adam_init(params)
    diverged, diverged_at = False, None
    for step in range(n_train):
        xs_idx = make_symbol_seq(20_000 + step, T)
        targets = teacher_rollout(jnp.array([1., 2., 3.], dtype=jnp.float64), xs_idx)
        loss_val = float(loss_jit(params, xs_idx, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        g = clip_grad(grad_fn(params, xs_idx, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project(params)
    if diverged:
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init)

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            xs_idx = make_symbol_seq(offset + i, T)
            targets = teacher_rollout(jnp.array([1., 2., 3.], dtype=jnp.float64), xs_idx)
            losses.append(float(loss_jit(params, xs_idx, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_var = eval_split(95_000, N_VAL)
    test_losses, test_var = eval_split(200_000, N_TEST)
    if not all(np.isfinite(val_losses)) or not all(np.isfinite(test_losses)):
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=n_train, lr=lr, seed=seed_init)
    nmse = float(np.mean(val_losses) / (np.mean(val_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_var) + 1e-12))
    return dict(nmse=nmse, test_nmse=test_nmse, diverged=False, diverged_at=None, lr=lr, seed=seed_init)


def train_with_grid(rollout_fn, make_params, project, lr_grid=LR_GRID, seeds=SEEDS):
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run(rollout_fn, make_params, project, lr, seed_init=1000 + seed)
            res["lr_tag"] = lr
            all_runs.append(res)
    finite = [r for r in all_runs if not r["diverged"]]
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    if not finite:
        return dict(status="all_diverged", n_diverged=n_diverged, n_total=len(all_runs))
    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))
    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite if r["lr_tag"] == best_lr]
    vals = [r["test_nmse"] for r in best_runs]
    return dict(status="ok", n_diverged=n_diverged, n_total=len(all_runs), best_lr=best_lr,
                test_nmse_median=float(np.median(vals)), test_nmse_mean=float(np.mean(vals)),
                test_nmse_std=float(np.std(vals)))


if __name__ == "__main__":
    print("=" * 78)
    print("Verify the negative-control teacher is genuinely noncommuting")
    print("=" * 78)
    ok = verify_noncommuting()
    print(f"NONCOMMUTING CONFIRMED: {ok}")

    print("\n" + "=" * 78)
    print(f"Negative control: RegularBlock (commutative, r={R}) vs GenericBlock (unrestricted, r={R}) "
          f"vs BoundedInterfaceFlag (r={FLAG_DU+FLAG_DV}) on S_3 word-composition tracking")
    print("=" * 78)
    res_reg = train_with_grid(regular_lookup_rollout, make_regular_lookup_params, project_regular_lookup)
    res_gen = train_with_grid(generic_lookup_rollout, make_generic_lookup_params, project_generic_lookup)
    res_flag = train_with_grid(flag_lookup_rollout, make_flag_lookup_params, project_flag_lookup)
    print(f"  RegularBlock (commutative, expected to lose):       {res_reg}")
    print(f"  GenericBlock (unrestricted dense, expected to win):  {res_gen}")
    print(f"  BoundedInterfaceFlag (nonlinear, input-coupled):     {res_flag}")
