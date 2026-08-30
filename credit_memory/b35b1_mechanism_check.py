"""B35b Part 1 -- mechanism check: does a d=2 regular-algebra local
block exploit a genuine generalized (repeated-pole) mode that d=1
(pure semisimple/diagonal) can only approximate via additional
semisimple modes?

Teacher: a hand-written, from-scratch (NOT jet-algebra-based) linear
2-state Jordan-block system, h_{t+1} = J h_t + b x_t, J=[[lambda,mu],
[0,lambda]] (mu!=0 -- the genuine generalized/repeated-pole coupling),
y_t = c.h_t. This is a completely independent implementation from the
regular-algebra code, so containment is a real test, not a tautology.

Students: LINEAR product-local recurrences (no phi nonlinearity --
clean linear system identification), h_{t+1} = alg_mult_blockwise(
theta, h, Q, d) + B_in*x_t, y_t = C_out.h_t, at matched r, matched
recurrent parameter count P=r (automatic: P=Q*d=r for ANY d, an exact
identity established in B35a-2), matched input dim (1), matched
readout capacity (C_out, B_in both length r, same for d=1 and d=2).

Run: python -m credit_memory.b35b1_mechanism_check
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, project_local_tails
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL

R = 16
T_TRAIN = 64
T_LONG = 512
N_TRAIN = 100
N_VAL = 15
N_TEST = 15
SEEDS = (0, 1, 2)
LR_GRID = (0.003, 0.01, 0.03, 0.1)
RHO_BASE = 0.95   # strictly-stable-pole range
RHO_NIL = 1.0

LAMBDA_T, MU_T = 0.85, 0.30
B_T = jnp.array([1.0, 0.7])   # teacher input vector (b0,b1)
C_T = jnp.array([0.6, 1.0])   # teacher readout vector (c0,c1)
J_T = jnp.array([[LAMBDA_T, MU_T], [0.0, LAMBDA_T]])


# =======================================================================
# Teacher: from-scratch Jordan-block LTI system.
# =======================================================================
def jordan_rollout(h0, xs, J=J_T, b=B_T, c=C_T):
    def step(h, x_t):
        h_next = J @ h + b * x_t
        return h_next, c @ h_next
    _, ys = jax.lax.scan(step, h0, xs)
    return ys


def verify_generalized_mode(T=40):
    """Impulse response regression: y_t ~ (c0_fit + c1_fit*t)*lambda^t.
    Confirm c1_fit is genuinely nonzero (not just numerically ~0)."""
    xs = jnp.zeros(T).at[0].set(1.0)
    ys = np.asarray(jordan_rollout(jnp.zeros(2), xs))
    t_idx = np.arange(T)
    basis = np.stack([LAMBDA_T ** t_idx, t_idx * LAMBDA_T ** t_idx], axis=1)
    coeffs, residuals, rank, sv = np.linalg.lstsq(basis, ys, rcond=None)
    fit = basis @ coeffs
    resid_norm = float(np.max(np.abs(fit - ys)))
    print(f"  impulse response fit to (c0+c1*t)*lambda^t: c0={coeffs[0]:.6f} c1={coeffs[1]:.6f}  "
          f"max|fit-actual|={resid_norm:.2e}  (c1 significantly nonzero: {abs(coeffs[1]) > 1e-3})")
    # pure-semisimple (c1=0) comparison fit for contrast
    basis_ss = basis[:, 0:1]
    coeffs_ss, *_ = np.linalg.lstsq(basis_ss, ys, rcond=None)
    fit_ss = basis_ss @ coeffs_ss
    resid_ss = float(np.max(np.abs(fit_ss - ys)))
    print(f"  pure-semisimple (c1=0) fit residual: max|fit-actual|={resid_ss:.2e}  "
          f"(vs {resid_norm:.2e} with c1 -- {resid_ss/max(resid_norm,1e-12):.1f}x worse)")
    return coeffs[1]


# =======================================================================
# Students.
# =======================================================================
def make_theta_lin(seed, Q, d, rho_base=RHO_BASE):
    rng = np.random.RandomState(seed)
    r = Q * d
    raw = jnp.array(rng.randn(r) * 0.2)
    if d > 1:
        raw = raw.reshape(Q, d).at[:, 0].set(jnp.array(rng.uniform(-0.6, 0.6, size=Q))).reshape(r)
    else:
        raw = jnp.array(rng.uniform(-0.6, 0.6, size=Q))
    return project_local_tails(raw, Q, d, rho_nil=RHO_NIL, rho_base=rho_base)


def make_student_params(seed, Q, d):
    rng = np.random.RandomState(seed + 555)
    r = Q * d
    theta = make_theta_lin(seed, Q, d)
    B_in = jnp.array(rng.randn(r) * (1.0 / np.sqrt(r)))
    C_out = jnp.array(rng.randn(r) * (1.0 / np.sqrt(r)))
    return dict(theta=theta, B_in=B_in, C_out=C_out)


def student_rollout(params, xs, Q, d):
    def step(h, x_t):
        h_next = alg_mult_blockwise(params["theta"], h, Q, d) + params["B_in"] * x_t
        return h_next, params["C_out"] @ h_next
    h0 = jnp.zeros(Q * d, dtype=jnp.float64)
    _, ys = jax.lax.scan(step, h0, xs)
    return ys


def project_params(params, Q, d, rho_base=RHO_BASE):
    return dict(params, theta=project_local_tails(params["theta"], Q, d, rho_nil=RHO_NIL, rho_base=rho_base))


# =======================================================================
# Analytic embedding (d=2 only) -- verified numerically against the
# from-scratch teacher, not assumed.
# =======================================================================
def embed_teacher_into_d2(Q=R // 2, d=2):
    r = Q * d
    theta = jnp.zeros((Q, d)).at[0, 0].set(LAMBDA_T).at[0, 1].set(MU_T).reshape(r)
    B_in = jnp.zeros((Q, d)).at[0, 0].set(B_T[1]).at[0, 1].set(B_T[0]).reshape(r)
    C_out = jnp.zeros((Q, d)).at[0, 0].set(C_T[1]).at[0, 1].set(C_T[0]).reshape(r)
    return dict(theta=theta, B_in=B_in, C_out=C_out)


def verify_exact_embedding(seed=0, T=50):
    print("\n" + "=" * 78)
    print("Analytic embedding of the teacher into the matching d=2 student (BEFORE any SGD)")
    print("=" * 78)
    Q, d = R // 2, 2
    params = embed_teacher_into_d2(Q, d)
    rng = np.random.RandomState(seed)
    xs = jnp.array(rng.randn(T) * 0.5)
    y_student = student_rollout(params, xs, Q, d)
    y_teacher = jordan_rollout(jnp.zeros(2), xs)
    max_diff = float(jnp.max(jnp.abs(y_student - y_teacher)))
    print(f"  max|y_student - y_teacher| over T={T}, random xs: {max_diff:.3e}  (expect ~0)")
    return max_diff < 1e-9


# =======================================================================
# Training (train/val/test split, LR selected on validation only).
# =======================================================================
def make_xs(seed, T):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(T) * 0.5)


def train_one_run(Q, d, lr, seed_init, n_train=N_TRAIN, T=T_TRAIN):
    def loss_fn(params, xs, targets):
        ys = student_rollout(params, xs, Q, d)
        return jnp.mean(0.5 * (ys - targets) ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_student_params(seed_init, Q, d)
    opt_state = adam_init(params)
    diverged, diverged_at = False, None
    for step in range(n_train):
        xs = make_xs(20_000 + step, T)
        targets = jordan_rollout(jnp.zeros(2), xs)
        loss_val = float(loss_jit(params, xs, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        g = clip_grad(grad_fn(params, xs, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project_params(params, Q, d)
    if diverged:
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init, params=params)

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            xs = make_xs(offset + i, T)
            targets = jordan_rollout(jnp.zeros(2), xs)
            losses.append(float(loss_jit(params, xs, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_var = eval_split(95_000, N_VAL)
    test_losses, test_var = eval_split(200_000, N_TEST)
    nmse = float(np.mean(val_losses) / (np.mean(val_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_var) + 1e-12))
    return dict(nmse=nmse, test_nmse=test_nmse, diverged=False, diverged_at=None, lr=lr, seed=seed_init, params=params)


def train_with_grid(Q, d, lr_grid=LR_GRID, seeds=SEEDS):
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run(Q, d, lr, seed_init=1000 + seed)
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
    return dict(status="ok", n_diverged=n_diverged, n_total=len(all_runs), best_lr=best_lr,
                nmse_mean=float(np.mean([r["nmse"] for r in best_runs])),
                test_nmse_mean=float(np.mean([r["test_nmse"] for r in best_runs])),
                best_runs=best_runs)


# =======================================================================
# Long-horizon impulse-response comparison (T_LONG >> T_TRAIN).
# =======================================================================
def impulse_response_comparison(best_runs_d1, best_runs_d2, T_long=T_LONG):
    print("\n" + "=" * 78)
    print(f"Long-horizon impulse-response comparison (T_long={T_long} >> T_train={T_TRAIN})")
    print("=" * 78)
    xs_imp = jnp.zeros(T_long).at[0].set(1.0)
    y_teacher = np.asarray(jordan_rollout(jnp.zeros(2), xs_imp))

    for tag, runs, Q, d in [("d=1", best_runs_d1, R, 1), ("d=2", best_runs_d2, R // 2, 2)]:
        errs_short, errs_long = [], []
        for r in runs:
            y_student = np.asarray(student_rollout(r["params"], xs_imp, Q, d))
            errs_short.append(float(np.max(np.abs(y_student[:T_TRAIN] - y_teacher[:T_TRAIN]))))
            errs_long.append(float(np.max(np.abs(y_student - y_teacher))))
        print(f"  {tag}: max|err| within training horizon (t<{T_TRAIN}) = {np.mean(errs_short):.4e}  "
              f"max|err| full long horizon (t<{T_long}) = {np.mean(errs_long):.4e}")


if __name__ == "__main__":
    print("=" * 78)
    print("Verify the teacher genuinely has a repeated-pole/generalized-mode impulse response")
    print("=" * 78)
    c1 = verify_generalized_mode()

    embed_ok = verify_exact_embedding()
    print(f"EXACT EMBEDDING PASS: {embed_ok}")

    print("\n" + "=" * 78)
    print(f"Training: d=1 (Q={R}) vs d=2 (Q={R//2}), matched r={R}, matched P={R}")
    print("=" * 78)
    res_d1 = train_with_grid(Q=R, d=1)
    res_d2 = train_with_grid(Q=R // 2, d=2)
    print(f"  d=1 (Q={R}, {R} independent spectral sectors):  best_lr={res_d1.get('best_lr')}  "
          f"VAL_NMSE={res_d1.get('nmse_mean')}  TEST_NMSE={res_d1.get('test_nmse_mean')}  "
          f"diverged={res_d1['n_diverged']}/{res_d1['n_total']}")
    print(f"  d=2 (Q={R//2}, {R//2} spectral sectors, each with 1 generalized coupling):  "
          f"best_lr={res_d2.get('best_lr')}  VAL_NMSE={res_d2.get('nmse_mean')}  "
          f"TEST_NMSE={res_d2.get('test_nmse_mean')}  diverged={res_d2['n_diverged']}/{res_d2['n_total']}")

    if res_d1["status"] == "ok" and res_d2["status"] == "ok":
        impulse_response_comparison(res_d1["best_runs"], res_d2["best_runs"])
