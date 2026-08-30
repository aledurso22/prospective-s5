"""B35b Part 2 -- decisive credit test: GenericBlockExactRTRL (p
scalars x p GENERIC dense d x d basis matrices, ordinary/unapproximated
module-wise RTRL, C_local=d*p) vs RegularBlock (our commutative regular
algebra, C_local=p). Both LINEAR (h_{t+1}=A(theta)@h+B_in*x_t), for a
clean, unconfounded credit-accounting comparison -- consistent with
Part 1's linear scope.

View 2A: matched architecture size (same d,p,Q,r,P) -- accounting +
correctness only, no performance claim.
View 2B: matched exact-credit budget C, via a PREDECLARED rule (fixed
BEFORE running/seeing results): at fixed local d, RegularBlock's
per-module credit is p=d, GenericBlock's is d*p=d^2 -- so at the same
total C, RegularBlock gets exactly d times more modules (hence d times
more r and P) than GenericBlock. Compared on the Part-1 generalized-
mode teacher AND one neutral (non-algebra) system-identification
teacher.

Run: python -m credit_memory.b35b2_generic_vs_regular
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, project_local_tails
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL
from credit_memory.b35b1_mechanism_check import jordan_rollout, J_T, B_T, C_T, RHO_BASE, RHO_NIL

D_LOCAL = 4          # fixed local factor size, both families
CREDIT_C = 64         # predeclared common exact-credit budget
N_TRAIN, N_VAL, N_TEST = 100, 15, 15
T_TRAIN = 64
SEEDS = (0, 1, 2)
LR_GRID = (0.003, 0.01, 0.03, 0.1)


# =======================================================================
# GenericBlock: p trainable scalars x p GENERIC (non-regular) dense
# d x d basis matrices, per module (independently drawn -- no shared
# structure that could accidentally create cross-module regularity).
# =======================================================================
def make_generic_basis(seed, Q, d, p):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(Q, p, d, d) / np.sqrt(d))


def make_generic_params(seed, Q, d, p, stable_radius=0.9):
    """theta initialized so A(theta) starts with spectral radius <
    stable_radius (checked per module) -- a fair, non-adversarial init,
    analogous to RegularBlock's base-coefficient clip."""
    rng = np.random.RandomState(seed)
    basis = make_generic_basis(seed + 10_000, Q, d, p)
    theta = np.zeros((Q, p))
    for q in range(Q):
        for _ in range(200):
            cand = rng.randn(p) * 0.3
            A = np.einsum("k,kij->ij", cand, np.asarray(basis[q]))
            if np.max(np.abs(np.linalg.eigvals(A))) < stable_radius:
                theta[q] = cand
                break
        else:
            theta[q] = cand * 0.1  # fallback, heavily damped
    return jnp.array(theta.reshape(Q * p)), basis


def generic_forward_batched(h_flat, theta_flat, basis, b_in_flat, x_t, Q, d, p):
    H = h_flat.reshape(Q, d)
    Theta = theta_flat.reshape(Q, p)
    A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
    H_next = jnp.einsum("qij,qj->qi", A_batch, H) + b_in_flat.reshape(Q, d) * x_t
    return H_next.reshape(Q * d)


def generic_rollout(h0, theta_flat, basis, b_in_flat, xs, Q, d, p):
    def step(h, x_t):
        h_next = generic_forward_batched(h, theta_flat, basis, b_in_flat, x_t, Q, d, p)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs


def generic_exact_module_rtrl_step(h_q, S_q, theta_q, basis_q, x_t, b_in_q):
    """ONE module, exact/unapproximated: S_q in R^{d,p}."""
    A_q = jnp.einsum("k,kij->ij", theta_q, basis_q)
    h_next = A_q @ h_q + b_in_q * x_t
    G_q = jnp.einsum("kij,j->ik", basis_q, h_q)   # G_q[:,k] = B_k @ h_q
    S_next = A_q @ S_q + G_q
    return h_next, S_next


_generic_rtrl_batched = jax.vmap(generic_exact_module_rtrl_step, in_axes=(0, 0, 0, 0, None, 0))


def generic_module_rtrl_grad(theta_flat, basis, b_in_flat, h0, xs, qs, Q, d, p):
    T = xs.shape[0]
    H = h0.reshape(Q, d)
    S = jnp.zeros((Q, d, p), dtype=jnp.float64)
    Theta = theta_flat.reshape(Q, p)
    B_in = b_in_flat.reshape(Q, d)
    g_total = jnp.zeros((Q, p), dtype=jnp.float64)
    S_traj = []
    for t in range(T):
        H_next, S_next = _generic_rtrl_batched(H, S, Theta, basis, xs[t], B_in)
        S_traj.append(S_next)
        dl_dh = qs[t].reshape(Q, d)   # upstream grad, per module
        g_total = g_total + jnp.einsum("qd,qdk->qk", dl_dh, S_next)
        H, S = H_next, S_next
    return g_total.reshape(Q * p), jnp.stack(S_traj)


# =======================================================================
# RegularBlock: LINEAR (h_{t+1}=alg_mult(theta,h)+B_in*x_t). Reduced
# eligibility s_{t+1}=theta*s_t+h_t (derived: G_t=M_{h_t} exactly, by
# commutativity M_u@v=M_v@u -- verified numerically below, not assumed).
# =======================================================================
def regular_rollout(h0, theta, b_in, xs, Q, d):
    def step(h, x_t):
        h_next = alg_mult_blockwise(theta, h, Q, d) + b_in * x_t
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs


def regular_reduced_grad(theta, b_in, h0, xs, qs, Q, d):
    T = xs.shape[0]
    h = h0
    s = jnp.zeros(Q * d, dtype=jnp.float64)
    g_total = jnp.zeros(Q * d, dtype=jnp.float64)
    s_traj = []
    for t in range(T):
        h_next = alg_mult_blockwise(theta, h, Q, d) + b_in * xs[t]
        s_next = alg_mult_blockwise(theta, s, Q, d) + h        # uses PRE-update h, per derivation
        s_traj.append(s_next)
        g_total = g_total + qs[t] * 0  # placeholder overwritten below via transpose action
        h, s = h_next, s_next
    return None, jnp.stack(s_traj)   # gradient contraction done separately (needs M_s^T action)


# =======================================================================
# View 2A -- matched architecture size: correctness + actual accounting.
# =======================================================================
def make_setting(seed, T, Q, d):
    rng = np.random.RandomState(seed)
    r = Q * d
    h0 = jnp.array(rng.randn(r) * 0.1)
    xs = jnp.array(rng.randn(T) * 0.4)
    qs = jnp.array(rng.randn(T, r) * 0.5)
    return h0, xs, qs


def run_view2a(Q=4, d=D_LOCAL, p=D_LOCAL, T=15, seeds=(0, 1, 2)):
    print("=" * 78)
    print(f"View 2A: matched architecture, Q={Q} d={d} p={p} (r={Q*d}, P={Q*p})")
    print("=" * 78)

    # ---- GenericBlock ----
    theta_g, basis = make_generic_params(seed=1, Q=Q, d=d, p=p)
    rng = np.random.RandomState(2)
    b_in_g = jnp.array(rng.randn(Q * d) * 0.3)

    def loss_generic(theta_flat, h0, xs, qs):
        Hs = generic_rollout(h0, theta_flat, basis, b_in_g, xs, Q, d, p)
        return jnp.sum(Hs * qs)

    grad_bptt_generic = jax.jit(jax.grad(loss_generic, argnums=0))

    worst_generic = 0.0
    worst_rank_deficit = 0
    for seed in seeds:
        h0, xs, qs = make_setting(seed, T, Q, d)
        g_b = grad_bptt_generic(theta_g, h0, xs, qs)
        g_r, S_traj = generic_module_rtrl_grad(theta_g, basis, b_in_g, h0, xs, qs, Q, d, p)
        rel = float(jnp.linalg.norm(g_r - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
        worst_generic = max(worst_generic, rel)
        # rank of S_t per module (generic max = min(d,p))
        S_np = np.asarray(S_traj[-1])   # (Q,d,p) at final step
        for q in range(Q):
            rank_q = np.linalg.matrix_rank(S_np[q], tol=1e-9)
            worst_rank_deficit = max(worst_rank_deficit, min(d, p) - rank_q)
        print(f"  GenericBlock seed={seed}: grad_rel_err={rel:.3e}  "
              f"sample module-0 rank(S_T)={np.linalg.matrix_rank(S_np[0], tol=1e-9)}/{min(d,p)}")
    print(f"  GenericBlock WORST grad_rel_err={worst_generic:.3e}  "
          f"worst rank deficit from generic max: {worst_rank_deficit} (0 = no accidental collapse)")

    actual_generic_credit = int(np.prod(S_traj[0].shape))  # (Q,d,p) -> Q*d*p, ACTUAL array
    symbolic_generic_credit = Q * d * p
    print(f"  GenericBlock actual persistent S array size={actual_generic_credit}  "
          f"symbolic Q*d*p={symbolic_generic_credit}  MATCH={actual_generic_credit==symbolic_generic_credit}")

    # ---- RegularBlock ----
    rng = np.random.RandomState(3)
    theta_r = project_local_tails(jnp.array(rng.randn(Q * d) * 0.3), Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
    b_in_r = jnp.array(rng.randn(Q * d) * 0.3)

    def loss_regular(theta, h0, xs, qs):
        Hs = regular_rollout(h0, theta, b_in_r, xs, Q, d)
        return jnp.sum(Hs * qs)

    grad_bptt_regular = jax.jit(jax.grad(loss_regular, argnums=0))

    from credit_memory.b35a_product_local_algebra import transpose_mult_blockwise
    worst_regular = 0.0
    for seed in seeds:
        h0, xs, qs = make_setting(seed, T, Q, d)
        g_b = grad_bptt_regular(theta_r, h0, xs, qs)
        _, s_traj = regular_reduced_grad(theta_r, b_in_r, h0, xs, qs, Q, d)
        g_r = jnp.zeros(Q * d, dtype=jnp.float64)
        for t in range(T):
            g_r = g_r + transpose_mult_blockwise(s_traj[t], qs[t], Q, d)
        rel = float(jnp.linalg.norm(g_r - g_b) / (jnp.linalg.norm(g_b) + 1e-12))
        worst_regular = max(worst_regular, rel)
        print(f"  RegularBlock  seed={seed}: grad_rel_err={rel:.3e}")
    print(f"  RegularBlock WORST grad_rel_err={worst_regular:.3e}")

    actual_regular_credit = int(s_traj[0].shape[0])   # (r,) ACTUAL array
    symbolic_regular_credit = Q * p   # = Q*d since p=d
    print(f"  RegularBlock actual persistent s array size={actual_regular_credit}  "
          f"symbolic Q*p={symbolic_regular_credit}  MATCH={actual_regular_credit==symbolic_regular_credit}")

    factor = symbolic_generic_credit / symbolic_regular_credit
    print(f"\n  Factor-d gap: GenericBlock credit / RegularBlock credit = {factor:.1f}  (predicted d={d})")
    ok = (worst_generic < 1e-8 and worst_regular < 1e-8 and
          actual_generic_credit == symbolic_generic_credit and
          actual_regular_credit == symbolic_regular_credit and
          worst_rank_deficit == 0 and abs(factor - d) < 1e-9)
    print(f"VIEW 2A PASS: {ok}")
    return ok


# =======================================================================
# View 2B -- matched exact-credit budget C, predeclared rule.
# =======================================================================
def make_dense_linear_teacher(seed, state_dim=8, radius=0.85):
    rng = np.random.RandomState(seed)
    M = rng.randn(state_dim, state_dim) / np.sqrt(state_dim)
    eig = np.max(np.abs(np.linalg.eigvals(M)))
    A = jnp.array(M * (radius / eig))
    b = jnp.array(rng.randn(state_dim))
    c = jnp.array(rng.randn(state_dim))

    def rollout(h0, xs):
        def step(h, x_t):
            h_next = A @ h + b * x_t
            return h_next, c @ h_next
        _, ys = jax.lax.scan(step, h0, xs)
        return ys
    return rollout, state_dim


def make_xs(seed, T):
    rng = np.random.RandomState(seed)
    return jnp.array(rng.randn(T) * 0.4)


def project_generic_params(params, basis, Q, d, p, rho_max=0.95):
    theta = params["theta"].reshape(Q, p)
    A_batch = jnp.einsum("qk,qkij->qij", theta, basis)
    eigmax = jnp.max(jnp.abs(jnp.linalg.eigvals(A_batch)), axis=1)
    scale = jnp.where(eigmax > rho_max, rho_max / (eigmax + 1e-12), 1.0)
    theta_new = (theta * scale[:, None]).reshape(Q * p)
    return dict(params, theta=theta_new)


def project_regular_params(params, Q, d):
    return dict(params, theta=project_local_tails(params["theta"], Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE))


def make_generic_student(seed_arch, Q, d, p):
    _, basis = make_generic_params(seed_arch, Q, d, p)

    def make_params(seed):
        rng = np.random.RandomState(seed + 555)
        theta0, _ = make_generic_params(seed, Q, d, p)
        b_in = jnp.array(rng.randn(Q * d) * (1.0 / np.sqrt(Q * d)))
        C_out = jnp.array(rng.randn(Q * d) * (1.0 / np.sqrt(Q * d)))
        return dict(theta=theta0, b_in=b_in, C_out=C_out)

    def rollout_y(params, xs):
        h0 = jnp.zeros(Q * d, dtype=jnp.float64)
        Hs = generic_rollout(h0, params["theta"], basis, params["b_in"], xs, Q, d, p)
        return Hs @ params["C_out"]

    def project(params):
        return project_generic_params(params, basis, Q, d, p)

    return rollout_y, make_params, project


def make_regular_student(Q, d):
    def make_params(seed):
        rng = np.random.RandomState(seed + 555)
        theta0 = project_local_tails(jnp.array(rng.randn(Q * d) * 0.2), Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
        b_in = jnp.array(rng.randn(Q * d) * (1.0 / np.sqrt(Q * d)))
        C_out = jnp.array(rng.randn(Q * d) * (1.0 / np.sqrt(Q * d)))
        return dict(theta=theta0, b_in=b_in, C_out=C_out)

    def rollout_y(params, xs):
        h0 = jnp.zeros(Q * d, dtype=jnp.float64)
        Hs = regular_rollout(h0, params["theta"], params["b_in"], xs, Q, d)
        return Hs @ params["C_out"]

    def project(params):
        return project_regular_params(params, Q, d)

    return rollout_y, make_params, project


def train_one_run_2b(rollout_y, make_params, project, teacher_rollout, teacher_state_dim, lr, seed_init,
                      n_train=N_TRAIN, T=T_TRAIN):
    def loss_fn(params, xs, targets):
        ys = rollout_y(params, xs)
        return jnp.mean(0.5 * (ys - targets) ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_params(seed_init)
    params = project(params)
    opt_state = adam_init(params)
    diverged, diverged_at = False, None
    for step in range(n_train):
        xs = make_xs(20_000 + step, T)
        targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
        loss_val = float(loss_jit(params, xs, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        g = clip_grad(grad_fn(params, xs, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project(params)
    if diverged:
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init)

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            xs = make_xs(offset + i, T)
            targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
            losses.append(float(loss_jit(params, xs, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_var = eval_split(95_000, N_VAL)
    test_losses, test_var = eval_split(200_000, N_TEST)
    if not all(np.isfinite(val_losses)) or not all(np.isfinite(test_losses)):
        return dict(nmse=None, test_nmse=None, diverged=True, diverged_at=n_train, lr=lr, seed=seed_init)
    nmse = float(np.mean(val_losses) / (np.mean(val_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_var) + 1e-12))
    return dict(nmse=nmse, test_nmse=test_nmse, diverged=False, diverged_at=None, lr=lr, seed=seed_init)


def train_with_grid_2b(rollout_y, make_params, project, teacher_rollout, teacher_state_dim,
                        lr_grid=LR_GRID, seeds=SEEDS):
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run_2b(rollout_y, make_params, project, teacher_rollout, teacher_state_dim,
                                    lr, seed_init=1000 + seed)
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
                test_nmse_mean=float(np.mean([r["test_nmse"] for r in best_runs])))


def run_view2b(d=D_LOCAL, C=CREDIT_C):
    p = d
    Q_reg = C // p
    Q_gen = C // (d * p)
    r_reg, P_reg = Q_reg * d, Q_reg * p
    r_gen, P_gen = Q_gen * d, Q_gen * p
    print("\n" + "=" * 78)
    print(f"View 2B: matched exact-credit budget C={C} (predeclared rule, d={d} fixed)")
    print(f"  RegularBlock: Q={Q_reg} r={r_reg} P={P_reg} credit={Q_reg*p}")
    print(f"  GenericBlock: Q={Q_gen} r={r_gen} P={P_gen} credit={Q_gen*d*p}")
    print("=" * 78)

    reg_rollout, reg_make_params, reg_project = make_regular_student(Q_reg, d)
    gen_rollout, gen_make_params, gen_project = make_generic_student(42, Q_gen, d, p)

    teachers = [
        ("JordanGeneralizedMode", jordan_rollout, 2),
        ("NeutralDenseLinear", make_dense_linear_teacher(seed=999, state_dim=8)[0], 8),
    ]

    results = {}
    for name, teacher_rollout, tdim in teachers:
        print(f"\n  --- teacher: {name} ---")
        res_reg = train_with_grid_2b(reg_rollout, reg_make_params, reg_project, teacher_rollout, tdim)
        res_gen = train_with_grid_2b(gen_rollout, gen_make_params, gen_project, teacher_rollout, tdim)
        results[name] = dict(regular=res_reg, generic=res_gen)
        print(f"    RegularBlock (r={r_reg},P={P_reg}): {res_reg}")
        print(f"    GenericBlock (r={r_gen},P={P_gen}): {res_gen}")
    return results


if __name__ == "__main__":
    ok_2a = run_view2a()
    if ok_2a:
        run_view2b()
    else:
        print("STOPPING: View 2A correctness/accounting failed -- repair before View 2B.")
