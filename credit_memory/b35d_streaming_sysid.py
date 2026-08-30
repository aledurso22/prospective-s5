"""B35d -- one targeted streaming (true online, no BPTT window, no
replay) system-identification application test. Architecture frozen:
reuses RegularBlock/ProductLocal (b35a_product_local_algebra.py),
GenericBlock (b35b2_generic_vs_regular.py), and RTU's OWN existing
per-step streaming exact-RTRL machinery (b28_rtu_faithful.py's
rtu_streaming_step) completely unmodified.

Scientific question: at fixed persistent exact-credit memory, can
RegularBlock use its saved credit budget for better CONTINUAL online
system identification than GenericBlock exact RTRL (RTU as a secondary
exact-RTRL reference, reused as-is)?

Protocol per step t: observe (x_t, u_t) -> predict x_hat_{t+1} -> observe
true x_{t+1} -> loss -> ONE exact-online parameter update (persistent
eligibility trace carried across the WHOLE stream, never reset,
no windowing) -> continue.

Teacher: a hand-written, INDEPENDENT 4-state linear system (NOT built
from ProductLocal/GenericBlock code) = blockdiag(oscillatory 2x2 block
rho^t*cos(omega*t+phase), repeated-pole 2x2 Jordan block
(c0+c1*t)*lambda^t). One regime change (rho, lambda both shift) at a
predeclared timestep, testing continual adaptation.

Design choice (stated explicitly, not hidden): each architecture's own
INPUT COUPLING (fixed random (r,5) matrix mapping [x_t,u_t] into the
recurrence) is FROZEN for RegularBlock/GenericBlock -- extending their
existing validated exact-RTRL formalism (theta only) to also credit an
input matrix would be new engineering beyond what B35a-c validated.
RTU's OWN existing rtu_streaming_step ALREADY trains its input
coupling (B_real/B_imag) as part of its native exact-RTRL scope, so it
is reused exactly as-is (features=5) -- this is a genuine asymmetry
between architectures, noted as a confound, not smoothed over. Only
the RECURRENT parameters (theta for Regular/Generic; all 4 RTU
families for RTU) are updated via each architecture's own EXISTING
exact-RTRL machinery; the linear readout C_out (pure output map, no
feedback into the recurrence) is updated via a direct per-step gradient
(no eligibility needed for a readout).

Run: python -m credit_memory.b35d_streaming_sysid
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

jax.config.update("jax_enable_x64", True)

from credit_memory.b35a_product_local_algebra import alg_mult_blockwise, project_local_tails
from credit_memory.b35b1_mechanism_check import RHO_BASE, RHO_NIL
from credit_memory.b35b2_generic_vs_regular import make_generic_params, generic_exact_module_rtrl_step
from credit_memory.b35c_matched_credit_frontier import regular_config, generic_config, rtu_config
from credit_memory.b28_rtu_faithful import make_rtu_params, rtu_streaming_init, rtu_streaming_step, RTU_FAMILIES

D_LOCAL = 4
BUDGETS = (128, 64)
SEEDS = (0, 1, 2)
T_TOTAL = 3000
T_CHANGE = 1500
LR_THETA = 0.02
LR_COUT = 0.02
CLIP_NORM = 5.0
U_SCALE = 0.3
X_DIM_TEACHER = 4
RECOVERY_THRESHOLD_MULT = 1.5

# ---------------------------------------------------------------------
# Teacher: hand-written, INDEPENDENT 4-state linear system. Verified by
# inspection: no import from b35a_product_local_algebra.py anywhere in
# this block.
# ---------------------------------------------------------------------
def make_A(rho, omega, lam, mu):
    R = rho * jnp.array([[jnp.cos(omega), -jnp.sin(omega)], [jnp.sin(omega), jnp.cos(omega)]])
    J = jnp.array([[lam, mu], [0.0, lam]])
    return block_diag(R, J)


A_PRE = make_A(rho=0.80, omega=0.6, lam=0.85, mu=0.30)
A_POST = make_A(rho=0.50, omega=0.6, lam=0.55, mu=0.30)
B_TEACHER = jnp.array([1.0, 0.5, 0.8, 0.6])


def teacher_step(x, u, t):
    A = jnp.where(t < T_CHANGE, A_PRE, A_POST)
    return A @ x + B_TEACHER * u


def make_teacher_trajectory(seed, T_total=T_TOTAL):
    rng = np.random.RandomState(seed)
    us = jnp.array(rng.randn(T_total) * U_SCALE)

    def step(x, inputs):
        u, t = inputs
        x_next = teacher_step(x, u, t)
        return x_next, x_next

    x0 = jnp.zeros(X_DIM_TEACHER, dtype=jnp.float64)
    t_idx = jnp.arange(T_total)
    _, xs_next_all = jax.lax.scan(step, x0, (us, t_idx))
    xs_t = jnp.concatenate([x0[None, :], xs_next_all[:-1]], axis=0)
    return xs_t, us, xs_next_all


from credit_memory.b35a_product_local_algebra import transpose_mult_blockwise
from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step


def clip_vec(g, max_norm=CLIP_NORM):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return g * scale


def project_generic_theta(theta_flat, basis, Q, d, p, rho_max=0.95):
    theta = theta_flat.reshape(Q, p)
    A_batch = jnp.einsum("qk,qkij->qij", theta, basis)
    eig = jnp.max(jnp.abs(jnp.linalg.eigvals(A_batch)), axis=1)
    scale = jnp.where(eig > rho_max, rho_max / (eig + 1e-12), 1.0)
    return (theta * scale[:, None]).reshape(Q * p)


# ---------------------------------------------------------------------
# RegularBlock streaming step (scanned, jitted). theta trained via its
# OWN existing exact-RTRL (reduced eligibility s_{t+1}=alg_mult(theta,
# s_t)+h_t, unaffected by the FIXED additive B_in term). C_out trained
# by a direct local gradient (pure readout, no eligibility needed).
# ---------------------------------------------------------------------
def make_regular_streaming_scan(Q, d, B_in, lr_theta=LR_THETA, lr_cout=LR_COUT):
    def step(carry, inputs):
        h, s, theta, C_out, opt_th, opt_co = carry
        x_t, u_t, x_next = inputs
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * jnp.sum(diff ** 2)
        dl_dh = C_out.T @ diff
        g_theta = clip_vec(transpose_mult_blockwise(s_next, dl_dh, Q, d))
        g_cout = clip_vec((jnp.outer(diff, h_next)).reshape(-1)).reshape(C_out.shape)

        theta_new, opt_th_new = adam_step(theta, g_theta, opt_th, lr_theta)
        theta_new = project_local_tails(theta_new, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
        C_out_new, opt_co_new = adam_step(C_out, g_cout, opt_co, lr_cout)

        finite = jnp.all(jnp.isfinite(h_next)) & jnp.isfinite(loss)
        new_carry = (h_next, s_next, theta_new, C_out_new, opt_th_new, opt_co_new)
        return new_carry, (loss, x_hat, finite)
    return step


# ---------------------------------------------------------------------
# GenericBlock streaming step (scanned, jitted). theta trained via its
# OWN existing exact module-wise RTRL (unapproximated).
# ---------------------------------------------------------------------
def make_generic_streaming_scan(Q, d, p, basis, B_in, lr_theta=LR_THETA, lr_cout=LR_COUT):
    def step(carry, inputs):
        h, S, theta, C_out, opt_th, opt_co = carry
        x_t, u_t, x_next = inputs
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        H = h.reshape(Q, d)
        Theta = theta.reshape(Q, p)
        A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
        H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
        G_batch = jnp.einsum("qkij,qj->qik", basis, H)
        S_next = jnp.einsum("qij,qjk->qik", A_batch, S) + G_batch
        h_next = H_next.reshape(Q * d)
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        loss = 0.5 * jnp.sum(diff ** 2)
        dl_dh = (C_out.T @ diff).reshape(Q, d)
        g_theta = clip_vec(jnp.einsum("qd,qdk->qk", dl_dh, S_next).reshape(-1))
        g_cout = clip_vec((jnp.outer(diff, h_next)).reshape(-1)).reshape(C_out.shape)

        theta_new, opt_th_new = adam_step(theta, g_theta, opt_th, lr_theta)
        theta_new = project_generic_theta(theta_new, basis, Q, d, p)
        C_out_new, opt_co_new = adam_step(C_out, g_cout, opt_co, lr_cout)

        finite = jnp.all(jnp.isfinite(h_next)) & jnp.isfinite(loss)
        new_carry = (h_next, S_next, theta_new, C_out_new, opt_th_new, opt_co_new)
        return new_carry, (loss, x_hat, finite)
    return step


def run_scanned_stream(step_fn, init_carry, xs_t, us_t, xs_next, T_total=T_TOTAL):
    inputs = (xs_t, us_t, xs_next)
    scan_fn = jax.jit(lambda carry, inputs: jax.lax.scan(step_fn, carry, inputs))
    final_carry, (losses, x_hats, finites) = scan_fn(init_carry, inputs)
    return losses, x_hats, finites


def verify_teacher_impulse_response(T=40):
    """Sanity: confirm the teacher genuinely contains BOTH rho^t*cos and
    (c0+c1 t) lambda^t components (pre-change matrix). Impulse each
    block on its OWN coordinate -- the two blocks are block-diagonal
    (no cross-coupling), so an impulse restricted to the oscillatory
    block never excites the repeated-pole block at all."""
    def rollout(x0, T):
        x = x0
        xs = [x]
        for t in range(T - 1):
            x = A_PRE @ x
            xs.append(x)
        return jnp.stack(xs)

    xs_osc = rollout(jnp.zeros(4).at[0].set(1.0), T)     # impulse on the OSCILLATORY block
    xs_rep = rollout(jnp.zeros(4).at[3].set(1.0), T)     # impulse on index 3 (the block's "b" coordinate,
    # which forces index 2 -- J=[[lam,mu],[0,lam]] means index 2 is the one that RECEIVES the generalized
    # coupling from index 3; impulsing index 2 itself gives pure lam^t decay with no growing term.

    t_idx = np.arange(T)
    rho, omega, lam = 0.80, 0.6, 0.85
    basis_osc = np.stack([rho ** t_idx * np.cos(omega * t_idx), rho ** t_idx * np.sin(omega * t_idx)], axis=1)
    coeffs_osc, *_ = np.linalg.lstsq(basis_osc, np.asarray(xs_osc[:, 0]), rcond=None)
    fit_osc = basis_osc @ coeffs_osc
    err_osc = float(np.max(np.abs(fit_osc - np.asarray(xs_osc[:, 0]))))

    basis_rep = np.stack([lam ** t_idx, t_idx * lam ** t_idx], axis=1)
    coeffs_rep, *_ = np.linalg.lstsq(basis_rep, np.asarray(xs_rep[:, 2]), rcond=None)
    fit_rep = basis_rep @ coeffs_rep
    err_rep = float(np.max(np.abs(fit_rep - np.asarray(xs_rep[:, 2]))))

    print(f"  osc-block coord fit to rho^t*(a*cos+b*sin): max|err|={err_osc:.2e} (expect ~0)")
    print(f"  rep-pole-block coord fit to (c0+c1 t) lambda^t: max|err|={err_rep:.2e} (expect ~0), c1={coeffs_rep[1]:.4f}")
    return err_osc < 1e-8 and err_rep < 1e-8 and abs(coeffs_rep[1]) > 1e-3


# =======================================================================
# Per-architecture initialization.
# =======================================================================
def init_regular_streaming(Q, d, seed):
    rng = np.random.RandomState(seed)
    theta0 = project_local_tails(jnp.array(rng.randn(Q * d) * 0.2), Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
    B_in = jnp.array(rng.randn(Q * d, X_DIM_TEACHER + 1) / np.sqrt(X_DIM_TEACHER + 1))
    C_out = jnp.array(rng.randn(X_DIM_TEACHER, Q * d) * (1.0 / np.sqrt(Q * d)))
    h0 = jnp.zeros(Q * d, dtype=jnp.float64)
    s0 = jnp.zeros(Q * d, dtype=jnp.float64)
    opt_th, opt_co = adam_init(theta0), adam_init(C_out)
    return (h0, s0, theta0, C_out, opt_th, opt_co), B_in


def init_generic_streaming(Q, d, p, seed):
    theta0, basis = make_generic_params(seed=seed, Q=Q, d=d, p=p)
    rng = np.random.RandomState(seed + 777)
    B_in = jnp.array(rng.randn(Q * d, X_DIM_TEACHER + 1) / np.sqrt(X_DIM_TEACHER + 1))
    C_out = jnp.array(rng.randn(X_DIM_TEACHER, Q * d) * (1.0 / np.sqrt(Q * d)))
    h0 = jnp.zeros(Q * d, dtype=jnp.float64)
    S0 = jnp.zeros((Q, d, p), dtype=jnp.float64)
    opt_th, opt_co = adam_init(theta0), adam_init(C_out)
    return (h0, S0, theta0, C_out, opt_th, opt_co), basis, B_in


def run_rtu_stream(hidden_dim, xs_t, us_t, xs_next, seed, T_total=T_TOTAL, lr_theta=LR_THETA, lr_cout=LR_COUT):
    """Reuses rtu_streaming_step (b28_rtu_faithful.py) UNMODIFIED, plus
    the SAME sensitivity-contraction pattern already validated in that
    module's own test_rtu_streaming_vs_bptt (np.einsum("ih,ih...->h...",
    dLdout_split, next_S[fam])) -- applied per-step online instead of
    accumulated-then-once. No new RTRL engineering."""
    rng = np.random.RandomState(seed)
    params = make_rtu_params(rng, hidden_dim, X_DIM_TEACHER + 1)
    stream_state = dict(real=np.zeros(hidden_dim), imag=np.zeros(hidden_dim),
                         S=rtu_streaming_init(hidden_dim, X_DIM_TEACHER + 1))
    C_out = jnp.array(rng.randn(X_DIM_TEACHER, 2 * hidden_dim) * (1.0 / np.sqrt(2 * hidden_dim)))
    opt_rtu, opt_co = adam_init(params), adam_init(C_out)

    losses, x_hats, finites = [], [], []
    xs_t_np, us_np, xs_next_np = np.asarray(xs_t), np.asarray(us_t), np.asarray(xs_next)
    for t in range(T_total):
        model_in = jnp.array(np.concatenate([xs_t_np[t], [us_np[t]]]))
        output, next_S = rtu_streaming_step(params, stream_state, model_in)
        h_next = np.asarray(output)
        x_hat = np.asarray(C_out) @ h_next
        diff = x_hat - xs_next_np[t]
        loss = 0.5 * float(np.sum(diff ** 2))
        finite = bool(np.all(np.isfinite(h_next))) and np.isfinite(loss)
        losses.append(loss); x_hats.append(x_hat); finites.append(finite)
        if not finite:
            break
        dl_dh = np.asarray(C_out).T @ diff
        dl_split = np.stack([dl_dh[:hidden_dim], dl_dh[hidden_dim:]], axis=0)
        grads = {fam: jnp.array(np.einsum("ih,ih...->h...", dl_split, next_S[fam])) for fam in RTU_FAMILIES}
        grads = {fam: clip_vec(grads[fam].reshape(-1)).reshape(grads[fam].shape) for fam in RTU_FAMILIES}
        params, opt_rtu = adam_step(params, grads, opt_rtu, lr_theta)
        g_cout = clip_vec(jnp.outer(jnp.array(diff), jnp.array(h_next)).reshape(-1)).reshape(C_out.shape)
        C_out, opt_co = adam_step(C_out, g_cout, opt_co, lr_cout)
    return jnp.array(losses), jnp.array(x_hats), jnp.array(finites)


# =======================================================================
# Metrics.
# =======================================================================
def compute_metrics(losses, x_hats, finites, xs_next, T_change=T_CHANGE,
                     pre_window=(800, 1500), post_window=(2300, 3000), threshold_mult=RECOVERY_THRESHOLD_MULT):
    n = int(losses.shape[0])
    finites_np = np.asarray(finites)
    n_nonfinite = int(np.sum(~finites_np))
    diverged = n < T_TOTAL or n_nonfinite > 0
    losses_np = np.asarray(losses)
    xs_next_np = np.asarray(xs_next)[:n]

    def window_nmse(a, b):
        a, b = max(0, a), min(n, b)
        if b <= a:
            return None
        seg_loss = np.mean(losses_np[a:b])
        seg_var = np.mean(xs_next_np[a:b] ** 2)
        return float(seg_loss / (seg_var + 1e-12))

    pre_nmse = window_nmse(*pre_window)
    post_nmse = window_nmse(*post_window)
    cum_loss = float(np.sum(losses_np))

    # trailing-window (width 20) online loss curve, for plotting/recovery
    W = 20
    trailing = np.convolve(losses_np, np.ones(W) / W, mode="valid")
    trailing_t = np.arange(W - 1, n)

    steps_to_recover = None
    if pre_nmse is not None:
        thresh_loss = pre_nmse * threshold_mult * np.mean(xs_next_np[max(0, T_change - 200):T_change] ** 2)
        post_start = T_change
        idxs = np.where(trailing_t >= post_start)[0]
        for i in idxs:
            if trailing[i] < thresh_loss:
                steps_to_recover = int(trailing_t[i] - post_start)
                break

    return dict(diverged=diverged, n_nonfinite=n_nonfinite, n_steps_completed=n, cum_loss=cum_loss,
                pre_nmse=pre_nmse, post_nmse=post_nmse, steps_to_recover=steps_to_recover,
                trailing_curve=(trailing_t.tolist(), trailing.tolist()))


# =======================================================================
# Validity checks (short frozen sequence, this NEW wiring specifically).
# =======================================================================
def validity_check_regular(Q=8, d=4, T=12, seed=0):
    print("  [RegularBlock] reduced-RTRL vs BPTT, THIS streaming wiring (concat[x,u] input, fixed B_in):")
    rng = np.random.RandomState(seed)
    theta = project_local_tails(jnp.array(rng.randn(Q * d) * 0.2), Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE)
    B_in = jnp.array(rng.randn(Q * d, 5) / np.sqrt(5))
    C_out = jnp.array(rng.randn(4, Q * d) * 0.1)
    xs_t, us, xs_next = make_teacher_trajectory(seed=999, T_total=T)

    def loss_bptt(theta):
        def step(h, inputs):
            x_t, u_t, x_next = inputs
            model_in = jnp.concatenate([x_t, jnp.array([u_t])])
            h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
            x_hat = C_out @ h_next
            diff = x_hat - x_next
            return h_next, 0.5 * jnp.sum(diff ** 2)
        h0 = jnp.zeros(Q * d, dtype=jnp.float64)
        _, losses = jax.lax.scan(step, h0, (xs_t, us, xs_next))
        return jnp.sum(losses)

    g_bptt = jax.grad(loss_bptt)(theta)

    step_fn = make_regular_streaming_scan(Q, d, B_in)
    h0, s0 = jnp.zeros(Q * d, dtype=jnp.float64), jnp.zeros(Q * d, dtype=jnp.float64)
    opt_th, opt_co = adam_init(theta), adam_init(C_out)
    h, s = h0, s0
    g_online_total = jnp.zeros(Q * d, dtype=jnp.float64)
    for t in range(T):
        x_t, u_t, x_next = xs_t[t], us[t], xs_next[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        h_next = alg_mult_blockwise(theta, h, Q, d) + B_in @ model_in
        s_next = alg_mult_blockwise(theta, s, Q, d) + h
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        dl_dh = C_out.T @ diff
        g_online_total = g_online_total + transpose_mult_blockwise(s_next, dl_dh, Q, d)
        h, s = h_next, s_next
    rel = float(jnp.linalg.norm(g_online_total - g_bptt) / (jnp.linalg.norm(g_bptt) + 1e-12))
    print(f"    relative grad error (accumulated online-RTRL vs BPTT over T={T}): {rel:.3e}")
    return rel < 1e-8


def validity_check_generic(Q=2, d=4, p=4, T=12, seed=0):
    print("  [GenericBlock] exact module-wise RTRL vs BPTT, THIS streaming wiring:")
    theta, basis = make_generic_params(seed=seed, Q=Q, d=d, p=p)
    rng = np.random.RandomState(seed + 1)
    B_in = jnp.array(rng.randn(Q * d, 5) / np.sqrt(5))
    C_out = jnp.array(rng.randn(4, Q * d) * 0.1)
    xs_t, us, xs_next = make_teacher_trajectory(seed=999, T_total=T)

    def loss_bptt(theta):
        def step(h, inputs):
            x_t, u_t, x_next = inputs
            model_in = jnp.concatenate([x_t, jnp.array([u_t])])
            Theta = theta.reshape(Q, p)
            A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
            H = h.reshape(Q, d)
            H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
            h_next = H_next.reshape(Q * d)
            x_hat = C_out @ h_next
            diff = x_hat - x_next
            return h_next, 0.5 * jnp.sum(diff ** 2)
        h0 = jnp.zeros(Q * d, dtype=jnp.float64)
        _, losses = jax.lax.scan(step, h0, (xs_t, us, xs_next))
        return jnp.sum(losses)

    g_bptt = jax.grad(loss_bptt)(theta)

    h, S = jnp.zeros(Q * d, dtype=jnp.float64), jnp.zeros((Q, d, p), dtype=jnp.float64)
    g_online_total = jnp.zeros((Q, p), dtype=jnp.float64)
    for t in range(T):
        x_t, u_t, x_next = xs_t[t], us[t], xs_next[t]
        model_in = jnp.concatenate([x_t, jnp.array([u_t])])
        H = h.reshape(Q, d)
        Theta = theta.reshape(Q, p)
        A_batch = jnp.einsum("qk,qkij->qij", Theta, basis)
        H_next = jnp.einsum("qij,qj->qi", A_batch, H) + (B_in @ model_in).reshape(Q, d)
        G_batch = jnp.einsum("qkij,qj->qik", basis, H)
        S_next = jnp.einsum("qij,qjk->qik", A_batch, S) + G_batch
        h_next = H_next.reshape(Q * d)
        x_hat = C_out @ h_next
        diff = x_hat - x_next
        dl_dh = (C_out.T @ diff).reshape(Q, d)
        g_online_total = g_online_total + jnp.einsum("qd,qdk->qk", dl_dh, S_next)
        h, S = h_next, S_next
    rel = float(jnp.linalg.norm(g_online_total.reshape(-1) - g_bptt) / (jnp.linalg.norm(g_bptt) + 1e-12))
    print(f"    relative grad error (accumulated online-RTRL vs BPTT over T={T}): {rel:.3e}")
    return rel < 1e-8


def verify_credit_allocation(C):
    print(f"  Actual allocated persistent eligibility, C={C}:")
    rc, gc, tc = regular_config(C), generic_config(C), rtu_config(C)
    s0 = jnp.zeros(rc["Q"] * rc["d"], dtype=jnp.float64)
    S0 = jnp.zeros((gc["Q"], gc["d"], gc["p"]), dtype=jnp.float64)
    from credit_memory.b28_rtu_faithful import rtu_streaming_init as rsi
    S_rtu = rsi(tc["hidden"], X_DIM_TEACHER + 1)
    rtu_actual = sum(int(np.prod(v.shape)) for v in S_rtu.values())
    print(f"    RegularBlock: actual s size={int(s0.shape[0])}  claimed credit={rc['credit']}  "
          f"MATCH={int(s0.shape[0])==rc['credit']}")
    print(f"    GenericBlock: actual S size={int(np.prod(S0.shape))}  claimed credit={gc['credit']}  "
          f"MATCH={int(np.prod(S0.shape))==gc['credit']}")
    print(f"    RTU: actual S size={rtu_actual}  formula credit(8h)={tc['credit']}  "
          f"NOTE: features={X_DIM_TEACHER+1} here (not 1), so RTU's OWN actual credit at this feature count "
          f"is 4h+4h*{X_DIM_TEACHER+1}={4*tc['hidden']+4*tc['hidden']*(X_DIM_TEACHER+1)}, "
          f"MATCH={rtu_actual==4*tc['hidden']+4*tc['hidden']*(X_DIM_TEACHER+1)}")


def rtu_config_streaming(C, features=X_DIM_TEACHER + 1):
    """RTU's OWN credit formula is 4h+4h*features (b28's actual
    rtu_streaming_init size), NOT the features=1-assumed 8h shortcut
    used in b35c. With features=5 here ([x_t,u_t]), naively reusing
    b35c's rtu_config(C) (hidden=C/8) would silently give RTU 3x the
    intended credit (24h vs 8h). FROZEN RULE: require actual credit <=
    C for every architecture (not nearest-feasible) -- use the LARGEST
    hidden_dim whose actual credit does not exceed C, via floor (not
    round, which could overshoot C, as it did at C=64: round gave
    hidden=3 -> credit=72 > 64, violating the budget). Report the
    realized (generally < C) credit plainly rather than forcing it."""
    h = max(1, int(np.floor(C / (4 + 4 * features))))
    credit = 4 * h + 4 * h * features
    assert credit <= C, f"RTU credit {credit} exceeds budget {C}"
    return dict(hidden=h, r=2 * h, P=4 * h + 4 * h * features, credit=credit)


# =======================================================================
# Main streaming experiment. FROZEN PROTOCOL (predeclared before any
# untouched evaluation seed was inspected):
#   - LR selected ONLY on TUNING_SEEDS (mean selection-window loss
#     averaged across tuning seeds), then FROZEN and applied unchanged
#     to every evaluation seed -- no per-eval-seed re-selection.
#   - EVAL_SEEDS are disjoint from TUNING_SEEDS and were not inspected
#     before this protocol was frozen (seed 0, used in the PILOT run
#     above, is deliberately excluded from EVAL_SEEDS for this reason).
# =======================================================================
LR_GRID_STREAM = (0.01, 0.02, 0.05)
SELECTION_WINDOW = (200, 800)   # strictly before the reported pre_window (800,1500)
TUNING_SEEDS = (100, 101)
EVAL_SEEDS = (11, 12, 13)


def _selection_score(losses):
    a, b = SELECTION_WINDOW
    seg = np.asarray(losses[a:b])
    if seg.size == 0 or not np.all(np.isfinite(seg)):
        return float("inf")
    return float(np.mean(seg))


def select_lr_regular(C, lr_grid=LR_GRID_STREAM, tuning_seeds=TUNING_SEEDS):
    rc = regular_config(C)
    scores = {}
    for lr in lr_grid:
        seed_scores = []
        for seed in tuning_seeds:
            xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
            carry, B_in = init_regular_streaming(rc["Q"], rc["d"], seed=1000 + seed)
            step_fn = make_regular_streaming_scan(rc["Q"], rc["d"], B_in, lr_theta=lr)
            losses, _, _ = run_scanned_stream(step_fn, carry, xs_t, us, xs_next)
            seed_scores.append(_selection_score(losses))
        scores[lr] = float(np.mean(seed_scores))
    return min(scores, key=scores.get), scores


def select_lr_generic(C, lr_grid=LR_GRID_STREAM, tuning_seeds=TUNING_SEEDS):
    gc = generic_config(C)
    scores = {}
    for lr in lr_grid:
        seed_scores = []
        for seed in tuning_seeds:
            xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
            carry, basis, B_in = init_generic_streaming(gc["Q"], gc["d"], gc["p"], seed=1000 + seed)
            step_fn = make_generic_streaming_scan(gc["Q"], gc["d"], gc["p"], basis, B_in, lr_theta=lr)
            losses, _, _ = run_scanned_stream(step_fn, carry, xs_t, us, xs_next)
            seed_scores.append(_selection_score(losses))
        scores[lr] = float(np.mean(seed_scores))
    return min(scores, key=scores.get), scores


def select_lr_rtu(C, lr_grid=LR_GRID_STREAM, tuning_seeds=TUNING_SEEDS):
    tc = rtu_config_streaming(C)
    scores = {}
    for lr in lr_grid:
        seed_scores = []
        for seed in tuning_seeds:
            xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
            losses, _, _ = run_rtu_stream(tc["hidden"], xs_t, us, xs_next, seed=1000 + seed, lr_theta=lr)
            seed_scores.append(_selection_score(losses))
        scores[lr] = float(np.mean(seed_scores))
    return min(scores, key=scores.get), scores


def run_one_regular(C, seed, lr):
    rc = regular_config(C)
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
    carry, B_in = init_regular_streaming(rc["Q"], rc["d"], seed=1000 + seed)
    step_fn = make_regular_streaming_scan(rc["Q"], rc["d"], B_in, lr_theta=lr)
    t0 = time.time()
    losses, x_hats, finites = run_scanned_stream(step_fn, carry, xs_t, us, xs_next)
    elapsed = time.time() - t0
    metrics = compute_metrics(losses, x_hats, finites, xs_next)
    metrics.update(r=rc["r"], P=rc["P"], credit=rc["credit"], elapsed=elapsed, seed=seed, C=C, lr=lr)
    return metrics


def run_one_generic(C, seed, lr):
    gc = generic_config(C)
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
    carry, basis, B_in = init_generic_streaming(gc["Q"], gc["d"], gc["p"], seed=1000 + seed)
    step_fn = make_generic_streaming_scan(gc["Q"], gc["d"], gc["p"], basis, B_in, lr_theta=lr)
    t0 = time.time()
    losses, x_hats, finites = run_scanned_stream(step_fn, carry, xs_t, us, xs_next)
    elapsed = time.time() - t0
    metrics = compute_metrics(losses, x_hats, finites, xs_next)
    metrics.update(r=gc["r"], P=gc["P"], credit=gc["credit"], elapsed=elapsed, seed=seed, C=C, lr=lr)
    return metrics


def run_one_rtu(C, seed, lr):
    tc = rtu_config_streaming(C)
    xs_t, us, xs_next = make_teacher_trajectory(seed=seed)
    t0 = time.time()
    losses, x_hats, finites = run_rtu_stream(tc["hidden"], xs_t, us, xs_next, seed=1000 + seed, lr_theta=lr)
    elapsed = time.time() - t0
    metrics = compute_metrics(losses, x_hats, finites, xs_next)
    metrics.update(r=tc["r"], P=tc["P"], credit=tc["credit"], elapsed=elapsed, seed=seed, C=C, lr=lr)
    return metrics


def measure_elig_step_time_regular(Q, d):
    B_in = jnp.zeros((Q * d, 5), dtype=jnp.float64)
    step_fn = jax.jit(make_regular_streaming_scan(Q, d, B_in))
    carry, _ = init_regular_streaming(Q, d, seed=0)
    x_t, u_t, x_next = jnp.zeros(4), 0.1, jnp.zeros(4)
    carry, _ = step_fn(carry, (x_t, u_t, x_next))
    jax.block_until_ready(carry)
    t0 = time.time()
    for _ in range(50):
        carry, _ = step_fn(carry, (x_t, u_t, x_next))
    jax.block_until_ready(carry)
    return (time.time() - t0) / 50


def measure_elig_step_time_generic(Q, d, p):
    theta0, basis = make_generic_params(seed=0, Q=Q, d=d, p=p)
    B_in = jnp.zeros((Q * d, 5), dtype=jnp.float64)
    step_fn = jax.jit(make_generic_streaming_scan(Q, d, p, basis, B_in))
    carry, _, _ = init_generic_streaming(Q, d, p, seed=0)
    x_t, u_t, x_next = jnp.zeros(4), 0.1, jnp.zeros(4)
    carry, _ = step_fn(carry, (x_t, u_t, x_next))
    jax.block_until_ready(carry)
    t0 = time.time()
    for _ in range(50):
        carry, _ = step_fn(carry, (x_t, u_t, x_next))
    jax.block_until_ready(carry)
    return (time.time() - t0) / 50


def measure_elig_step_time_rtu(hidden_dim):
    rng = np.random.RandomState(0)
    params = make_rtu_params(rng, hidden_dim, X_DIM_TEACHER + 1)
    stream_state = dict(real=np.zeros(hidden_dim), imag=np.zeros(hidden_dim),
                         S=rtu_streaming_init(hidden_dim, X_DIM_TEACHER + 1))
    u_t = jnp.zeros(X_DIM_TEACHER + 1)
    rtu_streaming_step(params, stream_state, u_t)
    t0 = time.time()
    for _ in range(50):
        rtu_streaming_step(params, stream_state, u_t)
    return (time.time() - t0) / 50


def run_experiment():
    print("=" * 78)
    print("VALIDITY CHECKS")
    print("=" * 78)
    ok_teacher = verify_teacher_impulse_response()
    ok_reg = validity_check_regular()
    ok_gen = validity_check_generic()
    for C in BUDGETS:
        verify_credit_allocation(C)
    print(f"\nALL VALIDITY CHECKS PASS: {ok_teacher and ok_reg and ok_gen}")
    if not (ok_teacher and ok_reg and ok_gen):
        print("STOPPING: validity checks failed.")
        return None

    all_results = []
    for C in BUDGETS:
        rc, gc, tc = regular_config(C), generic_config(C), rtu_config_streaming(C)
        print(f"\n{'='*78}\nC={C}: RegularBlock r={rc['r']} P={rc['P']} credit={rc['credit']}  |  "
              f"GenericBlock r={gc['r']} P={gc['P']} credit={gc['credit']}  |  "
              f"RTU(features=5) hidden={tc['hidden']} r={tc['r']} P={tc['P']} credit={tc['credit']} (realized)")
        print(f"{'='*78}")

        elig_reg = measure_elig_step_time_regular(rc["Q"], rc["d"])
        elig_gen = measure_elig_step_time_generic(gc["Q"], gc["d"], gc["p"])
        elig_rtu = measure_elig_step_time_rtu(tc["hidden"])
        print(f"  eligibility-step time: Regular={elig_reg*1e6:.2f}us  Generic={elig_gen*1e6:.2f}us  "
              f"RTU={elig_rtu*1e6:.2f}us")

        for name, run_fn, select_fn in [("RegularBlock", run_one_regular, select_lr_regular),
                                          ("GenericBlock", run_one_generic, select_lr_generic),
                                          ("RTU", run_one_rtu, select_lr_rtu)]:
            best_lr, lr_scores = select_fn(C)
            print(f"  [{name}] LR selected on TUNING_SEEDS={TUNING_SEEDS}: {lr_scores} -> best_lr={best_lr}")
            seed_results = []
            for seed in EVAL_SEEDS:
                m = run_fn(C, seed, best_lr)
                seed_results.append(m)
                print(f"  [{name}] seed={seed}: lr={m['lr']}  pre_nmse={m['pre_nmse']}  post_nmse={m['post_nmse']}  "
                      f"steps_to_recover={m['steps_to_recover']}  cum_loss={m['cum_loss']:.3e}  "
                      f"diverged={m['diverged']}  n_nonfinite={m['n_nonfinite']}  elapsed={m['elapsed']:.2f}s")
            pre_vals = [m["pre_nmse"] for m in seed_results if m["pre_nmse"] is not None]
            post_vals = [m["post_nmse"] for m in seed_results if m["post_nmse"] is not None]
            print(f"  [{name}] ACROSS SEEDS: pre_nmse median={np.median(pre_vals):.4e} mean={np.mean(pre_vals):.4e} "
                  f"std={np.std(pre_vals):.4e}  |  post_nmse median={np.median(post_vals):.4e} "
                  f"mean={np.mean(post_vals):.4e} std={np.std(post_vals):.4e}")
            all_results.append(dict(C=C, arch=name, seed_results=[
                {k: v for k, v in m.items() if k != "trailing_curve"} for m in seed_results
            ], trailing_curves=[m["trailing_curve"] for m in seed_results]))

    with open("/tmp/b35d_streaming_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to /tmp/b35d_streaming_results.json")
    return all_results


if __name__ == "__main__":
    run_experiment()
