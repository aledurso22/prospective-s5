"""B35a -- bounded product-local commutative response algebra.

Motivated by the single-long-jet dimension-normalization audit
(p2a_b34_dimension_norm_audit.py): in a length-r jet R[eps]/(eps^r),
EVERY global coefficient vector (theta, and every generated coefficient
a_t, b_t, kappa_t, c_t) participates in ONE length-r convolution
operator, with ||M_a||_inf = ||a||_1 exactly. Rescaling theta/kappa's
tail merely exposed the same r-dependent gain through a_t -- the
mechanism is generic to any r-length coefficient vector in a SINGLE
big factor, not specific to theta. Bounding the LOCAL factor size (not
rescaling coefficients as a function of total r) is the structural fix.

Algebra: A = product_{q=1}^Q R[eps_q]/(eps_q^d), r = Q*d, equal-size
factors. An element is represented as a flat (r,) vector, reshaped to
(Q,d) for all algebra operations:
  - multiplication is BLOCKWISE: (a*b)[q] = truncated_conv(a[q], b[q])
    within each d-dim factor (reusing b34a's exact Toeplitz
    regular-representation construction at size d); cross-factor
    multiplication is EXACTLY ZERO (a direct product of rings has
    componentwise multiplication -- this is exact, not an
    approximation).
  - the regular representation M_a of the WHOLE algebra is BLOCK-
    DIAGONAL: M_a = blockdiag(M_{a_1}, ..., M_{a_Q}), each block the
    d x d Toeplitz regular-rep matrix for factor q.

d=1 is the semisimple/diagonal endpoint (RTU-like, no nilpotent tail).
d=r (Q=1) is the old single long jet.
d in {2,4,8} are the new bounded product-local candidates.

STABILITY DESIGN (the actual fix): every A-valued quantity that enters
the recurrent multiplier is generated/parameterized PER FACTOR, with a
DIMENSION-INDEPENDENT local constraint -- NOT globally rescaled by
total r. Each factor's coefficients split into a semisimple/base
component (index 0) kept in a bounded range, and a nilpotent tail
(index 1..d-1) constrained to sum_i|.|<=RHO_NIL. This is applied
consistently to theta (the sole trainable A-valued parameter, m=1,
exactly mirroring b34a's P=r accounting) AND to every generated
coefficient (a_t, b_t, kappa_t, c_t) -- the tail cap is applied as a
PROJECTION (rescale only if exceeding, matching the R_V spectral
projection style already used for the bounded-interface flag), so it
remains valid DURING TRAINING for theta, not only at initialization.
Adding factors (Q) never changes any per-factor constant.

Run: python -m credit_memory.b35a_product_local_algebra
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

jax.config.update("jax_enable_x64", True)

from credit_memory.b34a_jet_algebra_correctness import make_M as make_M_local, X_DIM, GEN_HIDDEN, C1, C3

RHO_NIL = 1.0     # dimension-independent per-factor nilpotent-tail L1 cap
RHO_BASE = 1.0    # dimension-independent per-factor base-coefficient clip range


# =======================================================================
# Blockwise algebra primitives -- vectorized over Q via vmap, never a
# Python loop over factors (so cost/compile time is Q-scalable).
# =======================================================================
def alg_mult_blockwise(a, b, Q, d):
    """a, b: (r,)=(Q*d,) flat vectors. Returns (r,) = blockwise product."""
    A = a.reshape(Q, d)
    B = b.reshape(Q, d)
    M_batch = jax.vmap(lambda u: make_M_local(u, d))(A)   # (Q,d,d)
    out = jnp.einsum("qij,qj->qi", M_batch, B)             # (Q,d)
    return out.reshape(Q * d)


def make_M_full_blockdiag(a, Q, d):
    """EXPLICIT (r,r) block-diagonal regular-rep matrix -- for
    verification only, not used in the fast forward/RTRL path."""
    A = a.reshape(Q, d)
    blocks = [make_M_local(A[q], d) for q in range(Q)]
    return block_diag(*blocks)


def alg_one_blockwise(Q, d):
    """Multiplicative identity: (1,0,...,0) in every factor."""
    one_factor = jnp.eye(d)[0]
    return jnp.tile(one_factor, Q)


def phi_blockwise(y, Q, d):
    y2 = alg_mult_blockwise(y, y, Q, d)
    y3 = alg_mult_blockwise(y2, y, Q, d)
    return C1 * y + C3 * y3


def phi_prime_blockwise(y, Q, d):
    y2 = alg_mult_blockwise(y, y, Q, d)
    return C1 * alg_one_blockwise(Q, d) + 3.0 * C3 * y2


def transpose_mult_blockwise(u, q_vec, Q, d):
    """M_u^T @ q_vec via the direct per-block transpose-correlation
    formula (b34a's transpose_mult, applied per factor, vectorized)."""
    U = u.reshape(Q, d)
    Qv = q_vec.reshape(Q, d)

    def transpose_mult_one(u_row, q_row):
        out = []
        for p in range(d):
            out.append(jnp.dot(u_row[: d - p], q_row[p:d]))
        return jnp.stack(out)

    out = jax.vmap(transpose_mult_one)(U, Qv)
    return out.reshape(Q * d)


# =======================================================================
# Per-factor stability projection -- the actual fix. Applied to theta
# (trainable, after every optimizer step) and to every generated
# coefficient (a_t, b_t, kappa_t, c_t) at construction time, identically.
# =======================================================================
def project_local_tails(a, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE):
    """a: (r,) flat vector. Base column (index 0 of each factor) is
    clipped to [-rho_base,rho_base]; nilpotent tail (index>=1) of each
    factor is rescaled toward rho_nil ONLY IF its L1 norm exceeds it
    (same style as project_stable_R_V's spectral-radius projection) --
    a genuine per-factor constraint independent of Q, valid at init AND
    after training updates."""
    if d == 1:
        A = a.reshape(Q, 1)
        return jnp.clip(A, -rho_base, rho_base).reshape(Q)
    A = a.reshape(Q, d)
    base = jnp.clip(A[:, 0], -rho_base, rho_base)
    tail = A[:, 1:]
    l1 = jnp.sum(jnp.abs(tail), axis=1)
    scale = jnp.where(l1 > rho_nil, rho_nil / (l1 + 1e-12), 1.0)
    tail_scaled = tail * scale[:, None]
    return jnp.concatenate([base[:, None], tail_scaled], axis=1).reshape(Q * d)


# =======================================================================
# Frozen exogenous coefficient generator -- SAME small MLP structure as
# b34a's gen_forward (fan-in over X_DIM, GEN_HIDDEN hidden units,
# independent of Q/d), reshaped to (Q,d) and per-factor-projected.
# =======================================================================
def make_gen_params_local(seed, Q, d):
    r = Q * d
    rng = np.random.RandomState(seed)
    scale1 = 1.0 / np.sqrt(X_DIM)
    scale2 = 1.0 / np.sqrt(GEN_HIDDEN)
    return dict(
        Q=Q, d=d,
        W1=jnp.array(rng.randn(GEN_HIDDEN, X_DIM) * scale1),
        b1=jnp.array(rng.randn(GEN_HIDDEN) * 0.1),
        W_a=jnp.array(rng.randn(r, GEN_HIDDEN) * scale2), b_a=jnp.array(rng.randn(r) * 0.05),
        W_b=jnp.array(rng.randn(r, GEN_HIDDEN) * scale2), b_b=jnp.array(rng.randn(r) * 0.05),
        W_k=jnp.array(rng.randn(r, GEN_HIDDEN) * scale2), b_k=jnp.array(rng.randn(r) * 0.05),
        W_c=jnp.array(rng.randn(r, GEN_HIDDEN) * scale2), b_c=jnp.array(rng.randn(r) * 0.05),
    )


def gen_forward_local(x_t, gen_params, Q, d, rho_nil=RHO_NIL, rho_base=RHO_BASE):
    hid = jnp.tanh(gen_params["W1"] @ x_t + gen_params["b1"])
    a_raw = 0.25 * jnp.tanh(gen_params["W_a"] @ hid + gen_params["b_a"])
    b_raw = 0.1 * jnp.tanh(gen_params["W_b"] @ hid + gen_params["b_b"])
    c_raw = 0.1 * jnp.tanh(gen_params["W_c"] @ hid + gen_params["b_c"])
    k_raw_full = gen_params["W_k"] @ hid + gen_params["b_k"]
    kappa_raw = 0.1 * jnp.tanh(k_raw_full)
    if d > 1:
        K = kappa_raw.reshape(Q, d)
        # per-factor base kappa forced into a UNIT-guaranteed range
        # [0.2,0.8] -- SAME role as b34a's single-factor kappa0, now
        # applied identically to every factor (Q-independent).
        k_raw_base = k_raw_full.reshape(Q, d)[:, 0]
        kappa0 = 0.5 + 0.3 * jnp.tanh(k_raw_base)
        K = K.at[:, 0].set(kappa0)
        kappa_raw = K.reshape(Q * d)
    # per-factor tail cap applied to EVERY generated coefficient that
    # feeds the recurrent multiplier (a_t, kappa_t explicitly required;
    # b_t, c_t included too since both also multiply/add into y_t and
    # would reproduce the same r-dependent-gain failure mode otherwise
    # -- see module docstring / stability-design note).
    a_t = project_local_tails(a_raw, Q, d, rho_nil, rho_base=0.25)
    b_t = project_local_tails(b_raw, Q, d, rho_nil, rho_base=0.1)
    kappa_t = project_local_tails(kappa_raw, Q, d, rho_nil, rho_base=0.8)
    c_t = project_local_tails(c_raw, Q, d, rho_nil, rho_base=0.1)
    return a_t, b_t, kappa_t, c_t


# =======================================================================
# Model forward step -- structurally identical to b34a's h_step, using
# the blockwise algebra ops throughout. theta is the SOLE trainable
# A-valued parameter (m=1), reshaped/projected the same per-factor way.
# =======================================================================
def h_step_local(h, theta, x_t, gen_params, Q, d):
    a_t, b_t, kappa_t, c_t = gen_forward_local(x_t, gen_params, Q, d)
    A_theta_t = a_t + alg_mult_blockwise(kappa_t, theta, Q, d)
    y_t = alg_mult_blockwise(A_theta_t, h, Q, d) + alg_mult_blockwise(b_t, theta, Q, d) + c_t
    return phi_blockwise(y_t, Q, d)


def rollout_h_local(h0, theta, xs, gen_params, Q, d):
    def step(h, x_t):
        h_next = h_step_local(h, theta, x_t, gen_params, Q, d)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, xs)
    return Hs


def make_theta_local(seed, Q, d, base_std=0.3, tail_std=0.2):
    rng = np.random.RandomState(seed)
    r = Q * d
    raw = jnp.array(rng.randn(r) * tail_std)
    if d > 1:
        raw = raw.reshape(Q, d).at[:, 0].set(jnp.array(rng.randn(Q) * base_std)).reshape(r)
    return project_local_tails(raw, Q, d)


# =======================================================================
# Reduced RTRL (algebra-native persistent eligibility, s in R^r, exactly
# as in b34a's reduced_algebra_grad, using the blockwise algebra ops).
# =======================================================================
def reduced_algebra_grad_local(theta, h0, xs, qs, phases, gen_params, Q, d):
    T = xs.shape[0]
    h = h0
    s = jnp.zeros(Q * d, dtype=jnp.float64)
    g_total = jnp.zeros(Q * d, dtype=jnp.float64)
    s_traj = []

    def ell(y, ph):
        return jnp.sin(y + ph) + 0.5 * y ** 2

    def dell_dy(y, ph):
        return jnp.cos(y + ph) + y

    for t in range(T):
        x_t = xs[t]
        a_t, b_t, kappa_t, c_t = gen_forward_local(x_t, gen_params, Q, d)
        A_theta_t = a_t + alg_mult_blockwise(kappa_t, theta, Q, d)
        y_t = alg_mult_blockwise(A_theta_t, h, Q, d) + alg_mult_blockwise(b_t, theta, Q, d) + c_t
        d_t = phi_prime_blockwise(y_t, Q, d)
        inner = alg_mult_blockwise(A_theta_t, s, Q, d) + alg_mult_blockwise(kappa_t, h, Q, d) + b_t
        s = alg_mult_blockwise(d_t, inner, Q, d)
        s_traj.append(s)
        h_next = phi_blockwise(y_t, Q, d)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + transpose_mult_blockwise(s, dl_dh, Q, d)
        h = h_next
    return g_total, jnp.stack(s_traj)


def full_rtrl_grad_local(theta, h0, xs, qs, phases, gen_params, Q, d):
    T = xs.shape[0]
    r = Q * d
    h = h0
    S = jnp.zeros((r, r), dtype=jnp.float64)
    g_total = jnp.zeros(r, dtype=jnp.float64)
    S_traj = []

    def ell(y, ph):
        return jnp.sin(y + ph) + 0.5 * y ** 2

    def dell_dy(y, ph):
        return jnp.cos(y + ph) + y

    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda hh: h_step_local(hh, theta, x_t, gen_params, Q, d))(h)
        G_t = jax.jacobian(lambda th: h_step_local(h, th, x_t, gen_params, Q, d))(theta)
        S = J_t @ S + G_t
        S_traj.append(S)
        h_next = h_step_local(h, theta, x_t, gen_params, Q, d)
        y = qs[t] @ h_next
        dl_dh = dell_dy(y, phases[t]) * qs[t]
        g_total = g_total + dl_dh @ S
        h = h_next
    return g_total, jnp.stack(S_traj)


def loss_bptt_local(theta, h0, xs, qs, phases, gen_params, Q, d):
    Hs = rollout_h_local(h0, theta, xs, gen_params, Q, d)
    ys = jnp.einsum("ti,ti->t", Hs, qs)
    return jnp.sum(jnp.sin(ys + phases) + 0.5 * ys ** 2)


def make_grad_bptt_fn_local():
    return jax.jit(jax.grad(loss_bptt_local, argnums=0), static_argnums=(6, 7))


def make_setting_local(seed, T, Q, d):
    rng = np.random.RandomState(seed)
    r = Q * d
    theta = make_theta_local(seed, Q, d)
    h0 = jnp.array(rng.randn(r) * 0.15)
    xs = jnp.array(rng.randn(T, X_DIM) * 0.6)
    qs = jnp.array(rng.randn(T, r))
    phases = jnp.array(rng.uniform(0, 2 * np.pi, size=T))
    return theta, h0, xs, qs, phases
