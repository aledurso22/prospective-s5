"""Phase B26 -- exact feature bottleneck.

Architecture: a q-dim PERSISTENT recurrent state s_t, computed through
a WIDE (n-unit) nonlinear transition that is never itself persisted:

  a_t     = U s_t + E u_t + b        (n,)
  x_{t+1} = sigma(a_t)                (n,)  -- ephemeral, not persisted
  s_{t+1} = V^T x_{t+1}                (q,)  -- the only thing carried forward

U: (n,q), V: (n,q), E: (n,u_dim), b: (n,). sigma=tanh.

Exact sensitivity recurrence (Part 1's central identity):
  E_{t+1}^theta = A_lat,t E_t^theta + Q_t^theta,
  A_lat,t = V^T D_t U   (q x q, D_t=diag(sigma'(a_t)))
  Q_t^theta = d(s_{t+1})/d(theta) |_{s_t fixed}  (via JAX autodiff)

Three independent methods compared throughout: A (naive q x q RTRL via
per-step JAX autodiff Jacobians, NOT the hand-derived A_lat,t/Q_t
formula), B (the closed-form A_lat,t/Q_t recurrence), C (BPTT via
jax.grad on the full unrolled loss).

Run: python -m credit_memory.b26_feature_bottleneck
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Architecture.
# ---------------------------------------------------------------------------
def make_params(n, q, u_dim, seed, scale=0.6):
    rng = np.random.RandomState(seed)
    U = rng.randn(n, q) / np.sqrt(q) * scale
    V = rng.randn(n, q) / np.sqrt(n) * scale
    E = rng.randn(n, u_dim) / np.sqrt(u_dim) * scale
    b = rng.randn(n) * 0.1
    return dict(U=jnp.array(U), V=jnp.array(V), E=jnp.array(E), b=jnp.array(b), n=n, q=q,
                u_dim=u_dim)


def one_step(s, u, params):
    a = params["U"] @ s + params["E"] @ u + params["b"]
    x_next = jnp.tanh(a)
    s_next = params["V"].T @ x_next
    return s_next, a, x_next


def rollout(s0, U_seq, params):
    s = s0
    Ss = [s]
    for t in range(U_seq.shape[0]):
        s, _, _ = one_step(s, U_seq[t], params)
        Ss.append(s)
    return jnp.stack(Ss)


def family_shape(family, params):
    if family == "b":
        return (params["n"],)
    return params[family].shape


def family_size(family, params):
    return int(np.prod(family_shape(family, params)))


# ---------------------------------------------------------------------------
# Part 1: three independent methods.
# ---------------------------------------------------------------------------
def q_naive_rtrl(params, s0, U_seq, dLds_seq, family):
    """Method A: per-step JAX-autodiff Jacobians (independent of the
    hand-derived A_lat,t=V^T D_t U formula -- this is genuinely a
    different computational path, not just a relabeling)."""
    q = params["q"]
    shape = family_shape(family, params)
    m = family_size(family, params)
    theta0 = params[family].reshape(-1)

    s = s0
    Et = np.zeros((q, m))
    grad = np.zeros(m)
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]

        def f_s(ss):
            s_next, _, _ = one_step(ss, u_t, params)
            return s_next

        def f_theta(th):
            p = dict(params)
            p[family] = th.reshape(shape)
            s_next, _, _ = one_step(s, u_t, p)
            return s_next

        J_s = np.asarray(jax.jacobian(f_s)(s))
        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        Et = J_s @ Et + J_theta
        s = np.asarray(f_s(s))
        grad += np.asarray(dLds_seq[t]) @ Et
    return grad


def q_factorized_rtrl(params, s0, U_seq, dLds_seq, family, Et0=None):
    """Method B: closed-form A_lat,t=V^T D_t U recurrence, Q_t^theta via
    JAX autodiff of the direct-source term (not hand-derived, avoiding
    duplicate-bug risk, matching B25's established discipline)."""
    q, n = params["q"], params["n"]
    shape = family_shape(family, params)
    m = family_size(family, params)
    theta0 = params[family].reshape(-1)
    U_mat, V_mat = np.asarray(params["U"]), np.asarray(params["V"])

    s = s0
    Et = np.zeros((q, m)) if Et0 is None else np.array(Et0)
    grad = np.zeros(m)
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]
        a = np.asarray(params["U"] @ s + params["E"] @ u_t + params["b"])
        D = 1.0 - np.tanh(a) ** 2
        A_lat = V_mat.T @ (D[:, None] * U_mat)

        def f_theta(th):
            p = dict(params)
            p[family] = th.reshape(shape)
            s_next, _, _ = one_step(s, u_t, p)
            return s_next

        Q_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        Et = A_lat @ Et + Q_theta
        s_next, _, _ = one_step(s, u_t, params)
        s = np.asarray(s_next)
        grad += np.asarray(dLds_seq[t]) @ Et
    return grad


def bptt_grad(params, s0, U_seq, target_fn, family):
    theta0 = params[family]

    def loss_of(theta):
        p = dict(params, **{family: theta})
        Ss = rollout(s0, U_seq, p)
        return target_fn(Ss)

    g = jax.grad(loss_of)(theta0)
    return np.asarray(g).reshape(-1)


def dLds_from_target(params, s0, U_seq, target_fn):
    def loss_of_S(Ss):
        return target_fn(Ss)
    Ss = rollout(s0, U_seq, params)
    g = jax.grad(loss_of_S)(Ss)
    return np.asarray(g[1:])


# ---------------------------------------------------------------------------
# CORRECTION (per user review): the "reduction" claim needs a genuine
# equivalent-realization baseline, not a hypothetical one. The q-state
# system s_{t+1}=V^Tsigma(Us_t+Eu_t+b) has the exactly equivalent
# persistent-WIDE realization (D=0 case of Part 5's extended
# architecture, verified algebraically: s_{t+1}=V^Tx_{t+1}=
# V^Tsigma(UV^Tx_t+Eu_t+b)=V^Tsigma(Us_t+Eu_t+b) when s_t=V^Tx_t):
#   x_{t+1} = sigma(U (V^T x_t) + E u_t + b),   s_t = V^T x_t.
# Method A here tracks the GENUINE persistent n-dim object dx_t/dtheta
# (n x m), Method B is the already-implemented q_factorized_rtrl
# (q x m) on the mathematically-equivalent s_t, Method C is BPTT
# through the x-space rollout. Careful with s0=V^Tx0: for family='V'
# this has a direct dependency (x0 itself does NOT depend on any
# parameter -- it is a fixed external initial condition -- but the
# READOUT s0=V^Tx0 does), handled via the same Et0-seeding mechanism
# already validated in Part 5's falsification_test.
# ---------------------------------------------------------------------------
def wide_one_step(x, u, params):
    a = params["U"] @ (params["V"].T @ x) + params["E"] @ u + params["b"]
    return jnp.tanh(a)


def wide_rollout(x0, U_seq, params):
    x = x0
    Xs = [x]
    for t in range(U_seq.shape[0]):
        x = wide_one_step(x, U_seq[t], params)
        Xs.append(x)
    return jnp.stack(Xs)


def s_from_x(x, params):
    return params["V"].T @ x


def wide_naive_rtrl(params, x0, U_seq, dLds_seq, family):
    """Method A: genuine full wide RTRL, dx_t/dtheta in R^{n x m}."""
    n = params["n"]
    shape = family_shape(family, params)
    m = family_size(family, params)
    theta0 = params[family].reshape(-1)

    x = x0
    Ex = np.zeros((n, m))
    grad = np.zeros(m)
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]

        def f_x(xx):
            return wide_one_step(xx, u_t, params)

        def f_theta(th):
            p = dict(params, **{family: th.reshape(shape)})
            return wide_one_step(x, u_t, p)

        J_x = np.asarray(jax.jacobian(f_x)(x))
        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        Ex = J_x @ Ex + J_theta
        x = f_x(x)

        def g_theta(th):
            p = dict(params, **{family: th.reshape(shape)})
            return s_from_x(x, p)  # readout direct term at x_{t+1}, theta held free

        readout_direct = np.asarray(jax.jacobian(g_theta)(theta0))  # (q,m)
        S_next = readout_direct + np.asarray(params["V"]).T @ Ex
        grad += np.asarray(dLds_seq[t]) @ S_next
    return grad


def wide_realization_test(n, q, u_dim, seed=100, T_=6):
    """Three independent methods on the SAME (D=0) equivalent
    realization: A (full wide, n x m), B (reduced, q x m -- reusing
    q_factorized_rtrl on the mathematically-equivalent s_t), C (BPTT
    through the x-space rollout, an independent formulation of the
    reference from the one used elsewhere in this file)."""
    params = make_params(n=n, q=q, u_dim=u_dim, seed=seed)
    rng = np.random.RandomState(seed + 1)
    x0 = jnp.array(rng.randn(n) * 0.3)
    s0 = jnp.array(np.asarray(params["V"]).T @ np.asarray(x0))
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    target = jnp.array(rng.randn(T_, q) * 0.3)

    def target_fn(Ss):
        return 0.5 * jnp.sum((Ss[1:] - target) ** 2) / T_

    def true_loss(theta_dict):
        p = dict(params, **theta_dict)
        Xs = wide_rollout(x0, U_seq, p)
        Ss = Xs @ p["V"]  # (T+1,q) == V^T x_t per row
        return target_fn(Ss)

    true_grads = {}
    for family in ("U", "V", "E", "b"):
        g = jax.grad(lambda th, f=family: true_loss({f: th}))(params[family])
        true_grads[family] = np.asarray(g).reshape(-1)

    dLds = dLds_from_target(params, s0, U_seq, target_fn)

    wide_grads, reduced_grads = {}, {}
    for family in ("U", "V", "E", "b"):
        wide_grads[family] = wide_naive_rtrl(params, x0, U_seq, dLds, family)
        Et0 = None
        if family == "V":
            Et0 = np.asarray(jax.jacobian(lambda Vth: Vth.T @ x0)(params["V"])).reshape(q, -1)
        reduced_grads[family] = q_factorized_rtrl(params, s0, U_seq, dLds, family, Et0=Et0)

    errs = {}
    for family in ("U", "V", "E", "b"):
        errs[family] = dict(
            wide_vs_bptt=float(np.max(np.abs(wide_grads[family] - true_grads[family]))),
            reduced_vs_bptt=float(np.max(np.abs(reduced_grads[family] - true_grads[family]))),
            wide_vs_reduced=float(np.max(np.abs(wide_grads[family] - reduced_grads[family]))),
        )
    return errs


# ---------------------------------------------------------------------------
# Part 2: scaling accounting. IMPORTANT distinction stated explicitly:
# Method A (naive, above) and Method B (factorized) are BOTH q x m in
# THIS implementation, by design -- they are two independent
# computational paths to the same minimal object (autodiff-per-step vs
# closed-form A_lat,t), not a size comparison. The "full vs reduced"
# comparison Part 2 actually wants is against a HYPOTHETICAL naive
# implementation that does not realize x_t is ephemeral (not
# recurrent) and would mistakenly track its own n-dim sensitivity
# across time -- reported explicitly as such, not as what Method A
# implements.
# ---------------------------------------------------------------------------
def scaling_report(n, q, u_dim=2):
    m_U, m_V, m_E, m_b = n * q, n * q, n * u_dim, n
    families = dict(U=m_U, V=m_V, E=m_E, b=m_b)
    total_params = sum(families.values())
    reduced_floats = sum(q * m for m in families.values())          # actual: q x m per family
    naive_hypothetical_floats = sum(n * m for m in families.values())  # if x_t wrongly persisted
    forward_ops = n * q + n * u_dim + n           # a_t = Us+Eu+b, one matvec each dominant term
    eligibility_update_ops = sum(q * q * m for m in families.values())  # A_lat(qxq) @ Et(qxm)
    return dict(total_params=total_params, reduced_floats=reduced_floats,
                naive_hypothetical_floats=naive_hypothetical_floats,
                forward_ops=forward_ops, eligibility_update_ops=eligibility_update_ops)


# ---------------------------------------------------------------------------
# Parts 3/4: training loop using the factorized RTRL gradients themselves
# (a real use, matching B25's own discipline).
# ---------------------------------------------------------------------------
def _clip(g, maxnorm=1.0):
    nrm = np.linalg.norm(g)
    return g if nrm <= maxnorm else g * (maxnorm / nrm)


def train_student(n, q, u_dim, U_seq, target_fn, steps, lr, seed):
    params = make_params(n=n, q=q, u_dim=u_dim, seed=seed)
    rng = np.random.RandomState(seed + 1)
    s0 = jnp.array(rng.randn(q) * 0.2)
    losses = []
    for _ in range(steps):
        dLds = dLds_from_target(params, s0, U_seq, target_fn)
        grads = {}
        for family in ("U", "V", "E", "b"):
            g = q_factorized_rtrl(params, s0, U_seq, dLds, family)
            grads[family] = _clip(g, 1.0).reshape(family_shape(family, params))
        for family in ("U", "V", "E", "b"):
            params = dict(params, **{family: jnp.array(np.asarray(params[family]) - lr * grads[family])})
        Ss = rollout(s0, U_seq, params)
        loss = float(target_fn(Ss))
        losses.append(loss)
    return losses, params


# ---------------------------------------------------------------------------
# Part 5: causal-bottleneck falsification. Extended architecture with a
# GENUINELY persistent wide state x_t (not ephemeral):
#   x_{t+1} = D x_t + sigma(U (V^T x_t) + E u_t + b)
# s_t := V^T x_t is still computable, but the D x_t term is a real
# n-dim recurrence not reducible through s_t in general. Prediction:
# naively running the Part 1-4 q-only recurrence FAILS generically
# (D != 0) even for a loss defined purely on s_t=V^T x_t (since ds_t/
# dtheta still requires the full x_t dynamics). Restored to exactness
# in special cases where D's effect is invisible through s_t: D=0
# (trivial collapse) or V^T D = 0 (D's image lies in ker(V^T) -- a
# genuinely nontrivial D that is still unobservable through s_t).
# ---------------------------------------------------------------------------
def make_extended_params(n, q, u_dim, seed, D_mode="generic", D_scale=0.3):
    base = make_params(n=n, q=q, u_dim=u_dim, seed=seed)
    rng = np.random.RandomState(seed + 777)
    if D_mode == "zero":
        D = np.zeros((n, n))
    elif D_mode == "generic":
        M = rng.randn(n, n) / np.sqrt(n)
        eig = np.max(np.abs(np.linalg.eigvals(M)))
        D = M * (D_scale / eig)
    elif D_mode == "unobservable":
        # D's column space in ker(V^T): V^T is (q,n); build N spanning
        # the (n-q)-dim null space via SVD, D = N @ (arbitrary (n-q,n)).
        Vt = np.asarray(base["V"]).T
        _, _, Vh = np.linalg.svd(Vt, full_matrices=True)
        N = Vh[q:, :].T  # (n, n-q) orthonormal null-space basis of V^T
        M = rng.randn(n - q, n) / np.sqrt(n)
        D_raw = N @ M
        eig = np.max(np.abs(np.linalg.eigvals(D_raw))) if n - q > 0 else 1.0
        D = D_raw * (D_scale / max(eig, 1e-9)) if eig > 0 else D_raw
    else:
        raise ValueError(D_mode)
    return dict(base, D=jnp.array(D))


def extended_one_step(x, u, params):
    s = params["V"].T @ x
    a = params["U"] @ s + params["E"] @ u + params["b"]
    x_next = params["D"] @ x + jnp.tanh(a)
    return x_next


def extended_rollout(x0, U_seq, params):
    x = x0
    Xs = [x]
    for t in range(U_seq.shape[0]):
        x, = (extended_one_step(x, U_seq[t], params),)
        Xs.append(x)
    return jnp.stack(Xs)


def falsification_test(n, q, u_dim, D_mode, seed=60, T_=8, D_scale=0.3):
    params = make_extended_params(n=n, q=q, u_dim=u_dim, seed=seed, D_mode=D_mode, D_scale=D_scale)
    rng = np.random.RandomState(seed + 1)
    x0 = jnp.array(rng.randn(n) * 0.3)
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    target_s = jnp.array(rng.randn(T_, q) * 0.3)

    # TRUE gradient: BPTT through the FULL extended (n-dim) dynamics,
    # loss defined on the PROJECTED s_t=V^T x_t sequence only.
    def true_loss(theta_dict):
        p = dict(params, **theta_dict)
        Xs = extended_rollout(x0, U_seq, p)
        Ss = Xs[1:] @ p["V"]  # (T,q)
        return 0.5 * jnp.sum((Ss - target_s) ** 2) / T_

    true_grads = {}
    for family in ("U", "V", "E", "b"):
        g = jax.grad(lambda th: true_loss({family: th}))(params[family])
        true_grads[family] = np.asarray(g).reshape(-1)

    # NAIVE reduced: pretend s_t obeys the Part-1 recurrence, ignore D
    # entirely -- s0 seeded from the TRUE x0's projection. s0=V^T x0
    # depends on V directly; seed Et0 for family='V' with d(s0)/dV so
    # this initial-condition dependency isn't wrongly attributed to D
    # (a confound in the test, not the theory -- fixed here rather
    # than silently biasing the D=0 "should be exact" baseline).
    s0 = jnp.array(np.asarray(params["V"]).T @ np.asarray(x0))

    def target_fn(Ss):
        return 0.5 * jnp.sum((Ss[1:] - target_s) ** 2) / T_

    dLds = dLds_from_target(params, s0, U_seq, target_fn)
    naive_grads = {}
    for family in ("U", "V", "E", "b"):
        Et0 = None
        if family == "V":
            Et0 = np.asarray(jax.jacobian(lambda Vth: Vth.T @ x0)(params["V"]))
            Et0 = Et0.reshape(params["q"], -1)
        naive_grads[family] = q_factorized_rtrl(params, s0, U_seq, dLds, family, Et0=Et0)

    errs = {f: float(np.max(np.abs(naive_grads[f] - true_grads[f]))) for f in true_grads}
    return errs


def falsification_positive_control(n, q, u_dim, seed=60, T_=8, D_scale=0.3):
    """Per user review: V's nonzero error under 'unobservable' D is a
    STRUCTURAL finding, not an anomaly -- current-state unobservability
    (V^T D=0 AT ONE PARAMETER POINT) does not imply the condition is
    preserved under a perturbation of V itself, i.e. causal sufficiency
    must hold on a parameter-invariant family, not merely at one
    parameter point. This constructs the positive control the finding
    calls for: freeze V (exclude it from the compared families) and
    verify U,E,b are EXACTLY unaffected -- the naive q-only reduction
    is a full, exact positive control for every family whose tangent
    direction cannot perturb the V^T D=0 condition."""
    errs = falsification_test(n=n, q=q, u_dim=u_dim, D_mode="unobservable", seed=seed, T_=T_,
                               D_scale=D_scale)
    return {f: e for f, e in errs.items() if f != "V"}


# ---------------------------------------------------------------------------
# Part 6 (L=1 only -- explicit scope limit, see PHASE_B26.md): integrate
# the q-state feature bottleneck with B25/B25.1's bounded temporal
# interface. Joint state (h_t in R^r, s_t in R^q):
#   h_{t+1} = R h_t + B (W s_t) + Bu u_t
#   z_t     = C h_t                                (k-dim temporal readout)
#   s_{t+1} = V^T sigma(U s_t + E z_t + b)          (q-dim feature update)
# W: k x q reads the feature state back down to the temporal interface
# width k, closing the loop without a same-timestep circularity (h's
# injection at t+1 uses s_t, not s_{t+1}; s's update uses z_t=C h_t,
# the CURRENT temporal readout).
# Method A: naive full (r+q)-dim RTRL via per-step autodiff Jacobians.
# Method C: BPTT. Method B (reduced): temporal families (R,B,C,Bu,W)
# tracked in FULL r-space (B25's basis machinery, no further Krylov
# reduction attempted here -- explicit scope limit), feature families
# (U,V,E,b) tracked in full q-space, coupled via the SAME cross-term
# pattern as B25.1's cross-layer injection (now between heterogeneous
# r-space and q-space objects, not two temporal spaces).
# ---------------------------------------------------------------------------
def make_integrated_params(r, k, q, n, u_dim, seed):
    rng = np.random.RandomState(seed)
    def stable(rr):
        M = rng.randn(rr, rr) / np.sqrt(rr)
        eig = np.max(np.abs(np.linalg.eigvals(M)))
        return M * (0.85 / eig)
    R = stable(r)
    B = rng.randn(r, k) / np.sqrt(k) * 0.7
    C = rng.randn(k, r) / np.sqrt(r) * 0.7
    Bu = rng.randn(r, u_dim) / np.sqrt(u_dim) * 0.5
    W = rng.randn(k, q) / np.sqrt(q) * 0.6
    U = rng.randn(n, q) / np.sqrt(q) * 0.6
    V = rng.randn(n, q) / np.sqrt(n) * 0.6
    E = rng.randn(n, k) / np.sqrt(k) * 0.6
    b = rng.randn(n) * 0.1
    return dict(R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), Bu=jnp.array(Bu),
                W=jnp.array(W), U=jnp.array(U), V=jnp.array(V), E=jnp.array(E), b=jnp.array(b),
                r=r, k=k, q=q, n=n, u_dim=u_dim)


def integrated_step(h, s, u, p):
    z = p["C"] @ h
    h_next = p["R"] @ h + p["B"] @ (p["W"] @ s) + p["Bu"] @ u
    a = p["U"] @ s + p["E"] @ z + p["b"]
    s_next = p["V"].T @ jnp.tanh(a)
    return h_next, s_next


def integrated_rollout(h0, s0, U_seq, p):
    h, s = h0, s0
    Hs, Ss = [h], [s]
    for t in range(U_seq.shape[0]):
        h, s = integrated_step(h, s, U_seq[t], p)
        Hs.append(h)
        Ss.append(s)
    return jnp.stack(Hs), jnp.stack(Ss)


TEMPORAL_FAMILIES = ("R", "B", "C", "Bu", "W")
FEATURE_FAMILIES = ("U", "V", "E", "b")


def integrated_family_shape(family, p):
    return p[family].shape if family != "b" else (p["n"],)


def integrated_naive_grad(p, h0, s0, U_seq, dLdy_seq, family):
    """Method A: full (r+q)-dim per-step autodiff Jacobian, no reduction."""
    r, q = p["r"], p["q"]
    shape = integrated_family_shape(family, p)
    m = int(np.prod(shape))
    theta0 = p[family].reshape(-1)

    h, s = h0, s0
    S_full = np.zeros((r + q, m))
    grad = np.zeros(m)
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]

        def f_y(y):
            hh, ss = y[:r], y[r:]
            hn, sn = integrated_step(hh, ss, u_t, p)
            return jnp.concatenate([hn, sn])

        def f_theta(th):
            p2 = dict(p, **{family: th.reshape(shape)})
            hn, sn = integrated_step(h, s, u_t, p2)
            return jnp.concatenate([hn, sn])

        y = jnp.concatenate([h, s])
        J_y = np.asarray(jax.jacobian(f_y)(y))
        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        S_full = J_y @ S_full + J_theta
        h, s = integrated_step(h, s, u_t, p)
        grad += np.asarray(dLdy_seq[t]) @ S_full
    return grad


def integrated_bptt_grad(p, h0, s0, U_seq, target_fn, family):
    theta0 = p[family]
    def loss_of(theta):
        p2 = dict(p, **{family: theta})
        Hs, Ss = integrated_rollout(h0, s0, U_seq, p2)
        return target_fn(Hs, Ss)
    g = jax.grad(loss_of)(theta0)
    return np.asarray(g).reshape(-1)


def integrated_dLdy(p, h0, s0, U_seq, target_fn):
    r = p["r"]
    def loss_of_y(Ys):
        Hs, Ss = Ys[:, :r], Ys[:, r:]
        return target_fn(Hs, Ss)
    Hs, Ss = integrated_rollout(h0, s0, U_seq, p)
    Ys = jnp.concatenate([Hs, Ss], axis=1)
    g = jax.grad(loss_of_y)(Ys)
    return np.asarray(g[1:])


def integrated_reduced_grad(p, h0, s0, U_seq, dLdy_seq, family):
    """Method B: temporal families tracked in full r-space (B25 style,
    no further Krylov reduction here -- explicit scope limit), feature
    families in full q-space (B26 style), coupled each step via the
    SAME cross-term pattern as B25.1 (now between r-space and q-space)."""
    r, q, k, n = p["r"], p["q"], p["k"], p["n"]
    is_temporal = family in TEMPORAL_FAMILIES
    shape = integrated_family_shape(family, p)
    m = int(np.prod(shape))
    theta0 = p[family].reshape(-1)

    R, Bm, Cm, Bu, Wm = (np.asarray(p[x]) for x in ("R", "B", "C", "Bu", "W"))
    Um, Vm = np.asarray(p["U"]), np.asarray(p["V"])

    Eh = np.zeros((r, m))  # dh_t/dtheta
    Es = np.zeros((q, m))  # ds_t/dtheta
    grad = np.zeros(m)
    h, s = h0, s0
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]
        h_np, s_np = np.asarray(h), np.asarray(s)
        z_np = Cm @ h_np
        a_np = Um @ s_np + np.asarray(p["E"]) @ z_np + np.asarray(p["b"])
        D = 1.0 - np.tanh(a_np) ** 2  # (n,)
        A_lat = Vm.T @ (D[:, None] * Um)             # (q,q):  ds_next/ds
        Cross_sh = Vm.T @ (D[:, None] * np.asarray(p["E"])) @ Cm  # (q,r): ds_next/dh (via z=Ch)
        Cross_hs = Bm @ Wm                             # (r,q): dh_next/ds

        def f_theta(th):
            p2 = dict(p, **{family: th.reshape(shape)})
            hn, sn = integrated_step(h, s, u_t, p2)
            return jnp.concatenate([hn, sn])

        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))  # (r+q, m)
        Qh, Qs = J_theta[:r], J_theta[r:]

        Eh_next = R @ Eh + Cross_hs @ Es + Qh
        Es_next = Cross_sh @ Eh + A_lat @ Es + Qs
        Eh, Es = Eh_next, Es_next

        h, s = integrated_step(h, s, u_t, p)
        dLdy_t = np.asarray(dLdy_seq[t])
        grad += dLdy_t[:r] @ Eh + dLdy_t[r:] @ Es
    return grad


# ---------------------------------------------------------------------------
# Part 7 (light): end-to-end scaling accounting for the L=1 integrated
# model, extending Part 2's analytic-complexity approach.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CORRECTION (per user review): replace the training-failure-based q
# lower bound with an exact causal-memory/dimension test. A wide
# nonlinear q-small model CAN approximate a finite training sequence
# well via curve-fitting -- that is not evidence about persistent
# state capacity. The rigorous test: inject v in R^{q_teacher} once at
# t=0 (via u_0=v, u_dim=q_teacher), remove it (u_t=0 for t>=1), and ask
# whether v is exactly recoverable from s_T for a LATER T, on an open
# set of v. F: v -> s_T is smooth, R^q_teacher -> R^q.
#
# DIFFERENTIABLE-DECODER ARGUMENT (the exact statement, not just
# "F must be a local diffeomorphism"): IF a differentiable decoder G
# exists with G(F(v))=v on an open set, then by the chain rule
# DG(F(v)) @ DF(v) = I_{q_teacher}, which forces rank(DF(v))=q_teacher
# at every v in that set. But DF(v) has shape (q,q_teacher), so
# rank(DF(v))<=q always -- hence q<q_teacher makes rank(DF(v))=q_teacher
# IMPOSSIBLE, so no such G can exist, regardless of n. This is a
# structural (rank) fact, verified via autodiff, requiring no training
# at all. For q>=q_teacher, a numerically full-column-rank DF(v) shows
# only that the LOCAL DIFFERENTIAL OBSTRUCTION IS REMOVED / full local
# rank is achievable -- NOT a proof of global exact recoverability
# (existence of G is a separate, stronger claim not tested here).
# Training/reconstruction curves are kept only as an illustration.
# ---------------------------------------------------------------------------
def encode_then_recall_map(v, params, T_):
    n, q = params["n"], params["q"]
    s = jnp.zeros(q)
    zero_u = jnp.zeros(params["u_dim"])
    for t in range(T_):
        u_t = v if t == 0 else zero_u
        s, _, _ = one_step(s, u_t, params)
    return s


def causal_memory_rank_test(q, q_teacher, n, T_=2, seed=200):
    """Returns (J, singular values) for dF/dv at a generic random v,
    F: v -> s_T. IMPORTANT distinction, stated explicitly rather than
    collapsed into one rank number: for q<q_teacher, rank(J)<=q<q_teacher
    is a TRIVIAL shape fact (a (q,q_teacher) matrix with q<q_teacher
    columns exceeding rows cannot have full column rank) -- this alone
    already proves the structural obstruction, with no need for a
    numerical rank threshold. The genuinely nontrivial check is whether
    q>=q_teacher ACTUALLY ACHIEVES full column rank q_teacher (removing
    the obstruction) -- for that direction only, singular values are
    reported honestly rather than collapsed via an arbitrary tolerance
    (repeated tanh contraction over T_ steps genuinely degrades
    conditioning at larger T_ -- an expected nonlinear-dynamics effect,
    not a numerical bug -- so T_=2 is used for a clean, checked gap)."""
    params = make_params(n=n, q=q, u_dim=q_teacher, seed=seed)
    rng = np.random.RandomState(seed + 1)
    v0 = jnp.array(rng.randn(q_teacher) * 0.3)
    J = jax.jacobian(lambda v: encode_then_recall_map(v, params, T_))(v0)  # (q, q_teacher)
    J = np.asarray(J)
    S = np.linalg.svd(J, compute_uv=False)
    return J.shape, S


def causal_memory_illustration(q, q_teacher, n_list, T_=2, seed=201, steps=150, lr=0.05):
    """Training-based illustration ONLY (not the falsification itself):
    train a linear readout W_out (q -> q_teacher) on top of s_T to
    reconstruct v, over a fixed FAMILY of held-out random v's, and
    report reconstruction error vs n. Prediction (illustration of the
    rank fact above): error should NOT vanish as n grows when
    q<q_teacher (rank-limited), but should be drivable near 0 when
    q>=q_teacher."""
    rng = np.random.RandomState(seed)
    V_train = rng.randn(40, q_teacher) * 0.4
    V_test = rng.randn(20, q_teacher) * 0.4

    results = {}
    for n in n_list:
        params = make_params(n=n, q=q, u_dim=q_teacher, seed=seed + n)
        S_train = np.stack([np.asarray(encode_then_recall_map(jnp.array(v), params, T_))
                             for v in V_train])
        S_test = np.stack([np.asarray(encode_then_recall_map(jnp.array(v), params, T_))
                            for v in V_test])
        # closed-form least-squares linear readout (exact, no gradient descent needed
        # for THIS sub-step -- isolates the representational question cleanly)
        W_out, *_ = np.linalg.lstsq(S_train, V_train, rcond=None)
        pred_test = S_test @ W_out
        mse = float(np.mean((pred_test - V_test) ** 2))
        results[n] = mse
    return results


# ---------------------------------------------------------------------------
# CORRECTION (per user review): do not call end-to-end deep closed at
# L=1 only. Extend to L=2,3: a stack of the L=1 integrated layer, with
# layer l's h_l receiving the layer BELOW's temporal readout
# z_{l-1,t}=C_{l-1}h_{l-1,t} as its external input (matching B25/B25.1's
# routing convention). Key simplification vs B25.1: z_{l-1,t} enters
# h_l's update LINEARLY (via Bu_l), so the cross-layer term is a FIXED
# linear map Bu_l@C_{l-1} (no F_ab/G_ab nonlinear decomposition needed
# here -- the only nonlinearity in this architecture lives inside each
# layer's own local feature module, in the sigma applied to Us+Ez+b).
# ---------------------------------------------------------------------------
def build_integrated_stack(specs, seed):
    """specs: list of dicts with keys r,k,q,n (u_dim inferred: specs[0]
    keeps its own u_dim, layer l>0 uses u_dim=k_{l-1})."""
    layers = []
    for i, spec in enumerate(specs):
        u_dim = spec["u_dim"] if i == 0 else specs[i - 1]["k"]
        layers.append(make_integrated_params(r=spec["r"], k=spec["k"], q=spec["q"],
                                              n=spec["n"], u_dim=u_dim, seed=seed + i))
    return layers


def stack_integrated_step(hs, ss, u_ext, layers):
    lower_z = u_ext
    new_hs, new_ss, zs = [], [], []
    for l, p in enumerate(layers):
        h_next, s_next = integrated_step(hs[l], ss[l], lower_z, p)
        z = p["C"] @ hs[l]
        new_hs.append(h_next)
        new_ss.append(s_next)
        zs.append(z)
        lower_z = z
    return new_hs, new_ss, zs


def stack_integrated_rollout(h0s, s0s, U_seq, layers):
    hs, ss = h0s, s0s
    Hs, Ss = [hs], [ss]
    for t in range(U_seq.shape[0]):
        hs, ss, _ = stack_integrated_step(hs, ss, U_seq[t], layers)
        Hs.append(hs)
        Ss.append(ss)
    return Hs, Ss


def _flatten_hs_ss(hs, ss):
    return jnp.concatenate([h.reshape(-1) for h in hs] + [s.reshape(-1) for s in ss])


def _unflatten_hs_ss(flat, shapes_h, shapes_q):
    out_h, out_s = [], []
    i = 0
    for r_ in shapes_h:
        out_h.append(flat[i:i + r_])
        i += r_
    for q_ in shapes_q:
        out_s.append(flat[i:i + q_])
        i += q_
    return out_h, out_s


def stack_integrated_naive_grad(layers, h0s, s0s, U_seq, family_layer, dLdy_flat_seq):
    """Full (sum(r_l+q_l))-dim naive RTRL via per-step autodiff Jacobians."""
    L = len(layers)
    shapes_h = [p["r"] for p in layers]
    shapes_q = [p["q"] for p in layers]
    total_dim = sum(shapes_h) + sum(shapes_q)
    layer_idx, family = family_layer
    shape = integrated_family_shape(family, layers[layer_idx])
    m = int(np.prod(shape))
    theta0 = layers[layer_idx][family].reshape(-1)

    def step_flat(y_flat, u, layer_overrides):
        hs, ss = _unflatten_hs_ss(y_flat, shapes_h, shapes_q)
        layers_ = [dict(p, **ov) if ov else p for p, ov in zip(layers, layer_overrides)]
        new_hs, new_ss, _ = stack_integrated_step(hs, ss, u, layers_)
        return _flatten_hs_ss(new_hs, new_ss)

    y = _flatten_hs_ss(h0s, s0s)
    S = np.zeros((total_dim, m))
    grad = np.zeros(m)
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]

        def f_y(yy):
            return step_flat(yy, u_t, [{} for _ in layers])

        def f_theta(th):
            ov = [{} for _ in layers]
            ov[layer_idx] = {family: th.reshape(shape)}
            return step_flat(y, u_t, ov)

        J_y = np.asarray(jax.jacobian(f_y)(y))
        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        S = J_y @ S + J_theta
        y = np.asarray(f_y(y))
        grad += np.asarray(dLdy_flat_seq[t]) @ S
    return grad


def stack_integrated_bptt_grad(layers, h0s, s0s, U_seq, family_layer, target_fn):
    layer_idx, family = family_layer

    def loss_of(theta):
        layers_ = [dict(p) for p in layers]
        layers_[layer_idx] = dict(layers_[layer_idx], **{family: theta})
        Hs, Ss = stack_integrated_rollout(h0s, s0s, U_seq, layers_)
        return target_fn(Hs, Ss)

    theta0 = layers[layer_idx][family]
    g = jax.grad(loss_of)(theta0)
    return np.asarray(g).reshape(-1)


def stack_integrated_dLdy_flat(layers, h0s, s0s, U_seq, target_fn):
    shapes_h = [p["r"] for p in layers]
    shapes_q = [p["q"] for p in layers]

    def loss_of_flat_seq(flat_seq):
        Hs, Ss = [], []
        for t in range(flat_seq.shape[0]):
            hs, ss = _unflatten_hs_ss(flat_seq[t], shapes_h, shapes_q)
            Hs.append(hs)
            Ss.append(ss)
        return target_fn(Hs, Ss)

    Hs, Ss = stack_integrated_rollout(h0s, s0s, U_seq, layers)
    flat_seq = jnp.stack([_flatten_hs_ss(Hs[t], Ss[t]) for t in range(len(Hs))])
    g = jax.grad(loss_of_flat_seq)(flat_seq)
    return np.asarray(g[1:])


def stack_integrated_reduced_grad(layers, h0s, s0s, U_seq, dLdy_flat_seq, family_layer):
    """Reduced: per-layer (Eh_l in R^{r_l}, Es_l in R^{q_l}) local
    recursion (as in integrated_reduced_grad), PLUS a fixed LINEAR
    cross-layer term Bu_l@C_{l-1}@Eh_{l-1} injected into Eh_l for
    l>source_layer (no F_ab/G_ab needed -- z_{l-1,t} enters h_l's
    update linearly in this architecture)."""
    L = len(layers)
    layer_idx, family = family_layer
    shape = integrated_family_shape(family, layers[layer_idx])
    m = int(np.prod(shape))
    theta0 = layers[layer_idx][family].reshape(-1)

    Ehs = {l: np.zeros((layers[l]["r"], m)) for l in range(layer_idx, L)}
    Ess = {l: np.zeros((layers[l]["q"], m)) for l in range(layer_idx, L)}
    grad = np.zeros(m)

    hs, ss = h0s, s0s
    for t in range(U_seq.shape[0]):
        u_t = U_seq[t]
        new_hs, new_ss, zs = stack_integrated_step(hs, ss, u_t, layers)
        lower_inputs = [u_t] + zs[:-1]

        Eh_old = {l: Ehs[l].copy() for l in Ehs}
        Es_old = {l: Ess[l].copy() for l in Ess}
        Eh_new, Es_new = {}, {}

        for l in range(layer_idx, L):
            p = layers[l]
            r_l, q_l = p["r"], p["q"]
            R_l, Bm, Cm, Bu_l, Wm = (np.asarray(p[x]) for x in ("R", "B", "C", "Bu", "W"))
            Um, Vm = np.asarray(p["U"]), np.asarray(p["V"])
            h_np, s_np = np.asarray(hs[l]), np.asarray(ss[l])
            z_np = Cm @ h_np
            a_np = Um @ s_np + np.asarray(p["E"]) @ z_np + np.asarray(p["b"])
            D = 1.0 - np.tanh(a_np) ** 2
            A_lat = Vm.T @ (D[:, None] * Um)
            Cross_sh = Vm.T @ (D[:, None] * np.asarray(p["E"])) @ Cm
            Cross_hs = Bm @ Wm

            if l == layer_idx:
                def f_theta(th, p_=p, u_=lower_inputs[l], h_=hs[l], s_=ss[l]):
                    p2 = dict(p_, **{family: th.reshape(shape)})
                    hn, sn = integrated_step(h_, s_, u_, p2)
                    return jnp.concatenate([hn, sn])
                J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
                Qh, Qs = J_theta[:r_l], J_theta[r_l:]
            else:
                Qh, Qs = np.zeros((r_l, m)), np.zeros((q_l, m))

            cross_term = np.zeros((r_l, m))
            if l - 1 in Eh_old:
                C_prev = np.asarray(layers[l - 1]["C"])
                cross_term = Bu_l @ C_prev @ Eh_old[l - 1]
            # Direct term, ONLY at the first hop out of a family='C' source
            # (same lesson as B25.1): z_{source,t}=C_source@h_{source,t}
            # depends on C_source DIRECTLY (holding h_{source,t} fixed),
            # a pathway distinct from Qh/Qs (which only captures d(h_next)/dC
            # and d(s_next)/dC at the source layer itself) and easy to miss
            # since C ALSO serves as the cross-layer coupling variable here.
            if l == layer_idx + 1 and family == "C":
                h_src = np.asarray(hs[layer_idx])

                def g_theta(th, hh=h_src, r_src=layers[layer_idx]["r"]):
                    return th.reshape(layers[layer_idx]["k"], r_src) @ hh

                Zextra = np.asarray(jax.jacobian(g_theta)(theta0))  # (k_src, m)
                cross_term = cross_term + Bu_l @ Zextra

            Eh_new[l] = R_l @ Eh_old[l] + Cross_hs @ Es_old[l] + cross_term + Qh
            Es_new[l] = Cross_sh @ Eh_old[l] + A_lat @ Es_old[l] + Qs

        Ehs, Ess = Eh_new, Es_new
        hs, ss = new_hs, new_ss
        dLdy_t = dLdy_flat_seq[t]
        shapes_h = [p["r"] for p in layers]
        shapes_q = [p["q"] for p in layers]
        dLdh_t, dLds_t = _unflatten_hs_ss(dLdy_t, shapes_h, shapes_q)
        for l in range(layer_idx, L):
            grad += np.asarray(dLdh_t[l]) @ Ehs[l] + np.asarray(dLds_t[l]) @ Ess[l]
    return grad


def integrated_scaling_report(r, k, q, n, u_dim):
    m_temporal = dict(R=r * r, B=r * k, C=k * r, Bu=r * u_dim, W=k * q)
    m_feature = dict(U=n * q, V=n * q, E=n * k, b=n)
    total_params = sum(m_temporal.values()) + sum(m_feature.values())
    # forward state: r (temporal) + q (feature) -- x_t itself never persisted
    forward_state = r + q
    # persistent credit: temporal families tracked at r-dim, feature at q-dim
    temporal_credit_floats = sum(r * m for m in m_temporal.values())
    feature_credit_floats = sum(q * m for m in m_feature.values())
    total_credit_floats = temporal_credit_floats + feature_credit_floats
    forward_ops = r * r + r * k + k * r + r * u_dim + n * q + n * k + n  # dominant matvecs
    return dict(total_params=total_params, forward_state=forward_state,
                temporal_credit_floats=temporal_credit_floats,
                feature_credit_floats=feature_credit_floats,
                total_credit_floats=total_credit_floats, forward_ops=forward_ops)


def main():
    print("=" * 70)
    print("PART 1 -- standalone exactness (naive vs factorized vs BPTT)")
    print("=" * 70)
    for (n, q) in [(4, 2), (8, 4), (16, 8)]:
        params = make_params(n=n, q=q, u_dim=2, seed=1)
        rng = np.random.RandomState(2)
        s0 = jnp.array(rng.randn(q) * 0.3)
        T_ = 6
        U_seq = jnp.array(rng.randn(T_, 2) * 0.4)
        target = jnp.array(rng.randn(T_ + 1, q) * 0.3)
        target_fn = lambda Ss: 0.5 * jnp.sum((Ss - target) ** 2) / T_
        dLds = dLds_from_target(params, s0, U_seq, target_fn)
        for family in ("U", "V", "E", "b"):
            g_naive = q_naive_rtrl(params, s0, U_seq, dLds, family)
            g_fact = q_factorized_rtrl(params, s0, U_seq, dLds, family)
            g_bptt = bptt_grad(params, s0, U_seq, target_fn, family)
            e_nb = np.max(np.abs(g_naive - g_bptt))
            e_fb = np.max(np.abs(g_fact - g_bptt))
            print(f"  n={n:2d} q={q} {family}: |naive-bptt|={e_nb:.2e} |fact-bptt|={e_fb:.2e}")

    print()
    print("=" * 70)
    print("PART 1b -- GENUINE realization reduction (per user correction):")
    print("wide (n x m, dx_t/dtheta) vs reduced (q x m) vs BPTT, on the")
    print("EQUIVALENT persistent-wide D=0 realization x_{t+1}=sigma(UV^Tx_t+Eu_t+b)")
    print("=" * 70)
    for (n, q) in [(4, 2), (8, 4), (16, 6)]:
        errs = wide_realization_test(n=n, q=q, u_dim=1)
        for family, e in errs.items():
            print(f"  n={n:2d} q={q} {family}: wide-bptt={e['wide_vs_bptt']:.2e}  "
                  f"reduced-bptt={e['reduced_vs_bptt']:.2e}  "
                  f"wide-reduced={e['wide_vs_reduced']:.2e}")

    print()
    print("=" * 70)
    print("PART 2 -- scaling accounting (analytic, not wall-clock)")
    print("=" * 70)
    print(f"  {'n':>3} {'q':>3} {'params':>8} {'reduced':>10} {'naive_hyp':>10} "
          f"{'fwd_ops':>8} {'elig_ops':>10}")
    for q in (2, 4, 8):
        for n in (4, 8, 16, 32, 64):
            r_ = scaling_report(n, q)
            print(f"  {n:>3} {q:>3} {r_['total_params']:>8} {r_['reduced_floats']:>10} "
                  f"{r_['naive_hypothetical_floats']:>10} {r_['forward_ops']:>8} "
                  f"{r_['eligibility_update_ops']:>10}")

    print()
    print("=" * 70)
    print("PART 3 -- width genuinely adds transition capacity (fixed q)")
    print("=" * 70)
    rng = np.random.RandomState(50)
    n_teacher, q3 = 12, 3
    teacher = make_params(n=n_teacher, q=q3, u_dim=1, seed=50)
    s0_t = jnp.array(rng.randn(q3) * 0.3)
    T_ = 12
    U_seq = jnp.array(rng.randn(T_, 1) * 0.4)
    Ss_teacher = rollout(s0_t, U_seq, teacher)
    target = Ss_teacher[1:]
    target_fn = lambda Ss: 0.5 * jnp.sum((Ss[1:] - target) ** 2) / T_
    for n in (2, 4, 8, 16):
        losses, _ = train_student(n=n, q=q3, u_dim=1, U_seq=U_seq, target_fn=target_fn,
                                   steps=80, lr=0.05, seed=1)
        print(f"  n={n:2d}: final loss = {losses[-1]:.6f}")

    print()
    print("=" * 70)
    print("PART 4 -- causal-memory/dimension test (per user correction):")
    print("DIFFERENTIABLE-DECODER ARGUMENT, training as illustration ONLY.")
    print("F: v -> s_T. If a differentiable decoder G exists with G(F(v))=v")
    print("on an open set, chain rule forces DG(F(v))@DF(v)=I_qt, hence")
    print("rank(DF(v))=q_teacher there. But DF(v) has shape (q,q_teacher), so")
    print("rank<=q always -- q<q_teacher makes this IMPOSSIBLE (shape fact,")
    print("no numerics needed) => no such G exists, regardless of n.")
    print("q>=q_teacher: full column rank REMOVES THE LOCAL OBSTRUCTION --")
    print("NOT a proof of global exact recoverability (existence of G is a")
    print("separate, stronger claim not tested here).")
    print("=" * 70)
    q_teacher = 4
    print("  q < q_teacher: rank(DF)<=q<q_teacher BY SHAPE ALONE -- no decoder")
    print("  G can exist (chain-rule argument above), regardless of n:")
    for q in (1, 2, 3):
        for n in (8, 32):
            shape, S = causal_memory_rank_test(q=q, q_teacher=q_teacher, n=n)
            print(f"    q={q} n={n:2d}: J shape={shape} -> decoder existence ruled out "
                  f"by shape, S={np.round(S, 4)}")
    print("  q >= q_teacher: is the local differential obstruction removed?")
    for q in (4, 6):
        for n in (8, 32):
            shape, S = causal_memory_rank_test(q=q, q_teacher=q_teacher, n=n)
            ratio = S[q_teacher - 1] / S[0]
            print(f"    q={q} n={n:2d}: S={np.round(S, 4)}  smallest/largest={ratio:.2e} "
                  f"(nonzero -> local obstruction removed, NOT global recoverability proof)")
    print("  Illustration only (least-squares readout reconstruction MSE vs n):")
    for q in (2, 3, 4, 6):
        res = causal_memory_illustration(q=q, q_teacher=q_teacher, n_list=(4, 16, 64))
        tag = "< q_teacher" if q < q_teacher else ">= q_teacher"
        print(f"    q={q} ({tag}): {res}")

    print()
    print("=" * 70)
    print("PART 5 -- causal-bottleneck falsification")
    print("=" * 70)
    for D_mode in ("zero", "generic", "unobservable"):
        errs = falsification_test(n=8, q=3, u_dim=1, D_mode=D_mode)
        print(f"  D_mode={D_mode}: errs={ {k: f'{v:.2e}' for k, v in errs.items()} }")
    print("  Per user correction: V's nonzero error under 'unobservable' D is a")
    print("  STRUCTURAL finding (causal sufficiency must hold on a parameter-")
    print("  invariant family, not merely at one parameter point -- perturbing V")
    print("  generally destroys V^T D=0), not an anomaly. Positive control:")
    print("  freezing V (excluding it) gives an EXACT positive control for U,E,b:")
    pos = falsification_positive_control(n=8, q=3, u_dim=1)
    print(f"    {pos}")

    print()
    print("=" * 70)
    print("PART 6 -- integrated q+temporal model, L=1, 2, 3 (per user correction:")
    print("do not call deep closed until L=2,3 are checked)")
    print("=" * 70)
    print("  --- L=1 ---")
    p = make_integrated_params(r=3, k=2, q=3, n=6, u_dim=1, seed=7)
    rng = np.random.RandomState(8)
    h0 = jnp.array(rng.randn(3) * 0.3)
    s0 = jnp.array(rng.randn(3) * 0.3)
    T_ = 5
    U_seq = jnp.array(rng.randn(T_, 1) * 0.4)
    target_h = jnp.array(rng.randn(T_, 3) * 0.3)
    target_s = jnp.array(rng.randn(T_, 3) * 0.3)
    target_fn = lambda Hs, Ss: 0.5 * (jnp.sum((Hs[1:] - target_h) ** 2)
                                       + jnp.sum((Ss[1:] - target_s) ** 2)) / T_
    dLdy = integrated_dLdy(p, h0, s0, U_seq, target_fn)
    for family in TEMPORAL_FAMILIES + FEATURE_FAMILIES:
        g_naive = integrated_naive_grad(p, h0, s0, U_seq, dLdy, family)
        g_bptt = integrated_bptt_grad(p, h0, s0, U_seq, target_fn, family)
        g_red = integrated_reduced_grad(p, h0, s0, U_seq, dLdy, family)
        e_nb = np.max(np.abs(g_naive - g_bptt))
        e_rb = np.max(np.abs(g_red - g_bptt))
        print(f"    {family}: |naive-bptt|={e_nb:.2e}  |reduced-bptt|={e_rb:.2e}")

    print("  --- L=2 (source at earliest & final layer, all 9 families) ---")
    specs2 = [dict(r=3, k=2, q=2, n=4, u_dim=1), dict(r=3, k=2, q=2, n=4)]
    layers2 = build_integrated_stack(specs2, seed=5)
    rng = np.random.RandomState(6)
    h0s2 = [jnp.array(rng.randn(pp["r"]) * 0.3) for pp in layers2]
    s0s2 = [jnp.array(rng.randn(pp["q"]) * 0.3) for pp in layers2]
    T2 = 4
    U2 = jnp.array(rng.randn(T2, 1) * 0.4)
    target_h1_2 = jnp.array(rng.randn(T2, 3) * 0.3)
    target_s1_2 = jnp.array(rng.randn(T2, 2) * 0.3)
    target_fn2 = lambda Hs, Ss: 0.5 * (
        jnp.sum((jnp.stack([Hs[t][1] for t in range(1, T2 + 1)]) - target_h1_2) ** 2)
        + jnp.sum((jnp.stack([Ss[t][1] for t in range(1, T2 + 1)]) - target_s1_2) ** 2)) / T2
    dLdy2 = stack_integrated_dLdy_flat(layers2, h0s2, s0s2, U2, target_fn2)
    for layer_idx, family in [(0, f) for f in TEMPORAL_FAMILIES + FEATURE_FAMILIES] + \
                              [(1, f) for f in TEMPORAL_FAMILIES + FEATURE_FAMILIES]:
        g_naive = stack_integrated_naive_grad(layers2, h0s2, s0s2, U2, (layer_idx, family), dLdy2)
        g_bptt = stack_integrated_bptt_grad(layers2, h0s2, s0s2, U2, (layer_idx, family), target_fn2)
        g_red = stack_integrated_reduced_grad(layers2, h0s2, s0s2, U2, dLdy2, (layer_idx, family))
        e_nb = np.max(np.abs(g_naive - g_bptt))
        e_rb = np.max(np.abs(g_red - g_bptt))
        print(f"    layer{layer_idx}.{family}: |naive-bptt|={e_nb:.2e}  |reduced-bptt|={e_rb:.2e}")

    print("  --- L=3 (source at earliest/middle/final layer, loss on final layer) ---")
    specs3 = [dict(r=3, k=2, q=2, n=3, u_dim=1), dict(r=3, k=2, q=2, n=3), dict(r=3, k=2, q=2, n=3)]
    layers3 = build_integrated_stack(specs3, seed=15)
    rng = np.random.RandomState(16)
    h0s3 = [jnp.array(rng.randn(pp["r"]) * 0.3) for pp in layers3]
    s0s3 = [jnp.array(rng.randn(pp["q"]) * 0.3) for pp in layers3]
    U3 = jnp.array(rng.randn(T2, 1) * 0.4)
    target_h2_3 = jnp.array(rng.randn(T2, 3) * 0.3)
    target_s2_3 = jnp.array(rng.randn(T2, 2) * 0.3)
    target_fn3 = lambda Hs, Ss: 0.5 * (
        jnp.sum((jnp.stack([Hs[t][2] for t in range(1, T2 + 1)]) - target_h2_3) ** 2)
        + jnp.sum((jnp.stack([Ss[t][2] for t in range(1, T2 + 1)]) - target_s2_3) ** 2)) / T2
    dLdy3 = stack_integrated_dLdy_flat(layers3, h0s3, s0s3, U3, target_fn3)
    tests3 = [(0, f) for f in TEMPORAL_FAMILIES + FEATURE_FAMILIES] + \
             [(1, "C"), (1, "U")] + [(2, "R"), (2, "U")]
    for layer_idx, family in tests3:
        g_naive = stack_integrated_naive_grad(layers3, h0s3, s0s3, U3, (layer_idx, family), dLdy3)
        g_bptt = stack_integrated_bptt_grad(layers3, h0s3, s0s3, U3, (layer_idx, family), target_fn3)
        g_red = stack_integrated_reduced_grad(layers3, h0s3, s0s3, U3, dLdy3, (layer_idx, family))
        e_nb = np.max(np.abs(g_naive - g_bptt))
        e_rb = np.max(np.abs(g_red - g_bptt))
        print(f"    layer{layer_idx}.{family}: |naive-bptt|={e_nb:.2e}  |reduced-bptt|={e_rb:.2e}")

    print()
    print("=" * 70)
    print("PART 7 (light) -- integrated end-to-end scaling accounting")
    print("=" * 70)
    for n in (8, 16, 32, 64):
        r_ = integrated_scaling_report(r=3, k=2, q=4, n=n, u_dim=1)
        print(f"  n={n:3d}: {r_}")


if __name__ == "__main__":
    main()
