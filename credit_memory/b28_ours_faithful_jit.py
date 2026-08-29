"""Phase B28 -- "ours" (structured/factored exact-RTRL, full unreduced
temporal basis) ported into the IDENTICAL corrected outer scaffold
verified for RTU (commit 6c65863): same POPGym Autoencode environment,
observation representation/normalization, ScaleReward reward scaling,
gamma/lambda/entropy/ObGD/kappa/alpha, separate actor/critic networks,
Dense->LayerNorm->LeakyReLU encoder with sparse-init, action sampling,
episode/reset semantics, eligibility-trace reset, frame accounting,
v2 checkpoint/logging machinery. Reuses all of that machinery UNCHANGED
from b28_rtu_faithful_jit.py -- the only thing that changes is the
recurrent core (RTU -> our structured exact-RTRL architecture) and its
per-step credit propagation.

Architecture (B25/B27/B28's own, unchanged): h_{t+1} = (I_n⊗R)h_t +
(I_n⊗B)Phi((I_n⊗C)h_t, u_t), Phi a genuine MLP mixing all n*k entries
of z_t=(I_n⊗C)h_t and u_t. Exact per-step RTRL sensitivity propagated
via the analytic bounded-interface factorization:
  J_t = I_n⊗R + sum_ab F_ab,t ⊗ Q_ab,   Q_ab = B E_ab C
using the FULL unreduced temporal basis V_theta=I_r (naive/full-r
representation, as throughout B28) -- this experiment is about
representation/exact-online-learning transfer, NOT credit-memory
compression (ours' credit-storage cost is reported explicitly, not
matched to RTU's).

Run tests: python -m credit_memory.b28_ours_faithful_jit
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

import popgym

from credit_memory.b28_popgym_stage1 import one_hot_obs
from credit_memory.b25_nonlinear_credit import make_arch, family_dim, psi_flat, psi_from_flat
from credit_memory import b28_rtu_faithful_train as reft
from credit_memory.b28_rtu_faithful import ENCODER_FAMILIES, LEAKY_RELU_SLOPE, sparse_init
from credit_memory.b28_rtu_faithful_jit import (
    encoder_forward, encoder_jacobian, head_forward, categorical_entropy,
    obgd_step_size_jit, sample_mean_std_init, sample_mean_std_update,
    _running_update, _running_normalize, reward_scale_init, _reward_scale_update,
    _reward_scale_apply,
)

jax.config.update("jax_enable_x64", True)

OURS_FAMILIES = ("R", "B", "C", "psi")
ALL_OURS_NET_FAMILIES = ENCODER_FAMILIES + OURS_FAMILIES
OURS_FAMILY_ROLES = {**{f: "shared" for f in ENCODER_FAMILIES}}  # R,B,C,psi handled separately (whole-state, not per-unit)


# ---------------------------------------------------------------------------
# Pure-jnp versions of make_Q_all / compute_F_ab / direct_term (the
# originals in b25_nonlinear_credit.py use numpy ops on the JAX
# outputs, which is not jit-compatible when R/B/C/psi are traced
# parameters). Identical math, jnp-only.
# ---------------------------------------------------------------------------
def make_Q_all_jnp(B, C, k):
    """Q_ab = B @ E_ab @ C, (k,k,r,r), via a fixed one-hot basis E."""
    r = B.shape[0]
    E = jnp.eye(k * k).reshape(k, k, k, k)  # E[a,b] is the (k,k) one-hot at (a,b)
    # Q[a,b] = B @ E[a,b] @ C -- vectorize via einsum
    Q = jnp.einsum("ri,abij,js->abrs", B, E, C)
    return Q


def Phi_jnp(z_flat, u, psi):
    x = jnp.concatenate([z_flat, u])
    h = jnp.tanh(psi["W1"] @ x + psi["b1"])
    return psi["W2"] @ h + psi["b2"]


def forward_step_jnp(h, u, R, B, C, psi):
    n = h.shape[0]
    k = C.shape[0]
    z_flat = (h @ C.T).reshape(-1)
    phi_out = Phi_jnp(z_flat, u, psi)
    phi_r = phi_out.reshape(n, k)
    h_next = h @ R.T + phi_r @ B.T
    return h_next, z_flat


def compute_F_ab_jnp(h, u, R, B, C, psi, n, k):
    z_flat = (h @ C.T).reshape(-1)
    J_Phi = jax.jacobian(lambda zz: Phi_jnp(zz, u, psi))(z_flat)  # (nk,nk)
    J4 = J_Phi.reshape(n, k, n, k)          # [p,a,q,b]
    F = jnp.transpose(J4, (1, 3, 0, 2))     # [a,b,p,q]
    return F


def direct_term_jnp(family, h, u, R, B, C, psi, n, r, k, u_dim, hidden):
    """d(h_next)/d(theta_family)|_{h,u fixed}, shape (n,r,m_family)."""
    if family == "R":
        def f(flat):
            return forward_step_jnp(h, u, flat.reshape(r, r), B, C, psi)[0].reshape(-1)
        J = jax.jacobian(f)(R.reshape(-1))
    elif family == "B":
        def f(flat):
            return forward_step_jnp(h, u, R, flat.reshape(r, k), C, psi)[0].reshape(-1)
        J = jax.jacobian(f)(B.reshape(-1))
    elif family == "C":
        def f(flat):
            return forward_step_jnp(h, u, R, B, flat.reshape(k, r), psi)[0].reshape(-1)
        J = jax.jacobian(f)(C.reshape(-1))
    elif family == "psi":
        flat0 = psi_flat(psi)
        def f(flat):
            psi_ = psi_from_flat(flat, n, k, u_dim, hidden)
            return forward_step_jnp(h, u, R, B, C, psi_)[0].reshape(-1)
        J = jax.jacobian(f)(flat0)
    else:
        raise ValueError(family)
    return J.reshape(n, r, -1)


# ---------------------------------------------------------------------------
# ours params / network construction
# ---------------------------------------------------------------------------
RHO_MAX = 0.95


def project_stable_jnp(R_mat, rho_max=RHO_MAX):
    """JIT-compatible port of b28_stage2_ac_lambda.py's project_stable:
    RTU's parameterization keeps its spectral radius < 1 by
    CONSTRUCTION; ours' dense R has no such guarantee -- confirmed
    (again) empirically here: raw ObGD updates pushed R's spectral
    radius past 1 within ~200-400 frames, causing the recurrence to
    diverge geometrically (value -> 1e28 -> 1e116 within 600 frames).
    Rescales R toward rho_max whenever its spectral radius exceeds it;
    a no-op otherwise."""
    eigval_mag = jnp.max(jnp.abs(jnp.linalg.eigvals(R_mat)))
    scale = jnp.where(eigval_mag > rho_max, rho_max / eigval_mag, 1.0)
    return R_mat * scale, eigval_mag, scale


def make_ours_arch(rng, r, k, n, u_dim, hidden, seed):
    return make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)


def make_encoder_params_local(rng, in_dim, width, sparsity=0.9):
    from credit_memory.b28_rtu_faithful import make_encoder_params
    return make_encoder_params(rng, in_dim, width, sparsity)


def make_ours_network(rng, in_dim, width, r, k, n, hidden, head_out_dim, seed, sparsity=0.9):
    ours_arch = make_ours_arch(rng, r, k, n, width, hidden, seed)
    from credit_memory.b28_rtu_faithful_train import make_head_params
    return dict(
        enc=make_encoder_params_local(rng, in_dim, width, sparsity),
        ours=dict(R=ours_arch["R"], B=ours_arch["B"], C=ours_arch["C"], psi=ours_arch["psi"]),
        head=make_head_params(rng, n * r, head_out_dim, sparsity),
        cfg=dict(r=r, k=k, n=n, u_dim=width, hidden=hidden),
    )


def ours_network_param_count(net):
    total = 0
    for group in ("enc", "head"):
        for v in net[group].values():
            total += int(np.prod(v.shape))
    for k_ in ("R", "B", "C"):
        total += int(np.prod(net["ours"][k_].shape))
    for v in net["ours"]["psi"].values():
        total += int(np.prod(v.shape))
    return total


def ours_streaming_init(r, k, n, m_psi, enc_params):
    """X[fam] shapes: (n,r,m_fam) for R/B/C/psi (m_fam = that family's
    own flat param count); (n,r,*enc_param_shape) for each encoder
    family (matching enc_params' own shapes)."""
    m_R, m_B, m_C = r * r, r * k, k * r
    X = dict(R=jnp.zeros((n, r, m_R)), B=jnp.zeros((n, r, m_B)), C=jnp.zeros((n, r, m_C)),
             psi=jnp.zeros((n, r, m_psi)))
    for fam in ENCODER_FAMILIES:
        X[fam] = jnp.zeros((n, r) + enc_params[fam].shape)
    return dict(h=jnp.zeros((n, r)), X=X)


def ours_net_streaming_step_jit(ours_params, enc_params, h, X, obs_onehot, r, k, n, u_dim, hidden):
    """Pure-jnp streaming step for ours: encoder -> ours recurrence,
    exact RTRL sensitivity for ALL 8 families (4 encoder + R,B,C,psi),
    naive/full-r basis (V_theta=I_r)."""
    R, B, C, psi = ours_params["R"], ours_params["B"], ours_params["C"], ours_params["psi"]

    x_t = encoder_forward(obs_onehot, enc_params)
    h_next, z_flat = forward_step_jnp(h, x_t, R, B, C, psi)

    Q = make_Q_all_jnp(B, C, k)  # (k,k,r,r)
    F = compute_F_ab_jnp(h, x_t, R, B, C, psi, n, k)  # (k,k,n,n)

    new_X = {}
    for fam in OURS_FAMILIES:
        Xf = X[fam]
        Direct = direct_term_jnp(fam, h, x_t, R, B, C, psi, n, r, k, u_dim, hidden)  # (n,r,m)
        term_R = jnp.einsum("ij,pjc->pic", R, Xf)
        term_Q = jnp.einsum("abpq,abij,qjc->pic", F, Q, Xf)
        new_X[fam] = term_R + term_Q + Direct

    # Encoder-family chaining: d(h_next)/d(enc_theta) = d(h_next)/d(x_t) * d(x_t)/d(enc_theta)
    #   + recurrent term via R,Q acting on the OLD encoder-X.
    # d(pre-nonlinearity dependence on x_t) enters through B (input weight
    # analogous to RTU's B_real/B_imag): the direct_term for family "B"
    # already captures d(h_next)/dB|_h,u fixed; for x_t's OWN effect
    # (holding B fixed), we need d(h_next)/d(x_t) directly.
    def h_next_of_x(x):
        return forward_step_jnp(h, x, R, B, C, psi)[0].reshape(-1)
    dhnext_dx = jax.jacobian(h_next_of_x)(x_t).reshape(n, r, u_dim)  # (n,r,u_dim)

    enc_jacs = encoder_jacobian(obs_onehot, enc_params)
    for fam in ENCODER_FAMILIES:
        Xf = X[fam]  # (n,r,*enc_param_shape) -- may be >3D (e.g. W_enc)
        Jx = enc_jacs[fam]  # (u_dim, *enc_param_shape)
        term_R = jnp.einsum("ij,pj...->pi...", R, Xf)
        term_Q = jnp.einsum("abpq,abij,qj...->pi...", F, Q, Xf)
        direct_enc = jnp.tensordot(dhnext_dx, Jx, axes=([2], [0]))  # (n,r,*enc_param_shape)
        new_X[fam] = term_R + term_Q + direct_enc

    return h_next, z_flat, new_X


def ours_per_step_grad_jit(X, dLdh_flat):
    """dL/dtheta = sum_{p,i} dLdh[p,i] * X[fam][p,i,...] for each family
    (both ours' own R/B/C/psi and the encoder families -- all are
    GLOBAL/shared parameters here, no per-unit diagonal structure like
    RTU's, so every family sums fully over p,i)."""
    grads = {}
    for fam in OURS_FAMILIES:
        grads[fam] = jnp.einsum("pi,pi...->...", dLdh_flat, X[fam])
    for fam in ENCODER_FAMILIES:
        grads[fam] = jnp.einsum("pi,pi...->...", dLdh_flat, X[fam])
    return grads


# ---------------------------------------------------------------------------
# Correctness gate items 1-2: one-step forward dynamics + structured
# exact RTRL vs BPTT (independent jax.grad reference), for a SMALL
# config, encoder+ours combined (8 families).
# ---------------------------------------------------------------------------
def ours_rollout_bptt(ours_params, enc_params, obs_seq, r, n):
    def step(h, obs_onehot):
        x_t = encoder_forward(obs_onehot, enc_params)
        h_next, _ = forward_step_jnp(h, x_t, ours_params["R"], ours_params["B"],
                                      ours_params["C"], ours_params["psi"])
        return h_next, h_next
    h0 = jnp.zeros((n, r))
    _, Hs = jax.lax.scan(step, h0, obs_seq)
    return Hs  # (T,n,r)


def test_ours_streaming_vs_bptt(seed=21, T_=6, r=3, k=2, n=3, width=5, in_dim=4, hidden=6):
    rng = np.random.RandomState(seed)
    net = make_ours_network(rng, in_dim, width, r, k, n, hidden, head_out_dim=2, seed=seed)
    ours_params, enc_params = net["ours"], net["enc"]
    m_psi = family_dim("psi", dict(r=r, k=k, n=n, u_dim=width, hidden=hidden, psi=ours_params["psi"]))

    obs_seq = jnp.array((rng.rand(T_, in_dim) < 0.4).astype(np.float64))
    dLdH_seq = jnp.array(rng.randn(T_, n, r) * 0.3)

    all_params = {**ours_params, **enc_params}

    def loss_of(params_dict):
        op = {k_: params_dict[k_] for k_ in OURS_FAMILIES}
        ep = {k_: params_dict[k_] for k_ in ENCODER_FAMILIES}
        Hs = ours_rollout_bptt(op, ep, obs_seq, r, n)
        return jnp.sum(Hs * dLdH_seq)

    bptt_grads = jax.grad(loss_of)(all_params)
    # The streaming path's per-step gradients for R/B/C/psi are all FLAT
    # vectors (matching direct_term_jnp's jax.jacobian(...).reshape(n,r,-1)
    # convention), while jax.grad on the unflattened params returns
    # R/B/C's natural (r,r)/(r,k)/(k,r) matrix shapes and psi's own
    # nested dict. Flatten all four the same way for comparison.
    bptt_flat = dict(
        R=bptt_grads["R"].reshape(-1), B=bptt_grads["B"].reshape(-1), C=bptt_grads["C"].reshape(-1),
        psi=jnp.concatenate([bptt_grads["psi"]["W1"].ravel(), bptt_grads["psi"]["b1"],
                             bptt_grads["psi"]["W2"].ravel(), bptt_grads["psi"]["b2"]]),
    )

    stream = ours_streaming_init(r, k, n, m_psi, enc_params)
    grad_accum = {}
    for fam in ALL_OURS_NET_FAMILIES:
        if fam in OURS_FAMILIES:
            grad_accum[fam] = jnp.zeros(bptt_flat[fam].shape)
        else:
            grad_accum[fam] = jnp.zeros_like(all_params[fam])
    h = stream["h"]
    X = stream["X"]
    for t in range(T_):
        h_next, z_flat, X = ours_net_streaming_step_jit(ours_params, enc_params, h, X, obs_seq[t],
                                                          r, k, n, width, hidden)
        dLdh_t = dLdH_seq[t]
        g_t = ours_per_step_grad_jit(X, dLdh_t)
        for fam in ALL_OURS_NET_FAMILIES:
            grad_accum[fam] = grad_accum[fam] + g_t[fam]
        h = h_next

    errs = {}
    for fam in ALL_OURS_NET_FAMILIES:
        if fam in OURS_FAMILIES:
            errs[fam] = float(jnp.max(jnp.abs(grad_accum[fam] - bptt_flat[fam])))
        else:
            errs[fam] = float(jnp.max(jnp.abs(grad_accum[fam] - bptt_grads[fam])))

    # Also check one-step forward dynamics directly (item 1): the
    # streaming step's own h_next must match forward_step_jnp exactly
    # (same function, called identically) -- trivially true by
    # construction, verified here as an explicit regression guard.
    h0 = jnp.zeros((n, r))
    stream0 = ours_streaming_init(r, k, n, m_psi, enc_params)
    x0 = encoder_forward(obs_seq[0], enc_params)
    h_next_direct, _ = forward_step_jnp(h0, x0, ours_params["R"], ours_params["B"], ours_params["C"], ours_params["psi"])
    h_next_stream, _, _ = ours_net_streaming_step_jit(ours_params, enc_params, h0, stream0["X"], obs_seq[0],
                                                        r, k, n, width, hidden)
    errs["forward_dynamics"] = float(jnp.max(jnp.abs(h_next_direct - h_next_stream)))
    return errs


def main():
    print("=" * 70)
    print("Correctness gate 1-2: ours forward dynamics + structured exact RTRL vs BPTT")
    print("=" * 70)
    errs = test_ours_streaming_vs_bptt()
    for k, v in errs.items():
        print(f"  {k:16s} |err|={v:.2e}")
    assert all(v < 1e-6 for v in errs.values()), f"OURS RTRL PARITY FAILURE: {errs}"
    print("  PASS")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Full per-step learner update for ours -- mirrors
# b28_rtu_faithful_jit.py's full_update_step EXACTLY (same TD/entropy/
# ObGD/eligibility-reset/reward-scale/obs-normalize logic, verified
# there already), swapping only the recurrent core (RTU real/imag/S ->
# ours' h/X) and its per-step gradient extraction.
# ---------------------------------------------------------------------------
def ours_carry_from_net(net, stream, traces):
    return dict(
        params=dict(enc=net["enc"], ours=net["ours"], head=net["head"]),
        traces=dict(enc=traces["enc"], ours=traces["ours"], head=traces["head"]),
        h=stream["h"], X=stream["X"],
    )


def make_ours_carry(actor_net, actor_stream, actor_traces, critic_net, critic_stream, critic_traces,
                     obs_stats, reward_stats):
    return dict(
        actor=ours_carry_from_net(actor_net, actor_stream, actor_traces),
        critic=ours_carry_from_net(critic_net, critic_stream, critic_traces),
        obs_stats=dict(mean=jnp.asarray(obs_stats["mean"]), p=jnp.asarray(obs_stats["p"]),
                       count=jnp.asarray(obs_stats["count"]), var=jnp.asarray(obs_stats["var"])),
        reward_stats=dict(
            trace=jnp.asarray(reward_stats["trace"]),
            stats=dict(mean=jnp.asarray(reward_stats["stats"]["mean"]), p=jnp.asarray(reward_stats["stats"]["p"]),
                      count=jnp.asarray(reward_stats["stats"]["count"]), var=jnp.asarray(reward_stats["stats"]["var"])),
        ),
    )


def zero_ours_traces(net):
    m_psi = family_dim("psi", dict(r=net["cfg"]["r"], k=net["cfg"]["k"], n=net["cfg"]["n"],
                                    u_dim=net["cfg"]["u_dim"], hidden=net["cfg"]["hidden"],
                                    psi=net["ours"]["psi"]))
    traces = dict(
        enc={fam: jnp.zeros_like(net["enc"][fam]) for fam in ENCODER_FAMILIES},
        ours=dict(R=jnp.zeros_like(net["ours"]["R"]).reshape(-1), B=jnp.zeros_like(net["ours"]["B"]).reshape(-1),
                  C=jnp.zeros_like(net["ours"]["C"]).reshape(-1), psi=jnp.zeros(m_psi)),
        head={fam: jnp.zeros_like(net["head"][fam]) for fam in ("W", "b")},
    )
    return traces


def ours_full_update_step(carry, action, raw_reward, raw_obs_next, done,
                           gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                           actor_kappa, critic_kappa, r, k, n, u_dim, hidden):
    actor, critic = carry["actor"], carry["critic"]

    z_actor_cur = actor["h"].reshape(-1)
    z_critic_cur = critic["h"].reshape(-1)
    value_cur = head_forward(z_critic_cur, critic["params"]["head"])[0]
    logits_cur = head_forward(z_actor_cur, actor["params"]["head"])

    obs_stats_next = _running_update(carry["obs_stats"], raw_obs_next)
    obs_next_norm = _running_normalize(obs_stats_next, raw_obs_next)

    a_h_next, _, a_X_next = ours_net_streaming_step_jit(
        actor["params"]["ours"], actor["params"]["enc"], actor["h"], actor["X"], obs_next_norm, r, k, n, u_dim, hidden)
    c_h_next, _, c_X_next = ours_net_streaming_step_jit(
        critic["params"]["ours"], critic["params"]["enc"], critic["h"], critic["X"], obs_next_norm, r, k, n, u_dim, hidden)
    z_critic_next = c_h_next.reshape(-1)
    value_next_raw = head_forward(z_critic_next, critic["params"]["head"])[0]
    value_next = jnp.where(done > 0.5, 0.0, value_next_raw)

    reward_stats_next = _reward_scale_update(carry["reward_stats"], raw_reward, gamma, done)
    reward_norm = _reward_scale_apply(reward_stats_next, raw_reward)

    td_error = reward_norm + gamma * value_next - value_cur

    def logp_fn(z):
        logits = head_forward(z, actor["params"]["head"])
        logp = logits - jax.scipy.special.logsumexp(logits)
        return logp[action]

    def entropy_fn(z):
        return categorical_entropy(head_forward(z, actor["params"]["head"]))

    def value_fn(z):
        return head_forward(z, critic["params"]["head"])[0]

    dlogp_dz = jax.grad(logp_fn)(z_actor_cur)
    dentropy_dz = jax.grad(entropy_fn)(z_actor_cur)
    dactor_target_dh = (dlogp_dz + entropy_coef * jnp.sign(td_error) * dentropy_dz).reshape(n, r)
    dv_dh = jax.grad(value_fn)(z_critic_cur).reshape(n, r)

    g_actor = ours_per_step_grad_jit(actor["X"], dactor_target_dh)
    g_critic = ours_per_step_grad_jit(critic["X"], dv_dh)

    def actor_target(head_params):
        logits = head_forward(z_actor_cur, head_params)
        logp = logits - jax.scipy.special.logsumexp(logits)
        return logp[action] + entropy_coef * jnp.sign(td_error) * categorical_entropy(logits)

    def critic_target(head_params):
        return head_forward(z_critic_cur, head_params)[0]

    gh_actor = jax.grad(actor_target)(actor["params"]["head"])
    gh_critic = jax.grad(critic_target)(critic["params"]["head"])

    def update_group(traces_g, grads_g, keys):
        return {kk: gamma * lam * traces_g[kk] + grads_g[kk] for kk in keys}

    new_actor_traces_enc = update_group(actor["traces"]["enc"], g_actor, ENCODER_FAMILIES)
    new_actor_traces_ours = update_group(actor["traces"]["ours"], g_actor, OURS_FAMILIES)
    new_actor_traces_head = update_group(actor["traces"]["head"], gh_actor, ("W", "b"))

    new_critic_traces_enc = update_group(critic["traces"]["enc"], g_critic, ENCODER_FAMILIES)
    new_critic_traces_ours = update_group(critic["traces"]["ours"], g_critic, OURS_FAMILIES)
    new_critic_traces_head = update_group(critic["traces"]["head"], gh_critic, ("W", "b"))

    actor_trace_leaves = (list(new_actor_traces_enc.values()) + list(new_actor_traces_ours.values())
                          + list(new_actor_traces_head.values()))
    critic_trace_leaves = (list(new_critic_traces_enc.values()) + list(new_critic_traces_ours.values())
                           + list(new_critic_traces_head.values()))

    step_actor = obgd_step_size_jit(actor_alpha, actor_kappa, td_error, actor_trace_leaves)
    step_critic = obgd_step_size_jit(critic_alpha, critic_kappa, td_error, critic_trace_leaves)

    new_actor_enc = {kk: actor["params"]["enc"][kk] + step_actor * td_error * new_actor_traces_enc[kk] for kk in ENCODER_FAMILIES}
    new_actor_head = {kk: actor["params"]["head"][kk] + step_actor * td_error * new_actor_traces_head[kk] for kk in ("W", "b")}
    actor_R_raw = (actor["params"]["ours"]["R"].reshape(-1) + step_actor * td_error * new_actor_traces_ours["R"]).reshape(r, r)
    actor_R_proj, actor_R_eigval, actor_R_scale = project_stable_jnp(actor_R_raw)
    new_actor_ours = dict(
        R=actor_R_proj,
        B=(actor["params"]["ours"]["B"].reshape(-1) + step_actor * td_error * new_actor_traces_ours["B"]).reshape(r, k),
        C=(actor["params"]["ours"]["C"].reshape(-1) + step_actor * td_error * new_actor_traces_ours["C"]).reshape(k, r),
        psi=psi_from_flat(psi_flat(actor["params"]["ours"]["psi"]) + step_actor * td_error * new_actor_traces_ours["psi"],
                          n, k, u_dim, hidden),
    )

    new_critic_enc = {kk: critic["params"]["enc"][kk] + step_critic * td_error * new_critic_traces_enc[kk] for kk in ENCODER_FAMILIES}
    new_critic_head = {kk: critic["params"]["head"][kk] + step_critic * td_error * new_critic_traces_head[kk] for kk in ("W", "b")}
    critic_R_raw = (critic["params"]["ours"]["R"].reshape(-1) + step_critic * td_error * new_critic_traces_ours["R"]).reshape(r, r)
    critic_R_proj, critic_R_eigval, critic_R_scale = project_stable_jnp(critic_R_raw)
    new_critic_ours = dict(
        R=critic_R_proj,
        B=(critic["params"]["ours"]["B"].reshape(-1) + step_critic * td_error * new_critic_traces_ours["B"]).reshape(r, k),
        C=(critic["params"]["ours"]["C"].reshape(-1) + step_critic * td_error * new_critic_traces_ours["C"]).reshape(k, r),
        psi=psi_from_flat(psi_flat(critic["params"]["ours"]["psi"]) + step_critic * td_error * new_critic_traces_ours["psi"],
                          n, k, u_dim, hidden),
    )

    not_done = 1.0 - done

    def mask_traces(traces_g):
        return {kk: not_done * v for kk, v in traces_g.items()}

    new_actor = dict(
        params=dict(enc=new_actor_enc, ours=new_actor_ours, head=new_actor_head),
        traces=dict(enc=mask_traces(new_actor_traces_enc), ours=mask_traces(new_actor_traces_ours),
                    head=mask_traces(new_actor_traces_head)),
        h=not_done * a_h_next + done * actor["h"],
        X={fam: not_done * a_X_next[fam] + done * actor["X"][fam] for fam in ALL_OURS_NET_FAMILIES},
    )
    new_critic = dict(
        params=dict(enc=new_critic_enc, ours=new_critic_ours, head=new_critic_head),
        traces=dict(enc=mask_traces(new_critic_traces_enc), ours=mask_traces(new_critic_traces_ours),
                    head=mask_traces(new_critic_traces_head)),
        h=not_done * c_h_next + done * critic["h"],
        X={fam: not_done * c_X_next[fam] + done * critic["X"][fam] for fam in ALL_OURS_NET_FAMILIES},
    )

    new_carry = dict(actor=new_actor, critic=new_critic, obs_stats=obs_stats_next, reward_stats=reward_stats_next)
    diagnostics = dict(td_error=td_error, value_cur=value_cur, value_next=value_next,
                        step_actor=step_actor, step_critic=step_critic,
                        entropy=categorical_entropy(logits_cur), logits_cur=logits_cur,
                        actor_R_eigval=actor_R_eigval, actor_R_projected=(actor_R_scale < 1.0),
                        critic_R_eigval=critic_R_eigval, critic_R_projected=(critic_R_scale < 1.0))
    return new_carry, diagnostics
