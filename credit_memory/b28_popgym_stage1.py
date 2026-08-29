"""Phase B28 Stage 1 -- POPGym correctness harness.

Verifies, INSIDE actual actor/critic network plumbing (embedding ->
recurrent core -> policy head + value head, not standalone cells), that:

  OURS: exact online factorized RTRL (recurrent-core state sensitivity,
        reused unchanged from B25) matches BPTT for a genuine
        actor-critic-style loss on real POPGym trajectories.
  RTU:  exact published block-local RTRL (reused unchanged from B27)
        matches BPTT for the SAME loss structure.

Scope, stated explicitly: this verifies (a) recurrent-state
sensitivity and (b) the resulting policy/value LOSS gradient, using a
FIXED, precomputed advantage/return signal from a real rollout -- NOT
yet a full streaming online RL update rule with its own eligibility
mechanism (item 7's "RL eligibility trace" is a separate, not-yet-built
piece, kept explicitly distinct per instruction). The embedding is a
FIXED one-hot (not trained) for this stage, to isolate the recurrent-
core/head gradient question cleanly before adding a trainable
embedding family.

Run: python -m credit_memory.b28_popgym_stage1
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b25_nonlinear_credit import (
    make_arch, forward_step, rollout, dLdh_from_target, factorized_rtrl_run,
    bptt_reference_grads, family_dim,
)
from credit_memory.b27_noncommutative_advantage import (
    make_nonlinear_rtu_arch, nonlinear_rtu_rollout_scalar, nonlinear_rtu_pre_and_next,
    rtu_exact_credit_grad, rtu_block_matrix,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Trajectory collection from real POPGym environments (random policy --
# sufficient for a correctness check, which only needs SOME genuine
# trajectory, not a good one).
# ---------------------------------------------------------------------------
def collect_trajectory(env, T_, seed):
    obs, _ = env.reset(seed=seed)
    rng = np.random.RandomState(seed + 1)
    obses = [obs]
    actions = []
    rewards = []
    for t in range(T_):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        actions.append(a)
        rewards.append(r)
        obses.append(obs)
        if term or trunc:
            break
    return obses[:-1], actions, rewards  # obs_0..obs_{T-1} aligned with action_0..action_{T-1}


def one_hot_obs(obs, obs_kind):
    """obs_kind: 'discrete4' (RepeatFirst) or 'autoencode' (Tuple(2,4))."""
    if obs_kind == "discrete4":
        v = np.zeros(4)
        v[obs] = 1.0
        return v
    else:
        v = np.zeros(6)
        v[obs[0]] = 1.0
        v[2 + obs[1]] = 1.0
        return v


def make_advantages_returns(rewards, gamma=0.95):
    """Simple discounted return-to-go, used as both 'return' (value
    target) and 'advantage' (policy-gradient weight, undiscounted
    baseline-free) -- sufficient for a gradient-EXACTNESS check; not a
    claim about a good RL algorithm."""
    T_ = len(rewards)
    returns = np.zeros(T_)
    running = 0.0
    for t in reversed(range(T_)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns, returns.copy()  # advantage == return (no baseline) for this check


# ---------------------------------------------------------------------------
# OURS: embedding (fixed one-hot) -> B25 recurrent core -> policy/value heads.
# ---------------------------------------------------------------------------
def make_head_params(z_dim, num_actions, seed):
    rng = np.random.RandomState(seed)
    return dict(
        W_pi=jnp.array(rng.randn(num_actions, z_dim) / np.sqrt(z_dim) * 0.5),
        b_pi=jnp.array(rng.randn(num_actions) * 0.1),
        W_v=jnp.array(rng.randn(1, z_dim) / np.sqrt(z_dim) * 0.5),
        b_v=jnp.array(rng.randn(1) * 0.1),
    )


def actor_critic_loss(z_seq, actions, advantages, returns, head):
    """z_seq: (T, z_dim) -- z_{t+1} used to act at time t. Standard
    actor-critic loss (policy log-prob * advantage + value MSE)."""
    logits = z_seq @ head["W_pi"].T + head["b_pi"]  # (T, num_actions)
    logp = jax.nn.log_softmax(logits, axis=-1)
    actions_arr = jnp.array(actions)
    logp_a = jnp.take_along_axis(logp, actions_arr[:, None], axis=1)[:, 0]
    adv = jnp.array(advantages)
    policy_loss = -jnp.sum(logp_a * adv)
    values = (z_seq @ head["W_v"].T + head["b_v"])[:, 0]
    ret = jnp.array(returns)
    value_loss = 0.5 * jnp.sum((values - ret) ** 2)
    return (policy_loss + value_loss) / z_seq.shape[0]


def ours_target_fn(H, C, head, actions, advantages, returns):
    T_ = len(actions)
    z_seq = H[1:T_ + 1] @ C.T  # (T, k) per copy... need (n,k)->flatten
    n = H.shape[1]
    z_seq = z_seq.reshape(T_, -1)  # (T, n*k)
    return actor_critic_loss(z_seq, actions, advantages, returns, head)


def verify_ours_stage1(arch, h0, U_seq, actions, advantages, returns, head):
    """Recurrent-core families via factorized RTRL (unchanged from
    B25); head families via plain autodiff on the actual z sequence
    (no RTRL needed -- heads are not recurrent). BPTT: one jax.grad
    call over everything, the independent reference.

    REAL BUG FOUND AND FIXED HERE (caught by this Stage-1 harness,
    not present in B27's own frozen/accepted result): the loss reads
    z_seq=H@C.T DIRECTLY (the same pattern used throughout B25/B26/B27's
    own training loops), so C has TWO roles -- inside the recurrence
    (via Phi(C@h_t,...), already handled by factorized_rtrl_run's
    direct_term) AND as the loss's own direct readout (z_t=H_t@C.T,
    NOT captured by factorized_rtrl_run at all, since it only accounts
    for how C affects h_t's own recurrence). Fixed by adding the
    missing direct term explicitly: d(loss)/dC|_{H fixed}, i.e. the
    loss's OWN direct dependence on C holding the (already-computed)
    state sequence constant -- exactly the same "shared parameter,
    two roles" lesson as B25.1's z_direct_term and B26's C-cross-layer
    fix, recurring in a new context (loss-level readout, not
    cross-layer routing)."""
    C = arch["C"]

    def target_fn(H):
        return ours_target_fn(H, C, head, actions, advantages, returns)

    dLdh = dLdh_from_target(arch, h0, U_seq, target_fn)

    H_actual, _, _ = rollout(h0, U_seq, arch)

    ours_grads = {}
    for family in ("R", "B", "C", "psi"):
        g, d = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=False)
        if family == "C":
            def loss_of_C(C_):
                return ours_target_fn(H_actual, C_, head, actions, advantages, returns)
            extra = np.asarray(jax.grad(loss_of_C)(C)).reshape(-1)
            g = g + extra
        ours_grads[family] = g

    def head_loss(head_):
        H, _, _ = rollout(h0, U_seq, arch)
        return ours_target_fn(H, C, head_, actions, advantages, returns)

    head_grads = jax.grad(head_loss)(head)

    # BPTT reference: everything in one call
    def full_loss(R, B, C_, psi, head_):
        arch_ = dict(arch, R=R, B=B, C=C_, psi=psi)
        H, _, _ = rollout(h0, U_seq, arch_)
        return ours_target_fn(H, C_, head_, actions, advantages, returns)

    g_bptt = jax.grad(full_loss, argnums=(0, 1, 2, 3, 4))(
        arch["R"], arch["B"], arch["C"], arch["psi"], head)

    errs = {}
    errs["R"] = float(jnp.max(jnp.abs(g_bptt[0].reshape(-1) - ours_grads["R"])))
    errs["B"] = float(jnp.max(jnp.abs(g_bptt[1].reshape(-1) - ours_grads["B"])))
    errs["C"] = float(jnp.max(jnp.abs(g_bptt[2].reshape(-1) - ours_grads["C"])))
    from credit_memory.b25_nonlinear_credit import psi_flat
    errs["psi"] = float(jnp.max(jnp.abs(psi_flat(g_bptt[3]) - ours_grads["psi"])))
    for k in ("W_pi", "b_pi", "W_v", "b_v"):
        errs[f"head.{k}"] = float(jnp.max(jnp.abs(g_bptt[4][k] - head_grads[k])))
    return errs


# ---------------------------------------------------------------------------
# RTU: embedding (fixed one-hot) -> Nonlinear RTU core -> policy/value heads
# (heads read the FULL r_rtu-dim state directly, matching RTU's own
# full-state-head design from B27). rtu_exact_credit_grad (B27) was
# hardcoded to a fixed scalar-MSE head; generalized here to accept an
# arbitrary precomputed per-timestep dL/dh_next (T, r_rtu), matching
# B25's dLdh_seq pattern -- so the SAME verified block-decoupled
# recurrence (diag(f') update, 2-dim per-block trace) now composes
# with any downstream loss, not just the one it was originally tested
# against.
# ---------------------------------------------------------------------------
def rtu_rollout_states(h0, U_seq, arch):
    """Returns H: (T+1, r_rtu) full state sequence (H[0]=h0)."""
    def step(h, u):
        h_next, _ = nonlinear_rtu_pre_and_next(h, u, arch)
        return h_next, h_next
    _, Hs = jax.lax.scan(step, h0, U_seq)
    return jnp.concatenate([h0[None], Hs], axis=0)


def rtu_dLdh_from_loss(arch, h0, U_seq, loss_of_H):
    H = rtu_rollout_states(h0, U_seq, arch)
    g = jax.grad(loss_of_H)(H)
    return np.asarray(g[1:])  # (T, r_rtu), matches dLdh_from_target's convention


def rtu_exact_credit_grad_general(arch, U_seq, dLdh_seq, block_idx, family):
    """Same verified block-local recurrence as B27's rtu_exact_credit_grad,
    generalized to accept an arbitrary precomputed dL/dh_next sequence
    instead of a hardcoded scalar-MSE head."""
    i = block_idx
    theta_i, logr_i = arch["thetas"][i], arch["log_radii"][i]
    Wx1_i, Wx2_i = arch["Wx"][2 * i], arch["Wx"][2 * i + 1]
    A_block = rtu_block_matrix(theta_i, logr_i)

    if family in ("theta", "log_radius"):
        m = 1
    else:
        m = 2 * arch["u_dim"]

    h = jnp.zeros(arch["r_rtu"])
    E = np.zeros((2, m))
    grad = np.zeros(m)
    T_ = U_seq.shape[0]

    for t in range(T_):
        u_t = U_seq[t]
        h_next, pre = nonlinear_rtu_pre_and_next(h, u_t, arch)
        pre_i = np.asarray(pre[2 * i:2 * i + 2])
        fprime = 1.0 - np.tanh(pre_i) ** 2
        h_i = np.asarray(h[2 * i:2 * i + 2])

        def f_theta(th):
            if family == "theta":
                Ablk = rtu_block_matrix(th[0], logr_i)
                pre_local = Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
            elif family == "log_radius":
                Ablk = rtu_block_matrix(theta_i, th[0])
                pre_local = Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
            else:
                Wx1n, Wx2n = th[:arch["u_dim"]], th[arch["u_dim"]:]
                pre_local = A_block @ h_i + jnp.array([Wx1n @ u_t, Wx2n @ u_t])
            return pre_local

        if family == "theta":
            th0 = jnp.array([theta_i])
        elif family == "log_radius":
            th0 = jnp.array([logr_i])
        else:
            th0 = jnp.concatenate([Wx1_i, Wx2_i])
        Q_theta = np.asarray(jax.jacobian(f_theta)(th0))

        E_pre = np.asarray(A_block) @ E + Q_theta
        E = fprime[:, None] * E_pre

        dLdh_next_i = np.asarray(dLdh_seq[t])[2 * i:2 * i + 2]
        grad += dLdh_next_i @ E
        h = h_next
    return grad


def verify_rtu_stage1(arch, U_seq, actions, advantages, returns, head):
    r_rtu = arch["r_rtu"]
    h0 = jnp.zeros(r_rtu)

    def loss_of_H(H):
        T_ = len(actions)
        z_seq = H[1:T_ + 1]  # (T, r_rtu) -- full state IS the head's input for RTU
        return actor_critic_loss(z_seq, actions, advantages, returns, head)

    dLdh = rtu_dLdh_from_loss(arch, h0, U_seq, loss_of_H)

    ours_style_grads = {}
    for family in ("theta", "log_radius", "Wx"):
        gs = []
        for b in range(arch["n_blocks"]):
            gs.append(rtu_exact_credit_grad_general(arch, U_seq, dLdh, b, family))
        ours_style_grads[family] = np.concatenate(gs)

    def head_loss(head_):
        H = rtu_rollout_states(h0, U_seq, arch)
        T_ = len(actions)
        return actor_critic_loss(H[1:T_ + 1], actions, advantages, returns, head_)

    head_grads = jax.grad(head_loss)(head)

    def full_loss(thetas, log_radii, Wx, head_):
        arch_ = dict(arch, thetas=thetas, log_radii=log_radii, Wx=Wx)
        H = rtu_rollout_states(h0, U_seq, arch_)
        T_ = len(actions)
        return actor_critic_loss(H[1:T_ + 1], actions, advantages, returns, head_)

    g_bptt = jax.grad(full_loss, argnums=(0, 1, 2, 3))(
        arch["thetas"], arch["log_radii"], arch["Wx"], head)

    errs = {}
    errs["theta"] = float(jnp.max(jnp.abs(g_bptt[0] - ours_style_grads["theta"])))
    errs["log_radius"] = float(jnp.max(jnp.abs(g_bptt[1] - ours_style_grads["log_radius"])))
    errs["Wx"] = float(jnp.max(jnp.abs(g_bptt[2].reshape(-1) - ours_style_grads["Wx"])))
    for k in ("W_pi", "b_pi", "W_v", "b_v"):
        errs[f"head.{k}"] = float(jnp.max(jnp.abs(g_bptt[3][k] - head_grads[k])))
    return errs


def main():
    from popgym.envs.repeat_first import RepeatFirstEasy
    from popgym.envs.autoencode import AutoencodeEasy

    tasks = [
        ("RepeatFirst", RepeatFirstEasy(), "discrete4", 10),
        ("Autoencode", AutoencodeEasy(), "autoencode", 12),
    ]
    for name, env, kind, T_ in tasks:
        print("=" * 70)
        print(f"STAGE 1 CORRECTNESS HARNESS -- {name}")
        print("=" * 70)
        obses, actions, rewards = collect_trajectory(env, T_=T_, seed=0)
        U_seq = jnp.array(np.stack([one_hot_obs(o, kind) for o in obses]))
        advantages, returns = make_advantages_returns(rewards)
        u_dim = U_seq.shape[1]
        print(f"  T={len(actions)} u_dim={u_dim}")

        r, k, n = 3, 2, 3
        arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=8, seed=1)
        h0 = jnp.zeros((n, r))
        head = make_head_params(z_dim=n * k, num_actions=4, seed=2)
        errs = verify_ours_stage1(arch, h0, U_seq, actions, advantages, returns, head)
        print(f"  OURS  errs: {errs}")

        rtu_arch = make_nonlinear_rtu_arch(r_rtu=6, u_dim=u_dim, hidden=8, seed=1)
        head_rtu = make_head_params(z_dim=6, num_actions=4, seed=2)
        errs_rtu = verify_rtu_stage1(rtu_arch, U_seq, actions, advantages, returns, head_rtu)
        print(f"  RTU   errs: {errs_rtu}")
        print()


if __name__ == "__main__":
    main()
