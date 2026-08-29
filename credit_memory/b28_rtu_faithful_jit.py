"""Phase B28 -- JIT rewrite of the faithful RTU calibration hot path.

Commit 72a4f7b (b28_rtu_faithful.py + b28_rtu_faithful_train.py) is the
FROZEN semantic reference. This file is a PURE EXECUTION-OPTIMIZATION
rewrite: same architecture, same hyperparameters, same RTU equations,
same RTRL equations, same normalization semantics, same entropy term,
same ObGD, same num_envs=1, same frame accounting. Nothing here changes
what is computed -- only how (pure jnp instead of numpy, one pytree
carry, one jax.jit'd step function).

The online POPGym environment itself remains Python/eager (env.step()
is not JAX-compatible) -- only the model/RTRL/update math is compiled.

Run tests: python -m credit_memory.b28_rtu_faithful_jit
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util

import popgym

from credit_memory.b28_popgym_stage1 import one_hot_obs
from credit_memory import b28_rtu_faithful as ref  # frozen eager reference
from credit_memory import b28_rtu_faithful_train as reft  # frozen eager reference

jax.config.update("jax_enable_x64", True)

RTU_FAMILIES = ref.RTU_FAMILIES
ENCODER_FAMILIES = ref.ENCODER_FAMILIES
ALL_NET_FAMILIES = ref.ALL_NET_FAMILIES
NET_FAMILY_ROLES = reft.NET_FAMILY_ROLES


# ---------------------------------------------------------------------------
# Pure-jnp RTU + encoder forward and EXACT per-step RTRL sensitivity.
# Identical formulas to b28_rtu_faithful.py's rtu_streaming_step /
# net_streaming_step, rewritten with no numpy, no Python dict mutation,
# no host<->device round-trips -- a single functional step returning
# new state rather than mutating in place.
# ---------------------------------------------------------------------------
def rtu_g_phi_norm(nu_log, theta_log, eps=1e-8):
    r = jnp.exp(-jnp.exp(nu_log))
    theta = jnp.exp(theta_log)
    g = r * jnp.cos(theta)
    phi = r * jnp.sin(theta)
    norm = jnp.sqrt(1.0 - r**2) + eps
    return g, phi, norm, r


def encoder_forward(obs_onehot, enc_params, eps=1e-6):
    pre = obs_onehot @ enc_params["W_enc"] + enc_params["b_enc"]
    mean = jnp.mean(pre)
    var = jnp.var(pre)
    normed = (pre - mean) / jnp.sqrt(var + eps)
    ln_out = normed * enc_params["ln_scale"] + enc_params["ln_bias"]
    return jnp.where(ln_out >= 0, ln_out, ref.LEAKY_RELU_SLOPE * ln_out)


def encoder_jacobian(obs_onehot, enc_params):
    jacs = {}
    for fam in ENCODER_FAMILIES:
        def f(p, fam=fam):
            merged = dict(enc_params)
            merged[fam] = p
            return encoder_forward(obs_onehot, merged)
        jacs[fam] = jax.jacobian(f)(enc_params[fam])
    return jacs


def net_streaming_step_jit(rtu_params, enc_params, real, imag, S, obs_onehot, eps=1e-8):
    """Pure-jnp, functional version of net_streaming_step. Returns
    (new_real, new_imag, output, new_S) -- no mutation, no numpy."""
    nu_log, theta_log = rtu_params["nu_log"], rtu_params["theta_log"]
    g, phi, norm, r = rtu_g_phi_norm(nu_log, theta_log, eps)

    x_t = encoder_forward(obs_onehot, enc_params)
    u_real = rtu_params["B_real"] @ x_t
    u_imag = rtu_params["B_imag"] @ x_t
    pre_real = g * real - phi * imag + norm * u_real
    pre_imag = g * imag + phi * real + norm * u_imag
    new_real = jnp.tanh(pre_real)
    new_imag = jnp.tanh(pre_imag)
    output = jnp.concatenate([new_real, new_imag])

    d_real = 1.0 - new_real**2
    d_imag = 1.0 - new_imag**2
    d = jnp.stack([d_real, d_imag], axis=0)
    A = jnp.stack([jnp.stack([g, -phi]), jnp.stack([phi, g])])

    exp_nu = jnp.exp(nu_log)
    dg_dnu = -exp_nu * g
    dphi_dnu = -exp_nu * phi
    dnorm_dnu = exp_nu * r**2 / (jnp.sqrt(1 - r**2) + 1e-12)
    theta = jnp.exp(theta_log)
    dg_dtheta = -phi * theta
    dphi_dtheta = g * theta

    Bu = norm[:, None] * x_t[None, :]
    zeros_hF = jnp.zeros_like(Bu)

    J = dict(
        nu_log=jnp.stack([
            dg_dnu * real - dphi_dnu * imag + dnorm_dnu * u_real,
            dg_dnu * imag + dphi_dnu * real + dnorm_dnu * u_imag,
        ], axis=0),
        theta_log=jnp.stack([
            dg_dtheta * real - dphi_dtheta * imag,
            dg_dtheta * imag + dphi_dtheta * real,
        ], axis=0),
        B_real=jnp.stack([Bu, zeros_hF], axis=0),
        B_imag=jnp.stack([zeros_hF, Bu], axis=0),
    )

    enc_jacs = encoder_jacobian(obs_onehot, enc_params)
    B_real, B_imag = rtu_params["B_real"], rtu_params["B_imag"]
    for fam in ENCODER_FAMILIES:
        Jx = enc_jacs[fam]  # (features, *enc_param_shape)
        extra = (1,) * (Jx.ndim - 1)
        real_term = jnp.tensordot(B_real, Jx, axes=([1], [0])) * norm.reshape((-1,) + extra)
        imag_term = jnp.tensordot(B_imag, Jx, axes=([1], [0])) * norm.reshape((-1,) + extra)
        J[fam] = jnp.stack([real_term, imag_term], axis=0)

    new_S = {}
    for fam in ALL_NET_FAMILIES:
        Sf = S[fam]
        rotated = jnp.einsum("ijh,jh...->ih...", A, Sf)
        extra = (None,) * (Sf.ndim - 2)
        new_S[fam] = d[(slice(None), slice(None)) + extra] * (rotated + J[fam])

    return new_real, new_imag, output, new_S


def net_per_step_grad_jit(S, dLdz_split):
    grads = {}
    for fam, role in NET_FAMILY_ROLES.items():
        if role == "diag":
            grads[fam] = jnp.einsum("ih,ih...->h...", dLdz_split, S[fam])
        else:
            grads[fam] = jnp.einsum("ih,ih...->...", dLdz_split, S[fam])
    return grads


def obgd_step_size_jit(alpha, kappa, delta, trace_leaves):
    delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
    z_l1 = sum(jnp.sum(jnp.abs(z)) for z in trace_leaves)
    M = alpha * kappa * delta_bar * z_l1
    return jnp.where(M > 1.0, alpha / M, alpha)


def head_forward(z, head_params):
    return head_params["W"] @ z + head_params["b"]


def categorical_entropy(logits):
    logp = logits - jax.scipy.special.logsumexp(logits)
    p = jnp.exp(logp)
    return -jnp.sum(p * logp)


# ---------------------------------------------------------------------------
# Carry pytree: everything the compiled step needs, as jnp arrays only.
# Static shapes throughout (fixed by hidden_dim/width/in_dim/num_actions
# at construction time) -- no Python scalars/dicts of varying structure
# enter the compiled functions themselves (config is closed over via
# functools.partial with STATIC ints, not passed as pytree leaves).
# ---------------------------------------------------------------------------
def net_carry_from_eager(net, stream_state, traces):
    return dict(
        params=dict(
            enc={k: jnp.asarray(v) for k, v in net["enc"].items()},
            rtu={k: jnp.asarray(v) for k, v in net["rtu"].items()},
            head={k: jnp.asarray(v) for k, v in net["head"].items()},
        ),
        traces=dict(
            enc={k: jnp.asarray(traces[("enc", k)]) for k in net["enc"]},
            rtu={k: jnp.asarray(traces[("rtu", k)]) for k in net["rtu"]},
            head={k: jnp.asarray(traces[("head", k)]) for k in net["head"]},
        ),
        real=jnp.asarray(stream_state["real"]),
        imag=jnp.asarray(stream_state["imag"]),
        S={fam: jnp.asarray(stream_state["S"][fam]) for fam in ALL_NET_FAMILIES},
    )


def make_carry(actor_net, actor_stream, actor_traces, critic_net, critic_stream, critic_traces,
                obs_stats, reward_stats):
    return dict(
        actor=net_carry_from_eager(actor_net, actor_stream, actor_traces),
        critic=net_carry_from_eager(critic_net, critic_stream, critic_traces),
        obs_stats=dict(count=jnp.asarray(obs_stats["count"]), mean=jnp.asarray(obs_stats["mean"]),
                       M2=jnp.asarray(obs_stats["M2"])),
        reward_stats=dict(count=jnp.asarray(reward_stats["count"]), mean=jnp.asarray(reward_stats["mean"]),
                          M2=jnp.asarray(reward_stats["M2"])),
    )


def _running_update(stats, x):
    count = stats["count"] + 1.0
    delta = x - stats["mean"]
    mean = stats["mean"] + delta / count
    delta2 = x - mean
    M2 = stats["M2"] + delta * delta2
    return dict(count=count, mean=mean, M2=M2)


def _running_normalize(stats, x, eps=1e-8):
    var = stats["M2"] / jnp.maximum(stats["count"], 1.0)
    return (x - stats["mean"]) / (jnp.sqrt(var) + eps)


def _net_z(net_carry, hidden_dim):
    return jnp.concatenate([net_carry["real"], net_carry["imag"]])


def predict_step(carry):
    """Cheap: logits + value from the carry's CURRENT (already-consumed)
    state -- no gradients, no trace updates. Used for action sampling."""
    z_actor = jnp.concatenate([carry["actor"]["real"], carry["actor"]["imag"]])
    z_critic = jnp.concatenate([carry["critic"]["real"], carry["critic"]["imag"]])
    logits = head_forward(z_actor, carry["actor"]["params"]["head"])
    value = head_forward(z_critic, carry["critic"]["params"]["head"])[0]
    return logits, value


def full_update_step(carry, action, raw_reward, raw_obs_next, done,
                      gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                      actor_kappa, critic_kappa):
    """The complete per-step learner update: consumes the next
    observation (if not done) into BOTH networks using CURRENT
    (pre-update) params, computes the TD error from OLD z/value,
    computes exact per-step gradients via the CURRENT step's S
    (captured before this call's consumption), updates traces, computes
    ObGD step sizes, and applies the parameter update. Returns new_carry.
    `done` is a traced 0/1 float; both branches are always computed and
    combined via jnp.where (required for jax.jit)."""
    actor, critic = carry["actor"], carry["critic"]

    z_actor_cur = jnp.concatenate([actor["real"], actor["imag"]])
    z_critic_cur = jnp.concatenate([critic["real"], critic["imag"]])
    value_cur = head_forward(z_critic_cur, critic["params"]["head"])[0]
    logits_cur = head_forward(z_actor_cur, actor["params"]["head"])

    obs_stats_next = _running_update(carry["obs_stats"], raw_obs_next)
    obs_next_norm = _running_normalize(obs_stats_next, raw_obs_next)

    a_real, a_imag, _, a_S_next = net_streaming_step_jit(
        actor["params"]["rtu"], actor["params"]["enc"], actor["real"], actor["imag"], actor["S"], obs_next_norm
    )
    c_real, c_imag, _, c_S_next = net_streaming_step_jit(
        critic["params"]["rtu"], critic["params"]["enc"], critic["real"], critic["imag"], critic["S"], obs_next_norm
    )
    z_critic_next = jnp.concatenate([c_real, c_imag])
    value_next_raw = head_forward(z_critic_next, critic["params"]["head"])[0]
    value_next = jnp.where(done > 0.5, 0.0, value_next_raw)

    reward_stats_next = _running_update(carry["reward_stats"], raw_reward)
    reward_norm = _running_normalize(reward_stats_next, raw_reward)

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
    dactor_target_dz = dlogp_dz + entropy_coef * jnp.sign(td_error) * dentropy_dz
    dv_dz = jax.grad(value_fn)(z_critic_cur)

    hidden_dim = actor["real"].shape[0]
    dactor_split = jnp.stack([dactor_target_dz[:hidden_dim], dactor_target_dz[hidden_dim:]], axis=0)
    dv_split = jnp.stack([dv_dz[:hidden_dim], dv_dz[hidden_dim:]], axis=0)

    g_actor = net_per_step_grad_jit(actor["S"], dactor_split)
    g_critic = net_per_step_grad_jit(critic["S"], dv_split)

    def actor_target(head_params):
        logits = head_forward(z_actor_cur, head_params)
        logp = logits - jax.scipy.special.logsumexp(logits)
        return logp[action] + entropy_coef * jnp.sign(td_error) * categorical_entropy(logits)

    def critic_target(head_params):
        return head_forward(z_critic_cur, head_params)[0]

    gh_actor = jax.grad(actor_target)(actor["params"]["head"])
    gh_critic = jax.grad(critic_target)(critic["params"]["head"])

    def update_group(params_g, traces_g, grads_g):
        new_traces = {k: gamma * lam * traces_g[k] + grads_g[k] for k in params_g}
        return new_traces

    new_actor_traces_enc = update_group(actor["params"]["enc"], actor["traces"]["enc"],
                                         {k: g_actor[k] for k in ENCODER_FAMILIES})
    new_actor_traces_rtu = update_group(actor["params"]["rtu"], actor["traces"]["rtu"],
                                         {k: g_actor[k] for k in RTU_FAMILIES})
    new_actor_traces_head = update_group(actor["params"]["head"], actor["traces"]["head"], gh_actor)

    new_critic_traces_enc = update_group(critic["params"]["enc"], critic["traces"]["enc"],
                                          {k: g_critic[k] for k in ENCODER_FAMILIES})
    new_critic_traces_rtu = update_group(critic["params"]["rtu"], critic["traces"]["rtu"],
                                          {k: g_critic[k] for k in RTU_FAMILIES})
    new_critic_traces_head = update_group(critic["params"]["head"], critic["traces"]["head"], gh_critic)

    actor_trace_leaves = (list(new_actor_traces_enc.values()) + list(new_actor_traces_rtu.values())
                          + list(new_actor_traces_head.values()))
    critic_trace_leaves = (list(new_critic_traces_enc.values()) + list(new_critic_traces_rtu.values())
                           + list(new_critic_traces_head.values()))

    step_actor = obgd_step_size_jit(actor_alpha, actor_kappa, td_error, actor_trace_leaves)
    step_critic = obgd_step_size_jit(critic_alpha, critic_kappa, td_error, critic_trace_leaves)

    def apply_update(params_g, traces_g):
        return {k: params_g[k] + step_actor * td_error * traces_g[k] for k in params_g}

    def apply_update_critic(params_g, traces_g):
        return {k: params_g[k] + step_critic * td_error * traces_g[k] for k in params_g}

    new_actor_params = dict(
        enc=apply_update(actor["params"]["enc"], new_actor_traces_enc),
        rtu=apply_update(actor["params"]["rtu"], new_actor_traces_rtu),
        head=apply_update(actor["params"]["head"], new_actor_traces_head),
    )
    new_critic_params = dict(
        enc=apply_update_critic(critic["params"]["enc"], new_critic_traces_enc),
        rtu=apply_update_critic(critic["params"]["rtu"], new_critic_traces_rtu),
        head=apply_update_critic(critic["params"]["head"], new_critic_traces_head),
    )

    not_done = 1.0 - done
    new_actor = dict(
        params=new_actor_params,
        traces=dict(enc=new_actor_traces_enc, rtu=new_actor_traces_rtu, head=new_actor_traces_head),
        real=not_done * a_real + done * actor["real"],
        imag=not_done * a_imag + done * actor["imag"],
        S={fam: not_done * a_S_next[fam] + done * actor["S"][fam] for fam in ALL_NET_FAMILIES},
    )
    new_critic = dict(
        params=new_critic_params,
        traces=dict(enc=new_critic_traces_enc, rtu=new_critic_traces_rtu, head=new_critic_traces_head),
        real=not_done * c_real + done * critic["real"],
        imag=not_done * c_imag + done * critic["imag"],
        S={fam: not_done * c_S_next[fam] + done * critic["S"][fam] for fam in ALL_NET_FAMILIES},
    )

    new_carry = dict(
        actor=new_actor, critic=new_critic,
        obs_stats=obs_stats_next, reward_stats=reward_stats_next,
    )
    diagnostics = dict(td_error=td_error, value_cur=value_cur, value_next=value_next,
                        step_actor=step_actor, step_critic=step_critic,
                        entropy=categorical_entropy(logits_cur), logits_cur=logits_cur)
    return new_carry, diagnostics


# ---------------------------------------------------------------------------
# Deterministic multi-step parity harness: EAGER (b28_rtu_faithful_train
# machinery, numpy-based) vs JIT (this file), over an IDENTICAL fixed
# transition sequence, identical initial params/carry/RNG seed.
# ---------------------------------------------------------------------------
def record_fixed_trajectory(seed, n_steps, in_dim=6):
    """Records a real POPGym Autoencode trajectory (random policy) ONCE
    -- used identically by both eager and JIT runs, removing env
    randomness from the comparison."""
    env = popgym.envs.autoencode.AutoencodeEasy()
    rng = np.random.RandomState(seed)
    obs, _ = env.reset(seed=seed)
    obs_seq = [np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64)]
    action_seq, reward_seq, done_seq = [], [], []
    for _ in range(n_steps):
        a = int(rng.randint(0, env.action_space.n))
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        action_seq.append(a)
        reward_seq.append(float(r))
        done_seq.append(bool(done))
        obs_seq.append(np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64))
        if done:
            break
    return obs_seq, action_seq, reward_seq, done_seq


def run_eager_trajectory(seed, obs_seq, action_seq, reward_seq, done_seq,
                          hidden_dim, width, in_dim, num_actions,
                          gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                          actor_kappa, critic_kappa):
    """Mirrors b28_rtu_faithful_train.smoke_test's per-step logic
    EXACTLY, but consumes a PRE-RECORDED transition sequence and
    returns per-step diagnostics for parity comparison."""
    rng = np.random.RandomState(seed)
    actor = reft.make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic = reft.make_network(rng, in_dim, width, hidden_dim, 1)
    actor_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    critic_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    obs_stats = reft.running_stats_init((in_dim,))
    reward_stats = reft.running_stats_init(())
    actor_traces = reft.zero_traces(actor)
    critic_traces = reft.zero_traces(critic)

    u0 = obs_seq[0]
    reft.running_stats_update(obs_stats, u0)
    u0n = jnp.array(reft.running_stats_normalize(obs_stats, u0))
    z_actor, S_actor = reft.network_step(actor, actor_stream, u0n)
    z_critic, S_critic = reft.network_step(critic, critic_stream, u0n)

    diagnostics = []
    for t in range(len(action_seq)):
        a_t, r_t, done = action_seq[t], reward_seq[t], done_seq[t]

        logits_cur = reft.head_forward(z_actor, actor["head"])
        v_cur = float(reft.head_forward(z_critic, critic["head"])[0])

        reft.running_stats_update(reward_stats, r_t)
        r_t_norm = float(reft.running_stats_normalize(reward_stats, r_t))

        if done:
            v_next = 0.0
            z_next_actor = z_next_critic = S_next_actor = S_next_critic = None
        else:
            u_next = obs_seq[t + 1]
            reft.running_stats_update(obs_stats, u_next)
            u_next_n = jnp.array(reft.running_stats_normalize(obs_stats, u_next))
            z_next_actor, S_next_actor = reft.network_step(actor, actor_stream, u_next_n)
            z_next_critic, S_next_critic = reft.network_step(critic, critic_stream, u_next_n)
            v_next = float(reft.head_forward(z_next_critic, critic["head"])[0])

        td_error = r_t_norm + gamma * v_next - v_cur

        logp_fn = lambda z: (reft.head_forward(z, actor["head"]) - jax.scipy.special.logsumexp(reft.head_forward(z, actor["head"])))[a_t]
        entropy_fn = lambda z: reft.categorical_entropy(reft.head_forward(z, actor["head"]))
        dlogp_dz = np.asarray(jax.grad(logp_fn)(z_actor))
        dentropy_dz = np.asarray(jax.grad(entropy_fn)(z_actor))
        dactor_target_dz = dlogp_dz + entropy_coef * np.sign(td_error) * dentropy_dz
        dv_dz = np.asarray(jax.grad(lambda z: reft.head_forward(z, critic["head"])[0])(z_critic))

        dactor_split = np.stack([dactor_target_dz[:hidden_dim], dactor_target_dz[hidden_dim:]], axis=0)
        dv_split = np.stack([dv_dz[:hidden_dim], dv_dz[hidden_dim:]], axis=0)

        g_actor = reft.net_per_step_grad(S_actor, dactor_split, reft.NET_FAMILY_ROLES)
        g_critic = reft.net_per_step_grad(S_critic, dv_split, reft.NET_FAMILY_ROLES)

        for k, v in g_actor.items():
            grp = "rtu" if k in reft.RTU_FAMILIES else "enc"
            actor_traces[(grp, k)] = gamma * lam * actor_traces[(grp, k)] + v
        for k, v in g_critic.items():
            grp = "rtu" if k in reft.RTU_FAMILIES else "enc"
            critic_traces[(grp, k)] = gamma * lam * critic_traces[(grp, k)] + v

        gh_pi_W = np.asarray(jax.grad(lambda W: (W @ z_actor + actor["head"]["b"] - jax.scipy.special.logsumexp(W @ z_actor + actor["head"]["b"]))[a_t]
                                       + entropy_coef * np.sign(td_error) * reft.categorical_entropy(W @ z_actor + actor["head"]["b"]))(actor["head"]["W"]))
        gh_pi_b = np.asarray(jax.grad(lambda b: (actor["head"]["W"] @ z_actor + b - jax.scipy.special.logsumexp(actor["head"]["W"] @ z_actor + b))[a_t]
                                       + entropy_coef * np.sign(td_error) * reft.categorical_entropy(actor["head"]["W"] @ z_actor + b))(actor["head"]["b"]))
        gh_v_W = np.asarray(jax.grad(lambda W: (W @ z_critic + critic["head"]["b"])[0])(critic["head"]["W"]))
        gh_v_b = np.asarray(jax.grad(lambda b: (critic["head"]["W"] @ z_critic + b)[0])(critic["head"]["b"]))
        actor_traces[("head", "W")] = gamma * lam * actor_traces[("head", "W")] + gh_pi_W
        actor_traces[("head", "b")] = gamma * lam * actor_traces[("head", "b")] + gh_pi_b
        critic_traces[("head", "W")] = gamma * lam * critic_traces[("head", "W")] + gh_v_W
        critic_traces[("head", "b")] = gamma * lam * critic_traces[("head", "b")] + gh_v_b

        step_actor = reft.obgd_step_size(actor_alpha, actor_kappa, td_error, actor_traces)
        step_critic = reft.obgd_step_size(critic_alpha, critic_kappa, td_error, critic_traces)

        for (grp, k), ztr in actor_traces.items():
            actor[grp][k] = actor[grp][k] + step_actor * td_error * ztr
        for (grp, k), ztr in critic_traces.items():
            critic[grp][k] = critic[grp][k] + step_critic * td_error * ztr

        diagnostics.append(dict(
            z_actor=np.asarray(z_actor), z_critic=np.asarray(z_critic),
            logits_cur=np.asarray(logits_cur), value_cur=v_cur, value_next=v_next,
            td_error=td_error, entropy=float(reft.categorical_entropy(logits_cur)),
            step_actor=step_actor, step_critic=step_critic,
            actor_head_W=np.asarray(actor["head"]["W"]), critic_head_W=np.asarray(critic["head"]["W"]),
            actor_rtu_nu_log=np.asarray(actor["rtu"]["nu_log"]), critic_rtu_nu_log=np.asarray(critic["rtu"]["nu_log"]),
            obs_stats_mean=np.asarray(obs_stats["mean"]).copy(), reward_stats_mean=float(reward_stats["mean"]),
        ))

        if done:
            break
        z_actor, S_actor = z_next_actor, S_next_actor
        z_critic, S_critic = z_next_critic, S_next_critic

    return diagnostics


def run_jit_trajectory(seed, obs_seq, action_seq, reward_seq, done_seq,
                        hidden_dim, width, in_dim, num_actions,
                        gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                        actor_kappa, critic_kappa, jit_fns=None):
    rng = np.random.RandomState(seed)
    actor_net = reft.make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic_net = reft.make_network(rng, in_dim, width, hidden_dim, 1)
    actor_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    critic_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    obs_stats = reft.running_stats_init((in_dim,))
    reward_stats = reft.running_stats_init(())
    actor_traces = reft.zero_traces(actor_net)
    critic_traces = reft.zero_traces(critic_net)

    carry = make_carry(actor_net, actor_stream, actor_traces, critic_net, critic_stream, critic_traces,
                        obs_stats, reward_stats)

    u0 = obs_seq[0]
    obs_stats0 = _running_update(carry["obs_stats"], jnp.asarray(u0))
    u0n = _running_normalize(obs_stats0, jnp.asarray(u0))
    a_real, a_imag, _, a_S = net_streaming_step_jit(
        carry["actor"]["params"]["rtu"], carry["actor"]["params"]["enc"],
        carry["actor"]["real"], carry["actor"]["imag"], carry["actor"]["S"], u0n)
    c_real, c_imag, _, c_S = net_streaming_step_jit(
        carry["critic"]["params"]["rtu"], carry["critic"]["params"]["enc"],
        carry["critic"]["real"], carry["critic"]["imag"], carry["critic"]["S"], u0n)
    carry["actor"] = dict(carry["actor"], real=a_real, imag=a_imag, S=a_S)
    carry["critic"] = dict(carry["critic"], real=c_real, imag=c_imag, S=c_S)
    carry["obs_stats"] = obs_stats0

    update_fn = jit_fns["update"] if jit_fns else full_update_step

    diagnostics = []
    for t in range(len(action_seq)):
        a_t, r_t, done = action_seq[t], reward_seq[t], done_seq[t]
        obs_next = obs_seq[t + 1] if t + 1 < len(obs_seq) else np.zeros(in_dim)

        new_carry, diag = update_fn(
            carry, jnp.asarray(a_t), jnp.asarray(float(r_t)), jnp.asarray(obs_next),
            jnp.asarray(1.0 if done else 0.0),
            gamma, lam, entropy_coef, actor_alpha, critic_alpha, actor_kappa, critic_kappa,
        )
        diagnostics.append(dict(
            z_actor=np.asarray(jnp.concatenate([carry["actor"]["real"], carry["actor"]["imag"]])),
            z_critic=np.asarray(jnp.concatenate([carry["critic"]["real"], carry["critic"]["imag"]])),
            logits_cur=np.asarray(diag["logits_cur"]), value_cur=float(diag["value_cur"]),
            value_next=float(diag["value_next"]), td_error=float(diag["td_error"]),
            entropy=float(diag["entropy"]), step_actor=float(diag["step_actor"]),
            step_critic=float(diag["step_critic"]),
            actor_head_W=np.asarray(new_carry["actor"]["params"]["head"]["W"]),
            critic_head_W=np.asarray(new_carry["critic"]["params"]["head"]["W"]),
            actor_rtu_nu_log=np.asarray(new_carry["actor"]["params"]["rtu"]["nu_log"]),
            critic_rtu_nu_log=np.asarray(new_carry["critic"]["params"]["rtu"]["nu_log"]),
            obs_stats_mean=np.asarray(new_carry["obs_stats"]["mean"]),
            reward_stats_mean=float(new_carry["reward_stats"]["mean"]),
        ))
        carry = new_carry
        if done:
            break

    return diagnostics


# ---------------------------------------------------------------------------
# Parity comparison + recompilation check + benchmark.
# ---------------------------------------------------------------------------
_TRACE_COUNTER = {"n": 0}


def _full_update_step_traced(carry, action, raw_reward, raw_obs_next, done,
                              gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                              actor_kappa, critic_kappa):
    _TRACE_COUNTER["n"] += 1
    return full_update_step(carry, action, raw_reward, raw_obs_next, done,
                             gamma, lam, entropy_coef, actor_alpha, critic_alpha,
                             actor_kappa, critic_kappa)


def compare_trajectories(eager_diag, jit_diag):
    keys = ["z_actor", "z_critic", "logits_cur", "value_cur", "value_next", "td_error",
            "entropy", "step_actor", "step_critic", "actor_head_W", "critic_head_W",
            "actor_rtu_nu_log", "critic_rtu_nu_log", "obs_stats_mean", "reward_stats_mean"]
    max_abs = {k: 0.0 for k in keys}
    max_rel = {k: 0.0 for k in keys}
    n = min(len(eager_diag), len(jit_diag))
    for t in range(n):
        for k in keys:
            a = np.asarray(eager_diag[t][k], dtype=np.float64)
            b = np.asarray(jit_diag[t][k], dtype=np.float64)
            abs_err = np.max(np.abs(a - b))
            rel_err = np.max(np.abs(a - b) / (np.abs(a) + 1e-12))
            max_abs[k] = max(max_abs[k], float(abs_err))
            max_rel[k] = max(max_rel[k], float(rel_err))
    return max_abs, max_rel, n


def test_eager_vs_jit_parity(seed=42, n_steps=15, hidden_dim=6, width=10, in_dim=6, num_actions=4):
    obs_seq, action_seq, reward_seq, done_seq = record_fixed_trajectory(seed, n_steps, in_dim)
    hp = dict(gamma=0.99, lam=0.8, entropy_coef=0.095, actor_alpha=1.0, critic_alpha=1.0,
              actor_kappa=3.0, critic_kappa=2.0)
    eager_diag = run_eager_trajectory(seed, obs_seq, action_seq, reward_seq, done_seq,
                                       hidden_dim, width, in_dim, num_actions, **hp)
    jit_diag = run_jit_trajectory(seed, obs_seq, action_seq, reward_seq, done_seq,
                                   hidden_dim, width, in_dim, num_actions, **hp)
    return compare_trajectories(eager_diag, jit_diag)


def check_recompilation(hidden_dim=8, width=12, in_dim=6, num_actions=4, n_calls=5):
    _TRACE_COUNTER["n"] = 0
    jitted = jax.jit(_full_update_step_traced, static_argnums=())
    rng = np.random.RandomState(0)
    actor_net = reft.make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic_net = reft.make_network(rng, in_dim, width, hidden_dim, 1)
    actor_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    critic_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    obs_stats = reft.running_stats_init((in_dim,))
    reward_stats = reft.running_stats_init(())
    carry = make_carry(actor_net, actor_stream, reft.zero_traces(actor_net),
                        critic_net, critic_stream, reft.zero_traces(critic_net), obs_stats, reward_stats)
    hp = (0.99, 0.8, 0.095, 1.0, 1.0, 3.0, 2.0)
    for i in range(n_calls):
        new_carry, _ = jitted(carry, jnp.asarray(i % num_actions), jnp.asarray(0.1),
                               jnp.asarray(np.eye(in_dim)[i % in_dim]), jnp.asarray(0.0), *hp)
        jax.block_until_ready(new_carry)
        carry = new_carry
    return _TRACE_COUNTER["n"]  # should be 1 regardless of n_calls


def benchmark_published_scale(n_steps=30, hidden_dim=192, width=64, in_dim=6, num_actions=4):
    rng = np.random.RandomState(0)
    actor_net = reft.make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic_net = reft.make_network(rng, in_dim, width, hidden_dim, 1)
    actor_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    critic_stream = reft.network_streaming_init(hidden_dim, width, in_dim)
    obs_stats = reft.running_stats_init((in_dim,))
    reward_stats = reft.running_stats_init(())
    carry = make_carry(actor_net, actor_stream, reft.zero_traces(actor_net),
                        critic_net, critic_stream, reft.zero_traces(critic_net), obs_stats, reward_stats)

    jitted = jax.jit(full_update_step)
    hp = (0.99, 0.8, 0.095, 1.0, 1.0, 3.0, 2.0)

    obs_next = jnp.asarray(np.eye(in_dim)[0])
    t0 = time.time()
    new_carry, diag = jitted(carry, jnp.asarray(0), jnp.asarray(0.1), obs_next, jnp.asarray(0.0), *hp)
    jax.block_until_ready(new_carry)
    compile_time = time.time() - t0

    carry = new_carry
    t0 = time.time()
    for i in range(n_steps):
        obs_next = jnp.asarray(np.eye(in_dim)[i % in_dim])
        new_carry, diag = jitted(carry, jnp.asarray(i % num_actions), jnp.asarray(0.1),
                                  obs_next, jnp.asarray(0.0), *hp)
        jax.block_until_ready(new_carry)
        carry = new_carry
    steady_time = time.time() - t0

    return compile_time, steady_time / n_steps, n_steps


def main():
    print("=" * 70)
    print("Eager-vs-JIT multi-step parity (15 steps, real POPGym Autoencode)")
    print("=" * 70)
    max_abs, max_rel, n = test_eager_vs_jit_parity()
    print(f"  compared {n} steps")
    for k in max_abs:
        print(f"  {k:18s} max_abs={max_abs[k]:.2e}  max_rel={max_rel[k]:.2e}")
    worst_abs = max(max_abs.values())
    assert worst_abs < 1e-6, f"PARITY FAILURE: worst max_abs={worst_abs:.2e}"
    print(f"  PASS (worst max_abs={worst_abs:.2e})")

    print("=" * 70)
    print("Recompilation check (5 calls, same shapes)")
    print("=" * 70)
    n_traces = check_recompilation()
    print(f"  trace count over 5 calls = {n_traces} (expected 1)")
    assert n_traces == 1, f"UNEXPECTED RECOMPILATION: traced {n_traces} times"
    print("  PASS")

    print("=" * 70)
    print("Rerunning existing 10 faithful-RTU unit/parity tests (frozen reference)")
    print("=" * 70)
    ref.main()
    reft.main()

    print("=" * 70)
    print("Benchmark: published scale (hidden=192, width=64), JIT'd full_update_step")
    print("=" * 70)
    compile_time, steady_spf, n_steps = benchmark_published_scale()
    print(f"  first-call (compile+exec) time: {compile_time:.3f}s")
    print(f"  steady-state: {steady_spf:.5f} s/step over {n_steps} steps -> {1.0/steady_spf:.1f} frames/sec")
    for target in (100_000, 500_000, 1_000_000, 5_000_000):
        est_hours = target * steady_spf / 3600
        print(f"  estimated time for {target:,} frames: {est_hours:.2f} hours")


if __name__ == "__main__":
    main()
