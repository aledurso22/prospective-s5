"""Phase B28 -- faithful RTU calibration: heads, running normalization,
ObGD-with-entropy, separate actor/critic networks, frame-based training
loop, and the remaining parity/sanity unit tests (items 2-10 of the
requested 10; item 1 lives in b28_rtu_faithful.py).

Does NOT reintroduce "ours" and does NOT run any long training here --
this module provides the machinery plus short smoke tests; the actual
calibration run is launched separately once this file's tests pass.

Run tests: python -m credit_memory.b28_rtu_faithful_train
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

import popgym

from credit_memory.b28_rtu_faithful import (
    RTU_FAMILIES, ENCODER_FAMILIES, ALL_NET_FAMILIES,
    make_rtu_params, make_encoder_params, net_streaming_init, net_streaming_step,
    rtu_g_phi_norm,
)
from credit_memory.b28_popgym_stage1 import one_hot_obs

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Heads: linear policy (softmax) + linear value. Lecun-normal init
# (no sparse-init evidence found for head layers specifically).
# ---------------------------------------------------------------------------
HEAD_POLICY_FAMILIES = ("W_pi", "b_pi")
HEAD_VALUE_FAMILIES = ("W_v", "b_v")


def make_head_params(rng, in_dim, out_dim):
    scale = 1.0 / np.sqrt(in_dim)
    return dict(
        W=jnp.array(rng.randn(out_dim, in_dim) * scale),
        b=jnp.zeros(out_dim),
    )


def head_forward(z, head_params):
    return head_params["W"] @ z + head_params["b"]


# ---------------------------------------------------------------------------
# Running observation/reward normalization (Welford's online algorithm
# for mean/variance -- exact statistic form not stated in the paper
# text I could access; this is a standard, defensible choice, FLAGGED
# as not byte-confirmed against the exact repository code).
# ---------------------------------------------------------------------------
def running_stats_init(shape=()):
    return dict(count=1e-4, mean=np.zeros(shape), M2=np.ones(shape))


def running_stats_update(stats, x):
    x = np.asarray(x, dtype=np.float64)
    stats["count"] += 1
    delta = x - stats["mean"]
    stats["mean"] = stats["mean"] + delta / stats["count"]
    delta2 = x - stats["mean"]
    stats["M2"] = stats["M2"] + delta * delta2
    return stats


def running_stats_normalize(stats, x, eps=1e-8):
    var = stats["M2"] / max(stats["count"], 1.0)
    return (np.asarray(x) - stats["mean"]) / (np.sqrt(var) + eps)


# ---------------------------------------------------------------------------
# One full network (encoder+RTU+head) -- used twice (actor, critic),
# each with FULLY INDEPENDENT parameters and streaming state.
# ---------------------------------------------------------------------------
def make_network(rng, in_dim, width, hidden_dim, head_out_dim, sparsity=0.9):
    return dict(
        enc=make_encoder_params(rng, in_dim, width, sparsity),
        rtu=make_rtu_params(rng, hidden_dim, width),
        head=make_head_params(rng, 2 * hidden_dim, head_out_dim),
    )


def network_param_count(net):
    total = 0
    for group in ("enc", "rtu", "head"):
        for v in net[group].values():
            total += int(np.prod(v.shape))
    return total


def network_streaming_init(hidden_dim, width, in_dim):
    return net_streaming_init(hidden_dim, width, in_dim)


def network_step(net, stream_state, obs_onehot):
    """Returns (z, output, S) where z is the RTU output (2*hidden_dim,)
    fed to the head, and S is the sensitivity dict for enc+rtu families."""
    x_t, output, S = net_streaming_step(net["rtu"], net["enc"], stream_state, obs_onehot)
    return output, S


def net_per_step_grad(S, dLdz, family_groups):
    """dL/dtheta for each family in family_groups (a dict fam->'diag'
    or 'shared' role), given dL/dz split into (2,hidden_dim) real/imag
    components already."""
    grads = {}
    for fam, role in family_groups.items():
        if role == "diag":
            grads[fam] = np.einsum("ih,ih...->h...", dLdz, S[fam])
        else:
            grads[fam] = np.einsum("ih,ih...->...", dLdz, S[fam])
    return grads


NET_FAMILY_ROLES = {**{f: "diag" for f in RTU_FAMILIES}, **{f: "shared" for f in ENCODER_FAMILIES}}


# ---------------------------------------------------------------------------
# ObGD -- exact published formula (re-verified against memorax's
# _obgd_update in the parity audit): step = alpha/max(1, alpha*kappa*
# max(|delta|,1)*||z||_1), applied to the WHOLE trace group at once.
# ---------------------------------------------------------------------------
def obgd_step_size(alpha, kappa, delta, trace_dict):
    delta_bar = max(abs(delta), 1.0)
    z_l1 = sum(float(np.sum(np.abs(np.asarray(z)))) for z in trace_dict.values())
    M = alpha * kappa * delta_bar * z_l1
    return alpha / M if M > 1.0 else alpha


def flat_param_shapes(net):
    shapes = {}
    for group in ("enc", "rtu", "head"):
        for k, v in net[group].items():
            shapes[(group, k)] = v.shape
    return shapes


def zero_traces(net):
    traces = {}
    for group in ("enc", "rtu", "head"):
        for k, v in net[group].items():
            traces[(group, k)] = np.zeros(v.shape)
    return traces


# ---------------------------------------------------------------------------
# TEST 2 (of the 10 requested): RTU init-distribution sanity.
# ---------------------------------------------------------------------------
def test_rtu_init_distribution(seed=0, hidden_dim=2000):
    rng = np.random.RandomState(seed)
    params = make_rtu_params(rng, hidden_dim, features=1)
    r = np.exp(-np.exp(np.asarray(params["nu_log"])))
    theta = np.exp(np.asarray(params["theta_log"]))
    ok = (
        np.all(np.isfinite(r)) and np.all(np.isfinite(theta))
        and np.all(r > 0) and np.all(r < 1)
        and np.all(theta > 0) and np.all(theta < 6.28 + 1e-9)
    )
    stats = dict(r_mean=float(r.mean()), r_min=float(r.min()), r_max=float(r.max()),
                 theta_mean=float(theta.mean()), theta_min=float(theta.min()), theta_max=float(theta.max()))
    return ok, stats


# ---------------------------------------------------------------------------
# TEST 3: input-normalization-factor (norm=sqrt(1-r^2)+eps) parity.
# ---------------------------------------------------------------------------
def test_norm_scaling_parity(eps=1e-8):
    checks = []
    for r_val in (0.0, 0.5, 0.9, 0.9999):
        nu_log = np.log(-np.log(r_val)) if r_val > 0 else 50.0  # r=exp(-exp(nu_log)); for r->0 use large nu_log
        theta_log = 0.0
        g, phi, norm, r = rtu_g_phi_norm(jnp.array([nu_log]), jnp.array([theta_log]), eps)
        norm_expected = np.sqrt(max(1e-12, 1 - r_val**2)) + eps
        checks.append((r_val, float(r[0]), float(norm[0]), norm_expected))
    max_err = max(abs(c[2] - c[3]) for c in checks)
    return max_err, checks


# ---------------------------------------------------------------------------
# TEST 4: separate actor/critic ObGD groups -- verify independence
# (no cross-leakage) and formula correctness.
# ---------------------------------------------------------------------------
def test_obgd_group_independence(seed=2):
    rng = np.random.RandomState(seed)
    actor_traces = {"a": rng.randn(5, 3), "b": rng.randn(4)}
    critic_traces = {"c": rng.randn(6)}
    step_actor = obgd_step_size(1.0, 3.0, delta=0.7, trace_dict=actor_traces)
    step_critic = obgd_step_size(1.0, 2.0, delta=0.7, trace_dict=critic_traces)
    # mutate critic_traces heavily -- must not affect an already-computed actor step
    critic_traces["c"] *= 1000.0
    step_actor_recomputed = obgd_step_size(1.0, 3.0, delta=0.7, trace_dict=actor_traces)
    independent = abs(step_actor - step_actor_recomputed) < 1e-15
    # verify formula directly
    delta_bar = max(abs(0.7), 1.0)
    z_l1_actor = sum(np.sum(np.abs(v)) for v in actor_traces.values())
    M = 1.0 * 3.0 * delta_bar * z_l1_actor
    expected = 1.0 / M if M > 1.0 else 1.0
    formula_ok = abs(step_actor - expected) < 1e-12
    return independent, formula_ok, step_actor, step_critic


# ---------------------------------------------------------------------------
# TEST 5: entropy-gradient test (categorical policy).
# ---------------------------------------------------------------------------
def categorical_entropy(logits):
    logp = logits - jax.scipy.special.logsumexp(logits)
    p = jnp.exp(logp)
    return -jnp.sum(p * logp)


def test_entropy_gradient(seed=3):
    rng = np.random.RandomState(seed)
    logits = jnp.array(rng.randn(5))
    analytic_grad = jax.grad(categorical_entropy)(logits)
    eps = 1e-6
    fd_grad = np.zeros(5)
    for i in range(5):
        lp, lm = np.asarray(logits).copy(), np.asarray(logits).copy()
        lp[i] += eps
        lm[i] -= eps
        fd_grad[i] = (float(categorical_entropy(jnp.array(lp))) - float(categorical_entropy(jnp.array(lm)))) / (2 * eps)
    err = float(np.max(np.abs(np.asarray(analytic_grad) - fd_grad)))
    return err


# ---------------------------------------------------------------------------
# TEST 6: running observation/reward statistics test (Welford).
# ---------------------------------------------------------------------------
def test_running_stats(seed=4, n=20000):
    rng = np.random.RandomState(seed)
    true_mean, true_std = 3.0, 2.0
    samples = rng.randn(n) * true_std + true_mean
    stats = running_stats_init(())
    for x in samples:
        running_stats_update(stats, x)
    est_mean = stats["mean"]
    est_var = stats["M2"] / stats["count"]
    return abs(est_mean - true_mean), abs(np.sqrt(est_var) - true_std)


# ---------------------------------------------------------------------------
# TEST 7: encoder->RTU->head shape test (end to end, real POPGym obs).
# ---------------------------------------------------------------------------
def test_shapes(seed=5, hidden_dim=8, width=16, num_actions=4):
    rng = np.random.RandomState(seed)
    in_dim = 6  # Autoencode one-hot dim
    net = make_network(rng, in_dim, width, hidden_dim, num_actions)
    stream_state = network_streaming_init(hidden_dim, width, in_dim)
    obs_onehot = jnp.array(one_hot_obs((0, 2), "autoencode"))
    z, S = network_step(net, stream_state, obs_onehot)
    assert z.shape == (2 * hidden_dim,), z.shape
    for fam in RTU_FAMILIES:
        assert S[fam].shape[0] == 2 and S[fam].shape[1] == hidden_dim, (fam, S[fam].shape)
    assert S["W_enc"].shape == (2, hidden_dim, in_dim, width)
    logits = head_forward(z, net["head"])
    assert logits.shape == (num_actions,), logits.shape
    return True


# ---------------------------------------------------------------------------
# TEST 8: parameter-count report (actor vs critic).
# ---------------------------------------------------------------------------
def test_param_counts(seed=6, hidden_dim=192, width=64, num_actions=4, in_dim=6):
    rng = np.random.RandomState(seed)
    actor = make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic = make_network(rng, in_dim, width, hidden_dim, 1)
    return network_param_count(actor), network_param_count(critic)


# ---------------------------------------------------------------------------
# TEST 9: no accidental actor/critic parameter sharing.
# ---------------------------------------------------------------------------
def test_no_param_sharing(seed=7, hidden_dim=8, width=16, num_actions=4, in_dim=6):
    rng = np.random.RandomState(seed)
    actor = make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic = make_network(rng, in_dim, width, hidden_dim, 1)
    same_object = any(
        actor[g][k] is critic[g][k]
        for g in ("enc", "rtu") for k in actor[g]
    )
    # different seeds used in make_network calls (numpy RandomState is
    # STATEFUL and shared across calls above -- so also check VALUES differ)
    values_identical = all(
        np.allclose(np.asarray(actor["rtu"][k]), np.asarray(critic["rtu"][k]))
        for k in actor["rtu"] if actor["rtu"][k].shape == critic["rtu"][k].shape
    )
    return (not same_object), (not values_identical)


# ---------------------------------------------------------------------------
# TEST 10: short numerical smoke test -- real POPGym Autoencode steps
# through the FULL faithful pipeline (both networks, TD-error, entropy,
# ObGD update, running normalization), no NaN/inf.
# ---------------------------------------------------------------------------
def smoke_test(seed=8, n_steps=20, hidden_dim=8, width=16):
    rng = np.random.RandomState(seed)
    env = popgym.envs.autoencode.AutoencodeEasy()
    in_dim = 6
    num_actions = env.action_space.n

    actor = make_network(rng, in_dim, width, hidden_dim, num_actions)
    critic = make_network(rng, in_dim, width, hidden_dim, 1)
    actor_stream = network_streaming_init(hidden_dim, width, in_dim)
    critic_stream = network_streaming_init(hidden_dim, width, in_dim)

    obs_stats = running_stats_init((in_dim,))
    reward_stats = running_stats_init(())

    gamma, lam = 0.99, 0.8
    actor_alpha, critic_alpha = 1.0, 1.0
    actor_kappa, critic_kappa = 3.0, 2.0
    entropy_coef = 0.095

    actor_traces = zero_traces(actor)
    critic_traces = zero_traces(critic)

    obs, _ = env.reset(seed=seed)
    u0 = np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64)
    running_stats_update(obs_stats, u0)
    u0n = jnp.array(running_stats_normalize(obs_stats, u0))
    z_actor, S_actor = network_step(actor, actor_stream, u0n)
    z_critic, S_critic = network_step(critic, critic_stream, u0n)

    all_finite = True
    returns = []
    for t in range(n_steps):
        logits = head_forward(z_actor, actor["head"])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(rng.choice(len(probs), p=probs))

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        running_stats_update(reward_stats, r_t)
        r_t_norm = float(running_stats_normalize(reward_stats, r_t))

        v_cur = float(head_forward(z_critic, critic["head"])[0])
        if done:
            v_next = 0.0
            S_next_actor = S_next_critic = z_next_actor = z_next_critic = None
        else:
            u_next = np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)
            running_stats_update(obs_stats, u_next)
            u_next_n = jnp.array(running_stats_normalize(obs_stats, u_next))
            z_next_actor, S_next_actor = network_step(actor, actor_stream, u_next_n)
            z_next_critic, S_next_critic = network_step(critic, critic_stream, u_next_n)
            v_next = float(head_forward(z_next_critic, critic["head"])[0])

        td_error = r_t_norm + gamma * v_next - v_cur
        all_finite &= np.isfinite(td_error)

        logp_fn = lambda z: (head_forward(z, actor["head"]) - jax.scipy.special.logsumexp(head_forward(z, actor["head"])))[a_t]
        entropy_fn = lambda z: categorical_entropy(head_forward(z, actor["head"]))
        dlogp_dz = np.asarray(jax.grad(logp_fn)(z_actor))
        dentropy_dz = np.asarray(jax.grad(entropy_fn)(z_actor))
        dactor_target_dz = dlogp_dz + entropy_coef * np.sign(td_error) * dentropy_dz
        dv_dz = np.asarray(jax.grad(lambda z: head_forward(z, critic["head"])[0])(z_critic))

        dlogp_dz_split = np.stack([dactor_target_dz[:hidden_dim], dactor_target_dz[hidden_dim:]], axis=0)
        dv_dz_split = np.stack([dv_dz[:hidden_dim], dv_dz[hidden_dim:]], axis=0)

        g_actor = net_per_step_grad(S_actor, dlogp_dz_split, NET_FAMILY_ROLES)
        g_critic = net_per_step_grad(S_critic, dv_dz_split, NET_FAMILY_ROLES)

        for k, v in g_actor.items():
            actor_traces[("rtu" if k in RTU_FAMILIES else "enc", k)] = (
                gamma * lam * actor_traces[("rtu" if k in RTU_FAMILIES else "enc", k)] + v
            )
        for k, v in g_critic.items():
            critic_traces[("rtu" if k in RTU_FAMILIES else "enc", k)] = (
                gamma * lam * critic_traces[("rtu" if k in RTU_FAMILIES else "enc", k)] + v
            )
        gh_pi_W = np.asarray(jax.grad(lambda W: (W @ z_actor + actor["head"]["b"] - jax.scipy.special.logsumexp(W @ z_actor + actor["head"]["b"]))[a_t]
                                       + entropy_coef * np.sign(td_error) * categorical_entropy(W @ z_actor + actor["head"]["b"]))(actor["head"]["W"]))
        gh_pi_b = np.asarray(jax.grad(lambda b: (actor["head"]["W"] @ z_actor + b - jax.scipy.special.logsumexp(actor["head"]["W"] @ z_actor + b))[a_t]
                                       + entropy_coef * np.sign(td_error) * categorical_entropy(actor["head"]["W"] @ z_actor + b))(actor["head"]["b"]))
        gh_v_W = np.asarray(jax.grad(lambda W: (W @ z_critic + critic["head"]["b"])[0])(critic["head"]["W"]))
        gh_v_b = np.asarray(jax.grad(lambda b: (critic["head"]["W"] @ z_critic + b)[0])(critic["head"]["b"]))
        actor_traces[("head", "W")] = gamma * lam * actor_traces[("head", "W")] + gh_pi_W
        actor_traces[("head", "b")] = gamma * lam * actor_traces[("head", "b")] + gh_pi_b
        critic_traces[("head", "W")] = gamma * lam * critic_traces[("head", "W")] + gh_v_W
        critic_traces[("head", "b")] = gamma * lam * critic_traces[("head", "b")] + gh_v_b

        step_actor = obgd_step_size(actor_alpha, actor_kappa, td_error, actor_traces)
        step_critic = obgd_step_size(critic_alpha, critic_kappa, td_error, critic_traces)

        for (group, k), z_trace in actor_traces.items():
            actor[group][k] = actor[group][k] + step_actor * td_error * z_trace
        for (group, k), z_trace in critic_traces.items():
            critic[group][k] = critic[group][k] + step_critic * td_error * z_trace

        for group in ("enc", "rtu", "head"):
            for k, v in actor[group].items():
                all_finite &= bool(np.all(np.isfinite(np.asarray(v))))
            for k, v in critic[group].items():
                all_finite &= bool(np.all(np.isfinite(np.asarray(v))))

        returns.append(r_t)
        if done:
            break
        z_actor, S_actor = z_next_actor, S_next_actor
        z_critic, S_critic = z_next_critic, S_next_critic

    return all_finite, float(np.sum(returns)), t + 1


def main():
    print("=" * 70)
    print("TEST 2: RTU init-distribution sanity")
    print("=" * 70)
    ok, stats = test_rtu_init_distribution()
    print(f"  ok={ok}  stats={stats}")
    assert ok

    print("=" * 70)
    print("TEST 3: input-normalization-factor (norm) parity")
    print("=" * 70)
    max_err, checks = test_norm_scaling_parity()
    for r_val, r_got, norm_got, norm_exp in checks:
        print(f"  r_target={r_val:.4f}  r_got={r_got:.4f}  norm_got={norm_got:.6f}  norm_expected={norm_exp:.6f}")
    print(f"  max_err={max_err:.2e}")
    assert max_err < 1e-6

    print("=" * 70)
    print("TEST 4: separate actor/critic ObGD group independence")
    print("=" * 70)
    independent, formula_ok, step_actor, step_critic = test_obgd_group_independence()
    print(f"  independent={independent}  formula_ok={formula_ok}  step_actor={step_actor:.4f}  step_critic={step_critic:.4f}")
    assert independent and formula_ok

    print("=" * 70)
    print("TEST 5: entropy-gradient test (analytic vs finite-difference)")
    print("=" * 70)
    err = test_entropy_gradient()
    print(f"  |err|={err:.2e}")
    assert err < 1e-5

    print("=" * 70)
    print("TEST 6: running observation/reward statistics")
    print("=" * 70)
    mean_err, std_err = test_running_stats()
    print(f"  mean_err={mean_err:.4f}  std_err={std_err:.4f}")
    assert mean_err < 0.1 and std_err < 0.1

    print("=" * 70)
    print("TEST 7: encoder->RTU->head shape test")
    print("=" * 70)
    ok7 = test_shapes()
    print(f"  ok={ok7}")

    print("=" * 70)
    print("TEST 8: parameter-count report (published-scale: hidden=192, width=64)")
    print("=" * 70)
    n_actor, n_critic = test_param_counts()
    print(f"  actor params={n_actor}  critic params={n_critic}  total={n_actor + n_critic}")

    print("=" * 70)
    print("TEST 9: no accidental actor/critic parameter sharing")
    print("=" * 70)
    no_shared_obj, no_shared_val = test_no_param_sharing()
    print(f"  no_shared_object={no_shared_obj}  no_shared_values={no_shared_val}")
    assert no_shared_obj and no_shared_val

    print("=" * 70)
    print("TEST 10: short numerical smoke test (real POPGym Autoencode, small scale)")
    print("=" * 70)
    t0 = time.time()
    all_finite, ep_return, n_steps_run = smoke_test()
    print(f"  all_finite={all_finite}  episode_return={ep_return:.4f}  steps={n_steps_run}  "
          f"elapsed={time.time() - t0:.2f}s")
    assert all_finite

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
