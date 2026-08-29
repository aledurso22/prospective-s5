"""Phase B28 -- frozen-snapshot critic-calibration diagnostic.

Does NOT touch or stop the live 5M-frame seed-0 run (bi8gc78dh). Uses
the FROZEN JIT implementation (commit cfa1009) unchanged.

The live process cannot be introspected for its exact in-memory
parameters without stopping it (no IPC/debug hook exists), and its own
environment resets are unseeded (env.reset() with no seed argument),
so a separate process cannot reproduce its EXACT trajectory bit-for-
bit even with the same seed. Instead: train an independent "twin" run
from the same seed/hyperparameters/code to a frame count where the log
evidence shows the SAME plateau regime has clearly formed (value
~-15 to -25, entropy ~0.6-0.8 -- confirmed against the live run's own
log before trusting this snapshot), then freeze it completely and run
the requested calibration experiment. This is stated as a
representative-regime proxy, not a bit-exact snapshot of the live run.

Phase 1: train fresh to a representative checkpoint, verify its
regime matches the live run's plateau, save the full carry (actor+
critic params, obs_stats, reward_stats -- normalization frozen from
this point on).

Phase 2: roll out many fresh episodes using the FROZEN actor's own
stochastic policy (not uniform random), critic held fixed, reward/obs
normalizers held fixed (no further updates). Record phase, V(s_t),
exact normalized reward, and compute discounted MC return-to-go G_t
(gamma=0.99, correct terminal semantics) at every timestep. Report
V-G bias, correlation/regression, phase-conditional breakdowns,
percentile comparisons, and episode-level (not per-timestep) bootstrap
CI, plus the frozen policy's own return distribution/PLAY accuracy/
entropy/action distribution.

Run: python -m credit_memory.b28_rtu_snapshot_calibration
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

import popgym

from credit_memory.b28_popgym_stage1 import one_hot_obs
from credit_memory import b28_rtu_faithful_train as reft
from credit_memory.b28_rtu_faithful_jit import (
    make_carry, full_update_step, net_streaming_step_jit,
    _running_update, _running_normalize,
)

jax.config.update("jax_enable_x64", True)

HP = dict(gamma=0.99, lam=0.8, entropy_coef=0.095, actor_alpha=1.0, critic_alpha=1.0,
          actor_kappa=3.0, critic_kappa=2.0)
HIDDEN_DIM, WIDTH, IN_DIM, NUM_ACTIONS = 192, 64, 6, 4
GAMMA = HP["gamma"]


def stepped_update_simple(carry, action, raw_reward, raw_obs_next, done):
    return full_update_step(carry, action, raw_reward, raw_obs_next, done,
                             HP["gamma"], HP["lam"], HP["entropy_coef"], HP["actor_alpha"],
                             HP["critic_alpha"], HP["actor_kappa"], HP["critic_kappa"])


def reset_episode(env, carry):
    obs, _ = env.reset()
    u0 = np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64)
    obs_stats_new = _running_update(carry["obs_stats"], jnp.asarray(u0))
    u0n = _running_normalize(obs_stats_new, jnp.asarray(u0))
    a_real, a_imag, _, a_S = net_streaming_step_jit(
        carry["actor"]["params"]["rtu"], carry["actor"]["params"]["enc"],
        jnp.zeros(HIDDEN_DIM), jnp.zeros(HIDDEN_DIM),
        {fam: jnp.zeros_like(v) for fam, v in carry["actor"]["S"].items()}, u0n)
    c_real, c_imag, _, c_S = net_streaming_step_jit(
        carry["critic"]["params"]["rtu"], carry["critic"]["params"]["enc"],
        jnp.zeros(HIDDEN_DIM), jnp.zeros(HIDDEN_DIM),
        {fam: jnp.zeros_like(v) for fam, v in carry["critic"]["S"].items()}, u0n)
    new_carry = dict(carry)
    new_carry["actor"] = dict(carry["actor"], real=a_real, imag=a_imag, S=a_S)
    new_carry["critic"] = dict(carry["critic"], real=c_real, imag=c_imag, S=c_S)
    new_carry["obs_stats"] = obs_stats_new
    return new_carry


def train_twin_snapshot(seed=0, target_frames=300_000, log_every=20_000):
    """Phase 1: train a fresh twin run, verify it reaches the same
    plateau regime as the live run's log before trusting the
    snapshot."""
    stepped_jit = jax.jit(stepped_update_simple)
    rng = np.random.RandomState(seed)
    actor_net = reft.make_network(rng, IN_DIM, WIDTH, HIDDEN_DIM, NUM_ACTIONS)
    critic_net = reft.make_network(rng, IN_DIM, WIDTH, HIDDEN_DIM, 1)
    actor_stream = reft.network_streaming_init(HIDDEN_DIM, WIDTH, IN_DIM)
    critic_stream = reft.network_streaming_init(HIDDEN_DIM, WIDTH, IN_DIM)
    obs_stats = reft.running_stats_init((IN_DIM,))
    reward_stats = reft.running_stats_init(())
    carry = make_carry(actor_net, actor_stream, reft.zero_traces(actor_net),
                        critic_net, critic_stream, reft.zero_traces(critic_net),
                        obs_stats, reward_stats)

    env = popgym.envs.autoencode.AutoencodeEasy()
    action_rng = np.random.RandomState(seed + 10_000)
    carry = reset_episode(env, carry)

    frame = 0
    t0 = time.time()
    recent_values, recent_entropies = [], []
    while frame < target_frames:
        z_actor = jnp.concatenate([carry["actor"]["real"], carry["actor"]["imag"]])
        logits = reft.head_forward(z_actor, carry["actor"]["params"]["head"])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(action_rng.choice(len(probs), p=probs))

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        obs_next_arr = np.zeros(IN_DIM) if done else np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)

        carry, diag = stepped_jit(carry, jnp.asarray(a_t), jnp.asarray(float(r_t)),
                                   jnp.asarray(obs_next_arr), jnp.asarray(1.0 if done else 0.0))
        frame += 1
        recent_values.append(float(diag["value_cur"]))
        recent_entropies.append(float(diag["entropy"]))

        if done:
            carry = reset_episode(env, carry)

        if frame % log_every == 0:
            jax.block_until_ready(carry)
            print(f"  [twin] frame={frame:>8,d}  value_mean(last {log_every})={np.mean(recent_values):+.3f}  "
                  f"entropy_mean={np.mean(recent_entropies):.3f}  elapsed={time.time()-t0:.1f}s", flush=True)
            recent_values, recent_entropies = [], []

    print(f"Twin training done: {frame} frames in {time.time()-t0:.1f}s", flush=True)
    return carry


def rollout_frozen_episode(carry, env, action_rng):
    """One episode under the FROZEN actor policy, frozen critic, frozen
    normalizers (no updates at all -- pure forward passes). Returns a
    dict of per-timestep records."""
    stream_actor = dict(real=jnp.zeros(HIDDEN_DIM), imag=jnp.zeros(HIDDEN_DIM),
                         S={fam: jnp.zeros_like(v) for fam, v in carry["actor"]["S"].items()})
    stream_critic = dict(real=jnp.zeros(HIDDEN_DIM), imag=jnp.zeros(HIDDEN_DIM),
                          S={fam: jnp.zeros_like(v) for fam, v in carry["critic"]["S"].items()})

    obs, _ = env.reset()
    u0 = np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64)
    u0n = _running_normalize(carry["obs_stats"], jnp.asarray(u0))  # FROZEN normalizer, no update
    a_real, a_imag, _, a_S = net_streaming_step_jit(
        carry["actor"]["params"]["rtu"], carry["actor"]["params"]["enc"],
        stream_actor["real"], stream_actor["imag"], stream_actor["S"], u0n)
    c_real, c_imag, _, c_S = net_streaming_step_jit(
        carry["critic"]["params"]["rtu"], carry["critic"]["params"]["enc"],
        stream_critic["real"], stream_critic["imag"], stream_critic["S"], u0n)

    phases, values, rewards_norm, actions = [], [], [], []
    correct_play = 0
    n_play = 0
    while True:
        z_actor = jnp.concatenate([a_real, a_imag])
        z_critic = jnp.concatenate([c_real, c_imag])
        logits = reft.head_forward(z_actor, carry["actor"]["params"]["head"])
        value = float(reft.head_forward(z_critic, carry["critic"]["params"]["head"])[0])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(action_rng.choice(len(probs), p=probs))

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        r_norm = float(_running_normalize(carry["reward_stats"], jnp.asarray(float(r_t))))  # FROZEN

        is_play = (obs_next[0] == 0) if not done else None  # PLAY=0, WATCH=1 (popgym Mode enum)
        phases.append("play" if is_play else ("watch" if is_play is not None else "terminal"))
        values.append(value)
        rewards_norm.append(r_norm)
        actions.append(a_t)
        if is_play:
            n_play += 1
            if r_t > 0:
                correct_play += 1

        if done:
            break
        u_next = np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)
        u_next_n = _running_normalize(carry["obs_stats"], jnp.asarray(u_next))
        a_real, a_imag, _, a_S = net_streaming_step_jit(
            carry["actor"]["params"]["rtu"], carry["actor"]["params"]["enc"], a_real, a_imag, a_S, u_next_n)
        c_real, c_imag, _, c_S = net_streaming_step_jit(
            carry["critic"]["params"]["rtu"], carry["critic"]["params"]["enc"], c_real, c_imag, c_S, u_next_n)

    T = len(rewards_norm)
    G = np.zeros(T)
    running = 0.0
    for i in range(T - 1, -1, -1):
        running = rewards_norm[i] + GAMMA * running
        G[i] = running

    return dict(phases=phases, values=np.array(values), G=G, actions=np.array(actions),
                total_return=float(np.sum([r for r, ph in zip(rewards_norm, phases)])),  # normalized-units total
                raw_return=None, n_play=n_play, correct_play=correct_play, T=T)


def run_calibration_experiment(carry, n_episodes=300, seed=999):
    env = popgym.envs.autoencode.AutoencodeEasy()
    action_rng = np.random.RandomState(seed)
    episodes = []
    t0 = time.time()
    for ep in range(n_episodes):
        rec = rollout_frozen_episode(carry, env, action_rng)
        episodes.append(rec)
        if (ep + 1) % 50 == 0:
            print(f"  [calib] episode {ep+1}/{n_episodes}  elapsed={time.time()-t0:.1f}s", flush=True)
    return episodes


def analyze(episodes):
    all_V, all_G, all_phase = [], [], []
    per_ep_bias = []
    play_action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_play, total_correct = 0, 0
    ep_returns = []
    entropies_proxy = []

    for rec in episodes:
        V, G, phases = rec["values"], rec["G"], rec["phases"]
        all_V.extend(V.tolist())
        all_G.extend(G.tolist())
        all_phase.extend(phases)
        per_ep_bias.append(float(np.mean(V - G)))
        ep_returns.append(rec["total_return"])
        total_play += rec["n_play"]
        total_correct += rec["correct_play"]
        for a, ph in zip(rec["actions"], phases):
            if ph == "play":
                play_action_counts[a] += 1

    all_V, all_G = np.array(all_V), np.array(all_G)
    all_phase = np.array(all_phase)
    diff = all_V - all_G

    def pctl(x, ps=(1, 5, 25, 50, 75, 95, 99)):
        return {p: float(np.percentile(x, p)) for p in ps}

    report = {}
    report["n_episodes"] = len(episodes)
    report["n_timesteps"] = len(all_V)
    report["bias_mean"] = float(diff.mean())
    report["bias_median"] = float(np.median(diff))
    report["bias_MAE"] = float(np.mean(np.abs(diff)))
    report["bias_RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
    report["corr_V_G"] = float(np.corrcoef(all_V, all_G)[0, 1])
    slope, intercept = np.polyfit(all_V, all_G, 1)
    report["regression_G_on_V"] = dict(slope=float(slope), intercept=float(intercept))
    report["V_percentiles"] = pctl(all_V)
    report["G_percentiles"] = pctl(all_G)
    report["diff_percentiles"] = pctl(diff)

    report["phase_breakdown"] = {}
    for ph in ("watch", "play"):
        mask = all_phase == ph
        if mask.sum() > 0:
            report["phase_breakdown"][ph] = dict(
                n=int(mask.sum()), V_mean=float(all_V[mask].mean()), G_mean=float(all_G[mask].mean()),
                bias_mean=float(diff[mask].mean()), bias_RMSE=float(np.sqrt(np.mean(diff[mask] ** 2))),
            )
    # PLAY early/mid/late position buckets
    play_positions = []
    for rec in episodes:
        play_idx = [i for i, ph in enumerate(rec["phases"]) if ph == "play"]
        n = len(play_idx)
        for rank, i in enumerate(play_idx):
            bucket = "early" if rank < n / 3 else ("late" if rank >= 2 * n / 3 else "mid")
            play_positions.append((bucket, rec["values"][i], rec["G"][i]))
    for bucket in ("early", "mid", "late"):
        vs = np.array([v for b, v, g in play_positions if b == bucket])
        gs = np.array([g for b, v, g in play_positions if b == bucket])
        if len(vs):
            report.setdefault("play_position_breakdown", {})[bucket] = dict(
                n=len(vs), V_mean=float(vs.mean()), G_mean=float(gs.mean()), bias_mean=float((vs - gs).mean()))

    # Episode-level bootstrap CI on per-episode mean bias
    per_ep_bias = np.array(per_ep_bias)
    boot_means = []
    rng = np.random.RandomState(0)
    for _ in range(2000):
        sample = rng.choice(per_ep_bias, size=len(per_ep_bias), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    report["episode_level_bias"] = dict(
        mean=float(per_ep_bias.mean()), std=float(per_ep_bias.std()),
        bootstrap_ci_2_5=float(np.percentile(boot_means, 2.5)),
        bootstrap_ci_97_5=float(np.percentile(boot_means, 97.5)),
    )

    report["policy_performance"] = dict(
        return_mean=float(np.mean(ep_returns)), return_std=float(np.std(ep_returns)),
        play_accuracy=(total_correct / total_play) if total_play else None,
        play_action_distribution=(play_action_counts / max(play_action_counts.sum(), 1)).tolist(),
    )
    return report


def main():
    print("=" * 70, flush=True)
    print("Phase 1: training twin snapshot to representative plateau regime", flush=True)
    print("=" * 70, flush=True)
    carry = train_twin_snapshot(seed=0, target_frames=300_000, log_every=20_000)

    print("=" * 70, flush=True)
    print("Phase 2: frozen-snapshot calibration rollout (300 episodes, frozen actor/critic/normalizers)", flush=True)
    print("=" * 70, flush=True)
    episodes = run_calibration_experiment(carry, n_episodes=300)

    report = analyze(episodes)
    print("=" * 70, flush=True)
    print("CALIBRATION REPORT", flush=True)
    print("=" * 70, flush=True)
    import json
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
