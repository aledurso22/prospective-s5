"""Phase B28 -- faithful-as-practical RTU Autoencode calibration run.

Uses the FROZEN JIT implementation from commit cfa1009
(b28_rtu_faithful_jit.py) UNCHANGED -- this file only adds the outer
training loop, episode-boundary handling, lightweight in-jit
accumulator tracking (to avoid per-step host syncs), checkpointing, and
logging. No change to architecture, hyperparameters, RTU/RTRL
equations, normalization semantics, entropy term, ObGD, or num_envs (=1).

RTU only. No "ours". Not a Stage-2 comparison.

Run: python -m credit_memory.b28_rtu_calibration_run
"""
from __future__ import annotations

import json
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
CHECKPOINTS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]


def leaves_sq_diff(old_pytree, new_pytree):
    diffs = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), old_pytree, new_pytree)
    return sum(jax.tree_util.tree_leaves(diffs))


def all_finite(pytree):
    flags = jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), pytree)
    return jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: x, flags))


def stepped_update(carry, stats, action, raw_reward, raw_obs_next, done, hp):
    """Wraps the FROZEN full_update_step, adding lightweight
    accumulator bookkeeping (all pure jnp, jittable together) so the
    outer Python loop can avoid per-step host syncs."""
    old_actor_params = carry["actor"]["params"]
    old_critic_params = carry["critic"]["params"]

    new_carry, diag = full_update_step(
        carry, action, raw_reward, raw_obs_next, done,
        hp["gamma"], hp["lam"], hp["entropy_coef"], hp["actor_alpha"], hp["critic_alpha"],
        hp["actor_kappa"], hp["critic_kappa"],
    )

    upd_actor_sq = leaves_sq_diff(old_actor_params, new_carry["actor"]["params"])
    upd_critic_sq = leaves_sq_diff(old_critic_params, new_carry["critic"]["params"])

    new_stats = dict(
        n=stats["n"] + 1.0,
        sum_entropy=stats["sum_entropy"] + diag["entropy"],
        sum_value=stats["sum_value"] + diag["value_cur"],
        sum_value_sq=stats["sum_value_sq"] + diag["value_cur"] ** 2,
        sum_td=stats["sum_td"] + diag["td_error"],
        sum_td_sq=stats["sum_td_sq"] + diag["td_error"] ** 2,
        max_abs_td=jnp.maximum(stats["max_abs_td"], jnp.abs(diag["td_error"])),
        sum_step_actor=stats["sum_step_actor"] + diag["step_actor"],
        sum_step_critic=stats["sum_step_critic"] + diag["step_critic"],
        sum_upd_actor_sq=stats["sum_upd_actor_sq"] + upd_actor_sq,
        sum_upd_critic_sq=stats["sum_upd_critic_sq"] + upd_critic_sq,
    )
    return new_carry, new_stats


def zero_stats():
    return dict(n=jnp.asarray(0.0), sum_entropy=jnp.asarray(0.0), sum_value=jnp.asarray(0.0),
                sum_value_sq=jnp.asarray(0.0), sum_td=jnp.asarray(0.0), sum_td_sq=jnp.asarray(0.0),
                max_abs_td=jnp.asarray(0.0), sum_step_actor=jnp.asarray(0.0), sum_step_critic=jnp.asarray(0.0),
                sum_upd_actor_sq=jnp.asarray(0.0), sum_upd_critic_sq=jnp.asarray(0.0))


def rtu_r_stats(rtu_params):
    r = np.exp(-np.exp(np.asarray(rtu_params["nu_log"])))
    return dict(mean=float(r.mean()), min=float(r.min()), max=float(r.max()))


def run(seed=0, total_frames=5_000_000, log_every=2000, out_path=None):
    stepped_jit = jax.jit(lambda c, s, a, r, o, d: stepped_update(c, s, a, r, o, d, HP))

    rng = np.random.RandomState(seed)
    actor_net = reft.make_network(rng, IN_DIM, WIDTH, HIDDEN_DIM, NUM_ACTIONS)
    critic_net = reft.make_network(rng, IN_DIM, WIDTH, HIDDEN_DIM, 1)
    actor_stream = reft.network_streaming_init(HIDDEN_DIM, WIDTH, IN_DIM)
    critic_stream = reft.network_streaming_init(HIDDEN_DIM, WIDTH, IN_DIM)
    obs_stats = reft.running_stats_init((IN_DIM,))
    reward_stats = reft.reward_scale_init()
    carry = make_carry(actor_net, actor_stream, reft.zero_traces(actor_net),
                        critic_net, critic_stream, reft.zero_traces(critic_net),
                        obs_stats, reward_stats)
    stats = zero_stats()

    env = popgym.envs.autoencode.AutoencodeEasy()
    action_rng = np.random.RandomState(seed + 10_000)

    episode_returns = []
    all_episode_returns = []  # full cumulative learning curve
    ep_return = 0.0
    frame = 0
    next_checkpoint_idx = 0
    t_start = time.time()
    t_last_log = t_start
    frames_last_log = 0
    history = []  # checkpoint dicts, kept for later plotting

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

    carry = reset_episode(env, carry)

    print(f"=== B28 faithful-as-practical RTU Autoencode calibration -- seed={seed} ===")
    print(f"hp={HP} hidden={HIDDEN_DIM} width={WIDTH} num_envs=1")

    while frame < total_frames:
        logits, value = None, None
        z_actor = jnp.concatenate([carry["actor"]["real"], carry["actor"]["imag"]])
        logits = reft.head_forward(z_actor, carry["actor"]["params"]["head"])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(action_rng.choice(len(probs), p=probs))

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        ep_return += r_t

        if done:
            obs_next_arr = np.zeros(IN_DIM)
        else:
            obs_next_arr = np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)

        carry, stats = stepped_jit(carry, stats, jnp.asarray(a_t), jnp.asarray(float(r_t)),
                                    jnp.asarray(obs_next_arr), jnp.asarray(1.0 if done else 0.0))
        frame += 1

        if done:
            episode_returns.append(ep_return)
            all_episode_returns.append((frame, ep_return))
            ep_return = 0.0
            carry = reset_episode(env, carry)

        if frame % log_every == 0:
            jax.block_until_ready(carry["actor"]["params"]["rtu"]["nu_log"])
            fin = bool(all_finite(carry))
            if not fin:
                print(f"!!! NON-FINITE DETECTED at frame {frame} -- stopping.")
                break
            n = float(stats["n"])
            if n > 0:
                s = {k: float(v) for k, v in stats.items()}
                dt = time.time() - t_last_log
                fps = (frame - frames_last_log) / max(dt, 1e-9)
                print(f"  frame={frame:>9,d}  fps={fps:6.1f}  "
                      f"entropy={s['sum_entropy']/n:.4f}  "
                      f"value={s['sum_value']/n:+.4f}  "
                      f"td_mean={s['sum_td']/n:+.4f}  td_max_abs={s['max_abs_td']:.3f}  "
                      f"step_actor={s['sum_step_actor']/n:.4f}  step_critic={s['sum_step_critic']/n:.4f}  "
                      f"upd_actor_rms={np.sqrt(s['sum_upd_actor_sq']/n):.2e}  "
                      f"upd_critic_rms={np.sqrt(s['sum_upd_critic_sq']/n):.2e}  "
                      f"n_episodes={len(episode_returns)}")
            stats = zero_stats()
            t_last_log = time.time()
            frames_last_log = frame

        if next_checkpoint_idx < len(CHECKPOINTS) and frame >= CHECKPOINTS[next_checkpoint_idx]:
            jax.block_until_ready(carry)
            fin = bool(all_finite(carry))
            recent = episode_returns[-50:] if episode_returns else []
            r_actor = rtu_r_stats(carry["actor"]["params"]["rtu"])
            r_critic = rtu_r_stats(carry["critic"]["params"]["rtu"])
            elapsed = time.time() - t_start
            ckpt = dict(
                frame=frame, elapsed_s=elapsed, fps_avg=frame / elapsed,
                n_episodes=len(episode_returns),
                recent_return_mean=float(np.mean(recent)) if recent else None,
                recent_return_median=float(np.median(recent)) if recent else None,
                recent_return_min=float(np.min(recent)) if recent else None,
                recent_return_max=float(np.max(recent)) if recent else None,
                all_finite=fin,
                obs_stats_mean=np.asarray(carry["obs_stats"]["mean"]).tolist(),
                obs_stats_var=np.asarray(carry["obs_stats"]["var"]).tolist(),
                reward_scale_trace=float(carry["reward_stats"]["trace"]),
                reward_scale_var=float(carry["reward_stats"]["stats"]["var"]),
                rtu_r_actor=r_actor, rtu_r_critic=r_critic,
            )
            history.append(ckpt)
            print("=" * 70)
            print(f"CHECKPOINT frame={frame:,}  elapsed={elapsed/60:.1f}min  avg_fps={frame/elapsed:.1f}")
            print(f"  episodes so far: {len(episode_returns)}  "
                  f"recent(last 50) return mean={ckpt['recent_return_mean']}  "
                  f"median={ckpt['recent_return_median']}  "
                  f"min={ckpt['recent_return_min']}  max={ckpt['recent_return_max']}")
            print(f"  all_finite={fin}")
            print(f"  obs_stats mean={ckpt['obs_stats_mean']}")
            print(f"  obs_stats var={ckpt['obs_stats_var']}")
            print(f"  reward_stats mean={ckpt['reward_stats_mean']:.4f} var={ckpt['reward_stats_var']:.4f}")
            print(f"  RTU r (actor): {r_actor}   RTU r (critic): {r_critic}")
            print("=" * 70)
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(dict(history=history, episode_returns=all_episode_returns), f, indent=2)
            next_checkpoint_idx += 1
            if not fin:
                print("STOPPING due to non-finite state.")
                break

    print("DONE" if frame >= total_frames else "STOPPED EARLY")
    return history, all_episode_returns


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    run(seed=0, total_frames=5_000_000, out_path=out)
