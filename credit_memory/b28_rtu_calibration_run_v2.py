"""Phase B28 -- faithful-as-practical RTU Autoencode calibration run,
INSTRUMENTATION v2. This is now the driver for the CORRECTED-semantics
seed-0 calibration run.

The original seed-0 run (via credit_memory/b28_rtu_calibration_run.py,
v1 instrumentation) was INVALIDATED and stopped: two algorithmic
mismatches were found against Farr Appendix-B Algorithm 2 / Elsayed
stream-AC (see b28_rtu_faithful_jit.py's module docstring for the
full explanation) -- eligibility traces were not reset at episode
boundaries, and reward handling used the wrong transformation (mean-
centered normalization instead of Elsayed/Farr reward SCALING via a
discounted trace's variance). Both are now fixed in
b28_rtu_faithful_jit.py's full_update_step, verified against
independent literal references and the full eager-vs-JIT parity
suite. This file relaunches seed 0 FROM INITIALIZATION under the
corrected semantics, with the v2 richer diagnostics from the start.

Same architecture/hyperparameters otherwise: separate actor/critic
networks, RTU/RTRL equations, entropy term, ObGD, num_envs=1, frame
accounting. This file's own contribution is only the OUTER checkpoint
diagnostics and explicit log flushing.

New relative to v1:
  - policy entropy (already had window mean; now also a light sample
    for spread, though entropy is bounded/smooth so mean is usually
    sufficient -- kept simple: mean + min/max over the window);
  - action-distribution counts (per checkpoint window);
  - WATCH-vs-PLAY split: action/accuracy summary, using the
    OBSERVABLE mode flag (obs[0] from the environment's own Tuple
    observation) -- read externally, no env-internals access, no
    change to the learner;
  - critic value mean/std/min/max;
  - TD-error mean/std/abs-mean/max-abs;
  - actor/critic ObGD step-size median/p90/max, via a light periodic
    host-side SAMPLE (every 100 steps, not every step -- a small,
    bounded number of extra syncs, not a per-step cost);
  - actor/critic update norms (RMS, as v1) PLUS separately for encoder
    vs RTU parameter groups;
  - RTU radius (r) AND angle (theta) stats, plus checkpoint-to-
    checkpoint drift (computed by the outer driver comparing
    consecutive checkpoints, no new learner-side computation);
  - recent-return over last-50 AND last-500 episode windows;
  - explicit flush=True on every print, so a live run's log is
    readable mid-flight (unlike v1, which discovered this the hard
    way -- the log stayed at 0 bytes until process exit).

Launched for the corrected seed-0 calibration run per explicit review
instruction, after the reward-scaling and eligibility-trace-reset
corrections were verified.

Run: python -m credit_memory.b28_rtu_calibration_run_v2
"""
from __future__ import annotations

import json
import time
from collections import deque
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
STEP_SIZE_SAMPLE_EVERY = 100  # light host-side sampling for percentiles


def leaves_sq_diff(old_pytree, new_pytree):
    diffs = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), old_pytree, new_pytree)
    return sum(jax.tree_util.tree_leaves(diffs))


def all_finite(pytree):
    flags = jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), pytree)
    return jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: x, flags))


def stepped_update(carry, stats, action, raw_reward, raw_obs_next, done, hp):
    old_actor_enc, old_actor_rtu = carry["actor"]["params"]["enc"], carry["actor"]["params"]["rtu"]
    old_critic_enc, old_critic_rtu = carry["critic"]["params"]["enc"], carry["critic"]["params"]["rtu"]

    new_carry, diag = full_update_step(
        carry, action, raw_reward, raw_obs_next, done,
        hp["gamma"], hp["lam"], hp["entropy_coef"], hp["actor_alpha"], hp["critic_alpha"],
        hp["actor_kappa"], hp["critic_kappa"],
    )

    upd_actor_enc_sq = leaves_sq_diff(old_actor_enc, new_carry["actor"]["params"]["enc"])
    upd_actor_rtu_sq = leaves_sq_diff(old_actor_rtu, new_carry["actor"]["params"]["rtu"])
    upd_critic_enc_sq = leaves_sq_diff(old_critic_enc, new_carry["critic"]["params"]["enc"])
    upd_critic_rtu_sq = leaves_sq_diff(old_critic_rtu, new_carry["critic"]["params"]["rtu"])

    new_stats = dict(
        n=stats["n"] + 1.0,
        sum_entropy=stats["sum_entropy"] + diag["entropy"],
        min_entropy=jnp.minimum(stats["min_entropy"], diag["entropy"]),
        max_entropy=jnp.maximum(stats["max_entropy"], diag["entropy"]),
        sum_value=stats["sum_value"] + diag["value_cur"],
        sum_value_sq=stats["sum_value_sq"] + diag["value_cur"] ** 2,
        min_value=jnp.minimum(stats["min_value"], diag["value_cur"]),
        max_value=jnp.maximum(stats["max_value"], diag["value_cur"]),
        sum_td=stats["sum_td"] + diag["td_error"],
        sum_td_sq=stats["sum_td_sq"] + diag["td_error"] ** 2,
        sum_abs_td=stats["sum_abs_td"] + jnp.abs(diag["td_error"]),
        max_abs_td=jnp.maximum(stats["max_abs_td"], jnp.abs(diag["td_error"])),
        sum_step_actor=stats["sum_step_actor"] + diag["step_actor"],
        sum_step_critic=stats["sum_step_critic"] + diag["step_critic"],
        sum_upd_actor_enc_sq=stats["sum_upd_actor_enc_sq"] + upd_actor_enc_sq,
        sum_upd_actor_rtu_sq=stats["sum_upd_actor_rtu_sq"] + upd_actor_rtu_sq,
        sum_upd_critic_enc_sq=stats["sum_upd_critic_enc_sq"] + upd_critic_enc_sq,
        sum_upd_critic_rtu_sq=stats["sum_upd_critic_rtu_sq"] + upd_critic_rtu_sq,
    )
    return new_carry, new_stats, diag["step_actor"], diag["step_critic"]


def zero_stats():
    z = jnp.asarray(0.0)
    return dict(n=z, sum_entropy=z, min_entropy=jnp.asarray(1e9), max_entropy=jnp.asarray(-1e9),
                sum_value=z, sum_value_sq=z, min_value=jnp.asarray(1e9), max_value=jnp.asarray(-1e9),
                sum_td=z, sum_td_sq=z, sum_abs_td=z, max_abs_td=z,
                sum_step_actor=z, sum_step_critic=z,
                sum_upd_actor_enc_sq=z, sum_upd_actor_rtu_sq=z,
                sum_upd_critic_enc_sq=z, sum_upd_critic_rtu_sq=z)


def rtu_r_theta_stats(rtu_params):
    r = np.exp(-np.exp(np.asarray(rtu_params["nu_log"])))
    theta = np.exp(np.asarray(rtu_params["theta_log"]))
    return dict(r_mean=float(r.mean()), r_min=float(r.min()), r_max=float(r.max()),
                theta_mean=float(theta.mean()), theta_min=float(theta.min()), theta_max=float(theta.max()))


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
    all_episode_returns = []
    ep_return = 0.0
    frame = 0
    next_checkpoint_idx = 0
    t_start = time.time()
    t_last_log = t_start
    frames_last_log = 0
    history = []
    prev_r_theta = dict(actor=None, critic=None)

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    watch_steps, play_steps, play_correct = 0, 0, 0
    step_actor_samples, step_critic_samples = [], []

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

    print(f"=== B28 faithful-as-practical RTU Autoencode calibration v2 -- seed={seed} ===", flush=True)
    print(f"hp={HP} hidden={HIDDEN_DIM} width={WIDTH} num_envs=1", flush=True)

    while frame < total_frames:
        z_actor = jnp.concatenate([carry["actor"]["real"], carry["actor"]["imag"]])
        logits = reft.head_forward(z_actor, carry["actor"]["params"]["head"])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(action_rng.choice(len(probs), p=probs))
        action_counts[a_t] += 1

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        ep_return += r_t

        # WATCH-vs-PLAY split via the OBSERVABLE mode flag (obs is a
        # Tuple(mode, suit)). Confirmed from popgym source
        # (popgym.envs.autoencode.Mode): PLAY=0, WATCH=1 -- the
        # OPPOSITE of the naive guess -- read externally from the
        # returned observation, no env-internals access.
        is_play = (obs_next[0] == 0) if not done else None
        if is_play is not None:
            if is_play:
                play_steps += 1
                if r_t > 0:
                    play_correct += 1
            else:
                watch_steps += 1

        if done:
            obs_next_arr = np.zeros(IN_DIM)
        else:
            obs_next_arr = np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)

        carry, stats, step_a, step_c = stepped_jit(
            carry, stats, jnp.asarray(a_t), jnp.asarray(float(r_t)),
            jnp.asarray(obs_next_arr), jnp.asarray(1.0 if done else 0.0))
        frame += 1

        if frame % STEP_SIZE_SAMPLE_EVERY == 0:
            step_actor_samples.append(float(step_a))
            step_critic_samples.append(float(step_c))

        if done:
            episode_returns.append(ep_return)
            all_episode_returns.append((frame, ep_return))
            ep_return = 0.0
            carry = reset_episode(env, carry)

        if frame % log_every == 0:
            jax.block_until_ready(carry["actor"]["params"]["rtu"]["nu_log"])
            fin = bool(all_finite(carry))
            if not fin:
                print(f"!!! NON-FINITE DETECTED at frame {frame} -- stopping.", flush=True)
                break
            n = float(stats["n"])
            if n > 0:
                s = {k: float(v) for k, v in stats.items()}
                dt = time.time() - t_last_log
                fps = (frame - frames_last_log) / max(dt, 1e-9)
                print(f"  frame={frame:>9,d}  fps={fps:6.1f}  "
                      f"entropy={s['sum_entropy']/n:.4f} [{s['min_entropy']:.4f},{s['max_entropy']:.4f}]  "
                      f"value={s['sum_value']/n:+.4f}  "
                      f"td_mean={s['sum_td']/n:+.4f}  td_abs_mean={s['sum_abs_td']/n:.4f}  "
                      f"td_max_abs={s['max_abs_td']:.3f}  "
                      f"step_actor={s['sum_step_actor']/n:.4f}  step_critic={s['sum_step_critic']/n:.4f}  "
                      f"upd_actor(enc/rtu)_rms={np.sqrt(s['sum_upd_actor_enc_sq']/n):.2e}/{np.sqrt(s['sum_upd_actor_rtu_sq']/n):.2e}  "
                      f"upd_critic(enc/rtu)_rms={np.sqrt(s['sum_upd_critic_enc_sq']/n):.2e}/{np.sqrt(s['sum_upd_critic_rtu_sq']/n):.2e}  "
                      f"n_episodes={len(episode_returns)}", flush=True)
            stats = zero_stats()
            t_last_log = time.time()
            frames_last_log = frame

        if next_checkpoint_idx < len(CHECKPOINTS) and frame >= CHECKPOINTS[next_checkpoint_idx]:
            jax.block_until_ready(carry)
            fin = bool(all_finite(carry))
            recent50 = episode_returns[-50:] if episode_returns else []
            recent500 = episode_returns[-500:] if episode_returns else []
            r_theta_actor = rtu_r_theta_stats(carry["actor"]["params"]["rtu"])
            r_theta_critic = rtu_r_theta_stats(carry["critic"]["params"]["rtu"])
            drift_actor = None
            drift_critic = None
            if prev_r_theta["actor"] is not None:
                drift_actor = {k: r_theta_actor[k] - prev_r_theta["actor"][k] for k in r_theta_actor}
                drift_critic = {k: r_theta_critic[k] - prev_r_theta["critic"][k] for k in r_theta_critic}
            prev_r_theta["actor"], prev_r_theta["critic"] = r_theta_actor, r_theta_critic

            elapsed = time.time() - t_start
            total_play = watch_steps + play_steps
            action_dist = (action_counts / max(action_counts.sum(), 1)).tolist()

            def pct(samples, p):
                return float(np.percentile(samples, p)) if samples else None

            ckpt = dict(
                frame=frame, elapsed_s=elapsed, fps_avg=frame / elapsed,
                n_episodes=len(episode_returns),
                recent50_return_mean=float(np.mean(recent50)) if recent50 else None,
                recent50_return_median=float(np.median(recent50)) if recent50 else None,
                recent500_return_mean=float(np.mean(recent500)) if recent500 else None,
                recent500_return_median=float(np.median(recent500)) if recent500 else None,
                all_finite=fin,
                obs_stats_mean=np.asarray(carry["obs_stats"]["mean"]).tolist(),
                obs_stats_var=np.asarray(carry["obs_stats"]["var"]).tolist(),
                reward_scale_trace=float(carry["reward_stats"]["trace"]),
                reward_scale_mean=float(carry["reward_stats"]["stats"]["mean"]),
                reward_scale_var=float(carry["reward_stats"]["stats"]["var"]),
                rtu_r_theta_actor=r_theta_actor, rtu_r_theta_critic=r_theta_critic,
                rtu_drift_actor=drift_actor, rtu_drift_critic=drift_critic,
                action_distribution=action_dist,
                watch_steps=watch_steps, play_steps=play_steps,
                play_accuracy=(play_correct / play_steps) if play_steps else None,
                step_actor_median=pct(step_actor_samples, 50), step_actor_p90=pct(step_actor_samples, 90),
                step_actor_max=(max(step_actor_samples) if step_actor_samples else None),
                step_critic_median=pct(step_critic_samples, 50), step_critic_p90=pct(step_critic_samples, 90),
                step_critic_max=(max(step_critic_samples) if step_critic_samples else None),
            )
            history.append(ckpt)
            action_counts[:] = 0
            watch_steps, play_steps, play_correct = 0, 0, 0
            step_actor_samples, step_critic_samples = [], []

            print("=" * 70, flush=True)
            print(f"CHECKPOINT frame={frame:,}  elapsed={elapsed/60:.1f}min  avg_fps={frame/elapsed:.1f}", flush=True)
            print(f"  episodes so far: {len(episode_returns)}  "
                  f"recent50 mean/median={ckpt['recent50_return_mean']}/{ckpt['recent50_return_median']}  "
                  f"recent500 mean/median={ckpt['recent500_return_mean']}/{ckpt['recent500_return_median']}", flush=True)
            print(f"  all_finite={fin}", flush=True)
            print(f"  reward_scale: trace={ckpt['reward_scale_trace']:.4f} mean={ckpt['reward_scale_mean']:.4f} "
                  f"var={ckpt['reward_scale_var']:.4f}", flush=True)
            print(f"  action_distribution={action_dist}", flush=True)
            print(f"  watch_steps={ckpt['watch_steps']} play_steps={ckpt['play_steps']} "
                  f"play_accuracy={ckpt['play_accuracy']}", flush=True)
            print(f"  step_actor median/p90/max={ckpt['step_actor_median']}/{ckpt['step_actor_p90']}/{ckpt['step_actor_max']}", flush=True)
            print(f"  step_critic median/p90/max={ckpt['step_critic_median']}/{ckpt['step_critic_p90']}/{ckpt['step_critic_max']}", flush=True)
            print(f"  RTU actor r/theta: {r_theta_actor}  drift={drift_actor}", flush=True)
            print(f"  RTU critic r/theta: {r_theta_critic}  drift={drift_critic}", flush=True)
            print("=" * 70, flush=True)

            if out_path:
                with open(out_path, "w") as f:
                    json.dump(dict(history=history, episode_returns=all_episode_returns), f, indent=2)
            next_checkpoint_idx += 1
            if not fin:
                print("STOPPING due to non-finite state.", flush=True)
                break

    print("DONE" if frame >= total_frames else "STOPPED EARLY", flush=True)
    return history, all_episode_returns


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    run(seed=0, total_frames=5_000_000, out_path=out)
