"""Phase B28 -- OURS Autoencode calibration run, on the IDENTICAL
corrected outer scaffold verified for RTU (commit 6c65863). Same v2
checkpoint/diagnostic format as credit_memory/b28_rtu_calibration_run_v2.py
for direct comparison. Single-seed first-read transfer test -- NOT a
final statistical claim.

Capacity-matched configuration (see accompanying audit): r=16, k=2,
n=24 (nr=384, matching RTU hidden=192 -> 2*192=384 head-input features
exactly), encoder output width=64 (u_dim into the recurrence -- MUST
match RTU's actual Dense-layer output dimension exactly, per the
frozen scaffold; this is NOT the one-hot input dimension, which is a
separate IN_DIM=6), hidden=8 (Phi MLP width).

Run: python -m credit_memory.b28_ours_calibration_run
"""
from __future__ import annotations

import json
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

import popgym

from credit_memory.b28_popgym_stage1 import one_hot_obs
from credit_memory import b28_rtu_faithful_train as reft
from credit_memory.b25_nonlinear_credit import family_dim
from credit_memory.b28_rtu_faithful_jit import (
    head_forward, _running_update, _running_normalize,
)
from credit_memory.b28_ours_faithful_jit import (
    make_ours_network, ours_streaming_init, zero_ours_traces, make_ours_carry,
    ours_net_streaming_step_jit, ours_full_update_step, ours_network_param_count,
    reward_scale_init as ours_reward_scale_init,
)

jax.config.update("jax_enable_x64", True)

HP = dict(gamma=0.99, lam=0.8, entropy_coef=0.095, actor_alpha=1.0, critic_alpha=1.0,
          actor_kappa=3.0, critic_kappa=2.0)
# WIDTH is the ENCODER OUTPUT dimension (Dense->LayerNorm->LeakyReLU),
# i.e. the u_dim fed into the recurrence -- MUST match RTU's actual
# encoder output (64, per b28_rtu_calibration_run_v2.py's WIDTH=64),
# NOT the one-hot input dimension (that's IN_DIM=6, separate). An
# earlier draft of this file wrongly set WIDTH=6, conflating the two
# and giving ours a 6-dim recurrence input instead of RTU's actual
# 64-dim encoder output -- caught before launch, fixed here.
R, K, N, WIDTH, HIDDEN = 16, 2, 24, 64, 8
IN_DIM, NUM_ACTIONS = 6, 4
# First commitment horizon is 1M frames (not the full 5M) -- resumable
# via state_path so the SAME run can continue toward 2M/3M/4M/5M later
# without restarting, per explicit instruction.
CHECKPOINTS = [100_000, 250_000, 500_000, 1_000_000]


def leaves_sq_diff(old_pytree, new_pytree):
    diffs = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), old_pytree, new_pytree)
    return sum(jax.tree_util.tree_leaves(diffs))


def all_finite(pytree):
    flags = jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), pytree)
    return jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: x, flags))


def stepped_update(carry, stats, action, raw_reward, raw_obs_next, done):
    old_actor_enc, old_actor_ours = carry["actor"]["params"]["enc"], carry["actor"]["params"]["ours"]
    old_critic_enc, old_critic_ours = carry["critic"]["params"]["enc"], carry["critic"]["params"]["ours"]

    new_carry, diag = ours_full_update_step(
        carry, action, raw_reward, raw_obs_next, done,
        HP["gamma"], HP["lam"], HP["entropy_coef"], HP["actor_alpha"], HP["critic_alpha"],
        HP["actor_kappa"], HP["critic_kappa"], R, K, N, WIDTH, HIDDEN,
    )
    upd_actor_enc_sq = leaves_sq_diff(old_actor_enc, new_carry["actor"]["params"]["enc"])
    upd_actor_ours_sq = leaves_sq_diff(old_actor_ours, new_carry["actor"]["params"]["ours"])
    upd_critic_enc_sq = leaves_sq_diff(old_critic_enc, new_carry["critic"]["params"]["enc"])
    upd_critic_ours_sq = leaves_sq_diff(old_critic_ours, new_carry["critic"]["params"]["ours"])

    new_stats = dict(
        n=stats["n"] + 1.0,
        sum_entropy=stats["sum_entropy"] + diag["entropy"],
        min_entropy=jnp.minimum(stats["min_entropy"], diag["entropy"]),
        max_entropy=jnp.maximum(stats["max_entropy"], diag["entropy"]),
        sum_value=stats["sum_value"] + diag["value_cur"],
        sum_td=stats["sum_td"] + diag["td_error"],
        sum_abs_td=stats["sum_abs_td"] + jnp.abs(diag["td_error"]),
        max_abs_td=jnp.maximum(stats["max_abs_td"], jnp.abs(diag["td_error"])),
        sum_step_actor=stats["sum_step_actor"] + diag["step_actor"],
        sum_step_critic=stats["sum_step_critic"] + diag["step_critic"],
        sum_upd_actor_enc_sq=stats["sum_upd_actor_enc_sq"] + upd_actor_enc_sq,
        sum_upd_actor_ours_sq=stats["sum_upd_actor_ours_sq"] + upd_actor_ours_sq,
        sum_upd_critic_enc_sq=stats["sum_upd_critic_enc_sq"] + upd_critic_enc_sq,
        sum_upd_critic_ours_sq=stats["sum_upd_critic_ours_sq"] + upd_critic_ours_sq,
    )
    return new_carry, new_stats, diag["step_actor"], diag["step_critic"]


def zero_stats():
    z = jnp.asarray(0.0)
    return dict(n=z, sum_entropy=z, min_entropy=jnp.asarray(1e9), max_entropy=jnp.asarray(-1e9),
                sum_value=z, sum_td=z, sum_abs_td=z, max_abs_td=z,
                sum_step_actor=z, sum_step_critic=z,
                sum_upd_actor_enc_sq=z, sum_upd_actor_ours_sq=z,
                sum_upd_critic_enc_sq=z, sum_upd_critic_ours_sq=z)


def save_state(path, env, carry, episode_returns, all_episode_returns, frame, next_checkpoint_idx,
               action_rng_state, history, ep_return, stats, action_counts, watch_steps,
               play_steps, play_correct, step_actor_samples, step_critic_samples):
    """Pickle the FULL state needed to continue training bit-identically
    from a checkpoint: the popgym env object itself (its internal RNG
    and in-episode position -- verified pickle-roundtrip-faithful, see
    the split-run equivalence test), the learner carry (params, traces,
    streaming RTRL sensitivity, obs/reward-scale stats), the partial
    episode return, the action-sampling RNG, checkpoint bookkeeping, and
    the window-local diagnostic counters. This is intentionally NOT an
    episode-boundary-only checkpoint: carry["actor"/"critic"]["h"/"X"]
    already encode whatever observation the env is currently positioned
    on (see invariant note in run()), so no reset_episode() call is
    needed or correct on resume."""
    np_carry = jax.tree_util.tree_map(np.asarray, carry)
    np_stats = jax.tree_util.tree_map(np.asarray, stats)
    with open(path, "wb") as f:
        pickle.dump(dict(env=env, carry=np_carry, episode_returns=episode_returns,
                         all_episode_returns=all_episode_returns, frame=frame,
                         next_checkpoint_idx=next_checkpoint_idx,
                         action_rng_state=action_rng_state, history=history,
                         ep_return=ep_return, stats=np_stats,
                         action_counts=action_counts, watch_steps=watch_steps,
                         play_steps=play_steps, play_correct=play_correct,
                         step_actor_samples=step_actor_samples,
                         step_critic_samples=step_critic_samples), f)


def load_state(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    d["carry"] = jax.tree_util.tree_map(jnp.asarray, d["carry"])
    d["stats"] = jax.tree_util.tree_map(jnp.asarray, d["stats"])
    return d


def run(seed=0, total_frames=5_000_000, log_every=2000, out_path=None, state_path=None,
        resume=False, checkpoints=None):
    global CHECKPOINTS
    if checkpoints is not None:
        CHECKPOINTS = checkpoints
    stepped_jit = jax.jit(stepped_update)

    if resume and state_path is not None:
        saved = load_state(state_path)
        env = saved["env"]
        carry = saved["carry"]
        episode_returns = saved["episode_returns"]
        all_episode_returns = saved["all_episode_returns"]
        frame = saved["frame"]
        next_checkpoint_idx = saved["next_checkpoint_idx"]
        history = saved["history"]
        action_rng = np.random.RandomState(seed + 10_000)
        action_rng.set_state(saved["action_rng_state"])
        ep_return = saved["ep_return"]
        stats = saved["stats"]
        action_counts = saved["action_counts"]
        watch_steps = saved["watch_steps"]
        play_steps = saved["play_steps"]
        play_correct = saved["play_correct"]
        step_actor_samples = saved["step_actor_samples"]
        step_critic_samples = saved["step_critic_samples"]
        print(f"=== RESUMED from {state_path} at frame={frame:,} "
              f"(env+RNG+carry restored bit-for-bit; NOT resetting episode) ===", flush=True)
    else:
        env = popgym.envs.autoencode.AutoencodeEasy()
        rng = np.random.RandomState(seed)
        actor_net = make_ours_network(rng, IN_DIM, WIDTH, R, K, N, HIDDEN, NUM_ACTIONS, seed=seed)
        critic_net = make_ours_network(rng, IN_DIM, WIDTH, R, K, N, HIDDEN, 1, seed=seed + 1)
        m_psi = family_dim("psi", dict(r=R, k=K, n=N, u_dim=WIDTH, hidden=HIDDEN, psi=actor_net["ours"]["psi"]))
        actor_stream = ours_streaming_init(R, K, N, m_psi, actor_net["enc"])
        critic_stream = ours_streaming_init(R, K, N, m_psi, critic_net["enc"])
        obs_stats = reft.running_stats_init((IN_DIM,))
        reward_stats = ours_reward_scale_init()
        carry = make_ours_carry(actor_net, actor_stream, zero_ours_traces(actor_net),
                                 critic_net, critic_stream, zero_ours_traces(critic_net),
                                 obs_stats, reward_stats)
        episode_returns = []
        all_episode_returns = []
        frame = 0
        next_checkpoint_idx = 0
        history = []
        action_rng = np.random.RandomState(seed + 10_000)
        ep_return = 0.0
        stats = zero_stats()
        action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        watch_steps, play_steps, play_correct = 0, 0, 0
        step_actor_samples, step_critic_samples = [], []

    t_start = time.time()
    t_last_log = t_start
    frames_last_log = 0

    # Invariant maintained at every point in the main loop (including
    # right after this initial call, right after a done-triggered
    # reset_episode(), and right after an ordinary stepped_jit() update):
    # carry["actor"/"critic"]["h"/"X"] encode the CURRENT (about-to-be
    # -acted-on) observation, and `env` is positioned so env.step(a_t)
    # continues from that same observation. A checkpoint save captures
    # both halves of this invariant together, so resuming must NOT call
    # reset_episode() -- doing so would splice a fresh episode onto the
    # loaded learner state.
    def reset_episode(env, carry, seed=None):
        # gymnasium semantics: reset(seed=None) preserves the env's
        # existing internal RNG stream (does NOT reseed); reset(seed=N)
        # reseeds. Pass seed only for the very first reset of a fresh
        # (non-resumed) run, so env randomness is reproducible from
        # `seed` while later within-run episodes still vary.
        obs, _ = env.reset(seed=seed)
        u0 = np.asarray(one_hot_obs(obs, "autoencode"), dtype=np.float64)
        obs_stats_new = _running_update(carry["obs_stats"], jnp.asarray(u0))
        u0n = _running_normalize(obs_stats_new, jnp.asarray(u0))
        a_h, _, a_X = ours_net_streaming_step_jit(
            carry["actor"]["params"]["ours"], carry["actor"]["params"]["enc"],
            jnp.zeros((N, R)), {fam: jnp.zeros_like(v) for fam, v in carry["actor"]["X"].items()},
            u0n, R, K, N, WIDTH, HIDDEN)
        c_h, _, c_X = ours_net_streaming_step_jit(
            carry["critic"]["params"]["ours"], carry["critic"]["params"]["enc"],
            jnp.zeros((N, R)), {fam: jnp.zeros_like(v) for fam, v in carry["critic"]["X"].items()},
            u0n, R, K, N, WIDTH, HIDDEN)
        new_carry = dict(carry)
        new_carry["actor"] = dict(carry["actor"], h=a_h, X=a_X)
        new_carry["critic"] = dict(carry["critic"], h=c_h, X=c_X)
        new_carry["obs_stats"] = obs_stats_new
        return new_carry

    if not resume:
        carry = reset_episode(env, carry, seed=seed)

    print(f"=== B28 OURS Autoencode calibration -- seed={seed} r={R} k={K} n={N} width={WIDTH} hidden={HIDDEN} ===", flush=True)
    print(f"hp={HP} nr={N*R} num_envs=1", flush=True)

    while frame < total_frames:
        z_actor = carry["actor"]["h"].reshape(-1)
        logits = head_forward(z_actor, carry["actor"]["params"]["head"])
        probs = np.asarray(jax.nn.softmax(logits))
        probs = probs / probs.sum()
        a_t = int(action_rng.choice(len(probs), p=probs))
        action_counts[a_t] += 1

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        ep_return += r_t

        is_play = (obs_next[0] == 0) if not done else None
        if is_play is not None:
            if is_play:
                play_steps += 1
                if r_t > 0:
                    play_correct += 1
            else:
                watch_steps += 1

        obs_next_arr = np.zeros(IN_DIM) if done else np.asarray(one_hot_obs(obs_next, "autoencode"), dtype=np.float64)

        carry, stats, step_a, step_c = stepped_jit(carry, stats, jnp.asarray(a_t), jnp.asarray(float(r_t)),
                                                    jnp.asarray(obs_next_arr), jnp.asarray(1.0 if done else 0.0))
        frame += 1

        if frame % 100 == 0:
            step_actor_samples.append(float(step_a))
            step_critic_samples.append(float(step_c))

        if done:
            episode_returns.append(ep_return)
            all_episode_returns.append((frame, ep_return))
            ep_return = 0.0
            carry = reset_episode(env, carry)

        if frame % log_every == 0:
            jax.block_until_ready(carry["actor"]["params"]["ours"]["R"])
            fin = bool(all_finite(carry))
            if not fin:
                print(f"!!! NON-FINITE DETECTED at frame {frame} -- stopping.", flush=True)
                break
            n_ = float(stats["n"])
            if n_ > 0:
                s = {k: float(v) for k, v in stats.items()}
                dt = time.time() - t_last_log
                fps = (frame - frames_last_log) / max(dt, 1e-9)
                print(f"  frame={frame:>9,d}  fps={fps:6.1f}  "
                      f"entropy={s['sum_entropy']/n_:.4f} [{s['min_entropy']:.4f},{s['max_entropy']:.4f}]  "
                      f"value={s['sum_value']/n_:+.4f}  "
                      f"td_mean={s['sum_td']/n_:+.4f}  td_abs_mean={s['sum_abs_td']/n_:.4f}  "
                      f"td_max_abs={s['max_abs_td']:.3f}  "
                      f"step_actor={s['sum_step_actor']/n_:.4f}  step_critic={s['sum_step_critic']/n_:.4f}  "
                      f"upd_actor(enc/ours)_rms={np.sqrt(s['sum_upd_actor_enc_sq']/n_):.2e}/{np.sqrt(s['sum_upd_actor_ours_sq']/n_):.2e}  "
                      f"upd_critic(enc/ours)_rms={np.sqrt(s['sum_upd_critic_enc_sq']/n_):.2e}/{np.sqrt(s['sum_upd_critic_ours_sq']/n_):.2e}  "
                      f"n_episodes={len(episode_returns)}", flush=True)
            stats = zero_stats()
            t_last_log = time.time()
            frames_last_log = frame

        if next_checkpoint_idx < len(CHECKPOINTS) and frame >= CHECKPOINTS[next_checkpoint_idx]:
            jax.block_until_ready(carry)
            fin = bool(all_finite(carry))
            recent50 = episode_returns[-50:] if episode_returns else []
            recent500 = episode_returns[-500:] if episode_returns else []
            elapsed = time.time() - t_start
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
            print("=" * 70, flush=True)

            if out_path:
                with open(out_path, "w") as f:
                    json.dump(dict(history=history, episode_returns=all_episode_returns), f, indent=2)
            next_checkpoint_idx += 1
            if state_path:
                # Save the POST-increment next_checkpoint_idx: on
                # resume, CHECKPOINTS[next_checkpoint_idx] is the NEXT
                # (not-yet-consumed) checkpoint, unambiguously.
                save_state(state_path, env, carry, episode_returns, all_episode_returns, frame,
                           next_checkpoint_idx, action_rng.get_state(), history, ep_return,
                           stats, action_counts, watch_steps, play_steps, play_correct,
                           step_actor_samples, step_critic_samples)
                print(f"  state saved to {state_path} (resumable)", flush=True)
            if not fin:
                print("STOPPING due to non-finite state.", flush=True)
                break

    print("DONE" if frame >= total_frames else "STOPPED EARLY", flush=True)
    return history, all_episode_returns


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    state = sys.argv[2] if len(sys.argv) > 2 else None
    resume_flag = (len(sys.argv) > 3 and sys.argv[3] == "resume")
    run(seed=0, total_frames=1_000_000, out_path=out, state_path=state, resume=resume_flag)
