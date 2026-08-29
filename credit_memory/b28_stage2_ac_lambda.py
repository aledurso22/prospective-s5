"""Phase B28 Stage 2, items 3/5/6/8/9 -- streaming Actor-Critic(lambda)
on real POPGym Autoencode, ours-exact-RTRL vs Nonlinear-RTU-exact-RTRL,
identical outer-learner/encoder/heads/optimizer/init/env sequence.

Classical episodic semi-gradient Actor-Critic with eligibility traces
(Sutton & Barto style): TD(0) bootstrap for the critic target (no
gradient through the bootstrap state), separate eligibility traces for
policy and value parameters, each trace accumulated from the EXACT
per-step RTRL gradient g_t (recurrent parameters) or a direct local
gradient (head parameters, which have no recurrent role at all).
Labelled explicitly as our own best-faith instantiation of the
classical AC(lambda) algorithm family -- not a verified bit-for-bit
reproduction of a specific recent paper's exact hyperparameters.

Head design choice, stated explicitly: both architectures' heads read
the FULL raw recurrent state directly (z_t = h_t for RTU, z_t =
h_t.reshape(-1) for ours) rather than a further C-style readout. This
keeps the comparison apples-to-apples with RTU (whose head already
reads the full state, per B28 Stage 1's note that RTU has no
analogous dual-role-parameter concern) and sidesteps the dual-role
bug pattern from Stage 1 entirely for the recurrent-parameter
families, since none of R,B,C,psi/theta,log_radius,Wx appear directly
in the head -- only h_t does, and d(loss)/dh_t is exactly what the
streaming RTRL sensitivity is dotted against.

Run: python -m credit_memory.b28_stage2_ac_lambda
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

import popgym

from credit_memory.b25_nonlinear_credit import make_arch, psi_flat, psi_from_flat
from credit_memory.b27_noncommutative_advantage import make_nonlinear_rtu_arch
from credit_memory.b28_popgym_stage1 import one_hot_obs
from credit_memory.b28_stage2_streaming import (
    ours_streaming_init, ours_streaming_step, ours_per_step_grad, FAMILIES_OURS,
    rtu_streaming_init, rtu_streaming_step, rtu_per_step_grad, FAMILIES_RTU,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Param (de)serialization for each recurrent family -- flat vector <-> arch.
# ---------------------------------------------------------------------------
def ours_get_flat(arch, family):
    if family == "psi":
        return np.asarray(psi_flat(arch["psi"]))
    return np.asarray(arch[family]).reshape(-1)


RHO_MAX = 0.95


def project_stable(R_mat, rho_max=RHO_MAX):
    """RTU's parameterization (rho=exp(-|log_radius|)) keeps its
    spectral radius < 1 by CONSTRUCTION, regardless of how its
    parameters are updated online. Ours' dense R has no such built-in
    guarantee -- confirmed empirically: raw online SGD-style updates
    push R's spectral radius above 1 within ~10-15 steps of a single
    episode, causing geometric blow-up (delta -> 1e140 by step 19).
    This is an architecture-robustness fix needed for ANY fairly-
    compared non-tcg recurrent core under online updates (see
    memory: b18-init-instability), not a hack specific to this
    comparison -- RTU needs no analogous projection because its own
    parameterization already provides it for free.

    Returns (R_mat_out, event) so callers can log HOW OFTEN and HOW
    STRONGLY this activates -- per review request, frequent/strong
    activation is a real algorithmic finding about ours' dense-core
    parameterization, not an invisible numerical detail to bury."""
    eigval_mag = float(np.max(np.abs(np.linalg.eigvals(R_mat))))
    projected = eigval_mag > rho_max
    scale = (rho_max / eigval_mag) if projected else 1.0
    if projected:
        R_mat = R_mat * scale
    return R_mat, dict(pre_eigval=eigval_mag, projected=projected, scale=scale)


def ours_set_flat(arch, family, flat, r, k, n, u_dim, hidden, stability_log=None):
    if family == "R":
        R_mat = np.asarray(flat).reshape(r, r)
        R_mat, event = project_stable(R_mat)
        arch["R"] = jnp.array(R_mat)
        if stability_log is not None:
            stability_log.append(event)
    elif family == "B":
        arch["B"] = jnp.array(flat).reshape(r, k)
    elif family == "C":
        arch["C"] = jnp.array(flat).reshape(k, r)
    elif family == "psi":
        arch["psi"] = psi_from_flat(jnp.array(flat), n, k, u_dim, hidden)


def rtu_get_flat(arch, family):
    if family == "theta":
        return np.asarray(arch["thetas"])
    if family == "log_radius":
        return np.asarray(arch["log_radii"])
    if family == "Wx":
        return np.asarray(arch["Wx"]).reshape(-1)
    raise ValueError(family)


def rtu_set_flat(arch, family, flat, r_rtu, u_dim):
    if family == "theta":
        arch["thetas"] = jnp.array(flat)
    elif family == "log_radius":
        arch["log_radii"] = jnp.array(flat)
    elif family == "Wx":
        arch["Wx"] = jnp.array(flat).reshape(r_rtu, u_dim)


# ---------------------------------------------------------------------------
# Heads: linear policy (num_actions,z_dim) + linear value (z_dim,) -- shared
# structure for both architectures, only z_dim differs by construction
# (chosen equal below so it does not differ in practice).
# ---------------------------------------------------------------------------
TD_ERROR_CLIP = 5.0
OBGD_KAPPA = 2.0

# First attempt used a hard per-family update-norm clip (0.5). Diagnosed
# as WRONG after the first smoke run: "ours"'s raw per-step update
# saturated the clip on EVERY single step (upd_norm pinned at a constant
# 8.66e-01 for all 200 episodes) while RTU's raw update never approached
# it (~3e-3) -- i.e. the clip silently turned "ours"'s updates into a
# constant-magnitude, direction-only random walk (no real gradient
# descent) while leaving RTU's step size effectively untouched. A fixed
# clip cannot equalize two architectures with structurally different raw
# sensitivity scales. Replaced with ObGD (Overshooting-bounded Gradient
# Descent, the "optimizer / ObGD machinery" named in the review spec,
# per the streaming-RL literature): a SMOOTH, scale-normalizing
# effective step size computed from the trace norm and TD error each
# step, rather than a hard clip -- same formula, same kappa, applied
# identically to both architectures and to head params.


def obgd_step_size(alpha, delta, trace_dict, kappa=OBGD_KAPPA):
    z_sq_sum = sum(float(np.sum(np.asarray(z) ** 2)) for z in trace_dict.values())
    denom = max(1.0, alpha * kappa * abs(delta) * z_sq_sum)
    return alpha / denom


def make_head(z_dim, num_actions, seed):
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(z_dim)
    return dict(
        W_pi=jnp.array(rng.randn(num_actions, z_dim) * scale),
        b_pi=jnp.zeros(num_actions),
        W_v=jnp.array(rng.randn(1, z_dim) * scale),
        b_v=jnp.zeros(1),
    )


def policy_logits(z, head):
    return head["W_pi"] @ z + head["b_pi"]


def value_fn(z, head):
    return (head["W_v"] @ z + head["b_v"])[0]


def log_softmax(logits):
    return logits - jax.scipy.special.logsumexp(logits)


# ---------------------------------------------------------------------------
# One streaming AC(lambda) episode. `arch_kind` in {'ours','rtu'}.
# ---------------------------------------------------------------------------
def recompute_S_from_prefix(frozen_arch, arch_kind, obs_prefix):
    """S_recomputed: replay obs_prefix from its initial recurrent state
    using CURRENT (frozen at this instant) parameters throughout --
    the "no staleness" reference. `frozen_arch` must be a snapshot
    (dict copy; param arrays are immutable jnp arrays, so a shallow
    dict copy is a safe snapshot even though training continues to
    reassign arch's own entries afterward)."""
    if arch_kind == "ours":
        r, n = frozen_arch["r"], frozen_arch["n"]
        h0 = jnp.zeros((n, r))
        ss = ours_streaming_init(frozen_arch, h0)
        S = None
        for u in obs_prefix:
            _, _, S = ours_streaming_step(frozen_arch, ss, u)
        return S
    else:
        r_rtu = frozen_arch["r_rtu"]
        h0 = jnp.zeros(r_rtu)
        ss = rtu_streaming_init(frozen_arch, h0)
        S = None
        for u in obs_prefix:
            _, S = rtu_streaming_step(frozen_arch, ss, u)
        return S


def staleness_relative_error(S_live, S_recomputed, arch_kind, families):
    """||S_live - S_recomputed|| / ||S_recomputed|| per family. Distinct
    from an implementation bug: online RTRL under CONTINUOUSLY
    updated parameters propagates each step's trace contribution
    using whatever parameters were current AT THAT STEP, so the
    accumulated live trace need not equal a fresh replay of the same
    prefix under the parameters as they stand NOW -- that mismatch IS
    staleness, measured here, not asserted to be zero."""
    out = {}
    for fam in families:
        if arch_kind == "ours":
            live = np.asarray(S_live[fam]).reshape(-1)
            rec = np.asarray(S_recomputed[fam]).reshape(-1)
        else:
            live = np.concatenate([np.asarray(b).reshape(-1) for b in S_live[fam]])
            rec = np.concatenate([np.asarray(b).reshape(-1) for b in S_recomputed[fam]])
        out[fam] = float(np.linalg.norm(live - rec) / (np.linalg.norm(rec) + 1e-12))
    return out


def run_episode(env, arch, arch_kind, head, obs_kind, hp, rng, seed, collect_traj=False,
                 staleness_every=None):
    """hp: dict with gamma, lam_v, lam_pi, alpha_v, alpha_pi (ObGD base
    step sizes for the value/policy trace groups, recurrent+head combined).
    Mutates arch and head IN PLACE (numpy/jnp arrays reassigned).
    Returns dict of episode diagnostics."""
    stability_log = []
    if arch_kind == "ours":
        families = FAMILIES_OURS
        r, k, n, u_dim, hidden = arch["r"], arch["k"], arch["n"], arch["u_dim"], arch["hidden"]
        h0 = jnp.zeros((n, r))
        stream_state = ours_streaming_init(arch, h0)
        z_dim = n * r
        step_fn = ours_streaming_step
        per_step_grad_fn = ours_per_step_grad
        get_flat = lambda fam: ours_get_flat(arch, fam)
        set_flat = lambda fam, flat: ours_set_flat(arch, fam, flat, r, k, n, u_dim, hidden, stability_log)
        readout = lambda h: jnp.asarray(h).reshape(-1)
        dz_dh_shape = (n, r)
    else:
        families = FAMILIES_RTU
        r_rtu, u_dim = arch["r_rtu"], arch["u_dim"]
        h0 = jnp.zeros(r_rtu)
        stream_state = rtu_streaming_init(arch, h0)
        z_dim = r_rtu
        step_fn = rtu_streaming_step
        per_step_grad_fn = None  # RTU per-step grad handled via rtu_per_step_grad below
        get_flat = lambda fam: rtu_get_flat(arch, fam)
        set_flat = lambda fam, flat: rtu_set_flat(arch, fam, flat, r_rtu, u_dim)
        readout = lambda h: jnp.asarray(h)
        dz_dh_shape = (r_rtu,)

    z_trace_v = {fam: np.zeros_like(get_flat(fam)) for fam in families}
    z_trace_pi = {fam: np.zeros_like(get_flat(fam)) for fam in families}
    hz_pi = dict(W_pi=np.zeros_like(head["W_pi"]), b_pi=np.zeros_like(head["b_pi"]))
    hz_v = dict(W_v=np.zeros_like(head["W_v"]), b_v=np.zeros_like(head["b_v"]))

    obs, _ = env.reset(seed=seed)
    u0 = jnp.array(one_hot_obs(obs, obs_kind))
    obs_prefix = [u0]
    if arch_kind == "ours":
        h_cur, _, S_cur = step_fn(arch, stream_state, u0)
    else:
        h_cur, S_cur = step_fn(arch, stream_state, u0)
    z_cur = readout(h_cur)

    I = 1.0
    total_return = 0.0
    n_correct = 0
    n_steps = 0
    traj = [] if collect_traj else None
    staleness_log = []

    while True:
        if staleness_every is not None and n_steps > 0 and n_steps % staleness_every == 0:
            frozen_arch = dict(arch)
            S_recomputed = recompute_S_from_prefix(frozen_arch, arch_kind, obs_prefix)
            staleness_log.append(dict(
                step=n_steps,
                rel_err=staleness_relative_error(S_cur, S_recomputed, arch_kind, families),
            ))
        logits = policy_logits(z_cur, head)
        logp = log_softmax(logits)
        probs = np.asarray(jnp.exp(logp))
        probs = probs / probs.sum()
        a_t = int(rng.choice(len(probs), p=probs))

        obs_next, r_t, term, trunc, _ = env.step(a_t)
        done = term or trunc
        total_return += r_t
        n_correct += int(r_t > 0)
        n_steps += 1

        V_cur = float(value_fn(z_cur, head))
        if done:
            V_next = 0.0
            S_next = None
            z_next = None
        else:
            u_next = jnp.array(one_hot_obs(obs_next, obs_kind))
            obs_prefix.append(u_next)
            if arch_kind == "ours":
                h_next, _, S_next = step_fn(arch, stream_state, u_next)
            else:
                h_next, S_next = step_fn(arch, stream_state, u_next)
            z_next = readout(h_next)
            V_next = float(value_fn(z_next, head))

        delta = r_t + hp["gamma"] * V_next - V_cur
        delta = float(np.clip(delta, -TD_ERROR_CLIP, TD_ERROR_CLIP))

        dV_dz = np.asarray(jax.grad(lambda z: value_fn(z, head))(z_cur))
        dlogpi_dz = np.asarray(jax.grad(lambda z: log_softmax(policy_logits(z, head))[a_t])(z_cur))

        dV_dh = dV_dz.reshape(dz_dh_shape)
        dlogpi_dh = dlogpi_dz.reshape(dz_dh_shape)

        if arch_kind == "ours":
            g_v = per_step_grad_fn(S_cur, dV_dh)
            g_pi = per_step_grad_fn(S_cur, dlogpi_dh)
        else:
            n_blocks = arch["n_blocks"]
            g_v = rtu_per_step_grad(S_cur, dV_dh, n_blocks)
            g_pi = rtu_per_step_grad(S_cur, dlogpi_dh, n_blocks)

        for fam in families:
            z_trace_v[fam] = hp["gamma"] * hp["lam_v"] * z_trace_v[fam] + g_v[fam]
            z_trace_pi[fam] = hp["gamma"] * hp["lam_pi"] * z_trace_pi[fam] + I * g_pi[fam]

        gh_v_Wv = np.asarray(jax.grad(lambda W: value_fn(z_cur, {**head, "W_v": W}))(head["W_v"]))
        gh_v_bv = np.asarray(jax.grad(lambda b: value_fn(z_cur, {**head, "b_v": b}))(head["b_v"]))
        gh_pi_Wpi = np.asarray(jax.grad(lambda W: log_softmax(policy_logits(z_cur, {**head, "W_pi": W}))[a_t])(head["W_pi"]))
        gh_pi_bpi = np.asarray(jax.grad(lambda b: log_softmax(policy_logits(z_cur, {**head, "b_pi": b}))[a_t])(head["b_pi"]))

        hz_v["W_v"] = hp["gamma"] * hp["lam_v"] * hz_v["W_v"] + gh_v_Wv
        hz_v["b_v"] = hp["gamma"] * hp["lam_v"] * hz_v["b_v"] + gh_v_bv
        hz_pi["W_pi"] = hp["gamma"] * hp["lam_pi"] * hz_pi["W_pi"] + I * gh_pi_Wpi
        hz_pi["b_pi"] = hp["gamma"] * hp["lam_pi"] * hz_pi["b_pi"] + I * gh_pi_bpi

        value_traces = {**z_trace_v, "head_W_v": hz_v["W_v"], "head_b_v": hz_v["b_v"]}
        policy_traces = {**z_trace_pi, "head_W_pi": hz_pi["W_pi"], "head_b_pi": hz_pi["b_pi"]}
        step_v = obgd_step_size(hp["alpha_v"], delta, value_traces)
        step_pi = obgd_step_size(hp["alpha_pi"], delta, policy_traces)

        update_norm_sq = 0.0
        for fam in families:
            d_theta = step_v * delta * z_trace_v[fam] + step_pi * delta * z_trace_pi[fam]
            update_norm_sq += float(np.sum(d_theta ** 2))
            n_before = len(stability_log)
            set_flat(fam, get_flat(fam) + d_theta)
            if len(stability_log) > n_before:
                stability_log[-1]["step"] = n_steps

        head["W_v"] = head["W_v"] + step_v * delta * hz_v["W_v"]
        head["b_v"] = head["b_v"] + step_v * delta * hz_v["b_v"]
        head["W_pi"] = head["W_pi"] + step_pi * delta * hz_pi["W_pi"]
        head["b_pi"] = head["b_pi"] + step_pi * delta * hz_pi["b_pi"]

        I *= hp["gamma"]
        if collect_traj:
            traj.append(dict(t=n_steps, a=a_t, r=r_t, delta=delta))

        if done:
            break
        z_cur, S_cur = z_next, S_next

    return dict(
        ret=total_return, n_correct=n_correct, n_steps=n_steps,
        update_norm=float(np.sqrt(update_norm_sq)),
        traj=traj, staleness_log=staleness_log, stability_log=stability_log,
    )


def make_env(task):
    if task == "autoencode":
        return popgym.envs.autoencode.AutoencodeEasy(), "autoencode"
    if task == "repeat_first":
        return popgym.envs.repeat_first.RepeatFirstEasy(), "discrete4"
    raise ValueError(task)


def train(task, arch_kind, num_episodes, seed, hp, log_every=25, staleness_every_ep=20, staleness_step=15):
    env, obs_kind = make_env(task)
    u_dim = 6 if obs_kind == "autoencode" else 4
    num_actions = env.action_space.n
    rng = np.random.RandomState(seed)

    if arch_kind == "ours":
        r, k, n, hidden = 4, 2, 4, 8
        arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
        arch["u_dim"] = u_dim
        z_dim = n * r
        state_size = n * r
        param_count = r * r + r * k + k * r + psi_flat(arch["psi"]).shape[0]
        credit_floats = param_count * state_size  # (n,r,m) per family, summed
    else:
        r_rtu, hidden = 16, 8
        arch = make_nonlinear_rtu_arch(r_rtu=r_rtu, u_dim=u_dim, hidden=hidden, seed=seed)
        z_dim = r_rtu
        state_size = r_rtu
        param_count = r_rtu // 2 * 2 + r_rtu * u_dim
        credit_floats = 2 * r_rtu * (1 + u_dim)

    head = make_head(z_dim, num_actions, seed=seed + 1000)
    head_param_count = int(np.prod(head["W_pi"].shape) + head["b_pi"].shape[0]
                            + np.prod(head["W_v"].shape) + head["b_v"].shape[0])
    returns = []
    staleness_history = []
    stability_events = []  # flat list of {ep, step, pre_eigval, projected, scale}
    t0 = time.time()
    for ep in range(num_episodes):
        do_staleness = (ep % staleness_every_ep == 0)
        stats = run_episode(env, arch, arch_kind, head, obs_kind, hp, rng, seed=seed * 100000 + ep,
                             staleness_every=staleness_step if do_staleness else None)
        returns.append(stats["ret"])
        if stats["staleness_log"]:
            staleness_history.append(dict(ep=ep, log=stats["staleness_log"]))
        for e in stats["stability_log"]:
            stability_events.append(dict(ep=ep, **e))
        if (ep + 1) % log_every == 0:
            recent = returns[-log_every:]
            msg = (f"    [{arch_kind:5s} seed={seed}] ep {ep + 1:4d}/{num_episodes}  "
                   f"return(mean last {log_every})={np.mean(recent):+.4f}  "
                   f"n_steps={stats['n_steps']:3d}  upd_norm={stats['update_norm']:.2e}  "
                   f"elapsed={time.time() - t0:.1f}s")
            if stats["staleness_log"]:
                worst = max(max(e["rel_err"].values()) for e in stats["staleness_log"])
                msg += f"  staleness(max)={worst:.3f}"
            if stats["stability_log"]:
                n_proj = sum(1 for e in stats["stability_log"] if e["projected"])
                msg += f"  R_projections_this_ep={n_proj}/{len(stats['stability_log'])}"
            print(msg)
    stability_summary = summarize_stability(stability_events)
    return dict(returns=returns, arch=arch, head=head, state_size=state_size,
                param_count=param_count, head_param_count=head_param_count,
                credit_floats=credit_floats, staleness_history=staleness_history,
                stability_events=stability_events, stability_summary=stability_summary)


def summarize_stability(stability_events):
    """Per review item 2: report projection ACTIVITY explicitly rather
    than treat it as an invisible numerical detail. Empty for RTU
    (which has no analogous projection) and for architectures with
    zero R updates observed (e.g. a 0-episode run)."""
    if not stability_events:
        return dict(n_events=0, frac_projected=0.0,
                     median_pre_eigval=None, max_pre_eigval=None,
                     median_scale_when_projected=None, min_scale_when_projected=None)
    pre_eigvals = [e["pre_eigval"] for e in stability_events]
    projected_scales = [e["scale"] for e in stability_events if e["projected"]]
    n_projected = len(projected_scales)
    return dict(
        n_events=len(stability_events),
        frac_projected=n_projected / len(stability_events),
        median_pre_eigval=float(np.median(pre_eigvals)),
        max_pre_eigval=float(np.max(pre_eigvals)),
        median_scale_when_projected=float(np.median(projected_scales)) if n_projected else None,
        min_scale_when_projected=float(np.min(projected_scales)) if n_projected else None,
    )


def stability_events_in_window(stability_events, ep, up_to_step):
    return [e for e in stability_events if e["ep"] == ep and e["step"] <= up_to_step]


def run_multiseed(task, arch_kind, num_episodes, seeds, hp, log_every=999999):
    all_res = []
    for seed in seeds:
        res = train(task, arch_kind, num_episodes, seed, hp, log_every=log_every)
        all_res.append(res)
        final = np.mean(res["returns"][-max(10, num_episodes // 10):])

        print(f"  [{task}/{arch_kind}] seed={seed}  final return={final:+.4f}  "
              f"state={res['state_size']} "
              f"params={res['param_count']}+{res['head_param_count']}(head) "
              f"credit_floats={res['credit_floats']}")

        for entry in res["staleness_history"]:
            worst_step = max(entry["log"], key=lambda e: max(e["rel_err"].values()))
            n_proj_in_window = (
                sum(1 for e in stability_events_in_window(res["stability_events"], entry["ep"], worst_step["step"])
                    if e["projected"])
                if arch_kind == "ours" else None
            )
            worst = max(worst_step["rel_err"].values())
            proj_note = f"  R_projections_up_to_checkpoint={n_proj_in_window}" if n_proj_in_window is not None else ""
            print(f"    staleness ep={entry['ep']} step={worst_step['step']} "
                  f"max_rel_err={worst:.3f}{proj_note}")

        if arch_kind == "ours":
            s = res["stability_summary"]
            print(f"    R stability projection: n_events={s['n_events']}  "
                  f"frac_projected={s['frac_projected']:.3f}  "
                  f"pre_eigval(median/max)={s['median_pre_eigval']}/{s['max_pre_eigval']}  "
                  f"scale_when_projected(median/min)={s['median_scale_when_projected']}/{s['min_scale_when_projected']}")
    return all_res


def main():
    hp = dict(gamma=0.97, lam_v=0.8, lam_pi=0.8, alpha_v=0.5, alpha_pi=0.5)
    seeds = (0, 1, 2)

    print("=" * 70)
    print("PIPELINE SMOKE TEST ONLY (not the official simple-memory-control")
    print("condition -- that is item 4, RepeatFirst+QRC(lambda)): stream")
    print("AC(lambda) machinery sanity-checked on the short, cheap RepeatFirst")
    print("task before spending compute on the longer Autoencode episodes.")
    print("=" * 70)
    run_multiseed("repeat_first", "rtu", num_episodes=200, seeds=seeds[:1], hp=hp, log_every=50)
    run_multiseed("repeat_first", "ours", num_episodes=200, seeds=seeds[:1], hp=hp, log_every=50)

    print("=" * 70)
    print("Stage 2, item 3: AUTOENCODE + stream AC(lambda) -- RTU replication")
    print("sanity control, then ours-vs-RTU learning curves.")
    print("=" * 70)
    res_rtu_ae = run_multiseed("autoencode", "rtu", num_episodes=300, seeds=seeds, hp=hp, log_every=50)
    res_ours_ae = run_multiseed("autoencode", "ours", num_episodes=300, seeds=seeds, hp=hp, log_every=50)


if __name__ == "__main__":
    main()
