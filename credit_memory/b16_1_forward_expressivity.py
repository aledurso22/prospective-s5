"""Phase B16.1 -- forward-expressivity/scaling test for tied-pole
architectures. Ordinary BPTT training only; no new online-credit rule.
Tests whether model width N can grow independently of the number of
distinct temporal recurrence classes G required for the task.

Weight tying is implemented exactly (not just at init): after each
BPTT gradient computation, the flat gradient's rho[1]/theta[1]
entries are aggregated (summed) within each tied group and broadcast
back before the Adam update -- so all group members receive identical
updates at every step and remain exactly tied throughout training
(standard parameter-sharing backward pass, reusing tcg.flat_grads'
already-correct rho/theta chain rule unmodified).

Run:  python -m credit_memory.b16_1_forward_expressivity
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import set_l2_config
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b12_structural_spectral_theory import make_multi_delay_task
from credit_memory.b13_common_temporal_support import make_multi_freq_task

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b16_1")

T_TASK = 60
BATCH_TASK = 8
DELAY = 20
LR = 1e-3
STEPS = 300


def a1_slice(N_):
    """rho[1]/theta[1] occupy [2N, 4N) in the flatten() layout for
    L=2, M_IN=whatever: layer 0 occupies [0,2N) for its own a, then
    b[0] (size N*M_IN complex = 2*N*M_IN real), THEN layer 1's a."""
    return None  # computed dynamically per M_IN, see offset_a1 below


def offset_a1(N_, M_IN_):
    """Exact flatten() offset of rho[1] (theta[1] immediately follows,
    each length N_): layer 0's a (2N_) + layer 0's b (2*N_*M_IN_ real)."""
    return 2 * N_ + 2 * N_ * M_IN_


def make_group_map(N_, G, rng, mode="contiguous"):
    if mode == "contiguous":
        return np.array([j % G for j in range(N_)])
    elif mode == "random":
        return rng.randint(0, G, size=N_)
    raise ValueError(mode)


def init_tied_params(seed, N_, G, M_IN_, g_of_j, mag_range=(0.90, 0.995)):
    rng = np.random.RandomState(seed)
    old_M_IN = tcg.M_IN
    tcg.M_IN = M_IN_
    try:
        with set_l2_config(N_, T_TASK, BATCH_TASK):
            params = tcg.init_params(seed)
    finally:
        tcg.M_IN = old_M_IN
    mags = np.linspace(mag_range[0], mag_range[1], G)
    phases = rng.uniform(-np.pi, np.pi, G)
    rho_g = -np.log(1.0 / mags - 1.0)          # inverse sigmoid
    theta_g = phases
    params["rho"][1] = rho_g[g_of_j]
    params["theta"][1] = theta_g[g_of_j]
    params["a"] = tcg.a_of(params)
    return params


def tie_flat_gradient(g, N_, M_IN_, g_of_j, G):
    """Standard weight-tying backward pass: sum the gradient across all
    channels sharing a group, broadcast the sum back so every tied
    channel receives an identical update (keeping them exactly tied
    throughout training, not just at init)."""
    off = offset_a1(N_, M_IN_)
    for half in (0, 1):    # 0: rho block, 1: theta block
        seg = g[off + half * N_: off + half * N_ + N_]
        agg = np.zeros(G)
        for j in range(N_):
            agg[g_of_j[j]] += seg[j]
        for j in range(N_):
            seg[j] = agg[g_of_j[j]]
    return g


def train_tied(seed, N_, G, M_IN_, task_fn, delays_or_freqs, group_mode="contiguous",
              steps=STEPS, freeze_pole=False):
    rng = np.random.RandomState(4000 + seed)
    g_of_j = make_group_map(N_, G, rng, mode=group_mode)
    params = init_tied_params(seed, N_, G, M_IN_, g_of_j)

    old_M_IN = tcg.M_IN
    tcg.M_IN = M_IN_
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = 2, N_, T_TASK, DELAY, BATCH_TASK

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses = []
    try:
        for step in range(1, steps + 1):
            x, y = task_fn(rng, T_TASK, BATCH_TASK, M_IN_, delays_or_freqs) \
                if M_IN_ > 1 else _delayed_copy_batch(rng, T_TASK, BATCH_TASK, DELAY)
            h, yhat = tcg.forward(params, x)
            r = yhat - y
            if M_IN_ == 1:
                r[:DELAY] = 0.0
            loss = 0.5 * float(np.mean(r ** 2))
            q = tcg.spatial_q(params, h, r)
            Sa, Sb = tcg.sensitivities(params, h, x)
            lam = tcg.exact_lambda(params, q)
            G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
            g = tcg.flat_grads(G_ex, params)
            if not freeze_pole:
                tie_flat_gradient(g, N_, M_IN_, g_of_j, G)
            else:
                off = offset_a1(N_, M_IN_)
                g[off:off + 2 * N_] = 0.0
            flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
            params = tcg.pack(params, flat)
            losses.append(loss)
            if not np.isfinite(loss):
                break
    finally:
        tcg.M_IN = old_M_IN

    n_param_total = len(flat)
    n_param_pole = 2 * G      # effective free pole parameters (rho_g, theta_g), tied
    return dict(N=N_, G=G, seed=seed, steps_run=len(losses),
               final_loss=float(losses[-1]) if losses else None,
               median_late_loss=float(np.median(losses[-50:])) if len(losses) >= 50
               else (float(np.median(losses)) if losses else None),
               n_param_total=n_param_total, n_param_pole_effective=n_param_pole,
               S_credit=2 * G * N_, S_full=2 * N_ * N_)


def _delayed_copy_batch(rng, T_, BATCH_, delay):
    x = rng.randn(T_, BATCH_)
    y = np.concatenate([np.zeros((delay, BATCH_)), x[:-delay]], axis=0)
    return x, y


def multi_delay_wrapper(rng, T_, BATCH_, r_task, delays):
    return make_multi_delay_task(rng, T_, BATCH_, r_task, delays)


def multi_freq_wrapper(rng, T_, BATCH_, r_spectral, freqs):
    return make_multi_freq_task(rng, T_, BATCH_, r_spectral, freqs)


def g_grid_for(N_):
    cand = [1, 2, 4, 8, 16, N_]
    return sorted(set(g for g in cand if g <= N_))


def main() -> None:
    print("=" * 90)
    print("Phase B16.1: forward-expressivity / G x width scaling (BPTT only)")
    print("=" * 90)
    STEPS_A = 200
    SEEDS_A = [0, 1]

    # ---- Part A: G x width sweep, delayed-copy task ----
    print("\nPart A: G x width, delayed-copy task")
    a_results = []
    for N_ in (16, 32, 64, 128):
        for G in g_grid_for(N_):
            for seed in SEEDS_A:
                r = train_tied(seed, N_, G, 1, None, None, steps=STEPS_A)
                a_results.append(r)
        rows = [r for r in a_results if r["N"] == N_]
        summary = {}
        for G in g_grid_for(N_):
            gl = [r["median_late_loss"] for r in rows if r["G"] == G]
            summary[G] = float(np.median(gl))
        print(f"  N={N_}: median_late_loss by G = {summary}")

    # ---- Part B: temporal-complexity tasks, fixed representative widths ----
    print("\nPart B: temporal-complexity tasks (multi-delay, multi-freq)")
    b_results = []
    N_B = 32
    delay_freqs = [3, 11, 19, 27, 5, 13, 21, 29]
    for r_task in (1, 2, 4):
        for G in g_grid_for(N_B):
            for seed in SEEDS_A:
                delays = [5 + 5 * k for k in range(r_task)]
                r = train_tied(seed, N_B, G, r_task, multi_delay_wrapper, delays,
                              steps=STEPS_A)
                r["task"] = f"multi_delay_r{r_task}"
                b_results.append(r)
    for r_spec in (1, 2, 4):
        for G in g_grid_for(N_B):
            for seed in SEEDS_A:
                freqs = delay_freqs[:r_spec]
                r = train_tied(seed, N_B, G, r_spec, multi_freq_wrapper, freqs,
                              steps=STEPS_A)
                r["task"] = f"multi_freq_r{r_spec}"
                b_results.append(r)
    for task_name in sorted(set(r["task"] for r in b_results)):
        rows = [r for r in b_results if r["task"] == task_name]
        summary = {}
        for G in g_grid_for(N_B):
            gl = [r["median_late_loss"] for r in rows if r["G"] == G]
            if gl:
                summary[G] = float(np.median(gl))
        print(f"  {task_name}: median_late_loss by G = {summary}")

    # ---- Part E: group-assignment sensitivity (contiguous vs random) ----
    print("\nPart E: group assignment sensitivity")
    e_results = []
    for N_ in (32, 64):
        for G in (2, 4, 8):
            for mode in ("contiguous", "random"):
                for seed in SEEDS_A:
                    r = train_tied(seed, N_, G, 1, None, None, steps=STEPS_A,
                                   group_mode=mode)
                    r["group_mode"] = mode
                    e_results.append(r)
    for N_ in (32, 64):
        for G in (2, 4, 8):
            rows = [r for r in e_results if r["N"] == N_ and r["G"] == G]
            cont = np.median([r["median_late_loss"] for r in rows if r["group_mode"] == "contiguous"])
            rand = np.median([r["median_late_loss"] for r in rows if r["group_mode"] == "random"])
            print(f"  N={N_} G={G}: contiguous={cont:.4f} random={rand:.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc = dict(git=git, config=dict(T=T_TASK, BATCH=BATCH_TASK, DELAY=DELAY,
                                    LR=LR, steps=STEPS_A, seeds=SEEDS_A),
              part_a=a_results, part_b=b_results, part_e=e_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b16_1_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else str(o)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
