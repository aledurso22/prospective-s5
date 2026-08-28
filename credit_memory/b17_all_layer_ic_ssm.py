"""Phase B17 -- all-layer invariant-credit architecture test. tcg's
forward/spatial_q/sensitivities/exact_lambda/assemble/flat_grads/pack/
flatten are already generic in depth L (verified by inspection: every
one loops `for l in range(L)` off per-layer rho/theta/b arrays of equal
width N). This module generalizes B16.1/B16.2's tying machinery
(previously hardcoded to "the last of 2 layers") to tie an arbitrary
subset of layers, each with its own G_l, at arbitrary depth.

Ordinary BPTT only. No new persistent online-credit training rule
(Part F is gated on Parts A-E finding a useful regime). No S5.

Run:  python -m credit_memory.b17_all_layer_ic_ssm
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b12_structural_spectral_theory import make_multi_delay_task
from credit_memory.b13_common_temporal_support import make_multi_freq_task

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b17")

T_TASK = 60
BATCH_TASK = 8
LR = 1e-3
STEPS_MAIN = 200


# ---------------------------------------------------------------------------
# Generic-depth params/offsets/tying
# ---------------------------------------------------------------------------
def layer_input_dim(l, N_, M_IN_):
    return M_IN_ if l == 0 else N_


def layer_size(l, N_, M_IN_):
    M_l = layer_input_dim(l, N_, M_IN_)
    return 2 * N_ + 2 * N_ * M_l  # rho + theta + b.real + b.imag


def layer_offset(l, N_, M_IN_, L_):
    return sum(layer_size(k, N_, M_IN_) for k in range(l))


def init_params_L(seed, N_, L_, M_IN_):
    old = (tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY, tcg.M_IN)
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY, tcg.M_IN = L_, N_, T_TASK, BATCH_TASK, 0, M_IN_
    try:
        return tcg.init_params(seed)
    finally:
        tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY, tcg.M_IN = old


def make_group_map(N_, G, rng, mode="contiguous"):
    if G >= N_:
        return np.arange(N_)  # fully untied: each channel its own group
    if mode == "contiguous":
        return np.array([j % G for j in range(N_)])
    elif mode == "random":
        return rng.randint(0, G, size=N_)
    raise ValueError(mode)


def init_all_tied_params(seed, N_, L_, M_IN_, Gs, group_maps, mag_range=(0.90, 0.995)):
    """Gs: list of length L_, G_l per layer (G_l == N_ means untied)."""
    rng = np.random.RandomState(seed)
    params = init_params_L(seed, N_, L_, M_IN_)
    for l in range(L_):
        G_l = Gs[l]
        if G_l >= N_:
            continue
        g_of_j = group_maps[l]
        mags = np.linspace(mag_range[0], mag_range[1], G_l)
        phases = rng.uniform(-np.pi, np.pi, G_l)
        rho_g = -np.log(1.0 / mags - 1.0)
        params["rho"][l] = rho_g[g_of_j]
        params["theta"][l] = phases[g_of_j]
    params["a"] = tcg.a_of(params)
    return params


def tie_flat_gradient_all(g, N_, M_IN_, L_, Gs, group_maps):
    for l in range(L_):
        G_l = Gs[l]
        if G_l >= N_:
            continue
        off = layer_offset(l, N_, M_IN_, L_)
        g_of_j = group_maps[l]
        for half in (0, 1):  # rho block, theta block
            seg = g[off + half * N_: off + half * N_ + N_]
            agg = np.zeros(G_l)
            for j in range(N_):
                agg[g_of_j[j]] += seg[j]
            for j in range(N_):
                seg[j] = agg[g_of_j[j]]
    return g


def n_param_pole_effective(N_, L_, Gs):
    return sum(2 * min(G_l, N_) for G_l in Gs)


def s_credit_total(N_, L_, Gs, M_IN_):
    """Total exact persistent forward-credit state across all layers:
    layer l's OWN pole/routing gradient needs 2*G_l*M_l (B16's formula,
    M_l = that layer's own input width). This is a LOCAL, per-layer
    quantity (Sa[l]/Sb[l] in tcg.sensitivities are already exactly
    this, computed causally from layer l's own -- already-forward-
    computed -- input); it does not need to "chain" across layers."""
    return sum(2 * min(Gs[l], N_) * layer_input_dim(l, N_, M_IN_) for l in range(L_))


def s_full_total(N_, L_, M_IN_):
    return sum(2 * N_ * layer_input_dim(l, N_, M_IN_) for l in range(L_))


# ---------------------------------------------------------------------------
# Training loop (generalized train_tied2 from b16_2, arbitrary depth)
# ---------------------------------------------------------------------------
def train_all_tied(seed, N_, L_, Gs, M_IN_, task_fn, task_arg, group_mode="contiguous",
                   steps=STEPS_MAIN, keep_params=False):
    rng = np.random.RandomState(4000 + seed)
    group_maps = [make_group_map(N_, Gs[l], rng, mode=group_mode) for l in range(L_)]
    params = init_all_tied_params(seed, N_, L_, M_IN_, Gs, group_maps)

    old = (tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN)
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN = L_, N_, T_TASK, BATCH_TASK, M_IN_

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses = []
    try:
        for step in range(1, steps + 1):
            x, y = task_fn(rng, T_TASK, BATCH_TASK, M_IN_, task_arg)
            h, yhat = tcg.forward(params, x)
            r = yhat - y
            loss = 0.5 * float(np.mean(r ** 2))
            q = tcg.spatial_q(params, h, r)
            Sa, Sb = tcg.sensitivities(params, h, x)
            lam = tcg.exact_lambda(params, q)
            G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
            g = tcg.flat_grads(G_ex, params)
            tie_flat_gradient_all(g, N_, M_IN_, L_, Gs, group_maps)
            flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
            params = tcg.pack(params, flat)
            losses.append(loss)
            if not np.isfinite(loss):
                break
    finally:
        tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN = old

    out = dict(N=N_, L=L_, Gs=list(Gs), seed=seed, steps_run=len(losses),
              final_loss=float(losses[-1]) if losses else None,
              median_late_loss=float(np.median(losses[-50:])) if len(losses) >= 50
              else (float(np.median(losses)) if losses else None),
              n_param_total=len(flat), n_param_pole_effective=n_param_pole_effective(N_, L_, Gs),
              S_credit=s_credit_total(N_, L_, Gs, M_IN_), S_full=s_full_total(N_, L_, M_IN_))
    if keep_params:
        out["_params"] = params
        out["_M_IN"] = M_IN_
    return out


# ---------------------------------------------------------------------------
# Tasks (delay/freq/kexp reused from b12/b13/b16_2; hierarchical is new)
# ---------------------------------------------------------------------------
def delay_wrapper(rng, T_, BATCH_, M_IN_, delays):
    return make_multi_delay_task(rng, T_, BATCH_, M_IN_, delays)


def freq_wrapper(rng, T_, BATCH_, M_IN_, freqs):
    return make_multi_freq_task(rng, T_, BATCH_, M_IN_, freqs)


def k_exp_modes_wrapper(rng, T_, BATCH_, M_IN_, arg):
    K, mus = arg
    x = rng.randn(T_, BATCH_, 1)
    s = np.zeros((K, BATCH_))
    y = np.zeros((T_, BATCH_))
    for t in range(T_):
        s = mus[:, None] * s + x[t, :, 0][None, :]
        y[t] = s.mean(axis=0)
    return x, y


def hierarchical_wrapper(rng, T_, BATCH_, M_IN_, arg):
    """Two-timescale task requiring both a fast (short-delay) pathway and
    a slow (moving-average) pathway -- a genuinely multi-layer-useful
    target, not reducible to a single scalar recurrence class."""
    short_d, ma_w, long_d = arg
    x = rng.randn(T_, BATCH_, 1)
    xs = x[:, :, 0]
    csum = np.cumsum(xs, axis=0)
    ma = np.zeros_like(xs)
    for t in range(T_):
        lo = max(0, t - ma_w + 1)
        denom = t - lo + 1
        ma[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / denom
    y = np.zeros_like(xs)
    if short_d < T_:
        y[short_d:] += xs[:-short_d]
    if long_d < T_:
        y[long_d:] += ma[:-long_d]
    return x, y


def delays_for(r):
    return [5 + 5 * k for k in range(r)]


DELAY_FREQS_8 = [3, 11, 19, 27, 5, 13, 21, 29]


def freqs_for(r):
    return DELAY_FREQS_8[:r]


TASK_SPECS = [
    ("delay_r1", delay_wrapper, 1, delays_for(1)),
    ("delay_r4", delay_wrapper, 4, delays_for(4)),
    ("delay_r8", delay_wrapper, 8, delays_for(8)),
    ("freq_r1", freq_wrapper, 1, freqs_for(1)),
    ("kexp_K4", k_exp_modes_wrapper, 1, (4, np.linspace(0.75, 0.95, 4))),
    ("hierarchical", hierarchical_wrapper, 1, (3, 15, 30)),
]


# ---------------------------------------------------------------------------
# Part A: architectures
# ---------------------------------------------------------------------------
def architecture_Gs(arch, L_, N_):
    if arch == "A0_full":
        return [N_] * L_
    if arch == "A1_upper_only":
        return [N_] * (L_ - 1) + [1]
    if arch == "A2_all_tied":
        return [1] * L_
    if arch == "A3_lower_tied":
        return [1] * (L_ - 1) + [N_]
    raise ValueError(arch)


ARCHS = ["A0_full", "A1_upper_only", "A2_all_tied", "A3_lower_tied"]


# ---------------------------------------------------------------------------
# Part B: post-hoc diagnostics on already-trained models
# ---------------------------------------------------------------------------
def ablate_layer_loss(params, M_IN_, L_, N_, l_ablate, task_fn, task_arg, rng, n_batches=6):
    old = (tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN)
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN = L_, N_, T_TASK, BATCH_TASK, M_IN_
    try:
        params_ablated = dict(params)
        params_ablated["a"] = list(params["a"])
        params_ablated["a"][l_ablate] = np.zeros_like(params["a"][l_ablate])
        losses = []
        for _ in range(n_batches):
            x, y = task_fn(rng, T_TASK, BATCH_TASK, M_IN_, task_arg)
            _, yhat = tcg.forward(params_ablated, x)
            losses.append(0.5 * float(np.mean((yhat - y) ** 2)))
        return float(np.median(losses))
    finally:
        tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.M_IN = old


def layer_structural_rank(a_l, B_l, T_=60, tol_ratio=1e-9):
    N_ = len(a_l)
    cols = []
    state = np.zeros_like(B_l, dtype=complex)
    K = min(N_, T_)
    for _ in range(K):
        state = a_l[:, None] * state + B_l
        cols.append(state.copy())
    K_mat = np.concatenate(cols, axis=1)
    sv = np.linalg.svd(K_mat, compute_uv=False)
    if len(sv) == 0 or sv[0] < 1e-300:
        return 0
    return int(np.sum(sv > tol_ratio * sv[0]))


# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 90)
    print("Phase B17: all-layer invariant-credit architecture test")
    print("=" * 90)
    SEEDS = [0, 1]
    doc = {}

    # ---- Part A: architectures x depth x width x task ----
    print("\nPart A: A0-A3 x L x N x task")
    a_results = []
    for L_ in (2, 3, 4):
        for N_ in (32, 64):
            for arch in ARCHS:
                Gs = architecture_Gs(arch, L_, N_)
                for task_name, task_fn, M_IN_, arg in TASK_SPECS:
                    for seed in SEEDS:
                        r = train_all_tied(seed, N_, L_, Gs, M_IN_, task_fn, arg)
                        r["arch"] = arch
                        r["task"] = task_name
                        a_results.append(r)
            print(f"  L={L_} N={N_} done ({len(a_results)} rows so far)")
    doc["part_a"] = a_results
    with open(os.path.join(RESULTS_DIR, "b17_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    # ---- Part E (folded in): G_l in {2,4,8} for all-tied, reduced grid ----
    print("\nPart E: all-tied G_l in {2,4,8}, reduced grid (L in {2,3}, N=64)")
    e_results = []
    for L_ in (2, 3):
        for G_l in (2, 4, 8):
            Gs = [G_l] * L_
            for task_name, task_fn, M_IN_, arg in [TASK_SPECS[2], TASK_SPECS[5]]:  # delay_r8, hierarchical
                for seed in SEEDS:
                    r = train_all_tied(seed, 64, L_, Gs, M_IN_, task_fn, arg)
                    r["arch"] = f"A2_all_tied_G{G_l}"
                    r["task"] = task_name
                    e_results.append(r)
    # routing structure: block-preserving vs dense B at G=4 (E1 vs E2)
    # (uses group_mode as a stand-in structural axis: contiguous groups
    # naturally align with a block-diagonal-friendly channel ordering
    # while random groups force generic dense mixing across the same B)
    for group_mode in ("contiguous", "random"):
        for L_ in (2, 3):
            r = train_all_tied(0, 64, L_, [4] * L_, 8, delay_wrapper, delays_for(8),
                              group_mode=group_mode)
            r["arch"] = f"A2_all_tied_G4_{group_mode}"
            r["task"] = "delay_r8"
            e_results.append(r)
    doc["part_e"] = e_results
    print(f"  Part E done ({len(e_results)} rows)")

    # ---- Part B: post-hoc ablation + structural rank on a subset of Part A models ----
    print("\nPart B: layer ablation + structural rank (L=3, N=64 subset, kept params)")
    b_kept = []
    L_B, N_B = 3, 64
    for arch in ARCHS:
        Gs = architecture_Gs(arch, L_B, N_B)
        for task_name, task_fn, M_IN_, arg in [TASK_SPECS[2], TASK_SPECS[5]]:  # delay_r8, hierarchical
            r = train_all_tied(0, N_B, L_B, Gs, M_IN_, task_fn, arg, keep_params=True)
            b_kept.append((arch, task_name, task_fn, M_IN_, arg, r))

    b_results = []
    rng_ablate = np.random.RandomState(777)
    for arch, task_name, task_fn, M_IN_, arg, r in b_kept:
        params = r["_params"]
        base_loss = r["median_late_loss"]
        row = dict(arch=arch, task=task_name, L=L_B, N=N_B, base_loss=base_loss)
        for l_ablate in range(L_B):
            row[f"ablate_L{l_ablate}_loss"] = ablate_layer_loss(
                params, M_IN_, L_B, N_B, l_ablate, task_fn, arg, rng_ablate)
            row[f"structural_rank_L{l_ablate}"] = layer_structural_rank(
                params["a"][l_ablate], params["b"][l_ablate])
        b_results.append(row)
        print(f"  {arch:16s} {task_name:14s} base={base_loss:.4f} "
             f"ablate={[round(row[f'ablate_L{l}_loss'],4) for l in range(L_B)]} "
             f"rank={[row[f'structural_rank_L{l}'] for l in range(L_B)]}")
    doc["part_b"] = b_results
    with open(os.path.join(RESULTS_DIR, "b17_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc["git"] = git
    doc["config"] = dict(T=T_TASK, BATCH=BATCH_TASK, LR=LR, steps_main=STEPS_MAIN, seeds=SEEDS)
    out_path = os.path.join(RESULTS_DIR, "b17_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else
            None if isinstance(o, dict) and False else str(o)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
