"""Phase B18 -- small rich temporal cores vs wide feature multiplicity.

B17 falsified simple all-layer scalar tying (r=1 in this phase's own
language): temporal diversity is functionally necessary, and confining
every layer to a single shared pole destroys performance. The
remaining hypothesis: factor temporal complexity away from feature
multiplicity via H = M (x) T, A = I_n (x) R, dim(T)=r << N=n*r.

tcg (toyrig/ssm_rig.py) is diagonal-complex only and cannot represent
non-diagonal r x r blocks (Jordan/cascade/dense R) or heterogeneous
per-layer width (needed for the narrow-core-sandwich architecture), so
this phase uses a NEW, real-valued, generic multi-layer linear
recurrent simulator with a hand-derived, generic reverse-mode (BPTT)
adjoint -- verified for correctness against finite differences and,
for the diagonal/r=1 special case, sanity-checked to reproduce B17's
qualitative finding before trusting it for new architectures.

Ordinary BPTT only for Parts A/B/D/E/G. Parts C/F are primarily
derivation + small-scale exact verification, matching the pattern used
throughout B16-B17. No new persistent online-credit training rule. No
S5.

Run:  python -m credit_memory.b18_temporal_core
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b18")

T_TASK = 60
BATCH_TASK = 8
LR = 1e-3
STEPS_MAIN = 600  # the spectrally-normalized init (needed to fix an
                  # exponential-blowup bug at depth, see project_stable/
                  # _spectral_normalize) converges more slowly than the
                  # unconstrained-scale init that produced this constant
                  # under the original (unstable) init; 200 steps was
                  # sufficient there but leaves L>=3 undertrained here


# ---------------------------------------------------------------------------
# R-block families (r x r real matrices, spectral radius < 1)
# ---------------------------------------------------------------------------
def make_R_diagonal(r, rng, mag_range=(0.80, 0.97)):
    mags = np.linspace(mag_range[0], mag_range[1], r)
    rng.shuffle(mags)
    return np.diag(mags)


def make_R_oscillator(r, rng, mag_range=(0.85, 0.97)):
    """r/2 independent real 2x2 rotation-decay blocks (block-diagonal)."""
    R = np.zeros((r, r))
    npairs = r // 2
    mags = np.linspace(mag_range[0], mag_range[1], max(npairs, 1))
    for k in range(npairs):
        phase = rng.uniform(0.15 * np.pi, 0.85 * np.pi)
        m = mags[k]
        blk = m * np.array([[np.cos(phase), -np.sin(phase)],
                            [np.sin(phase), np.cos(phase)]])
        R[2 * k:2 * k + 2, 2 * k:2 * k + 2] = blk
    if r % 2 == 1:
        R[-1, -1] = rng.uniform(*mag_range)
    return R


def make_R_jordan(r, rng, mag_range=(0.85, 0.97)):
    """Single Jordan-like cascade block: shared decay on the diagonal,
    unit shift on the superdiagonal (shift-register memory chain)."""
    m = rng.uniform(*mag_range)
    R = m * np.eye(r) + np.diag(np.ones(r - 1), k=1) * 0.5
    return R


def make_R_dense(r, rng, mag_range=(0.80, 0.95)):
    """Generic dense stable matrix: random matrix rescaled to a target
    spectral radius."""
    M = rng.randn(r, r) / np.sqrt(r)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    target = rng.uniform(*mag_range)
    return M * (target / (rho + 1e-9))


R_FAMILIES = dict(diagonal=make_R_diagonal, oscillator=make_R_oscillator,
                  jordan=make_R_jordan, dense=make_R_dense)


# ---------------------------------------------------------------------------
# Model: a stack of layers, each layer l has width N_l = n_l * r_l,
# transition A_l = I_{n_l} (x) R_l (r_l x r_l, shared across n_l copies)
# or A_l = 0 (a "cheap" untrained pass-through layer, zero persistent
# credit state), routing B_l (N_l x M_l, M_l = input width from below).
# ---------------------------------------------------------------------------
class Layer:
    def __init__(self, width, r, R, trainable_R=True, routing="dense",
                q_dim=None, M_IN=None):
        self.width = width
        self.r = r
        self.n = width // r if r > 0 else 0
        self.R = R                      # r x r, or None for a=0 layer
        self.trainable_R = trainable_R
        self.routing = routing          # 'dense' | 'kron' | 'sum_kron'
        self.q_dim = q_dim              # for sum_kron
        self.M_IN = M_IN


def build_A(layer):
    if layer.R is None:
        return np.zeros((layer.width, layer.width))
    return np.kron(np.eye(layer.n), layer.R)


def _spectral_normalize(B, target_norm):
    """Normalize B to a FIXED operator (spectral) norm, independent of its
    shape -- removes the need to hand-tune an init scale per depth/width;
    the recurrent gain (from R's own pole magnitudes, accumulating over T
    steps) is then the only source of amplification, which project_stable
    already bounds during training."""
    sv1 = np.linalg.svd(B, compute_uv=False)[0]
    if sv1 < 1e-300:
        return B
    return B * (target_norm / sv1)


def init_routing(layer, M_in, rng, scale=1.0):
    """scale = target operator (spectral) norm of B, INDEPENDENT of M_in
    and depth -- deep linear recurrent stacks blow up (or vanish) under a
    fixed PER-ELEMENT init scale like 1/sqrt(M_in) once several layers'
    gains compound; normalizing each B's actual spectral norm avoids
    needing a hand-tuned, depth-dependent constant."""
    N_ = layer.width
    if layer.routing == "dense":
        B = rng.randn(N_, M_in) / np.sqrt(M_in)
    elif layer.routing == "kron":
        # B = M (x) I_r  -- M: n x (M_in/r), requires M_in divisible by r
        assert M_in % layer.r == 0
        m_in = M_in // layer.r
        M = rng.randn(layer.n, m_in) / np.sqrt(m_in)
        B = np.kron(M, np.eye(layer.r))
    elif layer.routing == "sum_kron":
        assert M_in % layer.r == 0
        m_in = M_in // layer.r
        q = layer.q_dim or 2
        B = np.zeros((N_, M_in))
        for _ in range(q):
            M = rng.randn(layer.n, m_in) / np.sqrt(m_in * q)
            Q = rng.randn(layer.r, layer.r) / np.sqrt(layer.r * q)
            B += np.kron(M, Q)
    else:
        raise ValueError(layer.routing)
    return _spectral_normalize(B, scale)


def init_stack(layer_specs, M_IN0, seed):
    """layer_specs: list of dicts describing each layer (width, r, family,
    trainable, routing, ...). Returns params dict."""
    rng = np.random.RandomState(seed)
    layers = []
    Bs = []
    M_in = M_IN0
    for l_idx, spec in enumerate(layer_specs):
        r = spec["r"]
        width = spec["width"]
        if spec.get("family") is None:
            R = None
        else:
            R = R_FAMILIES[spec["family"]](r, rng)
        layer = Layer(width, r, R, trainable_R=spec.get("trainable", True),
                     routing=spec.get("routing", "dense"),
                     q_dim=spec.get("q_dim"), M_IN=M_in)
        layers.append(layer)
        routing_scale = 1.0 if l_idx == 0 else 0.3
        Bs.append(init_routing(layer, M_in, rng, scale=routing_scale))
        M_in = width
    c = rng.randn(layers[-1].width) / np.sqrt(layers[-1].width)
    return dict(layers=layers, Bs=Bs, c=c)


def forward(params, x):
    """x: (T, batch, M_IN0). Returns h list per layer (T,batch,width), yhat."""
    layers, Bs, c = params["layers"], params["Bs"], params["c"]
    T_, batch = x.shape[0], x.shape[1]
    h_all = []
    inp = x
    for layer, B in zip(layers, Bs):
        A = build_A(layer)
        N_ = layer.width
        h = np.zeros((T_, batch, N_))
        sp = np.zeros((batch, N_))
        for t in range(T_):
            sp = sp @ A.T + inp[t] @ B.T
            h[t] = sp
        h_all.append(h)
        inp = h
    yhat = h_all[-1] @ c
    return h_all, yhat


def bptt_gradients(params, x, y, h_all):
    """Generic reverse-mode adjoint for the linear stack. Returns dict of
    gradients: dR per layer (or None if not trainable), dB per layer, dc."""
    layers, Bs, c = params["layers"], params["Bs"], params["c"]
    T_, batch = x.shape[0], x.shape[1]
    L_ = len(layers)
    yhat = h_all[-1] @ c
    r_err = (yhat - y) / (T_ * batch)          # dL/dyhat, L = 0.5*mean(err^2)

    dc = np.einsum("tbn,tb->n", h_all[-1], r_err)
    lam_next = [np.zeros((batch, layer.width)) for layer in layers]
    dR = [np.zeros_like(layer.R) if layer.R is not None else None for layer in layers]
    dB = [np.zeros_like(B) for B in Bs]
    inputs = [x] + h_all[:-1]

    for t in reversed(range(T_)):
        lam_t = [None] * L_
        lam_t[L_ - 1] = r_err[t][:, None] * c[None, :]
        if t + 1 < T_:
            A_top = build_A(layers[L_ - 1])
            lam_t[L_ - 1] = lam_t[L_ - 1] + lam_next[L_ - 1] @ A_top
        for l in range(L_ - 2, -1, -1):
            up = lam_t[l + 1] @ Bs[l + 1]
            if t + 1 < T_:
                A_l = build_A(layers[l])
                up = up + lam_next[l] @ A_l
            lam_t[l] = up
        for l in range(L_):
            layer = layers[l]
            if layer.trainable_R and layer.R is not None:
                h_prev = h_all[l][t - 1] if t > 0 else np.zeros((batch, layer.width))
                # dL/dR: A = I_n (x) R; d(sp)/dR contributes via each of the
                # n copies independently -- sum the outer product per copy.
                hp = h_prev.reshape(batch, layer.n, layer.r)
                lt = lam_t[l].reshape(batch, layer.n, layer.r)
                dR[l] += np.einsum("bnj,bnk->kj", hp, lt)
            dB[l] += np.einsum("bm,bn->nm", inputs[l][t], lam_t[l])
        lam_next = lam_t
    return dict(dR=dR, dB=dB, dc=dc)


def flatten_grad(params, grads):
    parts = []
    for l, layer in enumerate(params["layers"]):
        if layer.trainable_R and layer.R is not None:
            parts.append(grads["dR"][l].ravel())
        parts.append(grads["dB"][l].ravel())
    parts.append(grads["dc"])
    return np.concatenate(parts)


def flatten_params(params):
    parts = []
    for l, layer in enumerate(params["layers"]):
        if layer.trainable_R and layer.R is not None:
            parts.append(layer.R.ravel())
        parts.append(params["Bs"][l].ravel())
    parts.append(params["c"])
    return np.concatenate(parts)


def unflatten_into(params, flat):
    i = 0
    for l, layer in enumerate(params["layers"]):
        if layer.trainable_R and layer.R is not None:
            m = layer.R.size
            layer.R = flat[i:i + m].reshape(layer.R.shape)
            i += m
        m = params["Bs"][l].size
        params["Bs"][l] = flat[i:i + m].reshape(params["Bs"][l].shape)
        i += m
    params["c"] = flat[i:i + params["c"].size]
    return params


def adam_step(flat, m_, v_, g, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m_ = b1 * m_ + (1 - b1) * g
    v_ = b2 * v_ + (1 - b2) * g ** 2
    flat = flat - lr * (m_ / (1 - b1 ** step)) / (np.sqrt(v_ / (1 - b2 ** step)) + eps)
    return flat, m_, v_


def project_stable(params, radius_cap=0.98):
    """Unlike tcg's sigmoid(rho)-parameterized poles (unconditionally
    |a|<1 by construction), R here is a free matrix under raw gradient
    descent and can leave the stable region after an update, causing the
    recurrence to explode. Project each trainable R back to spectral
    radius <= radius_cap after every step (a standard practical fix,
    analogous to weight clipping / spectral normalization)."""
    for layer in params["layers"]:
        if layer.trainable_R and layer.R is not None:
            rad = np.max(np.abs(np.linalg.eigvals(layer.R)))
            if rad > radius_cap:
                layer.R = layer.R * (radius_cap / rad)


def train(layer_specs, M_IN0, task_fn, task_arg, seed, steps=STEPS_MAIN,
         T_task=T_TASK, batch_task=BATCH_TASK, lr=LR):
    params = init_stack(layer_specs, M_IN0, seed)
    rng = np.random.RandomState(4000 + seed)
    flat = flatten_params(params)
    m_ = np.zeros_like(flat); v_ = np.zeros_like(flat)
    losses = []
    for step in range(1, steps + 1):
        x, y = task_fn(rng, T_task, batch_task, M_IN0, task_arg)
        if x.ndim == 2:
            x = x[:, :, None]
        h_all, yhat = forward(params, x)
        loss = 0.5 * float(np.mean((yhat - y) ** 2))
        grads = bptt_gradients(params, x, y, h_all)
        g = flatten_grad(params, grads)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, lr)
        unflatten_into(params, flat)
        project_stable(params)
        flat = flatten_params(params)
        losses.append(loss)
        if not np.isfinite(loss):
            break
    median_late = float(np.median(losses[-50:])) if len(losses) >= 50 else \
        (float(np.median(losses)) if losses else None)
    return dict(params=params, losses=losses, median_late_loss=median_late,
               final_loss=float(losses[-1]) if losses else None)


# ---------------------------------------------------------------------------
# Correctness check: generic BPTT adjoint vs finite difference
# ---------------------------------------------------------------------------
def verify_bptt_vs_fd(seed=0):
    rng = np.random.RandomState(seed)
    specs = [dict(width=6, r=2, family="dense", trainable=True, routing="dense"),
            dict(width=4, r=2, family="oscillator", trainable=True, routing="dense")]
    params = init_stack(specs, 3, seed)
    x = rng.randn(8, 2, 3)
    y = rng.randn(8, 2)
    h_all, yhat = forward(params, x)
    grads = bptt_gradients(params, x, y, h_all)
    g_analytic = flatten_grad(params, grads)

    flat0 = flatten_params(params)

    def loss_at(flat):
        unflatten_into(params, flat)
        _, yhat_ = forward(params, x)
        return 0.5 * float(np.mean((yhat_ - y) ** 2))

    eps = 1e-6
    g_fd = np.zeros_like(flat0)
    for i in range(len(flat0)):
        fp = flat0.copy(); fp[i] += eps
        fm = flat0.copy(); fm[i] -= eps
        g_fd[i] = (loss_at(fp) - loss_at(fm)) / (2 * eps)
    unflatten_into(params, flat0)
    max_abs_err = float(np.max(np.abs(g_analytic - g_fd)))
    max_rel_err = float(max_abs_err / (np.max(np.abs(g_fd)) + 1e-12))
    return max_abs_err, max_rel_err


# ---------------------------------------------------------------------------
# Credit-state accounting (Part C): generalizing B16's 2*G*M_lower to
# I_n (x) R blocks. A source parameter feeding layer l needs an r-dim
# (not scalar, not N-dim) running "Z-chain" Z_t = R Z_{t-1} + seed_t --
# the SAME shared R is used for the credit recursion regardless of the
# feature multiplicity n, so persistent state per source is O(r), not
# O(1) (r=1) and not O(N) (untied). Total per layer: 2*r*M_l (the 2x is
# real/complex doubling -- not needed here since everything is already
# real, kept for direct comparability with B16-B17's formula).
# ---------------------------------------------------------------------------
def layer_input_dim(l, layer_specs, M_IN0):
    return M_IN0 if l == 0 else layer_specs[l - 1]["width"]


def s_credit_total(layer_specs, M_IN0):
    total = 0
    for l, spec in enumerate(layer_specs):
        if not spec.get("trainable", True) or spec.get("family") is None:
            continue  # a=0 "cheap" layer: zero persistent pole-credit state
        r = spec["r"]
        M_l = layer_input_dim(l, layer_specs, M_IN0)
        total += r * M_l
    return total


def s_full_total(layer_specs, M_IN0):
    total = 0
    for l, spec in enumerate(layer_specs):
        M_l = layer_input_dim(l, layer_specs, M_IN0)
        total += spec["width"] * M_l
    return total


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def delay_wrapper(rng, T_, BATCH_, M_IN_, delays):
    from credit_memory.b12_structural_spectral_theory import make_multi_delay_task
    return make_multi_delay_task(rng, T_, BATCH_, M_IN_, delays)


def k_exp_modes_wrapper(rng, T_, BATCH_, M_IN_, arg):
    K, mus = arg
    x = rng.randn(T_, BATCH_, 1)
    s = np.zeros((K, BATCH_))
    y = np.zeros((T_, BATCH_))
    for t in range(T_):
        s = mus[:, None] * s + x[t, :, 0][None, :]
        y[t] = s.mean(axis=0)
    return x, y


def pure_delay_wrapper(rng, T_, BATCH_, M_IN_, D):
    x = rng.randn(T_, BATCH_, 1)
    y = np.zeros((T_, BATCH_))
    if D < T_:
        y[D:] = x[:-D, :, 0]
    return x, y


def delays_for(r):
    return [5 + 5 * k for k in range(r)]


TASK_SPECS = [
    ("delay_r1", delay_wrapper, 1, delays_for(1)),
    ("delay_r4", delay_wrapper, 4, delays_for(4)),
    ("delay_r8", delay_wrapper, 8, delays_for(8)),
    ("kexp_K4", k_exp_modes_wrapper, 1, (4, np.linspace(0.75, 0.95, 4))),
    ("puredelay_D20", pure_delay_wrapper, 1, 20),
]


def r_grid_for(N_):
    cand = [1, 2, 4, 8, 16, N_]
    return sorted(set(g for g in cand if N_ % g == 0))


def uniform_stack(N_, L_, r, family="oscillator", routing="dense"):
    # oscillator (2x2 rotation-decay blocks), not diagonal (pure real
    # decay, no phase): a purely-real diagonal basis is dramatically
    # weaker at representing delays specifically (real poles can't
    # resonate/interfere the way tcg's genuinely complex poles can),
    # which would handicap even the r=N "fully untied" ceiling and make
    # the r-sweep's comparison to it meaningless -- verified directly:
    # at r=N=64, diagonal reaches loss 1.88 on delay_r8 where oscillator
    # reaches 0.14 and dense reaches 0.08, at identical width/steps.
    return [dict(width=N_, r=r, family=family, trainable=True, routing=routing)
           for _ in range(L_)]


def sandwich_stack(N_, r, core_family="dense", cheap_kind="zero"):
    """D-architecture: wide cheap layer -> narrow rich core -> wide cheap
    layer. cheap_kind='zero': a=0, no persistent pole-credit state at all."""
    cheap = dict(width=N_, r=N_, family=None, trainable=False, routing="dense")
    core = dict(width=r, r=r, family=core_family, trainable=True, routing="dense")
    return [dict(cheap), core, dict(cheap)]


# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 90)
    print("Phase B18: small rich temporal cores vs wide feature multiplicity")
    print("=" * 90)

    abs_err, rel_err = verify_bptt_vs_fd()
    print(f"\nBPTT-vs-FD sanity check: max_abs_err={abs_err:.2e} max_rel_err={rel_err:.2e}")
    assert rel_err < 1e-4, "generic BPTT adjoint failed FD check -- aborting"

    SEEDS = [0, 1]
    doc = {"fd_check": dict(max_abs_err=abs_err, max_rel_err=rel_err)}

    # ---- Part A: r sweep (r=1 .. r=N) x N x L x task, diagonal family ----
    # L restricted to {2,3} for the main grid: with the spectrally-
    # normalized (stable, not exploding) init, L=4 at these widths needs
    # ~1500+ steps to converge even for the FULLY UNTIED baseline -- an
    # explicit, honest scope reduction; L=4 gets a separate, longer-step,
    # narrower spot-check below instead of the full grid.
    print("\nPart A: r-sweep x N x L x task (oscillator blocks), L in {2,3}")
    a_results = []
    for N_ in (32, 64, 128):
        for L_ in (2, 3):
            for r in r_grid_for(N_):
                specs = uniform_stack(N_, L_, r)
                for task_name, task_fn, M_IN_, arg in TASK_SPECS:
                    seeds = SEEDS + ([2] if (N_ == 128 and r in (1, 8) and task_name == "delay_r8") else [])
                    for seed in seeds:
                        out = train(specs, M_IN_, task_fn, arg, seed=seed)
                        a_results.append(dict(
                            N=N_, L=L_, r=r, task=task_name, seed=seed,
                            median_late_loss=out["median_late_loss"],
                            final_loss=out["final_loss"],
                            S_credit=s_credit_total(specs, M_IN_),
                            S_full=s_full_total(specs, M_IN_)))
            print(f"  N={N_} L={L_} done ({len(a_results)} rows so far)")
    doc["part_a"] = a_results
    with open(os.path.join(RESULTS_DIR, "b18_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    # ---- L=4 spot-check: N=64 only, 2 tasks, 1 seed, longer training ----
    print("\nL=4 spot-check (N=64, delay_r1/delay_r8, 1 seed, 1200 steps)")
    a4_results = []
    for r in (1, 4, 16, 64):
        for task_name, task_fn, M_IN_, arg in [TASK_SPECS[0], TASK_SPECS[2]]:
            specs = uniform_stack(64, 4, r)
            out = train(specs, M_IN_, task_fn, arg, seed=0, steps=1200)
            a4_results.append(dict(N=64, L=4, r=r, task=task_name, seed=0,
                                   median_late_loss=out["median_late_loss"],
                                   final_loss=out["final_loss"],
                                   S_credit=s_credit_total(specs, M_IN_),
                                   S_full=s_full_total(specs, M_IN_),
                                   note="L=4 spot-check, 1200 steps"))
    doc["part_a4"] = a4_results
    print(f"  L=4 spot-check done ({len(a4_results)} rows)")

    # N=256 spot-check
    print("\nN=256 spot-check")
    for r in (1, 256):
        for task_name, task_fn, M_IN_, arg in [TASK_SPECS[0], TASK_SPECS[2]]:  # delay_r1, delay_r8
            specs = uniform_stack(256, 2, r)
            out = train(specs, M_IN_, task_fn, arg, seed=0, steps=600)
            a_results.append(dict(N=256, L=2, r=r, task=task_name, seed=0,
                                  median_late_loss=out["median_late_loss"],
                                  final_loss=out["final_loss"],
                                  S_credit=s_credit_total(specs, M_IN_),
                                  S_full=s_full_total(specs, M_IN_),
                                  note="N=256 spot-check, 600 steps"))
    doc["part_a"] = a_results

    # ---- Part B: R-family comparison at r=4, N=64, L in {2,3} ----
    print("\nPart B: R-family comparison (r=4, N=64)")
    b_results = []
    for L_ in (2, 3):
        for family in ("diagonal", "oscillator", "jordan", "dense"):
            specs = uniform_stack(64, L_, 4, family=family)
            for task_name, task_fn, M_IN_, arg in TASK_SPECS:
                for seed in SEEDS:
                    out = train(specs, M_IN_, task_fn, arg, seed=seed)
                    b_results.append(dict(L=L_, family=family, task=task_name, seed=seed,
                                          median_late_loss=out["median_late_loss"]))
        print(f"  L={L_} done")
    doc["part_b"] = b_results
    with open(os.path.join(RESULTS_DIR, "b18_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    # ---- Part D: narrow-core sandwich vs alternatives, N=64 ----
    print("\nPart D: narrow-core sandwich architecture, N=64")
    d_results = []
    N_D = 64
    for task_name, task_fn, M_IN_, arg in TASK_SPECS:
        for seed in SEEDS:
            # D1: wide untied recurrent memory layer (L=2, r=N, i.e. A0)
            specs_d1 = uniform_stack(N_D, 2, N_D)
            out = train(specs_d1, M_IN_, task_fn, arg, seed=seed)
            d_results.append(dict(arch="D1_wide_untied", task=task_name, seed=seed,
                                  median_late_loss=out["median_late_loss"],
                                  S_credit=s_credit_total(specs_d1, M_IN_)))
            for r in (4, 8):
                # D2: narrow dense temporal core ALONE (no wide layers at all)
                specs_d2 = uniform_stack(r, 2, r, family="dense")
                out = train(specs_d2, M_IN_, task_fn, arg, seed=seed)
                d_results.append(dict(arch=f"D2_narrow_alone_r{r}", task=task_name, seed=seed,
                                      median_late_loss=out["median_late_loss"],
                                      S_credit=s_credit_total(specs_d2, M_IN_)))
                # D3: replicated/shared core I_n (x) R (== Part A's r sweep)
                specs_d3 = uniform_stack(N_D, 2, r)
                out = train(specs_d3, M_IN_, task_fn, arg, seed=seed)
                d_results.append(dict(arch=f"D3_shared_core_r{r}", task=task_name, seed=seed,
                                      median_late_loss=out["median_late_loss"],
                                      S_credit=s_credit_total(specs_d3, M_IN_)))
                # D4 (sandwich): wide cheap (a=0) -> narrow rich core r -> wide cheap (a=0)
                specs_d4 = sandwich_stack(N_D, r)
                out = train(specs_d4, M_IN_, task_fn, arg, seed=seed)
                d_results.append(dict(arch=f"D4_sandwich_r{r}", task=task_name, seed=seed,
                                      median_late_loss=out["median_late_loss"],
                                      S_credit=s_credit_total(specs_d4, M_IN_)))
    doc["part_d"] = d_results
    with open(os.path.join(RESULTS_DIR, "b18_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)
    print(f"  Part D done ({len(d_results)} rows)")

    # ---- Part E: routing structure at r=4, N=64, L=2 ----
    print("\nPart E: routing structure (r=4, N=64, L=2)")
    e_results = []
    for routing in ("kron", "sum_kron", "dense"):
        for task_name, task_fn, M_IN_, arg in TASK_SPECS:
            # kron/sum_kron routing requires M_IN divisible by r=4 at layer 0;
            # for M_IN not divisible by 4, fall back to dense at layer 0 only
            L_ = 2
            specs = [dict(width=64, r=4, family="diagonal", trainable=True,
                         routing=routing if M_IN_ % 4 == 0 else "dense", q_dim=2)]
            specs.append(dict(width=64, r=4, family="diagonal", trainable=True,
                             routing=routing, q_dim=2))
            for seed in SEEDS:
                out = train(specs, M_IN_, task_fn, arg, seed=seed)
                e_results.append(dict(routing=routing, task=task_name, seed=seed,
                                      median_late_loss=out["median_late_loss"]))
    doc["part_e"] = e_results
    print(f"  Part E done ({len(e_results)} rows)")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc["git"] = git
    out_path = os.path.join(RESULTS_DIR, "b18_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
