"""PHASE 2 — gradient ground-truth experiment (handoff §16–§18, §30 items 2–6).

A tiny STACK of complex diagonal recurrent layers,

    h_t^(l) = a_l ⊙ h_{t-1}^(l) + B_l x_t^(l),   x_t^(l+1) = Re(h_t^(l)),
    yhat_t = Re(C h_t^(L)),   loss = sum_t 1/2 ||yhat_t - y*_t||^2,

small enough that exact BPTT (jax.grad, float64) is the cheap ground truth.
The question: how well do cheap/online gradient estimators align with BPTT,
and does the prospective error filter close the gap?

GRADIENT ALGEBRA (verified against jax.grad to ~1e-15 by the ref gate).
Two exact forms exist, related by the regrouping duality

    sum_t conj(u_t) S_t  ==  sum_t conj((Lambda u)_t) J_t     for ANY u_t,

where S_t = sum_{s<=t} a^{t-s} J_s is the accumulated RTRL sensitivity
(S^a_t = h_{t-1} + a S^a_{t-1}, S^B_t = x_t + a S^B_{t-1}), J_t is the
DIRECT Jacobian (h_{t-1} for a, x_t for B), and Lambda is the future filter
(Lambda q)_t = sum_k conj(a)^k q_{t+k}. Hence:

    BPTT (exact)   = (q, S)  ==  (lambda, J),   lambda = Lambda q (per layer,
                     with the stack-exact spatial input U_l, see below)
    online_full    = (q, S)         the online-LRU rule; EXACT iff L = 1
    spatial        = (q, J)         myopic in time and sensitivity
    tbptt1         = (q_t + conj(a) q_{t+1}, J)   Lambda truncated to 1 step
    prospective_J  = (e, J),  e_t = q_t - a q_{t-1}   <- the handoff §9
                     phase-exact slot: e is phase-matched to lambda, with
                     gain error |1 - conj(a) e^{iw}|^2
    prospective_S  = (e, S)         the handoff §11 literal slot; by the
                     duality this is (Lambda e, J) — the temporal filter is
                     applied twice. Both slots are measured; the data
                     arbitrates the memo ambiguity.
    vle_oracle     = per-mode optimal real gain on prospective_J (closed form
                     vs the exact adjoint — the ceiling of the VLE correction)
    pro_cascade    = cascade-Phi: phase-corrected SPATIAL backprop,
                     s_l = Phi_l(U_l s_{l+1}), s_{L-1} = Phi_{L-1} q_{L-1}.
                     Phases compose over the cascade, so s_l is phase-matched
                     to the full missing Lambda-cascade (the actual defect in
                     deep stacks); == prospective_J at L=1.

REGIMES. "broadband": white-noise inputs/targets (maximally broad error
spectrum — the gain distortion dominates and prospective loses). "narrowband":
inputs/targets are sums of 3 sinusoids placed on mode resonances (FFT-grid
quantized), so the error spectrum is concentrated where |1-conj(a)e^{iw}|^2
is nearly constant — the one configuration where the phase theorem could
convert into gradient alignment at L >= 2. (handoff §19/Phase 4.)

THE STACK SUBTLETY (handoff §10). The exact adjoint of layer l is

    lambda_t^(l) = U_l(lambda_t^(l+1)) + conj(a_l) lambda_{t+1}^(l),
    U_l(v) = Re(B_{l+1}^T conj(v)),

while the instantaneous spatial error is only q_t^(l) = U_l(q_t^(l+1)):
the upper layers' future credit is dropped. That Q-vs-q gap is where deep
online rules lose the future, and it grows with depth.

Complex-gradient convention (loss real, h holomorphic in a/B): with
G = sum_t conj(eps_t) (S or J)_t, dL/dRe = Re(G), dL/dIm = -Im(G).

GATES (hard asserts, handoff §17 nulls + §30 item 2):
  ref    manual (lambda, J) BPTT == jax.grad            (rel err < 1e-8)
  null1  L=1: online_full IS exact RTRL (cos > 1 - 1e-8; prospective must
         NOT beat it)
  null2  |a|=0: prospective == online_full              (< 1e-12)
  null3  prospective_S disabled == online_full bit-exact
  null4  shuffled q (phase control): reported, not gated — any phase
         advantage must vanish

Output: results/gradient_alignment.json + stdout tables.

Run:  python gradient_alignment.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Config (toy scale: exact BPTT must stay cheap)
# ---------------------------------------------------------------------------

T, N, D_IN, D_OUT = 256, 8, 4, 4
L_SWEEP = [1, 2, 4, 8]
MAG_SWEEP = [0.0, 0.5, 0.9, 0.99, 0.999]
SEED = 1234

GATE_TOL = 1e-8
NULL_TOL = 1e-12

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# Parameters (shared across sweep cells; only a changes with |a|)
# ---------------------------------------------------------------------------

def make_base_params(seed: int):
    rng = np.random.RandomState(seed)
    cplx = lambda *s: (rng.randn(*s) + 1j * rng.randn(*s)) / np.sqrt(2 * s[-1])
    dims = [D_IN] + [N] * (max(L_SWEEP) - 1)
    B_all = [cplx(N, dims[l]) for l in range(max(L_SWEEP))]
    C = cplx(D_OUT, N)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, N))   # fixed mode phases
    return B_all, C, phases


def cell_params(B_all, C, phases, L: int, mag: float):
    a = [mag * phases for _ in range(L)]                  # same phases each layer
    # keep per-mode stationary state variance O(1) across the |a| sweep:
    # Var(h_j) ~ |B_j,:|^2 / (1 - |a_j|^2), so scale B rows by sqrt(1 - |a|^2)
    scale = np.sqrt(1.0 - mag ** 2)
    B = [B_all[l] * scale for l in range(L)]
    return dict(a=a, B=B, C=C)


# ---------------------------------------------------------------------------
# Forward pass (numpy) + jax mirror for the autodiff reference
# ---------------------------------------------------------------------------

def forward_np(params, x: np.ndarray, L: int):
    a, B, C = params["a"], params["B"], params["C"]
    h = [np.zeros((T, N), np.complex128) for _ in range(L)]
    yhat = np.zeros((T, D_OUT))
    h_prev = [np.zeros(N, np.complex128) for _ in range(L)]
    for t in range(T):
        inp = x[t]
        for l in range(L):
            h_prev[l] = a[l] * h_prev[l] + B[l] @ inp
            h[l][t] = h_prev[l]
            inp = h_prev[l].real
        yhat[t] = (C @ h_prev[L - 1]).real
    return h, yhat


def make_jax_loss(L: int):
    def loss_fn(p, x, ystar):
        a = [p["a_re"][l] + 1j * p["a_im"][l] for l in range(L)]
        B = [p["B_re"][l] + 1j * p["B_im"][l] for l in range(L)]
        C = p["C_re"] + 1j * p["C_im"]

        def step(h_prev, x_t):
            inp = x_t
            h_new = []
            for l in range(L):
                h_l = a[l] * h_prev[l] + B[l] @ inp
                h_new.append(h_l)
                inp = h_l.real
            return h_new, (C @ h_new[L - 1]).real

        h0 = [jnp.zeros(N, jnp.complex128) for _ in range(L)]
        _, y = jax.lax.scan(step, h0, x)
        return 0.5 * jnp.sum((y - ystar) ** 2)

    return jax.jit(jax.value_and_grad(loss_fn))


def to_jax_params(params, L: int):
    out = {"a_re": [], "a_im": [], "B_re": [], "B_im": []}
    for l in range(L):
        out["a_re"].append(jnp.asarray(params["a"][l].real))
        out["a_im"].append(jnp.asarray(params["a"][l].imag))
        out["B_re"].append(jnp.asarray(params["B"][l].real))
        out["B_im"].append(jnp.asarray(params["B"][l].imag))
    out["C_re"] = jnp.asarray(params["C"].real)
    out["C_im"] = jnp.asarray(params["C"].imag)
    return out


def flat_jax_grad(g, L: int) -> np.ndarray:
    parts = []
    for l in range(L):
        parts += [np.asarray(g["a_re"][l]).ravel(), np.asarray(g["a_im"][l]).ravel(),
                  np.asarray(g["B_re"][l]).ravel(), np.asarray(g["B_im"][l]).ravel()]
    parts += [np.asarray(g["C_re"]).ravel(), np.asarray(g["C_im"]).ravel()]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Manual recursions (numpy): instantaneous spatial q, exact adjoint lambda,
# accumulated RTRL sensitivities S, direct Jacobian J
# ---------------------------------------------------------------------------

def spatial_errors(params, h, r, L: int):
    """Instantaneous spatial error q_t^(l) = d(loss_t)/d h_t^(l)."""
    B, C = params["B"], params["C"]
    q = [np.zeros((T, N), np.complex128) for _ in range(L)]
    q[L - 1] = np.einsum("kj,tk->tj", np.conj(C), r)          # C^H r_t
    for l in range(L - 2, -1, -1):
        q[l] = np.einsum("jm,tj->tm", B[l + 1], np.conj(q[l + 1])).real
    return q


def exact_adjoint(params, q, L: int):
    """lambda_t^(l) = U_l(lambda_t^(l+1)) + conj(a_l) lambda_{t+1}^(l)."""
    a, B = params["a"], params["B"]
    lam = [np.zeros((T, N), np.complex128) for _ in range(L)]
    lam_next = [np.zeros(N, np.complex128) for _ in range(L)]
    for t in range(T - 1, -1, -1):
        lam_next[L - 1] = q[L - 1][t] + np.conj(a[L - 1]) * lam_next[L - 1]
        for l in range(L - 2, -1, -1):
            up = np.einsum("jm,j->m", B[l + 1], np.conj(lam_next[l + 1])).real
            lam_next[l] = up + np.conj(a[l]) * lam_next[l]
        for l in range(L):
            lam[l][t] = lam_next[l]
    return lam


def layer_inputs(params, h, x, L: int):
    """x_t^(l): external input for l=0, Re(h_t^(l-1)) otherwise."""
    return [x] + [h[l].real for l in range(L - 1)]


def sensitivities(params, h, x, L: int):
    """Accumulated per-module RTRL: S^a_t = h_{t-1} + a S^a_{t-1},
    S^B_t[j, :] = x_t + a_j S^B_{t-1}[j, :]."""
    a = params["a"]
    xs = layer_inputs(params, h, x, L)
    Sa, Sb = [], []
    for l in range(L):
        M = xs[l].shape[1]
        sa = np.zeros((T, N), np.complex128)
        sb = np.zeros((T, N, M), np.complex128)
        h_prev = np.concatenate([np.zeros((1, N)), h[l][:-1]], axis=0)
        sa[0] = h_prev[0]
        sb[0] = np.broadcast_to(xs[l][0], (N, M)).copy()
        for t in range(1, T):
            sa[t] = h_prev[t] + a[l] * sa[t - 1]
            sb[t] = xs[l][t][None, :] + a[l][:, None] * sb[t - 1]
        Sa.append(sa)
        Sb.append(sb)
    return Sa, Sb


def direct_jacobians(params, h, x, L: int):
    """J^a_t = h_{t-1} (T, N), J^B_t = x_t broadcast (T, N, M)."""
    xs = layer_inputs(params, h, x, L)
    Ja, Jb = [], []
    for l in range(L):
        M = xs[l].shape[1]
        Ja.append(np.concatenate([np.zeros((1, N)), h[l][:-1]], axis=0))
        Jb.append(np.broadcast_to(xs[l][:, None, :], (T, N, M)))
    return Ja, Jb


# ---------------------------------------------------------------------------
# Error signals (per layer) and gradient assembly
# ---------------------------------------------------------------------------

def shift_back(q):    # q_{t-1}, zero at t=0
    out = np.zeros_like(q)
    out[1:] = q[:-1]
    return out


def shift_fwd(q):     # q_{t+1}, zero at t=T-1
    out = np.zeros_like(q)
    out[:-1] = q[1:]
    return out


def err_online(q, a):        return q
def err_tbptt1(q, a):        return q + np.conj(a)[None, :] * shift_fwd(q)

def err_prospective(q, a, enabled=True):
    # enabled=False is null 3: correction off == the ordinary online rule.
    if not enabled:
        return q
    return q - a[None, :] * shift_back(q)


def err_cascade(params, q, L: int):
    """Cascade-Phi: phase-corrected spatial backprop. Surrogate s_l for the
    exact adjoint lambda_l:  s_{L-1} = Phi_{L-1} q_{L-1},
    s_l = Phi_l(U_l s_{l+1})  with U_l(v) = Re(B_{l+1}^T conj(v))."""
    a, B = params["a"], params["B"]
    s = [None] * L
    s[L - 1] = err_prospective(q[L - 1], a[L - 1])
    for l in range(L - 2, -1, -1):
        down = np.einsum("jm,tj->tm", B[l + 1], np.conj(s[l + 1])).real
        s[l] = err_prospective(down, a[l])
    return s


def assemble_grad(params, h, r, x, L, eps, accumulated: bool):
    """Real gradient vector: pair error signals eps[l] with sensitivities.

    accumulated=True  -> pair with S (RTRL slot);
    accumulated=False -> pair with J (adjoint slot).
    G = sum_t conj(eps_t) (S|J)_t;  dRe = Re(G), dIm = -Im(G).
    """
    C = params["C"]
    if accumulated:
        Sa, Sb = sensitivities(params, h, x, L)
    else:
        Sa, Sb = direct_jacobians(params, h, x, L)
    parts = []
    for l in range(L):
        Ga = np.einsum("tj,tj->j", np.conj(eps[l]), Sa[l])
        Gb = np.einsum("tj,tjm->jm", np.conj(eps[l]), Sb[l])
        parts += [Ga.real.ravel(), -Ga.imag.ravel(),
                  Gb.real.ravel(), -Gb.imag.ravel()]
    Gc = np.einsum("tk,tj->kj", r, h[L - 1])
    parts += [Gc.real.ravel(), -Gc.imag.ravel()]
    return np.concatenate(parts)


def vle_oracle(params, h, r, x, L, eps_pro, g_ref):
    """Per-mode OPTIMAL real gain on the given error signals (closed form
    against the exact gradient g_ref — the ceiling of the VLE scalar-gain
    correction, not an online rule). Returns (gradient, gains per layer)."""
    g_pro = assemble_grad(params, h, r, x, L, eps_pro, accumulated=False)
    # Flat layout per layer block: [Re Ga (N), Im Ga (N), Re Gb (N*M),
    # Im Gb (N*M)]. Gather the per-mode index set explicitly.
    eps_vle = []
    gains = []
    idx = 0
    for l in range(L):
        M = params["B"][l].shape[1]
        g_l = np.zeros(N)
        for j in range(N):
            mode_idx = np.concatenate([
                [idx + j, idx + N + j],                       # Re/Im Ga_j
                idx + 2 * N + j * M + np.arange(M),           # Re Gb_j,: 
                idx + 2 * N + N * M + j * M + np.arange(M),   # Im Gb_j,: 
            ])
            u, v = g_pro[mode_idx], g_ref[mode_idx]
            g_l[j] = np.dot(u, v) / max(np.dot(u, u), 1e-300)
        gains.append(g_l)
        eps_vle.append(eps_pro[l] * g_l[None, :])
        idx += 2 * N * (1 + M)
    return assemble_grad(params, h, r, x, L, eps_vle, accumulated=False), gains


# ---------------------------------------------------------------------------
# Phase diagnostic: cross-spectral phase between an error signal and the
# exact adjoint lambda. The handoff's claim is about PHASE: e should be
# phase-aligned with lambda (offset ~ 0) while the instantaneous q lags by
# arg(1 - conj(a) e^{iw}). Cosine conflates phase and gain; this isolates
# phase. Null 4: shuffling q must destroy the alignment.
# ---------------------------------------------------------------------------

def phase_offset(sig, ref):
    """Power-weighted mean cross-spectral phase between sig and ref.

    sig, ref: (T, N) complex. Returns (mean phase offset in rad, mean
    coherence |C(omega)|) aggregated over modes and frequencies, weighted by
    cross-spectral power.
    """
    S = np.fft.fft(sig, axis=0)
    R = np.fft.fft(ref, axis=0)
    C = S * np.conj(R)                                  # (T, N) cross-spectrum
    w = np.abs(C)
    tot = w.sum()
    if tot < 1e-300:
        return 0.0, 0.0
    phase = float(np.abs(np.angle(C)).ravel() @ w.ravel() / tot)
    coh = float(np.abs(C.sum()) / tot)                  # 1 = perfectly aligned
    return phase, coh


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(est: np.ndarray, ref: np.ndarray, L: int) -> dict:
    cos = float(np.dot(est, ref) / (np.linalg.norm(est) * np.linalg.norm(ref)
                                    + 1e-300))
    rel = float(np.linalg.norm(est - ref) / (np.linalg.norm(ref) + 1e-300))
    ratio = float(np.linalg.norm(est) / (np.linalg.norm(ref) + 1e-300))
    # a-parameter block only (per layer block, the first 2N entries are a)
    ia = []
    idx = 0
    for l in range(L):
        M = params_cur["B"][l].shape[1]
        ia.append(np.arange(idx, idx + 2 * N))
        idx += 2 * N * (1 + M)
    ia = np.concatenate(ia)
    cos_a = float(np.dot(est[ia], ref[ia]) /
                  (np.linalg.norm(est[ia]) * np.linalg.norm(ref[ia]) + 1e-300))
    return dict(cos=cos, cos_a=cos_a, rel_err=rel, norm_ratio=ratio)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_cell(params, x, ystar, L: int, mag: float):
    global params_cur
    params_cur = params
    h, yhat = forward_np(params, x, L)
    r = yhat - ystar
    loss = 0.5 * float(np.sum(r ** 2))

    loss_and_grad = make_jax_loss(L)
    _, g_jax = loss_and_grad(to_jax_params(params, L),
                             jnp.asarray(x), jnp.asarray(ystar))
    ref = flat_jax_grad(g_jax, L)

    q = spatial_errors(params, h, r, L)
    lam = exact_adjoint(params, q, L)
    g_man = assemble_grad(params, h, r, x, L, lam, accumulated=False)
    gate = float(np.linalg.norm(g_man - ref) / np.linalg.norm(ref))
    assert gate < GATE_TOL, f"manual BPTT != jax.grad (rel err {gate:.2e})"

    a = params["a"]
    out = {"loss": loss, "gate_ref_rel": gate}
    ests = {}
    eps_pro = [err_prospective(q[l], a[l]) for l in range(L)]
    eps_casc = err_cascade(params, q, L)
    for name, eps_fn, acc in [
        ("online_full", err_online, True),
        ("spatial", err_online, False),
        ("tbptt1", err_tbptt1, False),
    ]:
        t1 = time.perf_counter()
        ests[name] = assemble_grad(params, h, r, x, L,
                                   [eps_fn(q[l], a[l]) for l in range(L)],
                                   accumulated=acc)
        out.setdefault("times", {})[name] = time.perf_counter() - t1
    t1 = time.perf_counter()
    ests["prospective_J"] = assemble_grad(params, h, r, x, L, eps_pro,
                                          accumulated=False)
    ests["prospective_S"] = assemble_grad(params, h, r, x, L, eps_pro,
                                          accumulated=True)
    ests["pro_cascade_J"] = assemble_grad(params, h, r, x, L, eps_casc,
                                          accumulated=False)
    out["times"]["prospective"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    g_vle, gains = vle_oracle(params, h, r, x, L, eps_pro, g_man)
    ests["vle_oracle"] = g_vle
    g_vlec, gains_c = vle_oracle(params, h, r, x, L, eps_casc, g_man)
    ests["vle_cascade"] = g_vlec
    out["times"]["vle_oracle"] = time.perf_counter() - t1
    out["vle_gain_mean_per_layer"] = [float(np.mean(g)) for g in gains]
    out["vle_cascade_gain_mean_per_layer"] = [float(np.mean(g))
                                              for g in gains_c]

    # null 3: prospective correction disabled (S slot) == online_full exactly
    g_off = assemble_grad(params, h, r, x, L,
                          [err_prospective(q[l], a[l], enabled=False)
                           for l in range(L)], accumulated=True)
    out["null3_maxdiff"] = float(np.max(np.abs(g_off - ests["online_full"])))
    assert out["null3_maxdiff"] < NULL_TOL

    # null 4: shuffled-q prospective (phase control), one global time perm
    rng = np.random.RandomState(7)
    perm = rng.permutation(T)
    ests["pro_shuffled"] = assemble_grad(
        params, h, r, x, L,
        [err_prospective(q[l][perm], a[l]) for l in range(L)],
        accumulated=False)

    # phase diagnostic (the handoff's phase claim, isolated from gain):
    # cross-spectral phase offset of each error signal vs the exact adjoint
    eps_shuf = [err_prospective(q[l][perm], a[l]) for l in range(L)]
    out["phase"] = dict(
        q=[phase_offset(q[l], lam[l]) for l in range(L)],
        pro=[phase_offset(eps_pro[l], lam[l]) for l in range(L)],
        casc=[phase_offset(eps_casc[l], lam[l]) for l in range(L)],
        shuf=[phase_offset(eps_shuf[l], lam[l]) for l in range(L)])

    for name, g in ests.items():
        out[name] = metrics(g, ref, L)

    # null 1: one layer -> online_full is exact RTRL
    if L == 1:
        assert out["online_full"]["cos"] > 1 - GATE_TOL, out["online_full"]
    # null 2: |a|=0 -> prospective == online_full
    if mag == 0.0:
        d = float(np.max(np.abs(ests["prospective_S"] - ests["online_full"])))
        assert d < NULL_TOL, d
    return out


def main() -> None:
    print("=" * 78)
    print("Phase 2 — gradient alignment vs exact BPTT (handoff §16–§18)")
    print("=" * 78)
    rng = np.random.RandomState(SEED)
    x = rng.randn(T, D_IN)
    ystar = rng.randn(T, D_OUT)
    B_all, C, phases = make_base_params(SEED)

    # narrowband regime: inputs/targets concentrated on 3 temporal
    # frequencies, FFT-quantized to mode resonances (handoff §19/Phase 4):
    # |1-conj(a)e^{iw}|^2 is then nearly constant over the error spectrum,
    # so the phase theorem — not the gain distortion — decides alignment.
    ks = sorted(set(
        int(round(th * T / (2 * np.pi))) % T
        for th in np.angle(phases)[[0, 2, 5]]))
    tt = np.arange(T)

    def narrowband(D):
        sig = np.zeros((T, D))
        for d in range(D):
            for k in ks:
                sig[:, d] += np.sin(2 * np.pi * k * tt / T
                                    + rng.uniform(0, 2 * np.pi))
        return sig / np.sqrt(len(ks))

    x_nb, ystar_nb = narrowband(D_IN), narrowband(D_OUT)
    print(f"narrowband freq bins k = {ks} "
          f"(omega = {['%.3f' % (2 * np.pi * k / T) for k in ks]})")

    cells = []
    for regime, x_r, y_r in [("broadband", x, ystar),
                             ("narrowband", x_nb, ystar_nb)]:
        for L in L_SWEEP:
            for mag in MAG_SWEEP:
                params = cell_params(B_all, C, phases, L, mag)
                out = run_cell(params, x_r, y_r, L, mag)
                cells.append(dict(regime=regime, L=L, mag=mag, **out))
                print(f"\n[{regime}] L={L}  |a|={mag}  loss={out['loss']:.4f}"
                      f"  (ref gate rel err {out['gate_ref_rel']:.2e})")
                print(f"  {'estimator':<14s} {'cos_full':>9s} {'cos_a':>9s} "
                      f"{'rel_err':>9s} {'norm_ratio':>10s}")
                for name in ["online_full", "spatial", "tbptt1",
                             "prospective_J", "pro_cascade_J", "prospective_S",
                             "vle_oracle", "vle_cascade", "pro_shuffled"]:
                    m = out[name]
                    print(f"  {name:<14s} {m['cos']:9.4f} {m['cos_a']:9.4f} "
                          f"{m['rel_err']:9.4f} {m['norm_ratio']:10.4f}")
                ph = out["phase"]
                summ = lambda k: (float(np.mean([p[0] for p in ph[k]])),
                                  float(np.mean([p[1] for p in ph[k]])))
                pq, pp, pc, ps = (summ("q"), summ("pro"), summ("casc"),
                                  summ("shuf"))
                print(f"  phase |offset| vs lambda (coherence): "
                      f"q {pq[0]:.3f} ({pq[1]:.2f}) | pro {pp[0]:.3f} "
                      f"({pp[1]:.2f}) | casc {pc[0]:.3f} ({pc[1]:.2f}) | "
                      f"shuf {ps[0]:.3f} ({ps[1]:.2f})")
                print(f"  vle gain mean/layer: "
                      f"{['%.3f' % g for g in out['vle_gain_mean_per_layer']]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch, seed=SEED,
               config=dict(T=T, N=N, D_IN=D_IN, D_OUT=D_OUT,
                           L_sweep=L_SWEEP, mag_sweep=MAG_SWEEP,
                           regimes=["broadband", "narrowband"],
                           narrowband_bins=ks,
                           dtype="float64/complex128"),
               cells=cells)
    path = os.path.join(RESULTS_DIR, "gradient_alignment.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print("\n" + "=" * 78)
    print(f"ALL GATES PASSED — wrote {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
