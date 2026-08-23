"""Can the per-mode credit gains be LEARNED (or at least amortized)?

Follow-up to optimal_credit_filter.py: oracle-fitted per-mode complex
gains on the online error signal beat online RTRL by +0.3-0.45 cosine in
the deep/slow regime, and the gains transfer across data. But cosine on a
probe is not task loss, and oracle-fitting used the exact gradient. This
script runs the actual training race.

Task: delayed continuous copy — input x_t ~ N(0,1), target y_t = x_{t-D}
with D = 50, T = 128, per-step regression loss (dense temporal credit).
Model: stacked complex diagonal RNN, L = 4, N = 16 modes, |a| = 0.95.

All arms share the forward model and the exact per-module RTRL
sensitivities S_t; they differ ONLY in the error signal:

  online       e_t = q_t                       (instantaneous spatial)
  prospective  e_t = q_t - a q_{t-1}           (ruled out in the rig;
                                                kept as control)
  oracle       e_t = w*_j q_t, w* least-squares-fit against exact BPTT
               on ONE probe batch at init (amortized calibration)
  calibrated   w re-fit on a probe batch every 200 steps (tracks drift)
  bptt         exact adjoint gradient (the ceiling; not online)

PREDECLARED BAR (before running): the gain arms WIN if their median final
loss (3 seeds) is <= 0.5 x online's AND <= 2 x bptt's. The prospective
control is expected to be <= online. If no gain-arm advantage: per-mode
gains do not help actual training despite the alignment win, and the
credit lane closes with that measured.

Gate: finite-difference check of the batched trainer's exact gradient at
init (rel err < 1e-5 on two random parameter entries).

Run:  python trained_credit_gains.py

OUTCOME (2026-08-24): registered bar NO WIN, both versions.
  Unclipped: online 27.5, prospective 6e29, oracle 8.8e25,
  calibrated 2.8e8, bptt 1e-4. Exploratory global-norm-clip (CLIP=1.0,
  all arms): online 688, prospective 224, oracle 9.1e9, calibrated
  1.9e19, bptt 1e-4. The gain-corrected online rules are training-
  UNSTABLE: the amplified slow-mode credit drives |a| past 1 and the
  run explodes, clip or no clip, while exact BPTT trains to 1e-4 — the
  approximation error has a systematic push toward the stability
  boundary. Credit-lane closing statement: per-mode gains help gradient
  ALIGNMENT (optimal_credit_filter.py, transfers across data) but not
  gradient DESCENT.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T, DELAY = 128, 50
L, N = 4, 16
MAG = 0.95
BATCH = 32
STEPS = 1500
CAL_EVERY = 200
SEEDS = [0, 1, 2]
LR = 1e-3
ARMS = ["online", "prospective", "oracle", "calibrated", "bptt"]
CLIP = 1.0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")


# ---------------------------------------------------------------------------
# Model (batched): h_t^(l) = a_l h_{t-1}^(l) + B_l x_t^(l), x^(l+1) = Re(h^l)
# readout every step: yhat_t = Re(c . h_t^(L))
# ---------------------------------------------------------------------------

def init_params(seed):
    rng = np.random.RandomState(seed)
    cplx = lambda *s: (rng.randn(*s) + 1j * rng.randn(*s)) / np.sqrt(2 * s[-1])
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, N))
    a = [MAG * phases for _ in range(L)]
    # deep-layer B scaled by (1-MAG) so layer outputs stay O(1) at init
    B = [cplx(N, 1)] + [cplx(N, N) * (1 - MAG) for _ in range(L - 1)]
    c = cplx(N).reshape(-1)
    return dict(a=a, b=B, c=c)


def forward(params, x):
    """x: (T, B) real. Returns h per layer (T, B, N) and yhat (T, B)."""
    a, B, c = params["a"], params["b"], params["c"]
    h = []
    inp = x[..., None]                                # (T, B, 1)
    for l in range(L):
        hl = np.zeros((T, x.shape[1], N), np.complex128)
        sp = np.zeros((x.shape[1], N), np.complex128)
        for t in range(T):
            sp = a[l] * sp + (B[l] @ inp[t].T).T      # (B, N)
            hl[t] = sp
        h.append(hl)
        inp = hl.real
    yhat = np.einsum("n,tbn->tb", c, h[-1]).real
    return h, yhat


def spatial_q(params, h, r):
    """Instantaneous spatial error. r: (T, B) residual; only used at the
    steps where the loss lives (dense here)."""
    a, B, c = params["a"], params["b"], params["c"]
    q = [np.zeros_like(hl) for hl in h]
    q[L - 1] = np.conj(c)[None, None, :] * r[..., None]  # conj(c) * r
    for l in range(L - 2, -1, -1):
        q[l] = np.einsum("jm,tbj->tbm", B[l + 1],
                         np.conj(q[l + 1])).real
    return q


def sensitivities(params, h, x):
    """Exact per-module RTRL: S^a (T,B,N), S^B (T,B,N,M_l)."""
    a, B = params["a"], params["b"]
    xs = [x[..., None]] + [h[l].real for l in range(L - 1)]
    Sa, Sb = [], []
    for l in range(L):
        M = xs[l].shape[2]
        sa = np.zeros((T, x.shape[1], N), np.complex128)
        sb = np.zeros((T, x.shape[1], N, M), np.complex128)
        h_prev = np.concatenate([np.zeros_like(h[l][:1]), h[l][:-1]], axis=0)
        sa[0] = h_prev[0]
        sb[0] = xs[l][0][:, None, :]
        for t in range(1, T):
            sa[t] = h_prev[t] + a[l] * sa[t - 1]
            sb[t] = xs[l][t][:, None, :] + a[l][None, :, None] * sb[t - 1]
        Sa.append(sa)
        Sb.append(sb)
    return Sa, Sb


def exact_lambda(params, q):
    """Batched stack-exact adjoint: lam_t^l = U_l(lam_t^{l+1}) + conj(a_l)
    lam_{t+1}^l."""
    a, B = params["a"], params["b"]
    lam = [np.zeros((T, q[0].shape[1], N), np.complex128) for _ in range(L)]
    lam_next = [np.zeros((q[0].shape[1], N), np.complex128)
                for _ in range(L)]
    for t in range(T - 1, -1, -1):
        lam_next[L - 1] = q[L - 1][t] + np.conj(a[L - 1]) * lam_next[L - 1]
        for l in range(L - 2, -1, -1):
            up = np.einsum("jm,bj->bm", B[l + 1],
                           np.conj(lam_next[l + 1])).real
            lam_next[l] = up + np.conj(a[l]) * lam_next[l]
        for l in range(L):
            lam[l][t] = lam_next[l]
    return lam


def assemble(params, h, x, r, err, Sa, Sb, direct=False):
    """Real parameter gradients from error signals err[l].
    direct=False pairs with the accumulated sensitivities (S-slot, the
    online family); direct=True pairs with the direct Jacobian (J-slot —
    the exact-BPTT position). Returns complex G per param group."""
    a, B, c = params["a"], params["b"], params["c"]
    xs = [x[..., None]] + [h[l].real for l in range(L - 1)]
    Ga, Gb = [], []
    for l in range(L):
        ce = np.conj(err[l])
        if direct:
            h_prev = np.concatenate([np.zeros_like(h[l][:1]), h[l][:-1]],
                                    axis=0)
            Ga.append(np.einsum("tbn,tbn->n", ce, h_prev))
            Gb.append(np.einsum("tbn,tbm->nm", ce, xs[l]))
        else:
            Ga.append(np.einsum("tbn,tbn->n", ce, Sa[l]))
            Gb.append(np.einsum("tbn,tbnm->nm", ce, Sb[l]))
    Gc = np.einsum("tb,tbn->n", r, h[L - 1])
    return dict(a=Ga, b=Gb, c=Gc)


def flat_grads(G):
    parts = []
    for l in range(L):
        parts += [G["a"][l].real, -G["a"][l].imag,
                  G["b"][l].real.ravel(), -G["b"][l].imag.ravel()]
    parts += [G["c"].real, -G["c"].imag]
    return np.concatenate(parts)


def pack(params, vec):
    out = dict(a=[], b=[], c=None)
    i = 0
    for l in range(L):
        n = params["a"][l].size
        re = vec[i:i + n]
        im = vec[i + n:i + 2 * n]
        out["a"].append(re + 1j * im)
        i += 2 * n
        m = params["b"][l].size
        k = 2 * m
        re = vec[i:i + m]
        im = vec[i + m:i + 2 * m]
        out["b"].append((re + 1j * im).reshape(params["b"][l].shape))
        i += k
    n = params["c"].size
    out["c"] = vec[i:i + n] + 1j * vec[i + n:i + 2 * n]
    return out


def flatten(params):
    parts = []
    for l in range(L):
        parts += [params["a"][l].real, params["a"][l].imag,
                  params["b"][l].real.ravel(), params["b"][l].imag.ravel()]
    parts += [params["c"].real, params["c"].imag]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Error signals + oracle gain fit
# ---------------------------------------------------------------------------

def err_of(arm, q, a_l, w_l):
    if arm == "online":
        return q
    if arm == "prospective":
        qm = np.concatenate([np.zeros_like(q[:1]), q[:-1]], axis=0)
        return q - a_l[None, None, :] * qm
    return q * w_l[None, None, :]


def fit_gains(params, rng):
    """Oracle per-mode complex gains: least squares of the online gradient
    block against the exact-BPTT block, per layer per mode, on one probe
    batch (32 sequences)."""
    x = rng.randn(T, BATCH)
    y = np.concatenate([np.zeros((DELAY, BATCH)),
                        x[:-DELAY]], axis=0)
    h, yhat = forward(params, x)
    r = yhat - y
    q = spatial_q(params, h, r)
    lam = exact_lambda(params, q)
    Sa, Sb = sensitivities(params, h, x)
    xs = [x[..., None]] + [h[l].real for l in range(L - 1)]
    w = []
    for l in range(L):
        h_prev = np.concatenate([np.zeros((1, BATCH, N)), h[l][:-1]],
                                axis=0)
        wl = np.zeros(N, np.complex128)
        for j in range(N):
            u_a = np.sum(np.conj(q[l][:, :, j]) * Sa[l][:, :, j])
            u_b = np.einsum("tb,tbm->m", np.conj(q[l][:, :, j]),
                            Sb[l][:, :, j, :])
            v_a = np.sum(np.conj(lam[l][:, :, j]) * h_prev[:, :, j])
            v_b = np.einsum("tb,tbm->m", np.conj(lam[l][:, :, j]), xs[l])
            u = np.concatenate([[u_a], u_b])
            v = np.concatenate([[v_a], v_b])
            alpha = np.vdot(v, u) / max(np.vdot(u, u).real, 1e-300)
            wl[j] = np.conj(alpha)
        w.append(wl)
    return w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_batch(params, rng, xy=None):
    if xy is None:
        x = rng.randn(T, BATCH)
        y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
    else:
        x, y = xy
    h, yhat = forward(params, x)
    r = yhat - y
    r[:DELAY] = 0.0                                  # no loss in the burn-in
    return 0.5 * float(np.mean(r ** 2)), h, r, x


def train_arm(arm, seed):
    params = init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    probe_rng = np.random.RandomState(77)
    w = None
    if arm == "oracle":
        w = fit_gains(params, probe_rng)
    if arm == "calibrated":
        w = fit_gains(params, probe_rng)

    flat = flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        loss, h, r, x = loss_batch(params, rng)
        q = spatial_q(params, h, r)
        Sa, Sb = sensitivities(params, h, x)
        if arm == "bptt":
            err = exact_lambda(params, q)
            G = assemble(params, h, x, r, err, Sa, Sb, direct=True)
            g = flat_grads(G)
        else:
            if arm == "calibrated" and step % CAL_EVERY == 0:
                w = fit_gains(params, probe_rng)
            err = [err_of(arm, q[l], params["a"][l],
                          w[l] if w else None) for l in range(L)]
            # S-slot pairing for the online family
            G = assemble(params, h, x, r, err, Sa, Sb)
            g = flat_grads(G)
        nrm = np.linalg.norm(g)
        if nrm > CLIP:
            g = g * (CLIP / nrm)               # uniform global-norm clip
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - LR * (m / (1 - b1 ** step)) / (
            np.sqrt(v / (1 - b2 ** step)) + eps)
        params = pack(params, flat)
        losses.append(loss)
        if step % 200 == 0:
            print(f"      {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    return dict(arm=arm, seed=seed, losses=losses,
                final_loss=float(np.mean(losses[-100:])),
                wall_time_sec=time.time() - t0)


def fd_gate():
    """Finite-difference check of the exact (bptt) gradient, on a SMALL
    config where fd is well conditioned (the full-size loss scale makes
    fd meaningless). The loss is a mean over (T, B) — factor included."""
    global L, N, T, BATCH, DELAY
    keep = (L, N, T, BATCH, DELAY)
    L, N, T, BATCH, DELAY = 2, 3, 12, 2, 4
    try:
        params = init_params(0)
        rng = np.random.RandomState(5)
        loss, h, r, x = loss_batch(params, rng)
        y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
        xy = (x, y)                                # FIXED batch for fd
        q = spatial_q(params, h, r)
        Sa, Sb = sensitivities(params, h, x)
        G = assemble(params, h, x, r, exact_lambda(params, q), Sa, Sb,
                     direct=True)
        g = flat_grads(G) / (T * BATCH)          # mean-convention loss
        flat = flatten(params)
        eps = 1e-6
        for idx in [3, 7, len(flat) // 2]:
            fp = flat.copy(); fp[idx] += eps
            fm = flat.copy(); fm[idx] -= eps
            lp = loss_batch(pack(params, fp), rng, xy=xy)[0]
            lm = loss_batch(pack(params, fm), rng, xy=xy)[0]
            fd = (lp - lm) / (2 * eps)
            rel = abs(fd - g[idx]) / max(abs(g[idx]), 1e-12)
            print(f"  fd gate idx {idx}: fd {fd:.6e} vs exact {g[idx]:.6e}  "
                  f"rel {rel:.2e}  {'PASS' if rel < 1e-4 else 'FAIL'}")
            assert rel < 1e-4
    finally:
        L, N, T, BATCH, DELAY = keep


def main() -> None:
    print("=" * 78)
    print("Trained credit gains — online rules race (delayed copy, D=50)")
    print("=" * 78)
    print("[gate]")
    fd_gate()
    print("[race]")
    results = {}
    for arm in ARMS:
        finals = []
        for seed in SEEDS:
            out = train_arm(arm, seed)
            finals.append(out["final_loss"])
            results[f"{arm}/s{seed}"] = out
        print(f"  {arm:<11s} final loss per seed: "
              f"{['%.4f' % f for f in finals]}  median "
              f"{np.median(finals):.4f}", flush=True)

    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final_loss"]
                                 for s in SEEDS])) for arm in ARMS}
    win = med["oracle"] <= 0.5 * med["online"] and \
        med["oracle"] <= 2 * med["bptt"]
    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"PREDECLARED BAR: oracle <= 0.5x online AND <= 2x bptt  ->  "
          f"{'WIN' if win else 'NO WIN'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(T=T, delay=DELAY, L=L, N=N, mag=MAG, batch=BATCH,
                           steps=STEPS, seeds=SEEDS, lr=LR,
                           cal_every=CAL_EVERY),
               medians=med, win=win)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "trained_credit_gains.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
