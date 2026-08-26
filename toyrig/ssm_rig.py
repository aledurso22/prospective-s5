"""Trained credit gains v2 — the FAIR race: constrained modes.

v1 outcome (2026-08-24): registered bar NO WIN — but v1 was UNFAIR to the
corrected rules: the modes a were parameterized unconstrained (a_re/a_im
free), so the gain-corrected arms could push |a| past 1, which no real
SSM parameterization allows (S5's bilinear map bounds |a| < 1; PESM used
sigmoid). The explosion was an artifact of an illegal move.

v2: a = sigmoid(rho) * e^{i theta} — magnitude in (0,1) by construction.
Same task, same budget, same bar. Plus one ablation arm that isolates the
explosion channel:

  oracle_B     gains applied to the B-gradients ONLY (the a-gradient is
               the plain online one) — this arm cannot destabilize the
               recurrence even in principle.

Arms: online, prospective, oracle, oracle_B, calibrated, bptt (exact).
Task: delayed continuous copy, y_t = x_{t-D}, D=50, T=128, per-step loss.
Model: L=4 stacked complex diagonal layers, N=16 modes, init |a| in
(0.90, 0.995). 3 seeds. Adam lr 1e-3, 1500 steps, grad-clip 1.0.

PREDECLARED BAR (unchanged from v1): the oracle arm WINS if its median
final loss <= 0.5 x online's AND <= 2 x bptt's, with all runs finite.
If the corrected rules still lose under the legal parametrization, the
credit lane closes with complete coverage.

Gate: fd check of the exact gradient, small config (rel err < 1e-4),
including the rho/theta reparameterization.

Run:  python trained_credit_gains.py
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
MAG = 0.95                       # only used for deep-B init scaling
BATCH = 32
STEPS = 1500
CAL_EVERY = 200
SEEDS = [0, 1, 2]
LR = 1e-3
ARMS = ["online", "prospective", "oracle", "oracle_B", "calibrated", "bptt"]
CLIP = 1.0
M_IN = 1                      # input channels (task-dependent)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")


# ---------------------------------------------------------------------------
# Model (batched): h_t^(l) = a_l h_{t-1}^(l) + B_l x_t^(l), x^(l+1) = Re(h^l)
# readout every step: yhat_t = Re(c . h_t^(L))
# ---------------------------------------------------------------------------

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def a_of(params):
    """Constrained modes: a_j = sigmoid(rho_j) e^{i theta_j} in (0,1)."""
    return [sig(r) * np.exp(1j * th)
            for r, th in zip(params["rho"], params["theta"])]


def init_params(seed):
    rng = np.random.RandomState(seed)
    cplx = lambda *s: (rng.randn(*s) + 1j * rng.randn(*s)) / np.sqrt(2 * s[-1])
    u0 = np.linspace(0.90, 0.995, N)
    rho = [np.log(u0 / (1 - u0)) for _ in range(L)]
    theta = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
    B = [cplx(N, M_IN)] + [cplx(N, N) * (1 - MAG) for _ in range(L - 1)]
    c = cplx(N).reshape(-1)
    params = dict(rho=rho, theta=theta, b=B, c=c)
    params["a"] = a_of(params)
    return params


def forward(params, x):
    """x: (T, B) real. Returns h per layer (T, B, N) and yhat (T, B)."""
    a, B, c = params["a"], params["b"], params["c"]
    h = []
    inp = x[..., None] if x.ndim == 2 else x
    for l in range(L):
        hl = np.zeros((T, x.shape[1], N), np.complex128)
        sp = np.zeros((x.shape[1], N), np.complex128)
        for t in range(T):
            sp = a[l] * sp + (B[l] @ inp[t].T).T
            hl[t] = sp
        h.append(hl)
        inp = hl.real
    yhat = np.einsum("n,tbn->tb", c, h[-1]).real
    return h, yhat


def spatial_q(params, h, r):
    a, B, c = params["a"], params["b"], params["c"]
    q = [np.zeros_like(hl) for hl in h]
    q[L - 1] = np.conj(c)[None, None, :] * r[..., None]
    for l in range(L - 2, -1, -1):
        q[l] = np.einsum("jm,tbj->tbm", B[l + 1],
                         np.conj(q[l + 1])).real
    return q


def sensitivities(params, h, x):
    a, B = params["a"], params["b"]
    xs = [x[..., None] if x.ndim == 2 else x] + [h[l].real for l in range(L - 1)]
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
    """direct=False: S-slot (online family); direct=True: J-slot (exact)."""
    a, B, c = params["a"], params["b"], params["c"]
    xs = [x[..., None] if x.ndim == 2 else x] + [h[l].real for l in range(L - 1)]
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


def flat_grads(G, params):
    """Complex Ga -> (g_rho, g_theta) via a = sigmoid(rho) e^{i theta}:
    d rho = sig'(rho) Re(G e^{i theta}),  d theta = -sig(rho) Im(G e^{i theta})."""
    parts = []
    for l in range(L):
        u = sig(params["rho"][l])
        Ge = G["a"][l] * np.exp(1j * params["theta"][l])
        parts += [u * (1 - u) * Ge.real, -u * Ge.imag,
                  G["b"][l].real.ravel(), -G["b"][l].imag.ravel()]
    parts += [G["c"].real, -G["c"].imag]
    return np.concatenate(parts)


def pack(params, vec):
    out = dict(rho=[], theta=[], b=[], c=None)
    i = 0
    for l in range(L):
        out["rho"].append(vec[i:i + N].copy())
        out["theta"].append(vec[i + N:i + 2 * N].copy())
        i += 2 * N
        m = params["b"][l].size
        re = vec[i:i + m]
        im = vec[i + m:i + 2 * m]
        out["b"].append((re + 1j * im).reshape(params["b"][l].shape))
        i += 2 * m
    out["c"] = vec[i:i + N] + 1j * vec[i + N:i + 2 * N]
    out["a"] = a_of(out)
    return out


def flatten(params):
    parts = []
    for l in range(L):
        parts += [params["rho"][l], params["theta"][l],
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


def fit_gains(params, x, r):
    """Oracle per-mode complex gains: LS of the online (S-slot) block
    against the exact (J-slot) block, per layer per mode, on the probe
    batch (x, r) supplied by the caller."""
    h, yhat = forward(params, x)
    q = spatial_q(params, h, r)
    lam = exact_lambda(params, q)
    Sa, Sb = sensitivities(params, h, x)
    xs = [x[..., None] if x.ndim == 2 else x] + [h[l].real for l in range(L - 1)]
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


def probe_batch(rng):
    """Default probe batch (delayed copy) for gain fitting."""
    x = rng.randn(T, BATCH)
    y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
    h, yhat = forward(init_params(0), x)
    r = yhat - y
    r[:DELAY] = 0.0
    return x, r


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
    r[:DELAY] = 0.0
    return 0.5 * float(np.mean(r ** 2)), h, r, x


def train_arm(arm, seed):
    params = init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    probe_rng = np.random.RandomState(77)
    w = fit_gains(params, *probe_batch(probe_rng)) \
        if arm in ("oracle", "oracle_B", "calibrated") else None

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
        elif arm == "oracle_B":
            G_on = assemble(params, h, x, r, q, Sa, Sb)
            err_w = [q[l] * w[l][None, None, :] for l in range(L)]
            G_w = assemble(params, h, x, r, err_w, Sa, Sb)
            G = dict(a=G_on["a"], b=G_w["b"], c=G_on["c"])
        else:
            if arm == "calibrated" and step % CAL_EVERY == 0:
                w = fit_gains(params, *probe_batch(probe_rng))
            err = [err_of(arm, q[l], params["a"][l],
                          w[l] if w is not None else None)
                   for l in range(L)]
            G = assemble(params, h, x, r, err, Sa, Sb)
        g = flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if nrm > CLIP:
            g = g * (CLIP / nrm)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - LR * (m / (1 - b1 ** step)) / (
            np.sqrt(v / (1 - b2 ** step)) + eps)
        params = pack(params, flat)
        losses.append(loss)
        if step % 200 == 0:
            amax = max(float(np.abs(aa).max()) for aa in params["a"])
            print(f"      {arm} s{seed} step {step}: loss {loss:.4f}  "
                  f"max|a| {amax:.4f}", flush=True)
    return dict(arm=arm, seed=seed, losses=losses,
                final_loss=float(np.mean(losses[-100:])),
                finite=bool(np.all(np.isfinite(losses))),
                wall_time_sec=time.time() - t0)


def fd_gate():
    """fd check of the exact gradient, small config, including the
    rho/theta reparameterization. Loss mean factor included."""
    global L, N, T, BATCH, DELAY
    keep = (L, N, T, BATCH, DELAY)
    L, N, T, BATCH, DELAY = 2, 3, 12, 2, 4
    try:
        params = init_params(0)
        rng = np.random.RandomState(5)
        loss, h, r, x = loss_batch(params, rng)
        y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
        xy = (x, y)
        q = spatial_q(params, h, r)
        Sa, Sb = sensitivities(params, h, x)
        G = assemble(params, h, x, r, exact_lambda(params, q), Sa, Sb,
                     direct=True)
        g = flat_grads(G, params) / (T * BATCH)
        flat = flatten(params)
        eps = 1e-6
        for idx in [1, 3, N + 2, len(flat) // 2]:
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
    print("Trained credit gains v2 — constrained modes (the fair race)")
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
              f"{np.median(finals):.4f}  finite {out['finite']}", flush=True)

    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final_loss"]
                                 for s in SEEDS])) for arm in ARMS}
    finite_all = all(results[f"{arm}/s{s}"]["finite"]
                     for arm in ARMS for s in SEEDS)
    win = (med["oracle"] <= 0.5 * med["online"]
           and med["oracle"] <= 2 * med["bptt"] and finite_all)
    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"PREDECLARED BAR: oracle <= 0.5x online AND <= 2x bptt, all "
          f"finite  ->  {'WIN' if win else 'NO WIN'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(T=T, delay=DELAY, L=L, N=N, batch=BATCH,
                           steps=STEPS, seeds=SEEDS, lr=LR, clip=CLIP,
                           cal_every=CAL_EVERY, version="v2-constrained"),
               medians=med, win=win)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "trained_credit_gains_v2.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
