"""Ablation demanded by the external review: what does the prospective
parameterization/constraint set buy over a generic learned preconditioner?

Route A (co_variational_metric.py) is, operationally, a meta-learned
per-mode diagonal complex preconditioner trained by one-step
meta-gradients — "the ML reviewer's description". The prospective
contribution, if any, must be in the constraints: the complex (phaseful)
per-mode family and the bounded-mode stability law. This experiment
strips them one at a time:

  routeA            complex w, sigmoid-bounded modes (the full version)
  routeA_real       w restricted to REAL values — strips the phase from
                    the metric family (does phase matter?)
  routeA_unbounded  complex w, modes a = exp(rho) e^{i theta} with rho
                    free (|a| may exceed 1) — strips the stability law

Task/budget identical to co_variational_metric.py (delayed copy D=50,
T=128, L=4, N=16, batch 32, Adam 1e-3, clip 1.0, 1500 steps, seeds 0-4).

Run:  python ablation_generic.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm

SEEDS = [0, 1, 2, 3, 4]
STEPS = 1500
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "ablation_generic")


# ---------------------------------------------------------------------------
# Unbounded-mode helpers (a = exp(rho) e^{i theta}, rho free)
# ---------------------------------------------------------------------------

def a_of_ub(rho, theta):
    return [np.exp(r) * np.exp(1j * th) for r, th in zip(rho, theta)]


def init_params_ub(seed):
    rng = np.random.RandomState(seed)
    cplx = lambda *s: (rng.randn(*s) + 1j * rng.randn(*s)) / np.sqrt(2 * s[-1])
    u0 = np.linspace(0.90, 0.995, tcg.N)
    rho = [np.log(u0) for _ in range(tcg.L)]
    theta = [rng.uniform(-np.pi, np.pi, tcg.N) for _ in range(tcg.L)]
    B = [cplx(tcg.N, tcg.M_IN)] + [cplx(tcg.N, tcg.N) * (1 - 0.95)
                                   for _ in range(tcg.L - 1)]
    c = cplx(tcg.N).reshape(-1)
    params = dict(rho=rho, theta=theta, b=B, c=c)
    params["a"] = a_of_ub(rho, theta)
    return params


def flatten_ub(params):
    parts = []
    for l in range(tcg.L):
        parts += [params["rho"][l], params["theta"][l],
                  params["b"][l].real.ravel(), params["b"][l].imag.ravel()]
    parts += [params["c"].real, params["c"].imag]
    return np.concatenate(parts)


def pack_ub(params, vec):
    out = dict(rho=[], theta=[], b=[], c=None)
    i = 0
    for l in range(tcg.L):
        out["rho"].append(vec[i:i + tcg.N].copy())
        out["theta"].append(vec[i + tcg.N:i + 2 * tcg.N].copy())
        i += 2 * tcg.N
        m = params["b"][l].size
        re = vec[i:i + m]
        im = vec[i + m:i + 2 * m]
        out["b"].append((re + 1j * im).reshape(params["b"][l].shape))
        i += 2 * m
    out["c"] = vec[i:i + tcg.N] + 1j * vec[i + tcg.N:i + 2 * tcg.N]
    out["a"] = a_of_ub(out["rho"], out["theta"])
    return out


def flat_grads_ub(G, params):
    """da/d rho = a, da/d theta = i a;  g_rho = Re(Ga a), g_theta = -Im(Ga a)."""
    parts = []
    for l in range(tcg.L):
        Ge = G["a"][l] * params["a"][l]
        parts += [Ge.real, -Ge.imag,
                  G["b"][l].real.ravel(), -G["b"][l].imag.ravel()]
    parts += [G["c"].real, -G["c"].imag]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# The ablated trainer (mirrors cvm.train_route routeA, with knobs)
# ---------------------------------------------------------------------------

def train_ablated(arm, seed):
    tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 128, 50, 1, 32
    bounded = arm != "routeA_unbounded"
    real_only = arm == "routeA_real"
    if bounded:
        params = tcg.init_params(seed)
        flat = tcg.flatten(params)
        pack = tcg.pack
        fgrads = tcg.flat_grads
    else:
        params = init_params_ub(seed)
        flat = flatten_ub(params)
        pack = pack_ub
        fgrads = flat_grads_ub
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = cvm.make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        G_use = cvm.scale_by_w(G, w)
        g = fgrads(G_use, params)
        g = cvm.clip(g)
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = pack(params, flat)

        # meta-gradient for w (same-chain form as routeA; per-mode)
        G_next = cvm.exact_grad(params_next, x, y)
        gN = fgrads(G_next, params_next)
        off = 0
        for l in range(tcg.L):
            A = G["a"][l] * params["a"][l] if not bounded else \
                G["a"][l] * np.exp(1j * params["theta"][l])
            u_mode = tcg.sig(params["rho"][l]) if bounded else 1.0
            sigp = u_mode * (1 - u_mode) if bounded else 1.0
            Gb = G["b"][l]
            M_ = Gb.shape[1]
            gN_rho = gN[off:off + tcg.N]
            gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
            gN_bre = gN[off + 2 * tcg.N:off + 2 * tcg.N + tcg.N * M_].reshape(
                tcg.N, M_)
            gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                        off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
            off += 2 * tcg.N + 2 * tcg.N * M_
            if bounded:
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            else:
                du = (gN_rho * A.real + gN_theta * (-1.0) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * A.imag + gN_theta * (1.0) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
            if real_only:
                w[l] = w[l].real + 0j
        params = params_next
        if step % 400 == 0:
            amax = max(float(np.abs(aa).max()) for aa in params["a"])
            print(f"      {arm} s{seed} step {step}: loss {loss:.4f}  "
                  f"max|a| {amax:.4f}", flush=True)
    losses = np.asarray(losses)
    return dict(arm=arm, seed=seed,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                w_final=[[[float(z.real), float(z.imag)] for z in wl]
                         for wl in w],
                wall_time_sec=time.time() - t0)


def main() -> None:
    print("=" * 78)
    print("Ablation: what do the prospective constraints buy?")
    print("=" * 78)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for arm in ["routeA", "routeA_real", "routeA_unbounded"]:
        finals = []
        for seed in SEEDS:
            out = train_ablated(arm, seed)
            finals.append(out["final_loss"])
            results[f"{arm}/s{seed}"] = out
            with open(os.path.join(RESULTS_DIR, f"{arm}_s{seed}.json"),
                      "w") as f:
                json.dump(out, f, indent=2)
        print(f"  {arm:<17s} finals {['%.4f' % x for x in finals]}  "
              f"median {np.median(finals):.4f}  "
              f"finite {all(results[f'{arm}/s{s}']['finite'] for s in SEEDS)}",
              flush=True)

    med = {arm: float(np.median([results[f"{arm}/s{s}"]["final_loss"]
                                 for s in SEEDS])) for arm in
           ["routeA", "routeA_real", "routeA_unbounded"]}
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, medians=med)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("-" * 78)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print("wrote summary.json")


if __name__ == "__main__":
    main()
