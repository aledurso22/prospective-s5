"""Wiener-Hopf oracle — the decisive ceiling for causal credit.

The PAC arc established the rank-0 case: the optimal CONSTANT causal
estimator c* = sum conj(a)^k rho(k) matches the learned phase but
leaves 90-99% of credit variance. That tested almost nothing: the
delay is 50, the credit horizon ~64 (tbptt). This script tests the
full classical object: the optimal causal FILTER (Wiener-Hopf), plus
the Nehari/Hankel floor, plus block-Procrustes (per-mode phase vs
subspace rotation), plus deployment of a frozen K=64 filter.

Components per seed {0,1,2}, at TRAINED routeA params (deterministic):

  WH probe   per (layer, mode): causal filter f in C^K mapping q -> lam
             by least squares, K in {1,2,4,8,16,32,64,96}. Report
             residual fraction and CREDIT COSINE (the orientation
             yardstick, not variance).
  HANKEL     per mode: top singular value of the truncated Hankel
             matrix of the anti-causal impulse response h_k = conj(a)^k,
             vs the closed form |a|/(1-|a|^2). The H-infinity distance
             to the best causal approximation.
  PROCRUSTES gradient-level: scalar per-mode vs U(2) pairs vs U(4)
             neighbor blocks (by |a| rank): residual of the optimal
             unitary correction.
  DEPLOY     frozen K=64 Wiener filter (estimated at trained params),
             deployed from scratch, paired protocol.

REGISTERED BARS (fixed before running):
  DEPLOY-MAIN: frozen K=64 closes >= 50% of the online -> routeA gap
  => estimation was the problem; adaptive Wiener is the algorithm.
  < 20% => LTI causal barrier at the training level.
  CEILING: report cosine vs K (no bar; diagnostic).

Run:  python wiener_oracle.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import train_cell, STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
K_GRID = [1, 2, 4, 8, 16, 32, 64, 96]
K_DEPLOY = 64
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "wiener_oracle")
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "factorize_w")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")


def wiener_fit(q, lam, K):
    """Per-mode causal filter f in C^K: min sum_t |lam_t - sum_k f_k q_{t-k}|^2."""
    T_, B, N_ = q.shape
    X = np.stack([np.roll(q, k, axis=0) for k in range(K)], axis=-1)
    X[:K - 1] = 0.0
    resid = np.zeros(N_)
    cos = np.zeros(N_)
    fs = np.zeros((N_, K), complex)
    for j in range(N_):
        Xf = X[K - 1:, :, j].reshape(-1, K)
        yv = lam[K - 1:, :, j].reshape(-1)
        f, *_ = np.linalg.lstsq(Xf, yv, rcond=None)
        pred = Xf @ f
        fs[j] = f
        resid[j] = 1.0 - float(np.sum(np.abs(yv - pred) ** 2)
                               / (np.sum(np.abs(yv) ** 2) + 1e-300))
        cos[j] = float(np.abs(np.vdot(yv, pred))
                       / (np.linalg.norm(yv) * np.linalg.norm(pred) + 1e-300))
    return fs, 1.0 - resid, cos   # resid returned as fraction remaining


def hankel_bound(a, trunc=200):
    """Top Hankel singular value of h_k = conj(a)^k (k>=1) vs |a|/(1-|a|^2)."""
    h = np.array([np.conj(a) ** k for k in range(1, trunc + 1)])
    n = trunc // 2
    H = np.stack([h[i:i + n] for i in range(n)], axis=0)
    s = np.linalg.svd(H, compute_uv=False)[0].real
    cf = abs(a) / (1 - abs(a) ** 2)
    return float(s), float(cf)


def deploy_filter(params0_seed, filters):
    """Train from scratch with the frozen per-mode causal filter on q."""
    params = tcg.init_params(params0_seed)
    rng = np.random.RandomState(1000 + params0_seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        err = []
        for l in range(tcg.L):
            ql = q[l]
            el = np.zeros(ql.shape, dtype=np.complex128)
            f = filters[l]
            K = f.shape[1]
            for k in range(K):
                el[k:] += f[:, k][None, None, :] * ql[:tcg.T - k]
            err.append(el)
        Sa, Sb = tcg.sensitivities(params, h, x)
        G_use = tcg.assemble(params, h, x, r, err, Sa, Sb)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return float(losses[-100:].mean()), bool(np.all(np.isfinite(losses)))


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    out = {}
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain...", flush=True)
        params, w = train_cell(4, 50, seed)
        w_saved = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        det = max(float(np.max(np.abs(w[l] - w_saved[l])))
                  for l in range(tcg.L))
        assert det < 1e-12
        rng = np.random.RandomState(900 + seed)
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        lam = tcg.exact_lambda(params, q)

        # WH probe
        wh = {}
        for l in range(tcg.L):
            per_k = {}
            for K in K_GRID:
                _, resid, cos = wiener_fit(q[l], lam[l], K)
                per_k[K] = dict(resid=float(np.median(resid)),
                                cos=float(np.median(cos)))
            wh[l] = per_k
            print(f"  L{l}: " + "  ".join(
                f"K{K} r{per_k[K]['resid']:.2f}/c{per_k[K]['cos']:.2f}"
                for K in K_GRID), flush=True)

        # Hankel bound
        hb = {}
        for l in range(tcg.L):
            ss, cf = [], []
            for j in range(tcg.N):
                s, c = hankel_bound(params["a"][l][j])
                ss.append(s)
                cf.append(c)
            hb[l] = dict(hankel_med=float(np.median(ss)),
                         closed_med=float(np.median(cf)),
                         hankel_max=float(np.max(ss)),
                         agree=float(np.corrcoef(ss, cf)[0, 1]))
            print(f"  L{l}: Hankel median {hb[l]['hankel_med']:.2f} "
                  f"(closed form {hb[l]['closed_med']:.2f}, "
                  f"max {hb[l]['hankel_max']:.1f}, corr {hb[l]['agree']:.3f})",
                  flush=True)

        # Procrustes at CREDIT level in mode space: for a block of k
        # modes, data points are (t,b) elements as columns z in C^k
        # (q values), target lam; R in U(k) = polar of Y Z^dagger.
        # k=1 is the scalar phase (c* direction).
        per_layer = {}
        for l in range(tcg.L):
            order = np.argsort([abs(aa) for aa in params["a"][l]])
            res = {}
            for k in (1, 2, 4):
                num = 0.0
                den = 0.0
                for g0 in range(0, tcg.N, k):
                    idx = order[g0:g0 + k]
                    z = q[l][:, :, idx].reshape(-1, k).T    # (k, M)
                    yv = lam[l][:, :, idx].reshape(-1, k).T
                    Mc = np.einsum("pm,rm->pr", yv, np.conj(z))
                    U_, S, Vh = np.linalg.svd(Mc)
                    R = U_ @ Vh
                    num += float(np.sum(np.abs(yv - R @ z) ** 2))
                    den += float(np.sum(np.abs(yv) ** 2))
                res[k] = num / (den + 1e-300)
            per_layer[l] = dict(u1=res[1], u2=res[2], u4=res[4])
            print(f"  L{l}: credit-Procrustes residual U1 {res[1]:.3f}  "
                  f"U2 {res[2]:.3f}  U4 {res[4]:.3f}", flush=True)

        # Deployment of frozen K=64 filter
        f64 = [wiener_fit(q[l], lam[l], K_DEPLOY)[0] for l in range(tcg.L)]
        fl, fin = deploy_filter(seed, f64)
        print(f"  deploy K=64: final {fl:.4f} finite {fin}", flush=True)
        out[seed] = dict(wh=wh, hankel=hb, procrustes=per_layer,
                         deploy=fl, finite=fin)

    med_deploy = float(np.median([out[s]["deploy"] for s in SEEDS]))
    gap = ref["online"] - ref["routeA"]
    frac = (ref["online"] - med_deploy) / gap
    win = frac >= 0.5
    print("-" * 70)
    print(f"deploy K=64 median {med_deploy:.4f} (online {ref['online']:.4f}, "
          f"routeA {ref['routeA']:.4f}) -> {frac:.2f} of gap")
    print(f"BAR DEPLOY-MAIN: >= 50% -> "
          f"{'ESTIMATION WAS THE PROBLEM' if win else 'NO WIN'}"
          f"{'' if frac >= 0.2 else '  => LTI causal barrier at training level'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, refs=ref, k_deploy=K_DEPLOY,
               per_seed={str(s): dict(deploy=out[s]["deploy"],
                                      hankel={str(l): out[s]["hankel"][l]
                                              for l in out[s]["hankel"]},
                                      procrustes={str(l): out[s]["procrustes"][l]
                                                  for l in out[s]["procrustes"]})
                         for s in SEEDS},
               wh={str(s): {str(l): {str(K): out[s]["wh"][l][K]
                                     for K in K_GRID}
                            for l in out[s]["wh"]} for s in SEEDS},
               med_deploy=med_deploy, frac=frac, win=bool(win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
