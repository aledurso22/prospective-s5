"""Recheck: is the learned metric secretly the prospective/natural mass?

The claim under test: "the prospective action does not derive the winning
algorithm" (Gap 2 of the double-check). The gap's load-bearing assumption
was that the learned per-mode metric w is NOT the curvature mass
(I + tau H)^{-1}. That was asserted, not measured. This script measures
it directly: at trained params, per (layer, mode), compute the loss's
Gauss-Newton curvature per coordinate (rho, theta, and the B row) by
finite differences on a fixed batch, and correlate with the learned |w|.

If |w| ~ 1/curvature, the learned metric IS approximately the prospective
mass in disguise (natural-gradient lineage; the meta-gradient is then
just its estimator) and "the action derives the algorithm" revives via
the curvature-mass route. If |w| is orthogonal to curvature, it is a
free MAML-fitted geometry and B+ stands.

Run:  python recheck_curvature.py
"""
from __future__ import annotations

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.train_cell import train_cell


def batch_loss(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    return 0.5 * float(np.mean(r ** 2))


def curvature(params, x, y, eps=1e-3):
    """Second derivative of the batch loss w.r.t. rho[l][j], theta[l][j],
    and the mean over the B-row entries, per (layer, mode)."""
    out = {}
    for l in range(tcg.L):
        for j in range(tcg.N):
            curvs = []
            for kind in ("rho", "theta"):
                for sign in (1, -1):
                    p2 = {k: (v.copy() if not isinstance(v, list)
                              else [a.copy() for a in v])
                          for k, v in params.items()}
                    p2["a"] = params["a"]
                    p2[kind][l][j] += sign * eps
                    p2["a"] = tcg.a_of(p2)
                    lp = batch_loss(p2, x, y)
                    p2[kind][l][j] -= 2 * sign * eps
                    p2["a"] = tcg.a_of(p2)
                    lm = batch_loss(p2, x, y)
                    curvs.append((lp + lm) / eps ** 2)
            base = 2 * batch_loss(params, x, y)
            curv = curvs[0] - base
            # B row representative curvature: perturb b[l][j,0] real part
            p2 = {k: (v.copy() if not isinstance(v, list)
                      else [a.copy() for a in v])
                  for k, v in params.items()}
            p2["a"] = params["a"]
            p2["b"][l][j, 0] += eps
            lp = batch_loss(p2, x, y)
            p2["b"][l][j, 0] -= 2 * eps
            lm = batch_loss(p2, x, y)
            curv_b = (lp + lm - 2 * batch_loss(params, x, y)) / eps ** 2
            out[(l, j)] = (abs(curv), abs(curv_b))
    return out


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 4, 16, 128, 50, 1, 32
    print("training routeA (L=4, D=50, seed 0)...", flush=True)
    params, w = train_cell(4, 50, 0)
    rng = np.random.RandomState(999)
    x, y = rng.randn(tcg.T, tcg.BATCH), None
    y = np.concatenate([np.zeros((tcg.DELAY, tcg.BATCH)), x[:-tcg.DELAY]],
                       axis=0)
    print("measuring curvature (finite differences)...", flush=True)
    curv = curvature(params, x, y)
    rows = []
    for l in range(tcg.L):
        for j in range(tcg.N):
            wj = abs(w[l][j])
            ca, cb = curv[(l, j)]
            rows.append(dict(layer=l, mode=j, wabs=wj, curv_a=ca,
                             curv_b=cb))
    for l in range(tcg.L):
        rs = [r for r in rows if r["layer"] == l]
        wa = np.array([r["wabs"] for r in rs])
        ca = np.array([r["curv_a"] for r in rs])
        cb = np.array([r["curv_b"] for r in rs])
        c1 = np.corrcoef(wa, 1.0 / np.maximum(ca, 1e-12))[0, 1]
        c2 = np.corrcoef(wa, 1.0 / np.maximum(cb, 1e-12))[0, 1]
        c3 = np.corrcoef(wa, ca)[0, 1]
        print(f"  layer {l}: corr(|w|, 1/curv_a) {c1:+.2f}   "
              f"corr(|w|, 1/curv_b) {c2:+.2f}   corr(|w|, curv_a) {c3:+.2f}")
    allr = rows
    wa = np.array([r["wabs"] for r in allr])
    ca = np.array([r["curv_a"] for r in allr])
    cb = np.array([r["curv_b"] for r in allr])
    print(f"  ALL: corr(|w|, 1/curv_a) "
          f"{np.corrcoef(wa, 1/np.maximum(ca, 1e-12))[0,1]:+.2f}   "
          f"corr(|w|, 1/curv_b) "
          f"{np.corrcoef(wa, 1/np.maximum(cb, 1e-12))[0,1]:+.2f}   "
          f"corr(|w|, curv_a) {np.corrcoef(wa, ca)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
