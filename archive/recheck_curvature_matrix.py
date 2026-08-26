"""Matrix-level recheck: is the learned metric the prospective mass?

The scalar check (recheck_curvature.py) found corr(|w|, 1/curv_a) ~ 0.
The reviewer correctly objected that magnitude correlation is not the
decisive test. The action-derived mobility on mode j is the 2x2 real
matrix

    M_j = (I + tau H_j)^{-1},   H_j = d^2 L / d(Re a_j, Im a_j)^2,

and the learned metric (multiplication of the mode's gradient by
conj(w_j), w_j = u + iv) is the conformal matrix

    W_j = [[u, v], [-v, u]].

This script runs the four registered diagnostics per layer:
  1. magnitude correlation  corr(|w_j|, ||M_j||_F)        (best tau)
  2. rotation content       phase(W_j) vs rotation(M_j)
  3. Frobenius cosine       <W_j, M_j> / (||W_j|| ||M_j||) (best tau)
  4. best scalar fit        c_l* = argmin_c sum_j ||W_j - c M_j||^2,
     residual fraction, and the STRUCTURAL FLOOR
     sum 2 v_j^2 / sum 2 (u_j^2 + v_j^2) — the part of W no symmetric
     mobility can ever fit (H real-symmetric => M symmetric => zero
     antisymmetric/rotational part), independent of tau.

If the residual approaches the floor and the floor is small, the
curvature-mass story lives. If the residual is large and the floor is
the dominant term, the load-bearing phaseful part of W is structurally
unreachable by (I + tau H)^{-1}.

Run:  python recheck_curvature_matrix.py
"""
from __future__ import annotations

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig.train_cell import train_cell

TAUS = np.logspace(-4, 4, 17)
H_FD = 1e-3


def batch_loss(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    return 0.5 * float(np.mean(r ** 2))


def with_da(params, l, j, da):
    """Copy params with a[l][j] shifted by complex da (forward reads
    params['a'] directly, so this bypasses the sigmoid parameterization —
    curvature is measured in the (Re a, Im a) coordinates the metric
    acts on)."""
    p2 = {k: (v.copy() if not isinstance(v, list) else [a.copy() for a in v])
          for k, v in params.items()}
    p2["a"] = [a.copy() for a in params["a"]]
    p2["a"][l][j] = p2["a"][l][j] + da
    return p2


def hessian_block(params, x, y, l, j, f0):
    """2x2 real Hessian of the loss w.r.t. (Re a_j, Im a_j), central FD."""
    e = H_FD
    fxp = batch_loss(with_da(params, l, j, e), x, y)
    fxm = batch_loss(with_da(params, l, j, -e), x, y)
    fyp = batch_loss(with_da(params, l, j, 1j * e), x, y)
    fym = batch_loss(with_da(params, l, j, -1j * e), x, y)
    fpp = batch_loss(with_da(params, l, j, e + 1j * e), x, y)
    fpm = batch_loss(with_da(params, l, j, e - 1j * e), x, y)
    fmp = batch_loss(with_da(params, l, j, -e + 1j * e), x, y)
    fmm = batch_loss(with_da(params, l, j, -e - 1j * e), x, y)
    fxx = (fxp - 2 * f0 + fxm) / e ** 2
    fyy = (fyp - 2 * f0 + fym) / e ** 2
    fxy = (fpp - fpm - fmp + fmm) / (4 * e ** 2)
    return np.array([[fxx, fxy], [fxy, fyy]])


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    print("training routeA (L=4, D=50, seed 0)...", flush=True)
    params, w = train_cell(4, 50, 0)
    rng = np.random.RandomState(999)
    x = rng.randn(tcg.T, tcg.BATCH)
    y = np.concatenate([np.zeros((tcg.DELAY, tcg.BATCH)), x[:-tcg.DELAY]],
                       axis=0)
    f0 = batch_loss(params, x, y)
    print(f"probe loss {f0:.6f}; measuring Hessian blocks...", flush=True)

    # collect W_j and eigendecompositions of H_j per layer
    layers = []
    for l in range(tcg.L):
        Ws, eigs, Qs, phases = [], [], [], []
        for j in range(tcg.N):
            u, v = w[l][j].real, w[l][j].imag
            Ws.append(np.array([[u, v], [-v, u]]))
            phases.append(np.arctan2(v, u))
            H = hessian_block(params, x, y, l, j, f0)
            ev, Q = np.linalg.eigh(H)
            eigs.append(ev)
            Qs.append(Q)
        layers.append(dict(W=Ws, eig=eigs, Q=Qs, phase=np.array(phases)))
        print(f"  layer {l} done; median |w| "
              f"{np.median([abs(w[l][j]) for j in range(tcg.N)]):.3f}  "
              f"median phase {np.median(np.abs(phases)):.3f} rad  "
              f"median eig(H) {np.median([e[0] for e in eigs]):.2e} / "
              f"{np.median([e[1] for e in eigs]):.2e}", flush=True)

    print("\ntau sweep (per-layer best tau by residual fraction):")
    for l, lay in enumerate(layers):
        W = np.array(lay["W"])                      # (N,2,2)
        wnorm2 = (W ** 2).sum(axis=(1, 2))          # 2(u^2+v^2)
        best = None
        for tau in TAUS:
            Ms, ok = [], []
            for j in range(tcg.N):
                lam = 1.0 + tau * lay["eig"][j]
                if np.any(np.abs(lam) < 1e-10):
                    ok.append(False)
                    Ms.append(np.zeros((2, 2)))
                    continue
                ok.append(True)
                Ms.append(lay["Q"][j] @ np.diag(1.0 / lam) @ lay["Q"][j].T)
            M = np.array(Ms)
            ok = np.array(ok)
            if ok.sum() < tcg.N // 2:
                continue
            Wk, Mk = W[ok], M[ok]
            ip = (Wk * Mk).sum(axis=(1, 2))         # <W_j, M_j>_F
            mnorm2 = (Mk ** 2).sum(axis=(1, 2))
            c = ip.sum() / mnorm2.sum()
            resid = 1.0 - ip.sum() ** 2 / (wnorm2[ok].sum() * mnorm2.sum())
            cos = ip / np.sqrt(wnorm2[ok] * mnorm2)
            wabs = np.abs([w[l][j] for j in range(tcg.N)])[ok]
            mnorm = np.sqrt(mnorm2)
            rc = np.corrcoef(wabs, mnorm)[0, 1] if ok.sum() > 2 else 0.0
            if best is None or resid < best["resid"]:
                best = dict(tau=tau, resid=resid, c=c,
                            cos=float(np.median(cos)), rc=rc)
        # structural floor: antisymmetric (rotational) energy of W
        u = W[:, 0, 0]
        v = W[:, 0, 1]
        floor = (2 * v ** 2).sum() / (2 * (u ** 2 + v ** 2)).sum()
        ph = np.median(np.abs(lay["phase"]))
        print(f"  layer {l}: best tau {best['tau']:.2e}  c* {best['c']:+.3e}  "
              f"resid {best['resid']:.3f}  floor {floor:.3f}  "
              f"medcos {best['cos']:+.3f}  corr(|w|,||M||) {best['rc']:+.2f}  "
              f"med|phase(W)| {ph:.2f} rad")

    print("\nreading: resid ~= floor and floor large  =>  the part of W that")
    print("matters (its rotation) is unreachable by ANY symmetric mobility;")
    print("resid << floor would mean the tau-fit found structure beyond the")
    print("structural bound (impossible for symmetric M — check for bugs).")


if __name__ == "__main__":
    main()
