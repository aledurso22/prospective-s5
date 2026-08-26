"""D4 controls — the defensible form of the stability theorem.

d4_stability.py's common-eta counterexample conflates orientation with
estimator-dependent effective step size (exact BPTT also diverges
there; Wiener descends at its own smaller eta). This script adds the
three controls the critique requires, all exact in the same minimal
linear-quadratic model:

1. EXACT stability margin per estimator: for real 2x2 M with
   eigenvalues mu, stability iff |1 - eta mu| < 1 for both, i.e.
   eta < 2 Re(mu)/|mu|^2. eta_max(E) = min over eigenvalues (0 if any
   Re(mu) <= 0). No grid.
2. TUNED comparison: each estimator at its own optimal stable rate
   eta* = Re(mu)/|mu|^2 of the binding eigenvalue; report asymptotic
   contraction rho* = |Im mu|/|mu| and GD outcome at eta*.
3. NORM-MATCHED delta-L: all estimators rescaled to the same update
   norm, isolating direction quality: dL = -eta g*^T d~ + eta^2/2
   d~^T H d~ at matched ||d~||, plus cos(d_E, g*).

Defensible theorem (predeclared reading): eta_max is NOT monotone in
credit MSE across estimators x |a| (report the correlation). The
stronger claim "better credit causes worse learning" holds only if the
tuned/norm-matched controls still separate phase from online.

Run:  python d4_controls.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from diagnostics.d4_stability import (rollout, credit_seqs, grad_of, wiener_fit_1mode,
                          apply_fir, M_matrix, T, NB_ENS, SEED, A_GRID)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "d4_controls")


def margins(M):
    """Exact eta_max and tuned rate for a real 2x2 learning Jacobian."""
    eigs = np.linalg.eigvals(M)
    etas = []
    for mu in eigs:
        if mu.real <= 1e-12:
            return 0.0, None, None
        etas.append(2.0 * mu.real / abs(mu) ** 2)
    eta_max = min(etas)
    # binding eigenvalue at eta_max-scaled; tuned rate per eigenvalue:
    # rho*(mu) = |Im mu|/|mu| at eta* = Re mu/|mu|^2
    rho_star = max(abs(mu.imag) / abs(mu) for mu in eigs)
    eta_star = min(mu.real / abs(mu) ** 2 for mu in eigs)
    return float(eta_max), float(eta_star), float(rho_star)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    bstar = 1.0 + 0.5j
    b0 = np.array([0.0, 0.0])
    rows = []
    for amag in A_GRID:
        a = amag * np.exp(0.3j)
        xs = [rng.randn(T) for _ in range(NB_ENS)]
        qs, lams = [], []
        for x in xs:
            q, lam = credit_seqs(a, bstar + 1.0, bstar, x)
            qs.append(q)
            lams.append(lam)
        f64 = wiener_fit_1mode(qs, lams, 64)
        cs = np.mean([np.mean(lam * np.conj(q)) for q, lam in zip(qs, lams)])
        cs /= np.mean([np.mean(np.abs(q) ** 2) for q in qs])
        phase = np.exp(1j * np.angle(cs))
        err_fns = {"exact": lambda q, lam: lam,
                   "online": lambda q, lam: q,
                   "wiener64": lambda q, lam: apply_fir(q, f64),
                   "phase": lambda q, lam: phase * q}
        mse = {}
        for name in err_fns:
            num = sum(np.sum(np.abs(lam - err_fns[name](q, lam)) ** 2)
                      for q, lam in zip(qs, lams))
            den = sum(np.sum(np.abs(lam) ** 2) for q, lam in zip(qs, lams))
            mse[name] = float(num / den)
        M = {name: np.mean([M_matrix(a, bstar, x, fn) for x in xs], axis=0)
             for name, fn in err_fns.items()}
        H = 0.5 * (M["exact"] + M["exact"].T)
        db0 = b0 - np.array([bstar.real, bstar.imag])
        gstar0 = M["exact"] @ db0
        for name in err_fns:
            eta_max, eta_star, rho_star = margins(M[name])
            d = M[name] @ db0
            dhat = d / (np.linalg.norm(d) + 1e-300)
            cos = float(gstar0 @ dhat / (np.linalg.norm(gstar0) + 1e-300))
            # norm-matched dL at the best shared step for directions
            num_ = gstar0 @ dhat
            curv = dhat @ H @ dhat
            eta_nm = num_ / max(curv, 1e-300)
            dl_nm = -eta_nm * num_ + 0.5 * eta_nm ** 2 * curv
            # tuned GD outcome at own eta*
            conv = None
            if eta_star is not None:
                b = b0.copy()
                for n in range(300):
                    x = rng.randn(T)
                    bc = b[0] + 1j * b[1]
                    q, lam = credit_seqs(a, bc, bstar, x)
                    g = grad_of(err_fns[name](q, lam), rollout(a, 1.0, x))
                    b = b - eta_star * g
                conv = float(np.linalg.norm(
                    b - np.array([bstar.real, bstar.imag])))
            rows.append(dict(amag=amag, est=name, mse=mse[name],
                             eta_max=eta_max, eta_star=eta_star,
                             rho_star=rho_star, cos=cos,
                             dl_normmatched=float(dl_nm), tuned_dist=conv,
                             hankel=amag / (1 - amag ** 2)))
            print(f"|a|={amag:<5g} {name:<9s} mse {mse[name]:.3f}  "
                  f"eta_max {eta_max:.4g}  rho* {rho_star if rho_star is None else f'{rho_star:.3f}'}  "
                  f"cos {cos:+.3f}  dL_nm {dl_nm:.4g}  "
                  f"tuned|db| {conv if conv is None else f'{conv:.3g}'}",
                  flush=True)

    # the defensible theorem: eta_max not monotone in credit MSE
    x = np.array([r["mse"] for r in rows])
    y = np.array([r["eta_max"] for r in rows])
    corr = float(np.corrcoef(x, y)[0, 1])
    print("-" * 70)
    print(f"corr(credit MSE, eta_max) across estimators x |a| = {corr:+.3f}")
    print("predeclared reading: |corr| small or negative => credit "
          "reconstruction error does not determine the stability margin")
    # norm-matched direction quality at slowest mode
    slow = [r for r in rows if r["amag"] == 0.995]
    print("norm-matched at |a|=0.995: "
          + "  ".join(f"{r['est']}: cos {r['cos']:+.3f} dL {r['dl_normmatched']:.3g}"
                      for r in slow))

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, rows=rows, corr_mse_etamax=corr),
                  f, indent=2, default=float)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
