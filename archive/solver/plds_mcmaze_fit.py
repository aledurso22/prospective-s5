"""PLDS FIT on MC_Maze — does the inner solver change what gets LEARNED?

plds_mcmaze.py fixed the parameters and measured the solve. This script
learns them: alternating optimization of the joint posterior energy

    E(s, theta) = sum_tj (s_tj - lam_j s_{t-1,j})^2 / (2(1-lam_j)^2)
                  + sum_tc [ rate_tc - y_tc ((C s_t)_c + d_c) ],
    rate = exp(C s + d),

over latents s (E-step: the inner solve, by arm) and parameters
theta = (lam, C, d) (M-step: Adam on the partial gradients at the current
solve — the envelope theorem makes ds*/dtheta vanish at the minimizer, so
the outer gradient is exact for newton and corrupted for a loose solve).

This is the B4 mechanism on real data: at stiff kappa, a loose inner solve
corrupts dE/d_lam (99% in the synthetic benchmark), so the two arms should
LEARN different dynamics. Predictions (declared before running):
  P1  the newton arm retains stiffer learned lambdas than the lbfgs arm
      (the loose arm's dynamics gradients are corrupted toward softness);
  P2  the newton arm reaches equal-or-better held-out Poisson
      log-likelihood (8 held-out trials), despite identical capacity;
  P3  the newton arm is faster per outer step (fewer inner NFEs).

Arms: inner = newton (K=8, tight) vs inner = lbfgs (maxiter=200 — the
realistic per-solve budget; it false-converges or crawls at stiff kappa).
Config: 32 train / 8 test trials from the cached 20 ms bins, M=8 latents,
PCA init for C, stiff lambda init (0.99..0.9999), Adam lr 1e-3, 200 outer
steps. Joint MAP (not marginal EM) — declared simplification.

Run:  python plds_mcmaze_fit.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import warnings

import numpy as np
from scipy.optimize import minimize as sp_min

warnings.filterwarnings("ignore")

from archive.solver.plds_mcmaze import (load_binned, energy_grad_vec, energy_val,
                         solve_newton)

# ---------------------------------------------------------------------------
# Config (declared)
# ---------------------------------------------------------------------------

M_LAT = 8
LAM_INIT = np.geomspace(0.99, 0.9999, M_LAT)
N_TRAIN, N_TEST = 32, 8
OUTER_STEPS = 200
OUTER_LR = 1e-3
NEWTON_K = 8
LBFGS_MAXITER = 200
EVAL_EVERY = 20

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results")


# ---------------------------------------------------------------------------
# Parameter gradients (partials at the current inner solve; envelope theorem)
# ---------------------------------------------------------------------------

def param_grads(s, y, C, d, lams):
    """Partial derivatives of E wrt (lams, C, d) at fixed s. Summed over time."""
    sig2 = (1 - lams) ** 2
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    r = s - lams * s_prev                              # (T, M)
    # dE/dlam_j = sum_t r^2/(sig2*(1-lam)) - r*s_prev/sig2   (per mode)
    g_lam = (np.sum(r ** 2, axis=0) / (sig2 * (1 - lams))
             - np.sum(r * s_prev, axis=0) / sig2)
    rate = np.exp(s @ C + d)                           # (T, Cch)
    resid = rate - y                                   # (T, Cch)
    g_C = s.T @ resid                                  # (M, Cch)
    g_d = resid.sum(axis=0)                            # (Cch,)
    return g_lam, g_C, g_d


def held_out_ll(s, y, C, d):
    """Poisson log-likelihood up to constants: sum y*log(rate) - rate."""
    rate = np.exp(s @ C + d)
    return float(np.sum(y * np.log(np.maximum(rate, 1e-12)) - rate))


# ---------------------------------------------------------------------------
# One arm of the alternating fit
# ---------------------------------------------------------------------------

def fit_arm(arm, counts_train, counts_test, seed=0):
    rng = np.random.RandomState(seed)
    T = counts_train.shape[1]
    C_ch = counts_train.shape[2]

    z = np.log(counts_train.reshape(-1, C_ch) + 0.5)
    z = z - z.mean(0)
    _, _, Vt = np.linalg.svd(z, full_matrices=False)
    C = Vt[:M_LAT] * 0.5
    d = np.log(counts_train.mean(axis=(0, 1)) + 0.5)
    raw = np.log(LAM_INIT / (1 - LAM_INIT))              # logit init
    lams = 1 / (1 + np.exp(-raw))

    # Adam state
    m = [np.zeros_like(raw), np.zeros_like(C), np.zeros_like(d)]
    v = [np.zeros_like(raw), np.zeros_like(C), np.zeros_like(d)]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    inner_nfe_total = 0
    t_start = time.time()

    def inner_all(lams, C, d):
        nonlocal inner_nfe_total
        sols = []
        for y in counts_train:
            s0 = np.zeros((T, M_LAT))
            if arm == "newton":
                s = solve_newton(y, C, d, lams, s0, NEWTON_K)
                inner_nfe_total += NEWTON_K
            else:
                fg = lambda sf: (
                    energy_val(sf.reshape(T, M_LAT), y, C, d, lams),
                    energy_grad_vec(sf.reshape(T, M_LAT), y, C, d,
                                    lams).reshape(-1))
                out = sp_min(fg, s0.reshape(-1), jac=True, method="L-BFGS-B",
                             options=dict(maxiter=LBFGS_MAXITER))
                s = out.x.reshape(T, M_LAT)
                inner_nfe_total += int(out.njev)
            sols.append(s)
        return sols

    for step in range(1, OUTER_STEPS + 1):
        lams = 1 / (1 + np.exp(-raw))
        sols = inner_all(lams, C, d)
        g_lam = np.zeros(M_LAT)
        g_C = np.zeros_like(C)
        g_d = np.zeros_like(d)
        for s, y in zip(sols, counts_train):
            gl, gc, gd_ = param_grads(s, y, C, d, lams)
            g_lam += gl
            g_C += gc
            g_d += gd_
        # lam gradient through the sigmoid
        g_raw = g_lam * lams * (1 - lams)
        for i, (g, p) in enumerate(zip([g_raw, g_C, g_d], [raw, C, d])):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g ** 2
            p -= OUTER_LR * (m[i] / (1 - b1 ** step)) / (
                np.sqrt(v[i] / (1 - b2 ** step)) + eps)   # in-place update

        if step % EVAL_EVERY == 0 or step == OUTER_STEPS:
            lams = 1 / (1 + np.exp(-raw))
            # E-step on test trials (newton always: measures model quality,
            # not solver speed) then held-out log-likelihood
            ll = 0.0
            for y in counts_test:
                s_te = solve_newton(y, C, d, lams, np.zeros((T, M_LAT)),
                                    NEWTON_K)
                ll += held_out_ll(s_te, y, C, d)
            ll /= N_TEST
            res = [np.linalg.norm(energy_grad_vec(s, y, C, d, lams))
                   for s, y in zip(sols[:4], counts_train[:4])]
            hist.append(dict(step=step, heldout_ll=ll,
                             lams=list(map(float, lams)),
                             inner_res_med=float(np.median(res))))
            print(f"    {arm} step {step:>3}: heldout ll {ll:.1f}  "
                  f"lam med {np.median(lams):.4f}  inner res "
                  f"{np.median(res):.1e}", flush=True)

    lams_final = 1 / (1 + np.exp(-raw))
    return dict(arm=arm, history=hist, lams_final=list(map(float, lams_final)),
                inner_nfe_total=inner_nfe_total,
                wall_time_sec=time.time() - t_start)


def main() -> None:
    print("=" * 78)
    print("PLDS fit on MC_Maze: does the inner solver change what is learned?")
    print("=" * 78)
    counts, _ = load_binned()
    counts_train, counts_test = counts[:N_TRAIN], counts[N_TRAIN:N_TRAIN + N_TEST]
    print(f"train {counts_train.shape}  test {counts_test.shape}")
    out = {}
    for arm in ["newton", "lbfgs"]:
        print(f"  arm: {arm}")
        out[arm] = fit_arm(arm, counts_train, counts_test)
        print(f"  {arm}: final lams "
              f"{['%.4f' % x for x in out[arm]['lams_final']]}")
        print(f"  {arm}: total inner NFEs {out[arm]['inner_nfe_total']}, "
              f"wall {out[arm]['wall_time_sec']:.0f}s")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch,
               config=dict(M_lat=M_LAT, outer_steps=OUTER_STEPS,
                           outer_lr=OUTER_LR, newton_K=NEWTON_K,
                           lbfgs_maxiter=LBFGS_MAXITER,
                           n_train=N_TRAIN, n_test=N_TEST,
                           note="joint MAP alternating optimization; PCA init"),
               arms=out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "plds_mcmaze_fit.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
