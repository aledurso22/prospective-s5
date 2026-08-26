"""PLDS BENCHMARK — the spotlight experiment suite.

Claim under test: prospective-metric structured Newton gives kappa-
INDEPENDENT, exact MAP inference for stiff non-Gaussian state-space models,
where standard optimizers degrade or silently fail. This is the solver's
habitat (mandatory tight solve + intrinsic stiffness + the answer IS the
solve), selected after the registered training nulls.

Suite:
  B1  main benchmark: kappa grid x 5 seeds x arms {newton, gd, anderson,
      broyden, lbfgs}; NFEs + wall time to rel residual 1e-8, final
      residual, state RMSE vs ground truth, solver status including
      FALSE-CONVERGENCE (solver reports success at res > 1e-3).
  B2  classical sanity gate: Gaussian-observation case — our Newton MAP
      must equal the closed-form Rauch-Tung-Striebel posterior mean
      (the exact classical answer) to ~1e-8.
  B3  capability cell: kappa = 1e10, T = 1000 — a regime where generic
      optimizers silently fail; report who converges and the state RMSE.
  B4  training relevance (mechanistic): at high kappa, the dynamics
      gradient dE/d_lam evaluated at a LOOSE solve (L-BFGS false-converged
      point) vs at the converged MAP — if loose solves corrupt the
      parameter gradient, solver quality is a TRAINING issue, not just an
      inference-speed issue.

Honest scope note (for the paper plan): the remaining steps to a full
submission are the public spike-data figure (Neural Latents Benchmark)
and the community baselines (LFADS variational posterior, Polya-Gamma
augmentation). This suite is the synthetic core + the gates.

Run:  python plds_benchmark.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from archive.solver.s5_state_inference import (
    make_problem, energy_grad, energy_val, hessian_terms, tridiag_solve_np,
    solve_newton, solve_gd, solve_anderson, solve_broyden, solve_lbfgs,
    RATE0,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAMBDAS = [0.99, 0.999, 0.9999]           # kappa ~ 4e4, 4e6, 4e8
SEEDS = [0, 1, 2, 3, 4]
TARGET_RES = 1e-8
MAX_STEPS = 2000
CAP_LAM = 1 - 1e-5                        # kappa = 1e10
CAP_T = 1000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results")

ARMS = ["newton", "gd", "anderson", "broyden", "lbfgs"]


def status_of(res):
    if not np.isfinite(res):
        return "diverged"
    if res <= 1e-6:
        return "converged"
    if res <= 1e-2:
        return "loose"
    return "failed"


# ---------------------------------------------------------------------------
# B1 — main benchmark
# ---------------------------------------------------------------------------

def run_one(arm, y, lam, s0, g0n, s_true):
    t0 = time.time()
    if arm == "newton":
        s, nfe = s0.copy(), 0
        res = 1.0
        while res > TARGET_RES and nfe < 100:
            s = solve_newton(y, lam, s, 1)
            nfe += 1
            res = float(np.linalg.norm(energy_grad(s, y, lam)) / g0n)
    elif arm == "lbfgs":
        s, njev = solve_lbfgs(y, lam, s0, MAX_STEPS)
        res = float(np.linalg.norm(energy_grad(s, y, lam)) / g0n)
        nfe = int(njev)
    else:
        fn = dict(gd=solve_gd, anderson=solve_anderson,
                  broyden=solve_broyden)[arm]
        s = fn(y, lam, s0, MAX_STEPS)
        res = float(np.linalg.norm(energy_grad(s, y, lam)) / g0n)
        nfe = MAX_STEPS
    wall = time.time() - t0
    st = status_of(res)
    # false convergence: L-BFGS stopped on its own criteria but is not loose
    false_conv = (arm == "lbfgs") and st in ("failed",)
    return dict(res=res, nfe=nfe, wall=wall, status=st,
                false_converged=false_conv,
                E=energy_val(s, y, lam) if np.isfinite(res) else None,
                rmse=(float(np.sqrt(np.mean((s - s_true) ** 2)))
                      if st in ("converged", "loose") else None))


def b1_main(rng):
    print("\n[B1] main benchmark (5 seeds; NFEs to rel residual 1e-8)")
    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            s_true, y = make_problem(lam, np.random.RandomState(100 + seed))
            s0 = np.zeros_like(s_true)
            g0n = np.linalg.norm(energy_grad(s0, y, lam))
            row = dict(lam=lam, kappa=((1 + lam) / (1 - lam)) ** 2,
                       seed=seed)
            for arm in ARMS:
                row[arm] = run_one(arm, y, lam, s0, g0n, s_true)
            rows.append(row)
        # summary across seeds
        r5 = [r for r in rows if r["lam"] == lam]
        print(f"\n  lam={lam:<7} kappa={row['kappa']:.1e}")
        for arm in ARMS:
            nfes = [r[arm]["nfe"] for r in r5]
            stats = [r[arm]["status"] for r in r5]
            rmses = [r[arm]["rmse"] for r in r5
                     if r[arm]["rmse"] is not None]
            fc = sum(r[arm]["false_converged"] for r in r5)
            print(f"    {arm:<9s} NFE med {int(np.median(nfes)):>5}  "
                  f"status {dict(zip(*np.unique(stats, return_counts=True)))}"
                  f"  false-conv {fc}"
                  + (f"  RMSE {np.median(rmses):.4f}" if rmses else ""))
    return rows


# ---------------------------------------------------------------------------
# B2 — Kalman gate (Gaussian case has a closed-form exact answer)
# ---------------------------------------------------------------------------

def rts_smoother(y, lam, sig_v2):
    """Scalar Rauch-Tung-Striebel smoother per mode (vectorized over modes).
    Prior: s_t = lam s_{t-1} + w, w ~ N(0, (1-lam)^2). Obs: y = s + N(0, sig_v2).
    Returns the exact posterior mean."""
    T, N = y.shape
    sig_w2 = (1 - lam) ** 2
    mf, Pf = np.zeros((T, N)), np.zeros((T, N))
    # prior matching the energy's chain term (s_{-1} = 0): s_0 ~ N(0, sig_w2)
    m_pred, P_pred = 0.0, sig_w2
    for t in range(T):
        K = P_pred / (P_pred + sig_v2)
        mf[t] = m_pred + K * (y[t] - m_pred)
        Pf[t] = (1 - K) * P_pred
        m_pred = lam * mf[t]
        P_pred = lam ** 2 * Pf[t] + sig_w2
    ms = np.zeros((T, N))
    ms[-1] = mf[-1]
    Ps_next = Pf[-1]
    for t in range(T - 2, -1, -1):
        P_pred = lam ** 2 * Pf[t] + sig_w2
        J = Pf[t] * lam / P_pred
        ms[t] = mf[t] + J * (ms[t + 1] - lam * mf[t])
        Ps_next = Pf[t] + J ** 2 * (Ps_next - P_pred)
    return ms


def b2_kalman_gate(rng):
    print("\n[B2] Kalman gate: newton MAP == RTS posterior mean (Gaussian)")
    lam = 0.999
    sig_v2 = 0.25
    s_true, _ = make_problem(lam, np.random.RandomState(7))
    y = s_true + np.sqrt(sig_v2) * rng.randn(*s_true.shape)

    def gauss_grad(s):
        sig2 = (1 - lam) ** 2
        s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
        r = (s - lam * s_prev) / sig2
        r_next = np.concatenate([r[1:], np.zeros_like(r[:1])], axis=0)
        return (r - lam * r_next) + (s - y) / sig_v2

    def gauss_hess(s):
        sig2 = (1 - lam) ** 2
        dg = (1 + lam ** 2) / sig2 + 1 / sig_v2
        dg = np.full_like(s, dg)
        dg[-1] = 1 / sig2 + 1 / sig_v2
        sub = np.full_like(s, -lam / sig2)
        sub[0] = 0.0
        return dg, sub

    s = np.zeros_like(y)
    for _ in range(2):                                    # quadratic: 1 step
        dg, sub = gauss_hess(s)
        s = s - tridiag_solve_np(dg, sub, gauss_grad(s))
    m_rts = rts_smoother(y, lam, sig_v2)
    err = float(np.max(np.abs(s - m_rts)) / np.max(np.abs(m_rts)))
    print(f"  max rel diff newton-MAP vs RTS mean = {err:.3e}  "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")
    assert err < 1e-6
    return dict(lam=lam, sig_v2=sig_v2, rel_diff=err)


# ---------------------------------------------------------------------------
# B3 — capability cell (kappa = 1e10)
# ---------------------------------------------------------------------------

def b3_capability(rng):
    print(f"\n[B3] capability cell: lam={CAP_LAM} (kappa=1e10), T={CAP_T}")
    lam = CAP_LAM
    eps = rng.randn(CAP_T, 8) * (1 - lam)
    s_true = np.zeros((CAP_T, 8))
    sp = np.zeros(8)
    for t in range(CAP_T):
        sp = lam * sp + eps[t]
        s_true[t] = sp
    y = rng.poisson(RATE0 * np.exp(s_true)).astype(np.float64)
    s0 = np.zeros_like(s_true)
    g0n = np.linalg.norm(energy_grad(s0, y, lam))
    row = {}
    for arm in ["newton", "lbfgs"]:
        row[arm] = run_one(arm, y, lam, s0, g0n, s_true)
        print(f"  {arm:<7s} NFE {row[arm]['nfe']:>5}  wall "
              f"{row[arm]['wall']:.2f}s  res {row[arm]['res']:.2e}  "
              f"status {row[arm]['status']}"
              + (f"  RMSE {row[arm]['rmse']:.4f}"
                 if row[arm]["rmse"] is not None else ""))
    return row


# ---------------------------------------------------------------------------
# B4 — training relevance: loose solves corrupt the dynamics gradient
# ---------------------------------------------------------------------------

def b4_gradient_corruption(rng):
    print("\n[B4] training relevance: dE/d_lam at converged vs loose solve")
    lam = 0.9999
    s_true, y = make_problem(lam, np.random.RandomState(11))
    s0 = np.zeros_like(s_true)

    def dE_dlam(s):
        sig2 = (1 - lam) ** 2
        s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
        r = s - lam * s_prev
        # d/d_lam of sum r^2/(2 sig2): r^2/sig2*(1/(1-lam)) - (r s_prev)/sig2
        return float(np.sum(r ** 2) / (sig2 * (1 - lam))
                     - np.sum(r * s_prev) / sig2)

    s_conv = solve_newton(y, lam, s0, 10)
    s_loose, _ = solve_lbfgs(y, lam, s0, 2000)
    res_conv = np.linalg.norm(energy_grad(s_conv, y, lam))
    res_loose = np.linalg.norm(energy_grad(s_loose, y, lam))
    g_conv = dE_dlam(s_conv)
    g_loose = dE_dlam(s_loose)
    rel = abs(g_loose - g_conv) / max(abs(g_conv), 1e-30)
    print(f"  state residuals: converged {res_conv:.2e} vs loose "
          f"{res_loose:.2e}")
    print(f"  dE/d_lam: converged {g_conv:.6f}  loose {g_loose:.6f}  "
          f"rel err {rel:.3f}")
    print("  => loose solves corrupt the dynamics gradient by "
          f"{100 * rel:.0f}% at kappa=4e8" if rel > 0.1 else
          f"  => gradient intact ({100 * rel:.1f}% error)")
    return dict(lam=lam, res_conv=float(res_conv),
                res_loose=float(res_loose), g_conv=g_conv, g_loose=g_loose,
                rel_err=float(rel))


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("PLDS benchmark — prospective Newton for stiff non-Gaussian inference")
    print("=" * 78)
    rng = np.random.RandomState(0)
    rows = b1_main(rng)
    gate = b2_kalman_gate(rng)
    cap = b3_capability(rng)
    corr = b4_gradient_corruption(rng)

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch,
               config=dict(lambdas=LAMBDAS, seeds=SEEDS,
                           target_res=TARGET_RES, max_steps=MAX_STEPS,
                           cap_lam=CAP_LAM, cap_T=CAP_T, rate0=RATE0),
               kalman_gate=gate, capability=cap, gradient_corruption=corr,
               rows=rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "plds_benchmark.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print("\n" + "=" * 78)
    print(f"wrote {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
