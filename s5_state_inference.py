"""When does the prospective solver beat SOTA? — stiff SSM STATE INFERENCE.

The registered training experiments taught us where the solver does NOT
matter: training is robust to loose equilibrium solves (both arms learned
the same with 50% residuals). The regime where the solver difference is
STRUCTURAL is inference: the equilibrium IS the answer.

Setting: a long-memory linear dynamical system (stiff AR(1) per mode,
|lam| -> 1, kappa ~ ((1+lam)/(1-lam))^2) with NON-GAUSSIAN observations —
Poisson counts, the classic point-process/PLDS setting. The exact Kalman
smoother does not exist here; the field's options are iterative
optimization of the posterior (L-BFGS is the strong baseline) or Gaussian
approximations. The posterior energy

    E(s) = sum_t (s_t - lam s_{t-1})^2 / (2 (1-lam)^2)
           + sum_t [ RATE0 * exp(s_t) - y_t s_t ]

is CONVEX (Poisson NLL + quadratic chain), so the MAP is unique and the
Newton system is exact: Hessian = stiff tridiagonal chain + diagonal
5e^s. The prospective arm solves it by tridiagonal associative scans;
the comparison arms are the prospective-deq solver zoo plus L-BFGS.

Question (predeclared): NFEs to reach relative gradient residual 1e-6 as
kappa grows from 4e2 to 4e8. Expectation from the theory: newton flat in
kappa (convex + exact Hessian => few steps), first-order arms grow like
kappa (GD) or sqrt(kappa) (L-BFGS/Anderson/Broyden at best).

Arms: newton (exact Hessian, 3 scans/step), gd (eta = 2/lambda_max bound),
anderson (m=5), broyden (m=5), lbfgs (scipy, the field baseline).

Gates:
  I1  tridiagonal solve == numpy dense solve on the Poisson Hessian
  I2  newton reaches rel residual <= 1e-8 in <= 20 steps at every lambda
  I3  all arms that converge reach the SAME maximizer (log-posterior
      agreement) — the claim is cost, not a different answer

Run:  python s5_state_inference.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T = 256
N_MODES = 16
LAMBDAS = [0.9, 0.99, 0.999, 0.9999]      # kappa ~ 3.6e2, 4e4, 4e6, 4e8
RATE0 = 5.0
TARGET_RES = 1e-8
MAX_STEPS = 2000
SEED = 0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

# ---------------------------------------------------------------------------
# Problem: ground-truth trajectory + Poisson observations + posterior
# ---------------------------------------------------------------------------

def make_problem(lam, rng):
    """Simulate the AR(1) chain (innovation 1-lam => O(sqrt((1-lam)/(1+lam)))
    stationary scale) and Poisson counts y_tj ~ Poisson(RATE0 * exp(s_tj))."""
    eps = rng.randn(T, N_MODES) * (1 - lam)
    s = np.zeros((T, N_MODES))
    sp = np.zeros(N_MODES)
    for t in range(T):
        sp = lam * sp + eps[t]
        s[t] = sp
    rate = RATE0 * np.exp(s)
    y = rng.poisson(rate).astype(np.float64)
    return s, y


def energy_grad(s, y, lam):
    """Gradient of E (above) at trajectory s. s, y: (T, N)."""
    sig2 = (1 - lam) ** 2
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    r = (s - lam * s_prev) / sig2
    r_next = np.concatenate([r[1:], np.zeros_like(r[:1])], axis=0)
    return (r - lam * r_next) + RATE0 * np.exp(s) - y


def energy_val(s, y, lam):
    sig2 = (1 - lam) ** 2
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    chain = np.sum((s - lam * s_prev) ** 2) / (2 * sig2)
    nll = np.sum(RATE0 * np.exp(s) - y * s)
    return float(chain + nll)


def hessian_terms(s, lam):
    """Exact Hessian: diag = (1+lam^2)/(1-lam)^2 + RATE0 e^s (last row lacks
    the forward coupling), off-diag = -lam/(1-lam)^2. SPD (convex E)."""
    sig2 = (1 - lam) ** 2
    dg = (1 + lam ** 2) / sig2 + RATE0 * np.exp(s)
    dg[-1] = 1.0 / sig2 + RATE0 * np.exp(s[-1])
    sub = np.full_like(s, -lam / sig2)
    sub[0] = 0.0
    return dg, sub


# ---------------------------------------------------------------------------
# Tridiagonal solve (numpy Thomas for this standalone script — the scan
# version is gated in pesm_s5_spectrum.py; I1 here gates against dense)
# ---------------------------------------------------------------------------

def tridiag_solve_np(dg, sub, g):
    """Solve tridiagonal H x = g along axis 0. diag real (T, ...),
    sub = H[t, t-1] (T, ...), sub[0] ignored. Thomas algorithm."""
    T = g.shape[0]
    d = np.empty_like(dg)
    v = np.empty_like(g)
    d[0] = dg[0]
    v[0] = g[0]
    for t in range(1, T):
        w = sub[t] / d[t - 1]
        d[t] = dg[t] - sub[t] * w
        v[t] = g[t] - w * v[t - 1]
    x = np.empty_like(g)
    x[-1] = v[-1] / d[-1]
    for t in range(T - 2, -1, -1):
        x[t] = (v[t] - np.conj(sub[t + 1]) * x[t + 1]) / d[t]
    return x


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def solve_newton(y, lam, s0, K):
    s = s0.copy()
    for _ in range(K):
        g = energy_grad(s, y, lam)
        dg, sub = hessian_terms(s, lam)
        s = s - tridiag_solve_np(dg, sub, g)
    return s


def solve_gd(y, lam, s0, K):
    # stable step from the Hessian diagonal bound: lam_max <= max(diag)
    eta = 1.8 / ((1 + lam) ** 2 / (1 - lam) ** 2 + RATE0 * np.exp(4.0))
    s = s0.copy()
    for _ in range(K):
        s = s - eta * energy_grad(s, y, lam)
    return s


def solve_anderson(y, lam, s0, K, m=5):
    eta = 1.8 / ((1 + lam) ** 2 / (1 - lam) ** 2 + RATE0 * np.exp(4.0))
    D = s0.size
    S = np.zeros((m, D))
    G = np.zeros((m, D))
    s = s0.reshape(-1).copy()
    for k in range(K):
        g = eta * energy_grad(s.reshape(s0.shape), y, lam).reshape(-1)
        S = np.roll(S, -1, axis=0); S[-1] = s
        G = np.roll(G, -1, axis=0); G[-1] = g
        count = min(k + 1, m)
        valid = np.arange(m) >= m - count
        Gram = G @ G.T
        valid2 = valid[:, None] & valid[None, :]
        Gram_m = np.where(valid2, Gram, np.eye(m)) + 1e-8 * np.eye(m)
        alpha = np.linalg.solve(Gram_m, np.ones(m))
        alpha = np.where(valid, alpha, 0.0)
        asum = alpha.sum()
        if abs(asum) > 1e-30:
            alpha = alpha / asum
        s = np.sum(alpha[:, None] * (S - G), axis=0)
    return s.reshape(s0.shape)


def solve_broyden(y, lam, s0, K, m=5):
    D = s0.size
    shape = s0.shape
    s = s0.reshape(-1).copy()
    g_prev = np.zeros(D)
    ds = np.zeros(D)
    U = np.zeros((m, D))
    V = np.zeros((m, D))
    for k in range(K):
        g = energy_grad(s.reshape(shape), y, lam).reshape(-1)
        dg = g - g_prev
        B_ds = ds + U.T @ (V @ ds)
        denom = float(np.dot(ds, ds))
        if denom > 1e-24:
            U = np.roll(U, -1, axis=0); U[-1] = (dg - B_ds) / denom
            V = np.roll(V, -1, axis=0); V[-1] = ds
        Mm = np.eye(m) + V @ U.T
        p = g - U.T @ np.linalg.solve(Mm, V @ g)
        s_new = s - p
        g_prev, ds, s = g, s_new - s, s_new
    return s.reshape(shape)


def solve_lbfgs(y, lam, s0, maxiter):
    shape = s0.shape

    def fg(sf):
        s = sf.reshape(shape)
        return energy_val(s, y, lam), energy_grad(s, y, lam).reshape(-1)

    res = scipy_minimize(fg, s0.reshape(-1), jac=True, method="L-BFGS-B",
                         options=dict(maxiter=maxiter, maxfun=10 * maxiter))
    return res.x.reshape(shape), res.njev


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def gate_I1():
    rng = np.random.RandomState(1)
    Tt, M = 40, 3
    s = rng.randn(Tt, M) * 0.3
    lam = 0.99
    dg, sub = hessian_terms(s, lam)
    g = rng.randn(Tt, M)
    x = tridiag_solve_np(dg, sub, g)
    err = 0.0
    for m in range(M):
        H = (np.diag(dg[:, m]) + np.diag(sub[1:, m], -1)
             + np.diag(sub[1:, m], 1))
        xd = np.linalg.solve(H, g[:, m])
        err = max(err, np.linalg.norm(x[:, m] - xd) / np.linalg.norm(xd))
    print(f"  I1 tridiag == dense (Poisson Hessian)  rel err = {err:.3e}  "
          f"{'PASS' if err < 1e-10 else 'FAIL'}")
    assert err < 1e-10


def rel_res(s, y, lam, g0n):
    return float(np.linalg.norm(energy_grad(s, y, lam)) / g0n)


def run_lambda(lam, rng):
    s_true, y = make_problem(lam, rng)
    kappa = ((1 + lam) / (1 - lam)) ** 2
    s0 = np.zeros_like(s_true)
    g0n = np.linalg.norm(energy_grad(s0, y, lam))

    # Newton: NFEs to target (convex + exact Hessian => few steps)
    s, nfe_n = s0.copy(), 0
    res = 1.0
    while res > TARGET_RES and nfe_n < MAX_STEPS:
        s = solve_newton(y, lam, s, 1)
        nfe_n += 1
        res = rel_res(s, y, lam, g0n)
    s_newton, newton_nfe, newton_res = s, nfe_n, res

    # iterative arms at the fixed budget
    outs = {}
    for arm in ["gd", "anderson", "broyden"]:
        fn = dict(gd=solve_gd, anderson=solve_anderson,
                  broyden=solve_broyden)[arm]
        t0 = time.time()
        s_a = fn(y, lam, s0, MAX_STEPS)
        dt = time.time() - t0
        outs[arm] = dict(res=rel_res(s_a, y, lam, g0n), nfe=MAX_STEPS,
                         wall=dt, E=energy_val(s_a, y, lam),
                         rmse=float(np.sqrt(np.mean((s_a - s_true) ** 2)))
                         if rel_res(s_a, y, lam, g0n) < 1e-3 else None)
    t0 = time.time()
    s_lbfgs, njev = solve_lbfgs(y, lam, s0, MAX_STEPS)
    dt = time.time() - t0
    outs["lbfgs"] = dict(res=rel_res(s_lbfgs, y, lam, g0n), nfe=int(njev),
                         wall=dt, E=energy_val(s_lbfgs, y, lam),
                         rmse=float(np.sqrt(np.mean((s_lbfgs - s_true) ** 2)))
                         if rel_res(s_lbfgs, y, lam, g0n) < 1e-3 else None)

    row = dict(lam=lam, kappa=kappa, newton=dict(
        res=newton_res, nfe=newton_nfe, E=energy_val(s_newton, y, lam),
        rmse=float(np.sqrt(np.mean((s_newton - s_true) ** 2)))), **outs)

    print(f"\n  lam={lam:<7} kappa={kappa:.1e}")
    print(f"    newton : NFE {row['newton']['nfe']:>5}  res "
          f"{row['newton']['res']:.2e}  E {row['newton']['E']:.6f}  "
          f"RMSE {row['newton']['rmse']:.4f}")
    for arm in ["gd", "anderson", "broyden", "lbfgs"]:
        o = row[arm]
        rmse = f"{o['rmse']:.4f}" if o["rmse"] is not None else "  n/c "
        print(f"    {arm:<8s}: NFE {o['nfe']:>5}  res {o['res']:.2e}  "
              f"E {o['E']:.6f}  RMSE {rmse}")
    return row


def main() -> None:
    print("=" * 78)
    print("Stiff SSM state inference (Poisson PLDS) — prospective vs SOTA")
    print("=" * 78)
    print("[gates]")
    gate_I1()
    rng = np.random.RandomState(SEED)
    rows = [run_lambda(lam, rng) for lam in LAMBDAS]

    # I2: newton converges in <= 20 NFEs everywhere
    ok2 = all(r["newton"]["nfe"] <= 20 and r["newton"]["res"] <= 1e-8
              for r in rows)
    # I3: converged arms agree on the maximizer (same E within 1e-4 rel)
    agree = []
    for r in rows:
        Es = [r["newton"]["E"]] + [r[a]["E"] for a in
                                   ("gd", "anderson", "broyden", "lbfgs")
                                   if r[a]["res"] < 1e-3]
        agree.append(max(Es) - min(Es) < 1e-4 * abs(Es[0]) if len(Es) > 1
                     else True)
    ok3 = all(agree)
    print("\n" + "=" * 78)
    print(f"I2 newton <= 20 NFEs to 1e-8 at all lambda: {ok2}")
    print(f"I3 converged arms agree on the maximizer:  {ok3}")
    assert ok2 and ok3

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch, seed=SEED,
               config=dict(T=T, N_modes=N_MODES, lambdas=LAMBDAS,
                           rate0=RATE0, target_res=TARGET_RES,
                           max_steps=MAX_STEPS),
               rows=rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "s5_state_inference.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
