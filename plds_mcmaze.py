"""REAL-DATA FIGURE — stiff PLDS inference on MC_Maze (NLB'21, DANDI 000128).

The synthetic benchmark (plds_benchmark.py) established kappa-independent
prospective-Newton MAP inference. This script is the contact with reality:
macaque motor/premotor spike trains, 20 ms count bins, Poisson observations
with a loading matrix — the setting where the exact Kalman form does not
exist and the field falls back on generic optimization.

DECLARED SCOPE: this experiment evaluates the INNER SOLVE on real data.
Model parameters (lambda grid, loading matrix C, bias d) are a fixed
heuristic init (PCA of smoothed log-rates); we do not claim a fitted model
quality comparison, only solver cost/robustness on the real problem.

Model (per trial): latents s_t in R^M (M=8 stiff AR(1) modes,
lambda in {0.99, ..., 0.9999} => kappa up to ~4e8), observations
y_tc counts, rate_tc = exp((C s_t)_c + d_c). Posterior energy

    E(s) = sum_tj (s_tj - lam_j s_{t-1,j})^2 / (2 (1-lam_j)^2)
           + sum_tc [ rate_tc - y_tc (C s_t)_c ]

is convex with a BLOCK-tridiagonal Hessian (M x M blocks): chain part +
C' diag(rate_t) C. Newton step = block-Thomas solve (exact Hessian).

Arms: newton (block-Thomas), lbfgs (scipy), gd (eta from spectral bound).
Metrics per trial: NFEs and wall time to rel residual 1e-8, final energy,
status (converged / loose / failed / false-converged), and cross-arm
agreement on the maximizer.

Run:  python plds_mcmaze.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NWB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "nlb", "mc_maze_train.nwb")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "nlb", "mc_maze_binned.npz")
BIN_MS = 20
N_TRIALS = 40               # trials used for the figure
M_LAT = 8
LAMBDAS = np.geomspace(0.99, 0.9999, M_LAT)   # stiff grid, kappa 4e4..4e8
TARGET_RES = 1e-8
MAX_STEPS = 2000
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")


# ---------------------------------------------------------------------------
# Data: load NWB once, cache binned counts
# ---------------------------------------------------------------------------

def load_binned():
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        return z["counts"], z["rate_mean"]
    from nlb_tools.nwb_interface import NWBDataset
    ds = NWBDataset(NWB_PATH, skip_fields=["behavior", "heldout_spikes"])
    d = ds.data["spikes"]                       # (6.95M ms, 137) counts at 1ms
    ti = ds.trial_info
    ti = ti[ti["success"] == True].iloc[:N_TRIALS]
    idx = d.index.values.astype("timedelta64[ns]").astype(np.int64)
    starts = ti["start_time"].values.astype("timedelta64[ns]").astype(np.int64)
    ends = ti["end_time"].values.astype("timedelta64[ns]").astype(np.int64)
    i0 = np.searchsorted(idx, starts)
    i1 = np.searchsorted(idx, ends)
    T_min = min((e - s) // BIN_MS for s, e in zip(i0, i1))
    trials = []
    for s, e in zip(i0, i1):
        blk = d.iloc[s:s + T_min * BIN_MS].values.astype(np.float64)
        trials.append(blk.reshape(T_min, BIN_MS, -1).sum(axis=1))
    counts = np.stack(trials)                   # (n_trials, T, 137)
    rate_mean = counts.mean(axis=(0, 1))        # mean count per bin/channel
    np.savez(CACHE_PATH, counts=counts, rate_mean=rate_mean)
    return counts, rate_mean


# ---------------------------------------------------------------------------
# Model energy (per trial): convex, block-tridiagonal Hessian
# ---------------------------------------------------------------------------

def unpack(C, d):
    return C, d


def energy_grad_vec(s, y, C, d, lams):
    """s: (T, M) latents, y: (T, C) counts. grad wrt s."""
    sig2 = (1 - lams) ** 2                       # (M,)
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    r = (s - lams * s_prev) / sig2
    r_next = np.concatenate([r[1:], np.zeros_like(r[:1])], axis=0)
    g_chain = r - lams * r_next                  # (T, M)
    rate = np.exp(s @ C + d)                   # (T, C)
    g_obs = (rate - y) @ C.T                     # (T, M)
    return g_chain + g_obs


def energy_val(s, y, C, d, lams):
    sig2 = (1 - lams) ** 2
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    chain = np.sum((s - lams * s_prev) ** 2 / (2 * sig2))
    rate = np.exp(s @ C + d)
    return float(chain + np.sum(rate - y * (s @ C + d)))


def hessian_blocks_full(s, y, C, d, lams):
    T, M = s.shape
    sig2 = (1 - lams) ** 2
    chain_diag = (1 + lams ** 2) / sig2          # (M,)
    rate = np.exp(s @ C + d)                   # (T, C)
    A = (np.einsum("tc,kc,lc->tkl", rate, C, C)
         + chain_diag[None, :, None] * np.eye(M)[None])
    A[-1] += (-lams ** 2 / sig2)[:, None] * np.eye(M)   # last row: no r_{t+1}
    B = np.zeros((T, M, M))
    B[1:] = -(lams / sig2)[:, None] * np.eye(M)[None]   # B_t = H[t, t-1]
    return A, B


def block_thomas(A, B, g):
    """Solve block-tridiagonal H x = g. A: (T,M,M) SPD diag blocks,
    B: (T,M,M) with B[t] = H[t, t-1], g: (T,M). Sequential block Thomas."""
    T, M = g.shape
    D = np.empty_like(A)
    v = np.empty_like(g)
    D[0] = A[0]
    v[0] = g[0]
    for t in range(1, T):
        W = np.linalg.solve(D[t - 1], B[t].T)        # D_{t-1}^{-1} B_t^T
        D[t] = A[t] - B[t] @ W
        v[t] = g[t] - B[t] @ np.linalg.solve(D[t - 1], v[t - 1])
    x = np.empty_like(g)
    x[-1] = np.linalg.solve(D[-1], v[-1])
    for t in range(T - 2, -1, -1):
        x[t] = np.linalg.solve(D[t], v[t] - B[t + 1].T @ x[t + 1])
    return x


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def solve_newton(y, C, d, lams, s0, K):
    s = s0.copy()
    for _ in range(K):
        g = energy_grad_vec(s, y, C, d, lams)
        A, B = hessian_blocks_full(s, y, C, d, lams)
        s = s - block_thomas(A, B, g)
    return s


def solve_gd(y, C, d, lams, s0, K):
    sig2 = (1 - lams) ** 2
    lam_max = ((1 + lams) ** 2 / sig2).max() + 4 * np.linalg.norm(C, 2) ** 2
    eta = 1.8 / lam_max
    s = s0.copy()
    for _ in range(K):
        s = s - eta * energy_grad_vec(s, y, C, d, lams)
    return s


def rel_res(s, y, C, d, lams, g0n):
    return float(np.linalg.norm(energy_grad_vec(s, y, C, d, lams)) / g0n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("PLDS inference on MC_Maze (real spike data) — prospective vs SOTA")
    print("=" * 78)
    counts, rate_mean = load_binned()
    n_trials, T, C_ch = counts.shape
    print(f"data: {n_trials} trials x {T} bins x {C_ch} channels "
          f"({BIN_MS} ms bins)")

    # heuristic parameter init: C from SVD of smoothed log-rates, d = log mean
    z = np.log(counts.reshape(-1, C_ch) + 0.5)   # (n_trials*T, 137)
    z = z - z.mean(0)
    U, Sv, Vt = np.linalg.svd(z, full_matrices=False)
    C = Vt[:M_LAT] * 0.5                          # (M, 137) loadings
    d = np.log(rate_mean + 0.5)
    lams = LAMBDAS
    print(f"kappa range: {((1+lams.min())/(1-lams.min()))**2:.1e} .. "
          f"{((1+lams.max())/(1-lams.max()))**2:.1e}")

    results = []
    for tr in range(n_trials):
        y = counts[tr]
        s0 = np.zeros((T, M_LAT))
        g0n = np.linalg.norm(energy_grad_vec(s0, y, C, d, lams))
        row = dict(trial=tr)
        # newton: NFEs to target
        s, nfe, res = s0.copy(), 0, 1.0
        t0 = time.time()
        while res > TARGET_RES and nfe < 100:
            s = solve_newton(y, C, d, lams, s, 1)
            nfe += 1
            res = rel_res(s, y, C, d, lams, g0n)
        row["newton"] = dict(nfe=nfe, res=res, wall=time.time() - t0,
                             E=energy_val(s, y, C, d, lams))
        # lbfgs (scipy, the field baseline)
        from scipy.optimize import minimize as sp_min
        t0 = time.time()
        fg = lambda sf: (energy_val(sf.reshape(T, M_LAT), y, C, d, lams),
                         energy_grad_vec(sf.reshape(T, M_LAT), y, C, d,
                                         lams).reshape(-1))
        out = sp_min(fg, s0.reshape(-1), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=MAX_STEPS, maxfun=10 * MAX_STEPS))
        s_lb = out.x.reshape(T, M_LAT)
        res_lb = rel_res(s_lb, y, C, d, lams, g0n)
        row["lbfgs"] = dict(nfe=int(out.njev), res=res_lb,
                            wall=time.time() - t0,
                            E=energy_val(s_lb, y, C, d, lams),
                            false_converged=bool(out.njev < 50
                                                 and res_lb > 1e-3))
        # gd
        t0 = time.time()
        s_gd = solve_gd(y, C, d, lams, s0, MAX_STEPS)
        row["gd"] = dict(nfe=MAX_STEPS, res=rel_res(s_gd, y, C, d, lams, g0n),
                         wall=time.time() - t0,
                         E=energy_val(s_gd, y, C, d, lams),
                         false_converged=False)
        results.append(row)
        if tr < 5 or (tr + 1) % 10 == 0:
            print(f"  trial {tr:>2}: newton {row['newton']['nfe']} NFE "
                  f"(res {row['newton']['res']:.1e}) | "
                  f"lbfgs {row['lbfgs']['nfe']} NFE "
                  f"(res {row['lbfgs']['res']:.1e}"
                  f"{', FALSE-CONV' if row['lbfgs']['false_converged'] else ''})"
                  f" | gd res {row['gd']['res']:.1e}")

    # agreement gate: energies of converged arms should match
    En = np.array([r["newton"]["E"] for r in results])
    El = np.array([r["lbfgs"]["E"] for r in results])
    agree = np.abs(En - El) / np.maximum(np.abs(En), 1e-30)
    print("\n" + "=" * 78)
    print(f"newton NFE median {int(np.median([r['newton']['nfe'] for r in results]))} "
          f"wall med {np.median([r['newton']['wall'] for r in results])*1e3:.0f} ms")
    print(f"lbfgs  NFE median {int(np.median([r['lbfgs']['nfe'] for r in results]))} "
          f"false-converged {sum(r['lbfgs']['false_converged'] for r in results)}/{n_trials}")
    print(f"energy agreement |dE|/|E| median {np.median(agree):.2e} "
          f"max {agree.max():.2e}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch,
               config=dict(bin_ms=BIN_MS, n_trials=n_trials, M_lat=M_LAT,
                           lambdas=list(map(float, lams)),
                           target_res=TARGET_RES, max_steps=MAX_STEPS,
                           scope="inner-solve evaluation; PCA heuristic params"),
               results=results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "plds_mcmaze.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
