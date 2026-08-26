"""D4 — exact closed-loop stability theory in the smallest exact model.

One complex mode, quadratic loss, affine gradients: for every causal
estimator E the gradient map is exactly g_E(b) = M_E (b - b*), with M_E
a real 2x2 matrix computable to machine precision (no linearization
error). Learning b_{n+1} = b_n - eta g_E(b_n) is locally stable iff
rho(I - eta M_E) < 1.

The counterexample: the Wiener-optimal filter (best credit MSE) has
rho > 1 at a step size where worse credit estimators (online,
phase-only) are stable and converge.

Plus three analytic/measurement pieces:

  * delta-L analysis. One-step loss change for update direction d:
        dL = -eta g*^T d + (eta^2/2) d^T H d   (H = M_exact, exact).
    Credit-MSE minimization and next-step-loss minimization are
    DIFFERENT objectives; routeA's meta-objective is the latter. The
    table shows the MSE ranking != the dL ranking != the stability
    ranking.
  * Adam: per-coordinate normalization divides away |M|; the stable
    region then depends on orientation, not gain — the exact version of
    "gain is redundant with Adam". Verified by simulation.
  * Batch-noise channel: P(rho_batch > 1) per estimator.
  * D5: eta_max(estimator) vs |a| against the Hankel floor
    sigma = |a|/(1 - |a|^2).

Model (mirrors the rig's top layer): h_t = sum_k a^{t-k} b x_k, real
white x, c = 1, r_t = Re(h_t(b) - h_t(b*)), q_t = r_t, S_t = h_t(1),
G = sum_t conj(err_t) S_t, real gradient (Re G, -Im G).

Run:  python d4_stability.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

T = 128
NB_ENS = 64
STEPS = 300
SEED = 0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "d4_stability")
A_GRID = [0.5, 0.9, 0.95, 0.99, 0.995]


def rollout(a, b, x):
    h = np.zeros(T, complex)
    for t in range(1, T):
        h[t] = a * h[t - 1] + b * x[t]
    return h


def credit_seqs(a, b, bstar, x):
    r = (rollout(a, b, x) - rollout(a, bstar, x)).real
    lam = np.zeros(T, complex)
    nxt = 0j
    for t in range(T - 1, -1, -1):
        nxt = r[t] + np.conj(a) * nxt
        lam[t] = nxt
    return r, lam


def grad_of(err, S):
    G = np.sum(np.conj(err) * S)
    return np.array([G.real, -G.imag])


def wiener_fit_1mode(qs, lams, K):
    """Causal FIR: min_f sum_t |lam_t - f . z_t|^2, z_t = q_{t-K+1..t}.
    Normal equations: [sum conj(z) z^T] f = sum lam conj(z)."""
    num = np.zeros(K, complex)
    R = np.zeros((K, K), complex)
    for q, lam in zip(qs, lams):
        for t in range(K - 1, T):
            z = q[t - np.arange(K)]          # z[k] = q[t-k], matches apply_fir
            R += np.outer(np.conj(z), z)
            num += lam[t] * np.conj(z)
    return np.linalg.solve(R + 1e-9 * np.eye(K), num)


def apply_fir(q, f):
    K = len(f)
    out = np.zeros(T, complex)
    for k in range(K):
        out[k:] += f[k] * q[:T - k]
    return out


def M_matrix(a, bstar, x, err_fn):
    M = np.zeros((2, 2))
    for i, (re, im) in enumerate([(1.0, 0.0), (0.0, 1.0)]):
        q, lam = credit_seqs(a, bstar + re + 1j * im, bstar, x)
        err = err_fn(q, lam)
        M[:, i] = grad_of(err, rollout(a, 1.0, x))
    return M


def spectral_radius(M, eta):
    return max(abs(np.linalg.eigvals(np.eye(2) - eta * M)))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    bstar = 1.0 + 0.5j
    out = {}
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

        err_fns = {
            "exact": lambda q, lam: lam,
            "online": lambda q, lam: q,
            "wiener64": lambda q, lam: apply_fir(q, f64),
            "phase": lambda q, lam: phase * q,
        }
        mse = {}
        for name in err_fns:
            num = sum(np.sum(np.abs(lam - err_fns[name](q, lam)) ** 2)
                      for q, lam in zip(qs, lams))
            den = sum(np.sum(np.abs(lam) ** 2) for q, lam in zip(qs, lams))
            mse[name] = float(num / den)

        M = {name: np.mean([M_matrix(a, bstar, x, fn) for x in xs], axis=0)
             for name, fn in err_fns.items()}
        etas = np.logspace(-3, 1.5, 300)
        eta_max = {}
        for name, Mm in M.items():
            stab = np.array([spectral_radius(Mm, e) < 1 for e in etas])
            eta_max[name] = float(etas[np.argmax(~stab)]) \
                if not stab.all() else float("inf")

        eta_test = min(0.9 * eta_max["wiener64"],
                       0.5 * min(eta_max["online"], eta_max["phase"]))
        eta_test = float(max(eta_test, 1e-3))

        # GD + Adam simulations from b0 = 0
        b0 = np.array([0.0, 0.0])
        sims, sims_adam = {}, {}
        for name, fn in err_fns.items():
            b = b0.copy()
            for n in range(STEPS):
                x = rng.randn(T)
                bc = b[0] + 1j * b[1]
                q, lam = credit_seqs(a, bc, bstar, x)
                g = grad_of(fn(q, lam), rollout(a, 1.0, x))
                b = b - eta_test * g
            sims[name] = float(np.linalg.norm(b - np.array([bstar.real,
                                                            bstar.imag])))
            b = b0.copy()
            mm = np.zeros(2)
            vv = np.zeros(2)
            for n in range(1, STEPS + 1):
                x = rng.randn(T)
                bc = b[0] + 1j * b[1]
                q, lam = credit_seqs(a, bc, bstar, x)
                g = grad_of(fn(q, lam), rollout(a, 1.0, x))
                mm = 0.9 * mm + 0.1 * g
                vv = 0.999 * vv + 0.001 * g ** 2
                b = b - eta_test * (mm / (1 - 0.9 ** n)) / (
                    np.sqrt(vv / (1 - 0.999 ** n)) + 1e-8)
            sims_adam[name] = float(np.linalg.norm(
                b - np.array([bstar.real, bstar.imag])))

        p_unstable = {}
        for name, fn in err_fns.items():
            cnt = sum(spectral_radius(M_matrix(a, bstar, x, fn), eta_test)
                      > 1 for x in xs)
            p_unstable[name] = cnt / len(xs)

        # delta-L analysis at b0: dL = -eta g*^T d + (eta^2/2) d^T H d
        H = M["exact"]
        H = 0.5 * (H + H.T)                     # Hessian (symmetric part)
        gstar0 = M["exact"] @ (b0 - np.array([bstar.real, bstar.imag]))
        dl = {}
        for name, fn in err_fns.items():
            d = M[name] @ (b0 - np.array([bstar.real, bstar.imag]))
            num_ = gstar0 @ d
            den_ = d @ H @ d
            eta_opt = num_ / max(den_, 1e-300)
            dl[name] = dict(align=float(num_), curv=float(den_),
                            eta_opt=float(eta_opt),
                            dl_at_test=float(-eta_test * num_
                                             + 0.5 * eta_test ** 2 * den_),
                            dl_opt=float(-eta_opt * num_
                                         + 0.5 * eta_opt ** 2 * den_))

        out[amag] = dict(mse=mse, eta_max=eta_max, eta_test=eta_test,
                         sim_gd=sims, sim_adam=sims_adam,
                         p_unstable=p_unstable, dl=dl,
                         hankel=amag / (1 - amag ** 2))
        print(f"|a|={amag}: hankel {amag / (1 - amag ** 2):.1f}  "
              f"eta_test {eta_test:.4g}")
        for name in err_fns:
            print(f"   {name:<9s} mse {mse[name]:.3f}  eta_max "
                  f"{eta_max[name]:.4g}  gd|db| {sims[name]:.4g}  "
                  f"adam|db| {sims_adam[name]:.4g}  P(rho>1) "
                  f"{p_unstable[name]:.2f}  dL@test "
                  f"{dl[name]['dl_at_test']:.3g}  dL@opt "
                  f"{dl[name]['dl_opt']:.3g}", flush=True)

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, model="single complex mode, quadratic",
                       per_amag={str(k): v for k, v in out.items()}),
                  f, indent=2, default=float)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
