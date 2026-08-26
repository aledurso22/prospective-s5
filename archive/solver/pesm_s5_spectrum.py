"""PESM x S5 — the prospective solver meets the actual S5 spectrum.

Experiment A (research handoff discussion): the prospective mechanism survives
discretization ONLY in the solver/metric slot. PESM demonstrated that on
hand-picked real lambdas (../pesm/results/EXPERIMENTS.md). Here we test it on
the REAL S5/HiPPO spectrum: complex, oscillatory AND near-unit modes at once.

The chain energy whose equilibrium IS the S5 forward computation:

    E(s) = sum_t 1/2 |s_t - a s_{t-1} - b_t|^2  +  beta/2 |tanh s_t - u_t|^2

per diagonal complex mode a = Lambda_bar (the S5 bilinear multiplier, from
ssm/shared/hippo.py + the baseline layer's Tustin formula). At beta = 0 the
equilibrium is exactly the S5 rollout s_t = a s_{t-1} + b_t. The Gauss-Newton
Hessian is per-mode Hermitian tridiagonal with kappa ~ ((1+|a|)/(1-|a|))^2,
spanning ~1e2..1e7 across the spectrum — the stiffest regime any of the
experiments in either repo has touched.

Solvers (same arms as PESM):
    gamma=1  damped Newton in the GN-Hessian metric (prospective): each step
             is one Hermitian tridiagonal solve = THREE associative scans
             (Riccati/LDL^H elimination + forward + backward substitution).
    gamma=0  Euclidean gradient steps, size eta (control).
    gamma=1/2 the blended metric [gamma H + (1-gamma)/eta I].

The tridiagonal scan solver is ported from ../pesm/ssm/pesm.py and generalized
to complex off-diagonals (LDL^H instead of LDL^T: |sub|^2 pivots, conj in the
backward substitution). Self-contained: the cluster only has this repo.

GATES (hard asserts):
  G1  scan solve == numpy dense solve on random Hermitian SPD tridiagonal
  G2  beta=0: ONE prospective (gamma=1) step from s0=0 is the exact
      equilibrium (rel residual < 1e-8) AND equals the first-order S5
      rollout s_t = a s_{t-1} + b_t (rel err < 1e-8) — the PESM<->S5
      consistency statement on the real spectrum
  G3  the same one step leaves the gamma=0 arm orders of magnitude behind

Run:  python pesm_s5_spectrum.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ssm.shared.hippo import hippo_init

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_MODES = 64          # HiPPO state size (the repo's default N)
T = 784               # chain length = sMNIST length, on purpose
DELTAS = [1e-3, 1e-2, 1e-1]     # S5 log_step init range is [1e-3, 1e-1]
BETAS = [0.0, 0.5]              # 0 = pure quadratic (S5 chain), 0.5 = anchored
GAMMAS = [0.0, 0.5, 1.0]        # 0 = Euclidean control, 1 = prospective
ETA = 0.25
K_LIST = [1, 2, 4, 8, 16, 32]
SEED = 0
GATE_TOL = 1e-8

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


# ---------------------------------------------------------------------------
# Hermitian tridiagonal solve via 3 associative scans
# (ported from ../pesm/ssm/pesm.py; generalized to complex sub-diagonal)
# ---------------------------------------------------------------------------

def _mobius_op(Mi, Mj):
    """Compose projective 2x2 Riccati maps (j later); renormalized."""
    N = jnp.einsum("...ij,...jk->...ik", Mj, Mi)
    n = jnp.max(jnp.abs(N), axis=(-2, -1), keepdims=True)
    return N / jnp.where(n > 0, n, 1.0)


def _aff_op(qi, qj):
    """Compose first-order affine maps s = a*s + b (j later)."""
    ai, bi = qi
    aj, bj = qj
    return aj * ai, aj * bi + bj


def tridiag_hermitian_scan(diag, sub, g):
    """Solve H x = g along axis 0: H Hermitian tridiagonal, SPD.

    Args:
        diag: (T, ...) real diagonal (positive).
        sub:  (T, ...) complex sub-diagonal; sub[0] ignored (sub[t] = H[t, t-1]).
        g:    (T, ...) complex right-hand side.

    LDL^H elimination with pivot d_t = diag_t - |sub_t|^2 / d_{t-1} (real
    Riccati), w_t = sub_t / d_{t-1}; backward substitution uses conj(w).
    """
    piv2 = -(jnp.abs(sub) ** 2)                                # real
    M = jnp.stack([jnp.stack([diag, piv2], axis=-1),
                   jnp.stack([jnp.ones_like(diag), jnp.zeros_like(diag)],
                             axis=-1)], axis=-2)               # (T,...,2,2)
    Nm = jax.lax.associative_scan(_mobius_op, M, axis=0)
    d = Nm[..., 0, 0] / Nm[..., 1, 0]                          # real pivots

    d_prev = jnp.concatenate([jnp.ones_like(d[:1]), d[:-1]], axis=0)
    _, v = jax.lax.associative_scan(_aff_op, (-sub / d_prev, g), axis=0)
    y = v / d

    w_next = jnp.concatenate(
        [jnp.conj(sub[1:]), jnp.zeros_like(sub[:1])], axis=0)
    _, x_rev = jax.lax.associative_scan(
        _aff_op, (jnp.flip(-w_next / d, axis=0), jnp.flip(y, axis=0)), axis=0)
    return jnp.flip(x_rev, axis=0)


# ---------------------------------------------------------------------------
# Energy, gradient, solver steps (complex chains, all ops elementwise over
# the mode axis; time is axis 0)
# ---------------------------------------------------------------------------

def energy_grad(s, a, b, u, beta):
    """g_t = dE/dconj(s_t) for E = sum 1/2|r|^2 + beta/2|tanh s - u|^2,
    r_t = s_t - a s_{t-1} - b_t."""
    s_prev = jnp.concatenate([jnp.zeros_like(s[:1]), s[:-1]], axis=0)
    r = s - a * s_prev - b
    r_next = jnp.concatenate([r[1:], jnp.zeros_like(r[:1])], axis=0)
    g = r - jnp.conj(a) * r_next
    if beta > 0:
        z = jnp.tanh(s)
        g = g + beta * (z - u) * jnp.conj(1 - z ** 2)
    return g


def solver_trajectory(a, b, u, beta, gamma, eta, K, s0):
    """K solver steps from s0; returns (s_K, rel residuals), where entry k
    is the relative residual BEFORE step k — i.e. rel[k] is the residual of
    s_k, so the residual after K steps is rel[K].

    gamma=1: damped Newton in the GN metric (one Hermitian tridiag solve per
    step). gamma=0: gradient steps s <- s - eta g. Blends solve
    [gamma H + (1-gamma)/eta I] d = g.
    """
    g0 = energy_grad(s0, a, b, u, beta)
    g0n = jnp.linalg.norm(g0, axis=0)

    def step(s, _):
        g = energy_grad(s, a, b, u, beta)
        rel = jnp.linalg.norm(g, axis=0) / g0n
        if gamma == 0.0:
            return s - eta * g, rel
        z = jnp.tanh(s)
        w = jnp.abs(1 - z ** 2) ** 2                       # GN anchor diag
        dg = 1 + jnp.abs(a) ** 2 + beta * w
        dg = dg.at[-1].set(1 + beta * w[-1])               # last row: no r_{T}
        sub = jnp.full_like(s, -a).at[0].set(0.0)
        A = gamma * dg + (1 - gamma) / eta
        off = gamma * sub
        return s - tridiag_hermitian_scan(A, off, g), rel

    return jax.lax.scan(step, s0, None, length=K)


solve_cell = jax.jit(solver_trajectory, static_argnums=(3, 4, 5, 6))


# ---------------------------------------------------------------------------
# S5 spectrum
# ---------------------------------------------------------------------------

def s5_multipliers(Delta: float) -> np.ndarray:
    """The actual S5 bilinear spectrum at step Delta (Tustin, as in
    ssm/baseline_s5/layer.py: discretize_bilinear)."""
    Lam, _, _ = hippo_init(N_MODES)
    Lam = np.asarray(Lam, np.complex128)
    return (1.0 + 0.5 * Delta * Lam) / (1.0 - 0.5 * Delta * Lam)


def kappa(a: np.ndarray) -> np.ndarray:
    return ((1 + np.abs(a)) / (1 - np.abs(a))) ** 2


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_G1_dense():
    rng = np.random.RandomState(0)
    Tt, M = 50, 4
    dg = rng.rand(Tt, M) + 2.0
    sub = (rng.randn(Tt, M) + 1j * rng.randn(Tt, M)) * 0.4
    sub[0] = 0
    g = rng.randn(Tt, M) + 1j * rng.randn(Tt, M)
    x = tridiag_hermitian_scan(jnp.asarray(dg), jnp.asarray(sub),
                               jnp.asarray(g))
    err = 0.0
    for m in range(M):
        H = (np.diag(dg[:, m]) + np.diag(sub[1:, m], -1)
             + np.diag(np.conj(sub[1:, m]), 1))
        xd = np.linalg.solve(H, g[:, m])
        err = max(err, np.linalg.norm(np.asarray(x)[:, m] - xd)
                  / np.linalg.norm(xd))
    print(f"  G1 scan==dense tridiag solve      rel err = {err:.3e}  "
          f"{'PASS' if err < 1e-10 else 'FAIL'}")
    assert err < 1e-10


def gate_G2_G3_one_step(a, b, u):
    """beta=0: one prospective step = exact equilibrium = S5 rollout."""
    s0 = jnp.zeros_like(b)
    s1, rel1 = solve_cell(jnp.asarray(a), b, u, 0.0, 1.0, ETA, 2, s0)
    rel = float(jnp.max(rel1[1]))                      # residual after 1 step
    # reference rollout s_t = a s_{t-1} + b_t, sequential numpy
    bn = np.asarray(b)
    an = np.asarray(a)[None, :]
    sref = np.zeros_like(bn)
    sp = np.zeros(bn.shape[1], np.complex128)
    for t in range(bn.shape[0]):
        sp = an[0] * sp + bn[t]
        sref[t] = sp
    err = float(np.max(np.abs(np.asarray(s1) - sref))
                / np.max(np.abs(sref)))
    print(f"  G2 one Newton step == exact eq    rel residual = {rel:.3e}  "
          f"{'PASS' if rel < GATE_TOL else 'FAIL'}")
    print(f"  G2 equilibrium == S5 rollout      rel err = {err:.3e}  "
          f"{'PASS' if err < GATE_TOL else 'FAIL'}")
    assert rel < GATE_TOL and err < GATE_TOL
    # G3: the gamma=0 arm after ONE step is orders of magnitude behind
    _, rel0 = solve_cell(jnp.asarray(a), b, u, 0.0, 0.0, ETA, 2, s0)
    r0 = float(jnp.max(rel0[1]))
    print(f"  G3 gamma=0 after 1 step           rel residual = {r0:.3f}  "
          f"(vs {rel:.1e} prospective)")
    assert r0 > 1e3 * rel


# ---------------------------------------------------------------------------
# Solver showdown: prospective Newton vs the field's solvers on the S5
# spectrum. Picard/Anderson/Broyden ported from prospective-deq/solvers.py
# under its uniform NFE contract: 1 gradient evaluation per step per arm
# (nfe = K + 1 counting the initial residual); the Newton arm's tridiagonal
# solves (3 scans each) are logged as structure, not hidden. The state is
# treated as ONE flat (T*N)-dim problem instance, as a DEQ user would.
# ---------------------------------------------------------------------------

def _flat_energy_grad(a, b, u, beta):
    shape = b.shape
    aj, bj, uj = jnp.asarray(a), jnp.asarray(b), jnp.asarray(u)

    def gfun(sf):
        return np.asarray(energy_grad(jnp.asarray(sf.reshape(shape)),
                                      aj, bj, uj, beta)).reshape(-1)
    return gfun


def anderson_solve(gfun, s0, K, m=5, eta=ETA):
    """Anderson mixing (depth m) on the gradient map f(s) = s - eta*gradE.
    Plain numpy loop: standalone solver comparison, no tracing needed."""
    D = s0.size
    S = np.zeros((m, D))
    G = np.zeros((m, D))
    s = s0.copy()
    for k in range(K):
        g = eta * gfun(s)                  # residual s - f(s)
        S = np.roll(S, -1, axis=0)
        S[-1] = s
        G = np.roll(G, -1, axis=0)
        G[-1] = g
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
    return s


def broyden_solve(gfun, s0, K, m=5):
    """Limited-memory Broyden root-find on gradE = 0 (Bai et al. DEQ style):
    B = I + U V^T with m secant updates, Woodbury step (one m x m solve).
    Plain numpy loop."""
    D = s0.size
    s = s0.copy()
    g_prev = np.zeros(D)
    ds = np.zeros(D)
    U = np.zeros((m, D))
    V = np.zeros((m, D))
    for k in range(K):
        g = gfun(s)
        dg = g - g_prev
        B_ds = ds + U.T @ (V @ ds)
        denom = float(np.dot(ds, ds))
        if denom > 1e-24:
            U = np.roll(U, -1, axis=0)
            U[-1] = (dg - B_ds) / denom
            V = np.roll(V, -1, axis=0)
            V[-1] = ds
        Mm = np.eye(m) + V @ U.T
        p = g - U.T @ np.linalg.solve(Mm, V @ g)
        s_new = s - p
        g_prev, ds, s = g, s_new - s, s_new
    return s


def solver_showdown():
    """Residual vs NFE for all four solvers, on the real S5 spectrum."""
    print("\n[solver showdown] rel residual ||gradE||/||gradE(s0)|| "
          "after K steps (NFE = K+1, uniform contract)")
    print("  (newton also performs K tridiagonal solves = 3K scans — "
          "logged as structure)")
    cells = [(1e-2, 0.0), (1e-2, 0.5), (0.1, 0.5)]
    out = []
    for Delta, beta in cells:
        a = s5_multipliers(Delta)
        rng = np.random.RandomState(3)
        b, u = make_drivers(a, beta, rng)
        gfun = _flat_energy_grad(a, np.asarray(b), np.asarray(u), beta)
        s0 = np.zeros(b.size)
        g0n = np.linalg.norm(gfun(s0))
        row = dict(Delta=Delta, beta=beta,
                   kappa_max=float(np.max(kappa(a))), arms={})
        for arm in ["newton", "gd", "anderson", "broyden"]:
            residuals = {}
            for K in K_LIST:
                if arm == "newton":
                    sK, _ = solve_cell(jnp.asarray(a), b, u, beta, 1.0, ETA,
                                       K, jnp.zeros_like(b))
                elif arm == "gd":
                    sK, _ = solve_cell(jnp.asarray(a), b, u, beta, 0.0, ETA,
                                       K, jnp.zeros_like(b))
                elif arm == "anderson":
                    sK = anderson_solve(gfun, s0, K)
                else:
                    sK = broyden_solve(gfun, s0, K)
                res = float(np.linalg.norm(gfun(np.asarray(sK).reshape(-1)))
                            / g0n)
                residuals[K] = res
            row["arms"][arm] = residuals
        out.append(row)
        print(f"\n  Delta={Delta:g} beta={beta} "
              f"(kappa_max={row['kappa_max']:.2e})")
        for arm in ["newton", "gd", "anderson", "broyden"]:
            r = row["arms"][arm]
            print(f"    {arm:<9s} " + "  ".join(f"K={K}: {r[K]:.2e}"
                                                for K in K_LIST))
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def make_drivers(a, beta, rng):
    """b scaled by (1-|a|) so equilibrium states are O(1) on every mode;
    u small complex anchors."""
    mag = np.abs(a)[None, :]
    b = (rng.randn(T, N_MODES) + 1j * rng.randn(T, N_MODES)) / np.sqrt(2)
    b = b * (1 - mag)
    b = jnp.asarray(b)
    if beta > 0:
        u = 0.3 * (rng.randn(T, N_MODES) + 1j * rng.randn(T, N_MODES))
        u = jnp.asarray(u / np.sqrt(2))
    else:
        u = jnp.zeros_like(b)
    return b, u


def run_cell(a, Delta, beta, rng):
    b, u = make_drivers(a, beta, rng)
    s0 = jnp.zeros_like(b)
    cell = dict(kappa_max=float(np.max(kappa(a))),
                kappa_med=float(np.median(kappa(a))),
                gammas={})
    for gamma in GAMMAS:
        _, rel = solve_cell(jnp.asarray(a), b, u, beta, gamma, ETA,
                            max(K_LIST) + 1, s0)
        rel = np.asarray(rel)                              # (K+1, N)
        cell["gammas"][gamma] = dict(
            median={K: float(np.median(rel[K])) for K in K_LIST},
            worst={K: float(np.max(rel[K])) for K in K_LIST})
    # phase buckets: near-real modes vs oscillatory modes (the S5 novelty)
    phase = np.abs(np.angle(a))
    osc = phase >= np.pi / 4
    out = {}
    for gamma in (0.0, 1.0):
        _, rel = solve_cell(jnp.asarray(a), b, u, beta, gamma, ETA, 5, s0)
        rn = np.asarray(rel[4])
        out[gamma] = dict(mem=float(np.median(rn[~osc])),
                          osc=float(np.median(rn[osc])))
    cell["residual_at_K4_by_phase"] = out
    print(f"  Delta={Delta:g} beta={beta}  "
          f"kappa_max={cell['kappa_max']:.2e} kappa_med={cell['kappa_med']:.2e}")
    for gamma in GAMMAS:
        g = cell["gammas"][gamma]
        print(f"    gamma={gamma:<4} rel residual (median): "
              + "  ".join(f"K={K}: {g['median'][K]:.2e}" for K in K_LIST)
              + f"   worst@32: {g['worst'][32]:.2e}")
    print(f"    K=4 by phase: gamma=1 mem {out[1.0]['mem']:.2e} / "
          f"osc {out[1.0]['osc']:.2e}   vs   gamma=0 mem "
          f"{out[0.0]['mem']:.2e} / osc {out[0.0]['osc']:.2e}")
    return cell


def main() -> None:
    print("=" * 78)
    print("PESM x S5 — prospective solver on the real S5/HiPPO spectrum")
    print("=" * 78)
    rng = np.random.RandomState(SEED)

    print("\n[gates]")
    gate_G1_dense()
    a0 = s5_multipliers(1e-2)
    b0, u0 = make_drivers(a0, 0.0, np.random.RandomState(1))
    gate_G2_G3_one_step(a0, b0, u0)

    print("\n[sweep]")
    cells = []
    for Delta in DELTAS:
        a = s5_multipliers(Delta)
        for beta in BETAS:
            cell = run_cell(a, Delta, beta, rng)
            cells.append(dict(Delta=Delta, beta=beta, **cell))

    showdown = solver_showdown()

    # the long-run contrast: stiffest cell, gamma=0 to K=4096
    a = s5_multipliers(1e-2)
    b, u = make_drivers(a, 0.5, np.random.RandomState(2))
    _, rel_long = solve_cell(jnp.asarray(a), b, u, 0.5, 0.0, ETA, 4097,
                             jnp.zeros_like(b))
    rl = np.asarray(rel_long)
    print(f"\n  long control run (Delta=1e-2, beta=0.5, gamma=0): "
          f"median rel residual K=32: {np.median(rl[32]):.3e}  "
          f"K=1024: {np.median(rl[1024]):.3e}  "
          f"K=4096: {np.median(rl[4096]):.3e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch, seed=SEED,
               config=dict(N_modes=N_MODES, T=T, deltas=DELTAS, betas=BETAS,
                           gammas=GAMMAS, eta=ETA, K_list=K_LIST,
                           dtype="float64/complex128"),
               long_control=dict(Delta=1e-2, beta=0.5, gamma=0.0,
                                 median_K32=float(np.median(rl[32])),
                                 median_K1024=float(np.median(rl[1024])),
                                 median_K4096=float(np.median(rl[4096]))),
               solver_showdown=showdown,
               cells=cells)
    path = os.path.join(RESULTS_DIR, "pesm_s5_spectrum.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print("\n" + "=" * 78)
    print(f"ALL GATES PASSED — wrote {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
