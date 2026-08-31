"""B37b -- trainability of the universal quotient chart.

Question: not correctness or expressivity (both settled in B37a, frozen), but
whether (q, u, B, C) can actually be LEARNED stably from data.

Model trained (the ONLY parameterization used; no new bases, no factorization,
no ProductLocal, no regularization, no stability projection, no FFT):
    z_{t+1} = u_theta(C(a)) z_t + B x_t,     y_hat_t = C z_t
Trainable: a (r), theta (r), B (r), C (r). u_theta(C_q) is formed with the
FROZEN B37a `mult_matrix` (rem(u * x^i, q) columnwise), which B37a verified
equals sum_k theta_k C_q^k. Ordinary offline BPTT/autodiff only -- no online
updates, to isolate chart conditioning from the RTRL algorithm.

Scoring is INPUT-OUTPUT equivalence, never recovery of the teacher's A.

Initialization arms:
  A. constructive  : (q,u,B,C) built from a known exact equivalent realization
                     of the teacher (loss ~0 at init).
  B. perturbed     : arm A plus relative noise of size EPS_PERTURB.
  C. generic stable: u(x)=x and q with random stable roots (so u(C_q)=C_q is
                     stable at init), B, C random. Uses only the existing
                     architecture -- it is a choice of initial values, not a
                     new parameterization.
"""
from __future__ import annotations

import functools
import json
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.b37a_universal_quotient import mult_matrix, companion   # FROZEN B37a primitives

R_VALUES = (4, 8)
N_TRAIN_SEQ, N_VAL_SEQ, N_TEST_SEQ = 32, 16, 16
T_SEQ = 64
N_STEPS = 400
LR_GRID = (3e-4, 1e-3, 3e-3)
EVAL_SEEDS = (0, 1, 2)
EPS_LADDER = (1e-6, 1e-4, 1e-2)
K_MARKOV = 40
DIVERGE = 1e12

FAMILIES = ("random_stable_diag", "distinct_real", "complex_conjugate", "repeated_poles",
            "exact_jordan", "multi_jordan_shared", "nearly_defective", "nonnormal", "stiff")


# ======================================================================
# Teachers: built from KNOWN Jordan data so the exact realization is
# constructible for arms A/B. blocks = [(lambda (may be complex), n), ...]
# with complex eigenvalues appearing in conjugate pairs.
# ======================================================================
def spec_for(family, r, rng):
    if family == "random_stable_diag":
        vals = rng.uniform(-0.9, 0.9, size=r)
        return [(float(v), 1) for v in vals], 1.0
    if family == "distinct_real":
        vals = np.linspace(-0.85, 0.85, r) + rng.randn(r) * 0.01
        return [(float(v), 1) for v in vals], 1.0
    if family == "complex_conjugate":
        blocks = []
        for _ in range(r // 2):
            mod, ang = rng.uniform(0.4, 0.9), rng.uniform(0.4, 2.6)
            z = mod * np.exp(1j * ang)
            blocks += [(complex(z), 1), (complex(np.conj(z)), 1)]
        while len(blocks) < r:
            blocks.append((float(rng.uniform(-0.8, 0.8)), 1))
        return blocks, 1.0
    if family == "repeated_poles":                      # repeated eigenvalues, SEMISIMPLE
        nd = max(1, r // 2)
        vals = rng.uniform(-0.85, 0.85, size=nd)
        reps = [vals[i % nd] for i in range(r)]
        return [(float(v), 1) for v in reps], 1.0
    if family == "exact_jordan":                        # ONE Jordan block of size r
        return [(float(rng.uniform(0.5, 0.85)), r)], 1.0
    if family == "multi_jordan_shared":                 # several blocks, SAME eigenvalue
        lam = float(rng.uniform(0.5, 0.85))
        part = [2] * (r // 2) if r % 2 == 0 else [3] + [2] * ((r - 3) // 2)
        return [(lam, n) for n in part], 1.0
    if family == "nearly_defective":                    # split a Jordan block by eps
        lam, eps = float(rng.uniform(0.5, 0.85)), 1e-6
        return [(lam + i * eps, 1) for i in range(r)], 1e4      # + ill-conditioned S
    if family == "nonnormal":                           # separated spectrum, very skewed basis
        vals = np.linspace(0.2, 0.9, r)
        return [(float(v), 1) for v in vals], 1e6
    if family == "stiff":
        mags = np.logspace(-3, np.log10(0.99), r)
        sg = rng.choice([-1.0, 1.0], size=r)
        return [(float(m * s), 1) for m, s in zip(mags, sg)], 1.0
    raise ValueError(family)


def jordan_from_blocks(blocks):
    """Complex Jordan form in block order."""
    r = sum(n for _, n in blocks)
    J = np.zeros((r, r), dtype=complex)
    o = 0
    for lam, n in blocks:
        J[o:o + n, o:o + n] = lam * np.eye(n) + (np.diag(np.ones(n - 1), 1) if n > 1 else 0)
        o += n
    return J


def real_jordan_and_pi(blocks):
    """Real Jordan form J_real (REAL matrix) and complex Pi with J_real @ Pi = Pi @ J.

    Complex eigenvalues must appear as ADJACENT conjugate pairs of 1x1 blocks
    (which is what `spec_for` produces); asserted rather than assumed.
    """
    r = sum(n for _, n in blocks)
    Jr = np.zeros((r, r))
    Pi = np.zeros((r, r), dtype=complex)
    o, i, p = 0, 0, len(blocks)
    while i < p:
        lam, n = blocks[i]
        if abs(np.imag(complex(lam))) > 1e-14:
            assert n == 1 and i + 1 < p, "complex blocks must be 1x1 and paired"
            lam2, n2 = blocks[i + 1]
            assert n2 == 1 and abs(complex(lam2) - np.conj(complex(lam))) < 1e-12
            al, be = float(np.real(lam)), float(np.imag(lam))
            Jr[o:o + 2, o:o + 2] = [[al, -be], [be, al]]
            Pi[o:o + 2, o:o + 2] = [[1.0, 1.0], [-1j, 1j]]
            o += 2; i += 2
        else:
            Jr[o:o + n, o:o + n] = np.real(lam) * np.eye(n) + (np.diag(np.ones(n - 1), 1) if n > 1 else 0)
            Pi[o:o + n, o:o + n] = np.eye(n)
            o += n; i += 1
    return Jr, Pi


def make_similarity(r, skew, rng):
    """Random invertible S; `skew` inflates cond(S) to make A nonnormal."""
    Q1, _ = np.linalg.qr(rng.randn(r, r))
    Q2, _ = np.linalg.qr(rng.randn(r, r))
    s = np.logspace(0, np.log10(max(skew, 1.0)), r)
    return Q1 @ np.diag(s) @ Q2


def make_teacher(family, r, seed):
    """A = S J_real S^-1 with S and J_real REAL, so A is exactly real.
    S_c = S @ Pi is the complex similarity to the complex Jordan form J,
    used only by the constructive initialization."""
    rng = np.random.RandomState(1000 + seed)
    blocks, skew = spec_for(family, r, rng)
    J = jordan_from_blocks(blocks)
    Jr, Pi = real_jordan_and_pi(blocks)
    S = make_similarity(r, skew, rng)
    A = S @ Jr @ np.linalg.inv(S)
    imag = float(np.max(np.abs(np.imag(A))))
    A = np.real(A)
    Sc = S.astype(complex) @ Pi
    resid = float(np.linalg.norm(A @ Sc - Sc @ J) / (1 + np.linalg.norm(A)))
    Bs, Cs = rng.randn(r), rng.randn(r)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return dict(A=A, Bs=Bs, Cs=Cs, blocks=blocks, S=Sc, rho=rho,
                condS=float(np.linalg.cond(Sc)), imag_leak=imag, resid_ASc=resid)


# ======================================================================
# Constructive exact realization (extends the B37a construction to
# complex-conjugate spectra; B37a itself is untouched).
# ======================================================================
def hermite_qu(blocks):
    """Distinct alphas, CONJUGATE-PAIRED wherever the lambdas are conjugate
    (so q and u come out real), plus Hermite-interpolated u."""
    r = sum(n for _, n in blocks)
    p = len(blocks)
    base = np.linspace(0.25, 0.85, p) if p > 1 else np.array([0.4])
    alphas = [None] * p
    assigned = [False] * p
    for i in range(p):
        if assigned[i]:
            continue
        lam, n = blocks[i]
        if abs(np.imag(complex(lam))) > 1e-14:
            j = next((k for k in range(i + 1, p)
                      if (not assigned[k]) and blocks[k][1] == n
                      and abs(complex(blocks[k][0]) - np.conj(complex(lam))) < 1e-12), None)
            if j is None:
                raise ValueError(f"complex block {i} has no conjugate partner")
            alphas[i] = complex(base[i], 0.11 + 0.037 * i)
            alphas[j] = np.conj(alphas[i])
            assigned[i] = assigned[j] = True
        else:
            alphas[i] = complex(base[i], 0.0)
            assigned[i] = True
    roots = []
    for (lam, n), al in zip(blocks, alphas):
        roots += [al] * n
    q_desc = np.poly(np.array(roots))
    a = q_desc[::-1][:r].copy()

    rows, rhs = [], []
    for (lam, n), al in zip(blocks, alphas):
        rows.append([al ** k for k in range(r)]); rhs.append(complex(lam))
        if n >= 2:
            rows.append([0.0 if k == 0 else k * al ** (k - 1) for k in range(r)]); rhs.append(1.0 + 0j)
    theta, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return a, theta, alphas, q_desc


def build_T_complex(blocks, alphas, q_desc, theta, r):
    import math
    cols = []
    for (lam, n), al in zip(blocks, alphas):
        cur, gs = q_desc.astype(complex).copy(), []
        for _ in range(n):
            out = [cur[0]]
            for c in cur[1:]:
                out.append(c + out[-1] * al)
            cur = np.array(out[:-1])
            v = np.zeros(r, dtype=complex); asc = cur[::-1]; v[:len(asc)] = asc
            gs.append(v)
        V = np.stack(gs, axis=1)
        P = np.zeros((n, n), dtype=complex)
        for j in range(1, n):
            c = np.polyval(np.polyder(np.poly1d(theta[::-1]), j), al) / math.factorial(j)
            P += c * np.diag(np.ones(n - j), j)
        e = np.zeros(n, dtype=complex); e[-1] = 1.0
        Wc, w = [], e.copy()
        for _ in range(n):
            Wc.append(w.copy()); w = P @ w
        cols.append(V @ np.stack(Wc[::-1], axis=1))
    return np.concatenate(cols, axis=1)


def constructive_realization(teacher, r):
    """Returns (a, theta, B, C) reproducing the teacher's I/O exactly, plus diagnostics."""
    blocks, S, A = teacher["blocks"], teacher["S"], teacher["A"]
    a_c, theta_c, alphas, q_desc = hermite_qu(blocks)
    Tc = build_T_complex(blocks, alphas, q_desc, theta_c, r)
    a = a_c.real.copy(); theta = theta_c.real.copy()
    a_imag = float(np.max(np.abs(a_c.imag))); th_imag = float(np.max(np.abs(theta_c.imag)))

    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))
    T_full = S @ np.linalg.inv(Tc)                    # A T = T M  (complex in general)
    best = None
    for c1, c2 in ((1, 0), (0, 1), (1, 1), (1, -1), (0.5, 1), (1, 0.5)):
        Tr = c1 * T_full.real + c2 * T_full.imag
        if np.linalg.matrix_rank(Tr) == r:
            cond = np.linalg.cond(Tr)
            if best is None or cond < best[0]:
                best = (cond, Tr)
    condT, T = best
    resid = float(np.linalg.norm(A @ T - T @ M) / (1 + np.linalg.norm(A) * np.linalg.norm(T)))
    B = np.linalg.solve(T, teacher["Bs"])
    C = teacher["Cs"] @ T
    return (a, theta, B, C), dict(resid_AT_TM=resid, condT=float(condT),
                                   a_imag=a_imag, theta_imag=th_imag)


def generic_stable_init(r, seed):
    """u(x)=x, q with random stable roots => u(C_q)=C_q stable. B, C random."""
    rng = np.random.RandomState(7000 + seed)
    roots = []
    while len(roots) < r:
        if len(roots) + 1 < r and rng.rand() < 0.4:
            mod, ang = rng.uniform(0.3, 0.85), rng.uniform(0.4, 2.6)
            roots += [mod * np.exp(1j * ang), mod * np.exp(-1j * ang)]
        else:
            roots.append(rng.uniform(-0.85, 0.85))
    a = np.poly(np.array(roots[:r]))[::-1][:r].real.copy()
    theta = np.zeros(r); theta[1 if r > 1 else 0] = 1.0
    return (a, theta, rng.randn(r) / np.sqrt(r), rng.randn(r) / np.sqrt(r))


# ======================================================================
# Model / training (plain BPTT).
# ======================================================================
def student_rollout(params, xs, r):
    a, theta, B, C = params
    M = mult_matrix(theta, a, r)

    def step(z, x):
        z_next = M @ z + B * x
        return z_next, C @ z_next
    z0 = jnp.zeros(r, dtype=jnp.float64)
    _, ys = jax.lax.scan(step, z0, xs)
    return ys


def student_rollout_states(params, xs, r):
    a, theta, B, C = params
    M = mult_matrix(theta, a, r)

    def step(z, x):
        z_next = M @ z + B * x
        return z_next, z_next
    z0 = jnp.zeros(r, dtype=jnp.float64)
    _, zs = jax.lax.scan(step, z0, xs)
    return zs


def teacher_outputs(teacher, xs):
    A, Bs, Cs = teacher["A"], teacher["Bs"], teacher["Cs"]
    h = np.zeros(A.shape[0])
    out = []
    for x in np.asarray(xs):
        h = A @ h + Bs * x
        out.append(Cs @ h)
    return np.array(out)


def make_dataset(teacher, n_seq, T, seed):
    rng = np.random.RandomState(seed)
    xs = rng.randn(n_seq, T) * 0.5
    ys = np.stack([teacher_outputs(teacher, xs[i]) for i in range(n_seq)])
    return jnp.array(xs), jnp.array(ys)


def batched_loss(params, xs, ys, r):
    pred = jax.vmap(lambda x: student_rollout(params, x, r))(xs)
    return jnp.mean((pred - ys) ** 2)


def markov_error(params, teacher, r, K=K_MARKOV):
    a, theta, B, C = [np.asarray(p) for p in params]
    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))
    A, Bs, Cs = teacher["A"], teacher["Bs"], teacher["Cs"]
    vs, vt, worst = B.copy(), Bs.copy(), 0.0
    for _ in range(K):
        gs, gt = float(C @ vs), float(Cs @ vt)
        if not np.isfinite(gs):
            return float("inf")
        worst = max(worst, abs(gs - gt) / (1 + abs(gt)))
        vs, vt = M @ vs, A @ vt
    return float(worst)


CHUNK = 10


@functools.lru_cache(maxsize=None)
def _build(r):
    """Compiled once per r; shapes are constant across the sweep."""
    def loss_of(p, xs, ys):
        return batched_loss(p, xs, ys, r)

    def chunk_step(carry, _):
        params, m, v, t = carry
        loss, g = jax.value_and_grad(loss_of)(params, carry_xs[0], carry_xs[1])
        return carry, loss

    def run_chunk(params, m, v, t0, xs, ys, lr):
        def one(carry, _):
            params, m, v, t = carry
            loss, g = jax.value_and_grad(loss_of)(params, xs, ys)
            t = t + 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, g))
            v = tuple(b2 * vi + (1 - b2) * gi ** 2 for vi, gi in zip(v, g))
            mh = tuple(mi / (1 - b1 ** t) for mi in m)
            vh = tuple(vi / (1 - b2 ** t) for vi in v)
            params = tuple(pp - lr * mhi / (jnp.sqrt(vhi) + eps)
                           for pp, mhi, vhi in zip(params, mh, vh))
            gn = jnp.stack([jnp.linalg.norm(gi) for gi in g])
            return (params, m, v, t), (loss, gn)
        (params, m, v, t), (losses, gns) = jax.lax.scan(
            one, (params, m, v, t0), None, length=CHUNK)
        return params, m, v, t, losses, gns
    return jax.jit(run_chunk), jax.jit(loss_of)


_DATA_CACHE = {}


def get_data(teacher, family, r, seed):
    key = (family, r, seed)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = (make_dataset(teacher, N_TRAIN_SEQ, T_SEQ, 20000 + seed),
                            make_dataset(teacher, N_VAL_SEQ, T_SEQ, 30000 + seed),
                            make_dataset(teacher, N_TEST_SEQ, T_SEQ, 40000 + seed))
    return _DATA_CACHE[key]


def diagnostics(params, teacher, r, xs_te):
    a, theta, B, C = [np.asarray(pp) for pp in params]
    if not np.all(np.isfinite(np.concatenate([a, theta, B, C]))):
        return dict(max_z=float("inf"), rho=float("inf"), condM=float("inf"),
                    markov=float("inf"))
    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))
    zs = np.asarray(jax.vmap(lambda x: student_rollout_states(params, x, r))(xs_te))
    max_z = float(np.max(np.abs(zs))) if np.all(np.isfinite(zs)) else float("inf")
    return dict(max_z=max_z,
                rho=float(np.max(np.abs(np.linalg.eigvals(M)))),
                condM=float(np.linalg.cond(M)),
                markov=markov_error(params, teacher, r))


def train_one(teacher, params0, family, r, lr, seed, n_steps=N_STEPS):
    (xs_tr, ys_tr), (xs_va, ys_va), (xs_te, ys_te) = get_data(teacher, family, r, seed)
    run_chunk, loss_of = _build(r)

    params = tuple(jnp.array(pp) for pp in params0)
    m = tuple(jnp.zeros_like(pp) for pp in params)
    v = tuple(jnp.zeros_like(pp) for pp in params)
    t = jnp.array(0.0)

    init_val = float(loss_of(params, xs_va, ys_va))
    init_diag = diagnostics(params, teacher, r, xs_te)
    best_val, best_params = init_val, params
    gsum, gcount, diverged, div_step = np.zeros(4), 0, False, None

    for c in range(n_steps // CHUNK):
        params, m, v, t, losses, gns = run_chunk(params, m, v, t, xs_tr, ys_tr, lr)
        losses = np.asarray(losses)
        gns = np.asarray(gns)
        ok = np.isfinite(gns).all(axis=1) & (np.abs(gns) < DIVERGE).all(axis=1)
        if ok.any():
            gsum += gns[ok].sum(axis=0); gcount += int(ok.sum())
        if not np.isfinite(losses).all() or float(np.max(losses)) > DIVERGE:
            diverged, div_step = True, (c + 1) * CHUNK
            break
        vl = float(loss_of(params, xs_va, ys_va))
        if np.isfinite(vl) and vl < best_val:
            best_val, best_params = vl, params

    use = best_params
    ynorm = float(jnp.mean(ys_te ** 2))
    test_nmse = float(loss_of(use, xs_te, ys_te)) / (ynorm + 1e-30)
    fv = float(loss_of(params, xs_va, ys_va))
    fn = float(loss_of(params, xs_te, ys_te)) / (ynorm + 1e-30)
    d = diagnostics(use, teacher, r, xs_te)
    g = (gsum / max(gcount, 1)).tolist()
    return dict(test_nmse=test_nmse, val_loss=best_val, init_val=init_val,
                final_val=fv if np.isfinite(fv) else float("inf"),
                final_nmse=fn if np.isfinite(fn) else float("inf"),
                improved=bool(best_val < init_val),
                init_max_z=init_diag["max_z"], init_markov=init_diag["markov"],
                init_nmse=float(loss_of(tuple(jnp.array(z) for z in params0),
                                        xs_te, ys_te)) / (ynorm + 1e-30),
                diverged=bool(diverged), div_step=div_step,
                gnorm_a=g[0], gnorm_theta=g[1], gnorm_B=g[2], gnorm_C=g[3], **d)


def perturb(params, eps, seed):
    rng = np.random.RandomState(50000 + seed)
    out = []
    for pp in params:
        pp = np.asarray(pp)
        out.append(pp + eps * (np.abs(pp) + 1.0) * rng.randn(*pp.shape))
    return tuple(out)


def main():
    rows = []
    t0 = time.time()
    for family in FAMILIES:
        for r in R_VALUES:
            for seed in EVAL_SEEDS:
                teacher = make_teacher(family, r, seed)
                cons, cdiag = constructive_realization(teacher, r)
                arms = {"A_constructive": cons}
                for e in EPS_LADDER:
                    arms[f"B_perturbed_{e:.0e}"] = perturb(cons, e, seed)
                arms["C_generic_stable"] = generic_stable_init(r, seed)
                for arm, p0 in arms.items():
                    best = None
                    for lr in LR_GRID:
                        res = train_one(teacher, p0, family, r, lr, seed)
                        if best is None or res["val_loss"] < best["val_loss"]:
                            best = dict(res, lr=lr)
                    row = dict(family=family, r=r, seed=seed, arm=arm,
                               rho_A=teacher["rho"], condS=teacher["condS"],
                               condT=cdiag["condT"], resid_AT_TM=cdiag["resid_AT_TM"],
                               **best)
                    rows.append(row)
                    print(f"[{time.time()-t0:7.1f}s] {family:20s} r={r} s={seed} {arm:16s} "
                          f"lr={best['lr']:.0e} nmse={best['test_nmse']:.3e} "
                          f"mk={best['markov']:.3e} val={best['val_loss']:.3e} "
                          f"rho={best['rho']:.3f} maxz={best['max_z']:.2e} "
                          f"init={best['init_nmse']:.2e} fin={best['final_nmse']:.2e} "
                          f"div={best['diverged']}", flush=True)
    with open("results/b37b/rows.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"done in {time.time()-t0:.1f}s -> results/b37b/rows.json")


if __name__ == "__main__":
    main()
