"""B37b cause isolation. The B37a-style constructive chart places the
interpolation nodes alpha_i on a FIXED grid in [0.25,0.85], independent of the
teacher spectrum, so u must be a high-degree oscillatory interpolant carrying
alpha_i -> lambda_i, with huge monomial coefficients. Hypothesis: THAT choice,
not the quotient chart itself, causes the coefficient->pole ill-conditioning.

Diagnostic: place alpha_i AT (or infinitesimally near) lambda_i instead, so
u ~ x. Same architecture, same parameter count, only the initialization moves.
The original negative result is preserved untouched; this only identifies its cause.
"""
import numpy as np
import jax.numpy as jnp
from credit_memory.b37a_universal_quotient import mult_matrix
from credit_memory.b37b_quotient_trainability import (
    FAMILIES, R_VALUES, EVAL_SEEDS, EPS_LADDER, make_teacher, build_T_complex,
    make_dataset, batched_loss, markov_error, perturb, train_one, LR_GRID, hermite_qu)


def hermite_qu_near(blocks, delta=1e-3):
    """alpha_i placed at lambda_i (offset by delta for repeated eigenvalues)."""
    r = sum(n for _, n in blocks)
    alphas, seen = [], {}
    for lam, n in blocks:
        lam = complex(lam)
        k = seen.get(round(lam.real, 12) + 1j * round(lam.imag, 12), 0)
        seen[round(lam.real, 12) + 1j * round(lam.imag, 12)] = k + 1
        off = delta * k
        alphas.append(lam + (off if abs(lam.imag) < 1e-14 else complex(off, 0.0)))
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


def realization(teacher, r, placement):
    blocks, S, A = teacher["blocks"], teacher["S"], teacher["A"]
    a_c, th_c, alphas, q_desc = placement(blocks)
    Tc = build_T_complex(blocks, alphas, q_desc, th_c, r)
    a, theta = a_c.real.copy(), th_c.real.copy()
    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))
    Tf = S @ np.linalg.inv(Tc)
    best = None
    rng = np.random.RandomState(0)
    combos = [(1, 0), (0, 1), (1, 1), (1, -1), (0.5, 1), (1, 0.5)]
    combos += [tuple(rng.randn(2)) for _ in range(20)]
    for c1, c2 in combos:
        Tr = c1 * Tf.real + c2 * Tf.imag
        c = np.linalg.cond(Tr)
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, Tr)
    if best is None or best[0] > 1e14:
        # numerically singular intertwiner: report rather than crash
        return None, dict(condT=float("inf"), resid=float("inf"),
                          theta_norm=float(np.linalg.norm(theta)),
                          a_norm=float(np.linalg.norm(a)), degenerate=True)
    condT, T = best
    resid = float(np.linalg.norm(A @ T - T @ M) / (1 + np.linalg.norm(A) * np.linalg.norm(T)))
    return (a, theta, np.linalg.solve(T, teacher["Bs"]), teacher["Cs"] @ T), \
           dict(condT=float(condT), resid=resid, theta_norm=float(np.linalg.norm(theta)),
                a_norm=float(np.linalg.norm(a)), degenerate=False)


def kappa(cons, r, eps, seed, reps=8):
    M0 = np.asarray(mult_matrix(jnp.array(cons[1]), jnp.array(cons[0]), r))
    e0 = np.sort_complex(np.linalg.eigvals(M0))
    ks, un = [], []
    for rep in range(reps):
        p = perturb(cons, eps, seed * 100 + rep)
        M = np.asarray(mult_matrix(jnp.array(p[1]), jnp.array(p[0]), r))
        e = np.sort_complex(np.linalg.eigvals(M))
        ks.append(np.max(np.abs(e - e0)) / eps)
        un.append(np.max(np.abs(e)) > 1.0)
    return float(np.median(ks)), float(np.mean(un))


print("=" * 112)
print("A. Interpolant magnitude and coefficient->pole conditioning: FIXED-GRID alpha vs alpha-AT-lambda")
print("=" * 112)
print(f"{'family':21s} {'r':>2s} | {'||theta||':>10s} {'cond(T)':>9s} {'kappa@1e-6':>11s} {'P(unst)':>8s} "
      f"| {'||theta||':>10s} {'cond(T)':>9s} {'kappa@1e-6':>11s} {'P(unst)':>8s}")
for f in FAMILIES:
    for r in R_VALUES:
        cells = []
        for place in (hermite_qu, hermite_qu_near):
            tn, ct, kk, uu = [], [], [], []
            for seed in EVAL_SEEDS:
                t = make_teacher(f, r, seed)
                cons, d = realization(t, r, place)
                if cons is None:
                    tn.append(d["theta_norm"]); ct.append(d["condT"])
                    kk.append(float("inf")); uu.append(1.0); continue
                k, u = kappa(cons, r, 1e-6, seed)
                tn.append(d["theta_norm"]); ct.append(d["condT"]); kk.append(k); uu.append(u)
            cells.append(f"{np.median(tn):10.2e} {np.median(ct):9.2e} {np.median(kk):11.2e} {np.mean(uu):8.2f}")
        print(f"{f:21s} {r:2d} | {cells[0]} | {cells[1]}")

print()
print("=" * 112)
print("B. Trainability under the SAME arms, but with alpha-at-lambda placement (400 steps, 3 seeds)")
print("=" * 112)
print(f"{'family':21s} {'r':>2s} {'A_near':>11s} {'B_near_1e-6':>12s} {'B_near_1e-4':>12s} "
      f"{'B_near_1e-2':>12s} {'div frac':>9s}")
for f in FAMILIES:
    for r in R_VALUES:
        res = {e: [] for e in ("A",) + EPS_LADDER}
        divs = []
        for seed in EVAL_SEEDS:
            t = make_teacher(f, r, seed)
            cons, dd = realization(t, r, hermite_qu_near)
            if cons is None:
                for key in res:
                    res[key].append(float("inf"))
                divs.append(True)
                continue
            for key in res:
                p0 = cons if key == "A" else perturb(cons, key, seed)
                best = None
                for lr in LR_GRID:
                    o = train_one(t, p0, f, r, lr, seed)
                    if best is None or o["val_loss"] < best["val_loss"]:
                        best = o
                res[key].append(best["test_nmse"]); divs.append(best["diverged"])
        print(f"{f:21s} {r:2d} " + " ".join(f"{np.median(res[k]):11.2e}" if k == 'A'
              else f"{np.median(res[k]):12.2e}" for k in res) + f" {np.mean(divs):9.2f}")
