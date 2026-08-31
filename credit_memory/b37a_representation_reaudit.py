"""B37a representation RE-AUDIT (correction only).

Scope: ONLY the universality / derogatory-representation claim. The exact
sensitivity results, the exact-rational check, the float64 conditioning
findings and the transient-amplification analysis are untouched and not rerun.

Universal criterion tested (the correct one):
        A T = T u(C_q)      i.e.   A = T u(C_q) T^{-1}
NOT  A T = T C_q, which is only the u(x)=x special case and would force A cyclic.

CONSTRUCTIVE THEOREM (explicit, not a numerical search):
Let A have Jordan blocks {(lambda_i, n_i)}_{i=1..p}, sum n_i = r. Pick DISTINCT
alpha_i and set q = prod_i (x-alpha_i)^{n_i} (monic, degree r).

(1) Basis for C_q ~ (+)_i J_{n_i}(alpha_i):  with g_{i,k} = q(x)/(x-alpha_i)^k,
    k=1..n_i, one has  x*g_{i,k} = g_{i,k-1} + alpha_i g_{i,k}  (g_{i,0}=q=0 in A),
    so V = [ ... g_{i,k} ... ] satisfies  C_q V = V D,  D = (+)_i J_{n_i}(alpha_i).
    Hence u(C_q) V = V u(D),  u(D) = (+)_i u(J_{n_i}(alpha_i)).

(2) Hermite conditions on u:  u(alpha_i) = lambda_i for every block, and
    u'(alpha_i) = 1 for blocks with n_i >= 2. Count = p_1 + 2 p_2 <= r, so
    Hermite interpolation gives deg u <= r-1 -- inside the model.

(3) Per block, u(J_n(alpha)) = lambda I + P with P = u'(alpha) N + ... nilpotent
    of rank n-1 (as u'(alpha) != 0). Taking W = [P^{n-1}e | ... | P e | e] with
    e = e_n gives P W = W N, hence u(J_n(alpha)) W = W J_n(lambda).

T = V * blockdiag(W_i) then satisfies EXACTLY  u(C_q) T = T A, i.e. A T' = T' u(C_q)
with T' = T^{-1}... reported here directly as  ||u(C_q) T - T A|| and equivalently
||A T^{-1} - T^{-1} u(C_q)||. ONE quotient suffices; no product of quotients.
"""
from __future__ import annotations

import math
import numpy as np
import jax.numpy as jnp

from credit_memory.b37a_universal_quotient import mult_matrix


def jordan_block(lam, n):
    J = np.eye(n) * lam
    if n > 1:
        J += np.diag(np.ones(n - 1), 1)
    return J


def build_target(blocks):
    r = sum(n for _, n in blocks)
    A = np.zeros((r, r))
    o = 0
    for lam, n in blocks:
        A[o:o + n, o:o + n] = jordan_block(lam, n)
        o += n
    return A


def jordan_type(M, lam, tol=1e-6):
    """Block sizes of M at lam via rank((M-lam I)^j). Returns None if the
    rank sequence is numerically inconsistent (sizes must sum to multiplicity)."""
    r = M.shape[0]
    P = M - lam * np.eye(r)
    scale = max(1.0, np.linalg.norm(M))
    ranks, Pk = [r], np.eye(r)
    for _ in range(r):
        Pk = Pk @ P
        s = np.linalg.svd(Pk, compute_uv=False)
        ranks.append(int((s > tol * scale).sum()))
    if any(ranks[i + 1] > ranks[i] for i in range(len(ranks) - 1)):
        return None
    counts = [ranks[j - 1] - ranks[j] for j in range(1, r + 1)]  # #blocks of size >= j
    sizes = []
    for j in range(len(counts) - 1, -1, -1):
        nxt = counts[j + 1] if j + 1 < len(counts) else 0
        e = counts[j] - nxt
        if e < 0:
            return None
        sizes += [j + 1] * e
    return sorted(sizes, reverse=True)


def construct(blocks, alpha_lo=0.20, alpha_hi=0.92):
    """Returns (a, theta, alphas, q_roots). Distinct alphas, Hermite u."""
    r = sum(n for _, n in blocks)
    p = len(blocks)
    alphas = ([alpha_lo] if p == 1 else
              list(np.linspace(alpha_lo, alpha_hi, p)))
    roots = []
    for (lam, n), al in zip(blocks, alphas):
        roots += [al] * n
    a = np.poly(np.array(roots))[::-1][:r].real.copy()   # ascending, drop leading x^r

    rows, rhs = [], []
    for (lam, n), al in zip(blocks, alphas):
        rows.append([al ** k for k in range(r)]); rhs.append(lam)
        if n >= 2:
            rows.append([0.0 if k == 0 else k * al ** (k - 1) for k in range(r)]); rhs.append(1.0)
    theta, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return a, theta, alphas


def poly_div_exact(num_desc, root):
    """Divide a descending-coefficient polynomial by (x - root); synthetic division."""
    out = [num_desc[0]]
    for c in num_desc[1:]:
        out.append(c + out[-1] * root)
    return np.array(out[:-1])            # drop remainder (exactly 0 in theory)


def build_T(blocks, a, theta, alphas):
    """Explicit T with u(C_q) T = T A  (construction (1)+(3) in the docstring)."""
    r = sum(n for _, n in blocks)
    q_desc = np.concatenate([[1.0], a[::-1]])            # descending, monic, degree r
    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))

    cols = []
    for (lam, n), al in zip(blocks, alphas):
        # g_k = q / (x-al)^k  for k=1..n, as ASCENDING coefficient vectors of length r
        cur = q_desc.copy()
        gs = []
        for k in range(1, n + 1):
            cur = poly_div_exact(cur, al)
            v = np.zeros(r)
            asc = cur[::-1]
            v[:len(asc)] = asc
            gs.append(v)
        V = np.stack(gs, axis=1)                          # r x n, columns g_1..g_n
        # P = u(J_n(al)) - lam I  in that basis; build via the Toeplitz of u^{(j)}(al)/j!
        P = np.zeros((n, n))
        du = np.array(theta, dtype=float)
        for j in range(1, n):
            # coefficient u^{(j)}(al)/j!  =  j-th Taylor coefficient of u at al
            c = np.polyval(np.polyder(np.poly1d(theta[::-1]), j), al) / math.factorial(j)
            P += c * np.diag(np.ones(n - j), j)
        # W = [P^{n-1} e | ... | P e | e], e = e_n
        e = np.zeros(n); e[-1] = 1.0
        Wcols, w = [], e.copy()
        for _ in range(n):
            Wcols.append(w.copy()); w = P @ w
        W = np.stack(Wcols[::-1], axis=1)                 # n x n
        cols.append(V @ W)
    T = np.concatenate(cols, axis=1)
    return T, M


def markov_err(A, M, T, K_max=40, seed=0):
    """Transformed ports: with u(C_q) T = T A, take Bq = T b, Cq = c T^{-1}.
    Then Cq (u(C_q))^k Bq = c A^k b."""
    rng = np.random.RandomState(seed + 3)
    r = A.shape[0]
    b, c = rng.randn(r), rng.randn(r)
    Bq = T @ b
    Cq = np.linalg.solve(T.T, c)          # c T^{-1}
    va, vm, ma, mm = b.copy(), Bq.copy(), [], []
    for _ in range(K_max):
        ma.append(c @ va); mm.append(Cq @ vm)
        va, vm = A @ va, M @ vm
    ma, mm = np.array(ma), np.array(mm)
    return float(np.max(np.abs(ma - mm)) / (1 + np.max(np.abs(ma))))


def audit(blocks, label, verbose=True):
    A = build_target(blocks)
    r = A.shape[0]
    a, theta, alphas = construct(blocks)
    T, M = build_T(blocks, a, theta, alphas)
    resid = float(np.linalg.norm(M @ T - T @ A) /
                  (1 + np.linalg.norm(A) * np.linalg.norm(T)))
    condT = float(np.linalg.cond(T))
    lam = blocks[0][0]
    target = sorted([n for _, n in blocks], reverse=True)
    got = jordan_type(M, lam)
    mk = markov_err(A, M, T)
    ok = (resid < 1e-9) and (mk < 1e-6)
    if verbose:
        gs = str(got) if got is not None else "unreliable"
        print(f"  {label:26s} r={r}  target={str(target):18s} u(C_q) type={gs:18s} "
              f"||u(C_q)T-TA||rel={resid:9.2e}  markov={mk:9.2e}  cond(T)={condT:9.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, resid, mk, condT, got, target


def partitions(n, mx=None):
    mx = mx or n
    if n == 0:
        yield []
        return
    for k in range(min(n, mx), 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest


def main():
    L = 0.62
    print("=" * 122)
    print("PART 1 -- the explicit counterexample from the correction:")
    print("  q=(x-alpha)^4,  u(x)=lambda+(x-alpha)^3   =>   u(C_q) = lambda*I + N^3")
    print("=" * 122)
    alpha, lam, r = 0.37, 0.62, 4
    a = np.poly(np.full(r, alpha))[::-1][:r].real.copy()
    theta = np.zeros(r); theta[0] = lam
    theta[:4] += (np.poly1d([1.0, -alpha]) ** 3).coefficients[::-1]
    M = np.asarray(mult_matrix(jnp.array(theta), jnp.array(a), r))
    N3 = M - lam * np.eye(r)
    sv = np.linalg.svd(N3, compute_uv=False)
    print(f"  rank(u(C_q) - lambda I) = {int((sv > 1e-8 * max(1, sv[0])).sum())}      "
          f"||(u(C_q)-lambda I)^2|| = {np.linalg.norm(N3 @ N3):.3e}")
    print(f"  eigenvalues = {np.round(np.sort(np.linalg.eigvals(M).real), 12)}")
    print(f"  Jordan type at lambda = {jordan_type(M, lam)}   (expected [2, 1, 1])")
    print("  => J_2+J_1+J_1 IS realizable by a SINGLE quotient. Prior caveat REFUTED.")

    print()
    print("=" * 122)
    print("PART 2 -- required derogatory targets, universal criterion  u(C_q) T = T A")
    print("=" * 122)
    required = [([(L, 1)] * 4, "lambda*I_4"), ([(L, 1)] * 6, "lambda*I_6"),
                ([(L, 2), (L, 1), (L, 1)], "J2+J1+J1"), ([(L, 2), (L, 2)], "J2+J2"),
                ([(L, 3), (L, 2)], "J3+J2")]
    req = [audit(b, lbl) for b, lbl in required]

    print()
    print("=" * 122)
    print("PART 3 -- ALL equal-eigenvalue partitions, r = 2..8")
    print("=" * 122)
    npass = ntot = 0
    worst_resid = worst_mk = 0.0
    typ_ok = typ_tot = 0
    for rr in range(2, 9):
        for part in partitions(rr):
            ok, resid, mk, condT, got, target = audit([(L, k) for k in part],
                                                       "+".join(f"J{k}" for k in part))
            npass += int(ok); ntot += 1
            worst_resid = max(worst_resid, resid); worst_mk = max(worst_mk, mk)
            if got is not None:
                typ_tot += 1; typ_ok += int(got == target)

    print()
    print("=" * 122)
    print(f"REQUIRED TARGETS      : {sum(x[0] for x in req)}/{len(req)} pass")
    print(f"ALL PARTITIONS r=2..8 : {npass}/{ntot} pass")
    print(f"worst ||u(C_q)T - TA||rel = {worst_resid:.3e}   worst markov err = {worst_mk:.3e}")
    print(f"independent Jordan-type confirmation (where rank sequence was numerically "
          f"reliable): {typ_ok}/{typ_tot}")
    print("=" * 122)


if __name__ == "__main__":
    main()
