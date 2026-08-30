"""B37a isolation -- EXACT RATIONAL verification of the reduced-eligibility
derivation, with zero floating-point roundoff.

Purpose: distinguish "the derivation is wrong" from "float64 roundoff on an
ill-conditioned instance". Every float64 number IS exactly a rational, and the
universal quotient model needs only + - * (q is monic, so synthetic division
requires no true division), so the entire computation can be carried out in
exact arithmetic with fractions.Fraction.

Reference derivative: a minimal forward-mode dual-number AD implemented over
Fraction. This differentiates the FORWARD MODEL MECHANICALLY -- it never uses
the analytic s-recursions -- so agreement is an independent confirmation of
the derivation, not a restatement of it.

Test: for random directions d_theta, d_a, d_B, verify EXACTLY
    D z_t[d] == M_{s_t} d      (as exact rationals, difference identically 0)

Run: python -m credit_memory.b37a_exact_rational_check
"""
from __future__ import annotations

from fractions import Fraction as F
import numpy as np


def to_F(arr):
    return [F(float(v)) for v in np.asarray(arr).ravel()]


# ---------------------------------------------------------------- dual numbers
class Dual:
    """value + eps*derivative, exact rational forward-mode AD."""
    __slots__ = ("v", "d")

    def __init__(self, v, d=F(0)):
        self.v, self.d = v, d

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v + o.v, self.d + o.d)
    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v - o.v, self.d - o.d)

    def __rsub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(o.v - self.v, o.d - self.d)

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v * o.v, self.v * o.d + self.d * o.v)
    __rmul__ = __mul__

    def __neg__(self):
        return Dual(-self.v, -self.d)


# ------------------------------------------------- exact polynomial primitives
def conv(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0 and not isinstance(pi, Dual):
            continue
        for j, qj in enumerate(q):
            out[i + j] = out[i + j] + pi * qj
    return out


def divmod_monic(p, a, r):
    """p ascending, len 2r-1; q = x^r + sum a_j x^j monic. Returns (v, rem)."""
    p = list(p)
    v = [F(0)] * r
    for i in range(2 * r - 2, r - 1, -1):
        c = p[i]
        v[i - r] = c
        for j in range(r):
            p[i - r + j] = p[i - r + j] - c * a[j]
        p[i] = F(0)
    return v, p[:r]


def alg_mult(c1, c2, a, r):
    return divmod_monic(conv(c1, c2), a, r)[1]


def mult_matrix(c, a, r):
    """columns i = rem(x^i c, q)."""
    cols = []
    for i in range(r):
        p = [F(0)] * (2 * r - 1)
        for k in range(r):
            p[i + k] = p[i + k] + c[k]
        cols.append(divmod_monic(p, a, r)[1])
    return cols  # cols[i][row]


def matvec_from_cols(cols, vec, r):
    return [sum((cols[i][row] * vec[i] for i in range(r)), F(0)) for row in range(r)]


# --------------------------------------------------------------- the two paths
def forward_dual(theta, a, B, z0, xs, r, m, T, which, direction):
    """Mechanical forward-mode AD of the forward model along `direction`
    in parameter block `which` in {'theta','a','B'}. Returns [dz_t] for t=1..T."""
    def lift(vals, seeds=None):
        return [Dual(v, seeds[i] if seeds else F(0)) for i, v in enumerate(vals)]

    th = lift(theta, direction if which == "theta" else None)
    aa = lift(a, direction if which == "a" else None)
    BB = [lift(B[j], direction[j * r:(j + 1) * r] if which == "B" else None) for j in range(m)]
    z = lift(z0)
    out = []
    for t in range(T):
        prod = conv(th, z)
        _, rem = divmod_monic(prod, aa, r)
        z = [rem[i] + sum((BB[j][i] * Dual(xs[t][j]) for j in range(m)), Dual(F(0)))
             for i in range(r)]
        out.append([zi.d for zi in z])
    return out


def reduced_path(theta, a, B, z0, xs, r, m, T, which, direction):
    """Analytic reduced eligibility, contracted with `direction`."""
    one = [F(1)] + [F(0)] * (r - 1)
    z = list(z0)
    s = [F(0)] * r
    out = []
    for t in range(T):
        prod = conv(theta, z)
        v_t, rem = divmod_monic(prod, a, r)
        z_next = [rem[i] + sum((B[j][i] * xs[t][j] for j in range(m)), F(0)) for i in range(r)]
        if which == "theta":
            s = [x + y for x, y in zip(alg_mult(theta, s, a, r), z)]
        elif which == "a":
            s = [x - y for x, y in zip(alg_mult(theta, s, a, r), v_t)]
        else:
            s = [x + xs[t][0] * y for x, y in zip(alg_mult(theta, s, a, r), one)]
        out.append(matvec_from_cols(mult_matrix(s, a, r),
                                    direction[:r] if which != "B" else direction[:r], r))
        z = z_next
    return out


def run(r, T, seed, case_params=None):
    rng = np.random.RandomState(seed)
    m = 1  # B-direction test uses channel 0 only, keeping exact arithmetic small
    if case_params is None:
        theta = to_F(rng.randn(r) * 0.3)
        a = to_F(rng.randn(r) * 0.3)
        B = [to_F(rng.randn(r) * 0.3)]
        z0 = to_F(rng.randn(r) * 0.3)
        xs = [[F(float(x))] for x in rng.randn(T) * 0.5]
    else:
        theta, a, B, z0, xs = case_params

    print(f"  r={r} T={T} seed={seed} (exact rational arithmetic, zero roundoff)")
    all_zero = True
    for which in ("theta", "a", "B"):
        direction = to_F(rng.randn(r))
        ad = forward_dual(theta, a, B, z0, xs, r, m, T, which, direction)
        red = reduced_path(theta, a, B, z0, xs, r, m, T, which, direction)
        worst = max(abs(ad[t][i] - red[t][i]) for t in range(T) for i in range(r))
        exact = (worst == 0)
        all_zero &= exact
        print(f"    d/d{which:6s}: max|AD - reduced| = {worst}  "
              f"{'EXACTLY ZERO' if exact else 'NONZERO -> derivation error'}")
    return all_zero


if __name__ == "__main__":
    print("=" * 78)
    print("EXACT RATIONAL CHECK: mechanical forward-mode AD vs analytic reduced trace")
    print("=" * 78)
    ok = True
    for r, T in ((2, 8), (4, 8), (8, 6), (16, 5), (32, 4)):
        ok &= run(r, T, seed=0)
    print("=" * 78)
    print(f"ALL DIRECTIONAL DERIVATIVES EXACTLY EQUAL: {ok}")
    print("=" * 78)
