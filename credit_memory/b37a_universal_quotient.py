"""B37a -- universal quotient recurrence and reduced exact eligibilities.

VERIFICATION ONLY. No training, no optimization, no FFT polynomial
arithmetic (direct O(r^2) convolution/synthetic division only).

=======================================================================
EXACT EQUATIONS USED
=======================================================================
Algebra:  A = R[x]/(q_a),  q_a(x) = x^r + sum_{j=0}^{r-1} a_j x^j
          u_theta(x) = sum_{k=0}^{r-1} theta_k x^k   (an element of A)
Basis of A: {1, x, ..., x^{r-1}}; an element is a length-r coefficient
vector in ASCENDING order (index k = coefficient of x^k).

Forward recurrence (per step t), with x_t in R^m, B in R^{r x m}:
    u_theta * z_t = q_a * v_t + r_t          (polynomial division, exact)
    z_{t+1} = r_t + B x_t
where deg(u*z) <= 2r-2, so deg(v_t) <= r-2 (stored as a length-r vector
with v_t[r-1] = 0) and deg(r_t) <= r-1.

Reduced (compressed) exact eligibilities -- each a SINGLE algebra
element (r scalars), not an r x r Jacobian:
    s^theta_{t+1} = rem(u s^theta_t, q) + z_t          s^theta_0 = 0
    s^a_{t+1}     = rem(u s^a_t,     q) - v_t          s^a_0     = 0
    s^{b_j}_{t+1} = rem(u s^{b_j}_t, q) + x_{j,t} * 1  s^{b_j}_0 = 0
(1 = multiplicative identity = e_0 = (1,0,...,0).)

Claim under test (reconstruction):
    D_theta z_t     = M_{s^theta_t}
    D_a z_t         = M_{s^a_t}
    D_{B[:,j]} z_t  = M_{s^{b_j}_t}
where M_c is the r x r matrix of multiplication-by-c in A, i.e.
column i of M_c is rem(x^i * c, q).

DERIVATION of the a-sensitivity (the novel term; the others are the
standard algebra-valued eligibility). From u z = q v + r, differentiate
w.r.t. a_j at fixed u, z:
    0 = (dq/da_j) v + q (dv/da_j) + (dr/da_j) = x^j v + q dv/da_j + dr/da_j
Reduce mod q (deg(dr/da_j) < r so it is its own remainder, and
rem(q * anything, q) = 0):
    dr/da_j = -rem(x^j v, q) = -(C_q^j) v     ==>   G^a_t = -M_{v_t}
Together with dz_{t+1}/dz_t = M_u this gives the s^a recursion above.

=======================================================================
PARAMETER ORDERING
=======================================================================
theta: length r, theta[k] = coefficient of x^k in u.
a:     length r, a[j]     = coefficient of x^j in q (q is MONIC, the
       leading x^r coefficient is fixed at 1 and is NOT a parameter).
B:     shape (r, m), column j = B[:, j] is the input vector for input
       channel j. Sensitivity s^{b_j} corresponds to column j.
Jacobians are indexed [output_coord, param_index], e.g.
(D_theta z_t)[i,k] = d z_t[i] / d theta[k].

=======================================================================
COMPANION CONVENTION
=======================================================================
COLUMN companion (multiplication-by-x operator on A in the ascending
coefficient basis):
    C_q[m, m-1] = 1        for m = 1..r-1     (subdiagonal ones)
    C_q[j, r-1] = -a_j     for j = 0..r-1     (last column)
so that (C_q c) is the coefficient vector of rem(x * c, q). Then
M_u = u(C_q) = sum_k theta_k C_q^k, and the eigenvalues of C_q are the
roots of q, while the eigenvalues of M_u = u(C_q) are u(lambda_i).

=======================================================================
THREE INDEPENDENT PATHS COMPARED
=======================================================================
A. REDUCED   : the s-recursions above; S reconstructed as M_s.
B. FULL RTRL : S_{t+1} = J_t S_t + G_t with J_t and G_t obtained by
               AUTODIFF of a single step (jax.jacobian), NOT from the
               analytic formulas -- so path B is independent of the
               derivation being tested.
C. BPTT/AD   : one end-to-end jax.jacobian of the whole scan output
               trajectory w.r.t. (theta, a, B). Independent of both the
               derivation and the RTRL accumulation order.

=======================================================================
TOLERANCES / PRECISION
=======================================================================
dtype: float64 everywhere (jax_enable_x64=True). Machine eps = 2.22e-16.
Preregistered PASS threshold on
    max_t ||S_t^reduced - S_t^full||_F / (1 + ||S_t^full||_F)
is < 1e-10, reported separately for theta, a, B, and against BOTH
reference paths (B and C). Cases whose forward trajectory overflows or
goes non-finite are reported separately and excluded from the pass
criterion (per "all non-overflowing cases").

Run: python -m credit_memory.b37a_universal_quotient
"""
from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PASS_THRESHOLD = 1e-10
T_STEPS = 40
M_INPUTS = 2
OVERFLOW_LIMIT = 1e150
R_VALUES = (2, 4, 8, 16, 32)


# =======================================================================
# Polynomial primitives (direct O(r^2), no FFT).
# =======================================================================
def divmod_monic(p, a, r):
    """Divide p (length 2r-1, ascending) by monic q = x^r + sum a_j x^j.
    Returns (v, rem): v length r (deg <= r-2, v[r-1]=0), rem length r."""
    v = jnp.zeros(r, dtype=p.dtype)
    for i in range(2 * r - 2, r - 1, -1):
        c = p[i]
        v = v.at[i - r].set(c)
        idx = jnp.arange(r) + (i - r)
        p = p.at[idx].add(-c * a)
        p = p.at[i].set(0.0)
    return v, p[:r]


def alg_mult(c1, c2, a, r):
    """Product of two algebra elements, reduced mod q."""
    prod = jnp.convolve(c1, c2)
    _, rem = divmod_monic(prod, a, r)
    return rem


def mult_matrix(c, a, r):
    """M_c: r x r matrix of multiplication-by-c. Column i = rem(x^i c, q)."""
    cols = []
    for i in range(r):
        p = jnp.zeros(2 * r - 1, dtype=c.dtype).at[i:i + r].add(c)
        _, rem = divmod_monic(p, a, r)
        cols.append(rem)
    return jnp.stack(cols, axis=1)


def companion(a, r):
    """Column companion matrix C_q (multiplication-by-x)."""
    C = jnp.zeros((r, r), dtype=a.dtype)
    if r > 1:
        C = C.at[jnp.arange(1, r), jnp.arange(0, r - 1)].set(1.0)
    C = C.at[:, r - 1].add(-a)
    return C


# =======================================================================
# Forward model.
# =======================================================================
def step_fn(z, theta, a, B, x, r):
    """One step. Returns (z_next, v_t)."""
    prod = jnp.convolve(theta, z)
    v, rem = divmod_monic(prod, a, r)
    return rem + B @ x, v


def rollout(theta, a, B, z0, xs, r):
    """Returns the full trajectory z_1..z_T, shape (T, r)."""
    def body(z, x):
        z_next, _ = step_fn(z, theta, a, B, x, r)
        return z_next, z_next
    _, zs = jax.lax.scan(body, z0, xs)
    return zs


# =======================================================================
# PATH A -- reduced (compressed) eligibilities.
# =======================================================================
def reduced_sensitivities(theta, a, B, z0, xs, r, m):
    """Propagates the r-scalar traces and reconstructs the r x r Jacobians."""
    T = xs.shape[0]
    one = jnp.zeros(r, dtype=theta.dtype).at[0].set(1.0)
    z = z0
    s_theta = jnp.zeros(r, dtype=theta.dtype)
    s_a = jnp.zeros(r, dtype=theta.dtype)
    s_b = [jnp.zeros(r, dtype=theta.dtype) for _ in range(m)]
    S_theta, S_a, S_b = [], [], []
    for t in range(T):
        z_next, v_t = step_fn(z, theta, a, B, xs[t], r)
        s_theta = alg_mult(theta, s_theta, a, r) + z
        s_a = alg_mult(theta, s_a, a, r) - v_t
        s_b = [alg_mult(theta, s_b[j], a, r) + xs[t, j] * one for j in range(m)]
        S_theta.append(mult_matrix(s_theta, a, r))
        S_a.append(mult_matrix(s_a, a, r))
        S_b.append(jnp.stack([mult_matrix(s_b[j], a, r) for j in range(m)], axis=-1))
        z = z_next
    return jnp.stack(S_theta), jnp.stack(S_a), jnp.stack(S_b)


# =======================================================================
# PATH B -- explicit full RTRL with AUTODIFF local Jacobians.
# =======================================================================
def full_rtrl_sensitivities(theta, a, B, z0, xs, r, m):
    T = xs.shape[0]
    z = z0
    S_theta = jnp.zeros((r, r), dtype=theta.dtype)
    S_a = jnp.zeros((r, r), dtype=theta.dtype)
    S_b = jnp.zeros((r, r, m), dtype=theta.dtype)
    out_theta, out_a, out_b = [], [], []
    for t in range(T):
        x_t = xs[t]
        J_t = jax.jacobian(lambda zz: step_fn(zz, theta, a, B, x_t, r)[0])(z)
        G_theta = jax.jacobian(lambda th: step_fn(z, th, a, B, x_t, r)[0])(theta)
        G_a = jax.jacobian(lambda aa: step_fn(z, theta, aa, B, x_t, r)[0])(a)
        G_b = jax.jacobian(lambda BB: step_fn(z, theta, a, BB, x_t, r)[0])(B)
        S_theta = J_t @ S_theta + G_theta
        S_a = J_t @ S_a + G_a
        S_b = jnp.einsum("ij,jkl->ikl", J_t, S_b) + G_b
        out_theta.append(S_theta)
        out_a.append(S_a)
        out_b.append(S_b)
        z, _ = step_fn(z, theta, a, B, x_t, r)
    return jnp.stack(out_theta), jnp.stack(out_a), jnp.stack(out_b)


# =======================================================================
# PATH C -- end-to-end BPTT/autodiff.
# =======================================================================
def autodiff_sensitivities(theta, a, B, z0, xs, r):
    def f(theta, a, B):
        return rollout(theta, a, B, z0, xs, r)
    jac = jax.jacobian(f, argnums=(0, 1, 2))(theta, a, B)
    return jac  # shapes (T,r,r), (T,r,r), (T,r,r,m)


# =======================================================================
# Test-case construction.
# =======================================================================
def poly_from_roots(roots, r):
    """Monic real polynomial from roots; returns a (length r, ascending)."""
    coeffs = np.poly(roots)          # descending, leading 1
    coeffs = np.real_if_close(coeffs, tol=1e6)
    assert coeffs.shape[0] == r + 1
    a = coeffs[::-1][:r].real.copy()  # ascending, drop leading x^r
    return a


def make_case(family, r, seed, m=M_INPUTS):
    """Returns (theta, a, B, z0, xs, info). Ensures rho(M_u) < 1."""
    rng = np.random.RandomState(seed)
    info = {}

    def rand_stable_roots(n, lo=0.05, hi=0.95):
        roots = []
        while len(roots) < n:
            if len(roots) + 1 < n and rng.rand() < 0.5:
                mod = rng.uniform(lo, hi)
                ang = rng.uniform(0.2, np.pi - 0.2)
                roots += [mod * np.exp(1j * ang), mod * np.exp(-1j * ang)]
            else:
                roots.append(rng.uniform(-hi, hi))
        return np.array(roots[:n])

    theta = None
    if family == "random_stable":
        a = poly_from_roots(rand_stable_roots(r), r)
        theta = rng.randn(r) * 0.5
    elif family == "real_distinct":
        roots = np.linspace(-0.9, 0.9, r) + rng.randn(r) * 0.01
        a = poly_from_roots(roots, r)
        theta = rng.randn(r) * 0.5
    elif family == "complex_conjugate":
        n_pairs = r // 2
        roots = []
        for i in range(n_pairs):
            mod = rng.uniform(0.3, 0.95)
            ang = rng.uniform(0.3, np.pi - 0.3)
            roots += [mod * np.exp(1j * ang), mod * np.exp(-1j * ang)]
        if r % 2 == 1:
            roots.append(rng.uniform(-0.9, 0.9))
        a = poly_from_roots(np.array(roots), r)
        theta = rng.randn(r) * 0.5
    elif family == "lambda_I":
        a = poly_from_roots(rand_stable_roots(r), r)
        theta = np.zeros(r)
        theta[0] = rng.uniform(0.3, 0.9)          # u = const  =>  M_u = lambda I
        info["note"] = "u constant => M_u = lambda*I exactly"
    elif family == "repeated_eigs":
        n_distinct = max(1, r // 4)
        vals = rng.uniform(-0.9, 0.9, size=n_distinct)
        roots = np.repeat(vals, int(np.ceil(r / n_distinct)))[:r]
        a = poly_from_roots(roots, r)
        theta = rng.randn(r) * 0.5
    elif family == "exact_jordan":
        lam = rng.uniform(0.5, 0.9)
        a = poly_from_roots(np.full(r, lam), r)   # q = (x-lam)^r
        theta = rng.randn(r) * 0.5
        info["note"] = "q=(x-lam)^r => C_q is a single Jordan block of size r"
    elif family == "multi_jordan_shared":
        # A single quotient ring is NONDEROGATORY (one Jordan block per
        # eigenvalue of C_q). Multiple Jordan blocks SHARING an eigenvalue
        # are still reachable for M_u = u(C_q): take q=(x-lam)^r and u with
        # u'(lam)=0, u''(lam)!=0, so u(C_q)-u(lam)I is nilpotent of index
        # ~r/2 with rank ~r/2 => TWO Jordan blocks sharing eigenvalue u(lam).
        lam = rng.uniform(0.4, 0.8)
        a = poly_from_roots(np.full(r, lam), r)
        # build u in the shifted basis: u = c0 + c2*(x-lam)^2 (no linear term)
        shifted = np.zeros(r)
        shifted[0] = rng.uniform(0.3, 0.7)
        if r >= 3:
            shifted[2] = rng.uniform(0.5, 1.0)
        theta = np.zeros(r)
        for k, ck in enumerate(shifted):
            if ck == 0:
                continue
            # expand ck*(x-lam)^k into the monomial basis
            binom = np.poly1d([1.0, -lam]) ** k
            c_asc = binom.coefficients[::-1]
            theta[:len(c_asc)] += ck * c_asc
        info["note"] = "u'(lam)=0, u''(lam)!=0 => M_u has 2 Jordan blocks sharing one eigenvalue"
    elif family == "nearly_defective":
        eps = 1e-8
        base = rng.uniform(0.4, 0.85)
        roots = np.array([base + (i % 2) * eps for i in range(r)])
        a = poly_from_roots(roots, r)
        theta = rng.randn(r) * 0.5
        info["note"] = f"roots split by eps={eps}"
    elif family == "nonnormal":
        roots = 0.95 * np.exp(1j * np.linspace(0, 0.25, r))
        roots = np.concatenate([roots[: r // 2], np.conj(roots[: r // 2])])
        if roots.shape[0] < r:
            roots = np.concatenate([roots, np.full(r - roots.shape[0], 0.9)])
        a = poly_from_roots(roots[:r], r)
        theta = rng.randn(r) * 0.5
        info["note"] = "tightly clustered roots near |z|=0.95 => strongly nonnormal companion"
    elif family == "stiff":
        mags = np.logspace(-6, np.log10(0.99), r)
        signs = rng.choice([-1.0, 1.0], size=r)
        a = poly_from_roots(mags * signs, r)
        theta = rng.randn(r) * 0.5
        info["note"] = "root magnitudes spanning 1e-6 .. 0.99"
    else:
        raise ValueError(family)

    a = jnp.array(a, dtype=jnp.float64)
    theta = jnp.array(theta, dtype=jnp.float64)

    # enforce spectral radius of M_u below 1 (scaling theta scales M_u linearly,
    # preserving every structural property used above)
    Mu = np.asarray(mult_matrix(theta, a, r))
    rho = float(np.max(np.abs(np.linalg.eigvals(Mu))))
    if not np.isfinite(rho) or rho > 0.9:
        theta = theta * (0.9 / (rho + 1e-30))
        Mu = np.asarray(mult_matrix(theta, a, r))
        rho = float(np.max(np.abs(np.linalg.eigvals(Mu))))
    info["rho_Mu"] = rho
    info["cond_Mu"] = float(np.linalg.cond(Mu))

    B = jnp.array(rng.randn(r, m) / np.sqrt(r), dtype=jnp.float64)
    z0 = jnp.array(rng.randn(r) * 0.1, dtype=jnp.float64)
    xs = jnp.array(rng.randn(T_STEPS, m) * 0.5, dtype=jnp.float64)
    return theta, a, B, z0, xs, info


# =======================================================================
# Comparison metric.
# =======================================================================
def rel_err(S_red, S_ref):
    """max_t ||S_red - S_ref||_F / (1 + ||S_ref||_F)."""
    T = S_red.shape[0]
    worst = 0.0
    for t in range(T):
        d = float(jnp.linalg.norm((S_red[t] - S_ref[t]).reshape(-1)))
        n = float(jnp.linalg.norm(S_ref[t].reshape(-1)))
        worst = max(worst, d / (1.0 + n))
    return worst


FAMILIES = ("random_stable", "real_distinct", "complex_conjugate", "lambda_I",
            "repeated_eigs", "exact_jordan", "multi_jordan_shared",
            "nearly_defective", "nonnormal", "stiff")


def run_case(family, r, seed):
    theta, a, B, z0, xs, info = make_case(family, r, seed)
    zs = rollout(theta, a, B, z0, xs, r)
    max_z = float(jnp.max(jnp.abs(zs)))
    if (not np.isfinite(max_z)) or max_z > OVERFLOW_LIMIT:
        return dict(family=family, r=r, seed=seed, overflow=True, max_z=max_z, **info)

    S_th_red, S_a_red, S_b_red = reduced_sensitivities(theta, a, B, z0, xs, r, M_INPUTS)
    S_th_full, S_a_full, S_b_full = full_rtrl_sensitivities(theta, a, B, z0, xs, r, M_INPUTS)
    S_th_ad, S_a_ad, S_b_ad = autodiff_sensitivities(theta, a, B, z0, xs, r)

    return dict(
        family=family, r=r, seed=seed, overflow=False, max_z=max_z,
        theta_vs_rtrl=rel_err(S_th_red, S_th_full),
        a_vs_rtrl=rel_err(S_a_red, S_a_full),
        B_vs_rtrl=rel_err(S_b_red, S_b_full),
        theta_vs_ad=rel_err(S_th_red, S_th_ad),
        a_vs_ad=rel_err(S_a_red, S_a_ad),
        B_vs_ad=rel_err(S_b_red, S_b_ad),
        rtrl_vs_ad_theta=rel_err(S_th_full, S_th_ad),
        **info)


# =======================================================================
# REPRESENTATION SIDE (independent of the sensitivity tests).
#
# Test R1 (u(x) = x, the classical controllable canonical form):
#   For a SISO minimal (A, b_*, c_*) let K = [b, Ab, ..., A^{r-1}b] be the
#   controllability matrix. Cayley-Hamilton gives A^r b = -sum_j a_j A^j b,
#   i.e. K a = -A^r b, which we SOLVE for a (rather than going through
#   ill-conditioned eigenvalue/np.poly root-finding). Then A K = K C_q
#   exactly, so A = K C_q K^{-1} with T = K, u(x) = x.
#   Markov parameters: c_* A^k b_* = (c_*K) C_q^k (K^{-1} b_*).
#
# Test R2 (nontrivial u): build A := T u(C_q) T^{-1} for random T, q, u
#   and verify the realization identity and Markov parameters directly.
#
# STRUCTURAL NOTE (reported, not hidden): a SINGLE quotient R[x]/(q) is
# always NONDEROGATORY -- C_q has exactly one Jordan block per distinct
# root. Multiple Jordan blocks sharing an eigenvalue are reachable for
# M_u = u(C_q) (see the multi_jordan_shared family) but NOT every Jordan
# structure is reachable from a single quotient of a given size; e.g.
# from q=(x-lam)^4 the attainable structures for u(C_q) are J_4, J_2+J_2
# and J_1^4, but not J_2+J_1+J_1. Realizing arbitrary derogatory
# structures needs a PRODUCT of quotients, not a single one.
# =======================================================================
def markov_params(A, b, c, K_max):
    out, v = [], b.copy()
    for _ in range(K_max):
        out.append(float(c @ v))
        v = A @ v
    return np.array(out)


def test_representation_R1(r, seed, K_max=60):
    rng = np.random.RandomState(seed)
    A = rng.randn(r, r) / np.sqrt(r)
    A = A * (0.9 / max(np.abs(np.linalg.eigvals(A)).max(), 1e-30))
    b = rng.randn(r)
    c = rng.randn(r)

    K = np.stack([np.linalg.matrix_power(A, k) @ b for k in range(r)], axis=1)
    condK = float(np.linalg.cond(K))
    a_vec = np.linalg.solve(K, -(np.linalg.matrix_power(A, r) @ b))
    Cq = np.asarray(companion(jnp.array(a_vec), r))

    err_similar = float(np.linalg.norm(A @ K - K @ Cq) / (1 + np.linalg.norm(A @ K)))
    err_recon = float(np.linalg.norm(A - K @ Cq @ np.linalg.inv(K)) / (1 + np.linalg.norm(A)))

    B_t = np.linalg.solve(K, b)
    C_t = c @ K
    mk_true = markov_params(A, b, c, K_max)
    mk_quot = markov_params(Cq, B_t, C_t, K_max)
    scale = 1.0 + np.max(np.abs(mk_true))
    err_markov = float(np.max(np.abs(mk_true - mk_quot)) / scale)
    return dict(r=r, seed=seed, cond_K=condK, err_similarity=err_similar,
                err_reconstruction=err_recon, err_markov=err_markov)


def test_representation_R2(r, seed, K_max=60):
    rng = np.random.RandomState(seed + 5000)
    theta, a, B, z0, xs, info = make_case("random_stable", r, seed + 777)
    Cq = np.asarray(companion(a, r))
    Mu = np.asarray(mult_matrix(theta, a, r))
    T = rng.randn(r, r) / np.sqrt(r)
    condT = float(np.linalg.cond(T))
    A = T @ Mu @ np.linalg.inv(T)

    err_recon = float(np.linalg.norm(A - T @ Mu @ np.linalg.inv(T)) / (1 + np.linalg.norm(A)))
    Mu_horner = np.zeros((r, r))
    for k in range(r - 1, -1, -1):
        Mu_horner = Mu_horner @ Cq + float(theta[k]) * np.eye(r)
    err_uCq = float(np.linalg.norm(Mu - Mu_horner) / (1 + np.linalg.norm(Mu)))

    b_star = rng.randn(r)
    c_star = rng.randn(r)
    B_t = np.linalg.solve(T, b_star)
    C_t = c_star @ T
    mk_true = markov_params(A, b_star, c_star, K_max)
    mk_quot = markov_params(Mu, B_t, C_t, K_max)
    scale = 1.0 + np.max(np.abs(mk_true))
    err_markov = float(np.max(np.abs(mk_true - mk_quot)) / scale)
    return dict(r=r, seed=seed, cond_T=condT, err_reconstruction=err_recon,
                err_u_of_Cq=err_uCq, err_markov=err_markov)


# =======================================================================
# Main.
# =======================================================================
def main():
    print("=" * 100)
    print("B37a -- universal quotient recurrence: reduced eligibilities vs full RTRL vs BPTT/autodiff")
    print(f"dtype=float64 (eps={np.finfo(np.float64).eps:.3e})  T={T_STEPS}  m={M_INPUTS}  "
          f"PASS threshold < {PASS_THRESHOLD:g}")
    print("=" * 100)

    rows, failures, overflows = [], [], []
    hdr = (f"{'family':22s} {'r':>3s} {'sd':>3s} {'rho(Mu)':>9s} {'cond(Mu)':>10s} "
           f"{'th|RTRL':>9s} {'a|RTRL':>9s} {'B|RTRL':>9s} {'th|AD':>9s} {'a|AD':>9s} {'B|AD':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for family in FAMILIES:
        for r in R_VALUES:
            for seed in (0, 1):
                res = run_case(family, r, seed)
                rows.append(res)
                if res["overflow"]:
                    overflows.append(res)
                    print(f"{family:22s} {r:3d} {seed:3d}   OVERFLOW/non-finite (max|z|={res['max_z']:.3e}) -- excluded")
                    continue
                errs = [res[k] for k in ("theta_vs_rtrl", "a_vs_rtrl", "B_vs_rtrl",
                                          "theta_vs_ad", "a_vs_ad", "B_vs_ad")]
                if max(errs) >= PASS_THRESHOLD:
                    failures.append(res)
                print(f"{family:22s} {r:3d} {seed:3d} {res['rho_Mu']:9.4f} {res['cond_Mu']:10.2e} "
                      f"{res['theta_vs_rtrl']:9.2e} {res['a_vs_rtrl']:9.2e} {res['B_vs_rtrl']:9.2e} "
                      f"{res['theta_vs_ad']:9.2e} {res['a_vs_ad']:9.2e} {res['B_vs_ad']:9.2e}")

    finite = [r_ for r_ in rows if not r_["overflow"]]
    worst = {k: max(r_[k] for r_ in finite) for k in
             ("theta_vs_rtrl", "a_vs_rtrl", "B_vs_rtrl", "theta_vs_ad", "a_vs_ad", "B_vs_ad")}
    print("-" * len(hdr))
    print("WORST OVER ALL NON-OVERFLOWING CASES:")
    for k, v in worst.items():
        print(f"   {k:16s} {v:.3e}   {'PASS' if v < PASS_THRESHOLD else 'FAIL'}")
    print(f"cases: {len(rows)} total, {len(finite)} non-overflowing, {len(overflows)} overflow, "
          f"{len(failures)} FAILURES")

    print("\n" + "=" * 100)
    print("REPRESENTATION SIDE")
    print("=" * 100)
    rep1, rep2 = [], []
    print("R1: u(x)=x, controllable canonical form  A = K C_q K^-1  (K = controllability matrix)")
    print(f"  {'r':>3s} {'sd':>3s} {'cond(K)':>11s} {'||AK-KCq||rel':>14s} {'||A-KCqK^-1||rel':>17s} {'markov err':>12s}")
    for r in R_VALUES:
        for seed in (0, 1):
            res = test_representation_R1(r, seed)
            rep1.append(res)
            print(f"  {res['r']:3d} {res['seed']:3d} {res['cond_K']:11.3e} {res['err_similarity']:14.3e} "
                  f"{res['err_reconstruction']:17.3e} {res['err_markov']:12.3e}")
    print("\nR2: nontrivial u,  A := T u(C_q) T^-1")
    print(f"  {'r':>3s} {'sd':>3s} {'cond(T)':>11s} {'||A-TuT^-1||rel':>16s} {'||Mu-u(Cq)||rel':>16s} {'markov err':>12s}")
    for r in R_VALUES:
        for seed in (0, 1):
            res = test_representation_R2(r, seed)
            rep2.append(res)
            print(f"  {res['r']:3d} {res['seed']:3d} {res['cond_T']:11.3e} {res['err_reconstruction']:16.3e} "
                  f"{res['err_u_of_Cq']:16.3e} {res['err_markov']:12.3e}")

    with open("/tmp/b37a_results.json", "w") as f:
        json.dump(dict(sensitivity=rows, representation_R1=rep1, representation_R2=rep2,
                       worst=worst, threshold=PASS_THRESHOLD, T=T_STEPS, m=M_INPUTS,
                       dtype="float64", eps=float(np.finfo(np.float64).eps)),
                  f, indent=2, default=str)
    print("\nSaved /tmp/b37a_results.json")

    print("\n" + "=" * 100)
    verdict = (len(failures) == 0)
    print(f"VERDICT (sensitivity exactness): {'PASS' if verdict else 'FAIL'} -- "
          f"all reduced traces match full RTRL and BPTT/autodiff below {PASS_THRESHOLD:g} "
          f"on all {len(finite)} non-overflowing cases." if verdict else
          f"VERDICT: FAIL -- {len(failures)} case(s) exceeded {PASS_THRESHOLD:g}.")
    if failures:
        sm = min(failures, key=lambda d: (d["r"], d["family"]))
        print(f"SMALLEST FAILING EXAMPLE: family={sm['family']} r={sm['r']} seed={sm['seed']}")
        print(f"  {sm}")
    print("=" * 100)
    return rows, failures


if __name__ == "__main__":
    main()
