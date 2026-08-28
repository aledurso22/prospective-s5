"""Phase B25 -- nonlinear temporal-credit separation.

Architecture:  h_{t+1} = (I_n (x) R) h_t + (I_n (x) B) Phi_psi(z_t, u_t),
               z_t = (I_n (x) C) h_t

R: r x r stable dense (tied across n copies). B: r x k. C: k x r.
Phi_psi: a genuine MLP mixing ALL n*k entries of z_t (and u_t)
nonlinearly -- this is what gives width-dependent nonlinear capacity.

Convention throughout: h_t stored as (n, r) -- copy index OUTER, temporal
coordinate INNER (standard reshape of the (I_n (x) *) Kronecker
structure). JAX (float64) supplies exact autodiff Jacobians and an
independent BPTT reference; the factorized forward-RTRL recurrence
itself is the novel object under test.

Run: python -m credit_memory.b25_nonlinear_credit
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Architecture construction.
# ---------------------------------------------------------------------------
def make_stable_dense(r, rng, radius=0.85):
    M = rng.randn(r, r) / np.sqrt(r)
    eig = np.max(np.abs(np.linalg.eigvals(M)))
    return M * (radius / eig)


def make_psi(n, k, u_dim, hidden, rng):
    in_dim = n * k + u_dim
    out_dim = n * k
    scale1 = 1.0 / np.sqrt(in_dim)
    scale2 = 1.0 / np.sqrt(hidden)
    return dict(
        W1=jnp.array(rng.randn(hidden, in_dim) * scale1),
        b1=jnp.array(rng.randn(hidden) * 0.1),
        W2=jnp.array(rng.randn(out_dim, hidden) * scale2),
        b2=jnp.array(rng.randn(out_dim) * 0.1),
    )


def psi_flat(psi):
    return jnp.concatenate([psi["W1"].ravel(), psi["b1"], psi["W2"].ravel(), psi["b2"]])


def psi_from_flat(flat, n, k, u_dim, hidden):
    in_dim = n * k + u_dim
    out_dim = n * k
    i = 0
    W1 = flat[i:i + hidden * in_dim].reshape(hidden, in_dim); i += hidden * in_dim
    b1 = flat[i:i + hidden]; i += hidden
    W2 = flat[i:i + out_dim * hidden].reshape(out_dim, hidden); i += out_dim * hidden
    b2 = flat[i:i + out_dim]
    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def Phi(z_flat, u, psi):
    """z_flat: (n*k,). u: (u_dim,). Returns (n*k,)."""
    x = jnp.concatenate([z_flat, u])
    h = jnp.tanh(psi["W1"] @ x + psi["b1"])
    return psi["W2"] @ h + psi["b2"]


def make_arch(r, k, n, u_dim, hidden, seed):
    rng = np.random.RandomState(seed)
    R = make_stable_dense(r, rng)
    B = rng.randn(r, k) / np.sqrt(k) * 0.8
    C = rng.randn(k, r) / np.sqrt(r) * 0.8
    psi = make_psi(n, k, u_dim, hidden, rng)
    return dict(R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), psi=psi,
                r=r, k=k, n=n, u_dim=u_dim, hidden=hidden)


def forward_step(h, u, R, B, C, psi):
    """h: (n,r). u: (u_dim,). Returns (h_next (n,r), z_flat (n*k,), phi_out (n*k,))."""
    n = h.shape[0]
    k = C.shape[0]
    z = h @ C.T                      # (n,k)
    z_flat = z.reshape(-1)
    phi_out = Phi(z_flat, u, psi)    # (n*k,)
    phi_r = phi_out.reshape(n, k)
    h_next = h @ R.T + phi_r @ B.T   # (n,r)
    return h_next, z_flat, phi_out


def rollout(h0, U, arch):
    """U: (T, u_dim). Returns H (T+1,n,r) states, Z (T,n*k), Y (T, n*k) phi outputs."""
    R, B, C, psi = arch["R"], arch["B"], arch["C"], arch["psi"]
    h = h0
    Hs = [h]
    Zs = []
    Ys = []
    for t in range(U.shape[0]):
        h, z, y = forward_step(h, U[t], R, B, C, psi)
        Hs.append(h)
        Zs.append(z)
        Ys.append(y)
    return jnp.stack(Hs), jnp.stack(Zs), jnp.stack(Ys)


# ---------------------------------------------------------------------------
# Part 1: the exact Jacobian identity.
#   J_Phi,t = dPhi/dz  (n*k x n*k, via JAX autodiff)
#   F_ab,t[p,q] = J_Phi,t[(p,a),(q,b)]   (unique expansion J_Phi,t = sum_ab F_ab (x) E_ab)
#   Q_ab = B E_ab C   (r x r, time-independent)
#   J_t := dh_{t+1}/dh_t  ==  I_n(x)R + sum_ab F_ab,t (x) Q_ab   -- verified against
#   a DIRECT full autodiff Jacobian of the forward step.
# ---------------------------------------------------------------------------
def make_E(k, a, b):
    E = np.zeros((k, k))
    E[a, b] = 1.0
    return E


def make_Q_all(B, C, k):
    """Returns array (k,k,r,r) of Q_ab = B @ E_ab @ C."""
    B_np, C_np = np.asarray(B), np.asarray(C)
    r = B_np.shape[0]
    Q = np.zeros((k, k, r, r))
    for a in range(k):
        for b in range(k):
            Q[a, b] = B_np @ make_E(k, a, b) @ C_np
    return Q


def compute_F_ab(h, u, arch):
    """Exact autodiff Jacobian of Phi wrt z, decomposed into F_ab (n,n) blocks."""
    n, k = arch["n"], arch["k"]
    C, psi = arch["C"], arch["psi"]
    z_flat = (h @ C.T).reshape(-1)
    J_Phi = jax.jacobian(lambda zz: Phi(zz, u, psi))(z_flat)  # (nk, nk)
    J4 = np.asarray(J_Phi).reshape(n, k, n, k)                # [p,a,q,b]
    F = np.transpose(J4, (1, 3, 0, 2))                        # [a,b,p,q]
    return F  # (k,k,n,n)


def predicted_jacobian(h, u, arch):
    n, k, r = arch["n"], arch["k"], arch["r"]
    R = np.asarray(arch["R"])
    Q = make_Q_all(arch["B"], arch["C"], k)   # (k,k,r,r)
    F = compute_F_ab(h, u, arch)              # (k,k,n,n)
    J = np.kron(np.eye(n), R)
    for a in range(k):
        for b in range(k):
            J = J + np.kron(F[a, b], Q[a, b])
    return J


def direct_jacobian(h, u, arch):
    R, B, C, psi = arch["R"], arch["B"], arch["C"], arch["psi"]
    n, r = h.shape
    f = lambda hh: forward_step(hh.reshape(n, r), u, R, B, C, psi)[0].reshape(-1)
    return np.asarray(jax.jacobian(f)(h.reshape(-1)))


def verify_jacobian_identity(arch, h, u):
    J_pred = predicted_jacobian(h, u, arch)
    J_direct = direct_jacobian(h, u, arch)
    return float(np.max(np.abs(J_pred - J_direct)))


# ---------------------------------------------------------------------------
# Part 2: the temporal algebra Alg{R, Q_ab}. d_T := dim of this algebra
# (a subspace of r x r matrices, closed under multiplication). Also:
#   rho   = dim K(R,B)     -- Krylov/reachability subspace of B under R
#   omega = dim K(R^T,C^T) -- observability subspace
#   deg(mu_R) = degree of R's minimal polynomial
# Bound: d_T <= min(r^2, deg(mu_R) + rho*omega).
# ---------------------------------------------------------------------------
def _orth_add(basis_mats, new_mats, tol=1e-9):
    """basis_mats: list of r x r matrices (orthonormal in Frobenius inner
    product, as flattened vectors). new_mats: candidates to add. Returns
    the updated orthonormal basis (list) via incremental QR/SVD."""
    r = new_mats[0].shape[0]
    flat_basis = [m.ravel() for m in basis_mats]
    for cand in new_mats:
        v = cand.ravel().astype(float)
        for b in flat_basis:
            v = v - (b @ v) * b
        nrm = np.linalg.norm(v)
        if nrm > tol:
            flat_basis.append(v / nrm)
    return [v.reshape(r, r) for v in flat_basis]


def algebra_closure(generators, tol=1e-9, max_iters=200):
    """generators: list of r x r matrices. Returns orthonormal basis
    (list of r x r matrices) of the smallest matrix SPAN containing the
    generators and closed under multiplication (a genuine algebra)."""
    basis = _orth_add([], generators, tol)
    for _ in range(max_iters):
        products = []
        for A in basis:
            for Bm in basis:
                products.append(A @ Bm)
        new_basis = _orth_add(basis, products, tol)
        if len(new_basis) == len(basis):
            return new_basis
        basis = new_basis
    return basis


def krylov_subspace(R, seed_cols, tol=1e-9):
    """Smallest R-invariant subspace of R^r containing span(seed_cols).
    seed_cols: r x m matrix (columns are seeds). Returns orthonormal
    basis matrix V (r x d)."""
    r = R.shape[0]
    vecs = [seed_cols[:, i] for i in range(seed_cols.shape[1])]
    basis = []
    for v in vecs:
        w = v.astype(float).copy()
        for b in basis:
            w = w - (b @ w) * b
        nrm = np.linalg.norm(w)
        if nrm > tol:
            basis.append(w / nrm)
    frontier = list(basis)
    for _ in range(r):
        new_frontier = []
        for v in frontier:
            w = (R @ v).astype(float)
            for b in basis:
                w = w - (b @ w) * b
            nrm = np.linalg.norm(w)
            if nrm > tol:
                basis.append(w / nrm)
                new_frontier.append(basis[-1])
        if not new_frontier:
            break
        frontier = new_frontier
    return np.stack(basis, axis=1) if basis else np.zeros((r, 0))


def minimal_poly_degree(R, tol=1e-9):
    r = R.shape[0]
    I = np.eye(r)
    basis = _orth_add([], [I])
    cur = I
    deg = 0
    for _ in range(r):
        cur = cur @ R
        new_basis = _orth_add(basis, [cur], tol)
        if len(new_basis) == len(basis):
            return deg + 1  # cur is dependent -> minimal poly has this degree
        basis = new_basis
        deg += 1
    return deg


def part2_temporal_algebra(arch):
    r, k = arch["r"], arch["k"]
    R = np.asarray(arch["R"])
    B = np.asarray(arch["B"])
    C = np.asarray(arch["C"])
    Q = make_Q_all(B, C, k)
    gens = [R] + [Q[a, b] for a in range(k) for b in range(k)]
    alg_basis = algebra_closure(gens)
    d_T = len(alg_basis)

    K_RB = krylov_subspace(R, B)
    rho = K_RB.shape[1]
    K_RtCt = krylov_subspace(R.T, C.T)
    omega = K_RtCt.shape[1]
    deg_mu = minimal_poly_degree(R)

    bound = min(r * r, deg_mu + rho * omega)
    return dict(d_T=d_T, rho=rho, omega=omega, deg_mu_R=deg_mu, bound=bound,
                bound_holds=(d_T <= bound))


# ---------------------------------------------------------------------------
# Part 3/4: forward RTRL -- naive (full r-dim basis, V=I_r) and factorized
# (basis V of the theory-predicted V_theta). Both driven by the SAME
# recurrence, parameterized by V; naive is simply the d=r special case.
# Direct per-step source terms d(h_{t+1})/d(theta)|_{h fixed} are computed
# via JAX autodiff (not hand-derived) to avoid re-deriving formulas that
# could introduce bugs independent of the algorithm under test.
# ---------------------------------------------------------------------------
def direct_term(family, h, u, arch):
    """Returns (n, r, m) exact d(h_next)/d(theta)|_{h fixed} for the given
    family in {'R','B','C','psi'}, m = number of scalar params in family."""
    R, B, C, psi = arch["R"], arch["B"], arch["C"], arch["psi"]
    n, r, k = arch["n"], arch["r"], arch["k"]
    if family == "R":
        def f(flat):
            return forward_step(h, u, flat.reshape(r, r), B, C, psi)[0].reshape(-1)
        J = jax.jacobian(f)(R.reshape(-1))
        m = r * r
    elif family == "B":
        def f(flat):
            return forward_step(h, u, R, flat.reshape(r, k), C, psi)[0].reshape(-1)
        J = jax.jacobian(f)(B.reshape(-1))
        m = r * k
    elif family == "C":
        def f(flat):
            return forward_step(h, u, R, B, flat.reshape(k, r), psi)[0].reshape(-1)
        J = jax.jacobian(f)(C.reshape(-1))
        m = k * r
    elif family == "psi":
        flat0 = psi_flat(psi)
        def f(flat):
            psi_ = psi_from_flat(flat, n, k, arch["u_dim"], arch["hidden"])
            return forward_step(h, u, R, B, C, psi_)[0].reshape(-1)
        J = jax.jacobian(f)(flat0)
        m = flat0.shape[0]
    else:
        raise ValueError(family)
    return np.asarray(J).reshape(n, r, m)


def family_dim(family, arch):
    r, k = arch["r"], arch["k"]
    if family == "R":
        return r * r
    if family == "B":
        return r * k
    if family == "C":
        return k * r
    if family == "psi":
        return psi_flat(arch["psi"]).shape[0]
    raise ValueError(family)


def basis_for_family(family, arch):
    """Returns V (r x d) -- theory-predicted V_theta. R,B: full r-dim
    (identity). C,psi: K(R,B) (Krylov reachability subspace of B)."""
    r = arch["r"]
    R = np.asarray(arch["R"])
    B = np.asarray(arch["B"])
    if family in ("R", "B"):
        return np.eye(r)
    if family in ("C", "psi"):
        V = krylov_subspace(R, B)
        return V if V.shape[1] > 0 else np.zeros((r, 0))
    raise ValueError(family)


def factorized_rtrl_run(family, arch, h0, U, dLdh_seq, use_naive=False):
    """Runs the forward-only recurrence (naive if use_naive else factorized)
    for one parameter family, accumulating the gradient contribution
    sum_t <dL/dh_{t+1}, S_{t+1}> as it goes -- O(1) memory in T beyond
    the running X_t / grad accumulator (no history cache)."""
    r, k, n = arch["r"], arch["k"], arch["n"]
    R = np.asarray(arch["R"])
    Q = make_Q_all(arch["B"], arch["C"], k)  # (k,k,r,r)
    V = np.eye(r) if use_naive else basis_for_family(family, arch)
    d = V.shape[1]
    m = family_dim(family, arch)

    Rmat = V.T @ R @ V                                    # (d,d)
    Qmat = np.einsum("ig,abij,jh->abgh", V, Q, V)          # (k,k,d,d)

    X = np.zeros((n, d, m))
    grad = np.zeros(m)
    h = h0
    T_ = U.shape[0]
    for t in range(T_):
        u_t = U[t]
        F = compute_F_ab(h, u_t, arch)                     # (k,k,n,n)
        Direct = direct_term(family, h, u_t, arch)          # (n,r,m)
        U_t = np.einsum("ig,pic->pgc", V, Direct)           # (n,d,m)

        term_R = np.einsum("gh,phc->pgc", Rmat, X)
        term_Q = np.einsum("abpq,abgh,qhc->pgc", F, Qmat, X)
        X = term_R + term_Q + U_t

        h_next, _, _ = forward_step(h, u_t, arch["R"], arch["B"], arch["C"], arch["psi"])
        S_next = np.einsum("ig,pgc->pic", V, X)             # reconstruct (n,r,m)
        grad += np.einsum("pi,pic->c", np.asarray(dLdh_seq[t]), S_next)
        h = h_next
    return grad, d


def bptt_reference_grads(arch, h0, U, target_fn):
    """target_fn(H) -> scalar loss, H: (T+1,n,r). Genuine autodiff BPTT
    via JAX -- the independent final reference."""
    R, B, C, psi = arch["R"], arch["B"], arch["C"], arch["psi"]
    flat0 = psi_flat(psi)

    def loss_of(Rm, Bm, Cm, psi_flat_):
        psi_ = psi_from_flat(psi_flat_, arch["n"], arch["k"], arch["u_dim"], arch["hidden"])
        arch_ = dict(arch, R=Rm, B=Bm, C=Cm, psi=psi_)
        H, _, _ = rollout(h0, U, arch_)
        return target_fn(H)

    gR, gB, gC, gpsi = jax.grad(loss_of, argnums=(0, 1, 2, 3))(R, B, C, flat0)
    return dict(R=np.asarray(gR).reshape(-1), B=np.asarray(gB).reshape(-1),
                C=np.asarray(gC).reshape(-1), psi=np.asarray(gpsi))


def combined_factorized_grads(arch, h0, U, dLdh_seq, use_naive=False):
    """Runs the SAME per-timestep loop once, computing all four families'
    forward-RTRL gradients together (shares the F_ab,t computation)."""
    r, k, n = arch["r"], arch["k"], arch["n"]
    R = np.asarray(arch["R"])
    Q = make_Q_all(arch["B"], arch["C"], k)
    families = ("R", "B", "C", "psi")
    Vs = {f: (np.eye(r) if use_naive else basis_for_family(f, arch)) for f in families}
    ds = {f: Vs[f].shape[1] for f in families}
    ms = {f: family_dim(f, arch) for f in families}
    Rmats = {f: Vs[f].T @ R @ Vs[f] for f in families}
    Qmats = {f: np.einsum("ig,abij,jh->abgh", Vs[f], Q, Vs[f]) for f in families}
    Xs = {f: np.zeros((n, ds[f], ms[f])) for f in families}
    grads = {f: np.zeros(ms[f]) for f in families}

    h = h0
    for t in range(U.shape[0]):
        u_t = U[t]
        F = compute_F_ab(h, u_t, arch)
        h_next, _, _ = forward_step(h, u_t, arch["R"], arch["B"], arch["C"], arch["psi"])
        dLdh_t = np.asarray(dLdh_seq[t])
        for f in families:
            Direct = direct_term(f, h, u_t, arch)
            U_t = np.einsum("ig,pic->pgc", Vs[f], Direct)
            term_R = np.einsum("gh,phc->pgc", Rmats[f], Xs[f])
            term_Q = np.einsum("abpq,abgh,qhc->pgc", F, Qmats[f], Xs[f])
            Xs[f] = term_R + term_Q + U_t
            S_next = np.einsum("ig,pgc->pic", Vs[f], Xs[f])
            grads[f] += np.einsum("pi,pic->c", dLdh_t, S_next)
        h = h_next
    return grads


def apply_grad_update(arch, grads, lr):
    r, k = arch["r"], arch["k"]
    R = np.asarray(arch["R"]) - lr * np.clip(grads["R"].reshape(r, r), -1, 1)
    B = np.asarray(arch["B"]) - lr * np.clip(grads["B"].reshape(r, k), -1, 1)
    C = np.asarray(arch["C"]) - lr * np.clip(grads["C"].reshape(k, r), -1, 1)
    flat0 = psi_flat(arch["psi"])
    flat_new = flat0 - lr * jnp.clip(grads["psi"], -1, 1)
    psi_new = psi_from_flat(flat_new, arch["n"], arch["k"], arch["u_dim"], arch["hidden"])
    return dict(arch, R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), psi=psi_new)


# ---------------------------------------------------------------------------
# Part 7: width-sweep memory accounting -- temporal quantities (d_T, rho,
# omega, per-family d_theta) vs. feature/coefficient storage (n*d*m for
# factorized, n*r*m for naive).
# ---------------------------------------------------------------------------
def memory_report(arch):
    n, r = arch["n"], arch["r"]
    rows = {}
    for f in ("R", "B", "C", "psi"):
        d = basis_for_family(f, arch).shape[1]
        m = family_dim(f, arch)
        rows[f] = dict(d_theta=d, m=m, naive_floats=n * r * m, factorized_floats=n * d * m)
    return rows


# ---------------------------------------------------------------------------
# Part 8: nonlinear capacity check. Teacher = a fixed, wider instance of
# the SAME architecture (genuinely nonlinear Phi). Approximators at the
# SAME (r,k) but growing n, trained via the factorized forward-RTRL
# gradients themselves (a real use, not just a checked-but-unused
# formula). Verify loss decreases with width.
# ---------------------------------------------------------------------------
def make_teacher_targets(r, k, n_teacher, u_dim, hidden, T_, seed):
    rng = np.random.RandomState(seed)
    teacher = make_arch(r, k, n_teacher, u_dim, hidden, seed=seed)
    h0 = jnp.array(rng.randn(n_teacher, r) * 0.3)
    U = jnp.array(rng.randn(T_, u_dim) * 0.4)
    H, Z, _ = rollout(h0, U, teacher)
    # scalar external readout: sum over teacher's n*k features per step
    y = jnp.sum(Z, axis=1)  # (T,)
    return U, y


def train_approximator(r, k, n, u_dim, hidden, U, y_target, steps, lr, seed):
    arch = make_arch(r, k, n, u_dim, hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.array(rng.randn(n, r) * 0.2)
    T_ = U.shape[0]

    losses = []
    for _ in range(steps):
        dLdh = dLdh_from_target(arch, h0, U, lambda H: (
            0.5 * jnp.sum((jnp.sum((H[1:] @ arch["C"].T).reshape(T_, -1), axis=1) - y_target) ** 2) / T_))
        grads = combined_factorized_grads(arch, h0, U, dLdh, use_naive=False)
        arch = apply_grad_update(arch, grads, lr)
        H, _, _ = rollout(h0, U, arch)
        Z = (H[1:] @ arch["C"].T).reshape(T_, -1)
        y = jnp.sum(Z, axis=1)
        loss = float(0.5 * jnp.sum((y - y_target) ** 2) / T_)
        losses.append(loss)
    return losses


def dLdh_from_target(arch, h0, U, target_fn):
    """dL/dh_{t+1} for t=0..T-1, via JAX autodiff on the target function
    applied to the actual rollout (used to drive the forward-RTRL
    gradient accumulation identically across naive/factorized methods)."""
    def loss_of_H(H):
        return target_fn(H)
    H, _, _ = rollout(h0, U, arch)
    g = jax.grad(loss_of_H)(H)   # (T+1,n,r)
    return np.asarray(g[1:])     # dL/dh_1 .. dL/dh_T


# ---------------------------------------------------------------------------
# Part 9 (scoped): multi-layer (L=2,3) naive-RTRL-vs-BPTT check. Each
# layer is its own instance of the SAME single-layer architecture;
# layer l's z output feeds layer l+1's external input u (standard
# feedforward-through-time stacking). Built generically via JAX autodiff
# Jacobians (full per-step state Jacobian + full per-step direct-param
# Jacobian of the WHOLE stack) -- no new per-layer algebra derivation
# needed for the NAIVE reference; the fully FACTORIZED deep-prefix
# extension (predicted bound d_temp_{j<-i} <= k_i*sum_{l=i}^j r_l) is
# flagged, not implemented, in the phase writeup -- an explicit scope
# limit given time budget, not a silent gap.
# ---------------------------------------------------------------------------
def stack_forward(hs, u_ext, archs):
    lower = u_ext
    new_hs = []
    zs = []
    for l, arch in enumerate(archs):
        h_next, z, _ = forward_step(hs[l], lower, arch["R"], arch["B"], arch["C"], arch["psi"])
        new_hs.append(h_next)
        zs.append(z)
        lower = z
    return new_hs, zs


def stack_rollout(h0s, U, archs):
    hs = h0s
    Hs = [hs]
    Zs = []
    for t in range(U.shape[0]):
        hs, zs = stack_forward(hs, U[t], archs)
        Hs.append(hs)
        Zs.append(zs)
    return Hs, Zs


def flatten_hs(hs):
    return jnp.concatenate([h.reshape(-1) for h in hs])


def unflatten_hs(flat, shapes):
    out = []
    i = 0
    for (n, r) in shapes:
        out.append(flat[i:i + n * r].reshape(n, r))
        i += n * r
    return out


def stack_naive_rtrl_grad(archs, h0s, U, family_layer, dLdh_flat_seq):
    """Naive forward RTRL for the WHOLE stack, for one (layer_idx,family)
    parameter, using JAX-computed per-step full state Jacobian and
    direct-parameter Jacobian (no algebra reduction -- the NAIVE
    reference at depth)."""
    shapes = [h.shape for h in h0s]

    def stack_step_flat(h_flat, u, layer_params):
        hs = unflatten_hs(h_flat, shapes)
        archs_ = [dict(a, **({k: v for k, v in lp.items()} if lp else {}))
                  for a, lp in zip(archs, layer_params)]
        new_hs, _ = stack_forward(hs, u, archs_)
        return flatten_hs(new_hs)

    layer_idx, family = family_layer
    if family == "R":
        theta0 = archs[layer_idx]["R"].reshape(-1)
        shape = archs[layer_idx]["R"].shape
    elif family == "B":
        theta0 = archs[layer_idx]["B"].reshape(-1)
        shape = archs[layer_idx]["B"].shape
    elif family == "C":
        theta0 = archs[layer_idx]["C"].reshape(-1)
        shape = archs[layer_idx]["C"].shape
    else:
        raise ValueError(family)
    m = theta0.shape[0]

    h_flat = flatten_hs(h0s)
    total_dim = h_flat.shape[0]
    S = np.zeros((total_dim, m))
    grad = np.zeros(m)
    for t in range(U.shape[0]):
        u_t = U[t]

        def f_h(hh):
            return stack_step_flat(hh, u_t, [{} for _ in archs])

        def f_theta(th):
            lp = [{} for _ in archs]
            lp[layer_idx] = {family: th.reshape(shape)}
            return stack_step_flat(h_flat, u_t, lp)

        J_h = np.asarray(jax.jacobian(f_h)(h_flat))
        J_theta = np.asarray(jax.jacobian(f_theta)(theta0))
        S = J_h @ S + J_theta
        h_flat = np.asarray(f_h(h_flat))
        grad += dLdh_flat_seq[t] @ S
    return grad


def stack_bptt_grad(archs, h0s, U, family_layer, target_fn):
    layer_idx, family = family_layer

    def loss_of(theta):
        archs_ = [dict(a) for a in archs]
        archs_[layer_idx] = dict(archs_[layer_idx], **{family: theta})
        Hs, _ = stack_rollout(h0s, U, archs_)
        return target_fn(Hs)

    theta0 = archs[layer_idx][family]
    g = jax.grad(loss_of)(theta0)
    return np.asarray(g).reshape(-1)


def part6_gauge_test(seed=21):
    rng = np.random.RandomState(seed)
    arch = make_arch(r=4, k=2, n=3, u_dim=2, hidden=6, seed=3)
    r = arch["r"]
    h0 = jnp.array(rng.randn(3, r) * 0.3)
    T_ = 4
    U = jnp.array(rng.randn(T_, 2) * 0.4)

    Traw = rng.randn(r, r)
    Uu, Ss, Vt = np.linalg.svd(Traw)
    Ss = np.clip(Ss, 0.3, 3.0)
    T = Uu @ np.diag(Ss) @ Vt
    Tinv = np.linalg.inv(T)

    arch_new = dict(arch, R=jnp.array(T @ np.asarray(arch["R"]) @ Tinv),
                     B=jnp.array(T @ np.asarray(arch["B"])),
                     C=jnp.array(np.asarray(arch["C"]) @ Tinv))
    h0_new = jnp.array(np.asarray(h0) @ T.T)

    H_old, Z_old, _ = rollout(h0, U, arch)
    H_new, Z_new, _ = rollout(h0_new, U, arch_new)
    z_err = float(np.max(np.abs(np.asarray(Z_old) - np.asarray(Z_new))))
    h_pred_new = np.einsum("ij,tpj->tpi", T, np.asarray(H_old))
    h_err = float(np.max(np.abs(h_pred_new - np.asarray(H_new))))

    res_old = part2_temporal_algebra(arch)
    res_new = part2_temporal_algebra(arch_new)
    algebra_match = (res_old["d_T"] == res_new["d_T"] and res_old["rho"] == res_new["rho"]
                      and res_old["omega"] == res_new["omega"])

    target_z = rng.randn(T_, 3, 2) * 0.3

    def make_target_fn(Cm):
        tgt = jnp.array(target_z)
        def target_fn(H):
            Z = H[1:] @ Cm.T
            return 0.5 * jnp.sum((Z - tgt) ** 2) / T_
        return target_fn

    tf_old, tf_new = make_target_fn(arch["C"]), make_target_fn(arch_new["C"])
    bptt_old = bptt_reference_grads(arch, h0, U, tf_old)
    bptt_new = bptt_reference_grads(arch_new, h0_new, U, tf_new)

    gR_pred = (T.T @ bptt_new["R"].reshape(r, r) @ Tinv.T).reshape(-1)
    gB_pred = (T.T @ bptt_new["B"].reshape(r, arch["k"])).reshape(-1)
    gC_pred = (bptt_new["C"].reshape(arch["k"], r) @ Tinv.T).reshape(-1)
    bptt_pullback_err = dict(
        R=float(np.max(np.abs(gR_pred - bptt_old["R"]))),
        B=float(np.max(np.abs(gB_pred - bptt_old["B"]))),
        C=float(np.max(np.abs(gC_pred - bptt_old["C"]))),
        psi=float(np.max(np.abs(bptt_new["psi"] - bptt_old["psi"]))),
    )

    dLdh_old = dLdh_from_target(arch, h0, U, tf_old)
    dLdh_new = dLdh_from_target(arch_new, h0_new, U, tf_new)
    fact_pullback_err = {}
    for family in ("R", "B", "C"):
        g_old, _ = factorized_rtrl_run(family, arch, h0, U, dLdh_old, use_naive=False)
        g_new, _ = factorized_rtrl_run(family, arch_new, h0_new, U, dLdh_new, use_naive=False)
        if family == "R":
            pred = (T.T @ g_new.reshape(r, r) @ Tinv.T).reshape(-1)
        elif family == "B":
            pred = (T.T @ g_new.reshape(r, arch["k"])).reshape(-1)
        else:
            pred = (g_new.reshape(arch["k"], r) @ Tinv.T).reshape(-1)
        fact_pullback_err[family] = float(np.max(np.abs(pred - g_old)))

    return dict(z_err=z_err, h_err=h_err, algebra_match=algebra_match,
                bptt_pullback_err=bptt_pullback_err, fact_pullback_err=fact_pullback_err)


def stack_dLdh_flat(archs, h0s, U, target_fn):
    shapes = [h.shape for h in h0s]

    def loss_of_flat_seq(flat_seq):
        Hs = [unflatten_hs(flat_seq[t], shapes) for t in range(flat_seq.shape[0])]
        return target_fn(Hs)

    Hs, _ = stack_rollout(h0s, U, archs)
    flat_seq = jnp.stack([flatten_hs(hs) for hs in Hs])
    g = jax.grad(loss_of_flat_seq)(flat_seq)
    return np.asarray(g[1:])


def main():
    print("=" * 70)
    print("PART 1 -- exact Jacobian identity")
    print("J_t == I(x)R + sum_ab F_ab,t(x)Q_ab  vs. direct autodiff Jacobian")
    print("=" * 70)
    for (r, k, n) in [(4, 1, 2), (4, 2, 4), (4, 2, 8), (3, 1, 16)]:
        arch = make_arch(r=r, k=k, n=n, u_dim=2, hidden=16, seed=0)
        rng = np.random.RandomState(2)
        h = jnp.array(rng.randn(n, r) * 0.4)
        u = jnp.array(rng.randn(2) * 0.5)
        err = verify_jacobian_identity(arch, h, u)
        print(f"  r={r} k={k} n={n:2d}: max|J_pred - J_direct| = {err:.2e}")

    print()
    print("=" * 70)
    print("PART 2 -- temporal algebra Alg{R,Q_ab}: d_T, rho, omega, bound")
    print("Sweep n at fixed r,k -- must be n-INDEPENDENT")
    print("=" * 70)
    for k in (1, 2):
        for n in (2, 4, 8, 16):
            arch = make_arch(r=4, k=k, n=n, u_dim=2, hidden=16, seed=5)
            res = part2_temporal_algebra(arch)
            print(f"  r=4 k={k} n={n:2d}: d_T={res['d_T']:2d} rho={res['rho']} "
                  f"omega={res['omega']} deg_mu_R={res['deg_mu_R']} "
                  f"bound={res['bound']} holds={res['bound_holds']}")

    print()
    print("=" * 70)
    print("PART 3/4 -- naive forward RTRL vs. factorized forward RTRL vs. BPTT")
    print("(generic case: rho=r, no reduction expected)")
    print("=" * 70)
    arch = make_arch(r=3, k=1, n=2, u_dim=1, hidden=4, seed=3)
    rng = np.random.RandomState(4)
    h0 = jnp.array(rng.randn(2, 3) * 0.3)
    T_ = 4
    U = jnp.array(rng.randn(T_, 1) * 0.4)
    target = jnp.array(rng.randn(T_ + 1, 2, 3) * 0.3)
    target_fn = lambda H: 0.5 * jnp.sum((H - target) ** 2) / T_
    dLdh = dLdh_from_target(arch, h0, U, target_fn)
    bptt = bptt_reference_grads(arch, h0, U, target_fn)
    for family in ("R", "B", "C", "psi"):
        g_naive, d_naive = factorized_rtrl_run(family, arch, h0, U, dLdh, use_naive=True)
        g_fact, d_fact = factorized_rtrl_run(family, arch, h0, U, dLdh, use_naive=False)
        e_nb = np.max(np.abs(g_naive - bptt[family]))
        e_fb = np.max(np.abs(g_fact - bptt[family]))
        print(f"  {family}: d_naive={d_naive} d_fact={d_fact}  "
              f"|naive-bptt|={e_nb:.2e}  |fact-bptt|={e_fb:.2e}")

    print()
    print("=" * 70)
    print("PART 5 -- genuine dimensional reduction (degenerate R, rho<r)")
    print("=" * 70)
    rng = np.random.RandomState(9)
    r = 5
    lam = 0.6
    D = np.diag([lam, lam, lam, 0.3, -0.4])
    Sm = rng.randn(r, r)
    Rdeg = np.linalg.solve(Sm, D) @ Sm
    arch = make_arch(r=r, k=1, n=2, u_dim=1, hidden=4, seed=3)
    arch = dict(arch, R=jnp.array(Rdeg))
    res2 = part2_temporal_algebra(arch)
    print(f"  degenerate R: d_T={res2['d_T']} rho={res2['rho']} omega={res2['omega']} "
          f"deg_mu_R={res2['deg_mu_R']}")
    h0 = jnp.array(rng.randn(2, r) * 0.3)
    U = jnp.array(rng.randn(T_, 1) * 0.4)
    target = jnp.array(rng.randn(T_ + 1, 2, r) * 0.3)
    target_fn = lambda H: 0.5 * jnp.sum((H - target) ** 2) / T_
    dLdh = dLdh_from_target(arch, h0, U, target_fn)
    bptt = bptt_reference_grads(arch, h0, U, target_fn)
    for family in ("C", "psi"):
        g_naive, d_naive = factorized_rtrl_run(family, arch, h0, U, dLdh, use_naive=True)
        g_fact, d_fact = factorized_rtrl_run(family, arch, h0, U, dLdh, use_naive=False)
        e_fb = np.max(np.abs(g_fact - bptt[family]))
        print(f"  {family}: d_naive={d_naive} d_fact={d_fact} (REDUCED)  |fact-bptt|={e_fb:.2e}")

    print()
    print("=" * 70)
    print("PART 6 -- gauge test")
    print("=" * 70)
    g = part6_gauge_test()
    print(f"  z invariance error: {g['z_err']:.2e}")
    print(f"  h transform consistency error: {g['h_err']:.2e}")
    print(f"  temporal algebra dims unchanged: {g['algebra_match']}")
    print(f"  BPTT gradient pullback errors: {g['bptt_pullback_err']}")
    print(f"  factorized-RTRL gradient pullback errors: {g['fact_pullback_err']}")

    print()
    print("=" * 70)
    print("PART 7 -- width sweep: temporal (n-independent) vs. feature/coefficient")
    print("storage (grows with n) -- reported separately")
    print("=" * 70)
    for n in (2, 4, 8, 16):
        arch = make_arch(r=4, k=2, n=n, u_dim=2, hidden=8, seed=5)
        res = part2_temporal_algebra(arch)
        mem = memory_report(arch)
        total_naive = sum(v["naive_floats"] for v in mem.values())
        total_fact = sum(v["factorized_floats"] for v in mem.values())
        print(f"  n={n:2d}: TEMPORAL d_T={res['d_T']} rho={res['rho']} omega={res['omega']} "
              f"(n-independent)  |  FEATURE naive_floats={total_naive} "
              f"factorized_floats={total_fact}")

    print()
    print("=" * 70)
    print("PART 8 -- nonlinear capacity check: does task quality improve with n?")
    print("=" * 70)
    U, y_target = make_teacher_targets(r=3, k=1, n_teacher=6, u_dim=1, hidden=8, T_=15, seed=50)
    for n in (1, 2, 4, 8):
        losses = train_approximator(r=3, k=1, n=n, u_dim=1, hidden=8, U=U, y_target=y_target,
                                     steps=150, lr=0.05, seed=1)
        print(f"  n={n:2d}: final loss = {losses[-1]:.6f}")

    print()
    print("=" * 70)
    print("PART 9 (scoped) -- naive forward RTRL vs BPTT at depth L=2,3")
    print("(NAIVE reference only, via autodiff Jacobians; the fully")
    print("factorized deep-prefix extension is flagged as future work,")
    print("not implemented here -- explicit scope limit, see PHASE_B25.md)")
    print("=" * 70)
    rng = np.random.RandomState(30)
    r, k = 3, 1
    n0, n1 = 2, 2
    arch0 = make_arch(r=r, k=k, n=n0, u_dim=1, hidden=4, seed=10)
    arch1 = make_arch(r=r, k=k, n=n1, u_dim=n0 * k, hidden=4, seed=11)
    archs = [arch0, arch1]
    h0s = [jnp.array(rng.randn(n0, r) * 0.3), jnp.array(rng.randn(n1, r) * 0.3)]
    T_ = 4
    U = jnp.array(rng.randn(T_, 1) * 0.4)
    target = jnp.array(rng.randn(T_, n1, r) * 0.3)
    target_fn = lambda Hs: 0.5 * jnp.sum(
        (jnp.stack([Hs[t][1] for t in range(1, T_ + 1)]) - target) ** 2) / T_
    dLdh_flat = stack_dLdh_flat(archs, h0s, U, target_fn)
    for layer_idx, family in [(0, "R"), (1, "R"), (0, "B"), (1, "C")]:
        g_naive = stack_naive_rtrl_grad(archs, h0s, U, (layer_idx, family), dLdh_flat)
        g_bptt = stack_bptt_grad(archs, h0s, U, (layer_idx, family), target_fn)
        err = np.max(np.abs(g_naive - g_bptt))
        print(f"  L=2 layer{layer_idx}.{family}: |naive-bptt|={err:.2e}")

    rng = np.random.RandomState(31)
    ns = [2, 2, 2]
    u_dims = [1, ns[0] * k, ns[1] * k]
    archs3 = [make_arch(r=r, k=k, n=ns[i], u_dim=u_dims[i], hidden=4, seed=20 + i) for i in range(3)]
    h0s3 = [jnp.array(rng.randn(ns[i], r) * 0.3) for i in range(3)]
    U3 = jnp.array(rng.randn(T_, 1) * 0.4)
    target3 = jnp.array(rng.randn(T_, ns[2], r) * 0.3)
    target_fn3 = lambda Hs: 0.5 * jnp.sum(
        (jnp.stack([Hs[t][2] for t in range(1, T_ + 1)]) - target3) ** 2) / T_
    dLdh_flat3 = stack_dLdh_flat(archs3, h0s3, U3, target_fn3)
    for layer_idx, family in [(0, "R"), (1, "B"), (2, "C"), (0, "C")]:
        g_naive = stack_naive_rtrl_grad(archs3, h0s3, U3, (layer_idx, family), dLdh_flat3)
        g_bptt = stack_bptt_grad(archs3, h0s3, U3, (layer_idx, family), target_fn3)
        err = np.max(np.abs(g_naive - g_bptt))
        print(f"  L=3 layer{layer_idx}.{family}: |naive-bptt|={err:.2e}")


if __name__ == "__main__":
    main()
