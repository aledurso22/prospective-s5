"""Phase B25.1 -- deep nonlinear factorized prefix RTRL.

Composes B25's local factorized temporal bases through depth for the
SAME architecture (no new architecture ideas):

  h_{l,t+1} = (I(x)R_l) h_{l,t} + (I(x)B_l) Phi_l(z_{l,t}, u_{l,t}),
  z_{l,t} = (I(x)C_l) h_{l,t},  u_{l,t} = z_{l-1,t}  (l>0)

Foundational new identity (the cross-layer analog of B25 Part 1):
  dh_{l+1,t+1}/dh_{l,t} = sum_ab G_ab,t (x) P_ab,
  G_ab,t = Jacobian of Phi_{l+1} wrt its u-argument, decomposed the
  same way as B25's F_ab,t; P_ab = B_{l+1} E_ab C_l (r_{l+1} x r_l).

A source's factorized sensitivity at its own layer i lives in V_{i<-i}
(B25's local basis: K(R_i,B_i) for C/psi sources, full r_i for R/B
sources). Propagated into layer l>i, it is seeded by the IMAGE of
V_{l-1<-i} under the cross-layer P_ab operators, then closed under
layer l's OWN {R_l, Q_ab^(l)} (needed in general, not just R_l alone --
see PHASE_B25_1.md Part 1 for why).

Run: python -m credit_memory.b25_1_deep_prefix
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b25_nonlinear_credit import (
    make_arch, forward_step, Phi, make_E, make_Q_all, compute_F_ab,
    basis_for_family, family_dim, direct_term, krylov_subspace,
    stack_forward, stack_rollout, flatten_hs, unflatten_hs,
    stack_bptt_grad, stack_naive_rtrl_grad, stack_dLdh_flat,
    part2_temporal_algebra,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Cross-layer Jacobian decomposition and P_ab operators.
# ---------------------------------------------------------------------------
def make_P_all(B_upper, C_lower, k_upper, k_lower):
    """P_ab = B_upper @ E_ab @ C_lower, a in 1..k_upper, b in 1..k_lower.
    Returns (k_upper,k_lower,r_upper,r_lower)."""
    B_np, C_np = np.asarray(B_upper), np.asarray(C_lower)
    r_upper = B_np.shape[0]
    r_lower = C_np.shape[1]
    P = np.zeros((k_upper, k_lower, r_upper, r_lower))
    for a in range(k_upper):
        for b in range(k_lower):
            P[a, b] = B_np[:, a:a + 1] @ C_np[b:b + 1, :]
    return P


def compute_G_ab(z_lower, z_upper, arch_upper, n_upper, k_upper, n_lower, k_lower):
    """Cross-derivative: Jacobian of Phi_upper wrt its u-argument
    (=z_lower), decomposed into (k_upper,k_lower,n_upper,n_lower) blocks
    the same way as compute_F_ab decomposes dPhi/dz."""
    G_full = jax.jacobian(lambda uu: Phi(z_upper, uu, arch_upper["psi"]))(z_lower)
    G4 = np.asarray(G_full).reshape(n_upper, k_upper, n_lower, k_lower)  # [p,a,q,b]
    return np.transpose(G4, (1, 3, 0, 2))  # [a,b,p,q]


def z_direct_term(h, arch):
    """d(z)/dC|_{h fixed} -- ONLY nonzero for family='C': z_t=(I(x)C)h_t
    depends on C directly (not just via h_t's own dependence on C),
    a pathway distinct from direct_term's d(h_next)/dC and easy to miss
    when only h's own sensitivity is propagated cross-layer. Returns
    (n,k,m), m=k*r."""
    n, r = h.shape
    k = arch["k"]
    def f(Cflat):
        return (h @ Cflat.reshape(k, r).T).reshape(-1)
    J = jax.jacobian(f)(arch["C"].reshape(-1))
    return np.asarray(J).reshape(n, k, k * r)


def verify_cross_layer_identity(arch_lower, arch_upper, h_lower, h_upper, u_ext):
    """Direct autodiff check of dh_upper_next/dh_lower vs sum_ab G_ab(x)P_ab."""
    n_lower, r_lower = h_lower.shape
    n_upper, r_upper = h_upper.shape
    k_lower, k_upper = arch_lower["k"], arch_upper["k"]

    def full_step_flat(hl_flat, hu_flat):
        hl = hl_flat.reshape(n_lower, r_lower)
        hu = hu_flat.reshape(n_upper, r_upper)
        hln, zl, _ = forward_step(hl, u_ext, arch_lower["R"], arch_lower["B"],
                                   arch_lower["C"], arch_lower["psi"])
        hun, zu, _ = forward_step(hu, zl, arch_upper["R"], arch_upper["B"],
                                   arch_upper["C"], arch_upper["psi"])
        return hln.reshape(-1), hun.reshape(-1)

    J_off = np.asarray(jax.jacobian(
        lambda hh: full_step_flat(hh, h_upper.reshape(-1))[1])(h_lower.reshape(-1)))
    J_off = J_off.reshape(n_upper, r_upper, n_lower, r_lower)

    z_lower = (h_lower @ arch_lower["C"].T).reshape(-1)
    z_upper = (h_upper @ arch_upper["C"].T).reshape(-1)
    G_ab = compute_G_ab(z_lower, z_upper, arch_upper, n_upper, k_upper, n_lower, k_lower)
    P_ab = make_P_all(arch_upper["B"], arch_lower["C"], k_upper, k_lower)

    J_pred = np.zeros((n_upper, r_upper, n_lower, r_lower))
    for a in range(k_upper):
        for b in range(k_lower):
            J_pred += np.einsum("pq,ij->piqj", G_ab[a, b], P_ab[a, b])
    return float(np.max(np.abs(J_pred - J_off)))


# ---------------------------------------------------------------------------
# Subspace closure under an arbitrary set of generator matrices (needed
# here, unlike B25's R-only krylov_subspace: a seed inside im(B_l) is
# NOT generally closed under R_l alone once k_l>1 -- Q_ab^(l) can route
# it to a different direction within im(B_l) that R_l-closure of the
# specific seed might not reach).
# ---------------------------------------------------------------------------
def subspace_closure(matrices, seed_vecs, tol=1e-9, max_iters=None):
    if not seed_vecs:
        r = matrices[0].shape[0] if matrices else 0
        return np.zeros((r, 0))
    r = seed_vecs[0].shape[0]
    max_iters = max_iters or r
    basis = []
    for v in seed_vecs:
        w = np.asarray(v, dtype=float).copy()
        for b in basis:
            w = w - (b @ w) * b
        nrm = np.linalg.norm(w)
        if nrm > tol:
            basis.append(w / nrm)
    frontier = list(basis)
    for _ in range(max_iters):
        new_frontier = []
        for v in frontier:
            for M in matrices:
                w = M @ v
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


# ---------------------------------------------------------------------------
# Deep prefix basis construction: V_{l<-i} for l=i..L-1.
# V_{i<-i} = B25's local basis (K(R_i,B_i) for C/psi sources, full r_i
# for R/B sources). For l>i, seeded by the image of V_{l-1<-i} under the
# cross-layer P_ab^(l) operators, closed under layer l's own {R_l,Q_ab^(l)}.
# ---------------------------------------------------------------------------
def build_prefix_bases(archs, source_layer, family):
    L = len(archs)
    bases = {}
    bases[source_layer] = basis_for_family(family, archs[source_layer])
    for l in range(source_layer + 1, L):
        V_prev = bases[l - 1]
        if V_prev.shape[1] == 0:
            bases[l] = np.zeros((archs[l]["r"], 0))
            continue
        k_l, k_prev = archs[l]["k"], archs[l - 1]["k"]
        P = make_P_all(archs[l]["B"], archs[l - 1]["C"], k_l, k_prev)  # (k_l,k_prev,r_l,r_prev)
        seed_vecs = []
        for a in range(k_l):
            for b in range(k_prev):
                img = P[a, b] @ V_prev  # (r_l, d_prev)
                for c in range(img.shape[1]):
                    seed_vecs.append(img[:, c])
        R_l = np.asarray(archs[l]["R"])
        Q_l = make_Q_all(archs[l]["B"], archs[l]["C"], k_l)
        gens = [R_l] + [Q_l[a, b] for a in range(k_l) for b in range(k_l)]
        bases[l] = subspace_closure(gens, seed_vecs)
    return bases  # dict layer -> (r_l, d_{l<-i}) basis matrix


# ---------------------------------------------------------------------------
# The deep factorized forward-RTRL recurrence itself. At the source
# layer i: B25's local recursion (+ local direct term). At l>i: local
# recursion (own F_ab,t, Q_ab^(l)) PLUS the cross-layer injection from
# layer l-1's own factorized state, via G_ab,t and the precomputed
# Pmat_ab^(l) structure constants. NO full n*r-by-parameter tensor is
# ever formed -- every object tracked is (n_l, d_{l<-i}, m).
# ---------------------------------------------------------------------------
def deep_factorized_grad(archs, h0s, U, source_layer, family, dLdh_flat_seq):
    L = len(archs)
    bases = build_prefix_bases(archs, source_layer, family)
    shapes = [(archs[l]["n"], archs[l]["r"]) for l in range(L)]
    m = family_dim(family, archs[source_layer])

    Rmats, Qmats = {}, {}
    for l in range(source_layer, L):
        V = bases[l]
        R_l = np.asarray(archs[l]["R"])
        Q_l = make_Q_all(archs[l]["B"], archs[l]["C"], archs[l]["k"])
        Rmats[l] = V.T @ R_l @ V
        Qmats[l] = np.einsum("ig,abij,jh->abgh", V, Q_l, V) if V.shape[1] > 0 else \
            np.zeros((archs[l]["k"], archs[l]["k"], 0, 0))

    Pmats = {}
    for l in range(source_layer + 1, L):
        V_l, V_prev = bases[l], bases[l - 1]
        k_l, k_prev = archs[l]["k"], archs[l - 1]["k"]
        if V_l.shape[1] == 0 or V_prev.shape[1] == 0:
            Pmats[l] = np.zeros((k_l, k_prev, V_l.shape[1], V_prev.shape[1]))
            continue
        P = make_P_all(archs[l]["B"], archs[l - 1]["C"], k_l, k_prev)
        Pmats[l] = np.einsum("ig,abij,jh->abgh", V_l, P, V_prev)

    Xs = {l: np.zeros((archs[l]["n"], bases[l].shape[1], m)) for l in range(source_layer, L)}
    grad = np.zeros(m)

    hs = h0s
    for t in range(U.shape[0]):
        u_t = U[t]
        new_hs, zs = stack_forward(hs, u_t, archs)
        lower_input = [u_t] + zs[:-1]  # lower_input[l] = the u fed to layer l
        dLdh_t = unflatten_hs(dLdh_flat_seq[t], shapes)

        # Snapshot ALL layers' X BEFORE any update this timestep -- every
        # propagation (local R/Q and cross-layer) must read dh_{*,t}/dtheta
        # (pre-update), never a sibling layer's already-advanced
        # dh_{*,t+1}/dtheta from earlier in this same timestep's loop.
        # (Bug caught here: T=1 sanity check gave a nonzero gradient where
        # BPTT gives exactly 0, since R0 cannot yet affect z0_0=C0@h0_0 at
        # the very first step -- traced to reading the just-updated
        # Xs[l-1] instead of its pre-update value.)
        Xs_old = {l: Xs[l].copy() for l in Xs}
        Xs_new = {}

        V_src = bases[source_layer]
        F_src = compute_F_ab(hs[source_layer], lower_input[source_layer], archs[source_layer])
        if V_src.shape[1] > 0:
            Direct = direct_term(family, hs[source_layer], lower_input[source_layer],
                                  archs[source_layer])
            U_t = np.einsum("ig,pic->pgc", V_src, Direct)
            term_R = np.einsum("gh,phc->pgc", Rmats[source_layer], Xs_old[source_layer])
            term_Q = np.einsum("abpq,abgh,qhc->pgc", F_src, Qmats[source_layer],
                                Xs_old[source_layer])
            Xs_new[source_layer] = term_R + term_Q + U_t
        else:
            Xs_new[source_layer] = Xs_old[source_layer]

        for l in range(source_layer + 1, L):
            if bases[l].shape[1] == 0:
                Xs_new[l] = Xs_old[l]
                continue
            F_l = compute_F_ab(hs[l], lower_input[l], archs[l])
            term_R = np.einsum("gh,phc->pgc", Rmats[l], Xs_old[l])
            term_Q = np.einsum("abpq,abgh,qhc->pgc", F_l, Qmats[l], Xs_old[l])
            term_cross = np.zeros_like(Xs_old[l])
            if bases[l - 1].shape[1] > 0:
                G_l = compute_G_ab(lower_input[l], zs[l], archs[l], archs[l]["n"], archs[l]["k"],
                                    archs[l - 1]["n"], archs[l - 1]["k"])
                term_cross = np.einsum("abpq,abgh,qhc->pgc", G_l, Pmats[l], Xs_old[l - 1])
            else:
                G_l = None
            # Extra pathway, ONLY at the first hop out of a family='C'
            # source: z_{src,t}=(I(x)C_src)h_{src,t} depends on C_src
            # DIRECTLY (holding h_src,t fixed), separate from h_src,t's
            # own dependence on C_src already carried by Xs_old[l-1].
            if l == source_layer + 1 and family == "C":
                if G_l is None:
                    G_l = compute_G_ab(lower_input[l], zs[l], archs[l], archs[l]["n"],
                                        archs[l]["k"], archs[l - 1]["n"], archs[l - 1]["k"])
                Zextra = z_direct_term(hs[source_layer], archs[source_layer])  # (n_src,k_src,m)
                B_l = np.asarray(archs[l]["B"])
                raw_extra = np.einsum("ia,abpq,qbc->pic", B_l, G_l, Zextra)
                term_cross = term_cross + np.einsum("ig,pic->pgc", bases[l], raw_extra)
            Xs_new[l] = term_R + term_Q + term_cross

        Xs = Xs_new
        for l in range(source_layer, L):
            if bases[l].shape[1] == 0:
                continue
            S_next = np.einsum("ig,pgc->pic", bases[l], Xs[l])
            grad += np.einsum("pi,pic->c", dLdh_t[l], S_next)

        hs = new_hs
    return grad, {l: bases[l].shape[1] for l in bases}


# ---------------------------------------------------------------------------
# Part 4: width-sweep memory accounting. Prefix basis dims (temporal)
# are computed purely from R,B,C,Q,P -- never touch n -- so they are
# n-independent by construction; reported alongside actual coefficient
# storage (which does scale with n) for the deep case.
# ---------------------------------------------------------------------------
def deep_memory_report(archs, source_layer, family):
    bases = build_prefix_bases(archs, source_layer, family)
    m = family_dim(family, archs[source_layer])
    naive_floats = 0
    fact_floats = 0
    for l in range(source_layer, len(archs)):
        n_l, r_l = archs[l]["n"], archs[l]["r"]
        d_l = bases[l].shape[1]
        naive_floats += n_l * r_l * m
        fact_floats += n_l * d_l * m
    return dict(dims={l: bases[l].shape[1] for l in bases}, naive_floats=naive_floats,
                factorized_floats=fact_floats)


def main():
    print("=" * 70)
    print("FOUNDATION -- cross-layer Jacobian identity")
    print("dh_{l+1,next}/dh_l == sum_ab G_ab,t (x) P_ab, vs direct autodiff")
    print("=" * 70)
    for (r0, k0, n0, r1, k1, n1) in [(3, 2, 2, 4, 1, 3), (4, 1, 4, 3, 2, 2), (2, 2, 8, 2, 2, 8)]:
        arch0 = make_arch(r=r0, k=k0, n=n0, u_dim=1, hidden=6, seed=1)
        arch1 = make_arch(r=r1, k=k1, n=n1, u_dim=n0 * k0, hidden=6, seed=2)
        rng = np.random.RandomState(3)
        h0 = jnp.array(rng.randn(n0, r0) * 0.3)
        h1 = jnp.array(rng.randn(n1, r1) * 0.3)
        u_ext = jnp.array(rng.randn(1) * 0.4)
        err = verify_cross_layer_identity(arch0, arch1, h0, h1, u_ext)
        print(f"  r0={r0} k0={k0} n0={n0} r1={r1} k1={k1} n1={n1}: err={err:.2e}")

    print()
    print("=" * 70)
    print("PART 2 -- three-way exactness (naive vs deep factorized vs BPTT)")
    print("L=2,3,4; sources at earliest/middle/final layer; all families")
    print("=" * 70)
    r, k = 3, 1
    for L in (2, 3, 4):
        ns = [2] * L
        u_dims = [1] + [ns[i] * k for i in range(L - 1)]
        archs = [make_arch(r=r, k=k, n=ns[i], u_dim=u_dims[i], hidden=4, seed=10 + i)
                  for i in range(L)]
        rng = np.random.RandomState(30 + L)
        h0s = [jnp.array(rng.randn(ns[i], r) * 0.3) for i in range(L)]
        T_ = 4
        U = jnp.array(rng.randn(T_, 1) * 0.4)
        target = jnp.array(rng.randn(T_, ns[-1], r) * 0.3)
        target_fn = lambda Hs: 0.5 * jnp.sum(
            (jnp.stack([Hs[t][L - 1] for t in range(1, T_ + 1)]) - target) ** 2) / T_
        dLdh_flat = stack_dLdh_flat(archs, h0s, U, target_fn)
        print(f"  --- L={L} ---")
        tests = [(0, "R"), (0, "C"), (0, "psi")]
        if L >= 2:
            tests += [(1, "R"), (1, "C")]
        if L >= 3:
            tests += [(L - 1, "C")]
        for layer_idx, family in tests:
            g_naive = stack_naive_rtrl_grad(archs, h0s, U, (layer_idx, family), dLdh_flat)
            g_bptt = stack_bptt_grad(archs, h0s, U, (layer_idx, family), target_fn)
            g_deep, dims = deep_factorized_grad(archs, h0s, U, layer_idx, family, dLdh_flat)
            err_nb = np.max(np.abs(g_naive - g_bptt))
            err_db = np.max(np.abs(g_deep - g_bptt))
            print(f"    layer{layer_idx}.{family}: dims={dims}  "
                  f"|naive-bptt|={err_nb:.2e}  |deep-bptt|={err_db:.2e}")

    print()
    print("=" * 70)
    print("PART 3 -- structural finding: downstream propagated dim is bounded")
    print("by the DOWNSTREAM layer's own rho_l, regardless of source family")
    print("=" * 70)
    rng = np.random.RandomState(40)
    r0, k0, n0 = 5, 1, 2
    r1, k1, n1 = 5, 1, 2
    arch0 = make_arch(r=r0, k=k0, n=n0, u_dim=1, hidden=4, seed=1)
    arch1 = make_arch(r=r1, k=k1, n=n1, u_dim=n0 * k0, hidden=4, seed=2)
    lam = 0.5
    D = np.diag([lam, lam, lam, 0.2, -0.3])
    Sm = rng.randn(r1, r1)
    arch1 = dict(arch1, R=jnp.array(np.linalg.solve(Sm, D) @ Sm))
    res1 = part2_temporal_algebra(arch1)
    print(f"  layer1 degenerate: rho1={res1['rho']} r1={r1} (layer0 fully generic)")
    archs = [arch0, arch1]
    h0s = [jnp.array(rng.randn(n0, r0) * 0.3), jnp.array(rng.randn(n1, r1) * 0.3)]
    T_ = 4
    U = jnp.array(rng.randn(T_, 1) * 0.4)
    target = jnp.array(rng.randn(T_, n1, r1) * 0.3)
    target_fn = lambda Hs: 0.5 * jnp.sum(
        (jnp.stack([Hs[t][1] for t in range(1, T_ + 1)]) - target) ** 2) / T_
    dLdh_flat = stack_dLdh_flat(archs, h0s, U, target_fn)
    for family in ("R", "B", "C", "psi"):
        g_bptt = stack_bptt_grad(archs, h0s, U, (0, family), target_fn)
        g_deep, dims = deep_factorized_grad(archs, h0s, U, 0, family, dLdh_flat)
        err_db = np.max(np.abs(g_deep - g_bptt))
        print(f"  source=layer0.{family}: dims={dims}  |deep-bptt|={err_db:.2e}")

    print()
    print("=" * 70)
    print("PART 5 -- genuine reduction: source-only vs all-layers degenerate")
    print("=" * 70)
    def degenerate_R(r, rng, n_repeat=3, lam=0.5):
        diag = [lam] * n_repeat + list(rng.uniform(0.2, 0.4, r - n_repeat))
        D = np.diag(diag)
        Sm = rng.randn(r, r)
        return np.linalg.solve(Sm, D) @ Sm

    rng = np.random.RandomState(41)
    r = 5
    archs_src_only = [make_arch(r=r, k=1, n=2, u_dim=(1 if i == 0 else 2), hidden=4, seed=10 + i)
                       for i in range(3)]
    archs_src_only[0] = dict(archs_src_only[0], R=jnp.array(degenerate_R(r, rng)))
    bases_src = build_prefix_bases(archs_src_only, 0, "C")
    print(f"  source-only degenerate: dims={ {l: bases_src[l].shape[1] for l in bases_src} } "
          f"(reduction does NOT persist through generic downstream layers)")

    rng = np.random.RandomState(42)
    archs_all_deg = []
    for i in range(3):
        a = make_arch(r=r, k=1, n=2, u_dim=(1 if i == 0 else 2), hidden=4, seed=10 + i)
        a = dict(a, R=jnp.array(degenerate_R(r, rng)))
        archs_all_deg.append(a)
    bases_all = build_prefix_bases(archs_all_deg, 0, "C")
    print(f"  ALL layers degenerate: dims={ {l: bases_all[l].shape[1] for l in bases_all} } "
          f"(reduction persists -- genuine 3-layer saving)")
    h0s = [jnp.array(rng.randn(2, r) * 0.3) for _ in range(3)]
    T_ = 4
    U = jnp.array(rng.randn(T_, 1) * 0.4)
    target = jnp.array(rng.randn(T_, 2, r) * 0.3)
    target_fn = lambda Hs: 0.5 * jnp.sum(
        (jnp.stack([Hs[t][2] for t in range(1, T_ + 1)]) - target) ** 2) / T_
    dLdh_flat = stack_dLdh_flat(archs_all_deg, h0s, U, target_fn)
    g_naive = stack_naive_rtrl_grad(archs_all_deg, h0s, U, (0, "C"), dLdh_flat)
    g_bptt = stack_bptt_grad(archs_all_deg, h0s, U, (0, "C"), target_fn)
    g_deep, dims = deep_factorized_grad(archs_all_deg, h0s, U, 0, "C", dLdh_flat)
    print(f"  exactness in the all-degenerate case: dims={dims}  "
          f"|naive-bptt|={np.max(np.abs(g_naive-g_bptt)):.2e}  "
          f"|deep-bptt|={np.max(np.abs(g_deep-g_bptt)):.2e}")
    mem = deep_memory_report(archs_all_deg, 0, "C")
    print(f"  memory: naive_floats={mem['naive_floats']} factorized_floats={mem['factorized_floats']}")

    print()
    print("=" * 70)
    print("PART 4 -- width sweep: temporal prefix dims (n-independent) vs")
    print("coefficient storage (grows with n), reported separately")
    print("=" * 70)
    for n in (2, 4, 8, 16):
        archs_w = [make_arch(r=3, k=1, n=n, u_dim=(1 if i == 0 else n * 1), hidden=4, seed=50 + i)
                   for i in range(3)]
        mem = deep_memory_report(archs_w, 0, "C")
        print(f"  n={n:2d}: TEMPORAL dims={mem['dims']} (n-independent)  |  "
              f"FEATURE naive_floats={mem['naive_floats']} factorized_floats={mem['factorized_floats']}")


if __name__ == "__main__":
    main()
