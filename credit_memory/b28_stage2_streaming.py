"""Phase B28 Stage 2 -- streaming RTRL + streaming RL, kept SEPARATE.

Per review item 2: recurrent RTRL and the outer streaming RL
eligibility trace are TWO DISTINCT mechanisms, not one vaguely named
"eligibility". The decomposition made explicit here:

  recurrent RTRL (persistent S_t = dh_t/dtheta, exact, forward-only)
    -> exact per-step gradient g_t = d(local_loss_t)/dtheta
    -> streaming RL eligibility z_t = gamma*lambda*z_{t-1} + g_t
    -> TD/actor-critic parameter update: theta += alpha * delta_t * z_t

This file provides the STREAMING (per-step, stateful) RTRL API for
both architectures -- factorized_rtrl_run/rtu_exact_credit_grad (B25,
B27) process a WHOLE trajectory at once; a genuine online RL loop
needs ONE step at a time, with S_t persisting as running state across
calls.

WHAT "NAIVE FULL-r" MEANS HERE, PRECISELY (per review request): this
is the ambient/generic full-RTRL representation, V_theta = I_r (the
r-dim identity), for EVERY family (R, B, C, psi) -- NOT the compressed
coefficient-basis algorithm from B25/B26. In the theory as originally
built (basis_for_family, b25_nonlinear_credit.py): families R,B
already use V=I_r in BOTH the "naive" and the "factorized/reduced"
mode (basis_for_family returns np.eye(r) for R,B unconditionally --
there is no compression available for these two families in the first
place). Families C,psi are where the reduced theory normally uses
V=K(R,B) (the Krylov/reachability subspace of B under R, dimension
<= r) instead of the full r-dim identity. use_naive=True (this file's
choice, for ALL four families) sets V=I_r for C,psi TOO, discarding
that reduction. Checked empirically for the Stage-2 architecture
(r=4,k=2, u_dim=4/6, 3 seeds): dim K(R,B) = r = 4 in every case tested
at initialization -- i.e. for this specific generic random init, the
reachability subspace already happens to be full rank, so the naive
and theoretically-reduced representations COINCIDE numerically here
(no compression is actually being lost to this specific set of
architectures at t=0). This has NOT been re-verified to hold at every
point along an online training trajectory (R,B drift under updates);
the naive/ambient choice was made specifically so correctness does not
depend on that holding.

Per-family live tensor shape (naive, ALL families): X[family] has
shape (n, r, m_family) where m_family = family_dim(family, arch) is
that family's own raw parameter count (r*r for R, r*k for B, k*r for
C, len(psi_flat) for psi) -- i.e. this is literally d(h_t)/d(theta),
the FULL (n,r,m_family) Jacobian block, with no basis compression
applied. Total persistent credit floats = n*r*(m_R+m_B+m_C+m_psi) =
n*r*param_count (the naive-representation storage cost actually used
by this streaming implementation -- an upper bound on, and for this
architecture numerically equal to, what the reduced/factorized
algorithm would need for the same trajectory). This must NOT be
attributed to "the compressed coefficient-basis algorithm" in any
Stage-2 comparison -- it is the correctness/robustness baseline
representation, reported as its own line item.

RELATION TO A_T, stated to avoid conflating two distinct objects:
A_T = Alg{R, Q_ab} (B27's temporal algebra, dimension d_T <= r^2) is a
STRUCTURAL/EXPRESSIVITY quantity -- whether the recurrent map's
generated matrix algebra is the full r x r matrix algebra (globally
coupled, B27's irreducibility argument) or a proper subalgebra
(independent blocks, e.g. RTU). K(R,B) (the reachability subspace used
for the C/psi families' theoretical credit-basis reduction) is a
separate, CREDIT-STORAGE quantity -- how large a subspace of R^r a
given family's sensitivity actually needs to span. Neither implies the
other in general; this file's naive representation (V_theta=I_r
throughout) makes no use of either quantity -- it exploits neither the
d_T<r^2 case for compression nor the dim K(R,B)<r case, which is
exactly why it remains correct regardless of what either quantity does
during online training.

Run: python -m credit_memory.b28_stage2_streaming
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b25_nonlinear_credit import (
    make_arch, forward_step, rollout, compute_F_ab, direct_term, make_Q_all,
    family_dim, dLdh_from_target, factorized_rtrl_run, bptt_reference_grads,
)
from credit_memory.b27_noncommutative_advantage import (
    make_nonlinear_rtu_arch, nonlinear_rtu_pre_and_next, rtu_block_matrix,
)

jax.config.update("jax_enable_x64", True)

FAMILIES_OURS = ("R", "B", "C", "psi")
FAMILIES_RTU = ("theta", "log_radius", "Wx")


# ---------------------------------------------------------------------------
# OURS: streaming (per-step) RTRL state, naive/full-r representation.
# ---------------------------------------------------------------------------
def ours_streaming_init(arch, h0):
    r, n = arch["r"], arch["n"]
    X = {family: np.zeros((n, r, family_dim(family, arch))) for family in FAMILIES_OURS}
    return dict(h=h0, X=X)


def ours_streaming_step(arch, stream_state, u_t):
    """Advances h and ALL families' persistent sensitivity trace by
    one step, using the CURRENT (possibly just-updated) arch
    parameters for THIS step's own contribution -- the old trace
    itself reflects whatever parameters were current at PAST steps
    (genuine online-RTRL staleness, not an approximation error).
    Returns (h_next, z_flat, S_raw) where S_raw[family] is the FULL
    (n,r,m) exact sensitivity dh_next/dtheta_family."""
    h = stream_state["h"]
    r, n, k = arch["r"], arch["n"], arch["k"]
    R = np.asarray(arch["R"])
    Q = make_Q_all(arch["B"], arch["C"], k)  # (k,k,r,r)
    F = compute_F_ab(h, u_t, arch)  # (k,k,n,n), shared across families

    S_raw = {}
    for family in FAMILIES_OURS:
        X = stream_state["X"][family]
        Direct = direct_term(family, h, u_t, arch)  # (n,r,m)
        term_R = np.einsum("ij,pjc->pic", R, X)
        term_Q = np.einsum("abpq,abij,qjc->pic", F, Q, X)
        X_new = term_R + term_Q + Direct
        stream_state["X"][family] = X_new
        S_raw[family] = X_new  # naive rep: X IS the full (n,r,m) sensitivity

    h_next, z_flat, _ = forward_step(h, u_t, arch["R"], arch["B"], arch["C"], arch["psi"])
    stream_state["h"] = h_next
    return h_next, z_flat, S_raw


def ours_per_step_grad(S_raw, dL_dh_next):
    """g_t[family] = <dL/dh_next, S_raw[family]> -- the exact per-step
    gradient for a LOCAL, single-timestep loss (e.g. a value or
    log-prob at THIS step), NOT yet an RL eligibility trace."""
    return {family: np.einsum("pi,pic->c", np.asarray(dL_dh_next), S_raw[family])
            for family in FAMILIES_OURS}


# ---------------------------------------------------------------------------
# RTU: streaming (per-step) RTRL state, block-local (already O(1)/block,
# no naive/reduced distinction needed -- RTU's own exact form IS block-
# decoupled by construction).
# ---------------------------------------------------------------------------
def rtu_streaming_init(arch, h0):
    n_blocks = arch["n_blocks"]
    u_dim = arch["u_dim"]
    E = {
        "theta": np.zeros((n_blocks, 2, 1)),
        "log_radius": np.zeros((n_blocks, 2, 1)),
        "Wx": np.zeros((n_blocks, 2, 2 * u_dim)),
    }
    return dict(h=h0, E=E)


def rtu_streaming_step(arch, stream_state, u_t):
    h = stream_state["h"]
    n_blocks = arch["n_blocks"]
    h_next, pre = nonlinear_rtu_pre_and_next(h, u_t, arch)

    S_raw = {family: np.zeros((arch["r_rtu"], 1 if family != "Wx" else 2 * arch["u_dim"]))
             for family in FAMILIES_RTU}
    # S_raw[family] here is (r_rtu, m_block) with a BLOCK-DIAGONAL structure --
    # only rows [2i,2i+1] are nonzero for a parameter belonging to block i;
    # represented compactly per-block, assembled into per-block contributions.
    S_by_block = {family: [] for family in FAMILIES_RTU}

    for i in range(n_blocks):
        theta_i, logr_i = arch["thetas"][i], arch["log_radii"][i]
        Wx1_i, Wx2_i = arch["Wx"][2 * i], arch["Wx"][2 * i + 1]
        A_block = rtu_block_matrix(theta_i, logr_i)
        pre_i = np.asarray(pre[2 * i:2 * i + 2])
        fprime = 1.0 - np.tanh(pre_i) ** 2
        h_i = np.asarray(h[2 * i:2 * i + 2])

        for family in FAMILIES_RTU:
            E_old = stream_state["E"][family][i]  # (2, m_family)
            if family == "theta":
                def f_theta(th):
                    Ablk = rtu_block_matrix(th[0], logr_i)
                    return Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
                th0 = jnp.array([theta_i])
            elif family == "log_radius":
                def f_theta(th):
                    Ablk = rtu_block_matrix(theta_i, th[0])
                    return Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
                th0 = jnp.array([logr_i])
            else:
                def f_theta(th):
                    Wx1n, Wx2n = th[:arch["u_dim"]], th[arch["u_dim"]:]
                    return A_block @ h_i + jnp.array([Wx1n @ u_t, Wx2n @ u_t])
                th0 = jnp.concatenate([Wx1_i, Wx2_i])
            Q_theta = np.asarray(jax.jacobian(f_theta)(th0))
            E_pre = np.asarray(A_block) @ E_old + Q_theta
            E_new = fprime[:, None] * E_pre
            stream_state["E"][family][i] = E_new
            S_by_block[family].append(E_new)

    stream_state["h"] = h_next
    return h_next, S_by_block  # S_by_block[family]: list of n_blocks (2,m) arrays


def rtu_per_step_grad(S_by_block, dL_dh_next, n_blocks):
    """g_t[family] = concatenation over blocks of <dL/dh_next[block], E_block>."""
    grads = {}
    for family in FAMILIES_RTU:
        gs = []
        for i in range(n_blocks):
            dLdh_i = np.asarray(dL_dh_next)[2 * i:2 * i + 2]
            gs.append(dLdh_i @ S_by_block[family][i])
        grads[family] = np.concatenate(gs)
    return grads


# ---------------------------------------------------------------------------
# Verification: the streaming (per-step, stateful) API must reproduce the
# whole-trajectory functions EXACTLY under FIXED parameters (no online
# update between steps -- that isolates "is the streaming refactor
# correct" from "what does staleness look like once parameters move",
# which is a separate, later question).
# ---------------------------------------------------------------------------
def verify_ours_streaming(seed=1):
    r, k, n, u_dim, hidden = 3, 2, 3, 2, 8
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.array(rng.randn(n, r) * 0.2)
    T_ = 9
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    dLdh_seq = jnp.array(rng.randn(T_, n, r) * 0.3)

    errs = {}
    for family in FAMILIES_OURS:
        grad_batch, _ = factorized_rtrl_run(family, arch, h0, U_seq, dLdh_seq, use_naive=True)

        stream_state = ours_streaming_init(arch, h0)
        grad_stream = np.zeros(family_dim(family, arch))
        for t in range(T_):
            _, _, S_raw = ours_streaming_step(arch, stream_state, U_seq[t])
            grad_stream += np.einsum("pi,pic->c", np.asarray(dLdh_seq[t]), S_raw[family])

        errs[family] = float(np.max(np.abs(grad_stream - grad_batch)))
    return errs


def verify_rtu_streaming(seed=1):
    from credit_memory.b28_popgym_stage1 import rtu_exact_credit_grad_general

    r_rtu, u_dim, hidden = 6, 2, 8
    arch = make_nonlinear_rtu_arch(r_rtu=r_rtu, u_dim=u_dim, hidden=hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.zeros(r_rtu)
    T_ = 9
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    dLdh_seq = jnp.array(rng.randn(T_, r_rtu) * 0.3)
    n_blocks = arch["n_blocks"]

    errs = {}
    for family in FAMILIES_RTU:
        grad_batch = np.zeros(1 if family != "Wx" else 2 * u_dim)
        grad_batch = np.concatenate([
            rtu_exact_credit_grad_general(arch, U_seq, dLdh_seq, i, family)
            for i in range(n_blocks)
        ])

        stream_state = rtu_streaming_init(arch, h0)
        grad_stream = {i: np.zeros(1 if family != "Wx" else 2 * u_dim) for i in range(n_blocks)}
        h = h0
        for t in range(T_):
            h, S_by_block = rtu_streaming_step(arch, stream_state, U_seq[t])
            for i in range(n_blocks):
                dLdh_i = np.asarray(dLdh_seq[t])[2 * i:2 * i + 2]
                grad_stream[i] += dLdh_i @ S_by_block[family][i]
        grad_stream_full = np.concatenate([grad_stream[i] for i in range(n_blocks)])

        errs[family] = float(np.max(np.abs(grad_stream_full - grad_batch)))
    return errs


def main():
    print("=" * 70)
    print("STAGE 2, item 2: streaming (per-step, stateful) RTRL API")
    print("verified against the whole-trajectory reference, FIXED params")
    print("=" * 70)
    errs_ours = verify_ours_streaming()
    for family, e in errs_ours.items():
        print(f"  OURS   family={family:6s} |err|={e:.2e}")
    errs_rtu = verify_rtu_streaming()
    for family, e in errs_rtu.items():
        print(f"  RTU    family={family:12s} |err|={e:.2e}")
    assert all(e < 1e-9 for e in errs_ours.values()), "REGRESSION: ours streaming API"
    assert all(e < 1e-9 for e in errs_rtu.values()), "REGRESSION: RTU streaming API"
    print("  PASS -- streaming API is exact under fixed parameters.")
    print("  (Staleness under CONTINUOUS parameter updates is a separate,")
    print("   expected phenomenon -- measured once online training exists,")
    print("   not tested here.)")


if __name__ == "__main__":
    main()
