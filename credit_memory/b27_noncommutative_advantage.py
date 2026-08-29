"""Phase B27 -- noncommutative temporal advantage falsification.

Tests whether B25/B25.1's surviving theorem (arbitrary nonlinear
feature computation behind a bounded B/C interface preserves an exact,
n-independent temporal module) buys any REPRESENTATIONAL advantage
over diagonal/RTU-style exact-RTRL recurrence -- not another "linear
in n" credit-cost check (already settled and retracted from novelty
claims in the B26 audit).

Reuses B25's architecture/algebra/factorized-RTRL machinery directly
(make_arch, algebra_closure, compute_F_ab, factorized_rtrl_run,
bptt_reference_grads, basis_for_family) -- no re-derivation of already-
verified exact-credit code. New code here: noncommutativity
diagnostics (commutator norms), teacher-usage verification (F_ab,t
trajectory + ablation), the RTU-style diagonal-linear-recurrence +
stateless-nonlinear-readout baseline WITH ITS OWN exact RTRL, training
loops, and the two controls.

Run: python -m credit_memory.b27_noncommutative_advantage
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b25_nonlinear_credit import (
    make_stable_dense, make_psi, psi_flat, psi_from_flat, Phi, make_arch,
    forward_step, rollout, make_E, make_Q_all, compute_F_ab,
    algebra_closure, krylov_subspace, minimal_poly_degree,
    part2_temporal_algebra, direct_term, family_dim, basis_for_family,
    factorized_rtrl_run, bptt_reference_grads, dLdh_from_target,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Part 1: build a genuinely noncommutative teacher, report diagnostics.
# ---------------------------------------------------------------------------
def commutator_norm(A, B_):
    return float(np.linalg.norm(A @ B_ - B_ @ A))


def commutant_dimension(generators, tol=1e-9):
    """dim{X : XA=AX for all A in generators}, via the linear system
    vec(XA-AX)=(A^T(x)I - I(x)A)vec(X)=0 stacked over all generators.
    CORRECTED (per review): this is reported as an INDEPENDENT
    consistency diagnostic only, not as a proof of irreducibility --
    "commutant dimension 1 implies irreducible" is the converse of
    Schur's lemma and is not generally valid. The actual irreducibility
    argument used elsewhere (noncommutativity_report /
    is_full_matrix_algebra) is direct: A_T subseteq M_r trivially, and
    dim(A_T)=r^2=dim(M_r) forces A_T=M_r exactly; M_r acting on R^r has
    no nontrivial invariant subspace (invariance under every
    elementary matrix E_ij forces {0} or the whole space), so the
    action is irreducible. Commutant=1 is then merely CONSISTENT with
    that (a known-irreducible algebra has trivial commutant by Schur's
    lemma, the direction that IS valid), not the source of the claim."""
    r = generators[0].shape[0]
    rows = [np.kron(A.T, np.eye(r)) - np.kron(np.eye(r), A) for A in generators]
    M = np.vstack(rows)
    rank = np.linalg.matrix_rank(M, tol=tol)
    return r * r - rank


def noncommutativity_report(arch):
    r, k = arch["r"], arch["k"]
    R = np.asarray(arch["R"])
    Q = make_Q_all(arch["B"], arch["C"], k)  # (k,k,r,r)
    res = part2_temporal_algebra(arch)

    rq_norms = []
    qq_norms = []
    for a in range(k):
        for b in range(k):
            rq_norms.append(commutator_norm(R, Q[a, b]))
            for c in range(k):
                for d in range(k):
                    if (a, b) < (c, d):
                        qq_norms.append(commutator_norm(Q[a, b], Q[c, d]))
    generators = [R] + [Q[a, b] for a in range(k) for b in range(k)]
    commutant_dim = commutant_dimension(generators)
    is_full = (res["d_T"] == r * r)
    # CORRECTED: irreducibility follows from the DIRECT argument
    # (A_T=M_r, and M_r has no nontrivial invariant subspace on R^r),
    # not from commutant_dim==1 (that converse of Schur's lemma is not
    # generally valid). commutant_dim is reported purely as an
    # independent consistency check, not used to derive this flag.
    is_irreducible = is_full
    return dict(d_T=res["d_T"], rho=res["rho"], omega=res["omega"],
                deg_mu_R=res["deg_mu_R"], bound=res["bound"],
                max_RQ_commutator=max(rq_norms) if rq_norms else 0.0,
                max_QQ_commutator=max(qq_norms) if qq_norms else 0.0,
                r_squared=r * r, commutant_dim=commutant_dim,
                is_full_matrix_algebra=is_full,
                is_irreducible=is_irreducible)


def make_noncommutative_teacher(r, k, n, u_dim, hidden, seed):
    """Genuinely feature-mixing Phi (a real MLP mixing all n*k entries
    of z, as in B25's own make_psi/Phi) -- the mixing is what lets the
    teacher ACTIVATE multiple distinct Q_ab combinations depending on
    state/input, not merely possess a noncommutative algebra it never
    uses."""
    return make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)


# ---------------------------------------------------------------------------
# Part 2: verify the teacher actually USES multiple noncommuting
# generators along real trajectories (not just that the algebra could,
# in principle, be large).
# ---------------------------------------------------------------------------
def teacher_usage_report(arch, h0, U_seq):
    """Records F_ab,t along a real rollout; checks (a) multiple Q_ab
    get non-negligible coefficients, (b) the ACTIVE combination varies
    over time, (c) removing the most important generator materially
    changes the output (ablation)."""
    k = arch["k"]
    n = arch["n"]
    H, Z, Y = rollout(h0, U_seq, arch)
    T_ = U_seq.shape[0]

    F_traj = []  # list of (k,k,n,n) per step
    for t in range(T_):
        F = compute_F_ab(H[t], U_seq[t], arch)
        F_traj.append(F)
    F_traj = np.stack(F_traj)  # (T,k,k,n,n)

    # per-(a,b) "activity" = RMS Frobenius norm of F_ab,t across time
    activity = np.sqrt(np.mean(np.sum(F_traj ** 2, axis=(-1, -2)), axis=0))  # (k,k)
    activity_flat = activity.reshape(-1)
    active_frac = float(np.sum(activity_flat > 0.05 * activity_flat.max()) / activity_flat.size)

    # temporal variation: coefficient of variation of ||F_ab,t||_F over time,
    # for the single most active (a,b)
    ab_star = np.unravel_index(np.argmax(activity), activity.shape)
    norms_t = np.linalg.norm(F_traj[:, ab_star[0], ab_star[1]], axis=(-1, -2))
    cv = float(np.std(norms_t) / (np.mean(norms_t) + 1e-12))

    # ablation: zero out the most active generator's numerator row
    # (b[.., ab_star[1] as the input-feature index is ambiguous for
    # general k -- ablate by zeroing psi's output channel(s) that feed
    # the identified (a,b) contribution most strongly is architecture-
    # specific; instead ablate at the SOURCE -- zero the corresponding
    # rows of B for output-feature a, which removes that Phi-output
    # channel's entire contribution to h_{t+1} for ALL b jointly, a
    # coarser but well-defined and interpretable ablation)
    B_ablated = np.asarray(arch["B"]).copy()
    a_idx = ab_star[0]
    B_ablated[:, :] = np.asarray(arch["B"])  # start from original
    # zero the a-th feature-channel's contribution across all n copies:
    # channel a corresponds to psi output indices [p*k + a for p in range(n)]
    psi_out_mask = np.ones(n * k)
    for p in range(n):
        psi_out_mask[p * k + a_idx] = 0.0

    def ablated_forward_step(h, u, arch_):
        R, B_, C, psi = arch_["R"], arch_["B"], arch_["C"], arch_["psi"]
        z = h @ C.T
        z_flat = z.reshape(-1)
        phi_out = Phi(z_flat, u, psi) * jnp.array(psi_out_mask)
        phi_r = phi_out.reshape(n, k)
        h_next = h @ R.T + phi_r @ B_.T
        return h_next

    h_abl = h0
    h_orig = h0
    max_diff = 0.0
    for t in range(T_):
        h_orig, _, _ = forward_step(h_orig, U_seq[t], arch["R"], arch["B"], arch["C"], arch["psi"])
        h_abl = ablated_forward_step(h_abl, U_seq[t], arch)
        max_diff = max(max_diff, float(jnp.max(jnp.abs(h_orig - h_abl))))

    return dict(activity=activity, active_frac=active_frac, cv_most_active=cv,
                ablation_max_diff=max_diff, ablated_channel=int(a_idx))


# ---------------------------------------------------------------------------
# Part 3: our exact online credit -- pure reuse of B25's already-verified
# factorized_rtrl_run / bptt_reference_grads. No new algorithm here; this
# section only re-confirms exactness ON THIS SPECIFIC noncommutative
# teacher before using it as ground truth for Part 4's comparison.
# ---------------------------------------------------------------------------
def verify_our_exact_credit(arch, h0, U_seq, target_fn):
    dLdh = dLdh_from_target(arch, h0, U_seq, target_fn)
    bptt = bptt_reference_grads(arch, h0, U_seq, target_fn)
    report = {}
    for family in ("R", "B", "C", "psi"):
        g_naive, d_naive = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=True)
        g_fact, d_fact = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=False)
        e_nb = float(np.max(np.abs(g_naive - bptt[family])))
        e_fb = float(np.max(np.abs(g_fact - bptt[family])))
        report[family] = dict(d_source_module=d_fact, err_naive_bptt=e_nb, err_fact_bptt=e_fb)
    return report


# ---------------------------------------------------------------------------
# Part 4: fair diagonal/RTU-style baseline. Architecturally correct
# version of "RTU-style exact RTRL" (matching Zucchet's LRU / RTU
# design principle): the RECURRENCE is purely LINEAR and BLOCK-DIAGONAL
# (r_diag/2 independent 2x2 rotation-scaling blocks, i.e. independent
# complex-conjugate eigenvalue pairs -- genuinely diagonalizable,
# genuinely decoupled). Nonlinearity is applied OUTSIDE the recurrence
# as a STATELESS readout Psi(C_diag@h_diag_t, u_t) -- NOT fed back into
# the state. This is what makes RTU/LRU-style RTRL exact and cheap:
# d(h_diag_{t+1})/d(R_diag) decouples per-block since R_diag is
# block-diagonal and nothing nonlinear re-enters the recurrence.
# Matched nonlinear capacity: Psi uses the SAME hidden width as the
# teacher's own Phi, so any imitation gap reflects recurrent temporal
# structure, not an obviously weaker readout.
# ---------------------------------------------------------------------------
def make_diag_block(theta, log_radius):
    """One 2x2 block: [[rho*cos(theta), -rho*sin(theta)], [rho*sin(theta), rho*cos(theta)]]
    -- a complex eigenvalue rho*exp(i*theta), radius clipped <1 for stability."""
    rho = jnp.exp(-jnp.abs(log_radius))  # in (0,1]
    c, s = jnp.cos(theta), jnp.sin(theta)
    return rho * jnp.array([[c, -s], [s, c]])


def make_full_state_head(in_dim, hidden, rng):
    """Stateless MLP head seeing the FULL recurrent state (not a
    k-dim bottleneck): R^in_dim -> R (scalar), matching the imitation
    task's shared scalar target directly. NOT fed back into the
    recurrence -- the diagonal exact-RTRL structural assumption
    (linear, decoupled recurrence) stays intact."""
    scale1 = 1.0 / np.sqrt(in_dim)
    scale2 = 1.0 / np.sqrt(hidden)
    return dict(W1=jnp.array(rng.randn(hidden, in_dim) * scale1),
                b1=jnp.array(rng.randn(hidden) * 0.1),
                W2=jnp.array(rng.randn(1, hidden) * scale2),
                b2=jnp.array(rng.randn(1) * 0.1))


def full_state_head_apply(head, x):
    h = jnp.tanh(head["W1"] @ x + head["b1"])
    return (head["W2"] @ h + head["b2"])[0]


def head_param_count(in_dim, hidden):
    return hidden * in_dim + hidden + hidden + 1


def solve_hidden_for_target_params(in_dim, target_params):
    """Smallest hidden width whose head_param_count >= target_params
    (used for the 'matched total parameter count' fairness regime)."""
    hidden = 1
    while head_param_count(in_dim, hidden) < target_params:
        hidden += 1
    return hidden


# ---------------------------------------------------------------------------
# CORRECTED BASELINE (per user review): the RTU paper defines NONLINEAR
# RTUs with the activation INSIDE each independent 2x2 block:
#   h1_t = f(g h1_{t-1} - phi h2_{t-1} + Wx1.x_t)
#   h2_t = f(g h2_{t-1} + phi h1_{t-1} + Wx2.x_t)
# f=tanh applied ELEMENT-WISE per unit (not mixing h1,h2 further) --
# blocks remain fully independent of each other (direct sum of small
# blocks), the structural property under test. Same strong full-state
# stateless MLP head as before.
# ---------------------------------------------------------------------------
def make_nonlinear_rtu_arch(r_rtu, u_dim, hidden, seed):
    assert r_rtu % 2 == 0
    n_blocks = r_rtu // 2
    rng = np.random.RandomState(seed)
    thetas = jnp.array(rng.uniform(0.05, np.pi - 0.05, n_blocks))
    log_radii = jnp.array(rng.uniform(0.1, 0.6, n_blocks))
    Wx = jnp.array(rng.randn(r_rtu, u_dim) / np.sqrt(u_dim) * 0.6)
    head = make_full_state_head(r_rtu + u_dim, hidden, rng)
    return dict(thetas=thetas, log_radii=log_radii, Wx=Wx, head=head,
                r_rtu=r_rtu, u_dim=u_dim, hidden=hidden, n_blocks=n_blocks)


def nonlinear_rtu_pre_and_next(h, u, arch):
    """Returns (h_next, pre) -- pre is the PRE-activation (needed for
    the exact block-local RTRL's f'(pre) factor)."""
    rho = jnp.exp(-jnp.abs(arch["log_radii"]))
    g = rho * jnp.cos(arch["thetas"])
    phi = rho * jnp.sin(arch["thetas"])
    h1, h2 = h[0::2], h[1::2]
    inp = arch["Wx"] @ u
    inp1, inp2 = inp[0::2], inp[1::2]
    pre1 = g * h1 - phi * h2 + inp1
    pre2 = g * h2 + phi * h1 + inp2
    pre = jnp.stack([pre1, pre2], axis=1).reshape(-1)
    h_next = jnp.tanh(pre)
    return h_next, pre


def nonlinear_rtu_rollout_scalar(h0, U_seq, arch):
    def step(h, u):
        h_next, _ = nonlinear_rtu_pre_and_next(h, u, arch)
        y = full_state_head_apply(arch["head"], jnp.concatenate([h_next, u]))
        return h_next, y

    _, Ys = jax.lax.scan(step, h0, U_seq)
    return Ys


def rtu_flat_params(arch):
    head = arch["head"]
    return jnp.concatenate([arch["thetas"], arch["log_radii"], arch["Wx"].reshape(-1),
                             head["W1"].reshape(-1), head["b1"], head["W2"].reshape(-1),
                             head["b2"]])


def rtu_from_flat(flat, r_rtu, u_dim, hidden):
    n_blocks = r_rtu // 2
    in_dim = r_rtu + u_dim
    i = 0
    thetas = flat[i:i + n_blocks]; i += n_blocks
    log_radii = flat[i:i + n_blocks]; i += n_blocks
    Wx = flat[i:i + r_rtu * u_dim].reshape(r_rtu, u_dim); i += r_rtu * u_dim
    W1 = flat[i:i + hidden * in_dim].reshape(hidden, in_dim); i += hidden * in_dim
    b1 = flat[i:i + hidden]; i += hidden
    W2 = flat[i:i + hidden].reshape(1, hidden); i += hidden
    b2 = flat[i:i + 1]
    head = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    return dict(thetas=thetas, log_radii=log_radii, Wx=Wx, head=head,
                r_rtu=r_rtu, u_dim=u_dim, hidden=hidden, n_blocks=n_blocks)


def train_rtu_bptt_adam(r_rtu, u_dim, hidden, U_train, y_train, steps, lr, seed):
    arch = make_nonlinear_rtu_arch(r_rtu=r_rtu, u_dim=u_dim, hidden=hidden, seed=seed)
    flat0 = rtu_flat_params(arch)
    n_seq, T_ = U_train.shape[0], U_train.shape[1]

    def loss_of(flat):
        p = rtu_from_flat(flat, r_rtu, u_dim, hidden)
        Ys = jax.vmap(lambda u_seq: nonlinear_rtu_rollout_scalar(jnp.zeros(r_rtu), u_seq, p))(U_train)
        return 0.5 * jnp.sum((Ys - y_train) ** 2) / (n_seq * T_)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(flat0)

    @jax.jit
    def one_step(flat, opt_state):
        loss, g = jax.value_and_grad(loss_of)(flat)
        updates, opt_state = optimizer.update(g, opt_state)
        flat = optax.apply_updates(flat, updates)
        return flat, opt_state, loss

    flat = flat0
    losses = []
    for _ in range(steps):
        flat, opt_state, loss = one_step(flat, opt_state)
        losses.append(float(loss))
    arch = rtu_from_flat(flat, r_rtu, u_dim, hidden)
    return losses, arch


def eval_rtu_mse(arch, U_batch, y_batch):
    r_rtu = arch["r_rtu"]
    Ys = jax.vmap(lambda u_seq: nonlinear_rtu_rollout_scalar(jnp.zeros(r_rtu), u_seq, arch))(U_batch)
    return float(jnp.mean((Ys - y_batch) ** 2))


def rtu_param_count(r_rtu, u_dim, hidden):
    n_blocks = r_rtu // 2
    return 2 * n_blocks + r_rtu * u_dim + head_param_count(r_rtu + u_dim, hidden)


# ---------------------------------------------------------------------------
# Nonlinear RTU's OWN exact block-local RTRL (per user review, item 5:
# derive and VERIFY the actual trace structure, don't assume the old
# linear baseline's accounting carries over). Each block's own params
# (theta_i,log_radius_i,Wx1_i,Wx2_i) produce a sensitivity trace
# confined to that block's own 2-dim state (diag(f'(pre)) does not mix
# blocks; the linear part is block-diagonal by construction) -- O(1)
# per own-parameter, independent of r_rtu/other blocks. Verified
# against BPTT below before its cost is trusted.
# ---------------------------------------------------------------------------
def rtu_block_matrix(theta, log_radius):
    rho = jnp.exp(-jnp.abs(log_radius))
    c, s = jnp.cos(theta), jnp.sin(theta)
    return rho * jnp.array([[c, -s], [s, c]])


def rtu_exact_credit_grad(arch, U_seq, y_target, block_idx, family):
    """Exact block-local RTRL for ONE block's OWN parameters. family in
    {'theta','log_radius','Wx'}. Returns the gradient contribution from
    this block alone (dL/dtheta_block), tracking only a 2 x m_block
    persistent trace -- verified against BPTT elsewhere."""
    n_blocks = arch["n_blocks"]
    i = block_idx
    theta_i, logr_i = arch["thetas"][i], arch["log_radii"][i]
    Wx1_i, Wx2_i = arch["Wx"][2 * i], arch["Wx"][2 * i + 1]
    A_block = rtu_block_matrix(theta_i, logr_i)

    if family == "theta":
        m = 1
    elif family == "log_radius":
        m = 1
    else:
        m = 2 * arch["u_dim"]  # Wx1_i and Wx2_i together

    h = jnp.zeros(arch["r_rtu"])
    E = np.zeros((2, m))
    grad = np.zeros(m)
    T_ = U_seq.shape[0]

    for t in range(T_):
        u_t = U_seq[t]
        h_next, pre = nonlinear_rtu_pre_and_next(h, u_t, arch)
        pre_i = np.asarray(pre[2 * i:2 * i + 2])
        fprime = 1.0 - np.tanh(pre_i) ** 2  # (2,)
        h_i = np.asarray(h[2 * i:2 * i + 2])

        def f_theta(th):
            if family == "theta":
                Ablk = rtu_block_matrix(th[0], logr_i)
                pre_local = Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
            elif family == "log_radius":
                Ablk = rtu_block_matrix(theta_i, th[0])
                pre_local = Ablk @ h_i + jnp.array([Wx1_i, Wx2_i]) @ u_t
            else:
                Wx1n, Wx2n = th[:arch["u_dim"]], th[arch["u_dim"]:]
                pre_local = A_block @ h_i + jnp.array([Wx1n @ u_t, Wx2n @ u_t])
            return pre_local  # (2,) direct term BEFORE tanh

        if family == "theta":
            th0 = jnp.array([theta_i])
        elif family == "log_radius":
            th0 = jnp.array([logr_i])
        else:
            th0 = jnp.concatenate([Wx1_i, Wx2_i])
        Q_theta = np.asarray(jax.jacobian(f_theta)(th0))  # (2, m)

        E_pre = np.asarray(A_block) @ E + Q_theta  # propagate through linear part, add direct
        E = fprime[:, None] * E_pre  # apply tanh' elementwise

        def head_out(hh):
            return full_state_head_apply(arch["head"], jnp.concatenate([hh, u_t]))

        dydh = np.asarray(jax.grad(head_out)(h_next))  # (r_rtu,)
        dydh_i = dydh[2 * i:2 * i + 2]  # (2,)
        y_val = float(head_out(h_next))
        err = (y_val - float(y_target[t])) / T_
        grad += err * (dydh_i @ E)
        h = h_next
    return grad


def rtu_credit_floats_verified(r_rtu, u_dim):
    """The VERIFIED (not assumed) per-block trace dimension: 2 (block
    state dim) per own scalar parameter. Total own params per block:
    theta,log_radius (1 each) + Wx1,Wx2 (u_dim each) = 2+2*u_dim.
    Credit per block = 2*(2+2*u_dim). Total = n_blocks*that
    = r_rtu*(2+2*u_dim) = 2*r_rtu*(1+u_dim) -- matches the corrected
    linear-block accounting exactly, confirming the earlier fix's
    asymptotic law was right; this derivation is now checked against
    an actual implemented-and-verified recurrence, not assumed."""
    return 2 * r_rtu * (1 + u_dim)


def make_diag_arch(r_diag, u_dim, hidden, seed):
    """r_diag must be even (pairs of 2x2 blocks). Full-state MLP head
    (per user correction 3) -- no k-dim readout bottleneck."""
    assert r_diag % 2 == 0
    n_blocks = r_diag // 2
    rng = np.random.RandomState(seed)
    thetas = jnp.array(rng.uniform(0.05, np.pi - 0.05, n_blocks))
    log_radii = jnp.array(rng.uniform(0.1, 0.6, n_blocks))
    B_diag = jnp.array(rng.randn(r_diag, u_dim) / np.sqrt(u_dim) * 0.6)
    head = make_full_state_head(r_diag + u_dim, hidden, rng)
    return dict(thetas=thetas, log_radii=log_radii, B=B_diag, head=head,
                r_diag=r_diag, u_dim=u_dim, hidden=hidden)


def diag_R_matrix(thetas, log_radii):
    n_blocks = thetas.shape[0]
    blocks = [make_diag_block(thetas[i], log_radii[i]) for i in range(n_blocks)]
    r_diag = 2 * n_blocks
    R = jnp.zeros((r_diag, r_diag))
    for i, blk in enumerate(blocks):
        R = R.at[2 * i:2 * i + 2, 2 * i:2 * i + 2].set(blk)
    return R


# ---------------------------------------------------------------------------
# Separate, structurally-matched REAL-diagonal baseline for Control A
# specifically (the commuting teacher's true generator is real-diagonal,
# no rotation) -- avoids a representational mismatch where the standard
# complex/2x2-block RTU parameterization can only reach a real
# eigenvalue at the hard-to-optimize theta=0/pi boundary of its own
# angle parameterization. NOT used for the main noncommutative-teacher
# test (which correctly uses the general complex/2x2-block baseline).
# ---------------------------------------------------------------------------
def make_real_diag_arch(r_diag, u_dim, hidden, seed):
    rng = np.random.RandomState(seed)
    raw_eigs = jnp.array(rng.uniform(-0.85, 0.85, r_diag))  # unconstrained-ish real eigenvalues
    B_diag = jnp.array(rng.randn(r_diag, u_dim) / np.sqrt(u_dim) * 0.6)
    head = make_full_state_head(r_diag + u_dim, hidden, rng)
    return dict(eigs=raw_eigs, B=B_diag, head=head, r_diag=r_diag, u_dim=u_dim, hidden=hidden)


def real_diag_rollout_scalar(h0, U_seq, arch):
    eigs = jnp.clip(arch["eigs"], -0.95, 0.95)  # keep stable during optimization

    def step(h, u):
        h_next = eigs * h + arch["B"] @ u
        y = full_state_head_apply(arch["head"], jnp.concatenate([h_next, u]))
        return h_next, y

    _, Ys = jax.lax.scan(step, h0, U_seq)
    return Ys


def real_diag_flat_params(arch):
    head = arch["head"]
    return jnp.concatenate([arch["eigs"], arch["B"].reshape(-1), head["W1"].reshape(-1),
                             head["b1"], head["W2"].reshape(-1), head["b2"]])


def real_diag_from_flat(flat, r_diag, u_dim, hidden):
    in_dim = r_diag + u_dim
    i = 0
    eigs = flat[i:i + r_diag]; i += r_diag
    B_diag = flat[i:i + r_diag * u_dim].reshape(r_diag, u_dim); i += r_diag * u_dim
    W1 = flat[i:i + hidden * in_dim].reshape(hidden, in_dim); i += hidden * in_dim
    b1 = flat[i:i + hidden]; i += hidden
    W2 = flat[i:i + hidden].reshape(1, hidden); i += hidden
    b2 = flat[i:i + 1]
    head = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    return dict(eigs=eigs, B=B_diag, head=head, r_diag=r_diag, u_dim=u_dim, hidden=hidden)


def train_real_diag_bptt_adam(r_diag, u_dim, hidden, U_train, y_train, steps, lr, seed):
    arch = make_real_diag_arch(r_diag=r_diag, u_dim=u_dim, hidden=hidden, seed=seed)
    flat0 = real_diag_flat_params(arch)
    n_seq, T_ = U_train.shape[0], U_train.shape[1]

    def loss_of(flat):
        p = real_diag_from_flat(flat, r_diag, u_dim, hidden)
        Ys = jax.vmap(lambda u_seq: real_diag_rollout_scalar(jnp.zeros(r_diag), u_seq, p))(U_train)
        return 0.5 * jnp.sum((Ys - y_train) ** 2) / (n_seq * T_)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(flat0)

    @jax.jit
    def one_step(flat, opt_state):
        loss, g = jax.value_and_grad(loss_of)(flat)
        updates, opt_state = optimizer.update(g, opt_state)
        flat = optax.apply_updates(flat, updates)
        return flat, opt_state, loss

    flat = flat0
    losses = []
    for _ in range(steps):
        flat, opt_state, loss = one_step(flat, opt_state)
        losses.append(float(loss))
    arch = real_diag_from_flat(flat, r_diag, u_dim, hidden)
    return losses, arch


def eval_real_diag_mse(arch, U_batch, y_batch):
    r_diag = arch["r_diag"]
    Ys = jax.vmap(lambda u_seq: real_diag_rollout_scalar(jnp.zeros(r_diag), u_seq, arch))(U_batch)
    return float(jnp.mean((Ys - y_batch) ** 2))


def real_diag_param_count(r_diag, u_dim, hidden):
    return r_diag + r_diag * u_dim + head_param_count(r_diag + u_dim, hidden)


def diag_rollout_scalar(h0, U_seq, arch):
    """lax.scan (not a Python loop) so this traces once. Full-state MLP
    head (per user correction 3): y_t = MLP([h_{t+1}, u_t]) -- sees the
    ENTIRE recurrent state, no k-dim readout bottleneck. Output is
    already scalar (matches the shared imitation target directly)."""
    R = diag_R_matrix(arch["thetas"], arch["log_radii"])

    def step(h, u):
        h_next = R @ h + arch["B"] @ u
        y = full_state_head_apply(arch["head"], jnp.concatenate([h_next, u]))
        return h_next, y

    _, Ys = jax.lax.scan(step, h0, U_seq)
    return Ys  # (T,)


def diag_flat_params(arch):
    head = arch["head"]
    return jnp.concatenate([arch["thetas"], arch["log_radii"], arch["B"].reshape(-1),
                             head["W1"].reshape(-1), head["b1"], head["W2"].reshape(-1),
                             head["b2"]])


def diag_from_flat(flat, r_diag, u_dim, hidden):
    n_blocks = r_diag // 2
    in_dim = r_diag + u_dim
    i = 0
    thetas = flat[i:i + n_blocks]; i += n_blocks
    log_radii = flat[i:i + n_blocks]; i += n_blocks
    B_diag = flat[i:i + r_diag * u_dim].reshape(r_diag, u_dim); i += r_diag * u_dim
    W1 = flat[i:i + hidden * in_dim].reshape(hidden, in_dim); i += hidden * in_dim
    b1 = flat[i:i + hidden]; i += hidden
    W2 = flat[i:i + hidden].reshape(1, hidden); i += hidden
    b2 = flat[i:i + 1]
    head = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    return dict(thetas=thetas, log_radii=log_radii, B=B_diag, head=head,
                r_diag=r_diag, u_dim=u_dim, hidden=hidden, n_blocks=n_blocks)


def train_diag_student_batch(r_diag, u_dim, hidden, U_batch, y_batch, steps, lr, seed):
    """CORRECTED (per user correction 2): trains on a BATCH of
    independently-sampled sequences (U_batch: (n_seq,T,u_dim), y_batch:
    (n_seq,T)), not one fixed trajectory -- tests whether the diagonal
    architecture can fit the teacher's input-output FUNCTION, not one
    observed sequence. Jitted + vmapped over the batch for speed."""
    arch = make_diag_arch(r_diag=r_diag, u_dim=u_dim, hidden=hidden, seed=seed)
    flat0 = diag_flat_params(arch)
    n_seq, T_ = U_batch.shape[0], U_batch.shape[1]

    def loss_of(flat):
        p = diag_from_flat(flat, r_diag, u_dim, hidden)
        Ys = jax.vmap(lambda u_seq: diag_rollout_scalar(jnp.zeros(r_diag), u_seq, p))(U_batch)
        return 0.5 * jnp.sum((Ys - y_batch) ** 2) / (n_seq * T_)

    @jax.jit
    def one_step(flat):
        loss, g = jax.value_and_grad(loss_of)(flat)
        g = jnp.clip(g, -1.0, 1.0)
        return flat - lr * g, loss

    flat = flat0
    losses = []
    for _ in range(steps):
        flat, loss = one_step(flat)
        losses.append(float(loss))
    arch = diag_from_flat(flat, r_diag, u_dim, hidden)
    return losses, arch


def eval_diag_batch_mse(arch, U_batch, y_batch):
    r_diag = arch["r_diag"]
    Ys = jax.vmap(lambda u_seq: diag_rollout_scalar(jnp.zeros(r_diag), u_seq, arch))(U_batch)
    return float(jnp.mean((Ys - y_batch) ** 2))


def diag_param_count(r_diag, u_dim, hidden):
    n_blocks = r_diag // 2
    return 2 * n_blocks + r_diag * u_dim + head_param_count(r_diag + u_dim, hidden)


# ---------------------------------------------------------------------------
# Shared imitation task: BOTH architectures reproduce a common SCALAR
# target (matching B25's own make_teacher_targets convention:
# y_t = sum over the readout's own feature dim), so the comparison
# is architecture-agnostic and not biased by internal dimensionality.
# ---------------------------------------------------------------------------
def teacher_scalar_target(teacher, h0, U_seq):
    H, Z, _ = rollout(h0, U_seq, teacher)
    return jnp.sum(Z, axis=1)  # (T,)


def generate_teacher_batch(teacher, n_seq, T_, u_dim, n_teacher, r_teacher, seed):
    """CRITICAL FIX (per user review): h0_teacher was previously random
    and UNOBSERVED by the student, making y=F(U_seq,h0_teacher) while
    the student only sees U_seq -- not a deterministic function of the
    student's own input, creating an irreducible test-error floor that
    could fully explain an apparent plateau on its own. Fixed: h0=0 for
    EVERY sequence (a genuine deterministic input-output function of
    U_seq alone); n_seq INDEPENDENTLY sampled persistently-exciting
    input sequences (per the earlier, still-valid correction 2 -- not
    one fixed trajectory)."""
    rng = np.random.RandomState(seed)
    U_batch = jnp.array(rng.randn(n_seq, T_, u_dim) * 0.5)
    h0 = jnp.zeros((n_teacher, r_teacher))
    y_list = [teacher_scalar_target(teacher, h0, U_batch[s]) for s in range(n_seq)]
    return U_batch, jnp.stack(y_list)


import optax


def ours_flat_params(arch):
    return jnp.concatenate([arch["R"].reshape(-1), arch["B"].reshape(-1), arch["C"].reshape(-1),
                             psi_flat(arch["psi"])])


def ours_from_flat(flat, r, k, n, u_dim, hidden):
    i = 0
    R = flat[i:i + r * r].reshape(r, r); i += r * r
    B_ = flat[i:i + r * k].reshape(r, k); i += r * k
    C_ = flat[i:i + k * r].reshape(k, r); i += k * r
    psi = psi_from_flat(flat[i:], n, k, u_dim, hidden)
    return dict(R=R, B=B_, C=C_, psi=psi, r=r, k=k, n=n, u_dim=u_dim, hidden=hidden)


def ours_rollout_scalar(h0, U_seq, arch):
    H, _, _ = rollout(h0, U_seq, arch)
    Z = (H[1:] @ arch["C"].T).reshape(U_seq.shape[0], -1)
    return jnp.sum(Z, axis=1)


def train_ours_bptt_adam(r, k, n, u_dim, hidden, U_train, y_train, steps, lr, seed):
    """CORRECTED (per user review, item 2): trains via ordinary
    BPTT/autodiff + Adam -- legitimate since factorized RTRL == BPTT
    has been verified repeatedly (and is re-verified on the FINAL
    trained model below). Isolates REPRESENTATION from the online
    algorithm's own (much slower, non-JIT-compatible) training loop --
    this subtest asks what each MODEL CLASS can represent, not how
    fast its online learning rule trains. JIT-compiled, same optimizer
    as the diagonal baseline for a fair comparison."""
    arch0 = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
    h0 = jnp.zeros((n, r))
    flat0 = ours_flat_params(arch0)
    n_seq, T_ = U_train.shape[0], U_train.shape[1]

    def loss_of(flat):
        p = ours_from_flat(flat, r, k, n, u_dim, hidden)
        Ys = jax.vmap(lambda u_seq: ours_rollout_scalar(h0, u_seq, p))(U_train)
        return 0.5 * jnp.sum((Ys - y_train) ** 2) / (n_seq * T_)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(flat0)

    @jax.jit
    def one_step(flat, opt_state):
        loss, g = jax.value_and_grad(loss_of)(flat)
        updates, opt_state = optimizer.update(g, opt_state)
        flat = optax.apply_updates(flat, updates)
        return flat, opt_state, loss

    flat = flat0
    losses = []
    for _ in range(steps):
        flat, opt_state, loss = one_step(flat, opt_state)
        losses.append(float(loss))
    arch = ours_from_flat(flat, r, k, n, u_dim, hidden)
    return losses, arch, h0


def eval_ours_bptt_mse(arch, h0, U_batch, y_batch):
    Ys = jax.vmap(lambda u_seq: ours_rollout_scalar(h0, u_seq, arch))(U_batch)
    return float(jnp.mean((Ys - y_batch) ** 2))


def train_diag_bptt_adam(r_diag, u_dim, hidden, U_train, y_train, steps, lr, seed):
    """Same Adam optimizer as ours_bptt_adam for a genuinely common
    training protocol across both architecture classes."""
    arch = make_diag_arch(r_diag=r_diag, u_dim=u_dim, hidden=hidden, seed=seed)
    flat0 = diag_flat_params(arch)
    n_seq, T_ = U_train.shape[0], U_train.shape[1]

    def loss_of(flat):
        p = diag_from_flat(flat, r_diag, u_dim, hidden)
        Ys = jax.vmap(lambda u_seq: diag_rollout_scalar(jnp.zeros(r_diag), u_seq, p))(U_train)
        return 0.5 * jnp.sum((Ys - y_train) ** 2) / (n_seq * T_)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(flat0)

    @jax.jit
    def one_step(flat, opt_state):
        loss, g = jax.value_and_grad(loss_of)(flat)
        updates, opt_state = optimizer.update(g, opt_state)
        flat = optax.apply_updates(flat, updates)
        return flat, opt_state, loss

    flat = flat0
    losses = []
    for _ in range(steps):
        flat, opt_state, loss = one_step(flat, opt_state)
        losses.append(float(loss))
    arch = diag_from_flat(flat, r_diag, u_dim, hidden)
    return losses, arch


def verify_ours_trained_model_exact(arch, h0, U_seq):
    """Post-hoc check (per user review, item 2 closing instruction):
    on the FINAL BPTT-trained model, re-verify factorized RTRL == BPTT
    to machine precision -- confirms the exact online algorithm still
    matches at the trained parameter point, not just at random init."""
    T_ = U_seq.shape[0]
    target = jnp.array(np.random.RandomState(0).randn(T_ + 1, arch["n"], arch["r"]) * 0.1)
    target_fn = lambda H: 0.5 * jnp.sum((H - target) ** 2) / T_
    dLdh = dLdh_from_target(arch, h0, U_seq, target_fn)
    bptt = bptt_reference_grads(arch, h0, U_seq, target_fn)
    errs = {}
    for family in ("R", "B", "C", "psi"):
        g_fact, _ = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=False)
        errs[family] = float(np.max(np.abs(g_fact - bptt[family])))
    return errs


def ours_scalar_output(arch, h0, U_seq):
    H, _, _ = rollout(h0, U_seq, arch)
    Z = (H[1:] @ arch["C"].T).reshape(U_seq.shape[0], -1)
    return jnp.sum(Z, axis=1)


def train_ours_student_batch(r, k, n, u_dim, hidden, U_batch, y_batch, steps, lr, seed):
    """CORRECTED (per user correction 2): trains on a BATCH of
    independently-sampled sequences (U_batch: (n_seq,T,u_dim), y_batch:
    (n_seq,T)) -- a smaller batch than the diagonal side's (B25's
    factorized_rtrl_run is not JIT-compatible, so batching multiplies
    wall-clock directly; kept small and stated explicitly, not silently
    matched to the diagonal side's much larger batch)."""
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.array(rng.randn(n, r) * 0.2)
    n_seq, T_ = U_batch.shape[0], U_batch.shape[1]

    losses = []
    for _ in range(steps):
        grads_acc = {f: np.zeros(family_dim(f, arch)) for f in ("R", "B", "C", "psi")}
        for s in range(n_seq):
            U_seq, y_target = U_batch[s], y_batch[s]

            def target_fn(H, y_target=y_target):
                Z = (H[1:] @ arch["C"].T).reshape(T_, -1)
                y = jnp.sum(Z, axis=1)
                return 0.5 * jnp.sum((y - y_target) ** 2) / T_

            dLdh = dLdh_from_target(arch, h0, U_seq, target_fn)
            for family in ("R", "B", "C", "psi"):
                g, _ = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=False)
                grads_acc[family] += np.asarray(g) / n_seq

        R = jnp.array(arch["R"]) - lr * jnp.clip(grads_acc["R"], -1.0, 1.0).reshape(r, r)
        B_ = jnp.array(arch["B"]) - lr * jnp.clip(grads_acc["B"], -1.0, 1.0).reshape(r, k)
        C_ = jnp.array(arch["C"]) - lr * jnp.clip(grads_acc["C"], -1.0, 1.0).reshape(k, r)
        flat0 = psi_flat(arch["psi"])
        psi_new = psi_from_flat(flat0 - lr * jnp.clip(grads_acc["psi"], -1.0, 1.0), n, k, u_dim, hidden)
        arch = dict(arch, R=R, B=B_, C=C_, psi=psi_new)

        total_loss = 0.0
        for s in range(n_seq):
            y_pred = ours_scalar_output(arch, h0, U_batch[s])
            total_loss += float(jnp.mean((y_pred - y_batch[s]) ** 2)) / n_seq
        losses.append(total_loss)
    return losses, arch, h0


def eval_ours_batch_mse(arch, h0, U_batch, y_batch):
    n_seq = U_batch.shape[0]
    total = 0.0
    for s in range(n_seq):
        y_pred = ours_scalar_output(arch, h0, U_batch[s])
        total += float(jnp.mean((y_pred - y_batch[s]) ** 2))
    return total / n_seq


def ours_param_count(r, k, n, u_dim, hidden):
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=0)
    return int(arch["R"].size + arch["B"].size + arch["C"].size
                + sum(v.size for v in arch["psi"].values()))


def ours_credit_floats(r, k, n, u_dim, hidden):
    """Persistent factorized-RTRL floats: sum over families of
    n * d_source_module * family_dim (the ACTUAL object B25/B25.1
    track, not a hypothetical)."""
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=0)
    total = 0
    for family in ("R", "B", "C", "psi"):
        d = basis_for_family(family, arch).shape[1]
        m = family_dim(family, arch)
        total += n * d * m
    return total


def diag_credit_floats(r_diag, u_dim, hidden):
    """CORRECTED (per user review, item 6): a 2x2 ROTATION block is not
    diagonal -- it mixes its own two coordinates. A parameter
    perturbation (theta, log_radius, or a B-entry in either of the
    block's two rows) therefore produces a genuinely 2-DIMENSIONAL
    persistent sensitivity trace confined to that block (decoupled
    from OTHER blocks, but not from its own pair) -- not one scalar
    float. Per block: 2 own params (theta,log_radius) + 2*u_dim
    B-params (u_dim per each of the block's 2 rows), each needing a
    2-dim trace: credit_per_block=(2+2*u_dim)*2. Total across
    n_blocks=r_diag/2: 2*r_diag*(1+u_dim). The head is STATELESS (no
    recurrence, its gradient is a plain one-step backprop, not
    persistent state at all). Asymptotically still O(r_diag) at fixed
    u_dim -- the constant factor changes, not the scaling law."""
    return 2 * r_diag * (1 + u_dim)


# ---------------------------------------------------------------------------
# Part 5: critical controls.
#
# Control A -- COMMUTING TEACHER: force Phi to depend on z through only
# ONE fixed linear combination (a rank-1 "gate" -- effectively
# collapsing all Q_ab into a single active direction, so the algebra
# closure never activates more than one generator at a time; formally
# this makes the realized F_ab,t rank-1 in (a,b) at every t, so the
# ACTIVE part of the algebra is abelian even though Alg(R,{Q_ab}) as a
# static object could still be large).
#
# Control B -- FIXED DENSE LINEAR TEACHER: drop Phi/nonlinearity
# entirely (pure LTI system, one fixed diagonalizable dense R, no
# state-dependent generator switching at all).
# ---------------------------------------------------------------------------
def make_commuting_teacher(r, k, n, u_dim, hidden, seed):
    """Same R,B,C as a generic teacher, but Phi is forced to route
    through a single fixed direction: Phi(z,u) = w * g(v.z + e.u + c)
    for fixed random w,v,e,c -- so at every t, dPhi/dz is RANK-1
    (outer product w (x) v), meaning F_ab,t = coeff_t * (fixed pattern)
    for ALL (a,b) simultaneously -- only one generator DIRECTION is
    ever active, so the realized algebra never needs more than one
    Q-direction regardless of R's own structure."""
    rng = np.random.RandomState(seed)
    R = make_stable_dense(r, rng)
    B = rng.randn(r, k) / np.sqrt(k) * 0.8
    C = rng.randn(k, r) / np.sqrt(r) * 0.8
    w = rng.randn(n * k) * 0.5
    v = rng.randn(n * k) / np.sqrt(n * k)
    e = rng.randn(u_dim) / np.sqrt(u_dim)
    c0 = rng.randn() * 0.1
    psi = dict(w=jnp.array(w), v=jnp.array(v), e=jnp.array(e), c0=jnp.array(c0),
               _commuting=True)
    return dict(R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), psi=psi,
                r=r, k=k, n=n, u_dim=u_dim, hidden=hidden)


def make_true_commuting_teacher(r, k, n, u_dim, hidden, seed):
    """CORRECTED Control A (per user review): rank(DPhi)=1 does NOT
    imply commuting -- a single active Q* can still fail to commute
    with R. This constructs TRUE commutativity BY DESIGN and verifies
    it numerically, rather than inferring it from rank: R is diagonal
    (real, distinct stable eigenvalues); each interface channel's
    B-column and C-row are aligned to the SAME single state coordinate
    (so every Q_ab=B[:,a]C[b,:] is itself diagonal -- a product of two
    diagonal matrices always commutes, [R,Q_ab]=0 exactly; and every
    pair Q_ab,Q_cd is diagonal-times-diagonal, so [Q_ab,Q_cd]=0 exactly
    too). Nonlinearity Phi still genuinely couples state/input
    (h_{t+1} is still a nonlinear function of h_t), but the algebra it
    activates is abelian by construction, not just low-rank."""
    rng = np.random.RandomState(seed)
    eigs = rng.uniform(-0.85, 0.85, r)
    while np.min(np.abs(np.diff(np.sort(eigs)))) < 0.02:  # keep eigenvalues distinct
        eigs = rng.uniform(-0.85, 0.85, r)
    R = np.diag(eigs)
    B = np.zeros((r, k))
    C = np.zeros((k, r))
    coord = rng.randint(0, r)  # ALL channels aligned to this one coordinate
    for a in range(k):
        B[coord, a] = rng.randn() * 0.7
        C[a, coord] = rng.randn() * 0.7
    psi = make_psi(n, k, u_dim, hidden, rng)
    return dict(R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), psi=psi,
                r=r, k=k, n=n, u_dim=u_dim, hidden=hidden), coord


def commuting_Phi(z_flat, u, psi):
    pre = psi["v"] @ z_flat + psi["e"] @ u + psi["c0"]
    return psi["w"] * jnp.tanh(pre)


def commuting_forward_step(h, u, R, B, C, psi):
    n, r = h.shape
    k = C.shape[0]
    z = h @ C.T
    z_flat = z.reshape(-1)
    phi_out = commuting_Phi(z_flat, u, psi)
    phi_r = phi_out.reshape(n, k)
    h_next = h @ R.T + phi_r @ B.T
    return h_next, z_flat, phi_out


def commuting_rollout(h0, U_seq, arch):
    h = h0
    Hs, Zs = [h], []
    for t in range(U_seq.shape[0]):
        h, z, _ = commuting_forward_step(h, U_seq[t], arch["R"], arch["B"], arch["C"], arch["psi"])
        Hs.append(h)
        Zs.append(z)
    return jnp.stack(Hs), jnp.stack(Zs)


def make_fixed_linear_teacher(r, k, n, u_dim, seed):
    """No Phi at all -- pure LTI: h_{t+1}=(I(x)R)h_t+(I(x)B)u_t_broadcast,
    z_t=(I(x)C)h_t. A single fixed diagonalizable dense R, no
    state-dependent generator switching."""
    rng = np.random.RandomState(seed)
    R = make_stable_dense(r, rng)
    B = rng.randn(r, u_dim) / np.sqrt(u_dim) * 0.7
    C = rng.randn(k, r) / np.sqrt(r) * 0.7
    return dict(R=jnp.array(R), B=jnp.array(B), C=jnp.array(C), r=r, k=k, n=n, u_dim=u_dim)


def fixed_linear_rollout(h0, U_seq, arch):
    R, B, C = arch["R"], arch["B"], arch["C"]
    h = h0
    Zs = []
    for t in range(U_seq.shape[0]):
        h = h @ R.T + U_seq[t] @ B.T  # single-copy (n=1) LTI, broadcast not needed
        Zs.append(h @ C.T)
    return jnp.stack(Zs)


def main():
    print("=" * 70)
    print("PART 1 -- noncommutative teacher construction and diagnostics")
    print("=" * 70)
    for (r, k) in [(3, 1), (3, 2), (4, 2), (5, 2)]:
        teacher = make_noncommutative_teacher(r=r, k=k, n=6, u_dim=2, hidden=16, seed=1)
        diag = noncommutativity_report(teacher)
        print(f"  r={r} k={k}: d_T={diag['d_T']} (r^2={diag['r_squared']}) rho={diag['rho']} "
              f"max_RQ_comm={diag['max_RQ_commutator']:.3f} max_QQ_comm={diag['max_QQ_commutator']:.3f}")

    print()
    print("=" * 70)
    print("PART 2 -- teacher usage verification (main r=4,k=2,n=6 teacher)")
    print("=" * 70)
    teacher = make_noncommutative_teacher(r=4, k=2, n=6, u_dim=2, hidden=16, seed=1)
    rng = np.random.RandomState(2)
    h0 = jnp.array(rng.randn(6, 4) * 0.3)
    U_seq = jnp.array(rng.randn(15, 2) * 0.5)
    usage = teacher_usage_report(teacher, h0, U_seq)
    print(f"  activity matrix (k x k):\n{usage['activity']}")
    print(f"  active_frac={usage['active_frac']:.2f}  cv_most_active={usage['cv_most_active']:.4f}  "
          f"ablation_max_diff={usage['ablation_max_diff']:.4f}")

    print()
    print("=" * 70)
    print("PART 3 -- our exact online credit on this specific teacher")
    print("=" * 70)
    T_ = 8
    U_seq3 = jnp.array(rng.randn(T_, 2) * 0.5)
    target = jnp.array(rng.randn(T_ + 1, 6, 4) * 0.3)
    target_fn = lambda H: 0.5 * jnp.sum((H - target) ** 2) / T_
    report = verify_our_exact_credit(teacher, h0, U_seq3, target_fn)
    for f, r_ in report.items():
        print(f"  {f}: {r_}")

    print()
    print("=" * 70)
    print("PART 6 (depth exactness, cheap reuse of B25.1 -- does NOT resolve the")
    print("representational question below, per user instruction)")
    print("=" * 70)
    from credit_memory.b25_nonlinear_credit import stack_dLdh_flat, stack_naive_rtrl_grad, stack_bptt_grad
    from credit_memory.b25_1_deep_prefix import deep_factorized_grad
    archs2 = [make_arch(r=4, k=2, n=4, u_dim=2, hidden=12, seed=1),
              make_arch(r=4, k=2, n=4, u_dim=4 * 2, hidden=12, seed=2)]
    rng5 = np.random.RandomState(5)
    h0s2 = [jnp.array(rng5.randn(4, 4) * 0.3), jnp.array(rng5.randn(4, 4) * 0.3)]
    T2 = 6
    U2 = jnp.array(rng5.randn(T2, 2) * 0.5)
    target2 = jnp.array(rng5.randn(T2, 4, 4) * 0.3)
    target_fn2 = lambda Hs: 0.5 * jnp.sum(
        (jnp.stack([Hs[t][1] for t in range(1, T2 + 1)]) - target2) ** 2) / T2
    dLdh2 = stack_dLdh_flat(archs2, h0s2, U2, target_fn2)
    for layer_idx, family in [(0, "R"), (0, "C"), (1, "R"), (1, "C")]:
        g_naive = stack_naive_rtrl_grad(archs2, h0s2, U2, (layer_idx, family), dLdh2)
        g_bptt = stack_bptt_grad(archs2, h0s2, U2, (layer_idx, family), target_fn2)
        g_deep, dims = deep_factorized_grad(archs2, h0s2, U2, layer_idx, family, dLdh2)
        e_nb = np.max(np.abs(g_naive - g_bptt))
        e_db = np.max(np.abs(g_deep - g_bptt))
        print(f"  L=2 layer{layer_idx}.{family}: dims={dims}  |naive-bptt|={e_nb:.2e}  |deep-bptt|={e_db:.2e}")

    print()
    print("=" * 70)
    print("A_T = M_r VERIFICATION (per user review) -- rigorous structural")
    print("characterization: DIRECT argument (dim A_T=r^2 forces A_T=M_r, which")
    print("has no nontrivial invariant subspace) + commutant as consistency check")
    print("=" * 70)
    teacher_m = make_noncommutative_teacher(r=4, k=2, n=6, u_dim=2, hidden=16, seed=1)
    diag_m = noncommutativity_report(teacher_m)
    print(f"  d_T={diag_m['d_T']} = r^2={diag_m['r_squared']} -> A_T=M_r: "
          f"{diag_m['is_full_matrix_algebra']} -> irreducible (direct argument): "
          f"{diag_m['is_irreducible']}")
    print(f"  commutant_dim={diag_m['commutant_dim']} (consistency check only, "
          f"NOT the source of the irreducibility claim)")

    print()
    print("=" * 70)
    print("CORRECTED BASELINE VERIFICATION -- Nonlinear RTU's own exact")
    print("block-local RTRL vs BPTT (per user review, item 5)")
    print("=" * 70)
    arch_rtu_v = make_nonlinear_rtu_arch(r_rtu=6, u_dim=2, hidden=8, seed=1)
    rngv = np.random.RandomState(2)
    Tv = 10
    Uv = jnp.array(rngv.randn(Tv, 2) * 0.4)
    yv = jnp.array(rngv.randn(Tv) * 0.3)
    for block_idx in range(3):
        for family in ("theta", "log_radius", "Wx"):
            if family == "theta":
                def loss_of(v, bi=block_idx):
                    p = dict(arch_rtu_v, thetas=arch_rtu_v["thetas"].at[bi].set(v))
                    return 0.5 * jnp.sum((nonlinear_rtu_rollout_scalar(jnp.zeros(6), Uv, p) - yv) ** 2) / Tv
                g_bptt = float(jax.grad(loss_of)(arch_rtu_v["thetas"][block_idx]))
                g_exact = rtu_exact_credit_grad(arch_rtu_v, Uv, yv, block_idx, family)[0]
                err = abs(g_bptt - g_exact)
            elif family == "log_radius":
                def loss_of(v, bi=block_idx):
                    p = dict(arch_rtu_v, log_radii=arch_rtu_v["log_radii"].at[bi].set(v))
                    return 0.5 * jnp.sum((nonlinear_rtu_rollout_scalar(jnp.zeros(6), Uv, p) - yv) ** 2) / Tv
                g_bptt = float(jax.grad(loss_of)(arch_rtu_v["log_radii"][block_idx]))
                g_exact = rtu_exact_credit_grad(arch_rtu_v, Uv, yv, block_idx, family)[0]
                err = abs(g_bptt - g_exact)
            else:
                def loss_of(v, bi=block_idx):
                    Wx = arch_rtu_v["Wx"].at[2 * bi:2 * bi + 2, :].set(v.reshape(2, 2))
                    p = dict(arch_rtu_v, Wx=Wx)
                    return 0.5 * jnp.sum((nonlinear_rtu_rollout_scalar(jnp.zeros(6), Uv, p) - yv) ** 2) / Tv
                v0 = jnp.concatenate([arch_rtu_v["Wx"][2 * block_idx], arch_rtu_v["Wx"][2 * block_idx + 1]])
                g_bptt = np.asarray(jax.grad(loss_of)(v0))
                g_exact = rtu_exact_credit_grad(arch_rtu_v, Uv, yv, block_idx, family)
                err = np.max(np.abs(g_bptt - g_exact))
            print(f"  block{block_idx}.{family}: |exact-bptt|={err:.2e}")

    print()
    print("=" * 70)
    print("POSITIVE CONTROL -- BLOCK-LOCAL teacher (an actual Nonlinear RTU")
    print("instance): RTU should fit near-perfectly; ours should too")
    print("=" * 70)
    rtu_teacher = make_nonlinear_rtu_arch(r_rtu=4, u_dim=2, hidden=16, seed=1)
    rngb = np.random.RandomState(1)
    U_train_bl = jnp.array(rngb.randn(20, 20, 2) * 0.5)
    y_train_bl = jnp.stack([nonlinear_rtu_rollout_scalar(jnp.zeros(4), U_train_bl[s], rtu_teacher)
                             for s in range(20)])
    U_test_bl = jnp.array(np.random.RandomState(999).randn(16, 20, 2) * 0.5)
    y_test_bl = jnp.stack([nonlinear_rtu_rollout_scalar(jnp.zeros(4), U_test_bl[s], rtu_teacher)
                            for s in range(16)])
    var_bl = float(jnp.var(y_train_bl))
    _, arch_rtu_bl = train_rtu_bptt_adam(r_rtu=4, u_dim=2, hidden=16, U_train=U_train_bl,
                                          y_train=y_train_bl, steps=800, lr=0.01, seed=10)
    test_mse_rtu_bl = eval_rtu_mse(arch_rtu_bl, U_test_bl, y_test_bl)
    print(f"  RTU r_rtu=4 on block-local teacher: NMSE={test_mse_rtu_bl/var_bl:.4f}")
    _, arch_o_bl, h0_o_bl = train_ours_bptt_adam(r=4, k=2, n=4, u_dim=2, hidden=16,
                                                  U_train=U_train_bl, y_train=y_train_bl,
                                                  steps=800, lr=0.01, seed=10)
    test_mse_o_bl = eval_ours_bptt_mse(arch_o_bl, h0_o_bl, U_test_bl, y_test_bl)
    print(f"  ours r=4,k=2,n=4 on block-local teacher: NMSE={test_mse_o_bl/var_bl:.4f}")

    print()
    print("=" * 70)
    print("DECISIVE COMPARISON -- GLOBALLY-COUPLED (A_T=M_r) teacher: ours vs")
    print("the CORRECT Nonlinear RTU baseline, h0=0, common BPTT+Adam")
    print("=" * 70)
    U_train, y_train = generate_teacher_batch(teacher_m, n_seq=20, T_=20, u_dim=2,
                                               n_teacher=6, r_teacher=4, seed=1)
    U_test, y_test = generate_teacher_batch(teacher_m, n_seq=16, T_=20, u_dim=2,
                                             n_teacher=6, r_teacher=4, seed=999)
    var_y = float(jnp.var(y_train))
    _, arch_o, h0_o = train_ours_bptt_adam(r=4, k=2, n=4, u_dim=2, hidden=16,
                                            U_train=U_train, y_train=y_train,
                                            steps=800, lr=0.01, seed=10)
    test_mse_o = eval_ours_bptt_mse(arch_o, h0_o, U_test, y_test)
    ours_params = ours_param_count(r=4, k=2, n=4, u_dim=2, hidden=16)
    ours_credit = ours_credit_floats(4, 2, 4, 2, 16)
    print(f"  OURS (total_state=16, params={ours_params}, credit={ours_credit}): "
          f"test_MSE={test_mse_o:.5f} NMSE={test_mse_o/var_y:.4f}")
    errs = verify_ours_trained_model_exact(arch_o, h0_o, U_train[0])
    print(f"  post-hoc exactness on trained model: {errs}")

    print("  Nonlinear RTU, MATCHED param-count regime:")
    for r_rtu in (4, 8, 16, 32, 64, 128):
        hidden_m = solve_hidden_for_target_params(r_rtu + 2, ours_params)
        _, arch_r = train_rtu_bptt_adam(r_rtu=r_rtu, u_dim=2, hidden=hidden_m,
                                         U_train=U_train, y_train=y_train,
                                         steps=800, lr=0.01, seed=10)
        test_mse = eval_rtu_mse(arch_r, U_test, y_test)
        p = rtu_param_count(r_rtu, 2, hidden_m)
        cf = rtu_credit_floats_verified(r_rtu, 2)
        print(f"    r_rtu={r_rtu:4d} hidden={hidden_m:3d} params={p:4d} credit={cf:4d}: "
              f"test_MSE={test_mse:.5f} NMSE={test_mse/var_y:.4f}")
    print("  Nonlinear RTU, STRONG (hidden=64) regime:")
    for r_rtu in (4, 8, 16, 32, 64, 128):
        _, arch_r = train_rtu_bptt_adam(r_rtu=r_rtu, u_dim=2, hidden=64,
                                         U_train=U_train, y_train=y_train,
                                         steps=800, lr=0.01, seed=10)
        test_mse = eval_rtu_mse(arch_r, U_test, y_test)
        p = rtu_param_count(r_rtu, 2, 64)
        cf = rtu_credit_floats_verified(r_rtu, 2)
        print(f"    r_rtu={r_rtu:4d} hidden=64  params={p:4d} credit={cf:4d}: "
              f"test_MSE={test_mse:.5f} NMSE={test_mse/var_y:.4f}")

    print()
    print("=" * 70)
    print("SUPPLEMENTARY -- TRUE commuting-teacher diagnostic (kept per user")
    print("instruction as supplementary, NOT the decisive falsifier)")
    print("=" * 70)
    teacher_c, coord = make_true_commuting_teacher(r=4, k=2, n=6, u_dim=2, hidden=16, seed=1)
    diag_c = noncommutativity_report(teacher_c)
    print(f"  coord={coord}  d_T={diag_c['d_T']} (r=4)  "
          f"max_RQ_comm={diag_c['max_RQ_commutator']:.2e}  max_QQ_comm={diag_c['max_QQ_commutator']:.2e}")

    print()
    print("=" * 70)
    print("SEED-ROBUSTNESS SWEEP (modest, 3 runs, per user review)")
    print("=" * 70)
    summary, failures = robustness_sweep(n_runs=3, r_rtu_points=(8, 64))
    for k, v in summary.items():
        print(f"  {k}: median={v['median']}  min={v['min']}  max={v['max']}  n={v['n']}")
    print(f"  failures: {failures if failures else 'none'}")


# ---------------------------------------------------------------------------
def robustness_sweep(n_runs=3, r_rtu_points=(8, 64)):
    results = {"block_local_rtu": [], "block_local_ours": [], "global_ours": []}
    for r_rtu in r_rtu_points:
        results[f"global_rtu_{r_rtu}"] = []
    failures = []

    for run in range(n_runs):
        teacher_seed = 1 + run * 17
        student_seed = 10 + run * 23
        data_seed = 1 + run * 31
        test_seed = 999 + run * 31

        try:
            # --- (A) block-local positive control ---
            rtu_teacher = make_nonlinear_rtu_arch(r_rtu=4, u_dim=2, hidden=16, seed=teacher_seed)
            rng_bl = np.random.RandomState(data_seed)
            U_train_bl = jnp.array(rng_bl.randn(20, 20, 2) * 0.5)
            y_train_bl = jnp.stack([nonlinear_rtu_rollout_scalar(jnp.zeros(4), U_train_bl[s], rtu_teacher)
                                     for s in range(20)])
            U_test_bl = jnp.array(np.random.RandomState(test_seed).randn(16, 20, 2) * 0.5)
            y_test_bl = jnp.stack([nonlinear_rtu_rollout_scalar(jnp.zeros(4), U_test_bl[s], rtu_teacher)
                                    for s in range(16)])
            var_bl = float(jnp.var(y_train_bl))
            if var_bl < 1e-8:
                failures.append((run, "block_local", "degenerate target variance"))
                continue

            _, arch_rtu_bl = train_rtu_bptt_adam(r_rtu=4, u_dim=2, hidden=16, U_train=U_train_bl,
                                                  y_train=y_train_bl, steps=800, lr=0.01, seed=student_seed)
            nmse_rtu_bl = eval_rtu_mse(arch_rtu_bl, U_test_bl, y_test_bl) / var_bl
            _, arch_o_bl, h0_o_bl = train_ours_bptt_adam(r=4, k=2, n=4, u_dim=2, hidden=16,
                                                          U_train=U_train_bl, y_train=y_train_bl,
                                                          steps=800, lr=0.01, seed=student_seed)
            nmse_o_bl = eval_ours_bptt_mse(arch_o_bl, h0_o_bl, U_test_bl, y_test_bl) / var_bl
            results["block_local_rtu"].append(nmse_rtu_bl)
            results["block_local_ours"].append(nmse_o_bl)

            # --- (B) global A_T=M_r teacher, verified before use ---
            global_teacher = make_noncommutative_teacher(r=4, k=2, n=6, u_dim=2, hidden=16, seed=teacher_seed)
            diag = noncommutativity_report(global_teacher)
            if not diag["is_full_matrix_algebra"]:
                failures.append((run, "global_teacher", f"dim A_T={diag['d_T']} != r^2=16, SKIPPED"))
                continue

            U_train, y_train = generate_teacher_batch(global_teacher, n_seq=20, T_=20, u_dim=2,
                                                       n_teacher=6, r_teacher=4, seed=data_seed)
            U_test, y_test = generate_teacher_batch(global_teacher, n_seq=16, T_=20, u_dim=2,
                                                     n_teacher=6, r_teacher=4, seed=test_seed)
            var_g = float(jnp.var(y_train))
            if var_g < 1e-8:
                failures.append((run, "global", "degenerate target variance"))
                continue

            _, arch_o, h0_o = train_ours_bptt_adam(r=4, k=2, n=4, u_dim=2, hidden=16,
                                                    U_train=U_train, y_train=y_train,
                                                    steps=800, lr=0.01, seed=student_seed)
            nmse_o = eval_ours_bptt_mse(arch_o, h0_o, U_test, y_test) / var_g
            results["global_ours"].append(nmse_o)

            for r_rtu in r_rtu_points:
                hidden_m = solve_hidden_for_target_params(
                    r_rtu + 2, ours_param_count(4, 2, 4, 2, 16))
                _, arch_r = train_rtu_bptt_adam(r_rtu=r_rtu, u_dim=2, hidden=hidden_m,
                                                 U_train=U_train, y_train=y_train,
                                                 steps=800, lr=0.01, seed=student_seed)
                nmse_r = eval_rtu_mse(arch_r, U_test, y_test) / var_g
                results[f"global_rtu_{r_rtu}"].append(nmse_r)
        except Exception as e:
            failures.append((run, "exception", str(e)))

    summary = {}
    for key, vals in results.items():
        if vals:
            arr = np.array(vals)
            summary[key] = dict(median=float(np.median(arr)), min=float(np.min(arr)),
                                 max=float(np.max(arr)), n=len(arr))
        else:
            summary[key] = dict(median=None, min=None, max=None, n=0)
    return summary, failures


if __name__ == "__main__":
    main()
