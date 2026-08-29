"""Phase B28 -- permanent regression test for the "dual-role parameter"
bug class found in Stage 1 (credit_memory/b28_popgym_stage1.py).

GENERAL PATTERN (not an ad-hoc fix for one parameter): a parameter
family theta can affect a loss through TWO distinct paths --

  1. RECURRENT: theta -> the recurrence -> h_t (t=1..T) -> loss.
     Handled by factorized_rtrl_run(theta, ..., dLdh) -- dLdh already
     captures the loss's dependence on the STATE, whatever form that
     dependence takes, evaluated with theta held at its CURRENT value.
  2. DIRECT/INSTANTANEOUS: theta appears explicitly in the loss
     formula itself, evaluated at a FIXED (already-computed) state
     sequence -- e.g. a readout z_t=H_t@C.T used directly downstream,
     not merely theta's effect on h_t's own future values.

The EXACT total gradient is always:

    dL/dtheta = recurrent_credit(theta) + direct_instantaneous(theta)

The loss is expressed through a SINGLE function `loss_fn(H, theta)`
taking the state sequence and the parameter as SEPARATE, explicit
arguments -- this is what makes the three distinct evaluations
(dLdh, the direct term, and the BPTT reference) unambiguous:

  dLdh                 : jax.grad wrt H of loss_fn(H, theta_current)
  direct_instantaneous : jax.grad wrt theta of loss_fn(H_actual, theta), H fixed
  BPTT reference       : jax.grad wrt theta of loss_fn(rollout(theta), theta)

(An earlier draft of this test conflated these three evaluations via a
closure that silently fixed theta in one of them -- caught by the
test itself giving a large, non-noise error on a case already known
to work from Stage 1, investigated rather than assumed to be a
genuine failure, and fixed by this explicit three-argument design.)

Run: python -m credit_memory.b28_regression_dual_role
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b25_nonlinear_credit import (
    make_arch, rollout, dLdh_from_target, factorized_rtrl_run,
)


def total_gradient_with_direct_role(arch, h0, U_seq, family, loss_fn):
    """loss_fn(H, theta) -- THE loss, with the state sequence and the
    parameter as explicit, separate arguments. Returns
    recurrent_credit(theta) + direct_instantaneous(theta), evaluated
    at the CURRENT arch[family]."""
    theta_current = arch[family]

    def target_fn_of_H(H):
        return loss_fn(H, theta_current)

    dLdh = dLdh_from_target(arch, h0, U_seq, target_fn_of_H)
    g_recurrent, _ = factorized_rtrl_run(family, arch, h0, U_seq, dLdh, use_naive=False)
    H_actual, _, _ = rollout(h0, U_seq, arch)
    g_direct = np.asarray(jax.grad(lambda th: loss_fn(H_actual, th))(theta_current)).reshape(-1)
    return g_recurrent + g_direct


def bptt_grad_dual_role(arch, h0, U_seq, family, loss_fn):
    """Ground truth: theta threaded through BOTH the rollout and the
    loss's own direct use of it, in one consistent jax.grad call."""
    def full_loss(theta):
        arch_ = dict(arch, **{family: theta})
        H, _, _ = rollout(h0, U_seq, arch_)
        return loss_fn(H, theta)
    return np.asarray(jax.grad(full_loss)(arch[family])).reshape(-1)


# ---------------------------------------------------------------------------
# Case 1: ARTIFICIAL dual role for family='R' (which has NO natural
# direct role -- R only ever affects h_t through the recurrence). This
# proves the decomposition pattern itself is general, not a fix
# specific to C's natural readout role.
# ---------------------------------------------------------------------------
def artificial_R_dual_role_case(seed=1):
    r, k, n, u_dim, hidden = 3, 2, 3, 2, 8
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.array(rng.randn(n, r) * 0.2)
    T_ = 8
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    target = jnp.array(rng.randn(T_ + 1, n, r) * 0.3)

    def loss_fn(H, R_theta):
        R_mat = R_theta.reshape(r, r)
        recurrent_part = 0.5 * jnp.sum((H - target) ** 2) / T_
        # artificial direct role: R has NO natural business appearing
        # outside the recurrence -- constructed purely to exercise the
        # general two-term pattern on a family that never needs it
        # naturally.
        direct_part = 0.1 * 0.5 * jnp.sum((H[-1] @ R_mat.T - target[-1]) ** 2)
        return recurrent_part + direct_part

    g_decomposed = total_gradient_with_direct_role(arch, h0, U_seq, "R", loss_fn)
    g_bptt = bptt_grad_dual_role(arch, h0, U_seq, "R", loss_fn)
    return float(np.max(np.abs(g_decomposed - g_bptt)))


# ---------------------------------------------------------------------------
# Case 2: the NATURALLY-OCCURRING case (family='C', found in B28 Stage 1).
# ---------------------------------------------------------------------------
def natural_C_dual_role_case(seed=1):
    from credit_memory.b28_popgym_stage1 import make_head_params, ours_target_fn

    r, k, n, u_dim, hidden = 3, 2, 3, 2, 8
    arch = make_arch(r=r, k=k, n=n, u_dim=u_dim, hidden=hidden, seed=seed)
    rng = np.random.RandomState(seed + 1)
    h0 = jnp.array(rng.randn(n, r) * 0.2)
    T_ = 8
    U_seq = jnp.array(rng.randn(T_, u_dim) * 0.4)
    head = make_head_params(z_dim=n * k, num_actions=4, seed=seed + 2)
    actions = list(rng.randint(0, 4, T_))
    advantages = rng.randn(T_) * 0.3
    returns = rng.randn(T_) * 0.3

    def loss_fn(H, C_theta):
        return ours_target_fn(H, C_theta.reshape(k, r), head, actions, advantages, returns)

    g_decomposed = total_gradient_with_direct_role(arch, h0, U_seq, "C", loss_fn)
    g_bptt = bptt_grad_dual_role(arch, h0, U_seq, "C", loss_fn)
    return float(np.max(np.abs(g_decomposed - g_bptt)))


def main():
    print("=" * 70)
    print("PERMANENT REGRESSION TEST -- dual-role parameter decomposition")
    print("dL/dtheta = recurrent_credit(theta) + direct_instantaneous(theta)")
    print("=" * 70)
    err_artificial = artificial_R_dual_role_case()
    print(f"  Case 1 (ARTIFICIAL dual role, family=R, proves the PATTERN is "
          f"general): |err|={err_artificial:.2e}")
    err_natural = natural_C_dual_role_case()
    print(f"  Case 2 (NATURAL dual role, family=C, the bug found in Stage 1): "
          f"|err|={err_natural:.2e}")
    assert err_artificial < 1e-9, "REGRESSION: artificial dual-role case failed"
    assert err_natural < 1e-9, "REGRESSION: natural dual-role case (C) failed"
    print("  PASS")


if __name__ == "__main__":
    main()
