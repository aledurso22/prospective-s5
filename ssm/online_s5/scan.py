"""ONLINE S5 — custom-VJP scan: forward is the exact S5 recurrence; the
backward is the online rule (no temporal cotangent transport).

Real-parameter interface (matching the rig's FD-gated convention
(Re G, -Im G) for complex parameters): all inputs/outputs real.

Per channel (x (T,) real; modes a (N,) complex from a_re+1j*a_im; input
weights Bb; readout C; passthrough D real):

  forward:  s_t = a * s_{t-1} + Bb * x_t ;  y_t = Re(C s_t) + D x_t
  backward (given dy (T,) real cotangent on y):
    q_t  = conj(C) * dy_t          (instantaneous credit to s_t)
    Sa_t = s_{t-1} + a * Sa_{t-1}  (causal sensitivities)
    Sb_t = x_t     + a * Sb_{t-1}
    Ga  = sum_t conj(q_t) * Sa_t ;  ct_a = (Re Ga, -Im Ga)
    Gb  = sum_t conj(q_t) * Sb_t ;  ct_B = (Re Gb, -Im Gb)
    Gc  = sum_t dy_t * s_t     ;    ct_C = (Re Gc, -Im Gc)
    dD  = sum_t dy_t * x_t
    dx_t = dy_t * (Re(C * Bb) + D)  (instantaneous input path ONLY —
                                     no A^T lam_{t+1}; this is what
                                     makes the rule online)

Validated against trained_credit_gains.py's online gradient to ~1e-7 on
identical single-layer configs (check_online_s5.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_vjp
def ssm_online(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x):
    """One channel, real interface. a_*, Bb_*, C_*: (N,) real; D: scalar;
    x: (T,) real. Returns y: (T,) real."""
    a = a_re + 1j * a_im
    Bb = Bb_re + 1j * Bb_im
    C = C_re + 1j * C_im
    Bu = x[:, None].astype(jnp.complex64) * Bb[None, :]

    def step(s, xu):
        s = a * s + xu
        return s, s

    _, s = jax.lax.scan(step, jnp.zeros_like(a), Bu)
    y = jnp.einsum("n,tn->t", C, s).real + D * x
    return y


def _fwd(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x):
    a = a_re + 1j * a_im
    Bb = Bb_re + 1j * Bb_im
    Bu = x[:, None].astype(jnp.complex64) * Bb[None, :]

    def step(s, xu):
        s = a * s + xu
        return s, s

    _, s = jax.lax.scan(step, jnp.zeros_like(a), Bu)
    y = jnp.einsum("n,tn->t", C_re + 1j * C_im, s).real + D * x
    return y, (s, a, Bb, C_re + 1j * C_im, D, x)


def _bwd(res, dy):
    s, a, Bb, C, D, x = res

    def bstep(carry, inp):
        Sa, Sb, ga_acc, gb_acc = carry
        s_prev, x_t, dy_t = inp
        Sa = s_prev + a * Sa
        Sb = x_t + a * Sb
        q_t = jnp.conj(C) * dy_t
        ga_acc = ga_acc + jnp.conj(q_t) * Sa
        gb_acc = gb_acc + jnp.conj(q_t) * Sb
        return (Sa, Sb, ga_acc, gb_acc), None

    s_prev = jnp.concatenate([jnp.zeros_like(s[:1]), s[:-1]], axis=0)
    (Sa_T, Sb_T, Ga, Gb), _ = jax.lax.scan(
        bstep,
        (jnp.zeros_like(a), jnp.zeros_like(a),
         jnp.zeros_like(a), jnp.zeros_like(a)),
        (s_prev, x, dy))
    Gc = jnp.einsum("t,tn->n", dy.astype(jnp.complex64), s)
    dD = jnp.dot(dy, x)
    dx = dy * (jnp.real(jnp.sum(C * Bb)) + D)
    return (Ga.real, -Ga.imag, Gb.real, -Gb.imag, Gc.real, -Gc.imag,
            dD, dx)


ssm_online.defvjp(_fwd, _bwd)


# ---------------------------------------------------------------------------
# Rotated variant: per-mode complex orientation w applied in the backward
# ---------------------------------------------------------------------------

def _rot_impl(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x, w_re, w_im):
    """Forward of ``ssm_online_rot`` — IDENTICAL to ``ssm_online`` (w unused)."""
    a = a_re + 1j * a_im
    Bb = Bb_re + 1j * Bb_im
    C = C_re + 1j * C_im
    Bu = x[:, None] * Bb[None, :]

    def step(s, xu):
        s = a * s + xu
        return s, s

    _, s = jax.lax.scan(step, jnp.zeros(a.shape, jnp.result_type(a, Bu)),
                        Bu)
    y = jnp.einsum("n,tn->t", C, s).real + D * x
    return y


@jax.custom_vjp
def ssm_online_rot(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x, w_re, w_im):
    """``ssm_online`` with a per-mode complex rotation in the backward.

    Forward is IDENTICAL to ``ssm_online`` (w is not used). The backward
    rotates the mode-gradient blocks before they leave the custom VJP:

        Ga <- conj(w) * Ga ;  Gb <- conj(w) * Gb ;  Gc, dD, dx untouched

    — the Route-A prescription (co_variational_metric.py's ``scale_by_w``):
    the readout gradient and the instantaneous input cotangent are NOT
    rotated. w = 1+0j reproduces the plain online gradient bit-for-bit.

    w enters through the residuals, so differentiating the backward output
    with respect to (w_re, w_im) — the Route-A meta-gradient through the
    one-step-lookahead update — is ordinary higher-order autodiff. The
    first-order cotangents for w are correctly zero (w does not affect the
    forward). Complex dtype follows promotion (complex64 in float32
    training, complex128 under x64 for finite-difference checks).
    """
    return _rot_impl(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x,
                     w_re, w_im)


def _rot_fwd(a_re, a_im, Bb_re, Bb_im, C_re, C_im, D, x, w_re, w_im):
    a = a_re + 1j * a_im
    Bb = Bb_re + 1j * Bb_im
    Bu = x[:, None] * Bb[None, :]

    def step(s, xu):
        s = a * s + xu
        return s, s

    _, s = jax.lax.scan(step, jnp.zeros(a.shape, jnp.result_type(a, Bu)),
                        Bu)
    y = jnp.einsum("n,tn->t", C_re + 1j * C_im, s).real + D * x
    return y, (s, a, Bb, C_re + 1j * C_im, D, x, w_re, w_im)


def _rot_bwd(res, dy):
    s, a, Bb, C, D, x, w_re, w_im = res
    w = w_re + 1j * w_im
    cdtype = jnp.result_type(s, a, C, dy)

    def bstep(carry, inp):
        Sa, Sb, ga_acc, gb_acc = carry
        s_prev, x_t, dy_t = inp
        Sa = s_prev + a * Sa
        Sb = x_t + a * Sb
        q_t = jnp.conj(C) * dy_t
        ga_acc = ga_acc + jnp.conj(q_t) * Sa
        gb_acc = gb_acc + jnp.conj(q_t) * Sb
        return (Sa, Sb, ga_acc, gb_acc), None

    z = jnp.zeros(a.shape, cdtype)
    s_prev = jnp.concatenate([jnp.zeros_like(s[:1]), s[:-1]], axis=0)
    (Sa_T, Sb_T, Ga, Gb), _ = jax.lax.scan(
        bstep, (z, z, z, z), (s_prev, x, dy))
    # ---- the Route-A rotation: conj(w) on the recurrence-mode blocks ----
    Ga = jnp.conj(w) * Ga
    Gb = jnp.conj(w) * Gb
    Gc = jnp.einsum("t,tn->n", dy.astype(s.dtype), s)
    dD = jnp.dot(dy, x)
    dx = dy * (jnp.real(jnp.sum(C * Bb)) + D)
    return (Ga.real, -Ga.imag, Gb.real, -Gb.imag, Gc.real, -Gc.imag,
            dD, dx, jnp.zeros_like(w_re), jnp.zeros_like(w_im))


ssm_online_rot.defvjp(_rot_fwd, _rot_bwd)
