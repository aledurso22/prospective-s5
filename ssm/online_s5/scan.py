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
