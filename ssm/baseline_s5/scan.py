"""BASELINE S5 — the standard first-order elementwise associative scan.

The S5 recurrence is first order and diagonal, so each time step is a scalar
affine map (a_t, b_t) acting on the state:

    s_t = a_t * s_{t-1} + b_t          (elementwise, complex)

Affine maps compose associatively,

    (a_j, b_j) o (a_i, b_i) = (a_j * a_i, a_j * b_i + b_j)

(element j is LATER in time than element i), which is exactly what
``jax.lax.associative_scan`` needs — the whole sequence resolves in
O(log T) parallel depth.

Three implementations, all the same math:

    elementwise_scan             parallel associative scan  (DEFAULT)
    elementwise_scan_sequential  python-loop reference      (tests only)
    elementwise_scan_lax         jax.lax.scan, sequential   (faster on XLA-CPU)

Arrays have TIME as the leading axis; all trailing axes ride along
elementwise.

Compare with ``ssm/prospective/scan.py``, where the second-order prospective
recurrence forces the same idea onto 2x2 matrix affine maps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _elem_binary_op(q_i: tuple[jnp.ndarray, jnp.ndarray],
                    q_j: tuple[jnp.ndarray, jnp.ndarray]
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compose two first-order affine maps; map i is earlier in time."""
    a_i, b_i = q_i
    a_j, b_j = q_j
    return a_j * a_i, a_j * b_i + b_j


def elementwise_scan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """S5-style associative scan of s_t = a_t * s_{t-1} + b_t with s_{-1} = 0.

    Args:
        a: (T, ...) complex multiplier sequence.
        b: (T, ...) complex input sequence.

    Returns:
        s: (T, ...) complex states s_0 .. s_{T-1}.
    """
    _, s = jax.lax.associative_scan(_elem_binary_op, (a, b), axis=0)
    return s


def elementwise_scan_sequential(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Sequential reference for ``elementwise_scan`` (s_{-1} = 0)."""
    T = a.shape[0]
    s = jnp.zeros_like(b[0])
    out = []
    for t in range(T):
        s = a[t] * s + b[t]
        out.append(s)
    return jnp.stack(out, axis=0)


def elementwise_scan_lax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Sequential (jax.lax.scan) version of ``elementwise_scan``.

    Same math as the associative scan; kept because XLA-CPU executes the
    Blelloch stages of ``associative_scan`` with high overhead, so a
    sequential scan is much faster on a few CPU cores. On GPU use "assoc".
    """

    def step(s, ab):
        a_t, b_t = ab
        s = a_t * s + b_t
        return s, s

    s0 = jnp.zeros_like(b[0])
    _, s = jax.lax.scan(step, s0, (a, b))
    return s


def select_scan(scan_impl: str):
    """Pick the baseline scan implementation.

    "assoc" — ``jax.lax.associative_scan`` (default; the parallel-scan proof).
    "lax"   — sequential ``jax.lax.scan`` (much faster on XLA-CPU).
    """
    if scan_impl == "assoc":
        return elementwise_scan
    if scan_impl == "lax":
        return elementwise_scan_lax
    raise ValueError(f"unknown scan_impl: {scan_impl}")
