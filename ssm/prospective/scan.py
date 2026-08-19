"""PROSPECTIVE — associative scan for the second-order prospective recurrence.

THIS IS NEW WORK. The baseline needs only a scalar affine map per step
(``ssm/baseline_s5/scan.py``); the prospective update is a TWO-step
recurrence, so its state must be stacked, z_t = [s_t ; s_{t-1}], and each
time step becomes a 2x2 MATRIX affine map:

    z_t = M_t z_{t-1} + b_t,     M = [[a1, a2], [1, 0]],   b_t = [x~_t ; 0]

Matrix affine maps still compose associatively,

    (M_j, b_j) o (M_i, b_i) = (M_j @ M_i, M_j @ b_i + b_j)

(map i applied FIRST / earlier in time), so the prospective recurrence is
still parallel-scan compatible — the point of this file.

Note the companion form [[a1, a2], [1, 0]] is NOT closed under matrix
multiplication, so a composed element must carry all four entries.

Implementations
    prospective_scan             literal (T,...,2,2) einsum scan   (reference)
    prospective_scan_sequential  python-loop reference             (tests only)
    prospective_scan_flat        SAME algebra on 6 flat components  (DEFAULT,
                                 what the layer runs)
    prospective_scan_companion_lax  sequential 2nd-order recurrence (CPU path)

All four agree to float32 tolerance — see ``test_scan.py`` [2] [3] [4b].
Arrays have TIME as the leading axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1. Literal 2x2 form — the readable statement of the algebra
# ---------------------------------------------------------------------------

def _mat_binary_op(q_i: tuple[jnp.ndarray, jnp.ndarray],
                   q_j: tuple[jnp.ndarray, jnp.ndarray]
                   ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compose two affine maps (M, b); map i is applied first (earlier).

        (M_j, b_j) o (M_i, b_i) = (M_j @ M_i, M_j @ b_i + b_j)
    """
    M_i, b_i = q_i
    M_j, b_j = q_j
    M = jnp.einsum("...ab,...bc->...ac", M_j, M_i)
    b = jnp.einsum("...ab,...b->...a", M_j, b_i) + b_j
    return M, b


def prospective_scan(M: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Associative scan of z_t = M_t z_{t-1} + b_t with z_{-1} = 0.

    Args:
        M: (T, ..., 2, 2) complex transition matrices.
        b: (T, ..., 2)    complex input vectors.

    Returns:
        z: (T, ..., 2) complex stacked states z_0 .. z_{T-1}.
    """
    _, z = jax.lax.associative_scan(_mat_binary_op, (M, b), axis=0)
    return z


def prospective_scan_sequential(M: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Sequential reference for ``prospective_scan`` (z_{-1} = 0)."""
    T = M.shape[0]
    z = jnp.zeros_like(b[0])
    out = []
    for t in range(T):
        z = jnp.einsum("...ab,...b->...a", M[t], z) + b[t]
        out.append(z)
    return jnp.stack(out, axis=0)


# ---------------------------------------------------------------------------
# 2. Flattened 2x2 form — the fast path the layer actually uses
#
# Identical composition algebra to ``_mat_binary_op``, written as pure
# ELEMENTWISE ops on the 6 components (m00, m01, m10, m11, b0, b1) instead of
# (T, ..., 2, 2) einsums: the full 2x2 matrix product (all four entries must
# be carried, the companion form is not closed under multiplication), but much
# faster on XLA. Validated against the sequential reference in test_scan.py.
# ---------------------------------------------------------------------------

def _flat_binary_op(e_i, e_j):
    m00_i, m01_i, m10_i, m11_i, b0_i, b1_i = e_i
    m00_j, m01_j, m10_j, m11_j, b0_j, b1_j = e_j
    return (m00_j * m00_i + m01_j * m10_i,
            m00_j * m01_i + m01_j * m11_i,
            m10_j * m00_i + m11_j * m10_i,
            m10_j * m01_i + m11_j * m11_i,
            m00_j * b0_i + m01_j * b1_i + b0_j,
            m10_j * b0_i + m11_j * b1_i + b1_j)


def prospective_scan_flat(a1: jnp.ndarray, a2: jnp.ndarray,
                          b0: jnp.ndarray) -> jnp.ndarray:
    """Associative scan of the prospective recurrence (flattened 2x2 form).

    Computes s_t of  z_t = [[a1,a2],[1,0]] z_{t-1} + [b0_t; 0]  (z_{-1} = 0),
    i.e. s_t = a1 s_{t-1} + a2 s_{t-2} + b0_t with s_{-1} = s_{-2} = 0.

    Args:
        a1, a2: (T, ...) complex companion coefficients (time-constant in
            practice, but the scan does not rely on that).
        b0: (T, ...) complex input sequence (x~_t).

    Returns:
        s: (T, ...) complex states s_0 .. s_{T-1}.
    """
    one = jnp.ones_like(a1)
    zero = jnp.zeros_like(a1)
    b1 = jnp.zeros_like(b0)
    *_, s, _ = jax.lax.associative_scan(
        _flat_binary_op, (a1, a2, one, zero, b0, b1), axis=0)
    return s


def prospective_scan_companion_lax(a1: jnp.ndarray, a2: jnp.ndarray,
                                   b0: jnp.ndarray) -> jnp.ndarray:
    """Sequential (jax.lax.scan) version of ``prospective_scan_flat`` — the
    direct second-order recurrence (identical math)."""

    def step(carry, xs):
        s_prev1, s_prev2 = carry
        a1_t, a2_t, b0_t = xs
        s = a1_t * s_prev1 + a2_t * s_prev2 + b0_t
        return (s, s_prev1), s

    z0 = (jnp.zeros_like(b0[0]), jnp.zeros_like(b0[0]))
    _, s = jax.lax.scan(step, z0, (a1, a2, b0))
    return s


def select_scan(scan_impl: str):
    """Pick the prospective scan implementation (companion/flat kernels).

    "assoc" — ``prospective_scan_flat``, the parallel associative scan
              (default; same affine-map algebra as ``prospective_scan``,
              elementwise ops instead of einsums).
    "lax"   — ``prospective_scan_companion_lax``, sequential (XLA-CPU).
    """
    if scan_impl == "assoc":
        return prospective_scan_flat
    if scan_impl == "lax":
        return prospective_scan_companion_lax
    raise ValueError(f"unknown scan_impl: {scan_impl}")
