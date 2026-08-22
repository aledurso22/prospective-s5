"""SHARED — complex-parameter plumbing and the S5 initializers.

Used by BOTH arms (``ssm.baseline_s5`` and ``ssm.prospective``): the two
layers are initialized identically, so any difference between them comes from
the recurrence, never from the initialization.

Contents
    ComplexParam    a complex flax parameter stored as two real arrays
    hippo_lambda_init / hippo_B_init / C_init / log_step_init
                    the S5 initializers (HiPPO-LegS, DPLR route)

The prospective arm reuses all of these except ``log_step_init``: the
prospective derivation contains no Delta, so that initializer has no
counterpart there.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from .hippo import hippo_init


# ---------------------------------------------------------------------------
# Parameter initializers (complex parameters stored as real/imag pairs)
# ---------------------------------------------------------------------------

def hippo_lambda_init(state_size: int):
    """Initializer for Lambda from HiPPO-LegS eigenvalues."""
    Lambda, _, _ = hippo_init(state_size)

    def init(key, shape, dtype=jnp.float32):
        assert shape == (state_size,)
        return jnp.asarray(Lambda, jnp.complex64)
    return init


def hippo_B_init(state_size: int, d_model: int):
    """Initializer for B in the diagonal basis: B~ = V^{-1} B_hippo, (N, H).

    Each of the H columns is initialized with the same HiPPO input vector
    transformed to the diagonal basis (S5-style).
    """
    _, _, B_tilde = hippo_init(state_size)  # (N, 1) complex

    def init(key, shape, dtype=jnp.float32):
        assert shape == (state_size, d_model)
        B = np.repeat(np.asarray(B_tilde), d_model, axis=1)  # (N, H)
        return jnp.asarray(B, jnp.complex64)
    return init


def C_init(state_size: int):
    """S5-style C init: C = lecun_normal(H, N) @ V (real-to-complex)."""
    _, V, _ = hippo_init(state_size)  # (N, N) unitary

    def init(key, shape, dtype=jnp.float32):
        H, N = shape
        C_real = nn.initializers.lecun_normal()(key, (H, N), jnp.float32)
        return (C_real @ jnp.asarray(V)).astype(jnp.complex64)
    return init


def log_step_init(key, shape, dtype=jnp.float32):
    """S5-style log-Delta init: log Delta ~ U[log(1e-3), log(1e-1)]."""
    return jax.random.uniform(
        key, shape, dtype=jnp.float32,
        minval=np.log(1e-3), maxval=np.log(1e-1))


# ---------------------------------------------------------------------------
# Complex parameter container
# ---------------------------------------------------------------------------

class ComplexParam(nn.Module):
    """A complex parameter stored as separate real/imag arrays."""
    shape: tuple
    init_fn: callable

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        def real_init(key, shape, dtype=jnp.float32):
            return jnp.asarray(self.init_fn(key, shape, dtype).real, jnp.float32)

        def imag_init(key, shape, dtype=jnp.float32):
            return jnp.asarray(self.init_fn(key, shape, dtype).imag, jnp.float32)

        re = self.param("re", real_init, self.shape)
        im = self.param("im", imag_init, self.shape)
        return re.astype(jnp.complex64) + 1j * im.astype(jnp.complex64)
