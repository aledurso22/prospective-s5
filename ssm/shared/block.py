"""SHARED — the S5 residual block wrapper.

    LayerNorm -> SSM -> dropout -> GLU -> dropout -> residual

The block is arm-agnostic: it takes whichever SSM layer it is handed
(``S5SSM`` or ``ProspectiveSSM``), so the two arms differ ONLY in the SSM
they plug in here. Everything around the recurrence is identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class S5Block(nn.Module):
    """LayerNorm -> SSM -> GLU/MLP -> residual block with dropout."""
    ssm: nn.Module
    d_model: int = 96
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: (batch, T, H) input sequence.
            train: whether dropout is active.
        Returns:
            (batch, T, H) output sequence.
        """
        z = nn.LayerNorm()(x)
        z = self.ssm(z)
        z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not train)
        # GLU: gated linear unit over the feature dim.
        z = nn.Dense(self.d_model)(z) * jax.nn.sigmoid(nn.Dense(self.d_model)(z))
        z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not train)
        return x + z
