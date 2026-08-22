"""SHARED — SequenceClassifier: the arm selector.

    linear encoder (1 -> H) -> L x S5Block -> mean pool over time
                            -> LayerNorm -> linear head

The ONLY place the two arms diverge is the `model_type` switch below, which
plugs either ``S5SSM`` (baseline) or ``ProspectiveSSM`` (prospective) into the
otherwise identical block stack. Same encoder, same block, same pooling, same
head, same training loop (`train.py`) — so a measured difference between the
arms is attributable to the recurrence alone.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from .shared.block import S5Block
from .baseline_s5.layer import S5SSM
from .prospective.layer import ProspectiveSSM


class SequenceClassifier(nn.Module):
    """Sequence-to-label classifier.

    Pipeline: per-timestep linear encoder (1 -> H) -> L stacked S5Blocks ->
    mean pooling over time -> linear classification head (H -> n_classes).
    """
    model_type: str = "baseline"   # {"baseline", "prospective"}
    d_model: int = 96              # H
    state_size: int = 64           # N
    n_layers: int = 3              # L
    n_classes: int = 10
    dropout_rate: float = 0.1
    scan_impl: str = "assoc"       # "assoc" | "lax" (CPU training accelerator)
    gamma: float = 1.0             # prospective strength; 0 = first-order
                                   # (explicit-Euler S5), 1 = the derivation.
    rho_init: float = 0.1          # prospective friction rho at init; the
                                   # physical mode is ~ (1 - rho), so the
                                   # memory horizon is ~ 1/rho_init tokens.
                                   # Use 1e-3 for long sequences (L ~ 1e3).

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: (batch, T) or (batch, T, 1) real input sequence.
            train: whether dropout is active.
        Returns:
            logits: (batch, n_classes)
        """
        if x.ndim == 2:
            x = x[..., None]
        x = nn.Dense(self.d_model)(x)                       # (B, T, H)
        for _ in range(self.n_layers):
            # ---- the ONLY difference between the two arms ----
            if self.model_type == "baseline":
                ssm = S5SSM(state_size=self.state_size, d_model=self.d_model,
                            scan_impl=self.scan_impl)
            elif self.model_type == "prospective":
                ssm = ProspectiveSSM(state_size=self.state_size,
                                     d_model=self.d_model,
                                     scan_impl=self.scan_impl,
                                     gamma=self.gamma,
                                     log_ratio_init=float(np.log(self.rho_init)))
            else:
                raise ValueError(f"unknown model_type: {self.model_type}")
            x = S5Block(ssm=ssm, d_model=self.d_model,
                        dropout_rate=self.dropout_rate)(x, train=train)
        x = jnp.mean(x, axis=1)                             # mean pool over time
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.n_classes)(x)                # (B, n_classes)
        return logits


def count_params(params) -> int:
    """Total number of scalar parameters in a flax param pytree."""
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


def build_model(model_type: str, d_model: int = 96, state_size: int = 64,
                n_layers: int = 3, n_classes: int = 10,
                dropout_rate: float = 0.1,
                scan_impl: str = "assoc",
                gamma: float = 1.0,
                rho_init: float = 0.1) -> SequenceClassifier:
    """Convenience constructor."""
    return SequenceClassifier(
        model_type=model_type, d_model=d_model, state_size=state_size,
        n_layers=n_layers, n_classes=n_classes, dropout_rate=dropout_rate,
        scan_impl=scan_impl, gamma=gamma, rho_init=rho_init)
