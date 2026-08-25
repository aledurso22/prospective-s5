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
from .online_s5.layer import OnlineS5SSM


class SequenceClassifier(nn.Module):
    """Sequence model: per-timestep linear encoder -> L stacked S5Blocks ->
    head.

    Default head: mean pooling over time -> LayerNorm -> linear head
    (sequence-to-label). With ``seq2seq=True``: LayerNorm -> per-timestep
    linear head (sequence-to-sequence, e.g. the copy task — loss masking
    selects the scored positions).
    """
    model_type: str = "baseline"   # {"baseline", "prospective", "online",
                                   #  "tbptt"}
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
    seq2seq: bool = False          # per-timestep head, no mean pooling
    tbptt_window: int = 0          # model_type="tbptt": backward truncation
                                   # window (forward unchanged)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: (batch, T), (batch, T, 1) or (batch, T, D_in) real input.
            train: whether dropout is active.
        Returns:
            logits: (batch, n_classes), or (batch, T, n_classes) if seq2seq.
        """
        if x.ndim == 2:
            x = x[..., None]
        x = nn.Dense(self.d_model)(x)                       # (B, T, H)
        for _ in range(self.n_layers):
            # ---- the ONLY difference between the two arms ----
            if self.model_type == "baseline":
                ssm = S5SSM(state_size=self.state_size, d_model=self.d_model,
                            scan_impl=self.scan_impl)
            elif self.model_type == "online":
                ssm = OnlineS5SSM(state_size=self.state_size,
                                  d_model=self.d_model)
            elif self.model_type == "tbptt":
                ssm = S5SSM(state_size=self.state_size, d_model=self.d_model,
                            scan_impl=self.scan_impl,
                            tbptt_window=self.tbptt_window)
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
        if self.seq2seq:
            x = nn.LayerNorm()(x)
            logits = nn.Dense(self.n_classes)(x)            # (B, T, K)
            return logits
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
                rho_init: float = 0.1,
                seq2seq: bool = False,
                tbptt_window: int = 0) -> SequenceClassifier:
    """Convenience constructor."""
    return SequenceClassifier(
        model_type=model_type, d_model=d_model, state_size=state_size,
        n_layers=n_layers, n_classes=n_classes, dropout_rate=dropout_rate,
        scan_impl=scan_impl, gamma=gamma, rho_init=rho_init,
        seq2seq=seq2seq, tbptt_window=tbptt_window)
