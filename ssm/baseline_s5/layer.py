"""BASELINE S5 — the reference diagonal state-space layer (control arm).

Continuous diagonal SSM  ds/dt = Lambda s + B~ x,  y = Re(C s) + D x,
discretized with the bilinear (Tustin) rule and run as a first-order
recurrence:

    s_t = Lambda_bar (*) s_{t-1} + B_bar x_t          (elementwise, complex)
    y_t = Re(C s_t) + D x_t

Trainable: log-Delta (per channel), Lambda (real/imag, HiPPO init), B, C, D.

This is the arm the prospective layer is measured against. Read this file
first, then ``ssm/prospective/layer.py`` — they are deliberately written with
the same structure, so the diff between them IS the contribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..shared.params import (
    ComplexParam, hippo_lambda_init, hippo_B_init, C_init, log_step_init,
)
from .scan import select_scan


def discretize_bilinear(Lambda: jnp.ndarray, B_tilde: jnp.ndarray,
                        Delta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bilinear (Tustin) discretization of the diagonal continuous SSM.

    Args:
        Lambda: (N,) complex continuous eigenvalues.
        B_tilde: (N, H) complex input matrix in the diagonal basis.
        Delta: (H,) real step sizes (one per channel).

    Returns:
        Lambda_bar: (H, N) complex discretized eigenvalues.
        B_bar: (H, N) complex discretized input matrix.
    """
    Delta = Delta[:, None]                    # (H, 1)
    Lambda = Lambda[None, :]                  # (1, N)
    BL = 1.0 / (1.0 - 0.5 * Delta * Lambda)   # (H, N)
    Lambda_bar = BL * (1.0 + 0.5 * Delta * Lambda)
    B_bar = (BL * Delta) * B_tilde.T          # (H, N)
    return Lambda_bar, B_bar


class S5SSM(nn.Module):
    """Baseline S5 diagonal SSM layer (first-order recurrence).

    s_t = Lambda_bar (*) s_{t-1} + B_bar x_t,   y_t = Re(C s_t) + D x_t

    Execution detail (math is unaffected): the (T, H, N) complex scan tensors
    are large, so the batch is walked in chunks with ``jax.lax.map`` and the
    per-sample function is rematerialized (``jax.checkpoint``) so backward
    does not store per-sample scan intermediates for every batch element.
    Channels are fully vectorized with ``vmap``.
    """
    state_size: int = 64     # N
    d_model: int = 96        # H
    scan_impl: str = "assoc"  # "assoc" (associative_scan) | "lax" (lax.scan)

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u: (batch, T, H) real input sequence.
        Returns:
            y: (batch, T, H) real output sequence.
        """
        H, N = self.d_model, self.state_size
        Lambda = ComplexParam((N,), hippo_lambda_init(N), name="Lambda")()
        B = ComplexParam((N, H), hippo_B_init(N, H), name="B")()
        C = ComplexParam((H, N), C_init(N), name="C")()
        D = self.param("D", nn.initializers.normal(1.0), (H,))
        log_step = self.param("log_step", log_step_init, (H,))
        Delta = jnp.exp(log_step)

        Lambda_bar, B_bar = discretize_bilinear(Lambda, B, Delta)  # (H, N)
        scan_fn = select_scan(self.scan_impl)

        def run_channel(xs) -> jnp.ndarray:
            # Per-channel scan: all tensors are (T, N) — small buffers keep
            # XLA compile/runtime memory low (4GB sandbox).
            x_h, Lb_h, Bb_h, C_h, D_h = xs
            # x_h: (T,) real -> states (T, N) complex
            Bu = x_h[:, None].astype(jnp.complex64) * Bb_h[None, :]   # (T, N)
            a = jnp.broadcast_to(Lb_h[None, :], Bu.shape)
            s = scan_fn(a, Bu)                                        # (T, N)
            y = jnp.einsum("n,tn->t", C_h, s).real                    # Re(C s_t)
            return y + D_h * x_h

        def run_batch(x: jnp.ndarray) -> jnp.ndarray:
            # x: (T, H) real -> y: (T, H). Channels are fully vectorized
            # (vmap): 96 x T x N complex64 tensors are ~40 MB, trivial for
            # any real GPU. (The old lax.map channel chunks were a 4 GB
            # CPU-sandbox workaround and serialize the GPU.)
            ys = jax.vmap(run_channel)(
                (x.T, Lambda_bar, B_bar, C, D))                       # (H, T)
            return ys.T

        # Batch is still chunked (lax.map) as a memory safety valve; raise
        # batch_size if your GPU has headroom, lower it on OOM.
        return jax.lax.map(jax.checkpoint(run_batch), u,
                           batch_size=16)
