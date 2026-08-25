"""ONLINE S5 layer — baseline S5 forward, streaming custom backward.

Structurally identical to ``ssm/baseline_s5/layer.py`` (same params,
same initializers, same discretization); only the gradient rule differs
(see ``scan.py``). The (Lambda, B, log_step) -> (Lambda_bar, B_bar)
bilinear chain is differentiated by autodiff outside the custom VJP.
Wired into ``ssm.model`` as model_type="online".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..shared.params import (
    ComplexParam, hippo_lambda_init, hippo_B_init, C_init, log_step_init,
)
from ..baseline_s5.layer import discretize_bilinear
from .scan import ssm_online, ssm_online_rot


class OnlineS5SSM(nn.Module):
    """Diagonal S5 layer with the online (streaming) gradient.

    Trainable: log-Delta (per channel), Lambda, B, C, D — exactly like
    the baseline arm. Forward identical; backward is the online rule.

    Route-A orientation: if the caller passes a ``"meta"`` collection with
    per-(channel, mode) arrays ``w_re``/``w_im`` (H, N) at this module's
    scope path, the backward rotates the mode-gradient blocks by conj(w)
    (see ``scan.ssm_online_rot``). Absent the collection, w = 1+0j and the
    layer is bit-identical to the plain online arm — train.py is unaware
    of the meta machinery.
    """
    state_size: int = 64     # N
    d_model: int = 96        # H

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        H, N = self.d_model, self.state_size
        Lambda = ComplexParam((N,), hippo_lambda_init(N), name="Lambda")()
        B = ComplexParam((N, H), hippo_B_init(N, H), name="B")()
        C = ComplexParam((H, N), C_init(N), name="C")()
        D = self.param("D", nn.initializers.normal(1.0), (H,))
        log_step = self.param("log_step", log_step_init, (H,))
        Delta = jnp.exp(log_step)

        w_re = self.scope.get_variable("meta", "w_re", None)
        w_im = self.scope.get_variable("meta", "w_im", None)
        if w_re is None or w_im is None:
            w_re = jnp.ones((H, N), jnp.float32)
            w_im = jnp.zeros((H, N), jnp.float32)

        Lambda_bar, B_bar = discretize_bilinear(Lambda, B, Delta)

        def run_channel(xs):
            x_h, a_h, Bb_h, C_h, D_h, wre_h, wim_h = xs
            return ssm_online_rot(a_h.real, a_h.imag, Bb_h.real, Bb_h.imag,
                                  C_h.real, C_h.imag, D_h, x_h,
                                  wre_h, wim_h)

        def run_batch(x: jnp.ndarray) -> jnp.ndarray:
            # x: (T, H) per sample -> y: (T, H); channels via vmap
            ys = jax.vmap(run_channel)((x.T, Lambda_bar, B_bar, C, D,
                                        w_re, w_im))
            return ys.T

        # batch walked with lax.map (memory valve), same as the baseline
        return jax.lax.map(jax.checkpoint(run_batch), u, batch_size=16)
