"""PROSPECTIVE — the Euler-discretized prospective SSM layer (treatment arm).

THIS IS THE CONTRIBUTION. Read ``ssm/baseline_s5/layer.py`` first; this file
mirrors it line for line so the delta is visible.

Derivation (continuous prospective dynamics, Zucchet et al. 2025 form, with
f_theta(s, t) = A s + B x_t):

    tau ds/dt = -s + f_theta(s, t) + tau d/dt f_theta(s, t)

Euler-discretized with rho = Delta t / tau:

    s_t = [(1 - rho) I + (1 + rho) A] s_{t-1} - A s_{t-2}
          + (1 + rho) B x_{t-1} - B x_{t-2}

which is the SECOND-ORDER recurrence the baseline does not have. Written in
companion form with a1 = (1-rho) + (1+rho) A and a2 = -A:

    z_t = [[a1, a2], [1, 0]] z_{t-1} + [x~_t ; 0],    z_t = [s_t ; s_{t-1}]
    x~_t = (1 + rho) B x_{t-1} - B x_{t-2}

A is diagonal (complex, HiPPO init — same as the baseline), so a1, a2 are
diagonal too and this is a per-channel 2x2 recurrence, evaluated with the
prospective associative scan of ``scan.py``. x~_t is a causal 2-tap
conv along time with kernel [(1+rho), -1] on the B-transformed inputs.

What differs from the baseline S5 layer, and nothing else:
    1. second-order recurrence (2x2 companion scan) instead of first-order;
    2. one extra per-channel parameter, log_ratio = log(rho);
    3. the input term x~_t (2-tap causal conv) instead of plain B x_t;
Initialization, C/D, the block, and the classifier are shared verbatim.
There is no Delta t-scaling of A and no clamping: the layer is the
derivation and nothing more, which is why it diverges (exact_failure.py).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from ..shared.params import (
    ComplexParam, hippo_lambda_init, hippo_B_init, C_init,
)
from .scan import select_scan


# ---------------------------------------------------------------------------
# Causal conv1d helper for the prospective input term x~_t
# ---------------------------------------------------------------------------

def causal_conv1d_time(v: jnp.ndarray, kernels: jnp.ndarray) -> jnp.ndarray:
    """Causal per-channel correlation-conv along time (zero left padding).

    Computes out[t, c] = sum_j kernels[c, j] * v[t - (K-1) + j, c],
    i.e. for K = 2:  out_t = w0_c * v_{t-1} + w1_c * v_t.

    Args:
        v: (T, C) input sequence (complex allowed).
        kernels: (C, K) per-channel convolution taps.

    Returns:
        (T, C) output sequence, same length as v.

    NOTE: this is the clean statement of the x~_t input term and is tested
    (test_scan.py [4]), but ``ProspectiveSSM`` below inlines the same math as
    two shifted adds — the grouped lax.conv backward pass hung XLA-CPU
    compilation.
    """
    T, C = v.shape
    K = kernels.shape[1]
    # NWC layout: (batch=1, time=T, channels=C); a grouped conv with
    # feature_group_count=C applies an independent kernel per channel.
    lhs = v[None, :, :]                          # (1, T, C)
    rhs = kernels.T[:, None, :].astype(v.dtype)  # (K, 1, C), match input dtype
    out = jax.lax.conv_general_dilated(
        lhs, rhs,
        window_strides=(1,),
        padding=[(K - 1, 0)],
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=C,
    )                                            # (1, T, C)
    return out[0]


# ---------------------------------------------------------------------------
# Prospective SSM layer
# ---------------------------------------------------------------------------

class ProspectiveSSM(nn.Module):
    """Prospective second-order diagonal SSM layer.

    Implements EXACTLY (see README.md §2):
        s_t = [(1-rho)I + (1+rho)A] s_{t-1} - A s_{t-2}
              + (1+rho) B x_{t-1} - B x_{t-2},      rho = Delta t / tau

    via the stacked 2x2 per-channel recurrence
        z_t = M z_{t-1} + [x~_t; 0],   M = [[a1, a2],[1, 0]],
        a1 = (1-rho) + (1+rho) A,   a2 = -A,
    evaluated with the prospective associative scan (flattened 2x2 kernel —
    EXACTLY the (M, b) composition algebra with elementwise ops instead of
    einsums; see scan.py). x~_t is produced by a causal conv1d with kernel
    [(1+rho), -1] acting on the B-transformed inputs (the kernel taps
    multiply [Bx_{t-1}, Bx_{t-2}]).

    THIS LAYER DOES NOT TRAIN, AND THAT IS THE RESULT. Run verbatim at the
    S5/HiPPO initialization it overflows float32 within ~14 steps. There is
    no stabilized variant here on purpose: the three reasons it fails are
    properties of the derivation and its discretization, not tuning
    problems. ``python exact_failure.py`` measures all three.

    Params: log_ratio = log(Delta t / tau) (per-channel, trainable), Lambda
    (HiPPO init, trainable real/imag), B, C, D. Nothing else — the
    derivation introduces no other free quantity.
    """
    state_size: int = 64     # N
    d_model: int = 96        # H
    # gamma = partial-prospection strength (README "Turning prospection off").
    #   gamma = 1 : the derivation verbatim (DEFAULT)
    #   gamma = 0 : the prospective term is gone and the recurrence collapses
    #               to first order, s_t = [(1-rho)I + rho*A] s_{t-1} + rho*B
    #               x_{t-1} -- explicit-Euler S5, same code path, same params.
    # Roots satisfy mu1*mu2 = gamma*A, so gamma also scales the parasitic root.
    gamma: float = 1.0
    # rho0 = exp(log_ratio_init) = Delta t / tau at initialization.
    log_ratio_init: float = float(np.log(0.1))
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
        # --- shared with the baseline: identical params, identical init ---
        Lambda = ComplexParam((N,), hippo_lambda_init(N), name="Lambda")()
        B = ComplexParam((N, H), hippo_B_init(N, H), name="B")()
        C = ComplexParam((H, N), C_init(N), name="C")()
        D = self.param("D", nn.initializers.normal(1.0), (H,))
        # --- prospective-only: the friction rho = Delta t / tau ---
        log_ratio = self.param(
            "log_ratio",
            lambda key, shape, dtype=jnp.float32: jnp.full(shape, self.log_ratio_init),
            (H,))
        # f_theta(s, t) = A s + B x with A the SSM matrix itself, exactly as
        # derived. No Delta t-scaling of A, no clamps: rho = Delta t / tau is
        # the only free parameter. At the HiPPO init this gives
        # max |mu| = 1433.60 (mu1*mu2 = gamma*A) — it diverges, by
        # construction of the scheme. See exact_failure.py.
        rho = jnp.exp(log_ratio)                                   # (H,)
        A = jnp.broadcast_to(Lambda[None, :], (H, N))              # A = Lambda

        # M = [[a1, a2], [1, 0]] per (channel h, state n); assembled per
        # channel inside run_channel.
        # Partial prospection (gamma=1 is the derivation as written):
        #   s_t = [(1-rho)I + (rho+gamma)A] s_{t-1} - gamma*A s_{t-2}
        #         + (rho+gamma) B x_{t-1} - gamma*B x_{t-2}
        g = self.gamma
        a1 = ((1.0 - rho)[:, None]
              + (rho + g)[:, None] * A)                            # (H, N)
        a2 = -g * A                                                # (H, N)

        # Causal conv kernel [(1+rho), -1] on the B-transformed inputs, i.e.
        #   x~_t = (1+rho) * Bx_{t-1} + (-1) * Bx_{t-2}.
        # causal_conv1d_time(v, [w0, w1]) gives out_t = w0 v_{t-1} + w1 v_t;
        # convolving the one-step-delayed sequence v'_t = Bx_{t-1} with
        # [w0, w1] = [-1, 1+rho] yields exactly x~_t.
        k0 = -g * jnp.ones((H,))                                   # taps on Bx_{t-2}
        k1 = rho + g                                               # taps on Bx_{t-1}
        kernels = jnp.stack([k0, k1], axis=1)                      # (H, 2)

        # B is used UNSCALED (exactly as written in the derivation); only A
        # carries the Delta t-scaling (see class docstring).
        B_eff = B.T                                                # (H, N)
        scan_fn = select_scan(self.scan_impl)

        def run_channel(xs) -> jnp.ndarray:
            # Per-channel scan: all tensors are (T, N) — at the sandbox
            # config (H<=96, T<=784, N<=64) a plain vmap is a few MB, so no
            # chunked lax.map is needed (it also blew up XLA compile time).
            x_h, a1_h, a2_h, Bb_h, k_h, C_h, D_h = xs
            # x_h: (T,) real
            T = x_h.shape[0]
            # B-transformed inputs in the diagonal basis: v[t, n] = Bb_h[n] x_h[t]
            v = x_h[:, None].astype(jnp.complex64) * Bb_h[None, :]      # (T, N)
            # Causal 2-tap "conv" expressed as explicit shifted adds — EXACTLY
            # out_t = k1 * v_{t-1} + k0 * v_{t-2} (identical math to
            # causal_conv1d_time with kernel [k0, k1] on v_delayed, but the
            # grouped lax.conv backward hung XLA-CPU compilation).
            v_d1 = jnp.concatenate(
                [jnp.zeros_like(v[:1]), v[:-1]], axis=0)                # v_{t-1}
            v_d2 = jnp.concatenate(
                [jnp.zeros_like(v[:1]), v_d1[:-1]], axis=0)             # v_{t-2}
            xtilde = k_h[1] * v_d1 + k_h[0] * v_d2                      # (T, N)

            # z_t = [[a1,a2],[1,0]] z_{t-1} + [x~_t; 0] via the
            # companion-structured scan (same composition algebra, see
            # scan.py); s_t = z_t[0].
            a1_seq = jnp.broadcast_to(a1_h[None, :], (T, N))
            a2_seq = jnp.broadcast_to(a2_h[None, :], (T, N))
            s = scan_fn(a1_seq, a2_seq, xtilde)                         # (T, N)

            y = jnp.einsum("n,tn->t", C_h, s).real                      # Re(C s_t)
            return y + D_h * x_h

        def run_batch(x: jnp.ndarray) -> jnp.ndarray:
            # x: (T, H) real -> y: (T, H). Channels are fully vectorized
            # (vmap): 96 x T x N complex64 tensors are ~40 MB, trivial for
            # any real GPU. (The old lax.map channel chunks were a 4 GB
            # CPU-sandbox workaround and serialize the GPU.)
            ys = jax.vmap(run_channel)(
                (x.T, a1, a2, B_eff, kernels, C, D))                    # (H, T)
            return ys.T

        # Batch is still chunked (lax.map) as a memory safety valve; raise
        # batch_size if your GPU has headroom, lower it on OOM.
        return jax.lax.map(jax.checkpoint(run_batch), u,
                           batch_size=16)
