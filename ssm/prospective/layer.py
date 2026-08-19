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
    4. A := Delta * Lambda with clamped ranges — the stability fallback
       documented in ProspectiveSSM's docstring and in README.md;
    5. a narrower log-Delta init range (``log_step_init_prospective``).
Initialization, C/D, the block, and the classifier are shared verbatim.
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
# Prospective-only initializer (the baseline's log_step_init is too wide here)
# ---------------------------------------------------------------------------

def log_step_init_prospective(key, shape, dtype=jnp.float32):
    """Prospective log-Delta init: U[log(1e-4), log(5e-4)] (stability range,
    see ProspectiveSSM docstring)."""
    return jax.random.uniform(
        key, shape, dtype=jnp.float32,
        minval=np.log(1e-4), maxval=np.log(5e-4))


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

    TWO MODES. ``exact=True`` (the default) runs the derivation verbatim:
    A = Lambda, B unscaled, no clamps. It DIVERGES at the HiPPO init, and
    that divergence is the scientific result (see ``exact_failure.py``).
    ``exact=False`` enables the fallback described next, which trains but
    flattens the spectrum.

    DOCUMENTED STABILITY FALLBACK, opt-in via exact=False (see README.md):
    the prospective Euler
    recurrence is a second-order difference equation; the eigenvalues of
    M_i = [[a1_i, a2_i], [1, 0]] satisfy mu1 * mu2 = -a2_i = A_i, so the
    recurrence has a parasitic mode with |mu| ~ |A_i|. With the (S5/DPLR)
    HiPPO eigenvalues (|Lambda| up to ~1.3e3 at N=64) the update is unstable
    for ANY rho unless A carries a Delta t scaling. We therefore take
        A := Delta * Lambda
    with a trainable log-Delta exactly like S5 (this is the consistent
    discrete-time reading of the derivation's f_theta), while B is used
    UNSCALED exactly as written in the derivation. The trainable ranges are
    clamped (jnp.clip on the log-params, transparent to the optimizer) to
        Delta in [1e-5, 5e-4],   rho in [1e-3, 0.25].
    Apart from the A-scaling and the clamps, the update is implemented
    EXACTLY as specified.

    MEMORY-HORIZON NOTE (empirically important): at init the physical
    companion eigenvalue is ~ (1 - rho0) — measured 0.900847 at rho0 = 0.1
    (test_scan.py test [3]); the ghost eigenvalue stays |mu_ghost| ~ |A| /
    (1 - rho) << 1, damped as designed. The friction rho therefore sets the
    memory horizon ~ 1/rho tokens. rho0 = 0.1 (horizon ~10) cripples
    long-sequence tasks such as sMNIST (L = 784): use log_ratio_init =
    log(1e-3) (horizon ~1000) there — exposed as --rho-init in train.py.

    Params: log_ratio = log(Delta t / tau) (per-channel, trainable), Lambda
    (HiPPO init, trainable real/imag), B, C, D — plus log_step = log Delta
    ONLY when exact=False (it has no counterpart in the derivation).
    """
    state_size: int = 64     # N
    d_model: int = 96        # H
    # exact=True (DEFAULT): the derivation verbatim — A = Lambda, B unscaled,
    # no clamps, the only free parameter being rho = Delta t / tau. This is
    # the arm that exhibits the failure (max |mu| = 1433.60 at the HiPPO init:
    # it overflows within a few steps). exact=False enables the documented
    # stability fallback (A := Delta*Lambda + the clamps) that makes the layer
    # trainable at the cost of flattening the spectrum — see the class
    # docstring and README "Stability note".
    exact: bool = True
    # rho0 = exp(log_ratio_init); see README for the stability discussion.
    log_ratio_init: float = float(np.log(0.1))
    # Stability-fallback clamps — used only when exact=False.
    clip_log_step: tuple = (float(np.log(1e-5)), float(np.log(5e-4)))
    clip_log_ratio: tuple = (float(np.log(1e-3)), float(np.log(0.25)))
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
        if self.exact:
            # ---- THE DERIVATION VERBATIM ----------------------------------
            # f_theta(s, t) = A s + B x with A the SSM matrix itself. No
            # Delta t-scaling of A, no clamps: rho = Delta t / tau is the only
            # free parameter, exactly as written. At the HiPPO init this gives
            # max |mu| = 1433.60 (mu1*mu2 = -a2 = A), i.e. it diverges — that
            # is the result, not a bug.
            rho = jnp.exp(log_ratio)                               # (H,)
            A = jnp.broadcast_to(Lambda[None, :], (H, N))          # A = Lambda
        else:
            # ---- documented stability fallback (opt-in) --------------------
            log_step = self.param("log_step", log_step_init_prospective, (H,))
            log_ratio = jnp.clip(log_ratio, *self.clip_log_ratio)
            log_step = jnp.clip(log_step, *self.clip_log_step)
            rho = jnp.exp(log_ratio)                               # (H,)
            Delta = jnp.exp(log_step)                              # (H,)
            # A := Delta * Lambda (Delta t-scaled diagonal operator), (H, N)
            A = Delta[:, None] * Lambda[None, :]

        # M = [[a1, a2], [1, 0]] per (channel h, state n); assembled per
        # channel inside run_channel.
        a1 = ((1.0 - rho)[:, None]
              + (1.0 + rho)[:, None] * A)                          # (H, N)
        a2 = -A                                                    # (H, N)

        # Causal conv kernel [(1+rho), -1] on the B-transformed inputs, i.e.
        #   x~_t = (1+rho) * Bx_{t-1} + (-1) * Bx_{t-2}.
        # causal_conv1d_time(v, [w0, w1]) gives out_t = w0 v_{t-1} + w1 v_t;
        # convolving the one-step-delayed sequence v'_t = Bx_{t-1} with
        # [w0, w1] = [-1, 1+rho] yields exactly x~_t.
        k0 = -jnp.ones((H,))                                       # taps on Bx_{t-2}
        k1 = 1.0 + rho                                             # taps on Bx_{t-1}
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
