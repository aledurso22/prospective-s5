"""Phase B28 -- faithful RTU-only calibration path, separate from the
small B28 harness (b28_stage2_ac_lambda.py, left UNCHANGED).

Implements the CONFIRMED published/repository ingredients from the
paper-parity audit against Farr, Reddi, D'Eramo, Peters, "Streaming
Reinforcement Learning under Partial Observability with Real-Time
Recurrent Learning" (arXiv:2605.24709v2) and its code,
github.com/noahfarr/memorax:

  - separate actor and critic networks, each its own encoder+RTU+head
    (memorax/algorithms/stream_ac.py: StreamAC holds independent
    actor_network, critic_network);
  - encoder: one-hot(obs) -> Dense(64) -> LayerNorm -> LeakyReLU
    (memorax/examples/ppo_popgym.py's FeatureExtractor pattern; the
    exact stream-ac-POPGym script was not found in the repo, so this
    is confirmed by a *sibling* POPGym example using the same
    FeatureExtractor/Network primitives, not the literal stream-AC
    script -- flagged, not silently assumed identical);
  - RTU cell: EXACT math and EXACT per-step RTRL sensitivity formulas
    taken directly from memorax/networks/sequence_models/rtu.py's
    RTUCell.__call__ and RTUCell.local_jacobian (g=r*cos(theta),
    phi=r*sin(theta), norm=sqrt(1-r^2)+eps input-scaling, tanh
    activation, log-parameterized r/theta init);
  - online observation AND reward normalization via running
    mean/variance (paper: "normalized observations and rewards online
    with running statistics" -- exact statistic form, e.g. Welford
    vs. exponential-moving, not stated in the paper text I could
    access; implemented here as a standard unbiased running
    mean/var, flagged as a reasonable-but-unconfirmed choice);
  - 90% sparse initialization: the `sparse(sparsity=0.9)` initializer
    IS confirmed to exist in the repo (memorax/networks/initializers/
    sparse.py) with default 90%, but its exact USAGE SITE in the
    stream-AC-POPGym config could NOT be confirmed (no
    stream_ac_popgym.py example exists in the repo; generic FFN/
    Projection blocks default to lecun_normal, NOT sparse). Applied
    here to the encoder's Dense(64) kernel as the most likely
    candidate, FLAGGED explicitly as an unconfirmed assumption, not a
    verified fact;
  - gamma=.99, lambda=.8, entropy_coefficient=.095, actor/critic
    alpha=1.0, actor kappa=3.0, critic kappa=2.0 (Table 3, confirmed
    from paper text);
  - ObGD formula: independently re-derived from
    memorax/algorithms/stream_ac.py's _obgd_update, confirmed IDENTICAL
    to the corrected formula already implemented in
    b28_stage2_ac_lambda.py (delta_bar=max(|delta|,1),
    M=alpha*kappa*delta_bar*||z||_1 over the WHOLE actor-or-critic
    trace set combined, step=alpha/max(1,M));
  - softmax policy (Categorical head);
  - frame-based training accounting (not episode-based).

num_envs is left EXPLICITLY CONFIGURABLE and UNRESOLVED: the actual
POPGym stream-AC experimental num_envs value could not be found in the
repo (no stream_ac_popgym.py). Defaults to 1 (single-stream, the
conservative/literal-"streaming" reading) per explicit instruction NOT
to increase it merely for accelerator throughput without confirming it
matches the published regime.

This is a from-scratch reimplementation matching the CONFIRMED math
exactly (verified against the RTUCell.local_jacobian formulas), NOT a
port of memorax's own phantom-injection autodiff mechanism (compute_
phantom/inject_phantom) -- that is a different IMPLEMENTATION technique
for computing the same exact-RTRL gradient; the algorithm/mathematics
match, the code mechanism does not.

Run tests: python -m credit_memory.b28_rtu_faithful
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Faithful RTU cell: params, forward, and EXACT per-step RTRL sensitivity.
# Formulas taken directly from memorax's RTUCell (rtu.py) __call__ and
# local_jacobian. Unbatched (no leading batch/num_envs axis here; the
# outer training loop handles num_envs=1 by construction).
# ---------------------------------------------------------------------------
RTU_FAMILIES = ("nu_log", "theta_log", "B_real", "B_imag")


def rtu_init_nu_log(rng, hidden_dim, r_min=0.0, r_max=1.0):
    u = rng.uniform(size=hidden_dim)
    return np.log(-0.5 * np.log(u * (r_max**2 - r_min**2) + r_min**2))


def rtu_init_theta_log(rng, hidden_dim, max_phase=6.28):
    u = rng.uniform(size=hidden_dim)
    return np.log(max_phase * u)


def lecun_normal(rng, shape):
    fan_in = shape[-1] if len(shape) > 1 else shape[0]
    std = 1.0 / np.sqrt(fan_in)
    return rng.randn(*shape) * std


def make_rtu_params(rng, hidden_dim, features, r_min=0.0, r_max=1.0, max_phase=6.28):
    return dict(
        nu_log=jnp.array(rtu_init_nu_log(rng, hidden_dim, r_min, r_max)),
        theta_log=jnp.array(rtu_init_theta_log(rng, hidden_dim, max_phase)),
        B_real=jnp.array(lecun_normal(rng, (hidden_dim, features))),
        B_imag=jnp.array(lecun_normal(rng, (hidden_dim, features))),
    )


def rtu_g_phi_norm(nu_log, theta_log, eps=1e-8):
    r = jnp.exp(-jnp.exp(nu_log))
    theta = jnp.exp(theta_log)
    g = r * jnp.cos(theta)
    phi = r * jnp.sin(theta)
    norm = jnp.sqrt(1.0 - r**2) + eps
    return g, phi, norm, r


def rtu_forward(params, carry_real, carry_imag, u_t, eps=1e-8):
    """One step. Returns (new_real, new_imag, output, pre_real, pre_imag)."""
    g, phi, norm, r = rtu_g_phi_norm(params["nu_log"], params["theta_log"], eps)
    u_real = params["B_real"] @ u_t
    u_imag = params["B_imag"] @ u_t
    pre_real = g * carry_real - phi * carry_imag + norm * u_real
    pre_imag = g * carry_imag + phi * carry_real + norm * u_imag
    new_real = jnp.tanh(pre_real)
    new_imag = jnp.tanh(pre_imag)
    output = jnp.concatenate([new_real, new_imag])
    return new_real, new_imag, output, pre_real, pre_imag


def rtu_streaming_init(hidden_dim, features):
    """Sensitivity dict, one (2,hidden_dim,...) entry per family --
    matches RTUCell.initialize_sensitivity exactly."""
    return dict(
        nu_log=np.zeros((2, hidden_dim)),
        theta_log=np.zeros((2, hidden_dim)),
        B_real=np.zeros((2, hidden_dim, features)),
        B_imag=np.zeros((2, hidden_dim, features)),
    )


def rtu_streaming_step(params, stream_state, u_t, eps=1e-8):
    """Advances (carry_real, carry_imag) and ALL families' sensitivity
    by one step. stream_state: dict with 'real','imag' (hidden_dim,)
    and 'S' (the per-family sensitivity dict). Mutates stream_state
    in place; returns (output, next_sensitivity) where
    next_sensitivity[family] has shape (2,hidden_dim,...) -- axis 0
    indexes (real,imag)."""
    carry_real, carry_imag = stream_state["real"], stream_state["imag"]
    nu_log, theta_log = params["nu_log"], params["theta_log"]
    g, phi, norm, r = rtu_g_phi_norm(nu_log, theta_log, eps)

    new_real, new_imag, output, pre_real, pre_imag = rtu_forward(
        params, carry_real, carry_imag, u_t, eps
    )
    d_real = 1.0 - np.asarray(new_real) ** 2  # tanh'(pre) = 1 - tanh(pre)^2
    d_imag = 1.0 - np.asarray(new_imag) ** 2

    exp_nu = np.exp(np.asarray(nu_log))
    g_np, phi_np, norm_np, r_np = (np.asarray(g), np.asarray(phi), np.asarray(norm), np.asarray(r))
    dg_dnu = -exp_nu * g_np
    dphi_dnu = -exp_nu * phi_np
    dnorm_dnu = exp_nu * r_np**2 / (np.sqrt(1 - r_np**2) + 1e-12)

    theta = np.exp(np.asarray(theta_log))
    dg_dtheta = -phi_np * theta
    dphi_dtheta = g_np * theta

    u_real = np.asarray(params["B_real"]) @ np.asarray(u_t)
    u_imag = np.asarray(params["B_imag"]) @ np.asarray(u_t)

    S = stream_state["S"]
    Bu = norm_np[:, None] * np.asarray(u_t)[None, :]  # (hidden_dim, features)
    zeros_hF = np.zeros_like(Bu)

    J = dict(
        nu_log=np.stack([
            dg_dnu * np.asarray(carry_real) - dphi_dnu * np.asarray(carry_imag) + dnorm_dnu * u_real,
            dg_dnu * np.asarray(carry_imag) + dphi_dnu * np.asarray(carry_real) + dnorm_dnu * u_imag,
        ], axis=0),
        theta_log=np.stack([
            dg_dtheta * np.asarray(carry_real) - dphi_dtheta * np.asarray(carry_imag),
            dg_dtheta * np.asarray(carry_imag) + dphi_dtheta * np.asarray(carry_real),
        ], axis=0),
        B_real=np.stack([Bu, zeros_hF], axis=0),
        B_imag=np.stack([zeros_hF, Bu], axis=0),
    )

    A = np.array([[g_np, -phi_np], [phi_np, g_np]])  # (2,2,hidden_dim) -- per-unit rotation-scale
    d = np.stack([d_real, d_imag], axis=0)  # (2, hidden_dim)

    next_S = {}
    for fam in RTU_FAMILIES:
        Sf = S[fam]  # (2, hidden_dim, ...)
        # rotated[i,h,...] = sum_j A[i,j,h] * Sf[j,h,...]
        rotated = np.einsum("ijh,jh...->ih...", A, Sf)
        next_S[fam] = d[:, :, *([None] * (Sf.ndim - 2))] * (rotated + J[fam])

    stream_state["real"], stream_state["imag"] = new_real, new_imag
    stream_state["S"] = next_S
    return output, next_S


def rtu_direct_term_for_input(params, u_t, eps=1e-8):
    """d(pre_real)/d(u_t) = norm*B_real, d(pre_imag)/d(u_t) = norm*B_imag
    -- used to chain sensitivity through an upstream encoder whose
    output IS u_t."""
    g, phi, norm, r = rtu_g_phi_norm(params["nu_log"], params["theta_log"], eps)
    norm_np = np.asarray(norm)
    return norm_np[:, None] * np.asarray(params["B_real"]), norm_np[:, None] * np.asarray(params["B_imag"])


# ---------------------------------------------------------------------------
# Faithful encoder: one-hot(obs) -> Dense(width) -> LayerNorm -> LeakyReLU.
# Confirmed pattern from memorax/examples/ppo_popgym.py's FeatureExtractor
# (a SIBLING POPGym example using the same Network/FeatureExtractor
# primitives -- the literal stream-ac-POPGym script was not found in the
# repo, so this encoder shape is a well-corroborated but not
# byte-confirmed match for the exact stream-AC config).
# Sparse (90%) init applied to the Dense kernel -- FLAGGED ASSUMPTION,
# not a confirmed usage site (see module docstring).
# ---------------------------------------------------------------------------
ENCODER_FAMILIES = ("W_enc", "b_enc", "ln_scale", "ln_bias")
LEAKY_RELU_SLOPE = 0.01  # JAX/flax default negative_slope


def sparse_init(rng, shape, sparsity=0.9):
    """Matches memorax/networks/initializers/sparse.py exactly: uniform
    in [-1/sqrt(fan_in), 1/sqrt(fan_in)], then zero out ceil(sparsity*
    fan_in) entries per OUTPUT column via a per-column random
    permutation mask. shape=(fan_in,fan_out)."""
    fan_in, fan_out = shape
    limit = 1.0 / np.sqrt(fan_in)
    weights = rng.uniform(-limit, limit, size=shape)
    n_zero = int(np.ceil(sparsity * fan_in))
    mask = np.zeros(shape)
    for j in range(fan_out):
        perm = rng.permutation(fan_in)
        keep = perm >= n_zero
        mask[:, j] = keep.astype(np.float64)
    return weights * mask


def make_encoder_params(rng, in_dim, width=64, sparsity=0.9):
    return dict(
        W_enc=jnp.array(sparse_init(rng, (in_dim, width), sparsity)),
        b_enc=jnp.zeros(width),
        ln_scale=jnp.ones(width),
        ln_bias=jnp.zeros(width),
    )


def encoder_forward(obs_onehot, enc_params, eps=1e-6):
    """obs_onehot: (in_dim,) -- already one-hot outside this function
    (the ONE-HOT step itself has no trainable parameters)."""
    pre = obs_onehot @ enc_params["W_enc"] + enc_params["b_enc"]
    mean = jnp.mean(pre)
    var = jnp.var(pre)
    normed = (pre - mean) / jnp.sqrt(var + eps)
    ln_out = normed * enc_params["ln_scale"] + enc_params["ln_bias"]
    out = jnp.where(ln_out >= 0, ln_out, LEAKY_RELU_SLOPE * ln_out)
    return out


def encoder_jacobian(obs_onehot, enc_params):
    """d(x_t)/d(theta) for theta in each encoder family, at fixed
    obs_onehot -- obs_onehot has NO trainable params of its own (a
    fixed one-hot of the current observation), so this is the FULL
    sensitivity of the encoder output to its own parameters."""
    jacs = {}
    for fam in ENCODER_FAMILIES:
        def f(p, fam=fam):
            merged = dict(enc_params)
            merged[fam] = p
            return encoder_forward(obs_onehot, merged)
        jacs[fam] = jax.jacobian(f)(enc_params[fam])
    return jacs


# ---------------------------------------------------------------------------
# Combined per-network streaming step: encoder -> RTU -> exact RTRL for
# BOTH encoder and RTU families, chained through the encoder's output
# acting as the RTU's input u_t.
# ---------------------------------------------------------------------------
ALL_NET_FAMILIES = ENCODER_FAMILIES + RTU_FAMILIES


def net_streaming_init(hidden_dim, width, in_dim):
    S = rtu_streaming_init(hidden_dim, width)
    S["W_enc"] = np.zeros((2, hidden_dim, in_dim, width))
    S["b_enc"] = np.zeros((2, hidden_dim, width))
    S["ln_scale"] = np.zeros((2, hidden_dim, width))
    S["ln_bias"] = np.zeros((2, hidden_dim, width))
    return dict(real=np.zeros(hidden_dim), imag=np.zeros(hidden_dim), S=S)


def net_streaming_step(rtu_params, enc_params, stream_state, obs_onehot, eps=1e-8):
    """One step through encoder->RTU, advancing sensitivity for ALL
    8 families (4 encoder + 4 RTU). Returns (x_t, output, next_S)."""
    carry_real, carry_imag = stream_state["real"], stream_state["imag"]
    nu_log, theta_log = rtu_params["nu_log"], rtu_params["theta_log"]
    g, phi, norm, r = rtu_g_phi_norm(nu_log, theta_log, eps)
    g_np, phi_np, norm_np, r_np = (np.asarray(g), np.asarray(phi), np.asarray(norm), np.asarray(r))

    x_t = encoder_forward(obs_onehot, enc_params)
    x_t_np = np.asarray(x_t)

    new_real, new_imag, output, pre_real, pre_imag = rtu_forward(
        rtu_params, carry_real, carry_imag, x_t, eps
    )
    d_real = 1.0 - np.asarray(new_real) ** 2
    d_imag = 1.0 - np.asarray(new_imag) ** 2
    d = np.stack([d_real, d_imag], axis=0)
    A = np.array([[g_np, -phi_np], [phi_np, g_np]])

    # --- RTU-family J terms (identical to rtu_streaming_step) ---
    exp_nu = np.exp(np.asarray(nu_log))
    dg_dnu = -exp_nu * g_np
    dphi_dnu = -exp_nu * phi_np
    dnorm_dnu = exp_nu * r_np**2 / (np.sqrt(1 - r_np**2) + 1e-12)
    theta = np.exp(np.asarray(theta_log))
    dg_dtheta = -phi_np * theta
    dphi_dtheta = g_np * theta
    u_real = np.asarray(rtu_params["B_real"]) @ x_t_np
    u_imag = np.asarray(rtu_params["B_imag"]) @ x_t_np
    Bu = norm_np[:, None] * x_t_np[None, :]
    zeros_hF = np.zeros_like(Bu)

    J = dict(
        nu_log=np.stack([
            dg_dnu * np.asarray(carry_real) - dphi_dnu * np.asarray(carry_imag) + dnorm_dnu * u_real,
            dg_dnu * np.asarray(carry_imag) + dphi_dnu * np.asarray(carry_real) + dnorm_dnu * u_imag,
        ], axis=0),
        theta_log=np.stack([
            dg_dtheta * np.asarray(carry_real) - dphi_dtheta * np.asarray(carry_imag),
            dg_dtheta * np.asarray(carry_imag) + dphi_dtheta * np.asarray(carry_real),
        ], axis=0),
        B_real=np.stack([Bu, zeros_hF], axis=0),
        B_imag=np.stack([zeros_hF, Bu], axis=0),
    )

    # --- encoder-family J terms: chain d(x_t)/d(enc_theta) through the
    # SAME norm*B_real / norm*B_imag input-coupling used for B_real/B_imag
    # above, since x_t enters the RTU exactly like an input vector. ---
    enc_jacs = encoder_jacobian(obs_onehot, enc_params)
    B_real_np, B_imag_np = np.asarray(rtu_params["B_real"]), np.asarray(rtu_params["B_imag"])
    for fam in ENCODER_FAMILIES:
        Jx = np.asarray(enc_jacs[fam])  # (features, *enc_param_shape)
        # (hidden_dim, *enc_param_shape) via tensordot over the features axis
        real_term = np.tensordot(B_real_np, Jx, axes=([1], [0])) * norm_np.reshape((-1,) + (1,) * (Jx.ndim - 1))
        imag_term = np.tensordot(B_imag_np, Jx, axes=([1], [0])) * norm_np.reshape((-1,) + (1,) * (Jx.ndim - 1))
        J[fam] = np.stack([real_term, imag_term], axis=0)

    S = stream_state["S"]
    next_S = {}
    for fam in ALL_NET_FAMILIES:
        Sf = S[fam]
        rotated = np.einsum("ijh,jh...->ih...", A, Sf)
        next_S[fam] = d[(slice(None), slice(None)) + (None,) * (Sf.ndim - 2)] * (rotated + J[fam])

    stream_state["real"], stream_state["imag"] = new_real, new_imag
    stream_state["S"] = next_S
    return x_t, output, next_S


# ---------------------------------------------------------------------------
# TEST 1: one-step RTU recurrence + exact RTRL sensitivity, verified
# against full BPTT (independent jax.grad reference) over a short
# synthetic sequence, for every family.
# ---------------------------------------------------------------------------
def rtu_rollout_bptt(params, U_seq, hidden_dim):
    def step(carry, u_t):
        real, imag = carry
        g, phi, norm, r = rtu_g_phi_norm(params["nu_log"], params["theta_log"])
        u_real = params["B_real"] @ u_t
        u_imag = params["B_imag"] @ u_t
        pre_real = g * real - phi * imag + norm * u_real
        pre_imag = g * imag + phi * real + norm * u_imag
        new_real, new_imag = jnp.tanh(pre_real), jnp.tanh(pre_imag)
        out = jnp.concatenate([new_real, new_imag])
        return (new_real, new_imag), out

    carry0 = (jnp.zeros(hidden_dim), jnp.zeros(hidden_dim))
    _, outputs = jax.lax.scan(step, carry0, U_seq)
    return outputs  # (T, 2*hidden_dim)


def test_rtu_streaming_vs_bptt(seed=0, T_=9, hidden_dim=5, features=3):
    rng = np.random.RandomState(seed)
    params = make_rtu_params(rng, hidden_dim, features)
    U_seq = jnp.array(rng.randn(T_, features) * 0.5)
    dLdout_seq = jnp.array(rng.randn(T_, 2 * hidden_dim) * 0.3)

    def loss_of(p):
        outs = rtu_rollout_bptt(p, U_seq, hidden_dim)
        return jnp.sum(outs * dLdout_seq)

    bptt_grads = jax.grad(loss_of)(params)

    stream_state = dict(real=np.zeros(hidden_dim), imag=np.zeros(hidden_dim),
                         S=rtu_streaming_init(hidden_dim, features))
    grad_accum = {fam: np.zeros_like(np.asarray(params[fam])) for fam in RTU_FAMILIES}
    for t in range(T_):
        output, next_S = rtu_streaming_step(params, stream_state, U_seq[t])
        dLdout_t = np.asarray(dLdout_seq[t])
        dLdreal, dLdimag = dLdout_t[:hidden_dim], dLdout_t[hidden_dim:]
        dLdout_split = np.stack([dLdreal, dLdimag], axis=0)  # (2,hidden_dim)
        for fam in RTU_FAMILIES:
            # next_S[fam]: (2,hidden_dim,...) -- sum ONLY over the real/imag
            # axis i; nu_log/theta_log/B_real/B_imag are all per-unit
            # (diagonal) parameters, so the hidden-unit axis h (and any
            # trailing param-shape axes) must be PRESERVED, not summed.
            grad_accum[fam] += np.einsum("ih,ih...->h...", dLdout_split, next_S[fam])

    errs = {}
    for fam in RTU_FAMILIES:
        errs[fam] = float(np.max(np.abs(grad_accum[fam] - np.asarray(bptt_grads[fam]))))
    return errs


# ---------------------------------------------------------------------------
# TEST 2: combined encoder+RTU streaming exact-RTRL (all 8 families,
# including the 4 encoder families chained through the RTU input) vs
# full BPTT.
# ---------------------------------------------------------------------------
def net_rollout_bptt(rtu_params, enc_params, obs_seq, hidden_dim):
    def step(carry, obs_onehot):
        real, imag = carry
        x_t = encoder_forward(obs_onehot, enc_params)
        g, phi, norm, r = rtu_g_phi_norm(rtu_params["nu_log"], rtu_params["theta_log"])
        u_real = rtu_params["B_real"] @ x_t
        u_imag = rtu_params["B_imag"] @ x_t
        pre_real = g * real - phi * imag + norm * u_real
        pre_imag = g * imag + phi * real + norm * u_imag
        new_real, new_imag = jnp.tanh(pre_real), jnp.tanh(pre_imag)
        out = jnp.concatenate([new_real, new_imag])
        return (new_real, new_imag), out

    carry0 = (jnp.zeros(hidden_dim), jnp.zeros(hidden_dim))
    _, outputs = jax.lax.scan(step, carry0, obs_seq)
    return outputs


def test_net_streaming_vs_bptt(seed=1, T_=7, hidden_dim=4, width=6, in_dim=3):
    rng = np.random.RandomState(seed)
    rtu_params = make_rtu_params(rng, hidden_dim, width)
    enc_params = make_encoder_params(rng, in_dim, width)
    obs_seq = jnp.array((rng.rand(T_, in_dim) < 0.4).astype(np.float64))  # sparse-ish one-hot-like
    dLdout_seq = jnp.array(rng.randn(T_, 2 * hidden_dim) * 0.3)

    def loss_of(params_dict):
        rp = {k: params_dict[k] for k in RTU_FAMILIES}
        ep = {k: params_dict[k] for k in ENCODER_FAMILIES}
        outs = net_rollout_bptt(rp, ep, obs_seq, hidden_dim)
        return jnp.sum(outs * dLdout_seq)

    all_params = {**rtu_params, **enc_params}
    bptt_grads = jax.grad(loss_of)(all_params)

    stream_state = net_streaming_init(hidden_dim, width, in_dim)
    grad_accum = {fam: np.zeros_like(np.asarray(all_params[fam])) for fam in ALL_NET_FAMILIES}
    for t in range(T_):
        x_t, output, next_S = net_streaming_step(rtu_params, enc_params, stream_state, obs_seq[t])
        dLdout_t = np.asarray(dLdout_seq[t])
        dLdreal, dLdimag = dLdout_t[:hidden_dim], dLdout_t[hidden_dim:]
        dLdout_split = np.stack([dLdreal, dLdimag], axis=0)
        for fam in RTU_FAMILIES:
            # per-unit (diagonal) parameters -- keep the hidden-unit axis
            grad_accum[fam] += np.einsum("ih,ih...->h...", dLdout_split, next_S[fam])
        for fam in ENCODER_FAMILIES:
            # SHARED across all hidden units (same x_t feeds every unit)
            # -- sum over the hidden-unit axis too
            grad_accum[fam] += np.einsum("ih,ih...->...", dLdout_split, next_S[fam])

    errs = {}
    for fam in ALL_NET_FAMILIES:
        errs[fam] = float(np.max(np.abs(grad_accum[fam] - np.asarray(bptt_grads[fam]))))
    return errs


def main():
    print("=" * 70)
    print("TEST 1: faithful RTU streaming exact-RTRL vs BPTT")
    print("=" * 70)
    errs = test_rtu_streaming_vs_bptt()
    for fam, e in errs.items():
        print(f"  family={fam:10s} |err|={e:.2e}")
    assert all(e < 1e-8 for e in errs.values()), "REGRESSION: faithful RTU streaming RTRL"
    print("  PASS")

    print("=" * 70)
    print("TEST 2: combined encoder+RTU streaming exact-RTRL vs BPTT (8 families)")
    print("=" * 70)
    errs2 = test_net_streaming_vs_bptt()
    for fam, e in errs2.items():
        print(f"  family={fam:10s} |err|={e:.2e}")
    assert all(e < 1e-6 for e in errs2.values()), "REGRESSION: encoder+RTU streaming RTRL"
    print("  PASS")


if __name__ == "__main__":
    main()
