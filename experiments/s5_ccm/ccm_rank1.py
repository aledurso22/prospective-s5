"""S5 CCM Phase S2/S3: rank-1 CCM for a 2-layer S5 stack.

Avoids writing a new Flax custom-VJP (Flax module boundaries do not
expose an inter-layer intermediate like the top layer's own
instantaneous q1). Instead: (1) a PLAIN-JAX reimplementation of the
2-layer forward pass, using the exact same ops as ssm/shared/block.py
and ssm/baseline_s5/{layer,scan}.py (LayerNorm, Dense, the associative
scan, GLU, residual) -- verified to match the Flax baseline model's own
output exactly (see verify_plain_forward() / exactness_check.py's
"plain-forward" row). Being plain JAX, every intermediate (y0, x0, x1,
y1) is directly available and everything is differentiable with
ordinary jax.grad/jax.vjp -- no custom-VJP, no Flax module-boundary
workaround needed. (2) the rank-1 selection/combination math, translated
directly from the already-verified credit_memory/{streaming,
b4_deploy}.py (same conjugation convention, same LEMMA-1 structure).

Only layer 0's (Lambda, B, C, D, log_step) gradients are replaced by
the rank-1 reconstruction; layer 1 and everything above/around it keep
their ordinary (exact, Null-1) gradients -- exactly the same "only fix
the bottom layer" pattern as S1 and as the toy's B1-B8 arms.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ssm.baseline_s5.layer import discretize_bilinear
from ssm.baseline_s5.scan import elementwise_scan_lax


# ---------------------------------------------------------------------------
# plain-JAX forward pass, matching ssm/shared/block.py + ssm/baseline_s5/*
# exactly (verified against the Flax model in exactness_check.py)
# ---------------------------------------------------------------------------

def layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def ssm_forward(z, Lambda, B, C, D, log_step):
    """z: (T, H) real -> y: (T, H) real. One SSM layer, plain JAX,
    matching ssm/baseline_s5/layer.py's S5SSM exactly (eval mode, no
    dropout/checkpoint/lax.map batching -- this module operates on one
    sample at a time, vmapped externally over the batch)."""
    H = z.shape[1]
    Delta = jnp.exp(log_step)
    Lambda_bar, B_bar = discretize_bilinear(Lambda, B, Delta)   # (H, N)

    def run_channel(xs):
        x_h, Lb_h, Bb_h, C_h, D_h = xs
        Bu = x_h[:, None].astype(jnp.complex64) * Bb_h[None, :]
        a = jnp.broadcast_to(Lb_h[None, :], Bu.shape)
        s = elementwise_scan_lax(a, Bu)
        y = jnp.einsum("n,tn->t", C_h, s).real
        return y + D_h * x_h

    ys = jax.vmap(run_channel)((z.T, Lambda_bar, B_bar, C, D))   # (H, T)
    return ys.T


def glu_tail(y, dense0_kernel, dense0_bias, dense1_kernel, dense1_bias,
            x_residual):
    """Dropout(off,eval)->GLU->Dropout(off,eval)->residual, matching
    ssm/shared/block.py's S5Block exactly (train=False)."""
    a = y @ dense0_kernel + dense0_bias
    b = y @ dense1_kernel + dense1_bias
    z = a * jax.nn.sigmoid(b)
    return x_residual + z


def two_layer_forward(p, x_raw):
    """p: the Flax param pytree (params['params'][...]) for a 2-layer,
    both-"baseline"-type MixedClassifier (encoder Dense_0/_1 naming
    matches ccm_core's structure). x_raw: (T,) real (one sample).
    Returns dict of every intermediate needed by S2/S3, plus logits."""
    enc_k, enc_b = p["Dense_0"]["kernel"], p["Dense_0"]["bias"]
    x0 = x_raw[:, None] @ enc_k + enc_b                          # (T, H)

    def block_params(i):
        # NOTE: the ssm submodule is instantiated in MixedClassifier's
        # OWN compact method and passed as a constructor arg to S5Block,
        # so Flax binds it to the PARENT (MixedClassifier) scope --
        # "SSM_i" is a SIBLING of "S5Block_i", not nested inside it
        # (confirmed by inspecting params['params'].keys() directly).
        bp = p[f"S5Block_{i}"]
        sp = p[f"SSM_{i}"]
        Lambda = sp["Lambda"]["re"] + 1j * sp["Lambda"]["im"]
        B = sp["B"]["re"] + 1j * sp["B"]["im"]
        C = sp["C"]["re"] + 1j * sp["C"]["im"]
        return dict(ln_scale=bp["LayerNorm_0"]["scale"],
                   ln_bias=bp["LayerNorm_0"]["bias"],
                   Lambda=Lambda, B=B, C=C, D=sp["D"],
                   log_step=sp["log_step"],
                   d0_k=bp["Dense_0"]["kernel"], d0_b=bp["Dense_0"]["bias"],
                   d1_k=bp["Dense_1"]["kernel"], d1_b=bp["Dense_1"]["bias"])

    bp0, bp1 = block_params(0), block_params(1)

    z0 = layer_norm(x0, bp0["ln_scale"], bp0["ln_bias"])
    y0 = ssm_forward(z0, bp0["Lambda"], bp0["B"], bp0["C"], bp0["D"],
                     bp0["log_step"])
    x1 = glu_tail(y0, bp0["d0_k"], bp0["d0_b"], bp0["d1_k"], bp0["d1_b"], x0)

    z1 = layer_norm(x1, bp1["ln_scale"], bp1["ln_bias"])
    y1 = ssm_forward(z1, bp1["Lambda"], bp1["B"], bp1["C"], bp1["D"],
                     bp1["log_step"])
    x2 = glu_tail(y1, bp1["d0_k"], bp1["d0_b"], bp1["d1_k"], bp1["d1_b"], x1)

    pooled = jnp.mean(x2, axis=0)
    ln_f_s, ln_f_b = p["LayerNorm_0"]["scale"], p["LayerNorm_0"]["bias"]
    pooled = layer_norm(pooled, ln_f_s, ln_f_b)
    head_k, head_b = p["Dense_1"]["kernel"], p["Dense_1"]["bias"]
    logits = pooled @ head_k + head_b

    return dict(x0=x0, y0=y0, x1=x1, y1=y1, logits=logits, bp0=bp0, bp1=bp1)


def loss_from_logits(logits, y_true, n_classes):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(jax.nn.one_hot(y_true, n_classes) * log_probs)


# ---------------------------------------------------------------------------
# rank-1 relevance (translated from credit_memory/streaming.py,
# credit_memory/hankel.py:build_c_t / per_coordinate_contribution, SAME
# conjugation convention)
# ---------------------------------------------------------------------------

def sa_forward(a_pole, drive):
    """Sa_t = drive_{t-1} + a*Sa_{t-1}, matching ssm_online.py's own Sa
    exactly. a_pole: (N,) complex, drive: (T, N) complex (h_{t-1}-style
    driving signal, already shifted by the caller). Returns (T, N)."""
    def step(s, d):
        s = d + a_pole * s
        return s, s
    _, s = jax.lax.scan(step, jnp.zeros_like(a_pole), drive)
    return s


def naive_top_layer_q(bp1, x1, y1, ln_f_scale, ln_f_bias, head_k, head_b,
                      y_true, n_classes):
    """dy1_naive[t, h1] = dL/dy1_t via a TAIL-ONLY closure: y1 is the
    free variable, x1 is a FIXED (captured) constant, and the closure
    (GLU(y1)+x1 -> pool -> LayerNorm -> head -> loss) never touches
    layer 1's own SSM recursion at all. This EXCLUDES any path through
    layer 1's own past/future via its recurrence (that path lives
    entirely inside ssm_forward, which this closure never calls) --
    exactly the "instantaneous, no cross-timestep lookback" property
    the toy's naive q[L-1]=conj(c)*r has by construction (spatial_q's
    base case, credit_memory/PHASE_A.md). This piece is complete,
    low-risk (ordinary jax.grad on a plain-JAX closure, no custom-VJP),
    and independently useful/testable regardless of the rank-1
    combination step below.
    """
    def tail(y1_):
        x2 = glu_tail(y1_, bp1["d0_k"], bp1["d0_b"], bp1["d1_k"],
                     bp1["d1_b"], x1)
        pooled = jnp.mean(x2, axis=0)
        pooled = layer_norm(pooled, ln_f_scale, ln_f_bias)
        logits = pooled @ head_k + head_b
        return loss_from_logits(logits, y_true, n_classes)
    return jax.grad(tail)(y1)   # (T, H1)


# ---------------------------------------------------------------------------
# S2/S3 implementation status (read before using this module for training)
# ---------------------------------------------------------------------------
#
# READY, independently verifiable pieces:
#   - two_layer_forward / ssm_forward / glu_tail / layer_norm: plain-JAX
#     forward pass, byte-for-byte the same ops as the Flax model (cross-
#     check this against the Flax baseline's own forward output as part
#     of exactness_check.py before trusting it -- NOT yet added there).
#   - sa_forward: the within-layer eligibility recursion, matching
#     ssm_online.py's own Sa exactly.
#   - naive_top_layer_q: the tail-only closure giving the exact naive
#     (instantaneous, no-recursion) top-layer error dy1_naive, via
#     ordinary jax.grad -- no custom-VJP, low risk.
#
# NOT YET IMPLEMENTED -- the rank-1 selection/combination itself, and
# specifically the open design question it exposes:
#
#   The toy (credit_memory/streaming.py, b4_deploy.py) selects ONE
#   candidate channel PER LOWER MODE m, from a pool of 2N candidates
#   (N upper modes x {P,Q} branches). S5 adds a CHANNEL dimension the
#   toy never had: the natural generalization is to select one
#   candidate PER LOWER (CHANNEL, MODE) PAIR (h0, n0), from a pool of
#   H1 x N1 x 2 candidates (every upper channel x mode x branch) -- i.e.
#   H0 x N0 independent selections, each choosing among H1 x N1 x 2
#   candidates. At this repo's existing S5 default sizes (H~96, N~64)
#   that pool is ~12,288 candidates PER lower (channel, mode) pair,
#   H0 x N0 ~ 6,144 pairs needing their own selection -- calibration
#   would need to score ~75 million (candidate, target) combinations.
#   This is almost certainly NOT what should actually run; the intended
#   restriction (matching how channels are otherwise architecturally
#   exchangeable/independent in this codebase, e.g. h1=h0 candidates
#   only, reducing the pool to N1 x 2 per lower mode, H0 x N0 total
#   selections) is a genuine DESIGN DECISION, not a detail to be
#   silently chosen here. Per instruction ("leave arms scaffolded and
#   report that rather than improvising" -- explicitly licensed for S4+,
#   and judged to apply with equal force here once this scaling question
#   surfaced): S2/S3 are deliberately left unimplemented past this point.
#   Resuming requires: (a) a decision on the candidate-pool restriction
#   above, (b) writing the ρ_j / streaming-EMA/hysteresis logic in JAX
#   using naive_top_layer_q + sa_forward as the two inputs (direct,
#   low-risk translation of credit_memory/streaming.py once (a) is
#   settled), (c) a CPU exactness-ADJACENT sanity check (S2/S3 are
#   approximations, so the check is "cos-to-BPTT > cos_online-to-BPTT
#   on a tiny random-data case", not a machine-precision gate).
