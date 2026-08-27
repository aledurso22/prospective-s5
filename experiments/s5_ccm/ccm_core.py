"""S5 CCM Phase S1 core: the mixed-model_type construction (S1, full
exact causal CCM) plus the graph-splitting machinery S2/S3 (rank-1) build
on.

Design note (see README "Design note: how S1/S2/S3 are built" for the
full derivation):

S1 (full exact causal CCM) needs NO new custom-VJP code. A 2-layer stack
with the TOP layer using ssm.baseline_s5 (ordinary autodiff, exact BPTT)
and the BOTTOM layer using ssm.online_s5 (the existing Sa/Sb custom-VJP,
credit_memory/PHASE_A.md's LEMMA 1) gives, under plain jax.grad, EXACT
BPTT gradients for the bottom layer's OWN SSM recurrence parameters
(Lambda, B, C, D, log_step) -- verified in exactness_check.py to
~1e-7 (float32). This is because the online custom-VJP's formula
(Ga=sum_t conj(q_t) Sa_t) is LEMMA-1-correct for WHATEVER cotangent it
receives; mixing in a baseline top layer makes that cotangent exact
instead of the defective (instantaneous-only) one two "online" layers
would produce for each other.

S2/S3 (rank-1) need access to the top layer's own INSTANTANEOUS q1 (not
the exact adjoint) -- exactly what the toy's B3/B4 rank-1 mechanism
uses. Rather than writing a new joint custom-VJP to expose q1 across the
Flax module boundary, this module SPLITS the forward computation at the
layer-0/layer-1 boundary and uses two small, ordinary jax.vjp calls:

  1. dL/dx1 (exact cotangent on layer-0's block OUTPUT / layer-1's block
     INPUT) via jax.vjp of the "suffix" (layer 1 onwards) -- ordinary
     autodiff, exact, no custom-VJP involved.
  2. dy0 (exact cotangent on layer-0's SSM's OWN raw output, i.e. dL/dy0)
     via a SMALL local jax.vjp of layer 0's own GLU/Dropout/residual
     tail, using dL/dx1 from step 1.
  3. dy1 (layer 1's OWN cotangent, from the ordinary autodiff already
     computed in step 1's vjp) gives q1_t[h1,n1] = conj(C1[h1,n1]) *
     dy1_t[h1] -- the exact analogue of the toy's naive/instantaneous
     top-layer error, NOT the exact adjoint (matching B3/B4's rank-1
     mechanism, not S1's exact one).

All three of these are ordinary JAX operations (no custom_vjp writing);
the rank-1 SELECTION/tracking math (credit_memory/streaming.py,
credit_memory/b4_deploy.py) is then applied directly in plain JAX,
translated faithfully from the already-verified numpy implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from ssm.shared.block import S5Block
from ssm.baseline_s5.layer import S5SSM, discretize_bilinear
from ssm.online_s5.layer import OnlineS5SSM
from ssm.shared.params import (ComplexParam, hippo_lambda_init,
                               hippo_B_init, C_init, log_step_init)


def build_mixed_classifier(model_types, d_model, state_size, n_classes,
                           dropout_rate=0.0, scan_impl="lax",
                           seq2seq=False):
    """A SequenceClassifier variant with a DIFFERENT model_type per layer
    (ssm.model.SequenceClassifier only supports one uniform type).
    model_types[l] in {"baseline", "online"}. Used directly for S0
    (all "online"), S1 (bottom "online", top "baseline"), and Sref (all
    "baseline"); S2/S3 additionally split the graph around this same
    structure (see ccm_rank1.py)."""

    class MixedClassifier(nn.Module):
        @nn.compact
        def __call__(self, x, train=False):
            if x.ndim == 2:
                x = x[..., None]
            x = nn.Dense(d_model)(x)
            for i, lt in enumerate(model_types):
                if lt == "baseline":
                    ssm = S5SSM(state_size=state_size, d_model=d_model,
                               scan_impl=scan_impl, name=f"SSM_{i}")
                elif lt == "online":
                    ssm = OnlineS5SSM(state_size=state_size,
                                      d_model=d_model, name=f"SSM_{i}")
                else:
                    raise ValueError(lt)
                x = S5Block(ssm=ssm, d_model=d_model,
                           dropout_rate=dropout_rate,
                           name=f"S5Block_{i}")(x, train=train)
            if seq2seq:
                x = nn.LayerNorm()(x)
                return nn.Dense(n_classes)(x)
            x = jnp.mean(x, axis=1)
            x = nn.LayerNorm()(x)
            return nn.Dense(n_classes)(x)

    return MixedClassifier()


def s5_channel_poles(params, layer_idx, d_model, state_size):
    """Discretized (H, N) complex poles Lambda_bar for one SSM layer,
    read directly from its own params (Lambda, log_step) -- architecture
    only, no data. Used by the rank-1 selector's candidate pole set.

    NOTE: the ssm submodule is instantiated in MixedClassifier's own
    compact method and passed as a constructor arg to S5Block, so Flax
    binds it to the PARENT scope -- "SSM_i" is a SIBLING of "S5Block_i"
    in params['params'], not nested inside it."""
    p = params["params"][f"SSM_{layer_idx}"]
    Lambda = p["Lambda"]["re"] + 1j * p["Lambda"]["im"]
    Delta = jnp.exp(p["log_step"])
    B = p["B"]["re"] + 1j * p["B"]["im"]
    Lambda_bar, _ = discretize_bilinear(Lambda, B, Delta)   # (H, N)
    return Lambda_bar
