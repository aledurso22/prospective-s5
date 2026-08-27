"""S5 CCM Phase S1 -- mandatory exactness gate.

Verifies, on a tiny deterministic CPU config, that S1 (full exact causal
CCM) reproduces the BPTT gradient to numerical precision, for every
parameter block it claims to reproduce, BEFORE any cluster training is
authorized.

Construction (see experiments/s5_ccm/README.md "Design note: how S1 is
built" for the full derivation): a 2-layer S5 stack where the TOP layer
uses the ordinary/baseline SSM (ssm.baseline_s5 -- plain jax.grad gives
exact BPTT through it, no custom VJP) and the BOTTOM layer uses the
existing ONLINE SSM (ssm.online_s5 -- credit_memory/PHASE_A.md's LEMMA 1
custom-VJP, `Ga=sum_t conj(q_t) Sa_t`). Ordinary `jax.grad` over this
MIXED stack:
  - gives the top layer's OWN gradient exactly (no approximation there,
    it's plain autodiff);
  - backprops the EXACT dL/dy_top-layer-input cotangent down through the
    LayerNorm/GLU/Dropout/residual coupling (also plain autodiff, exact);
  - feeds the bottom layer's online custom-VJP this EXACT cotangent
    (instead of the defective one it would receive if the top layer were
    ALSO "online") -- and the online custom-VJP's own formula
    (Ga=sum_t conj(q_t) Sa_t) is LEMMA-1-correct for *any* input
    cotangent, so it now reproduces the bottom layer's exact BPTT
    gradient too, computed via the SAME cheap Sa/Sb forward machinery
    the online rule already uses -- no reverse-time pass for the bottom
    layer at all.

This claim is the entire content of this file's check. If it fails,
S1 must not be trusted and the cluster run must not proceed.

Run (CPU, tiny config, seconds):
  python -m experiments.s5_ccm.exactness_check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

# NOTE: intentionally NOT enabling jax_enable_x64 -- ssm/shared/params.py's
# ComplexParam/hippo initializers hardcode complex64/float32, matching the
# existing S5 test suite's own convention (tests/test_online_s5_jax.py:
# "JAX online S5 gradient vs the numpy rig (~1e-7)"). The exactness gate
# below is set to this repo's established float32 bar, not full float64
# machine precision (which Phase A/B1-B8's pure-numpy toy checks use).

from ssm.model import SequenceClassifier


def build_mixed_model(model_types, **kwargs):
    """A 2-layer SequenceClassifier variant with a DIFFERENT model_type
    per layer (ssm.model.SequenceClassifier only supports one uniform
    type; this is the minimal extension needed for S1, kept local to
    this experiment folder rather than editing ssm/model.py)."""
    from ssm.shared.block import S5Block
    from ssm.baseline_s5.layer import S5SSM
    from ssm.online_s5.layer import OnlineS5SSM
    from flax import linen as nn

    class MixedClassifier(nn.Module):
        d_model: int
        state_size: int
        n_classes: int
        dropout_rate: float
        scan_impl: str
        seq2seq: bool

        @nn.compact
        def __call__(self, x, train=False):
            if x.ndim == 2:
                x = x[..., None]
            x = nn.Dense(self.d_model)(x)
            for i, lt in enumerate(model_types):
                # explicit, TYPE-INDEPENDENT names so params initialized
                # under one model_type's tree apply directly to another
                # (S5SSM vs OnlineS5SSM otherwise get different default
                # Flax scope names, e.g. "S5SSM_0" vs "OnlineS5SSM_0")
                if lt == "baseline":
                    ssm = S5SSM(state_size=self.state_size,
                               d_model=self.d_model,
                               scan_impl=self.scan_impl, name=f"SSM_{i}")
                elif lt == "online":
                    ssm = OnlineS5SSM(state_size=self.state_size,
                                      d_model=self.d_model, name=f"SSM_{i}")
                else:
                    raise ValueError(lt)
                x = S5Block(ssm=ssm, d_model=self.d_model,
                           dropout_rate=self.dropout_rate,
                           name=f"S5Block_{i}")(x, train=train)
            if self.seq2seq:
                x = nn.LayerNorm()(x)
                return nn.Dense(self.n_classes)(x)
            x = jnp.mean(x, axis=1)
            x = nn.LayerNorm()(x)
            return nn.Dense(self.n_classes)(x)

    return MixedClassifier(**kwargs)


def loss_fn(params, model, x, y):
    logits = model.apply(params, x, train=False)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(jax.nn.one_hot(y, logits.shape[-1])
                             * log_probs, axis=-1))


def block_leaf_paths(params):
    """Flatten the param pytree into {path_string: array} for per-block
    reporting (path includes the layer index, e.g.
    'params/S5Block_0/OnlineS5SSM_0/Lambda/re')."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    out = {}
    for path, leaf in flat:
        key = "/".join(str(p.key if hasattr(p, "key") else p.idx)
                       for p in path)
        out[key] = leaf
    return out


def cos_relerr(g1_flat, g2_flat):
    v1 = jnp.concatenate([jnp.ravel(v) for v in g1_flat])
    v2 = jnp.concatenate([jnp.ravel(v) for v in g2_flat])
    cos = jnp.abs(jnp.vdot(v1, v2)) / (jnp.linalg.norm(v1)
                                       * jnp.linalg.norm(v2) + 1e-300)
    rel = jnp.linalg.norm(v1 - v2) / (jnp.linalg.norm(v2) + 1e-300)
    norm_ratio = jnp.linalg.norm(v1) / (jnp.linalg.norm(v2) + 1e-300)
    return float(cos), float(rel), float(norm_ratio)


def check_plain_forward_matches_flax(d_model, state_size, n_classes,
                                     seq_len, batch, seed):
    """S2/S3's plain-JAX forward reimplementation (ccm_rank1.py) must
    match the Flax baseline model's own forward pass exactly before any
    S2/S3 code built on top of it is trusted -- checked here since
    ccm_rank1.py documents this as a required, not-yet-automated check."""
    from experiments.s5_ccm.ccm_rank1 import two_layer_forward
    key = jax.random.PRNGKey(seed + 999)
    kx, kinit = jax.random.split(key)
    x = jax.random.normal(kx, (batch, seq_len), jnp.float32)
    model = build_mixed_model(["baseline", "baseline"], d_model=d_model,
                              state_size=state_size, n_classes=n_classes,
                              dropout_rate=0.0, scan_impl="lax",
                              seq2seq=False)
    params = model.init(kinit, x, train=False)
    logits_flax = model.apply(params, x, train=False)
    p = params["params"]
    logits_plain = jnp.stack([two_layer_forward(p, x[b])["logits"]
                              for b in range(batch)])
    rel = float(jnp.linalg.norm(logits_flax - logits_plain)
               / (jnp.linalg.norm(logits_flax) + 1e-30))
    return rel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--state-size", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    print("=" * 78)
    print("S5 CCM Phase S1: mandatory exactness gate (tiny CPU config)")
    print(f"d_model={args.d_model} state_size={args.state_size} "
         f"T={args.seq_len} batch={args.batch}")
    print("=" * 78)

    key = jax.random.PRNGKey(args.seed)
    kx, ky, kinit = jax.random.split(key, 3)
    x = jax.random.normal(kx, (args.batch, args.seq_len), jnp.float32)
    y = jax.random.randint(ky, (args.batch,), 0, args.n_classes)

    common = dict(d_model=args.d_model, state_size=args.state_size,
                 n_classes=args.n_classes, dropout_rate=0.0,
                 scan_impl="lax", seq2seq=False)

    # reference: BPTT (both layers baseline, ordinary jax.grad)
    model_bptt = build_mixed_model(["baseline", "baseline"], **common)
    params = model_bptt.init(kinit, x, train=False)
    g_bptt = jax.grad(loss_fn)(params, model_bptt, x, y)

    # S0 online: both layers online (existing, unmodified)
    model_online = build_mixed_model(["online", "online"], **common)
    g_online = jax.grad(loss_fn)(params, model_online, x, y)

    # S1 full causal CCM: bottom=online, top=baseline
    model_s1 = build_mixed_model(["online", "baseline"], **common)
    g_s1 = jax.grad(loss_fn)(params, model_s1, x, y)

    leaves_bptt = block_leaf_paths(g_bptt)
    leaves_s1 = block_leaf_paths(g_s1)
    leaves_online = block_leaf_paths(g_online)

    print("\nPer-block comparison (S1 vs BPTT):")
    rows = []
    all_pass = True
    for key_ in sorted(leaves_bptt):
        gb = jnp.ravel(leaves_bptt[key_])
        gs = jnp.ravel(leaves_s1[key_])
        go = jnp.ravel(leaves_online[key_])
        cos_s1, rel_s1, nr_s1 = cos_relerr([gs], [gb])
        cos_on, rel_on, nr_on = cos_relerr([go], [gb])
        # SCOPE, precisely: Phase A's (E2) claims to exactly reconstruct
        # the bottom recurrent layer's OWN SSM parameters (Lambda, B, C,
        # D, log_step -- the "a"/"b"/readout blocks the toy's construction
        # targets). It does NOT claim (and structurally cannot, via this
        # mixed-model-type construction) to fix gradients for params that
        # sit STRICTLY BEFORE the bottom SSM's input -- here, S5Block_0's
        # own LayerNorm -- because those receive their gradient via the
        # online custom-VJP's `dx` (still the defective, instantaneous-
        # only input-cotangent, unchanged regardless of what the layer
        # above is). This is a genuinely NEW category of parameter versus
        # the toy (which had no learned pre-recurrence layer at all) --
        # see README "Transfer differences: toy vs S5" #1.
        is_ssm0_recurrence_block = key_.startswith("params/SSM_0/")
        gate = (rel_s1 < 1e-4) if is_ssm0_recurrence_block else True
        # top-layer (S5Block_1) and the encoder/head are IDENTICAL
        # across all three models by construction (same ordinary-
        # autodiff path in every arm here) -- rel_s1 should be ~0
        # there too, but the LOAD-BEARING claim is specifically the
        # bottom recurrent block, which is what differs between
        # "online" and "S1".
        all_pass = all_pass and gate
        rows.append(dict(block=key_, cos_s1_vs_bptt=cos_s1,
                         rel_err_s1_vs_bptt=rel_s1, norm_ratio_s1=nr_s1,
                         cos_online_vs_bptt=cos_on,
                         rel_err_online_vs_bptt=rel_on,
                         is_ssm0_recurrence_block=is_ssm0_recurrence_block,
                         gate_pass=bool(gate)))
        flag = "" if gate else "  <-- FAIL"
        print(f"  {key_:45s} S1: cos={cos_s1:.10f} rel={rel_s1:.2e}   "
             f"online: cos={cos_on:.4f} rel={rel_on:.4f}{flag}")

    print("-" * 78)
    print(f"ALL params/SSM_0/* (bottom recurrence params) GATES PASS "
         f"(rel_err < 1e-4): {all_pass}")
    ln0 = next((r for r in rows
               if r["block"] == "params/S5Block_0/LayerNorm_0/scale"), None)
    if ln0:
        print(f"NOTE (expected, documented, not a failure): "
             f"S5Block_0/LayerNorm_0 rel_err={ln0['rel_err_s1_vs_bptt']:.3f}"
             f" -- out of S1's claimed scope (params strictly BEFORE the "
             f"bottom SSM's input still see the online rule's defective "
             f"instantaneous dx; see README 'Transfer differences' #1).")
    if not all_pass:
        print("STOP: S1 does not reproduce BPTT to numerical precision on "
             "the SSM_0 recurrence parameters. Do not proceed to cluster "
             "training with this construction.")

    plain_forward_rel = check_plain_forward_matches_flax(
        args.d_model, args.state_size, args.n_classes, args.seq_len,
        args.batch, args.seed)
    plain_forward_pass = plain_forward_rel < 1e-4
    print(f"\nS2/S3 plain-JAX forward vs Flax model (ccm_rank1.py): "
         f"rel_err={plain_forward_rel:.2e}  "
         f"{'PASS' if plain_forward_pass else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=vars(args), rows=rows,
              all_ssm0_recurrence_gates_pass=bool(all_pass),
              plain_forward_rel_err=plain_forward_rel,
              plain_forward_pass=bool(plain_forward_pass))
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        "exactness_check_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
