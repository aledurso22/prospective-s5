"""Correctness tests: associative scans must match sequential references.

Verifies (rtol < 1e-4, atol < 1e-5):
  1. The S5-style elementwise associative scan vs. a sequential loop.
  2. The prospective per-channel 2x2 affine associative scan vs. a sequential
     loop, including a check with transition matrices of the prospective form
     M = [[a1, a2], [1, 0]] built from HiPPO eigenvalues.

Which arm each test covers:
  [1]      BASELINE  ssm/baseline_s5/scan.py
  [2] [3] [4b] [4]   PROSPECTIVE  ssm/prospective/{scan,layer}.py
  [5]      BASELINE  ssm/baseline_s5/layer.py   (layer vs sequential ref)
  [6]      PROSPECTIVE  ssm/prospective/layer.py (layer vs sequential ref)

Run:  python test_scan.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ssm.shared.hippo import hippo_init
from ssm.baseline_s5.scan import (
    elementwise_scan,
    elementwise_scan_sequential,
)
from ssm.prospective.scan import (
    prospective_scan,
    prospective_scan_sequential,
)

RTOL = 1e-4
ATOL = 1e-5


def check(name: str, got: jnp.ndarray, ref: jnp.ndarray) -> None:
    got, ref = np.asarray(got), np.asarray(ref)
    ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
    err = np.max(np.abs(got - ref))
    rel = err / (np.max(np.abs(ref)) + 1e-12)
    print(f"  {name:<45s} max abs err = {err:.3e}  max rel = {rel:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name} mismatch: max abs err {err:.3e}")


def test_elementwise_scan() -> None:
    print("[1] S5-style elementwise associative scan vs sequential")
    key = jax.random.PRNGKey(0)
    T, H, N = 512, 8, 32
    k1, k2, k3 = jax.random.split(key, 3)
    # Stable complex multipliers |a| < 1 (as after bilinear discretization).
    mag = jax.random.uniform(k1, (T, H, N), minval=0.5, maxval=0.999)
    ang = jax.random.uniform(k2, (T, H, N), minval=-np.pi, maxval=np.pi)
    a = (mag * jnp.exp(1j * ang)).astype(jnp.complex64)
    b = (jax.random.normal(k3, (T, H, N), dtype=jnp.float32)
         + 1j * jax.random.normal(k3, (T, H, N), dtype=jnp.float32))
    b = b.astype(jnp.complex64)

    check("elementwise_scan", elementwise_scan(a, b),
          elementwise_scan_sequential(a, b))


def test_prospective_scan_random() -> None:
    print("[2] Prospective 2x2 affine scan vs sequential (random stable M)")
    key = jax.random.PRNGKey(1)
    T, H, N = 512, 8, 32
    k1, k2 = jax.random.split(key)
    # Random stable 2x2 matrices: scale random matrices to have spectral
    # radius < 1 (guaranteed by ||.||_2 < 1 via 0.9/sqrt(2) scaling).
    M = jax.random.normal(k1, (T, H, N, 2, 2), dtype=jnp.float32)
    M = (0.45 * M).astype(jnp.complex64)
    b = jax.random.normal(k2, (T, H, N, 2), dtype=jnp.float32).astype(jnp.complex64)

    check("prospective_scan (random M)", prospective_scan(M, b),
          prospective_scan_sequential(M, b))


def test_prospective_scan_hippo() -> None:
    print("[3] Prospective 2x2 affine scan vs sequential (HiPPO-form M)")
    T, H = 784, 4
    N = 64
    Lambda, _, B_tilde = hippo_init(N)
    rho = 0.1        # Delta t / tau
    # A stable companion is needed for a T=784 correctness check, so the
    # HiPPO spectrum is scaled down BY HAND here. The layer itself applies
    # no such scaling — run verbatim it diverges, by design (exact_failure.py).
    Delta = 2e-4
    A = Delta * Lambda
    a1 = (1.0 - rho) + (1.0 + rho) * A               # (N,)
    a2 = -A                                          # (N,)
    one = np.ones_like(a1)
    zero = np.zeros_like(a1)
    M1 = np.stack([np.stack([a1, a2], axis=-1),
                   np.stack([one, zero], axis=-1)], axis=-2)  # (N, 2, 2)
    M = np.broadcast_to(M1[None, None], (T, H, N, 2, 2))
    M = jnp.asarray(np.ascontiguousarray(M), jnp.complex64)

    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (T, H), dtype=jnp.float32)
    B = jnp.asarray(np.repeat(B_tilde, H, axis=1), jnp.complex64)  # (N, H)
    B_eff = B.T                                                    # (H, N) unscaled
    v = x[:, :, None].astype(jnp.complex64) * B_eff[None, :, :]    # (T, H, N)
    # x~_t = (1+rho) v_{t-1} - v_{t-2}
    vpad = jnp.concatenate([jnp.zeros_like(v[:2]), v], axis=0)
    xtilde = (1.0 + rho) * vpad[1:-1] - vpad[:-2]
    b = jnp.stack([xtilde, jnp.zeros_like(xtilde)], axis=-1)

    check("prospective_scan (HiPPO M)", prospective_scan(M, b),
          prospective_scan_sequential(M, b))

    # Also report the eigenvalue magnitudes of the prospective M matrices
    # (stability diagnostic).
    eigs = np.linalg.eigvals(np.asarray(M1))          # (N, 2)
    print(f"  max |eig(M_i)| for rho={rho}: {np.abs(eigs).max():.6f} "
          f"(must be <= 1 for stability)")


def test_companion_scans() -> None:
    print("[4b] Flattened/lax prospective scans vs sequential ref")
    from ssm.prospective.scan import (prospective_scan_flat,
                                      prospective_scan_companion_lax)
    T, N = 784, 64
    Lambda, _, _ = hippo_init(N)
    # scaled-down spectrum as in [3]: a stable companion for the long-window
    # check (the layer itself runs A = Lambda and diverges, by design)
    rho, Delta = 0.1, 2e-4
    A = Delta * Lambda
    a1 = (1.0 - rho) + (1.0 + rho) * A
    a2 = -A
    a1s = jnp.asarray(np.broadcast_to(a1[None], (T, N)), jnp.complex64)
    a2s = jnp.asarray(np.broadcast_to(a2[None], (T, N)), jnp.complex64)
    rng = np.random.RandomState(5)
    b0 = jnp.asarray(rng.randn(T, N).astype(np.float32) + 0j)

    # Sequential reference via the literal 2x2 form
    one = np.ones_like(a1)
    zero = np.zeros_like(a1)
    M1 = np.stack([np.stack([a1, a2], -1), np.stack([one, zero], -1)], -2)
    M = jnp.asarray(np.ascontiguousarray(
        np.broadcast_to(M1[None], (T, N, 2, 2))), jnp.complex64)
    b = jnp.stack([b0, jnp.zeros_like(b0)], axis=-1)
    ref = prospective_scan_sequential(M, b)[..., 0]

    check("prospective_scan_flat (assoc)",
          prospective_scan_flat(a1s, a2s, b0), ref)
    check("prospective_scan_companion_lax",
          prospective_scan_companion_lax(a1s, a2s, b0), ref)


def test_causal_conv() -> None:
    print("[4] Causal conv1d (grouped lax.conv) vs manual shift reference")
    from ssm.prospective.layer import causal_conv1d_time
    T, C = 64, 7
    rng = np.random.RandomState(0)
    v = jnp.asarray(rng.randn(T, C).astype(np.float32))
    kernels = jnp.asarray(rng.randn(C, 2).astype(np.float32))
    got = causal_conv1d_time(v, kernels)
    vpad = jnp.concatenate([jnp.zeros((1, C), v.dtype), v], axis=0)
    ref = kernels[:, 0][None] * vpad[:-1] + kernels[:, 1][None] * vpad[1:]
    check("causal_conv1d_time", got, ref)


def _get_complex(params, name):
    return params[name]["re"] + 1j * params[name]["im"]


def test_s5_layer_end_to_end() -> None:
    print("[5] S5SSM layer (scan path) vs fully sequential reference")
    from ssm.baseline_s5.layer import S5SSM, discretize_bilinear
    T, H, N = 128, 4, 16
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.randn(2, T, H).astype(np.float32))
    layer = S5SSM(state_size=N, d_model=H)
    variables = layer.init(jax.random.PRNGKey(0), x)
    params = variables["params"]
    got = layer.apply(variables, x)

    Lambda = _get_complex(params, "Lambda")
    B = _get_complex(params, "B")
    C = _get_complex(params, "C")
    D = params["D"]
    Delta = jnp.exp(params["log_step"])
    Lambda_bar, B_bar = discretize_bilinear(Lambda, B, Delta)  # (H, N)

    def ref_batch(xb):  # xb: (T, H)
        s = jnp.zeros((H, N), jnp.complex64)
        ys = []
        for t in range(T):
            s = Lambda_bar * s + B_bar * xb[t][:, None].astype(jnp.complex64)
            ys.append(jnp.einsum("hn,hn->h", C, s).real + D * xb[t])
        return jnp.stack(ys, 0)

    check("S5SSM layer", got, jax.vmap(ref_batch)(x))


def _prospective_layer_check(T: int, label: str) -> None:
    """ProspectiveSSM (the derivation verbatim) vs a fully sequential reference.

    The layer runs A = Lambda with no scaling and no clamps, so it diverges —
    the check runs over a short window where the state is still finite.
    """
    from ssm.prospective.layer import ProspectiveSSM
    H, N = 4, 16
    rng = np.random.RandomState(2)
    x = jnp.asarray(rng.randn(2, T, H).astype(np.float32))
    layer = ProspectiveSSM(state_size=N, d_model=H)
    variables = layer.init(jax.random.PRNGKey(0), x)
    params = variables["params"]
    got = layer.apply(variables, x)

    Lambda = _get_complex(params, "Lambda")
    B = _get_complex(params, "B")
    C = _get_complex(params, "C")
    D = params["D"]
    assert "log_step" not in params, "the derivation has no Delta parameter"
    rho = jnp.exp(params["log_ratio"])                                      # (H,)
    A = jnp.broadcast_to(Lambda[None, :], (H, N))                           # A = Lambda
    B_eff = B.T                                                             # (H, N) unscaled
    a1 = (1.0 - rho)[:, None] + (1.0 + rho)[:, None] * A
    a2 = -A

    def ref_batch(xb):  # xb: (T, H)
        v = xb[:, :, None].astype(jnp.complex64) * B_eff[None]              # (T, H, N)
        vpad = jnp.concatenate([jnp.zeros_like(v[:2]), v], axis=0)
        xtilde = (1.0 + rho)[None, :, None] * vpad[1:-1] - vpad[:-2]        # (T, H, N)
        s_prev1 = jnp.zeros((H, N), jnp.complex64)   # s_{t-1}
        s_prev2 = jnp.zeros((H, N), jnp.complex64)   # s_{t-2}
        ys = []
        for t in range(T):
            s = a1 * s_prev1 + a2 * s_prev2 + xtilde[t]
            ys.append(jnp.einsum("hn,hn->h", C, s).real + D * xb[t])
            s_prev2, s_prev1 = s_prev1, s
        return jnp.stack(ys, 0)

    check(label, got, jax.vmap(ref_batch)(x))


def test_prospective_layer_end_to_end() -> None:
    print("[6] ProspectiveSSM layer (scan+conv path) vs fully sequential ref")
    # The derivation verbatim (A = Lambda); T is short because this arm
    # diverges — see exact_failure.py.
    _prospective_layer_check(8, "ProspectiveSSM layer")


def main() -> None:
    print("=" * 72)
    print("Scan correctness tests (associative scan vs sequential reference)")
    print("=" * 72)
    test_elementwise_scan()
    test_prospective_scan_random()
    test_prospective_scan_hippo()
    test_companion_scans()
    test_causal_conv()
    test_s5_layer_end_to_end()
    test_prospective_layer_end_to_end()
    print("=" * 72)
    print("ALL SCAN TESTS PASSED (rtol < 1e-4)")
    print("=" * 72)


if __name__ == "__main__":
    main()
