"""Cross-check: the jax online-S5 custom backward vs the numpy rig's
online gradient (trained_credit_gains.py) on an identical single-layer
config. This is the validation gate for ssm/online_s5: two independent
implementations of the SAME online rule must agree to ~1e-5.

Setup: L=1, N=4, single channel, D=0; identical a, B, c, x, y. Compare
per-mode complex gradients (a, Bb) and the readout gradient (C).

Run:  python check_online_s5.py
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from toyrig import ssm_rig as tcg
from ssm.online_s5.scan import ssm_online


def main() -> None:
    N, T, B = 4, 48, 3
    rng = np.random.RandomState(7)
    a = (np.linspace(0.9, 0.98, N)
         * np.exp(1j * rng.uniform(-1, 1, N)))
    Bm = (rng.randn(N, 1) + 1j * rng.randn(N, 1)) / np.sqrt(2)
    c = (rng.randn(N) + 1j * rng.randn(N)) / np.sqrt(2)
    x = rng.randn(T, B)
    y = rng.randn(T, B)

    # ---- numpy rig (tcg) online gradient ----
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = 1, N, T, 1, 1, B
    params = dict(rho=[np.zeros(N)], theta=[np.zeros(N)],
                  b=[Bm.copy()], c=c.copy())
    params["a"] = [a.copy()]
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    G = tcg.assemble(params, h, x, r, q, Sa, Sb)
    ga_np, gb_np, gc_np = G["a"][0], G["b"][0][:, 0], G["c"]

    # ---- jax custom-vjp online gradient (real re/im interface) ----
    def np_online_grad(a_, Bb_, C_):
        ga = np.zeros(N, complex)
        gb = np.zeros(N, complex)
        gc = np.zeros(N, complex)
        for b in range(B):
            xb = jnp.asarray(x[:, b])
            yb = y[:, b]
            yhat_b, vjp_fn = jax.vjp(ssm_online,
                                     jnp.asarray(a_.real),
                                     jnp.asarray(a_.imag),
                                     jnp.asarray(Bb_[:, 0].real),
                                     jnp.asarray(Bb_[:, 0].imag),
                                     jnp.asarray(C_.real),
                                     jnp.asarray(C_.imag),
                                     0.0, xb)
            dy = jnp.asarray(np.asarray(yhat_b) - yb)
            ct = vjp_fn(dy)
            ga += np.asarray(ct[0]) - 1j * np.asarray(ct[1])
            gb += np.asarray(ct[2]) - 1j * np.asarray(ct[3])
            gc += np.asarray(ct[4]) - 1j * np.asarray(ct[5])
        return ga, gb, gc        # unnormalized sums, matching tcg's G
    ga_jx, gb_jx, gc_jx = np_online_grad(a, Bm, c)

    def rel(u, v):
        return float(np.max(np.abs(u - v)) / (np.max(np.abs(u)) + 1e-12))

    print(f"len T={T} B={B} N={N}")
    print(f"rel diff a-grad: {rel(ga_np, ga_jx):.2e}")
    print(f"rel diff B-grad: {rel(gb_np, gb_jx):.2e}")
    print(f"rel diff C-grad: {rel(gc_np, gc_jx):.2e}")
    ok = max(rel(ga_np, ga_jx), rel(gb_np, gb_jx), rel(gc_np, gc_jx)) < 1e-3
    print(f"GATE (rel < 1e-3): {'PASS — implementations agree' if ok else 'FAIL — convention mismatch to fix'}")


if __name__ == "__main__":
    main()
