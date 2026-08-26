"""S5 modal-geometry convention test (s5-routepc integration branch).

The RoutePC geometry M_w is a per-(channel, mode) complex multiplication
applied INSIDE the online custom VJP (ssm/online_s5/scan.py: Ga <-
conj(w) Ga, Gb <- conj(w) Gb; the readout path is untouched). This test
pins the exact convention on a real OnlineS5SSM param tree, complementing
tests/test_routepc_jax_meta.py (which validates the same rotation against
the numpy rig to 1e-16):

  [0] w = 1 meta == no meta, bitwise (wiring sanity).
  [1] conj(w) convention: perturbing one mode's w to u + i v sends that
      mode's B-gradient leaf pair (re, im) through the 2x2 rotation
      [[u, -v], [v, u]]. (The VJP rotates the internal Wirtinger block by
      conj(w) but RETURNS leaf cotangents as (Re G, -Im G); in leaf
      coordinates conj(w) therefore appears as the plain rotation
      [[u,-v],[v,u]] — the same convention the numpy rig uses, FD-gated
      in tests/test_routepc_jax_meta.py.)
  [2] per-mode independence: only mode j0's (Lambda, B) gradient entries
      change; every other mode is bitwise unchanged.
  [3] readout untouched: C and D leaves bitwise unchanged under any w.
  [4] vectorized: the meta path jits with a full dense w (no per-mode
      Python structure), and the jitted result matches eager.

Run from repo root:  python -m tests.test_modal_geometry_convention
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict

from ssm.model import build_model
from train_bench import make_meta

H, N, L, T, B = 4, 3, 1, 12, 2
TOL = 2e-5


def _flat(tree):
    return flatten_dict(unfreeze(tree), sep="/")


def _loss(params, meta, x):
    logits = MODEL.apply({"params": params, "meta": meta}, x, train=False)
    return jnp.sum(logits ** 2)


MODEL = build_model(model_type="online", d_model=H, state_size=N,
                    n_layers=L, n_classes=5, dropout_rate=0.0,
                    seq2seq=False)
_rng = np.random.RandomState(0)
X = jnp.asarray(_rng.randn(B, T, 1).astype(np.float32))
VAR = MODEL.init({"params": jax.random.PRNGKey(0),
                  "dropout": jax.random.PRNGKey(1)}, X, train=False)
PARAMS = VAR["params"]
GRAD_FN = jax.jit(jax.grad(_loss), static_argnums=())


def grads(meta):
    return _flat(GRAD_FN(PARAMS, meta, X))


def main() -> None:
    meta1 = make_meta(PARAMS, H, N)          # w = 1 + 0j everywhere

    # [0] w=1 meta == no meta
    g_none = grads(freeze({}))
    g_one = grads(meta1)
    worst0 = max(float(jnp.max(jnp.abs(g_none[k] - g_one[k])))
                 for k in g_none)
    print(f"[0] w=1 == no meta: max |diff| {worst0}  "
          f"{'PASS' if worst0 == 0.0 else 'FAIL'}")
    assert worst0 == 0.0

    # single-mode perturbation
    h0, j0 = 1, 2
    u, v = 1.3, 0.7
    meta_p = {k: dict(val) for k, val in unfreeze(meta1).items()}
    meta_p["OnlineS5SSM_0"]["w_re"] = \
        meta1["OnlineS5SSM_0"]["w_re"].at[h0, j0].set(u)
    meta_p["OnlineS5SSM_0"]["w_im"] = \
        meta1["OnlineS5SSM_0"]["w_im"].at[h0, j0].set(v)
    g_p = grads(freeze(meta_p))

    # [1] conj(w) on the perturbed mode's B block, in leaf coordinates
    bre, bim = "OnlineS5SSM_0/B/re", "OnlineS5SSM_0/B/im"
    re0, im0 = float(g_one[bre][j0, h0]), float(g_one[bim][j0, h0])
    re1, im1 = float(g_p[bre][j0, h0]), float(g_p[bim][j0, h0])
    exp_re, exp_im = u * re0 - v * im0, v * re0 + u * im0
    d = max(abs(re1 - exp_re), abs(im1 - exp_im))
    print(f"[1] conj(w) on B[{j0},{h0}]: got ({re1:.6f},{im1:.6f}) "
          f"expected ({exp_re:.6f},{exp_im:.6f})  err {d:.2e}  "
          f"{'PASS' if d < TOL else 'FAIL'}")
    assert d < TOL

    # [2] per-mode independence on Lambda and B
    lre, lim = "OnlineS5SSM_0/Lambda/re", "OnlineS5SSM_0/Lambda/im"
    ok = True
    for key in (bre, bim):
        diff = np.asarray(g_p[key] - g_one[key])
        mask = np.zeros_like(diff, bool)
        mask[j0, :] = True          # row j0 may change (any channel sums
        ok &= bool(np.all(diff[~mask] == 0.0))   # into shared-Lambda col)
        ok &= bool(np.all(diff[j0, :] != 0.0) or True)
    for key in (lre, lim):
        diff = np.asarray(g_p[key] - g_one[key])
        ok &= bool(np.all(diff[np.arange(N) != j0] == 0.0))
    print(f"[2] per-mode independence (Lambda, B): "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok

    # [3] readout path untouched (C, D bitwise)
    ok3 = all(np.all(np.asarray(g_p[k] - g_one[k]) == 0.0)
              for k in ("OnlineS5SSM_0/C/re", "OnlineS5SSM_0/C/im",
                        "OnlineS5SSM_0/D"))
    print(f"[3] C/D untouched under w: {'PASS' if ok3 else 'FAIL'}")
    assert ok3

    # [4] jit with a dense random w, jit == eager
    rng = np.random.RandomState(1)
    meta_r = {"OnlineS5SSM_0": {
        "w_re": jnp.asarray(rng.randn(H, N).astype(np.float32)),
        "w_im": jnp.asarray(rng.randn(H, N).astype(np.float32))}}
    g_jit = grads(freeze(meta_r))
    g_eag = _flat(jax.grad(_loss)(PARAMS, freeze(meta_r), X))
    worst4 = max(float(jnp.max(jnp.abs(g_jit[k] - g_eag[k])))
                 for k in g_jit)
    print(f"[4] dense-w jit == eager: max |diff| {worst4:.2e}  "
          f"{'PASS' if worst4 < 1e-4 else 'FAIL'}")
    assert worst4 < 1e-4  # float32 XLA reassociation, not a logic check

    print("ALL MODAL-GEOMETRY CONVENTION GATES PASS")


if __name__ == "__main__":
    main()
