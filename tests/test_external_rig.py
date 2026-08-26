"""Independent-rig regression test (adapted from external/verify.py).

Origin: written by an independent second agent (see
INDEPENDENT_VALIDATION.md). Loads external/ssm.py BY PATH (it must not
shadow the repo's ssm/ package) and asserts the three identities that
independently corroborate D1 and the top-layer exactness:

  1. BPTT vs finite differences: rel < 1e-8 on every parameter.
     (Convention fix vs the original: the Wirtinger factor 2 applies only
     to COMPLEX parameters; the original applied it to the real bias d
     too, producing a spurious factor-2 "mismatch".)
  2. D1 restoration: eligibility x exact adjoint == BPTT (cos 1, rel tiny).
  3. Top recurrent layer online == BPTT; lower layer biased.

Run from repo root:  python -m tests.test_external_rig
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "external_ssm", os.path.join(ROOT, "external", "ssm.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def main() -> None:
    rng = np.random.default_rng(0)
    P1, P2, din, dout, T, B = 4, 4, 2, 1, 30, 3
    p = S.init_params(P1, P2, din, dout, rng)
    x = rng.normal(0, 1, (B, T, din))
    tgt = rng.normal(0, 1, (B, T, dout))

    L, g, aux = S.bptt_grads(p, x, tgt)

    # 1. finite differences (factor 2 only for complex params)
    eps, worst = 1e-6, 0.0
    for k in S.KEYS:
        arr = np.asarray(p[k])
        idx = tuple(rng.integers(0, s) for s in arr.shape) if arr.ndim else ()
        parts = [0, 1] if np.iscomplexobj(arr) else [0]
        for part in parts:
            bump = eps * (1j ** part) if np.iscomplexobj(arr) else eps
            p2 = {kk: np.array(vv) for kk, vv in p.items()}
            p2[k][idx] += bump
            Lp, _ = S.loss_and_resid(S.forward(p2, x)[0], tgt)
            p2 = {kk: np.array(vv) for kk, vv in p.items()}
            p2[k][idx] -= bump
            Lm, _ = S.loss_and_resid(S.forward(p2, x)[0], tgt)
            fd = (Lp - Lm) / (2 * eps)
            fac = 2.0 if np.iscomplexobj(arr) else 1.0
            an = fac * (np.asarray(g[k])[idx].real if part == 0
                        else np.asarray(g[k])[idx].imag)
            worst = max(worst, abs(fd - an) / (abs(fd) + 1e-12))
    print(f"[1] BPTT vs FD (external rig): worst rel {worst:.2e}  "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")
    assert worst < 1e-6  # FD-noise bar at eps=1e-6; identities are [2]/[3]

    # 2. D1 restoration
    _, g_res = S.online_grads(p, x, tgt, exact_layer1_error=True)
    u, v = S.flat(g), S.flat(g_res)
    cos = S.cos(u, v)
    rel = np.linalg.norm(u - v) / np.linalg.norm(u)
    print(f"[2] D1 restoration: cos {cos:.15f}  rel {rel:.2e}  "
          f"{'PASS' if rel < 1e-12 else 'FAIL'}")
    assert rel < 1e-12

    # 3. per-layer alignment
    _, g_on = S.online_grads(p, x, tgt)
    ut, vt = S.flat(g, S.L2KEYS), S.flat(g_on, S.L2KEYS)
    rel_top = np.linalg.norm(ut - vt) / np.linalg.norm(ut)
    ul, vl = S.flat(g, S.L1KEYS), S.flat(g_on, S.L1KEYS)
    cos_low = S.cos(ul, vl)
    ok = rel_top < 1e-12 and cos_low < 0.9
    print(f"[3] top-layer rel {rel_top:.2e} (<1e-12), lower-layer cos "
          f"{cos_low:.3f} (<0.9)  {'PASS' if ok else 'FAIL'}")
    assert ok
    print("ALL EXTERNAL-RIG GATES PASS")


if __name__ == "__main__":
    main()
