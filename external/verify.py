import numpy as np
from ssm import *

rng = np.random.default_rng(0)
P1, P2, din, dout, T, B = 4, 4, 2, 1, 30, 3
p = init_params(P1, P2, din, dout, rng)
x = rng.normal(0, 1, (B, T, din))
tgt = rng.normal(0, 1, (B, T, dout))

L, g, aux = bptt_grads(p, x, tgt)

# ---- 1. finite differences vs BPTT ----------------------------------------
# For real L and complex z: dL/du = 2 Re(g), dL/dv = 2 Im(g)
print("1. BPTT vs finite differences")
eps = 1e-6
worst = 0.0
for k in KEYS:
    arr = np.asarray(p[k])
    idx = tuple(rng.integers(0, s) for s in arr.shape) if arr.ndim else ()
    for part in ([0, 1] if np.iscomplexobj(arr) else [0]):
        bump = eps * (1j ** part) if np.iscomplexobj(arr) else eps
        p2 = {kk: np.array(vv) for kk, vv in p.items()}
        p2[k][idx] += bump
        Lp, _ = loss_and_resid(*(forward(p2, x)[0], tgt))
        p2 = {kk: np.array(vv) for kk, vv in p.items()}
        p2[k][idx] -= bump
        Lm, _ = loss_and_resid(*(forward(p2, x)[0], tgt))
        fd = (Lp - Lm) / (2 * eps)
        an = 2 * (np.asarray(g[k])[idx].real if part == 0 else np.asarray(g[k])[idx].imag)
        rel = abs(fd - an) / (abs(fd) + 1e-12)
        worst = max(worst, rel)
        print(f"   {k:3s} {'re' if part==0 else 'im'}  fd={fd: .8e}  analytic={an: .8e}  rel={rel:.2e}")
print(f"   worst relative error: {worst:.2e}\n")

# ---- 2. D1 restoration: eligibility x EXACT adjoint == BPTT ----------------
print("2. Restoring the exact adjoint in the online path (D1 check)")
_, g_restored = online_grads(p, x, tgt, exact_layer1_error=True)
u, v = flat(g), flat(g_restored)
print(f"   cos(restored, BPTT)   = {cos(u, v):.15f}")
print(f"   relative error        = {np.linalg.norm(u - v) / np.linalg.norm(u):.3e}\n")

# ---- 3. top recurrent layer should already be exact online -----------------
print("3. Per-layer online-vs-BPTT alignment (no correction)")
_, g_on = online_grads(p, x, tgt)
for name, keys in [("top layer (a2,C1)", L2KEYS), ("lower layer (a1,B1)", L1KEYS)]:
    u, v = flat(g, keys), flat(g_on, keys)
    print(f"   {name:22s} cos={cos(u,v):.6f}   relerr={np.linalg.norm(u-v)/np.linalg.norm(u):.3e}")
