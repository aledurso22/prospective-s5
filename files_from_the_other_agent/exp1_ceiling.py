import numpy as np
from ssm import *


def fit_w(g_on, g_ex, kind, P1):
    on, ex = mode_view(g_on), mode_view(g_ex)
    if kind == 'identity':
        return np.ones(P1, complex)
    if kind == 'global_complex':
        num = np.vdot(on.ravel(), ex.ravel()); den = np.vdot(on.ravel(), on.ravel()).real
        return np.full(P1, num / (den + 1e-30))
    num = np.einsum('pn,pn->p', np.conj(on), ex)
    den = np.einsum('pn,pn->p', np.conj(on), on).real + 1e-30
    return (num / den) if kind == 'mode_complex' else (num.real / den).astype(complex)


def task(rng, B, T, din, dout, lag):
    x = rng.normal(0, 1, (B, T, din))
    tgt = np.zeros((B, T, dout))
    tgt[:, lag:, 0] = x[:, :-lag, 0]
    return x, tgt


rng = np.random.default_rng(1)
P1 = P2 = 12; din = dout = 1; T = 120; lag = 40; B = 16
KINDS = ['identity', 'global_complex', 'mode_real', 'mode_complex']

rows = {k: {'in': [], 'out': []} for k in KINDS}
for seed in range(8):
    r = np.random.default_rng(100 + seed)
    p = init_params(P1, P2, din, dout, r)
    xa, ta = task(r, B, T, din, dout, lag)      # fit window
    xb, tb = task(r, B, T, din, dout, lag)      # held-out window
    _, gexA, _ = bptt_grads(p, xa, ta); _, gonA = online_grads(p, xa, ta)
    _, gexB, _ = bptt_grads(p, xb, tb); _, gonB = online_grads(p, xb, tb)
    for k in KINDS:
        w = fit_w(gonA, gexA, k, P1)
        for tag, (go, ge) in [('in', (gonA, gexA)), ('out', (gonB, gexB))]:
            rows[k][tag].append(cos(flat(apply_w(go, w), L1KEYS), flat(ge, L1KEYS)))

print(f"Layer-1 gradient alignment vs exact BPTT   (P={P1} modes, T={T}, lag={lag}, 8 seeds)")
print(f"{'geometry':<18}{'in-window':>22}{'held-out':>22}")
for k in KINDS:
    a, b = np.array(rows[k]['in']), np.array(rows[k]['out'])
    print(f"{k:<18}{np.median(a):>12.3f} [{a.min():.3f},{a.max():.3f}]"
          f"{np.median(b):>12.3f} [{b.min():.3f},{b.max():.3f}]")

# --- pole dependence of the learned correction ------------------------------
print("\nLearned per-mode complex w vs its own pole (pooled over seeds)")
W, R, TH = [], [], []
for seed in range(8):
    r = np.random.default_rng(100 + seed)
    p = init_params(P1, P2, din, dout, r)
    xa, ta = task(r, B, T, din, dout, lag)
    _, gex, _ = bptt_grads(p, xa, ta); _, gon = online_grads(p, xa, ta)
    W.append(fit_w(gon, gex, 'mode_complex', P1)); R.append(np.abs(p['a1'])); TH.append(np.angle(p['a1']))
W, R = np.concatenate(W), np.concatenate(R)
for lo, hi in [(0.90, 0.94), (0.94, 0.97), (0.97, 0.99), (0.99, 1.0)]:
    m = (R >= lo) & (R < hi)
    if m.sum():
        print(f"   |a1| in [{lo:.2f},{hi:.2f})  n={m.sum():3d}   "
              f"median |w|={np.median(np.abs(W[m])):7.3f}   median |arg w|={np.median(np.abs(np.angle(W[m]))):.3f} rad")
