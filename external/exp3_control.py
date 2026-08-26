import numpy as np, warnings
warnings.filterwarnings('ignore')
from ssm import *
from exp2_train import batch, clip, Adam, evaluate

# Arms:
#   online       Zucchet online gradient, no correction
#   bptt         exact gradient, no correction
#   online_w     RoutePC: online gradient + causally learned per-mode complex w
#   bptt_w       CONTROL: exact gradient + the SAME causally learned w.
#                If w is repairing missing credit it should do nothing here,
#                because there is no missing credit to repair.


def run(arm, seed, steps, T, B, P, eta, lr_w, thr, probe_every=150):
    r = np.random.default_rng(1000 + seed)
    p = init_params(P, P, 1, 1, r)
    w = np.ones(P, complex); mw = Adam(P, lr_w); prev = None
    curve = []
    for n in range(steps):
        x, tg = batch(r, B, T)
        if arm.startswith('bptt'):
            _, g0, _ = bptt_grads(p, x, tg)
        else:
            _, g0 = online_grads(p, x, tg)
        g0, _ = clip(g0, thr)
        if arm.endswith('_w'):
            if prev is not None:
                gp, tv = mode_view(prev), mode_view(g0)
                w = w - mw.step(-eta * np.einsum('pn,pn->p', np.conj(gp), tv))
            prev = g0
            g = apply_w(g0, w)
        else:
            g = g0
        for k in KEYS:
            p[k] = p[k] - eta * g[k]
        for k in ('a1', 'a2'):
            m = np.abs(p[k]); bad = m > 0.9995
            if bad.any():
                p[k][bad] = p[k][bad] / m[bad] * 0.9995
        if (n + 1) % probe_every == 0:
            curve.append(evaluate(p, 7, T, None))
    return curve, w


ARMS = ['online', 'bptt', 'online_w', 'bptt_w']
STEPS, T, B, P, ETA, THR = 1200, 100, 8, 10, 0.003, 200.0
LRW = 0.01
curves = {a: [] for a in ARMS}
for seed in [1, 2, 3, 4, 5]:
    for a in ARMS:
        c, w = run(a, seed, STEPS, T, B, P, ETA, LRW, THR)
        curves[a].append(c)
    print(f"  seed {seed} done", flush=True)

print(f"\nEvaluation MSE vs training step (median over 5 seeds).  predict-zero = 1.00")
probes = [(i + 1) * 150 for i in range(STEPS // 150)]
print("step      " + "".join(f"{s:>9}" for s in probes))
for a in ARMS:
    M = np.median(np.array(curves[a]), axis=0)
    print(f"{a:<10}" + "".join(f"{v:>9.4f}" for v in M))

print("\nbest-along-curve MSE (per seed), i.e. tuning the stopping point per arm:")
for a in ARMS:
    b = [min(c) for c in curves[a]]
    print(f"  {a:<10} median {np.median(b):.4f}   per-seed " + " ".join(f"{z:.4f}" for z in b))
