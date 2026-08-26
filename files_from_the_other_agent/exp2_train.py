import numpy as np, sys
from ssm import *


class Adam:
    def __init__(s, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        s.m = np.zeros(shape, complex); s.v = np.zeros(shape); s.t = 0
        s.lr, s.b1, s.b2, s.eps = lr, b1, b2, eps
    def step(s, g):
        s.t += 1
        s.m = s.b1 * s.m + (1 - s.b1) * g
        s.v = s.b2 * s.v + (1 - s.b2) * np.abs(g) ** 2
        mh = s.m / (1 - s.b1 ** s.t); vh = s.v / (1 - s.b2 ** s.t)
        return s.lr * mh / (np.sqrt(vh) + s.eps)


B1_, B2_ = 0.90, 0.97          # two-timescale cascade the architecture can realise

def batch(rng, B, T, lag=None):
    """target = IIR(B2_) o IIR(B1_) applied to x  -> needs BOTH layers' memory."""
    x = rng.normal(0, 1, (B, T, 1))
    z = np.zeros((B, T)); u = np.zeros((B, T))
    zs = np.zeros(B); us = np.zeros(B)
    for t in range(T):
        zs = B1_ * zs + x[:, t, 0]; us = B2_ * us + zs
        z[:, t] = zs; u[:, t] = us
    u = u / np.sqrt(np.mean(u[:, T // 2:] ** 2))
    return x, u[:, :, None]


def clip(g, thr):
    n = np.linalg.norm(flat(g))
    return ({k: v * (thr / n) for k, v in g.items()}, thr / n) if n > thr else (g, 1.0)


def evaluate(p, rng_eval, T, lag, nb=8, B=16):
    r = np.random.default_rng(rng_eval)
    tot = 0.0
    for _ in range(nb):
        x, tg = batch(r, B, T, lag)
        y, _ = forward(p, x)
        tot += float(np.mean((y - tg) ** 2))
    return tot / nb


def run(arm, seed, steps, T, lag, B, P1, P2, eta, lr_w, thr):
    r = np.random.default_rng(1000 + seed)
    p = init_params(P1, P2, 1, 1, r)
    w = np.ones(P1, complex)
    mw = Adam(P1, lr_w)
    prev = None                      # (g_used_for_update, w_at_that_step)
    for n in range(steps):
        x, tg = batch(r, B, T, lag)
        if arm == 'bptt':
            _, g, _ = bptt_grads(p, x, tg); g, _ = clip(g, thr)
        else:
            _, g_on = online_grads(p, x, tg); g_on, _ = clip(g_on, thr)
            if arm == 'online':
                g = g_on
            elif arm == 'oracle_w':
                _, g_ex, _ = bptt_grads(p, x, tg)
                on, ex = mode_view(g_on), mode_view(g_ex)
                num = np.einsum('pn,pn->p', np.conj(on), ex)
                den = np.einsum('pn,pn->p', np.conj(on), on).real + 1e-30
                g = apply_w(g_on, num / den)
            else:                                     # routePC / routeA
                # CORRECT step, using the gradient observed at the *current* params
                if prev is not None:
                    g_prev, _ = prev
                    if arm == 'routeA':
                        _, teach, _ = bptt_grads(p, x, tg); teach, _ = clip(teach, thr)
                    else:
                        teach = g_on
                    gp, tv = mode_view(g_prev), mode_view(teach)
                    rhat = -eta * np.einsum('pn,pn->p', np.conj(gp), tv)
                    w = w - mw.step(rhat)
                g = apply_w(g_on, w)
                prev = (g_on, w.copy())
        for k in KEYS:
            p[k] = p[k] - eta * g[k]
        for k in ('a1', 'a2'):                       # keep the system stable
            m = np.abs(p[k]); bad = m > 0.9995
            if bad.any():
                p[k][bad] = p[k][bad] / m[bad] * 0.9995
    return evaluate(p, 7, T, lag), w


if __name__ == '__main__':
    steps, T, lag, B, P1, P2 = 900, 100, 35, 8, 10, 10
    eta, lr_w, thr = float(sys.argv[1]), 0.02, 50.0
    ARMS = ['bptt', 'online', 'routePC', 'routeA', 'oracle_w']
    res = {a: [] for a in ARMS}
    for seed in range(5):
        for a in ARMS:
            L, w = run(a, seed, steps, T, lag, B, P1, P2, eta, lr_w, thr)
            res[a].append(L)
        print(f"  seed {seed}: " + "  ".join(f"{a}={res[a][-1]:.4f}" for a in ARMS), flush=True)
    print(f"\neta={eta}  steps={steps}  T={T}  lag={lag}  P={P1}  5 seeds")
    print(f"{'arm':<12}{'median':>10}{'  per-seed'}")
    for a in ARMS:
        v = np.array(res[a])
        print(f"{a:<12}{np.median(v):>10.4f}   " + " ".join(f"{z:.4f}" for z in v))
    on, bp = np.median(res['online']), np.median(res['bptt'])
    for a in ['routePC', 'routeA', 'oracle_w']:
        print(f"  {a:<10} closes {100*(on-np.median(res[a]))/(on-bp):5.1f}% of the online->BPTT gap")
