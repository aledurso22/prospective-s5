import numpy as np, warnings
warnings.filterwarnings('ignore')
from ssm import *
from exp2_train import batch, clip, Adam, evaluate

# Same four arms, but the BASE optimiser is Adam instead of plain SGD.
# If w is acting as a per-mode preconditioner, Adam should already supply
# most of that benefit and w's advantage should shrink sharply.
def run(arm, seed, steps, T, B, P, eta, lr_w, thr, probe=150):
    r = np.random.default_rng(1000+seed)
    p = init_params(P,P,1,1,r); w = np.ones(P,complex); mw = Adam(P,lr_w); prev=None
    opt = {k: Adam(np.asarray(p[k]).shape, eta) for k in KEYS}
    curve=[]
    for n in range(steps):
        x,tg = batch(r,B,T)
        g0 = (bptt_grads(p,x,tg)[1] if arm.startswith('bptt') else online_grads(p,x,tg)[1])
        g0,_ = clip(g0,thr)
        if arm.endswith('_w'):
            if prev is not None:
                gp,tv = mode_view(prev), mode_view(g0)
                w = w - mw.step(-eta*np.einsum('pn,pn->p',np.conj(gp),tv))
            prev = g0; g = apply_w(g0,w)
        else: g = g0
        for k in KEYS: p[k] = p[k] - opt[k].step(g[k])
        for k in ('a1','a2'):
            m=np.abs(p[k]); bad=m>0.9995
            if bad.any(): p[k][bad]=p[k][bad]/m[bad]*0.9995
        if (n+1)%probe==0: curve.append(evaluate(p,7,T,None))
    return curve

ARMS=['online','bptt','online_w','bptt_w']
out={a:[] for a in ARMS}
for seed in [1,2,3,4,5]:
    for a in ARMS: out[a].append(run(a,seed,1200,100,8,10,0.01,0.01,200.))
    print(f"  seed {seed} done", flush=True)
print("\nBASE OPTIMISER = ADAM.  best-along-curve MSE (predict-zero = 1.00)")
for a in ARMS:
    b=[min(c) for c in out[a]]
    print(f"  {a:<10} median {np.median(b):.4f}   per-seed " + " ".join(f"{z:.4f}" for z in b))
