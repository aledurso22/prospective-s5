"""B37c -- MODEL-INDEPENDENT conditioning of each teacher's realization problem.
Nothing here depends on the parameterization; it bounds what any realization
must contend with (similarity conditioning, Gramian/Hankel conditioning,
transient gain). Used only to interpret residual failure."""
import numpy as np
from credit_memory.b37b_quotient_trainability import FAMILIES, make_teacher

def gram(A, V, n=400):
    W, X = np.zeros_like(A), V.reshape(-1, 1) if V.ndim == 1 else V
    P = X.copy()
    for _ in range(n):
        W = W + P @ P.T
        P = A @ P
    return W

print(f"{'family':21s} {'r':>2s} {'rho(A)':>7s} {'cond(S)':>9s} {'Gam_H=max||A^k||':>16s} "
      f"{'cond(Wc)':>10s} {'cond(Wo)':>10s} {'Hankel sv cond':>14s} {'peak |g_k|':>11s}")
out = {}
for f in FAMILIES:
    for r in (4, 8):
        gh, cwc, cwo, hsv, pk, cs = [], [], [], [], [], []
        for seed in (0, 1, 2):
            t = make_teacher(f, r, seed)
            A, B, C = t["A"], t["Bs"], t["Cs"]
            P, g = np.eye(r), 0.0
            for _ in range(40):
                P = A @ P; g = max(g, float(np.linalg.norm(P, 2)))
            Wc, Wo = gram(A, B), gram(A.T, C)
            hs = np.sqrt(np.clip(np.linalg.eigvals(Wc @ Wo).real, 1e-300, None))
            hs = np.sort(hs)[::-1]
            gk = [float(C @ np.linalg.matrix_power(A, k) @ B) for k in range(40)]
            gh.append(g); cs.append(t["condS"])
            cwc.append(np.linalg.cond(Wc)); cwo.append(np.linalg.cond(Wo))
            hsv.append(hs[0] / max(hs[-1], 1e-300)); pk.append(max(abs(np.array(gk))))
        m = lambda v: float(np.median(v))
        out[(f, r)] = dict(gamma=m(gh), hsv=m(hsv))
        print(f"{f:21s} {r:2d} {m([make_teacher(f,r,s)['rho'] for s in (0,1,2)]):7.4f} "
              f"{m(cs):9.2e} {m(gh):16.3e} {m(cwc):10.2e} {m(cwo):10.2e} {m(hsv):14.2e} {m(pk):11.2e}")
