"""B37b control: is arm C's ~1e-3 NMSE a convergence floor or just 400 steps?
Same generic stable init, same LR grid, 10x the optimization budget."""
import json, time
import numpy as np
from credit_memory.b37b_quotient_trainability import (
    FAMILIES, R_VALUES, EVAL_SEEDS, LR_GRID, make_teacher, generic_stable_init, train_one)

rows, t0 = [], time.time()
for f in FAMILIES:
    for r in R_VALUES:
        for seed in EVAL_SEEDS:
            teacher = make_teacher(f, r, seed)
            p0 = generic_stable_init(r, seed)
            best = None
            for lr in LR_GRID:
                res = train_one(teacher, p0, f, r, lr, seed, n_steps=4000)
                if best is None or res["val_loss"] < best["val_loss"]:
                    best = dict(res, lr=lr)
            rows.append(dict(family=f, r=r, seed=seed, **best))
            print(f"[{time.time()-t0:7.1f}s] {f:20s} r={r} s={seed} lr={best['lr']:.0e} "
                  f"nmse={best['test_nmse']:.3e} mk={best['markov']:.3e} "
                  f"rho={best['rho']:.3f} div={best['diverged']}", flush=True)
json.dump(rows, open("results/b37b/armC_long.json", "w"), indent=1)
print("done", time.time()-t0)
