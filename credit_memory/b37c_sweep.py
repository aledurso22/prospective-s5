"""B37c section 4 -- same teachers, data, seeds, budget, LR grid and validation
protocol as B37b; only the parameterization differs."""
import json, time
import numpy as np
from credit_memory.b37b_quotient_trainability import (
    FAMILIES, R_VALUES, EVAL_SEEDS, LR_GRID, make_teacher)
from credit_memory.b37c_productlocal_native import (
    spec_from_blocks, real_dim, generic_init, exact_init, perturb_params, train_one)

EPS_LADDER = (1e-6, 1e-4, 1e-2)
rows, t0 = [], time.time()
for family in FAMILIES:
    for r in R_VALUES:
        for seed in EVAL_SEEDS:
            teacher = make_teacher(family, r, seed)
            spec = spec_from_blocks(teacher["blocks"])
            assert real_dim(spec) == r
            ex, exd = exact_init(teacher, spec)
            arms = {"C_generic_stable": generic_init(spec, seed), "A_exact": ex}
            for e in EPS_LADDER:
                arms[f"B_perturbed_{e:.0e}"] = perturb_params(ex, e, seed)
            for arm, p0 in arms.items():
                steps = (400, 4000) if arm == "C_generic_stable" else (400,)
                for ns in steps:
                    best = None
                    for lr in LR_GRID:
                        res = train_one(teacher, p0, family, r, spec, lr, seed, n_steps=ns)
                        if best is None or res["val_loss"] < best["val_loss"]:
                            best = dict(res, lr=lr)
                    rows.append(dict(family=family, r=r, seed=seed, arm=arm, steps=ns,
                                     spec="".join(f"{k}{d}" for k, d in spec),
                                     condT=exd["condT"], resid_AT_TM=exd["resid_AT_TM"],
                                     condS=teacher["condS"], rho_A=teacher["rho"], **best))
                    print(f"[{time.time()-t0:7.1f}s] {family:20s} r={r} s={seed} {arm:18s} "
                          f"n={ns:4d} lr={best['lr']:.0e} nmse={best['test_nmse']:.3e} "
                          f"mk={best['markov']:.3e} rho={best['rho']:.3f} "
                          f"gam={best['gamma_H']:.2e} maxz={best['max_z']:.2e} "
                          f"div={best['diverged']}", flush=True)
json.dump(rows, open("results/b37c/rows.json", "w"), indent=1)
print(f"done in {time.time()-t0:.1f}s")
