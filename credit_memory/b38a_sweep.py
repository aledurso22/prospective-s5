"""B38a section 4 -- matched A/B training. Identical init, data, chunk schedule,
Adam state and evaluation; only the gradient routine differs."""
import json, time
import numpy as np
from credit_memory.b38a_train import (
    FAMILIES_ALL, L_VALUES, LR_GRID, EVAL_SEEDS, train, markov_err, make_teacher_norm)

rows, t0 = [], time.time()
for f in FAMILIES_ALL:
    for seed in EVAL_SEEDS:
        for L in L_VALUES:
            per_arm = {}
            for arm in ("A", "B"):
                best = None
                for lr in LR_GRID:
                    s = time.time()
                    o = train(f, 8, seed, arm, L, lr)
                    o["wall"] = time.time() - s
                    o["lr"] = lr
                    if best is None or o["val_loss"] < best["val_loss"]:
                        best = o
                per_arm[arm] = best
                rows.append(dict(family=f, r=8, seed=seed, L=L, arm=arm, lr=best["lr"],
                                 test_nmse=best["test_nmse"], val_loss=best["val_loss"],
                                 markov=markov_err(best["params"], best["teacher"],
                                                   best["spec"]),
                                 diverged=best["diverged"], wall=best["wall"],
                                 curve=[float(c) for c in best["curve"]]))
            ca = np.array(per_arm["A"]["curve"]); cb = np.array(per_arm["B"]["curve"])
            n = min(len(ca), len(cb))
            traj = float(np.max(np.abs(ca[:n] - cb[:n]) / (1 + np.abs(ca[:n])))) if n else np.nan
            rows[-1]["traj_dev"] = traj; rows[-2]["traj_dev"] = traj
            print(f"[{time.time()-t0:7.1f}s] {f:20s} s={seed} L={L:3d} | "
                  f"A nmse={per_arm['A']['test_nmse']:.3e} lr={per_arm['A']['lr']:.0e} | "
                  f"B nmse={per_arm['B']['test_nmse']:.3e} lr={per_arm['B']['lr']:.0e} | "
                  f"traj_dev={traj:.2e}", flush=True)
json.dump(rows, open("results/b38a/sweep.json", "w"), indent=1)
print(f"done in {time.time()-t0:.1f}s")
