"""B35c diagnostic audit (NOT a re-freeze): why does the RegularBlock/
GenericBlock gap on Task A (generalized-mode) jump from ~3x at C=128 to
~21.5x at C=256? Reuses the EXACT frozen architecture, predeclared
sizing rule, LR grid, and seeds from b35c_matched_credit_frontier.py
(imported, not modified) -- only adds instrumentation (train-loss and
periodic validation-loss trajectories, per-seed breakdown) that does
not alter the training procedure itself. Does NOT change the frozen
headline numbers in PHASE_B35C.md.

Run: python -m credit_memory.b35c_diagnostic_audit
"""
from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.p2a_expressivity_credit_frontier import adam_init, adam_step, clip_grad, DIVERGENCE_LOSS_CEIL
from credit_memory.b35b1_mechanism_check import jordan_rollout
from credit_memory.b35c_matched_credit_frontier import (
    regular_config, generic_config, make_regular_student, make_generic_student,
    make_xs, LR_GRID, SEEDS, N_TRAIN, N_VAL, N_TEST, T_TRAIN,
)

CHECKPOINT_EVERY = 5


def train_one_run_instrumented(rollout_y, make_params, project, teacher_rollout, teacher_state_dim, lr, seed_init,
                                n_train=N_TRAIN, T=T_TRAIN):
    def loss_fn(params, xs, targets):
        ys = rollout_y(params, xs)
        return jnp.mean(0.5 * (ys - targets) ** 2)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    def eval_split(params, offset, n):
        losses, ref_var = [], []
        for i in range(n):
            xs = make_xs(offset + i, T)
            targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
            losses.append(float(loss_jit(params, xs, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return float(np.mean(losses) / (np.mean(ref_var) + 1e-12))

    params = make_params(seed_init)
    params = project(params)
    opt_state = adam_init(params)
    train_losses = []
    val_nmse_curve = []   # (step, val_nmse) checkpoints
    diverged, diverged_at = False, None
    for step in range(n_train):
        xs = make_xs(20_000 + step, T)
        targets = teacher_rollout(jnp.zeros(teacher_state_dim), xs)
        loss_val = float(loss_jit(params, xs, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        train_losses.append(loss_val)
        if step % CHECKPOINT_EVERY == 0:
            val_nmse_curve.append((step, eval_split(params, 95_000, N_VAL)))
        g = clip_grad(grad_fn(params, xs, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project(params)
    if diverged:
        return dict(diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init,
                    train_losses=train_losses, val_nmse_curve=val_nmse_curve,
                    nmse=None, test_nmse=None)

    val_nmse_curve.append((n_train, eval_split(params, 95_000, N_VAL)))
    val_nmse = eval_split(params, 95_000, N_VAL)
    test_nmse = eval_split(params, 200_000, N_TEST)
    train_nmse_final = float(np.mean(train_losses[-10:])) if len(train_losses) >= 10 else float(np.mean(train_losses))
    return dict(diverged=False, diverged_at=None, lr=lr, seed=seed_init,
                train_losses=train_losses, val_nmse_curve=val_nmse_curve,
                nmse=val_nmse, test_nmse=test_nmse, train_nmse_final=train_nmse_final)


def audit_cell(name, rollout_y, make_params, project, C, lr_grid=LR_GRID, seeds=SEEDS):
    print(f"\n{'='*78}\n{name} (C={C})\n{'='*78}")
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run_instrumented(rollout_y, make_params, project, jordan_rollout, 2,
                                              lr, seed_init=1000 + seed)
            res["lr_tag"] = lr
            all_runs.append(res)

    finite = [r for r in all_runs if not r["diverged"]]
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))
    best_lr = min(by_lr, key=by_lr.get) if by_lr else None
    best_runs = [r for r in finite if r["lr_tag"] == best_lr] if best_lr is not None else []

    print(f"  divergence: {n_diverged}/{len(all_runs)} total runs")
    print(f"  by_lr validation NMSE means: {by_lr}")
    print(f"  SELECTED best_lr = {best_lr}")
    print(f"\n  Per-seed detail at selected best_lr:")
    for r in best_runs:
        print(f"    seed={r['seed']}: train_nmse(last10 steps)={r['train_nmse_final']:.4e}  "
              f"val_nmse={r['nmse']:.4e}  test_nmse={r['test_nmse']:.4e}  "
              f"final_train_loss={r['train_losses'][-1]:.4e}  n_train_steps_completed={len(r['train_losses'])}")

    if best_runs:
        test_vals = [r["test_nmse"] for r in best_runs]
        val_vals = [r["nmse"] for r in best_runs]
        print(f"\n  Across-seed test NMSE: median={np.median(test_vals):.4e}  mean={np.mean(test_vals):.4e}  "
              f"std={np.std(test_vals):.4e}  min={np.min(test_vals):.4e}  max={np.max(test_vals):.4e}")
        print(f"  Across-seed val  NMSE: median={np.median(val_vals):.4e}  mean={np.mean(val_vals):.4e}  "
              f"std={np.std(val_vals):.4e}  min={np.min(val_vals):.4e}  max={np.max(val_vals):.4e}")

        print(f"\n  Learning curves (val NMSE at checkpoints, every {CHECKPOINT_EVERY} steps) per seed:")
        for r in best_runs:
            curve = r["val_nmse_curve"]
            sparse = curve[::4] + [curve[-1]] if len(curve) > 4 else curve
            curve_str = ", ".join(f"step{s}={v:.3e}" for s, v in sparse)
            print(f"    seed={r['seed']}: {curve_str}")

        print(f"\n  Train-loss curves (sparse checkpoints) per seed:")
        for r in best_runs:
            tl = r["train_losses"]
            n = len(tl)
            idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])) if n > 0 else []
            tl_str = ", ".join(f"step{i}={tl[i]:.3e}" for i in idxs)
            print(f"    seed={r['seed']}: {tl_str}")

        # is it still decreasing at the end?
        for r in best_runs:
            curve = r["val_nmse_curve"]
            if len(curve) >= 3:
                last3 = [v for _, v in curve[-3:]]
                still_decreasing = last3[-1] < last3[0] * 0.98
                print(f"    seed={r['seed']}: val NMSE last-3 checkpoints={['%.3e'%v for v in last3]}  "
                      f"still clearly decreasing at stop: {still_decreasing}")

    return dict(name=name, C=C, all_runs=all_runs, best_lr=best_lr, n_diverged=n_diverged,
                n_total=len(all_runs))


def run_longer_continuation(make_params, rollout_y, project, C, seeds=SEEDS, lr=None, n_train_long=500):
    print(f"\n{'='*78}\nDIAGNOSTIC-ONLY longer-training continuation (n_train={n_train_long}), C={C}, lr={lr}")
    print("(This does NOT replace the frozen B35c number.)")
    print(f"{'='*78}")
    for seed in seeds:
        res = train_one_run_instrumented(rollout_y, make_params, project, jordan_rollout, 2,
                                          lr, seed_init=1000 + seed, n_train=n_train_long)
        if res["diverged"]:
            print(f"  seed={seed}: DIVERGED at step {res['diverged_at']}")
            continue
        curve = res["val_nmse_curve"]
        sparse = curve[::len(curve) // 10] + [curve[-1]] if len(curve) > 10 else curve
        curve_str = ", ".join(f"step{s}={v:.3e}" for s, v in sparse)
        print(f"  seed={seed}: final test_nmse={res['test_nmse']:.4e}  val curve: {curve_str}")


if __name__ == "__main__":
    results = {}
    for C in (128, 256):
        rc, gc = regular_config(C), generic_config(C)
        reg = make_regular_student(rc["Q"], rc["d"])
        gen = make_generic_student(100 + C, gc["Q"], gc["d"], gc["p"])
        results[f"Regular_C{C}"] = audit_cell(f"RegularBlock", *reg, C)
        results[f"Generic_C{C}"] = audit_cell(f"GenericBlock", *gen, C)

    # Longer-training continuation for GenericBlock at C=256 (diagnostic only)
    C = 256
    gc = generic_config(C)
    gen = make_generic_student(100 + C, gc["Q"], gc["d"], gc["p"])
    best_lr_256 = results[f"Generic_C{C}"]["best_lr"]
    run_longer_continuation(gen[1], gen[0], gen[2], C, lr=best_lr_256, n_train_long=500)

    with open("/tmp/b35c_diagnostic_audit.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "all_runs"} for k, v in results.items()},
                   f, indent=2, default=str)
    print("\nSaved summary to /tmp/b35c_diagnostic_audit.json")
