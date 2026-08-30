"""B35a-4 -- representation-teacher sanity comparison: RTU, old single
jet, product-local d in {2,4,8}, on B_long (old single-jet teacher),
C_multi (old independent-multipole teacher), and a NEW E_local teacher
(product-local, multiple distinct semisimple sectors + nonzero
nilpotent tails within sectors -- genuinely requires both spectral
diversity AND within-sector generalized coupling).

Modest feasible scale: everything at r=64 (View-1-style SHARED h0/W,
no decoupled-readout machinery -- appropriate for a phase-diagram
SANITY check, not a rigorous matched-budget benchmark; a light
train/val split is used for LR selection, no held-out test set, and
the LR grid/seed count is intentionally small).

Run: python -m credit_memory.b35a4_representation_teachers
"""
from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.p2a_expressivity_credit_frontier import (
    make_rollout, make_sequence, make_readout, Teacher, adam_init, adam_step, clip_grad,
    DIVERGENCE_LOSS_CEIL, T_SEQ, Y_DIM,
    rtu_make_params, rtu_step, rtu_param_count,
    jet_make_theta, make_jet_step, jet_param_count, jet_make_gen_params,
    make_teacher_B_jet, make_teacher_C_multipole,
)
from credit_memory.b35a_product_local_algebra import (
    h_step_local, make_theta_local, make_gen_params_local, project_local_tails,
    alg_mult_blockwise,
)

R = 64
N_TRAIN = 80
N_VAL = 10
SEEDS = (0, 1)
LR_GRID = (0.01, 0.03, 0.1)


# ---------------------------------------------------------------------
# NEW teacher: E_local -- product-local, d=4 (Q=16), multiple distinct
# semisimple sectors (4 clusters of base coefficients) each carrying a
# NONZERO nilpotent tail (genuine generalized/non-diagonal local
# dynamics, not scalar) -- a clean positive control for product-local.
# ---------------------------------------------------------------------
def make_teacher_E_local(seed=781, Q=16, d=4):
    r = Q * d
    rng = np.random.RandomState(seed)
    base_clusters = [0.9, 0.5, 0.1, -0.4]  # multiple independently tunable spectral sectors
    per_cluster = Q // len(base_clusters)
    bases = []
    for b in base_clusters:
        bases += [b] * per_cluster
    while len(bases) < Q:
        bases.append(base_clusters[-1])
    bases = np.array(bases[:Q])

    theta_raw = np.zeros((Q, d))
    theta_raw[:, 0] = bases
    if d > 1:
        theta_raw[:, 1:] = rng.randn(Q, d - 1) * 0.35   # NONZERO nilpotent tail within every sector
    theta_star = project_local_tails(jnp.array(theta_raw.reshape(r)), Q, d)

    gen_params = make_gen_params_local(seed=seed + 1000, Q=Q, d=d)
    step = lambda h, p, x: h_step_local(h, p, jnp.stack([x, 0.0, 0.0, 0.0]), gen_params, Q, d)
    rollout = make_rollout(step)
    W = make_readout(seed + 1, r)
    return Teacher(f"E_local_Q{Q}d{d}", r, rollout, theta_star, W)


# ---------------------------------------------------------------------
# Shared-h0/W training loop (View-1 style), with an OPTIONAL per-step
# projection hook (used only by product-local students).
# ---------------------------------------------------------------------
def train_one_run(rollout_fn, make_params_fn, param_count_fn, teacher, lr, seed_init,
                   project_fn=None, n_train=N_TRAIN, n_val=N_VAL, T=T_SEQ,
                   seq_seed_offset=20_000, val_seed_offset=95_000):
    def loss_fn(params, h0, xs, targets, W):
        Hs = rollout_fn(h0, params, xs)
        Ys = Hs @ W.T
        return jnp.mean(0.5 * jnp.sum((Ys - targets) ** 2, axis=1))
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_params_fn(seed_init)
    if project_fn is not None:
        params = project_fn(params)
    opt_state = adam_init(params)
    diverged, diverged_at = False, None
    for step in range(n_train):
        h0, xs = make_sequence(seq_seed_offset + step, T, teacher.state_dim)
        targets = teacher.targets(h0, xs)
        loss_val = float(loss_jit(params, h0, xs, targets, teacher.W))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at = True, step
            break
        g = clip_grad(grad_fn(params, h0, xs, targets, teacher.W))
        params, opt_state = adam_step(params, g, opt_state, lr)
        if project_fn is not None:
            params = project_fn(params)
    if diverged:
        return dict(nmse=None, diverged=True, diverged_at=diverged_at, lr=lr, seed=seed_init)

    losses, ref_var = [], []
    for i in range(n_val):
        h0, xs = make_sequence(val_seed_offset + i, T, teacher.state_dim)
        targets = teacher.targets(h0, xs)
        losses.append(float(loss_jit(params, h0, xs, targets, teacher.W)))
        ref_var.append(float(jnp.mean(targets ** 2)))
    if not all(np.isfinite(losses)):
        return dict(nmse=None, diverged=True, diverged_at=n_train, lr=lr, seed=seed_init)
    nmse = float(np.mean(losses) / (np.mean(ref_var) + 1e-12))
    return dict(nmse=nmse, diverged=False, diverged_at=None, lr=lr, seed=seed_init)


def train_with_grid(rollout_fn, make_params_fn, param_count_fn, teacher, lr_grid=LR_GRID, seeds=SEEDS,
                     project_fn=None):
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_one_run(rollout_fn, make_params_fn, param_count_fn, teacher, lr,
                                 seed_init=1000 + seed, project_fn=project_fn)
            res["lr_tag"] = lr
            all_runs.append(res)
    finite = [r for r in all_runs if not r["diverged"]]
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    if not finite:
        return dict(status="all_diverged", n_diverged=n_diverged, n_total=len(all_runs),
                     best_lr=None, nmse_mean=None, nmse_values=[])
    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))
    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite if r["lr_tag"] == best_lr]
    nmse_values = [r["nmse"] for r in best_runs]
    return dict(status="ok", n_diverged=n_diverged, n_total=len(all_runs), best_lr=best_lr,
                nmse_mean=float(np.mean(nmse_values)), nmse_values=nmse_values)


# ---------------------------------------------------------------------
# Architectures at r=64.
# ---------------------------------------------------------------------
def build_architectures():
    RTU_HIDDEN = R // 2
    jet_gen = jet_make_gen_params(seed=2000 + R, r=R)

    archs = dict(
        RTU=dict(rollout=make_rollout(lambda h, p, x: rtu_step(h, p, x, RTU_HIDDEN)),
                  make_params=lambda seed: rtu_make_params(seed, RTU_HIDDEN),
                  param_count=rtu_param_count, project=None),
        OldSingleJet=dict(rollout=make_rollout(make_jet_step(jet_gen, R)),
                           make_params=lambda seed: jet_make_theta(seed, R),
                           param_count=jet_param_count, project=None),
    )
    for d in (2, 4, 8):
        Q = R // d
        gen_params_d = make_gen_params_local(seed=3000 + d, Q=Q, d=d)
        archs[f"ProductLocal_d{d}"] = dict(
            rollout=make_rollout(lambda h, p, x, gp=gen_params_d, Q=Q, d=d: h_step_local(
                h, p, jnp.stack([x, 0.0, 0.0, 0.0]), gp, Q, d)),
            make_params=lambda seed, Q=Q, d=d: make_theta_local(seed, Q, d),
            param_count=lambda p: int(p.shape[0]),
            project=lambda p, Q=Q, d=d: project_local_tails(p, Q, d),
        )
    return archs


def run_comparison():
    print("=" * 78)
    print("B35a-4: RTU / old single jet / product-local d=2,4,8  x  B_long / C_multi / E_local  (r=64)")
    print("=" * 78)
    teacher_B, _ = make_teacher_B_jet(seed=778, r=R, gen_seed=2000 + R)
    teacher_C = make_teacher_C_multipole(seed=779, hidden_dim=R // 2)
    teacher_E = make_teacher_E_local(seed=781, Q=16, d=4)
    teachers = [teacher_B, teacher_C, teacher_E]
    archs = build_architectures()

    results = []
    for arch_name, a in archs.items():
        for teacher in teachers:
            res = train_with_grid(a["rollout"], a["make_params"], a["param_count"], teacher,
                                   project_fn=a["project"])
            results.append(dict(arch=arch_name, teacher=teacher.name, **res))
            if res["status"] == "all_diverged":
                print(f"  {arch_name:16s} vs {teacher.name:16s}  ALL DIVERGED ({res['n_diverged']}/{res['n_total']})")
            else:
                print(f"  {arch_name:16s} vs {teacher.name:16s}  best_lr={res['best_lr']}  "
                      f"NMSE={res['nmse_mean']:.4e}  diverged={res['n_diverged']}/{res['n_total']}")
    with open("/tmp/b35a4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /tmp/b35a4_results.json")
    return results


if __name__ == "__main__":
    run_comparison()
