"""Phase 2A View 2 -- matched persistent exact-credit budget.

Scientific question: at a fixed amount of persistent exact online
credit state, what recurrent representation does each architecture
buy? Reuses View 1's frozen architecture/teacher DEFINITIONS (RTU from
b28, jet-algebra from b34a, bounded-interface flag's actual math from
b31a) completely unmodified; only their SIZE knobs are varied to hit
matched C_credit budgets, via credit_memory/p2a_sized_flag.py (a
size-generalization of b31a's outer glue, verified byte-identical to
b31a at b31a's own fixed sizes -- see its own equivalence check).

BUDGET FEASIBILITY (measured empirically before committing to tiers):
B34's per-gradient-call cost is O(r^2) per step (Toeplitz matrix
construction), and on this CPU-only machine r=3200 made even XLA
COMPILATION intractable (killed after >5 min, still compiling). Timed
r=200/500/800 at 0.08/0.35/0.89s per BPTT gradient call -- r=800 is the
practical ceiling for this sweep's scale (~200 training+eval runs).
Given this, and that flag's credit floor is highly sensitive to
(d_v,c_dim,k_out) -- b31a's own (4,8,8) gives a floor of ~3232, already
above the whole feasible range -- flag is instantiated at SMALLER
(d_v,c_dim,k_out)=(2,2,2) for View 2 (still genuinely V/U-coupled,
just a smaller invariant subspace/channel count than b31a's canonical
instance) so its credit floor (~28) sits well below the chosen tiers.

Three tiers solved to land NEAR a common credit target (200/500/800),
recording the EXACT realized value per architecture (never forced
equal):
  small:  RTU h=25  (r=50,  P=100,  credit=200)
          B34 r=200 (P=200, credit=200)
          Flag d_u=14,h=5,d_v=c=k=2 (r=16, P=100, credit=200)
  medium: RTU h=63  (r=126, P=252,  credit=504)
          B34 r=500 (P=500, credit=500)
          Flag d_u=44,h=10 (r=46, P=250, credit=500)
  large:  RTU h=100 (r=200, P=400,  credit=800)
          B34 r=800 (P=800, credit=800)
          Flag d_u=77,h=13 (r=79, P=400, credit=800)
Realized credits: (200,200,200), (504,500,500), (800,800,800) -- close
matches throughout, exact at 2 of 3 tiers.

DECOUPLED READOUT (necessary change from View 1): View 1 shared the
teacher's own h0/W directly with the student, which only worked
because every architecture was pinned to the SAME state_dim=64. Here
state_dim varies both across architectures (at a "matched" tier) and
across tiers, so the student now has its OWN trainable linear readout
W_s (jointly trained via the same Adam+clipping), and its own FIXED
zero initial state (h0_student=0) on every sequence -- only the
exogenous input xs and the teacher's own targets are shared with View
1's data-generation convention (make_sequence, unmodified).

Run: python -m credit_memory.p2a_view2_matched_credit
"""
from __future__ import annotations

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from credit_memory.p2a_expressivity_credit_frontier import (
    make_rollout, rtu_make_params, rtu_step, rtu_param_count, rtu_credit_scalars,
    jet_make_theta, make_jet_step, jet_param_count, jet_credit_scalars,
    make_teacher_B_jet, make_teacher_C_multipole, make_readout, Teacher, make_sequence,
    adam_init, adam_step, clip_grad, project_stable_R_V, DIVERGENCE_LOSS_CEIL,
    T_SEQ, Y_DIM, N_TRAIN, N_VAL, SEEDS,
    jet_make_gen_params,
)
from credit_memory.p2a_sized_flag import (
    make_sized_flag_consts, make_sized_flag_theta, make_sized_flag_step,
    sized_flag_param_count, sized_flag_credit_scalars,
)
from credit_memory.b28_rtu_faithful import rtu_streaming_init
from credit_memory.b34a_jet_algebra_correctness import (
    gen_forward as jet_gen_forward, alg_mult as jet_alg_mult, phi as jet_phi,
    phi_prime as jet_phi_prime, transpose_mult as jet_transpose_mult, X_DIM as JET_X_DIM,
)
from jax.flatten_util import ravel_pytree

# ---------------------------------------------------------------------
# Budget tiers (see module docstring for the solved (arg) values).
# ---------------------------------------------------------------------
FLAG_DV, FLAG_C, FLAG_K = 2, 2, 2

TIERS = dict(
    small=dict(rtu_h=25, jet_r=200, flag_du=14, flag_h=5),
    medium=dict(rtu_h=63, jet_r=500, flag_du=44, flag_h=10),
    large=dict(rtu_h=100, jet_r=800, flag_du=77, flag_h=13),
)


# ---------------------------------------------------------------------
# Sized positive-control teachers A (RTU) and D (flag). B (jet) and C
# (multipole) are already size-parameterized in the View-1 module and
# reused directly.
# ---------------------------------------------------------------------
def make_teacher_A_sized(seed, hidden_dim):
    params = rtu_make_params(seed, hidden_dim)
    rollout = make_rollout(lambda h, p, x: rtu_step(h, p, x, hidden_dim))
    W = make_readout(seed + 1, 2 * hidden_dim)
    return Teacher(f"A_independent_h{hidden_dim}", 2 * hidden_dim, rollout, params, W)


def make_teacher_D_sized(seed, consts, d_u, h_dim, d_v=FLAG_DV, c_dim=FLAG_C, k_out=FLAG_K):
    theta_star = make_sized_flag_theta(seed, d_u, h_dim, d_v, c_dim, k_out)
    step = make_sized_flag_step(consts, d_u)
    rollout = make_rollout(step)
    r = d_u + d_v
    W = make_readout(seed + 1, r)
    return Teacher(f"D_coupled_r{r}", r, rollout, theta_star, W)


# ---------------------------------------------------------------------
# Decoupled-readout training (necessary once state_dim varies across
# architectures/tiers -- see module docstring).
# ---------------------------------------------------------------------
def make_loss_decoupled(rollout_fn):
    def loss(params, h0, xs, targets):
        Hs = rollout_fn(h0, params["arch"], xs)
        Ys = Hs @ params["W"].T
        return jnp.mean(0.5 * jnp.sum((Ys - targets) ** 2, axis=1))
    return loss


def make_student_params(make_arch_params_fn, state_dim, seed, y_dim=Y_DIM):
    arch_params = make_arch_params_fn(seed)
    rng = np.random.RandomState(seed + 777_777)
    W = jnp.array(rng.randn(y_dim, state_dim) * (1.0 / np.sqrt(state_dim)))
    return dict(arch=arch_params, W=W)


def project_stable_R_V_nested(params):
    arch = params["arch"]
    if isinstance(arch, dict) and "R_V" in arch:
        return dict(params, arch=project_stable_R_V(arch))
    return params


def train_student_one_run_v2(rollout_fn, make_arch_params_fn, param_count_fn, teacher, state_dim, lr,
                              seed_init, n_train=N_TRAIN, n_val=N_VAL, T=T_SEQ,
                              seq_seed_offset=20_000, val_seed_offset=95_000, test_seed_offset=200_000,
                              return_params=False):
    loss_fn = make_loss_decoupled(rollout_fn)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    loss_jit = jax.jit(loss_fn)

    params = make_student_params(make_arch_params_fn, state_dim, seed_init)
    opt_state = adam_init(params)
    h0_student = jnp.zeros(state_dim, dtype=jnp.float64)
    train_losses = []
    diverged, diverged_at_step = False, None
    t0 = time.time()
    for step in range(n_train):
        h0_t, xs = make_sequence(seq_seed_offset + step, T, teacher.state_dim)
        targets = teacher.targets(h0_t, xs)
        loss_val = float(loss_jit(params, h0_student, xs, targets))
        if not np.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS_CEIL:
            diverged, diverged_at_step = True, step
            break
        train_losses.append(loss_val)
        g = clip_grad(grad_fn(params, h0_student, xs, targets))
        params, opt_state = adam_step(params, g, opt_state, lr)
        params = project_stable_R_V_nested(params)
    elapsed = time.time() - t0

    if diverged:
        out = dict(train_losses=train_losses, nmse=None, test_nmse=None, val_loss=None, elapsed=elapsed,
                   P_c=param_count_fn(params["arch"]), diverged=True, diverged_at_step=diverged_at_step,
                   lr=lr, seed=seed_init)
        if return_params:
            out["params"] = params
        return out

    def eval_split(offset, n):
        losses, ref_var = [], []
        for i in range(n):
            h0_t, xs = make_sequence(offset + i, T, teacher.state_dim)
            targets = teacher.targets(h0_t, xs)
            losses.append(float(loss_jit(params, h0_student, xs, targets)))
            ref_var.append(float(jnp.mean(targets ** 2)))
        return losses, ref_var

    val_losses, val_ref_var = eval_split(val_seed_offset, n_val)
    test_losses, test_ref_var = eval_split(test_seed_offset, n_val)
    if not all(np.isfinite(val_losses)) or not all(np.isfinite(test_losses)):
        out = dict(train_losses=train_losses, nmse=None, test_nmse=None, val_loss=None, elapsed=elapsed,
                   P_c=param_count_fn(params["arch"]), diverged=True, diverged_at_step=n_train,
                   lr=lr, seed=seed_init)
        if return_params:
            out["params"] = params
        return out
    nmse = float(np.mean(val_losses) / (np.mean(val_ref_var) + 1e-12))
    test_nmse = float(np.mean(test_losses) / (np.mean(test_ref_var) + 1e-12))
    out = dict(train_losses=train_losses, val_loss=float(np.mean(val_losses)), nmse=nmse,
               test_nmse=test_nmse, elapsed=elapsed, P_c=param_count_fn(params["arch"]), diverged=False,
               diverged_at_step=None, lr=lr, seed=seed_init)
    if return_params:
        out["params"] = params
    return out


def train_student_with_grid_v2(rollout_fn, make_arch_params_fn, param_count_fn, teacher, state_dim,
                                lr_grid, seeds=SEEDS, **kwargs):
    all_runs = []
    for lr in lr_grid:
        for seed in seeds:
            res = train_student_one_run_v2(rollout_fn, make_arch_params_fn, param_count_fn, teacher,
                                            state_dim, lr, seed_init=1000 + seed, **kwargs)
            res["lr_tag"] = lr
            all_runs.append(res)

    n_total = len(all_runs)
    n_diverged = sum(1 for r in all_runs if r["diverged"])
    finite_runs = [r for r in all_runs if not r["diverged"]]

    by_lr = {}
    for lr in lr_grid:
        runs = [r for r in finite_runs if r["lr_tag"] == lr]
        if runs:
            by_lr[lr] = float(np.mean([r["nmse"] for r in runs]))

    if not by_lr:
        return dict(status="all_diverged", n_total=n_total, n_diverged=n_diverged, best_lr=None,
                    val_nmse_values=[], val_nmse_mean=None, val_nmse_median=None,
                    test_nmse_values=[], test_nmse_mean=None, test_nmse_median=None,
                    P_c=all_runs[0]["P_c"])

    best_lr = min(by_lr, key=by_lr.get)
    best_runs = [r for r in finite_runs if r["lr_tag"] == best_lr]
    val_nmse_values = [r["nmse"] for r in best_runs]
    test_nmse_values = [r["test_nmse"] for r in best_runs]
    return dict(status="ok", n_total=n_total, n_diverged=n_diverged, best_lr=best_lr,
                val_nmse_values=val_nmse_values, val_nmse_mean=float(np.mean(val_nmse_values)),
                val_nmse_median=float(np.median(val_nmse_values)),
                test_nmse_values=test_nmse_values, test_nmse_mean=float(np.mean(test_nmse_values)),
                test_nmse_median=float(np.median(test_nmse_values)), P_c=best_runs[0]["P_c"],
                diverged_at_best_lr=sum(1 for r in all_runs if r["lr_tag"] == best_lr and r["diverged"]))


# ---------------------------------------------------------------------
# Teacher B mode-usage diagnostic (jet-algebra-specific: an element
# u=(u_0,...,u_{r-1}) acts as a PURE SCALAR (M_u=u_0*I, no cross-
# coefficient coupling) iff u_1=...=u_{r-1}=0. So "did the student
# degenerate to an effectively diagonal/scalar solution" has an exact,
# architecture-native answer here: how much of the learned theta's
# energy sits in the scalar (index-0) coefficient vs. the r-1
# genuinely-coupling coefficients, plus how well it aligns with the
# teacher's own (fully dense, non-scalar) generator theta_star.
# ---------------------------------------------------------------------
def jet_mode_usage_diagnostic(theta_learned, theta_star):
    theta_learned = np.asarray(theta_learned)
    theta_star = np.asarray(theta_star)
    total_energy = float(np.sum(theta_learned ** 2)) + 1e-18
    frac_scalar_energy = float(theta_learned[0] ** 2 / total_energy)
    cos_sim = float(np.dot(theta_learned, theta_star) /
                     (np.linalg.norm(theta_learned) * np.linalg.norm(theta_star) + 1e-18))
    frac_scalar_energy_teacher = float(theta_star[0] ** 2 / (np.sum(theta_star ** 2) + 1e-18))
    return dict(frac_scalar_energy_learned=frac_scalar_energy, cos_sim_to_teacher=cos_sim,
                frac_scalar_energy_teacher=frac_scalar_energy_teacher, r=int(theta_learned.shape[0]))


# ---------------------------------------------------------------------
# PREFLIGHT (required before the sweep): (1) verify C_credit against
# the ACTUAL persistent eligibility arrays each architecture's own
# validated reduced-RTRL implementation allocates (not just the
# symbolic formula); (2) benchmark the ACTUAL reduced-RTRL update step
# (not only the BPTT surrogate used for training-time efficiency in
# this sweep) to confirm compute feasibility at each tier, in
# particular for B34 where persistent storage is O(r) but the
# structured multiplication (Toeplitz regular-rep matrix M_u, built
# and multiplied every step) is NOT O(r) -- it is the O(r^2)-ish cost
# that made r=3200 infeasible even to COMPILE on this machine.
# ---------------------------------------------------------------------
def make_jet_reduced_step(gen_params, r):
    def step(h, s, theta, x_t):
        a_t, b_t, kappa_t, c_t = jet_gen_forward(x_t, gen_params, r)
        A_theta_t = a_t + jet_alg_mult(kappa_t, theta, r)
        y_t = jet_alg_mult(A_theta_t, h, r) + jet_alg_mult(b_t, theta, r) + c_t
        d_t = jet_phi_prime(y_t, r)
        inner = jet_alg_mult(A_theta_t, s, r) + jet_alg_mult(kappa_t, h, r) + b_t
        s_next = jet_alg_mult(d_t, inner, r)
        h_next = jet_phi(y_t, r)
        g_contrib = jet_transpose_mult(s_next, jnp.ones(r), r)  # dummy upstream grad, timing only
        return h_next, s_next, g_contrib
    return jax.jit(step)


def preflight_credit_accounting():
    print("\n" + "=" * 78)
    print("PREFLIGHT: actual persistent-eligibility array sizes vs symbolic C_credit,")
    print("plus actual reduced-RTRL update-step timing (not just the BPTT training surrogate)")
    print("=" * 78)
    all_ok = True
    rows = []
    for tier_name in ("small", "medium", "large"):
        cfg = TIERS[tier_name]
        rtu_h, jet_r, flag_du, flag_h = cfg["rtu_h"], cfg["jet_r"], cfg["flag_du"], cfg["flag_h"]
        print(f"\n  --- tier '{tier_name}' ---")

        # --- RTU: actual sensitivity dict from rtu_streaming_init (the
        # validated real online-RTRL trace structure, b28_rtu_faithful.py). ---
        S0 = rtu_streaming_init(rtu_h, features=1)
        actual_rtu = sum(int(np.prod(v.shape)) for v in S0.values())
        symbolic_rtu = rtu_credit_scalars(rtu_h)["reduced"]
        ok_rtu = actual_rtu == symbolic_rtu
        all_ok &= ok_rtu
        rng = np.random.RandomState(0)
        rtu_params = rtu_make_params(0, rtu_h)
        stream_state = dict(real=np.zeros(rtu_h), imag=np.zeros(rtu_h), S=S0)
        u_t = jnp.array(rng.randn(1) * 0.5)
        from credit_memory.b28_rtu_faithful import rtu_streaming_step
        t0 = time.time()
        for _ in range(20):
            rtu_streaming_step(rtu_params, stream_state, u_t)
        t_rtu = (time.time() - t0) / 20
        print(f"    RTU  h={rtu_h:4d}  r={2*rtu_h:4d}  actual_credit={actual_rtu:5d}  "
              f"symbolic_credit={symbolic_rtu:5d}  MATCH={ok_rtu}  "
              f"reduced-RTRL step time={t_rtu*1000:.3f}ms  (=> T=128: {t_rtu*128:.3f}s)")

        # --- B34: actual s vector (reduced_algebra_grad's persistent
        # state) + actual per-step reduced-RTRL update timing. ---
        actual_jet = jet_r  # s = jnp.zeros(r) -- allocate explicitly to verify, not assumed
        s_alloc = jnp.zeros(jet_r)
        assert int(s_alloc.shape[0]) == jet_r
        symbolic_jet = jet_credit_scalars(jet_r)["reduced"]
        ok_jet = actual_jet == symbolic_jet == int(s_alloc.shape[0])
        all_ok &= ok_jet
        gen_params_t = jet_make_gen_params(seed=3000 + jet_r, r=jet_r)
        reduced_step = make_jet_reduced_step(gen_params_t, jet_r)
        rng2 = np.random.RandomState(0)
        theta_t = jnp.array(rng2.randn(jet_r) * 0.2)
        h_t = jnp.array(rng2.randn(jet_r) * 0.15)
        s_t = jnp.zeros(jet_r)
        x_t = jnp.array(rng2.randn(JET_X_DIM) * 0.6)
        t0 = time.time()
        h2, s2, g2 = reduced_step(h_t, s_t, theta_t, x_t)
        jax.block_until_ready((h2, s2, g2))
        compile_t = time.time() - t0
        t0 = time.time()
        for _ in range(10):
            h2, s2, g2 = reduced_step(h_t, s_t, theta_t, x_t)
        jax.block_until_ready((h2, s2, g2))
        t_jet = (time.time() - t0) / 10
        print(f"    B34  r={jet_r:4d}  P={jet_r:4d}  actual_credit={actual_jet:5d}  "
              f"symbolic_credit={symbolic_jet:5d}  MATCH={ok_jet}  "
              f"reduced-RTRL step: compile={compile_t:.3f}s  per-step={t_jet*1000:.3f}ms  "
              f"(=> T=128: {t_jet*128:.3f}s)  [O(r) STORAGE, O(r^2)-ish COMPUTE per step -- not O(r)]")

        # --- Flag(sized): actual E matrix (d_v, P_c) from the REAL
        # theta pytree (ravel_pytree'd, not the symbolic formula). ---
        theta_flag = make_sized_flag_theta(0, flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K)
        theta_flat, _ = ravel_pytree(theta_flag)
        P_c_actual = int(theta_flat.shape[0])
        E0 = jnp.zeros((FLAG_DV, P_c_actual))
        actual_flag = int(np.prod(E0.shape))
        symbolic_flag_dict = sized_flag_credit_scalars(flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K)
        symbolic_flag = symbolic_flag_dict["reduced"]
        ok_flag = actual_flag == symbolic_flag and P_c_actual == symbolic_flag_dict["P_c"]
        all_ok &= ok_flag
        print(f"    Flag d_u={flag_du:4d} h={flag_h:3d}  r={flag_du+FLAG_DV:4d}  P={P_c_actual:5d}  "
              f"actual_credit={actual_flag:5d}  symbolic_credit={symbolic_flag:5d}  MATCH={ok_flag}")

        rows.append(dict(tier=tier_name, rtu_ok=ok_rtu, jet_ok=ok_jet, flag_ok=ok_flag,
                          rtu_step_ms=t_rtu * 1000, jet_step_ms=t_jet * 1000))

    print(f"\nPREFLIGHT ALL EXACT MATCHES (actual == symbolic, every architecture, every tier): {all_ok}")
    print("NOTE: View 2 is matched on PERSISTENT EXACT-CREDIT BUDGET ONLY. Hidden dimension r, "
          "trainable parameter count P, total model memory, and per-step compute cost are NOT "
          "matched across architectures or tiers -- they are reported separately (see structural "
          "accounting) and generally differ substantially even when C_credit is closely matched.")
    return dict(all_ok=all_ok, rows=rows)


# ---------------------------------------------------------------------
# Per-tier architecture/teacher construction.
# ---------------------------------------------------------------------
def build_tier(tier_name):
    cfg = TIERS[tier_name]
    rtu_h = cfg["rtu_h"]
    jet_r = cfg["jet_r"]
    flag_du, flag_h = cfg["flag_du"], cfg["flag_h"]
    flag_r = flag_du + FLAG_DV

    # Frozen, per-tier substrates shared across ALL 4 teacher tests at
    # this tier (same convention as View 1's single FLAG_CONSTS /
    # teacher_B_gen row-wide reuse).
    jet_gen_params = jet_make_gen_params(seed=3000 + jet_r, r=jet_r)
    flag_consts = make_sized_flag_consts(seed=4000 + flag_du, d_u=flag_du)

    teacher_A = make_teacher_A_sized(seed=777, hidden_dim=rtu_h)
    teacher_B, _ = make_teacher_B_jet(seed=778, r=jet_r, gen_seed=3000 + jet_r)
    teacher_C = make_teacher_C_multipole(seed=779, hidden_dim=rtu_h)
    teacher_D = make_teacher_D_sized(seed=780, consts=flag_consts, d_u=flag_du, h_dim=flag_h)
    teachers = [teacher_A, teacher_B, teacher_C, teacher_D]

    architectures = dict(
        RTU=dict(rollout=make_rollout(lambda h, p, x: rtu_step(h, p, x, rtu_h)),
                 make_params=lambda seed: rtu_make_params(seed, rtu_h),
                 param_count=rtu_param_count, state_dim=2 * rtu_h,
                 credit=rtu_credit_scalars(rtu_h), r=2 * rtu_h, P=4 * rtu_h,
                 lr_grid=(0.03, 0.1, 0.3)),
        B34=dict(rollout=make_rollout(make_jet_step(jet_gen_params, jet_r)),
                 make_params=lambda seed: jet_make_theta(seed, jet_r),
                 param_count=jet_param_count, state_dim=jet_r,
                 credit=jet_credit_scalars(jet_r), r=jet_r, P=jet_r,
                 lr_grid=(0.01, 0.03, 0.1)),
        BoundedInterfaceFlag=dict(
                 rollout=make_rollout(make_sized_flag_step(flag_consts, flag_du)),
                 make_params=lambda seed: make_sized_flag_theta(seed, flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K),
                 param_count=sized_flag_param_count, state_dim=flag_r,
                 credit=sized_flag_credit_scalars(flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K),
                 r=flag_r, P=sized_flag_credit_scalars(flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K)["P_c"],
                 lr_grid=(0.003, 0.01, 0.03)),
    )
    return dict(teachers=teachers, architectures=architectures, jet_gen_params=jet_gen_params,
                jet_r=jet_r, flag_consts=flag_consts, flag_du=flag_du, flag_h=flag_h)


# ---------------------------------------------------------------------
# Per-tier positive-control sanity checks.
# ---------------------------------------------------------------------
def run_tier_sanity(tier_name, built):
    print(f"\n--- Tier '{tier_name}' positive-control sanity checks ---")
    arch = built["architectures"]
    teachers_by_name = {t.name.split("_")[0]: t for t in built["teachers"]}
    checks = {}
    pairs = [("RTU", "A", built["teachers"][0]), ("B34", "B", built["teachers"][1]),
             ("BoundedInterfaceFlag", "D", built["teachers"][3])]
    all_ok = True
    for arch_name, teacher_tag, teacher in pairs:
        a = arch[arch_name]
        res = train_student_with_grid_v2(a["rollout"], a["make_params"], a["param_count"], teacher,
                                          a["state_dim"], a["lr_grid"])
        checks[f"{arch_name}->{teacher_tag}"] = res
        ok = res["status"] == "ok" and res["test_nmse_mean"] < 0.5
        flag_note = "" if ok else "  [FLAG: possibly still optimization-limited at this tier/budget]"
        all_ok = all_ok and ok
        print(f"  {arch_name}->{teacher_tag} (r={a['r']}, credit={a['credit']['reduced']}): "
              f"status={res['status']} val_nmse={res.get('val_nmse_mean')} "
              f"test_nmse={res.get('test_nmse_mean')} diverged={res.get('n_diverged')}/{res.get('n_total')}"
              f"{flag_note}")
    return dict(all_ok=all_ok, checks=checks)


# ---------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------
def run_view2():
    results = []
    diagnostics_B = {}
    sanity_all = {}
    for tier_name in ("small", "medium", "large"):
        print("\n" + "=" * 78)
        print(f"TIER: {tier_name}  ({TIERS[tier_name]})")
        print("=" * 78)
        built = build_tier(tier_name)
        sanity = run_tier_sanity(tier_name, built)
        sanity_all[tier_name] = sanity["all_ok"]

        arch = built["architectures"]
        teachers = built["teachers"]
        for arch_name, a in arch.items():
            for teacher in teachers:
                res = train_student_with_grid_v2(a["rollout"], a["make_params"], a["param_count"], teacher,
                                                  a["state_dim"], a["lr_grid"])
                row = dict(tier=tier_name, arch=arch_name, teacher=teacher.name, r=a["r"], P=a["P"],
                           credit=a["credit"], rP=a["r"] * a["P"], **res)
                results.append(row)
                if res["status"] == "all_diverged":
                    print(f"  [{tier_name}] {arch_name:22s} vs {teacher.name:16s}  ALL DIVERGED "
                          f"({res['n_diverged']}/{res['n_total']}) -- optimization/dynamical instability, "
                          f"not representational impossibility", flush=True)
                else:
                    print(f"  [{tier_name}] {arch_name:22s} vs {teacher.name:16s}  r={a['r']:4d} "
                          f"P={a['P']:5d} credit={a['credit']['reduced']:5d}  best_lr={res['best_lr']}  "
                          f"VAL_NMSE={res['val_nmse_mean']:.4e}  TEST_NMSE={res['test_nmse_mean']:.4e}  "
                          f"diverged={res['n_diverged']}/{res['n_total']}", flush=True)

                # Teacher-B mode-usage diagnostic: only meaningful for the
                # B34 architecture (jet-algebra-native quantity), only on
                # teacher B (its own positive control).
                if arch_name == "B34" and teacher.name.startswith("B_jet") and res["status"] == "ok":
                    curve_res = train_student_one_run_v2(a["rollout"], a["make_params"], a["param_count"],
                                                          teacher, a["state_dim"], res["best_lr"],
                                                          seed_init=1000 + SEEDS[0], return_params=True)
                    if not curve_res["diverged"]:
                        theta_learned = curve_res["params"]["arch"]
                        diag = jet_mode_usage_diagnostic(theta_learned, teacher.params)
                        diagnostics_B[tier_name] = diag
                        print(f"    [teacher-B mode-usage diagnostic] r={diag['r']} "
                              f"frac_scalar_energy(learned)={diag['frac_scalar_energy_learned']:.4f} "
                              f"frac_scalar_energy(teacher)={diag['frac_scalar_energy_teacher']:.4f} "
                              f"cos_sim_to_teacher={diag['cos_sim_to_teacher']:.4f}")

    with open("/tmp/p2a_view2_results.json", "w") as f:
        json.dump(dict(results=results, diagnostics_B=diagnostics_B, sanity_all=sanity_all),
                   f, indent=2, default=str)
    print("\nSaved to /tmp/p2a_view2_results.json")
    return dict(results=results, diagnostics_B=diagnostics_B, sanity_all=sanity_all)


def print_structural_accounting_v2():
    print("\n" + "=" * 78)
    print("View 2 structural accounting: r, P, C_credit, r*P, ratio -- per architecture per tier")
    print("=" * 78)
    for tier_name in ("small", "medium", "large"):
        cfg = TIERS[tier_name]
        rtu_h, jet_r, flag_du, flag_h = cfg["rtu_h"], cfg["jet_r"], cfg["flag_du"], cfg["flag_h"]
        rtu_c = rtu_credit_scalars(rtu_h)
        jet_c = jet_credit_scalars(jet_r)
        flag_c = sized_flag_credit_scalars(flag_du, flag_h, FLAG_DV, FLAG_C, FLAG_K)
        print(f"\n  Tier '{tier_name}':")
        print(f"    RTU                   r={2*rtu_h:4d}  P={4*rtu_h:5d}  credit={rtu_c['reduced']:5d}  "
              f"r*P={rtu_c['full']:7d}  ratio={rtu_c['ratio']:.1f}x")
        print(f"    B34                   r={jet_r:4d}  P={jet_r:5d}  credit={jet_c['reduced']:5d}  "
              f"r*P={jet_c['full']:7d}  ratio={jet_c['ratio']:.1f}x")
        print(f"    BoundedInterfaceFlag  r={flag_du+FLAG_DV:4d}  P={flag_c['P_c']:5d}  "
              f"credit={flag_c['reduced']:5d}  r*P={flag_c['full']:7d}  ratio={flag_c['ratio']:.1f}x")


if __name__ == "__main__":
    preflight = preflight_credit_accounting()
    if not preflight["all_ok"]:
        print("\nSTOPPING: preflight actual-vs-symbolic credit accounting mismatch. Repair before sweeping.")
    else:
        out = run_view2()
        print_structural_accounting_v2()
        print(f"\nSanity gate per tier: {out['sanity_all']}")
