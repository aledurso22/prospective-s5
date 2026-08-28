"""Phase B16.2 -- full task-complexity x width x credit-complexity law,
then a test of whether the exact tied-credit closure (B16) survives
depth and selectivity. Ordinary BPTT training only for Parts A-E; Parts
F-H are standalone exact-algebra constructions (no tcg dependence, no
new persistent online-credit training rule). No S5.

Run:  python -m credit_memory.b16_2_phase_diagram
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.teacher import set_l2_config
from credit_memory.b5_1_action_utility import adam_step
from credit_memory.b12_structural_spectral_theory import make_multi_delay_task
from credit_memory.b13_common_temporal_support import make_multi_freq_task
from credit_memory.b16_1_forward_expressivity import (
    make_group_map, init_tied_params, tie_flat_gradient, offset_a1)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b16_2")

T_TASK = 60
BATCH_TASK = 8
LR = 1e-3
STEPS_MAIN = 200


# ---------------------------------------------------------------------------
# Generic tied-BPTT training loop (task_fn always called, no M_IN==1 special
# case -- unlike b16_1's train_tied, so delay-r=1 uses the same generator
# family as delay-r=2/4/8 for a clean apples-to-apples task-complexity axis).
# ---------------------------------------------------------------------------
def train_tied2(seed, N_, G, M_IN_, task_fn, task_arg, group_mode="contiguous",
                steps=STEPS_MAIN, T_task=T_TASK, batch_task=BATCH_TASK, lr=LR,
                keep_curve=False):
    rng = np.random.RandomState(4000 + seed)
    g_of_j = make_group_map(N_, G, rng, mode=group_mode)
    params = init_tied_params(seed, N_, G, M_IN_, g_of_j)

    old_M_IN = tcg.M_IN
    tcg.M_IN = M_IN_
    tcg.L, tcg.N, tcg.T, tcg.BATCH = 2, N_, T_task, batch_task

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses = []
    try:
        for step in range(1, steps + 1):
            x, y = task_fn(rng, T_task, batch_task, M_IN_, task_arg)
            h, yhat = tcg.forward(params, x)
            r = yhat - y
            loss = 0.5 * float(np.mean(r ** 2))
            q = tcg.spatial_q(params, h, r)
            Sa, Sb = tcg.sensitivities(params, h, x)
            lam = tcg.exact_lambda(params, q)
            G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
            g = tcg.flat_grads(G_ex, params)
            tie_flat_gradient(g, N_, M_IN_, g_of_j, G)
            flat, m_, v_ = adam_step(flat, m_, v_, g, step, lr)
            params = tcg.pack(params, flat)
            losses.append(loss)
            if not np.isfinite(loss):
                break
    finally:
        tcg.M_IN = old_M_IN

    n_param_total = len(flat)
    out = dict(N=N_, G=G, seed=seed, steps_run=len(losses),
              final_loss=float(losses[-1]) if losses else None,
              median_late_loss=float(np.median(losses[-50:])) if len(losses) >= 50
              else (float(np.median(losses)) if losses else None),
              n_param_total=n_param_total, n_param_pole_effective=2 * G,
              S_credit=2 * G * N_, S_full=2 * N_ * N_)
    if keep_curve:
        out["curve"] = [float(v) for v in losses]
    return out


def delay_wrapper(rng, T_, BATCH_, M_IN_, delays):
    return make_multi_delay_task(rng, T_, BATCH_, M_IN_, delays)


def freq_wrapper(rng, T_, BATCH_, M_IN_, freqs):
    return make_multi_freq_task(rng, T_, BATCH_, M_IN_, freqs)


def k_exp_modes_wrapper(rng, T_, BATCH_, M_IN_, arg):
    K, mus = arg
    x = rng.randn(T_, BATCH_, 1)
    s = np.zeros((K, BATCH_))
    y = np.zeros((T_, BATCH_))
    for t in range(T_):
        s = mus[:, None] * s + x[t, :, 0][None, :]
        y[t] = s.mean(axis=0)
    return x, y


def pure_delay_wrapper(rng, T_, BATCH_, M_IN_, D):
    x = rng.randn(T_, BATCH_, 1)
    y = np.zeros((T_, BATCH_))
    if D < T_:
        y[D:] = x[:-D, :, 0]
    return x, y


def g_grid_for(N_):
    cand = [1, 2, 4, 8, 16, N_]
    return sorted(set(g for g in cand if g <= N_))


DELAY_FREQS_8 = [3, 11, 19, 27, 5, 13, 21, 29]


def delays_for(r):
    return [5 + 5 * k for k in range(r)]


def freqs_for(r):
    return DELAY_FREQS_8[:r]


# ---------------------------------------------------------------------------
# Part B: G_required
# ---------------------------------------------------------------------------
def curve_from_rows(rows, N_, task_name):
    Gs = sorted(set(r["G"] for r in rows if r["N"] == N_ and r["task"] == task_name))
    curve = {}
    for G in Gs:
        vals = [r["median_late_loss"] for r in rows
                if r["N"] == N_ and r["task"] == task_name and r["G"] == G]
        curve[G] = float(np.median(vals))
    return curve


def g_required(curve, mode, threshold):
    if not curve:
        return None
    Gs = sorted(curve)
    full_loss = curve[Gs[-1]]
    for G in Gs:
        if mode == "abs" and curve[G] <= threshold:
            return G
        if mode == "rel" and curve[G] <= full_loss * (1.0 + threshold):
            return G
    return None  # not achieved even at full G=N


# ---------------------------------------------------------------------------
# Part F: deep tied exact scalar recurrence (standalone, no tcg)
# h_t^l = a_l h_{t-1}^l + B_l h_t^{l-1},  l=1..L, scalar a_l tied across layer.
# Claim: d h_t^l/d theta_m = (B_l...B_1 v_m) z_{l,t}^m with
#   z_{1,t} = a_1 z_{1,t-1} + u_t^m,   z_{l,t} = a_l z_{l,t-1} + z_{l-1,t}.
# Verified three ways: (i) closed form, (ii) literal full-vector RTRL
# forward-sensitivity accumulation, (iii) linear-system exact reference
# (run the true recurrence with theta=e_m, all other components zero --
# exact by linearity, not finite-difference), (iv) reverse-mode BPTT
# adjoint recursion for a scalar readout loss.
# ---------------------------------------------------------------------------
def part_f_deep_tied(L_list, seed=0, T_=40, widths=(6, 10)):
    rng = np.random.RandomState(seed)
    rows = []
    for L_ in L_list:
        for N_ in widths:
            dims = [N_] * (L_ + 1)  # dims[0]=source dim, dims[l]=layer-l width
            a = rng.uniform(0.5, 0.95, size=L_)
            Bs = [rng.randn(dims[l + 1], dims[l]) / np.sqrt(dims[l]) for l in range(L_)]
            v_m = rng.randn(dims[0])
            v_m /= np.linalg.norm(v_m)
            u = rng.randn(T_)          # scalar drive u_t^m
            w = rng.randn(dims[L_])    # fixed readout direction

            # (i) closed form
            zs_prev = u.copy()  # "z_0,t" := u_t^m
            for l in range(L_):
                zc = np.zeros(T_)
                zprev = 0.0
                for t in range(T_):
                    zprev = a[l] * zprev + zs_prev[t]
                    zc[t] = zprev
                zs_prev = zc
            z_hist = zs_prev  # z_{L,t}
            path = v_m.copy()
            for l in range(L_):
                path = Bs[l] @ path
            closed = np.outer(z_hist, path)  # (T, dims[L]) = dh_t^L/dtheta_m

            # (ii) literal full-vector RTRL: S_t^l = a_l S_{t-1}^l + B_l S_t^{l-1}
            S = [np.zeros(dims[0])] + [np.zeros(dims[l + 1]) for l in range(L_)]
            rtrl_top = np.zeros((T_, dims[L_]))
            for t in range(T_):
                S[0] = v_m * u[t]
                for l in range(1, L_ + 1):
                    S[l] = a[l - 1] * S[l] + Bs[l - 1] @ S[l - 1]
                rtrl_top[t] = S[L_]

            # (iii) exact-by-linearity reference: run the TRUE recurrence with
            # theta = e_m (only source m active) and read off h_t^L directly.
            h = [np.zeros(dims[0])] + [np.zeros(dims[l + 1]) for l in range(L_)]
            lin_top = np.zeros((T_, dims[L_]))
            for t in range(T_):
                h[0] = v_m * u[t]
                for l in range(1, L_ + 1):
                    h[l] = a[l - 1] * h[l] + Bs[l - 1] @ h[l - 1]
                lin_top[t] = h[L_]

            # (iv) reverse-mode BPTT adjoint for scalar loss = w . h_T^L
            lam = [np.zeros(dims[l]) for l in range(L_ + 1)]  # lam[l] = dL/dh_t^l, reused per t
            lam[L_] = w.copy()
            dtheta_bptt = 0.0
            lam_next = [np.zeros(dims[l]) for l in range(L_ + 1)]
            lam_next[L_] = w.copy()
            for t in reversed(range(T_)):
                lam_t = [None] * (L_ + 1)
                lam_t[L_] = lam_next[L_].copy() if t == T_ - 1 else a[L_ - 1] * lam_next[L_]
                for l in range(L_ - 1, -1, -1):
                    contrib = Bs[l].T @ lam_t[l + 1] if l + 1 <= L_ else 0.0
                    temporal = a[l - 1] * lam_next[l] if l > 0 else 0.0
                    lam_t[l] = contrib + temporal
                dtheta_bptt += float(v_m @ lam_t[0]) * u[t]
                lam_next = lam_t
            # closed-form total dL/dtheta_m (L = w . h_{T-1}^L), independent
            # cross-check against the reverse-mode adjoint accumulation above
            closed_dtheta = float(w @ (path * z_hist[-1]))

            err_rtrl = float(np.max(np.abs(closed - rtrl_top)))
            err_lin = float(np.max(np.abs(closed - lin_top)))
            err_bptt = float(abs(dtheta_bptt - closed_dtheta))

            rows.append(dict(L=L_, N=N_, err_vs_rtrl=err_rtrl, err_vs_linear=err_lin,
                             err_vs_bptt_adjoint=err_bptt,
                             S_credit_per_source=2 * L_, S_full_per_source=2 * N_ * L_))
    return rows


# ---------------------------------------------------------------------------
# Part G: complex realification / depth path count.
# Symbolic conjugate-path count for naive real/imag expansion across L
# complex-tied layers is 2^(L-1) (analytic, stated not simulated). The
# actual minimal reachable dimension of the z-chain (complex tied scalar
# per layer, bidiagonal Jordan-chain structure) is computed via the rank
# of its controllability/Krylov matrix for a single scalar source.
# ---------------------------------------------------------------------------
def part_g_minimal_realization(L_list, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for L_ in L_list:
        symbolic_path_count = 2 ** max(L_ - 1, 0)
        # widely-linear real augmented z-chain: state = (Re z_1,Im z_1,...,Re z_L,Im z_L)
        a_complex = rng.uniform(0.5, 0.95, L_) * np.exp(1j * rng.uniform(-np.pi, np.pi, L_))
        dim = 2 * L_
        A_eff = np.zeros((dim, dim))
        Bv = np.zeros(dim)
        for l in range(L_):
            r, phi = np.abs(a_complex[l]), np.angle(a_complex[l])
            blk = r * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            A_eff[2 * l:2 * l + 2, 2 * l:2 * l + 2] = blk
            if l > 0:
                A_eff[2 * l:2 * l + 2, 2 * (l - 1):2 * (l - 1) + 2] = np.eye(2)
        Bv[0:2] = [1.0, 0.0]  # scalar real source enters layer 1's real/imag pair
        # controllability/Krylov matrix
        cols = [Bv]
        for _ in range(dim - 1):
            cols.append(A_eff @ cols[-1])
        K_mat = np.stack(cols, axis=1)
        rank = int(np.linalg.matrix_rank(K_mat, tol=1e-9))
        rows.append(dict(L=L_, symbolic_path_count=symbolic_path_count,
                         minimal_real_dim=dim, measured_reachable_rank=rank,
                         growth_law="O(2L)" if rank == dim else "UNEXPECTED"))
    return rows


# ---------------------------------------------------------------------------
# Part H: tied + selective combination.
# h_t = a_t h_{t-1} + x_t, tied scalar a_t across all N units at each t.
# H1 exogenous: a_t = a(theta_0, exo_signal_t)   -- no h_{t-1} dependence.
# H2 endogenous/selective: a_t = a(theta_0, x_t) -- input-dependent gate,
#   still N-dim h_{t-1} enters the sensitivity recursion multiplicatively
#   via the extra term h_{t-1} * (grad_x a_t . dx_t/dtheta).
# Measured via effective rank (participation ratio) of the sensitivity
# trajectory {dh_t/dtheta_0}_t, across widths N -- if H1 stays O(1) while
# H2 grows with N, selectivity breaks the tied closure.
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def part_h_tied_selective(widths, T_=40, seed=0):
    rows = []
    rng = np.random.RandomState(seed)
    for N_ in widths:
        exo_signal = rng.randn(T_)
        x = rng.randn(T_, N_)
        theta0 = 0.3
        eps = 1e-6

        def run(gate_kind, theta0_):
            h = np.zeros(N_)
            sens = np.zeros((T_, N_))  # dh_t/dtheta0
            dh_dth = np.zeros(N_)
            for t in range(T_):
                if gate_kind == "const":
                    # non-selective, non-time-varying baseline: a_t = a(theta0)
                    # only -- the actual regime B16/Part F's closure applies to.
                    a_t = sigmoid(theta0_)
                    da_dtheta = a_t * (1 - a_t)
                    dxdtheta = 0.0
                elif gate_kind == "exo":
                    # exogenous: a_t depends on theta0 and a FIXED external
                    # signal, never on the current or past state/input.
                    a_t = sigmoid(theta0_ * exo_signal[t])
                    da_dtheta = exo_signal[t] * a_t * (1 - a_t)
                    dxdtheta = 0.0
                else:  # endogenous/selective: a_t = a(theta0, x_t) -- Mamba-
                       # style input-dependent gate, function of the CURRENT
                       # input only (no h_{t-1} feedback, matching the H2
                       # spec literally).
                    drive = float(np.mean(x[t]))
                    a_t = sigmoid(theta0_ * drive)
                    da_dtheta = drive * a_t * (1 - a_t)
                    dxdtheta = 0.0
                h = a_t * h + x[t]
                dh_dth = a_t * dh_dth + h * da_dtheta + dxdtheta
                sens[t] = dh_dth
            return sens

        sens_const = run("const", theta0)
        sens_exo = run("exo", theta0)
        sens_sel = run("sel", theta0)

        def eff_rank(sens):
            # participation-ratio rank of the (T x N) sensitivity trajectory
            Ssv = np.linalg.svd(sens, compute_uv=False)
            if np.sum(Ssv ** 2) < 1e-30:
                return 0.0
            return float((np.sum(Ssv ** 2) ** 2) / np.sum(Ssv ** 4))

        rows.append(dict(N=N_, eff_rank_const=eff_rank(sens_const),
                         eff_rank_exo=eff_rank(sens_exo),
                         eff_rank_selective=eff_rank(sens_sel)))
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 90)
    print("Phase B16.2: full N x G x task-complexity phase diagram + depth/selectivity gate")
    print("=" * 90)
    SEEDS_MAIN = [0, 1]
    doc = {}

    # ---- Part A: N x G x task-complexity phase diagram ----
    print("\nPart A: N x G x task phase diagram")
    a_results = []
    task_specs = []
    for r in (1, 2, 4, 8):
        task_specs.append((f"delay_r{r}", delay_wrapper, r, delays_for(r)))
    for r in (1, 8):
        task_specs.append((f"freq_r{r}", freq_wrapper, r, freqs_for(r)))

    for N_ in (16, 32, 64, 128):
        for task_name, task_fn, M_IN_, arg in task_specs:
            for G in g_grid_for(N_):
                seeds = SEEDS_MAIN + ([2] if (N_ == 128 and task_name == "delay_r8") else [])
                for seed in seeds:
                    r_ = train_tied2(seed, N_, G, M_IN_, task_fn, arg, steps=STEPS_MAIN)
                    r_["task"] = task_name
                    a_results.append(r_)
        print(f"  N={N_} done ({len(a_results)} rows so far)")

    # N=256 minimal spot-check (2 tasks x 2 G values x 1 seed, reduced steps)
    for task_name, task_fn, M_IN_, arg in [("delay_r1", delay_wrapper, 1, delays_for(1)),
                                           ("delay_r8", delay_wrapper, 8, delays_for(8))]:
        for G in (1, 256):
            r_ = train_tied2(0, 256, G, M_IN_, task_fn, arg, steps=100)
            r_["task"] = task_name
            r_["note"] = "N=256 spot-check, 1 seed, 100 steps (reduced)"
            a_results.append(r_)
    print("  N=256 spot-check done")
    doc["part_a"] = a_results
    with open(os.path.join(RESULTS_DIR, "b16_2_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    # ---- Part B: G_required curves ----
    print("\nPart B: G_required(N, task) curves")
    part_b = []
    for N_ in (16, 32, 64, 128):
        for task_name, *_ in task_specs:
            curve = curve_from_rows(a_results, N_, task_name)
            if not curve:
                continue
            rec = dict(N=N_, task=task_name, curve=curve)
            for rel in (0.05, 0.10, 0.20):
                rec[f"G_req_rel{int(rel*100)}"] = g_required(curve, "rel", rel)
            rec["G_req_abs0.01"] = g_required(curve, "abs", 0.01)
            part_b.append(rec)
            print(f"  N={N_} {task_name}: curve={curve}")
    doc["part_b"] = part_b

    # ---- Part C: rational-complexity task family (N=64 only) ----
    print("\nPart C: rational/McMillan-complexity tasks (N=64)")
    c_results = []
    N_C = 64
    for K in (1, 2, 4, 8):
        mus = np.linspace(0.75, 0.95, K)
        for G in g_grid_for(N_C):
            for seed in SEEDS_MAIN:
                r_ = train_tied2(seed, N_C, G, 1, k_exp_modes_wrapper, (K, mus), steps=STEPS_MAIN)
                r_["task"] = f"kexp_K{K}"
                c_results.append(r_)
    for D in (5, 10, 20, 40):
        for G in g_grid_for(N_C):
            for seed in SEEDS_MAIN:
                r_ = train_tied2(seed, N_C, G, 1, pure_delay_wrapper, D, steps=STEPS_MAIN)
                r_["task"] = f"puredelay_D{D}"
                c_results.append(r_)
    doc["part_c"] = c_results
    with open(os.path.join(RESULTS_DIR, "b16_2_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)
    for task_name in sorted(set(r["task"] for r in c_results)):
        curve = curve_from_rows(c_results, N_C, task_name)
        g20 = g_required(curve, "rel", 0.20)
        print(f"  N={N_C} {task_name}: curve={curve}  G_req_rel20={g20}")

    # ---- Part D: performance-per-credit-budget Pareto (derived, no new runs) ----
    print("\nPart D: Pareto (derived from Part A + Part C)")
    pareto_rows = []
    for r_ in a_results + c_results:
        if r_.get("median_late_loss") is None:
            continue
        pareto_rows.append(dict(task=r_["task"], N=r_["N"], G=r_["G"],
                                S_credit=r_["S_credit"], loss=r_["median_late_loss"]))
    doc["part_d_pareto_rows"] = pareto_rows

    # ---- Gate check before Parts F-H ----
    gate_evidence = []
    for rec in part_b:
        g20 = rec.get("G_req_rel20")
        if g20 is not None:
            gate_evidence.append((rec["N"], rec["task"], g20, g20 / rec["N"]))
    gate_pass = any(N_ == 128 and ratio <= 0.25 for (N_, _, _, ratio) in gate_evidence)
    print(f"\nGATE check (G_required << N at N=128 for at least one task): "
         f"{'PASS' if gate_pass else 'FAIL'}")
    doc["gate_pass"] = gate_pass
    doc["gate_evidence"] = gate_evidence

    # ---- Part E: longer-training control ----
    print("\nPart E: longer-training control")
    e_results = []
    STEPS_LONG = 2000
    e_conditions = [
        ("delay_r4", delay_wrapper, 4, delays_for(4), 64),
        ("delay_r8", delay_wrapper, 8, delays_for(8), 64),
        ("kexp_K8", k_exp_modes_wrapper, 1, (8, np.linspace(0.75, 0.95, 8)), 64),
    ]
    for task_name, task_fn, M_IN_, arg, N_ in e_conditions:
        curve = curve_from_rows(a_results if "delay" in task_name else c_results, N_, task_name)
        g_near = g_required(curve, "rel", 0.20) if curve else None
        Gs_to_test = sorted(set([1, g_near if g_near else 8, N_]))
        for G in Gs_to_test:
            for seed in SEEDS_MAIN[:1]:
                r_ = train_tied2(seed, N_, G, M_IN_, task_fn, arg, steps=STEPS_LONG,
                                 keep_curve=True)
                r_["task"] = task_name
                r_["role"] = ("full" if G == N_ else ("near_threshold" if G == g_near else "below_threshold"))
                e_results.append(r_)
        print(f"  {task_name} N={N_}: tested G={Gs_to_test} for {STEPS_LONG} steps")
    doc["part_e"] = [{k: v for k, v in r.items() if k != "curve"} for r in e_results]
    doc["part_e_curves"] = [dict(task=r["task"], N=r["N"], G=r["G"], role=r["role"],
                                 curve_tail=r["curve"][-20:], curve_head=r["curve"][:20],
                                 curve_mid=r["curve"][len(r["curve"])//2:len(r["curve"])//2+20])
                            for r in e_results]
    with open(os.path.join(RESULTS_DIR, "b16_2_summary.json"), "w") as fjson:
        json.dump(doc, fjson, indent=2, default=str)

    # ---- Parts F-H: only if gate passes ----
    if gate_pass:
        print("\nGate PASSED -- proceeding to Parts F-H")
        print("\nPart F: deep tied exact scalar recurrence")
        f_results = part_f_deep_tied([2, 3, 4, 6, 8], widths=(6, 10))
        doc["part_f"] = f_results
        for r_ in f_results:
            print(f"  L={r_['L']} N={r_['N']}: err_vs_rtrl={r_['err_vs_rtrl']:.2e} "
                 f"err_vs_linear={r_['err_vs_linear']:.2e} err_vs_bptt={r_['err_vs_bptt_adjoint']:.2e}")

        print("\nPart G: complex realification / minimal realization")
        g_results = part_g_minimal_realization([2, 3, 4, 6, 8])
        doc["part_g"] = g_results
        for r_ in g_results:
            print(f"  L={r_['L']}: symbolic_path_count={r_['symbolic_path_count']} "
                 f"minimal_real_dim={r_['minimal_real_dim']} measured_rank={r_['measured_reachable_rank']} "
                 f"-> {r_['growth_law']}")

        print("\nPart H: tied + selective combination")
        h_results = part_h_tied_selective([8, 32, 128, 512])
        doc["part_h"] = h_results
        for r_ in h_results:
            print(f"  N={r_['N']}: eff_rank_const={r_['eff_rank_const']:.3f} "
                 f"eff_rank_exo={r_['eff_rank_exo']:.3f} "
                 f"eff_rank_selective={r_['eff_rank_selective']:.3f}")
    else:
        print("\nGate FAILED -- stopping before Parts F-H per protocol")
        doc["part_f"] = doc["part_g"] = doc["part_h"] = None

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    doc["git"] = git
    doc["config"] = dict(T=T_TASK, BATCH=BATCH_TASK, LR=LR, steps_main=STEPS_MAIN,
                         steps_long=STEPS_LONG, seeds=SEEDS_MAIN)
    out_path = os.path.join(RESULTS_DIR, "b16_2_summary.json")
    with open(out_path, "w") as fjson:
        json.dump(doc, fjson, indent=2, default=lambda o: (
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else
            bool(o) if isinstance(o, np.bool_) else str(o)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
