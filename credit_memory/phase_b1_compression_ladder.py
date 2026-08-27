"""B1.1-B1.4 -- compression ladder: does a tiny causal state preserve the
exact Phase-A causal-dual gradient correction?

Representation-capacity / oracle-fitting experiment only. No online
learning, no optimizer-state adaptation, no RoutePC/prospective residual,
no Meta-SGD/Meta-Adam. All classes are driven ONLY by Sa0 (the existing,
already-implemented within-layer eligibility trace) and read out against
q0 (the existing, already-implemented naive spatial error) -- exactly the
two signals already available to the online rule at the point Phase-A's
P/Q system is driven. Fitting uses BPTT gradients as an OFFLINE oracle
teacher only (per B1.3); no exact gradient is used in any deployed
causal-training sense here (nothing is deployed -- this is pure
representation-capacity analysis).

Classes (per lower-layer mode m; N0=N1=N since toyrig.ssm_rig uses a
uniform per-layer mode count):

  C0  online baseline:            G[m] = sum_t conj(q0_t[m]) Sa0_t[m]
  C1  complex-linear, 1 state:    z_t = alpha z_{t-1} + Sa0_t
                                   G[m] = sum_t c conj(q0_t) z_t
  C2  widely-linear, 1 state:     z_t = alpha z_{t-1} + beta conj(z_{t-1})
                                        + Sa0_t + delta conj(Sa0_t)
                                   G[m] = sum_t [c conj(q0_t) z_t
                                                + d q0_t conj(z_t)]
  C3  exact-teacher-class,
      2 states PER (j,m) PAIR:    z1_t[j,m] = pole1[j,m] z1_{t-1}[j,m] + Sa0_t[m]
                                   z2_t[j,m] = pole2[j,m] z2_{t-1}[j,m] + Sa0_t[m]
                                   G[m] = sum_t sum_j [w1[j,m] conj(q1_t[j]) z1_t[j,m]
                                                       + w2[j,m] q1_t[j] conj(z2_t[j,m])]
      positive control: pole1=a1[j], pole2=conj(a1[j]),
                         w1=B1[j,m]/2, w2=conj(B1[j,m])/2
      (this is exactly Phase A's (E2) -- see PHASE_A.md)

gamma (the drive-signal scale in the handoff's schematic z_t=alpha
z_{t-1}+gamma u_t) is gauge-fixed to 1 for C1/C2: gamma and the readout
scale c are not separately identifiable from the gradient objective
alone (see PHASE_B1.md), so fixing gamma=1 removes a non-identifiable
direction rather than restricting the model class.

Run:  python -m credit_memory.phase_b1_compression_ladder
"""
from __future__ import annotations

import json
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax

from toyrig import ssm_rig as tcg
from credit_memory.teacher import compute_teacher, draw_trajectory, set_l2_config

jax.config.update("jax_enable_x64", True)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

N, T, BATCH = 6, 40, 8
TRAIN_SEEDS = list(range(0, 8))
VAL_SEEDS = list(range(8, 11))
TEST_SEEDS = list(range(11, 19))
N_FIT_TRAJ = 2       # per train seed, used in the fit loss
N_EVAL_TRAJ = 2       # per train seed, held out (train-seed generalization)
N_TEST_TRAJ = 3       # per test/val seed, held out entirely
STEPS = 800
LR = 3e-2


# ---------------------------------------------------------------------------
# data collection (numpy, via the exact teacher machinery -- unchanged)
# ---------------------------------------------------------------------------

def collect(seed, n_traj, traj_offset=0):
    """n_traj independent trajectories for one architecture seed."""
    with set_l2_config(N, T, BATCH):
        params = tcg.init_params(seed)
        rows = []
        for k in range(n_traj):
            rng = np.random.RandomState(50000 + seed * 1000 + traj_offset
                                        + k)
            x, r = draw_trajectory(params, rng, T, BATCH)
            out = compute_teacher(params, x, r)
            rows.append(out)
    return params, rows


def stack(rows_list):
    """rows_list: list of teacher-output dicts -> stacked jnp arrays,
    axis 0 = pooled (seed, trajectory) index."""
    Sa0 = jnp.asarray(np.stack([r["Sa0"] for r in rows_list]))
    q0 = jnp.asarray(np.stack([r["q0"] for r in rows_list]))
    q1 = jnp.asarray(np.stack([r["q1"] for r in rows_list]))
    G_bptt = jnp.asarray(np.stack([r["G_bptt"] for r in rows_list]))
    G_online = jnp.asarray(np.stack([r["G_online"] for r in rows_list]))
    return dict(Sa0=Sa0, q0=q0, q1=q1, G_bptt=G_bptt, G_online=G_online)


# ---------------------------------------------------------------------------
# C1 / C2 model: reparameterization, recursion, readout
# ---------------------------------------------------------------------------

def sig(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def c1_unpack(raw):
    """raw: (N, 4) -> alpha (N,) complex |alpha|<1, c (N,) complex."""
    rho, theta, c_re, c_im = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    alpha = sig(rho) * jnp.exp(1j * theta)
    c = c_re + 1j * c_im
    return alpha, jnp.zeros_like(alpha), jnp.zeros_like(alpha), c, \
        jnp.zeros_like(alpha)


def c2_unpack(raw):
    """raw: (N, 10) -> alpha, beta (|alpha|+|beta|<1), delta, c, d."""
    (rho_mag, rho_split, th_a, th_b, d_re, d_im,
     c_re, c_im, dd_re, dd_im) = [raw[:, i] for i in range(10)]
    mag = sig(rho_mag)
    split = sig(rho_split)
    alpha = mag * split * jnp.exp(1j * th_a)
    beta = mag * (1 - split) * jnp.exp(1j * th_b)
    delta = d_re + 1j * d_im
    c = c_re + 1j * c_im
    d = dd_re + 1j * dd_im
    return alpha, beta, delta, c, d


def widely_linear_scan(alpha, beta, delta, Sa0):
    """Sa0: (T, BATCH, N) complex -> z: (T, BATCH, N) complex, via
    z_t = alpha z_{t-1} + beta conj(z_{t-1}) + Sa0_t + delta conj(Sa0_t)."""
    drive = Sa0 + delta[None, None, :] * jnp.conj(Sa0)

    def step(z_prev, drive_t):
        z_t = alpha[None, :] * z_prev + beta[None, :] * jnp.conj(z_prev) \
            + drive_t
        return z_t, z_t

    z0 = jnp.zeros((Sa0.shape[1], Sa0.shape[2]), jnp.complex128)
    _, z = jax.lax.scan(step, z0, drive)
    return z


def c1c2_gradient(raw, unpack_fn, Sa0, q0):
    """Sa0, q0: (T, BATCH, N) -> G: (N,) complex, raw sum over (t, batch)."""
    alpha, beta, delta, c, d = unpack_fn(raw)
    z = widely_linear_scan(alpha, beta, delta, Sa0)
    term1 = c[None, None, :] * jnp.conj(q0) * z
    term2 = d[None, None, :] * q0 * jnp.conj(z)
    return jnp.sum(term1 + term2, axis=(0, 1))


def batched_gradient(raw, unpack_fn, Sa0_stack, q0_stack):
    return jax.vmap(lambda Sa0, q0: c1c2_gradient(raw, unpack_fn, Sa0, q0))(
        Sa0_stack, q0_stack)


# ---------------------------------------------------------------------------
# C3 model: exact-teacher-class, per (j, m) pair
# ---------------------------------------------------------------------------

def c3_unpack(raw):
    """raw: (N, N, 8) [j, m, :] -> pole1, pole2 (|.|<1), w1, w2 (N,N) each."""
    rho1, th1, rho2, th2, w1r, w1i, w2r, w2i = [raw[..., i]
                                                for i in range(8)]
    pole1 = sig(rho1) * jnp.exp(1j * th1)
    pole2 = sig(rho2) * jnp.exp(1j * th2)
    w1 = w1r + 1j * w1i
    w2 = w2r + 1j * w2i
    return pole1, pole2, w1, w2


def c3_exact_raw(a1, B1):
    """The analytically known exact Phase-A solution, as C3 raw params."""
    N_ = a1.shape[0]
    pole1 = np.broadcast_to(a1[:, None], (N_, N_))
    pole2 = np.conj(pole1)
    w1 = B1 / 2.0
    w2 = np.conj(B1) / 2.0
    rho1 = np.log(np.abs(pole1) / (1 - np.abs(pole1)))
    th1 = np.angle(pole1)
    rho2 = np.log(np.abs(pole2) / (1 - np.abs(pole2)))
    th2 = np.angle(pole2)
    raw = np.stack([rho1, th1, rho2, th2, w1.real, w1.imag,
                    w2.real, w2.imag], axis=-1)
    return jnp.asarray(raw)


def c3_gradient(raw, Sa0, q1):
    """Sa0, q1: (T, BATCH, N) -> G: (N,) complex. pole1/pole2/w1/w2 shape
    (N1, N0) = (j, m)."""
    pole1, pole2, w1, w2 = c3_unpack(raw)   # each (N1, N0)

    def scan_pole(pole):
        def step(z_prev, Sa0_t):
            z_t = pole[None, :, :] * z_prev + Sa0_t[:, None, :]
            return z_t, z_t
        z0 = jnp.zeros((Sa0.shape[1], pole.shape[0], pole.shape[1]),
                       jnp.complex128)
        _, z = jax.lax.scan(step, z0, Sa0)
        return z   # (T, BATCH, N1, N0)

    Z1 = scan_pole(pole1)
    Z2 = scan_pole(pole2)
    # Phase-A (E2): term1 = (1/2) B1[j,m] conj(q1) P[j,m], with P using
    # pole a1; term2 = (1/2) conj(B1[j,m]) q1 Q[j,m] (Q itself, NOT
    # conj(Q)), with Q using pole conj(a1). Here Z1 plays P's role
    # exactly (same pole a1=pole1), Z2 plays Q's role exactly (same pole
    # conj(a1)=pole2) -- no extra conjugate on Z2.
    term1 = jnp.einsum("jm,tbj,tbjm->m", w1, jnp.conj(q1), Z1)
    term2 = jnp.einsum("jm,tbj,tbjm->m", w2, q1, Z2)
    return term1 + term2


def batched_c3_gradient(raw, Sa0_stack, q1_stack):
    return jax.vmap(lambda Sa0, q1: c3_gradient(raw, Sa0, q1))(
        Sa0_stack, q1_stack)


# ---------------------------------------------------------------------------
# metrics + fitting
# ---------------------------------------------------------------------------

def cos_np(u, v):
    u = np.ravel(u); v = np.ravel(v)
    return float(np.abs(np.vdot(v, u)) / (np.linalg.norm(u)
                                          * np.linalg.norm(v) + 1e-300))


def relerr_np(u, v):
    u = np.ravel(u); v = np.ravel(v)
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-300))


def cos_jax(u, v):
    return jnp.abs(jnp.vdot(v, u)) / (jnp.linalg.norm(u)
                                      * jnp.linalg.norm(v) + 1e-300)


def fit_c1c2(unpack_fn, n_params_per_mode, fit_data, seed_key=0,
             loss_kind="grad"):
    """loss_kind: 'grad' (1 - mean cos over fit set, primary) or
    'state_mse' (informative control, C1 only -- fits z to a coupling-
    weighted projection of the true P channel, see PHASE_B1.md)."""
    key = jax.random.PRNGKey(seed_key)
    raw = 0.1 * jax.random.normal(key, (N, n_params_per_mode))
    opt = optax.adam(LR)
    opt_state = opt.init(raw)

    Sa0_stack, q0_stack, G_bptt_stack = (fit_data["Sa0"], fit_data["q0"],
                                         fit_data["G_bptt"])

    def loss_grad(raw):
        G = batched_gradient(raw, unpack_fn, Sa0_stack, q0_stack)
        c = jax.vmap(cos_jax)(G, G_bptt_stack)
        return jnp.mean(1.0 - c)

    def loss_state_mse(raw):
        alpha, beta, delta, c, d = unpack_fn(raw)
        z = jax.vmap(lambda Sa0: widely_linear_scan(alpha, beta, delta,
                                                     Sa0))(Sa0_stack)
        target = fit_data["state_target"]
        return jnp.mean(jnp.abs(z - target) ** 2)

    loss_fn = loss_grad if loss_kind == "grad" else loss_state_mse

    @jax.jit
    def step(raw, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(raw)
        updates, opt_state = opt.update(grad, opt_state)
        raw = optax.apply_updates(raw, updates)
        return raw, opt_state, loss

    history = []
    for i in range(STEPS):
        raw, opt_state, loss = step(raw, opt_state)
        if i % 100 == 0 or i == STEPS - 1:
            history.append(float(loss))
    return raw, history


def evaluate(unpack_fn, raw, rows, use_c3=False):
    """rows: list of teacher-output dicts (one per trajectory) -> list of
    per-trajectory metric dicts."""
    out_rows = []
    for r in rows:
        if use_c3:
            G = np.asarray(c3_gradient(raw, jnp.asarray(r["Sa0"]),
                                       jnp.asarray(r["q1"])))
        else:
            G = np.asarray(c1c2_gradient(raw, unpack_fn,
                                         jnp.asarray(r["Sa0"]),
                                         jnp.asarray(r["q0"])))
        G_bptt, G_online = r["G_bptt"], r["G_online"]
        c_comp = cos_np(G, G_bptt)
        c_on = cos_np(G_online, G_bptt)
        gap = max(1.0 - c_on, 1e-12)
        out_rows.append(dict(
            cos=c_comp, rel_err=relerr_np(G, G_bptt),
            norm_ratio=float(np.linalg.norm(G)
                             / (np.linalg.norm(G_bptt) + 1e-300)),
            cos_online=c_on,
            improvement_over_online=c_comp - c_on,
            frac_gap_recovered=float((c_comp - c_on) / gap)))
    return out_rows


def median_of(rows, key):
    return float(np.median([r[key] for r in rows]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("Phase B1: compression ladder (C0-C3), L=2, N=%d, T=%d, BATCH=%d"
          % (N, T, BATCH))
    print("=" * 78)

    # ---- collect data
    train_fit_rows, train_eval_rows = [], {}
    for s in TRAIN_SEEDS:
        _, rows = collect(s, N_FIT_TRAJ + N_EVAL_TRAJ)
        train_fit_rows += rows[:N_FIT_TRAJ]
        train_eval_rows[s] = rows[N_FIT_TRAJ:]
    val_rows = {s: collect(s, N_TEST_TRAJ, traj_offset=500)[1]
               for s in VAL_SEEDS}
    test_rows = {}
    test_params = {}
    for s in TEST_SEEDS:
        params, rows = collect(s, N_TEST_TRAJ, traj_offset=500)
        test_rows[s] = rows
        test_params[s] = params

    fit_data = stack(train_fit_rows)

    # ---- C0: online baseline (no fitting)
    c0_test = {s: [dict(cos=cos_np(r["G_online"], r["G_bptt"]),
                        rel_err=relerr_np(r["G_online"], r["G_bptt"]),
                        norm_ratio=float(np.linalg.norm(r["G_online"])
                                         / (np.linalg.norm(r["G_bptt"])
                                            + 1e-300)))
                    for r in test_rows[s]] for s in TEST_SEEDS}
    print("[C0 online] test median cos:",
          round(median_of(sum(c0_test.values(), []), "cos"), 4))

    # ---- C1: complex-linear, gradient-weighted fit
    raw_c1, hist_c1 = fit_c1c2(c1_unpack, 4, fit_data, seed_key=1)
    c1_train_eval = {s: evaluate(c1_unpack, raw_c1, train_eval_rows[s])
                     for s in TRAIN_SEEDS}
    c1_val = {s: evaluate(c1_unpack, raw_c1, val_rows[s]) for s in VAL_SEEDS}
    c1_test = {s: evaluate(c1_unpack, raw_c1, test_rows[s])
              for s in TEST_SEEDS}
    print("[C1 complex-linear] fit loss history:", hist_c1)
    print("[C1] test median cos:",
          round(median_of(sum(c1_test.values(), []), "cos"), 4),
          " frac_gap_recovered:",
          round(median_of(sum(c1_test.values(), []),
                          "frac_gap_recovered"), 4))

    # ---- C1, state-MSE control: target = |B1|^2-weighted projection of
    # the true P channel onto a single aggregate channel (see PHASE_B1.md
    # for the rationale; this is an *informative*, non-primary control).
    state_targets = []
    for r in train_fit_rows:
        B1 = r["B1"]                       # (N1, N0)
        P = r["P"]                          # (T, BATCH, N1, N0)
        w = (np.abs(B1) ** 2)
        w = w / (w.sum(axis=0, keepdims=True) + 1e-30)   # (N1, N0)
        target = np.einsum("jm,tbjm->tbm", w, P)
        state_targets.append(target)
    fit_data_mse = dict(fit_data)
    fit_data_mse["state_target"] = jnp.asarray(np.stack(state_targets))
    raw_c1_mse, hist_c1_mse = fit_c1c2(c1_unpack, 4, fit_data_mse,
                                       seed_key=2, loss_kind="state_mse")
    c1_mse_test = {s: evaluate(c1_unpack, raw_c1_mse, test_rows[s])
                  for s in TEST_SEEDS}
    print("[C1 state-MSE control] fit loss history:", hist_c1_mse)
    print("[C1 state-MSE] test median cos:",
          round(median_of(sum(c1_mse_test.values(), []), "cos"), 4))

    # ---- C2: widely-linear, gradient-weighted fit
    raw_c2, hist_c2 = fit_c1c2(c2_unpack, 10, fit_data, seed_key=3)
    c2_train_eval = {s: evaluate(c2_unpack, raw_c2, train_eval_rows[s])
                     for s in TRAIN_SEEDS}
    c2_val = {s: evaluate(c2_unpack, raw_c2, val_rows[s]) for s in VAL_SEEDS}
    c2_test = {s: evaluate(c2_unpack, raw_c2, test_rows[s])
              for s in TEST_SEEDS}
    print("[C2 widely-linear] fit loss history:", hist_c2)
    print("[C2] test median cos:",
          round(median_of(sum(c2_test.values(), []), "cos"), 4),
          " frac_gap_recovered:",
          round(median_of(sum(c2_test.values(), []),
                          "frac_gap_recovered"), 4))

    # ---- C3: positive control (exact analytic parameters, per test seed
    # -- pole/weights depend on that seed's OWN a1/B1, so this is NOT a
    # cross-seed generalization claim, it is a same-architecture sanity
    # check that the model CLASS contains the exact solution).
    c3_positive_control = {}
    for s in TEST_SEEDS:
        r0 = test_rows[s][0]
        raw_exact = c3_exact_raw(np.asarray(r0["a1"]), np.asarray(r0["B1"]))
        c3_positive_control[s] = evaluate(None, raw_exact, test_rows[s],
                                          use_c3=True)
    flat_pc = sum(c3_positive_control.values(), [])
    print("[C3 positive control] median cos:",
          round(median_of(flat_pc, "cos"), 6),
          " median rel_err:", round(median_of(flat_pc, "rel_err"), 8))

    # ---- C3: optional free fit (informative only; per B1, optimizer
    # failure here is not evidence against the representation)
    raw_c3_init = jnp.zeros((N, N, 8))
    raw_c3_fit, hist_c3 = None, None
    try:
        key = jax.random.PRNGKey(7)
        raw = 0.05 * jax.random.normal(key, (N, N, 8))
        opt = optax.adam(LR)
        opt_state = opt.init(raw)
        q1_stack = fit_data["q1"]

        def loss_c3(raw):
            G = batched_c3_gradient(raw, fit_data["Sa0"], q1_stack)
            c = jax.vmap(cos_jax)(G, fit_data["G_bptt"])
            return jnp.mean(1.0 - c)

        @jax.jit
        def step3(raw, opt_state):
            loss, grad = jax.value_and_grad(loss_c3)(raw)
            updates, opt_state = opt.update(grad, opt_state)
            raw = optax.apply_updates(raw, updates)
            return raw, opt_state, loss

        hist_c3 = []
        for i in range(STEPS):
            raw, opt_state, loss = step3(raw, opt_state)
            if i % 100 == 0 or i == STEPS - 1:
                hist_c3.append(float(loss))
        raw_c3_fit = raw
    except Exception as e:                          # pragma: no cover
        hist_c3 = [f"FIT_FAILED: {e}"]
    c3_fit_test = None
    if raw_c3_fit is not None:
        c3_fit_test = {s: evaluate(None, raw_c3_fit, test_rows[s],
                                   use_c3=True) for s in TEST_SEEDS}
        print("[C3 free fit] test median cos:",
              round(median_of(sum(c3_fit_test.values(), []), "cos"), 4))

    # ---- diagnostic: WITHIN-SEED fit/eval (fit C1/C2 on 2 trajectories
    # of a SINGLE fixed architecture, evaluate on 1 held-out trajectory of
    # the SAME architecture -- isolates "can the model class represent
    # this at all for one network" from "does a pooled cross-seed fit
    # transfer". Not a violation of B1.3 (this is a clearly separate,
    # clearly labeled diagnostic, not the headline generalization claim).
    within_seed_diag = {}
    diag_seeds = TEST_SEEDS[:4]
    for s in diag_seeds:
        params, rows = collect(s, 3, traj_offset=90000)
        fit_rows, held_rows = rows[:2], rows[2:]
        fdat = stack(fit_rows)
        raw1, _ = fit_c1c2(c1_unpack, 4, fdat, seed_key=100 + s)
        raw2, _ = fit_c1c2(c2_unpack, 10, fdat, seed_key=200 + s)
        within_seed_diag[s] = dict(
            C1=evaluate(c1_unpack, raw1, held_rows),
            C2=evaluate(c2_unpack, raw2, held_rows),
            C0=[dict(cos=cos_np(r["G_online"], r["G_bptt"]),
                     rel_err=relerr_np(r["G_online"], r["G_bptt"]))
                for r in held_rows])
        print(f"[within-seed diag] seed {s}: "
              f"C0={within_seed_diag[s]['C0'][0]['cos']:.3f}  "
              f"C1={within_seed_diag[s]['C1'][0]['cos']:.3f}  "
              f"C2={within_seed_diag[s]['C2'][0]['cos']:.3f}")

    # ---- assemble report
    def summarize(d):
        flat = sum(d.values(), [])
        return dict(median_cos=median_of(flat, "cos"),
                   median_rel_err=median_of(flat, "rel_err"),
                   median_norm_ratio=median_of(flat, "norm_ratio"),
                   per_seed={str(s): d[s] for s in d})

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(
        git=git,
        config=dict(N=N, T=T, BATCH=BATCH, steps=STEPS, lr=LR,
                   train_seeds=TRAIN_SEEDS, val_seeds=VAL_SEEDS,
                   test_seeds=TEST_SEEDS, n_fit_traj=N_FIT_TRAJ,
                   n_eval_traj=N_EVAL_TRAJ, n_test_traj=N_TEST_TRAJ),
        param_counts_per_mode=dict(C0=0, C1=4, C2=10,
                                   C3_per_jm_pair=8,
                                   C3_total="8 * N^2 (or 0 fitted, "
                                            "positive control only)"),
        C0_online_test=summarize(c0_test),
        C1_test=summarize(c1_test),
        C1_train_seed_heldout_traj=summarize(c1_train_eval),
        C1_val=summarize(c1_val),
        C1_state_mse_control_test=summarize(c1_mse_test),
        C1_fit_loss_history=hist_c1,
        C1_state_mse_fit_loss_history=hist_c1_mse,
        C2_test=summarize(c2_test),
        C2_train_seed_heldout_traj=summarize(c2_train_eval),
        C2_val=summarize(c2_val),
        C2_fit_loss_history=hist_c2,
        C3_positive_control=summarize(c3_positive_control),
        C3_free_fit_test=(summarize(c3_fit_test)
                          if c3_fit_test is not None else None),
        C3_free_fit_loss_history=hist_c3,
        within_seed_diagnostic={str(s): within_seed_diag[s]
                               for s in within_seed_diag},
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b1_compression_ladder_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print("-" * 78)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
