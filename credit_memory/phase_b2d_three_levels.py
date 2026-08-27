"""B2D -- three levels of information for the reduced-order system, at
the same r-ladder, same seeds, same disjoint test trajectories as
phase_b2bc_hankel_truncation.py (so all four columns -- C0, L1, L2, L3
-- are directly comparable):

  L1 architecture only:  Wc analytic (as always) + Wo from an isotropic
                          prior S=I (no calibration data at all).
  L2 causal calibration: Wc analytic + Wo from calibration-estimated S
                          (this IS phase_b2bc_hankel_truncation.py's
                          result; reused here, not recomputed, for an
                          exact apples-to-apples row).
  L3 exact-gradient oracle: r free complex-linear channels per mode,
                          full q1-based (not q0-collapsed) readout --
                          strictly richer than L1/L2's balanced-truncated
                          family (contains it as a special case), fit by
                          gradient descent against BPTT on the SAME
                          calibration trajectories, tested on the SAME
                          held-out trajectories. Upper-bound capacity
                          test only -- optimizer failure here is not
                          evidence against the representation (same
                          caveat as B1's C3).

Run:  python -m credit_memory.phase_b2d_three_levels
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
from credit_memory.hankel import (build_F, analytic_Wc, estimate_S,
                                  solve_Wo, balanced_transform,
                                  reduced_system, reduced_gradient)
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, R_LADDER, cos_np,
    relerr_np, collect_rows)

jax.config.update("jax_enable_x64", True)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

STEPS = 600
LR = 3e-2


# ---------------------------------------------------------------------------
# L1: architecture-only (isotropic Wo prior, S=I, no data)
# ---------------------------------------------------------------------------

def l1_architecture_only(a1, B1, test_rows, r_ladder):
    f_diag = build_F(a1)
    Wc = analytic_Wc(f_diag)
    S_iso = np.eye(2 * N, dtype=np.complex128)
    Wo = solve_Wo(f_diag, S_iso)
    per_mode = {}
    for m in range(N):
        T_bal, Tinv_bal, _ = balanced_transform(Wc, Wo)
        per_mode[m] = dict(T_bal=T_bal, Tinv_bal=Tinv_bal)
    out = {}
    for r in r_ladder:
        F_r_d_r = {m: reduced_system(f_diag, np.ones(2 * N, np.complex128),
                                     per_mode[m]["T_bal"],
                                     per_mode[m]["Tinv_bal"], r)
                  for m in range(N)}
        rows = []
        for r_idx, row in enumerate(test_rows):
            G_hat = np.zeros(N, np.complex128)
            for m in range(N):
                F_r, d_r = F_r_d_r[m]
                _, G_hat[m] = reduced_gradient(
                    F_r, d_r, per_mode[m]["T_bal"], row["Sa0"][:, :, m],
                    row["q1"], B1[:, m], r)
            G_bptt, G_online = row["G_bptt"], row["G_online"]
            c_hat, c_on = cos_np(G_hat, G_bptt), cos_np(G_online, G_bptt)
            gap = max(1.0 - c_on, 1e-12)
            rows.append(dict(cos=c_hat, cos_online=c_on,
                             rel_err=relerr_np(G_hat, G_bptt),
                             frac_gap_recovered=float((c_hat - c_on)
                                                      / gap)))
        out[r] = rows
    return out


# ---------------------------------------------------------------------------
# L3: exact-gradient oracle -- r free complex-linear channels/mode, full
# q1-based readout (upper bound; strictly richer than the balanced-
# truncated family)
# ---------------------------------------------------------------------------

def sig(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def unpack_oracle(raw, r):
    """raw: (N, r*(1+4N)) -> beta (N,r) complex, w1,w2 (N,r,N) complex."""
    idx = 0
    rho = raw[:, idx:idx + r]; idx += r
    theta = raw[:, idx:idx + r]; idx += r
    beta = sig(rho) * jnp.exp(1j * theta)
    w1r = raw[:, idx:idx + r * N].reshape(N, r, N); idx += r * N
    w1i = raw[:, idx:idx + r * N].reshape(N, r, N); idx += r * N
    w2r = raw[:, idx:idx + r * N].reshape(N, r, N); idx += r * N
    w2i = raw[:, idx:idx + r * N].reshape(N, r, N); idx += r * N
    return beta, w1r + 1j * w1i, w2r + 1j * w2i


def oracle_gradient(raw, r, Sa0_all, q1):
    """Sa0_all: (T,BATCH,N) [drive per mode m = Sa0_all[:,:,m]];
    q1: (T,BATCH,N). Returns G: (N,) complex."""
    beta, w1, w2 = unpack_oracle(raw, r)   # (N,r), (N,r,N), (N,r,N)

    def per_mode(m, beta_m, w1_m, w2_m):
        def step(z_prev, Sa0_t):
            z_t = beta_m[None, :] * z_prev + Sa0_t[:, None]
            return z_t, z_t
        z0 = jnp.zeros((Sa0_all.shape[1], r), jnp.complex128)
        _, z = jax.lax.scan(step, z0, Sa0_all[:, :, m])   # (T,BATCH,r)
        c1 = jnp.einsum("ij,tbj->tbi", w1_m, jnp.conj(q1))  # (T,BATCH,r)
        c2 = jnp.einsum("ij,tbj->tbi", w2_m, q1)             # (T,BATCH,r)
        g_t = jnp.sum(c1 * z + c2 * jnp.conj(z), axis=-1)    # (T,BATCH)
        return jnp.sum(g_t)

    G = jnp.stack([per_mode(m, beta[m], w1[m], w2[m]) for m in range(N)])
    return G


def fit_oracle(r, cal_rows, seed_key):
    Sa0_stack = jnp.asarray(np.stack([row["Sa0"] for row in cal_rows]))
    q1_stack = jnp.asarray(np.stack([row["q1"] for row in cal_rows]))
    G_bptt_stack = jnp.asarray(np.stack([row["G_bptt"] for row in cal_rows]))

    n_raw = r * (2 + 4 * N)   # rho, theta (r each) + w1r,w1i,w2r,w2i (r*N each)
    key = jax.random.PRNGKey(seed_key)
    raw = 0.05 * jax.random.normal(key, (N, n_raw))
    opt = optax.adam(LR)
    opt_state = opt.init(raw)

    def loss_fn(raw):
        def one(Sa0, q1, G_bptt):
            G = oracle_gradient(raw, r, Sa0, q1)
            c = jnp.abs(jnp.vdot(G_bptt, G)) / (jnp.linalg.norm(G)
                                                * jnp.linalg.norm(G_bptt)
                                                + 1e-300)
            return 1.0 - c
        return jnp.mean(jax.vmap(one)(Sa0_stack, q1_stack, G_bptt_stack))

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


def l3_exact_gradient_oracle(cal_rows, test_rows, r_ladder, seed):
    out, histories = {}, {}
    for r in r_ladder:
        if r > 2 * N:
            continue
        raw, hist = fit_oracle(r, cal_rows, seed_key=1000 * seed + r)
        histories[r] = hist
        rows = []
        for row in test_rows:
            G_hat = np.asarray(oracle_gradient(
                raw, r, jnp.asarray(row["Sa0"]), jnp.asarray(row["q1"])))
            G_bptt, G_online = row["G_bptt"], row["G_online"]
            c_hat, c_on = cos_np(G_hat, G_bptt), cos_np(G_online, G_bptt)
            gap = max(1.0 - c_on, 1e-12)
            rows.append(dict(cos=c_hat, cos_online=c_on,
                             rel_err=relerr_np(G_hat, G_bptt),
                             frac_gap_recovered=float((c_hat - c_on)
                                                      / gap)))
        out[r] = rows
    return out, histories


def main() -> None:
    print("=" * 78)
    print("Phase B2D: three information levels, same seeds/splits as B2B/C")
    print("=" * 78)

    l1_all, l3_all, l3_hist_all = {}, {}, {}
    r_ladder_l3 = [1, 2, 4]     # oracle fit cost grows fast with r; kept
                                 # to the informative small-r range
    for seed in SEEDS:
        params, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        a1, B1 = cal_rows[0]["a1"], cal_rows[0]["B1"]

        l1 = l1_architecture_only(a1, B1, test_rows, R_LADDER)
        for r, rows in l1.items():
            l1_all.setdefault(r, []).extend(rows)

        l3, hist = l3_exact_gradient_oracle(cal_rows, test_rows,
                                            r_ladder_l3, seed)
        for r, rows in l3.items():
            l3_all.setdefault(r, []).extend(rows)
        l3_hist_all[seed] = hist
        print(f"seed {seed}: L1 r=4 cos={np.median([x['cos'] for x in l1[4]]):.3f}"
              f"  L3 r=1 cos={np.median([x['cos'] for x in l3.get(1, [dict(cos=float('nan'))])]):.3f}"
              f"  L3 r=4 cos={np.median([x['cos'] for x in l3.get(4, [dict(cos=float('nan'))])]):.3f}")

    print("-" * 78)
    print("L1 architecture-only (isotropic prior):")
    for r in R_LADDER:
        rows = l1_all[r]
        print(f"  r={r:>2d}: median cos={np.median([x['cos'] for x in rows]):.4f}"
              f"  median frac_gap_recovered="
              f"{np.median([x['frac_gap_recovered'] for x in rows]):.4f}")
    print("L3 exact-gradient oracle (upper bound):")
    for r in r_ladder_l3:
        rows = l3_all[r]
        print(f"  r={r:>2d}: median cos={np.median([x['cos'] for x in rows]):.4f}"
              f"  median frac_gap_recovered="
              f"{np.median([x['frac_gap_recovered'] for x in rows]):.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(
        git=git,
        config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                   n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ,
                   r_ladder=R_LADDER, r_ladder_l3=r_ladder_l3,
                   oracle_steps=STEPS, oracle_lr=LR),
        L1_architecture_only={
            str(r): dict(
                median_cos=float(np.median([x["cos"] for x in rows])),
                median_rel_err=float(np.median([x["rel_err"]
                                               for x in rows])),
                median_frac_gap_recovered=float(np.median(
                    [x["frac_gap_recovered"] for x in rows])),
                rows=rows)
            for r, rows in l1_all.items()},
        L3_exact_gradient_oracle={
            str(r): dict(
                median_cos=float(np.median([x["cos"] for x in rows])),
                median_rel_err=float(np.median([x["rel_err"]
                                               for x in rows])),
                median_frac_gap_recovered=float(np.median(
                    [x["frac_gap_recovered"] for x in rows])),
                rows=rows)
            for r, rows in l3_all.items()},
        L3_fit_loss_histories=l3_hist_all,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_b2d_three_levels_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
