"""B3B -- causal-teacher rank-1 diagnostic.

Can a good rank-1 direction be found without BPTT, if temporarily
allowed to see the exact forward P/Q teacher state x_t on calibration?
x_t is BPTT-free (P,Q are built only from Sa0 and a1 -- forward-only, no
reverse-time pass); this is what makes it a legitimate "no BPTT" input,
distinct from G_bptt (which needs exact_lambda's reverse pass, used
here only for held-out evaluation).

Model: x_hat_t = v * z_t,  z_t = beta z_{t-1} + u_t (u_t = Sa0_t[m]),
v in C^{2N} a single fixed rank-1 embedding direction, beta a single
complex pole (|beta|<1). This is a strict SUBSET of B2D's L3 r=1 oracle
family: L3 allowed fully free q1-readout weights w1,w2 (2N complex each,
unconstrained); here the readout is tied to c_t via a single v (Ghat =
sum_t c_t^dagger v z_t = v . rho_beta, rho_beta[p] := sum_t conj(c_t[p])
z_t), i.e. exactly a genuine rank-1 subspace of the ORIGINAL (P,Q)
coordinates, not an arbitrary bilinear form in q1.

Primary objective (as specified): normalized squared error of the
POOLED (t,batch) contraction, |Ghat - G|^2 / |G|^2 -- NOT a cosine loss,
though cosine is still reported for comparability with the rest of the
ladder. Fit on calibration trajectories only (no BPTT in the loss);
BPTT is evaluation-only, on held-out test trajectories.

Run:  python -m credit_memory.phase_b3b_teacher_rank1
"""
from __future__ import annotations

import json
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax

from credit_memory.hankel import build_F, build_c_t
from credit_memory.phase_b2bc_hankel_truncation import (
    N, T, BATCH, SEEDS, N_CAL_TRAJ, N_TEST_TRAJ, collect_rows, cos_np,
    relerr_np)

jax.config.update("jax_enable_x64", True)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

STEPS = 800
LR = 3e-2


def sig(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def unpack(raw):
    """raw: (2N+2,) -> beta (complex scalar), v (2N,) complex."""
    rho, theta = raw[0], raw[1]
    beta = sig(rho) * jnp.exp(1j * theta)
    v = raw[2:2 + 2 * N] + 1j * raw[2 + 2 * N:2 + 4 * N]
    return beta, v


def teacher_rank1_gradient(raw, u_t, c_t):
    """u_t: (T,BATCH) complex, c_t: (T,BATCH,2N) complex -> Ghat scalar."""
    beta, v = unpack(raw)

    def step(z_prev, u):
        z_t = beta * z_prev + u
        return z_t, z_t

    z0 = jnp.zeros((u_t.shape[1],), jnp.complex128)
    _, z = jax.lax.scan(step, z0, u_t)                     # (T,BATCH)
    xhat = v[None, None, :] * z[:, :, None]                 # (T,BATCH,2N)
    g_t = jnp.sum(jnp.conj(c_t) * xhat, axis=-1)
    return jnp.sum(g_t)


def fit(cal_rows, m, seed_key):
    u_stack = jnp.asarray(np.stack([row["Sa0"][:, :, m] for row in cal_rows]))
    c_stack = jnp.asarray(np.stack(
        [build_c_t(row["q1"], row["B1"][:, m]) for row in cal_rows]))
    G_stack = jnp.asarray(np.stack([row["G_causal"][m] for row in cal_rows]))

    key = jax.random.PRNGKey(seed_key)
    raw = 0.05 * jax.random.normal(key, (2 + 4 * N,))
    opt = optax.adam(LR)
    opt_state = opt.init(raw)

    def loss_fn(raw):
        def one(u_t, c_t, G):
            Ghat = teacher_rank1_gradient(raw, u_t, c_t)
            return jnp.abs(Ghat - G) ** 2 / (jnp.abs(G) ** 2 + 1e-30)
        return jnp.mean(jax.vmap(one)(u_stack, c_stack, G_stack))

    @jax.jit
    def step(raw, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(raw)
        updates, opt_state = opt.update(grad, opt_state)
        raw = optax.apply_updates(raw, updates)
        return raw, opt_state, loss

    history = []
    for i in range(STEPS):
        raw, opt_state, loss = step(raw, opt_state)
        if i % 200 == 0 or i == STEPS - 1:
            history.append(float(loss))
    return raw, history


def evaluate(raw_by_mode, test_rows):
    rows = []
    for row in test_rows:
        G_hat = np.zeros(N, np.complex128)
        for m in range(N):
            u_t = jnp.asarray(row["Sa0"][:, :, m])
            c_t = jnp.asarray(build_c_t(row["q1"], row["B1"][:, m]))
            G_hat[m] = np.asarray(teacher_rank1_gradient(raw_by_mode[m],
                                                          u_t, c_t))
        G_bptt, G_online = row["G_bptt"], row["G_online"]
        c_hat, c_on = cos_np(G_hat, G_bptt), cos_np(G_online, G_bptt)
        gap = max(1.0 - c_on, 1e-12)
        rows.append(dict(cos=c_hat, cos_online=c_on,
                         rel_err=relerr_np(G_hat, G_bptt),
                         norm_ratio=float(np.linalg.norm(G_hat)
                                          / (np.linalg.norm(G_bptt)
                                             + 1e-300)),
                         frac_gap_recovered=float((c_hat - c_on) / gap)))
    return rows


def main() -> None:
    print("=" * 78)
    print(f"Phase B3B: causal-teacher rank-1 diagnostic, {len(SEEDS)} seeds")
    print("=" * 78)

    all_test_rows = {}
    all_histories = {}
    for seed in SEEDS:
        _, cal_rows = collect_rows(seed, N_CAL_TRAJ, offset=0)
        _, test_rows = collect_rows(seed, N_TEST_TRAJ, offset=9000)
        raw_by_mode = {}
        hist_by_mode = {}
        for m in range(N):
            raw, hist = fit(cal_rows, m, seed_key=2000 + seed * 10 + m)
            raw_by_mode[m] = raw
            hist_by_mode[m] = hist
        rows = evaluate(raw_by_mode, test_rows)
        all_test_rows[seed] = rows
        all_histories[seed] = hist_by_mode
        print(f"seed {seed}: median cos={np.median([x['cos'] for x in rows]):.4f}"
              f"  median frac_gap_recovered="
              f"{np.median([x['frac_gap_recovered'] for x in rows]):.4f}")

    flat = sum(all_test_rows.values(), [])
    med_cos = float(np.median([x["cos"] for x in flat]))
    med_frac = float(np.median([x["frac_gap_recovered"] for x in flat]))
    print("-" * 78)
    print(f"TEST median cos: {med_cos:.4f}  median frac_gap_recovered: "
          f"{med_frac:.4f}")
    verdict = ("reaches ~0.90-0.94 -> rank-1 IS causally identifiable "
              "from the forward teacher (B3B passes)"
              if med_cos >= 0.88 else
              "fails badly -> the exact-gradient oracle exploits "
              "information not present in the causal teacher projection "
              "(B3B fails)")
    print(f"verdict: {verdict}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(N=N, T=T, BATCH=BATCH, seeds=SEEDS,
                         n_cal_traj=N_CAL_TRAJ, n_test_traj=N_TEST_TRAJ,
                         steps=STEPS, lr=LR),
              per_seed_test_rows=all_test_rows,
              fit_loss_histories=all_histories,
              median_cos=med_cos, median_frac_gap_recovered=med_frac,
              verdict=verdict)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "phase_b3b_teacher_rank1_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
