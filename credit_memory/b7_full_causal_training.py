"""B7C-E: end-to-end training with the full exact Phase-A causal (P/Q)
credit system (credit_memory/full_causal.py, no compression, no
selection -- verified exact against BPTT in b7_verify_exact.py), compared
against online, the best-supported B6 rank-1 arm (T2: reactive EMA +
hysteresis, no prospective extrapolation), and BPTT itself.

No new theory, no new approximation, no prospective predictor, no new
selector idea, no S5 launch, no hyperparameter search.

Run:  python -m credit_memory.b7_full_causal_training
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.full_causal import full_causal_gradient
from credit_memory.b5_train import (set_config, draw_task_batch, loss_of,
                                    L, N, T, DELAY, BATCH, LR, N_CAL_TRAJ,
                                    CHECKPOINTS)
from credit_memory.b6_prospective_tracking import (
    causal_prefix_selection, single_batch_observation, hysteretic_select,
    T2_GAMMA, HYSTERESIS_MARGIN)
from credit_memory.b5_1_action_utility import adam_step

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b7")

SEEDS = list(range(8))
STEPS = 600


def cos_np(u, v):
    return float(np.abs(np.vdot(v, u))
                / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def relerr_np(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-300))


def block_slices(params):
    idx = 0
    slices = {}
    for l in range(L):
        slices[f"a{l}"] = (idx, idx + 2 * N)
        idx += 2 * N
        m = params["b"][l].size
        slices[f"b{l}"] = (idx, idx + 2 * m)
        idx += 2 * m
    slices["c"] = (idx, idx + 2 * N)
    idx += 2 * N
    return slices, idx


def train(arm, seed, clip):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)
    diag_rng = np.random.RandomState(55555 + seed)

    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)

    top_j_by_mode = None
    rho_cur = None
    if arm == "a1_rank1":
        rho_cur, top_j_by_mode = causal_prefix_selection(params, cal_rng,
                                                          f_diag)

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    losses, diagnostics = [], []
    finite = True

    for step in range(1, STEPS + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)

        if arm == "online":
            G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        elif arm == "bptt":
            lam = tcg.exact_lambda(params, q)
            G = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        elif arm == "a1_rank1":
            for m in range(N):
                c_m = build_c_t(q[1], B1[:, m])
                r_obs = single_batch_observation(f_diag, Sa[0][:, :, m],
                                                 c_m)
                rho_cur[m] = (1 - T2_GAMMA) * rho_cur[m] + T2_GAMMA * r_obs
                new_sel, _ = hysteretic_select(rho_cur[m],
                                               top_j_by_mode.get(m),
                                               HYSTERESIS_MARGIN)
                top_j_by_mode[m] = new_sel
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                          q[1], Sa[0], Sb[0])
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        elif arm == "a2_full_causal":
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = full_causal_gradient(a1, B1, N, q[1], Sa[0], Sb[0])
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:], c=G_online["c"])
        else:
            raise ValueError(arm)

        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if clip > 0 and nrm > clip:
            g = g * (clip / nrm)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]
        f_diag = build_F(a1)

        losses.append(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(g)):
            finite = False
            break

        if step in CHECKPOINTS or step == 1:
            x_d, y_d = draw_task_batch(diag_rng)
            _, h_d, r_d = loss_of(params, x_d, y_d)
            q_d = tcg.spatial_q(params, h_d, r_d)
            Sa_d, Sb_d = tcg.sensitivities(params, h_d, x_d)
            lam_d = tcg.exact_lambda(params, q_d)
            G_bptt_d = tcg.assemble(params, h_d, x_d, r_d, lam_d, Sa_d,
                                    Sb_d, direct=True)
            g_bptt_d = tcg.flat_grads(G_bptt_d, params)

            if arm == "online":
                G_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d, Sb_d)
            elif arm == "bptt":
                G_d = G_bptt_d
            elif arm == "a1_rank1":
                G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d,
                                          Sb_d)
                Ga0_d, Gb0_d = b4_layer0_gradient(f_diag, top_j_by_mode,
                                                  B1, N, q_d[1], Sa_d[0],
                                                  Sb_d[0])
                G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:],
                          c=G_online_d["c"])
            elif arm == "a2_full_causal":
                G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d,
                                          Sb_d)
                Ga0_d, Gb0_d = full_causal_gradient(a1, B1, N, q_d[1],
                                                    Sa_d[0], Sb_d[0])
                G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:],
                          c=G_online_d["c"])
            g_train_d = tcg.flat_grads(G_d, params)

            slices, _ = block_slices(params)
            def sl(vec, key):
                a, b = slices[key]
                return vec[a:b]

            diagnostics.append(dict(
                step=step,
                cos_whole=cos_np(g_train_d, g_bptt_d),
                rel_err_whole=relerr_np(g_train_d, g_bptt_d),
                cos_a0=cos_np(sl(g_train_d, "a0"), sl(g_bptt_d, "a0")),
                cos_b0=cos_np(sl(g_train_d, "b0"), sl(g_bptt_d, "b0"))))

    return dict(arm=arm, seed=seed, clip=clip, finite=finite,
               steps_run=len(losses),
               final_loss=float(losses[-1]) if losses else None,
               best_loss=float(np.min(losses)) if losses else None,
               median_late_loss=float(np.median(losses[-100:]))
               if len(losses) >= 100 else
               (float(np.median(losses)) if losses else None),
               losses=losses, diagnostics=diagnostics)


def main() -> None:
    print("=" * 90)
    print("Phase B7C-E: online / rank-1 A-CCM / full-causal CCM / BPTT")
    print("=" * 90)

    arms = ["online", "a1_rank1", "a2_full_causal", "bptt"]
    all_runs = []
    for clip in [0.0, 1.0]:
        for arm in arms:
            for seed in SEEDS:
                out = train(arm, seed, clip)
                all_runs.append(out)
                d600 = next((d for d in out["diagnostics"]
                            if d["step"] == 600), None)
                print(f"[{arm} s{seed} clip{clip}] final_loss="
                      f"{out['final_loss']:.4f}  "
                      f"cos_whole@600={d600['cos_whole'] if d600 else float('nan'):.6f}"
                      f"  finite={out['finite']}")

    print("-" * 90)
    for clip in [0.0, 1.0]:
        print(f"-- clip={clip} --")
        for arm in arms:
            rows = [r for r in all_runs if r["arm"] == arm
                   and r["clip"] == clip and r["finite"]]
            if not rows:
                continue
            finals = [r["final_loss"] for r in rows]
            print(f"  {arm:16s}: median final_loss={np.median(finals):.4f}"
                  f"  n_finite={len(rows)}/{len(SEEDS)}")
            for step in CHECKPOINTS:
                cs = [d["cos_whole"] for r in rows for d in r["diagnostics"]
                     if d["step"] == step]
                if cs:
                    print(f"      step {step}: median cos_whole vs BPTT="
                          f"{np.median(cs):.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(L=L, N=N, T=T, DELAY=DELAY, BATCH=BATCH,
                         steps=STEPS, lr=LR, seeds=SEEDS,
                         checkpoints=CHECKPOINTS),
              runs=all_runs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b7_full_causal_training_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
