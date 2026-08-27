"""B5: small end-to-end causal training validation.

Trains the frozen L=2 toy rig (toyrig.ssm_rig, same convention as
Phase A/B1-B4) on the delayed continuous-copy task with four possible
layer-0 gradient rules ("arms"):

  A0  online       existing scalable online rule, unchanged (err=q[0])
  A1  b4_arch      B4 rank-1 correction, channel selected from
                    architecture only (routing-weighted controllability
                    score |B1[j,m]|^2/(1-|a1[j]|^2)), frozen for all of
                    training, no data/BPTT ever used to pick it
  A2  b4_causal     B4 rank-1 correction, channel selected via a short
                    causal calibration prefix (streaming estimator,
                    credit_memory/streaming.py; no parameter updates
                    during calibration, no BPTT), frozen for the rest of
                    training -- PRIMARY arm
  bptt              exact BPTT teacher (evaluation/performance reference
                    only; never informs A1/A2)

Layers >=1 and the readout "c" always use the unchanged online rule in
every arm (Null-1: the online rule is already exact there; B3/B4 never
found or claimed a defect above layer 0).

No S5, no RoutePC, no Meta-Adam, no prospective coding, no exact/BPTT
information inside A0/A1/A2's training update (bptt arm is a separate,
clearly-labeled reference arm).

Run:  python -m credit_memory.b5_train --arm b4_causal --seed 0 --clip 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F
from credit_memory.streaming import StreamingRelevance
from credit_memory.b4_deploy import b4_layer0_gradient

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b5")

# ---------------------------------------------------------------------------
# fixed small-pilot config (deliberately small, per B5's own framing)
# ---------------------------------------------------------------------------
L, N, T, DELAY, BATCH = 2, 6, 60, 20, 8
STEPS = 600
LR = 1e-3
N_CAL_TRAJ = 4                    # matches B1-B4's calibration convention
CHECKPOINTS = [0, 100, 300, 600]  # gradient-mechanism probe steps


def set_config():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = L, N, T, DELAY, BATCH


def draw_task_batch(rng):
    x = rng.randn(T, BATCH)
    y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
    return x, y


def loss_of(params, x, y):
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:DELAY] = 0.0
    return 0.5 * float(np.mean(r ** 2)), h, r


def architecture_only_selector(a1, B1):
    """A1: routing-weighted controllability score, zero data, zero BPTT.
    P-type (pole a1[j]) and Q-type (pole conj(a1[j])) channels have
    identical |pole| and |weight| magnitude for the same j (the score is
    symmetric under conjugation), so ties are broken by always preferring
    the P-type index -- an arbitrary but fully deterministic, documented
    choice, not a data-dependent one."""
    score = np.abs(B1) ** 2 / (1.0 - np.abs(a1[:, None]) ** 2)  # (N,N) [j,m]
    return {m: int(np.argmax(score[:, m])) for m in range(N)}   # P-type (<N)


def causal_calibration_selector(params, cal_rng, f_diag):
    """A2 primary protocol: short causal prefix, NO parameter updates,
    streaming estimator only (credit_memory/streaming.py), no BPTT."""
    estimators = {m: StreamingRelevance(f_diag, BATCH, mode="windowed")
                 for m in range(N)}
    from credit_memory.hankel import build_c_t
    for _ in range(N_CAL_TRAJ):
        x, y = draw_task_batch(cal_rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, _ = tcg.sensitivities(params, h, x)
        for m in range(N):
            c_m = build_c_t(q[1], params["b"][1][:, m])
            for t in range(T):
                estimators[m].step(Sa[0][t, :, m], c_m[t])
    return {m: int(estimators[m].top_channel(1)[0]) for m in range(N)}


def train(arm, seed, clip, out_path=None):
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    cal_rng = np.random.RandomState(777 + seed)     # disjoint from
                                                      # training's own
                                                      # random stream
    diag_rng = np.random.RandomState(55555 + seed)   # disjoint diagnostic
                                                      # batches

    a1, B1 = params["a"][1], params["b"][1]
    f_diag = build_F(a1)

    selector_info = dict(kind=arm)
    top_j_by_mode = None
    if arm == "b4_arch":
        top_j_by_mode = architecture_only_selector(a1, B1)
        selector_info["top_j_by_mode"] = top_j_by_mode
    elif arm == "b4_causal":
        top_j_by_mode = causal_calibration_selector(params, cal_rng, f_diag)
        selector_info["top_j_by_mode"] = top_j_by_mode
        selector_info["n_cal_traj"] = N_CAL_TRAJ

    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    losses = []
    diagnostics = []
    finite = True
    clip_fires = 0
    t0 = time.time()

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
        elif arm in ("b4_arch", "b4_causal"):
            G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)
            Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, B1, N,
                                          q[1], Sa[0], Sb[0])
            G = dict(a=[Ga0] + G_online["a"][1:],
                     b=[Gb0] + G_online["b"][1:],
                     c=G_online["c"])
        else:
            raise ValueError(arm)

        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if clip > 0 and nrm > clip:
            g = g * (clip / nrm)
            clip_fires += 1
        m_ = b1_ * m_ + (1 - b1_) * g
        v_ = b2_ * v_ + (1 - b2_) * g ** 2
        flat = flat - LR * (m_ / (1 - b1_ ** step)) / (
            np.sqrt(v_ / (1 - b2_ ** step)) + eps)
        params = tcg.pack(params, flat)
        a1, B1 = params["a"][1], params["b"][1]     # refresh after update
        f_diag = build_F(a1)

        losses.append(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(g)):
            finite = False
            print(f"  [{arm} s{seed} clip{clip}] NON-FINITE at step {step}, "
                  f"stopping early")
            break

        if step in CHECKPOINTS or step == 1:
            # snapshot diagnostic: fresh disjoint batch, POST-update
            # params (this step's update already applied above) -- both
            # the arm's own gradient rule and the exact BPTT reference
            # are recomputed on the SAME batch/params for a genuine
            # apples-to-apples comparison. exact_lambda is called here
            # ONLY for this offline probe; it never touches `g`/the
            # actual parameter update above.
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
                G_d = G_bptt_d          # this arm IS the teacher
            elif arm in ("b4_arch", "b4_causal"):
                G_online_d = tcg.assemble(params, h_d, x_d, r_d, q_d, Sa_d,
                                          Sb_d)
                Ga0_d, Gb0_d = b4_layer0_gradient(
                    f_diag, top_j_by_mode, B1, N, q_d[1], Sa_d[0], Sb_d[0])
                G_d = dict(a=[Ga0_d] + G_online_d["a"][1:],
                          b=[Gb0_d] + G_online_d["b"][1:],
                          c=G_online_d["c"])
            g_train_d = tcg.flat_grads(G_d, params)

            cos_d = float(np.abs(np.vdot(g_bptt_d, g_train_d)) / (
                np.linalg.norm(g_train_d) * np.linalg.norm(g_bptt_d)
                + 1e-300))
            rel_d = float(np.linalg.norm(g_train_d - g_bptt_d)
                          / (np.linalg.norm(g_bptt_d) + 1e-300))
            diagnostics.append(dict(step=step, cos_train_vs_bptt=cos_d,
                                    rel_err_train_vs_bptt=rel_d))
            print(f"  [{arm} s{seed} clip{clip}] step {step}: loss "
                  f"{loss:.4f}  cos_vs_bptt {cos_d:.3f}")

    wall = time.time() - t0
    out = dict(arm=arm, seed=seed, clip=clip, steps_run=len(losses),
              losses=losses, finite=finite,
              p_clip=(clip_fires / len(losses)) if losses and clip > 0
              else 0.0,
              final_loss=float(losses[-1]) if losses else None,
              best_loss=float(np.min(losses)) if losses else None,
              median_late_loss=float(np.median(losses[-100:]))
              if len(losses) >= 100 else
              (float(np.median(losses)) if losses else None),
              diagnostics=diagnostics, selector_info=selector_info,
              wall_time_sec=wall,
              config=dict(L=L, N=N, T=T, DELAY=DELAY, BATCH=BATCH,
                         steps=STEPS, lr=LR, n_cal_traj=N_CAL_TRAJ,
                         checkpoints=CHECKPOINTS))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        git = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        out["git"] = git
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["online", "b4_arch", "b4_causal", "bptt"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--clip", type=float, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    out_path = args.out or os.path.join(
        RESULTS_DIR,
        f"b5_{args.arm}_clip{args.clip}_s{args.seed}.json")
    train(args.arm, args.seed, args.clip, out_path=out_path)


if __name__ == "__main__":
    main()
