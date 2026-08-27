"""B5.1 -- action-utility audit. Strictly diagnostic: does NOT change the
CCM (B4 rank-1) algorithm, does NOT run new seeds, does NOT launch S5.

Reuses the exact 8 B5 seeds and their already-selected A2 (b4_causal)
channels (loaded from results/credit_memory/b5/b5_b4_causal_clip{clip}_
s{seed}.json -- never re-calibrated here). Replays the ONLINE arm's
exact training trajectory (same RNG draws as credit_memory/b5_train.py's
"online" arm, so intermediate states are bitwise-reproducible, not a new
experiment) up to three checkpoints (100/300/600, matching B5's own
"early/middle/late" checkpoints), and at each one:

  B5.1A: clone (params, Adam m/v) at that checkpoint; compute g_on,
         g_CCM, g_BPTT on the SAME next batch; apply ONE Adam step with
         each (three independent optimizer-state copies, same starting
         m/v, no reset); evaluate all three on the SAME fixed held-out
         post-update batch; report delta L for each.
  B5.1B: compare the resulting parameter UPDATE vectors (post-Adam-
         transform), not just the raw gradients.
  B5.1C: break the online-to-BPTT gradient defect, CCM's repair of it,
         and BPTT's actual update-norm share, down by parameter block
         (lower a, lower b, upper [a+b], readout c).
  B5.1D: summarize the existing B5 bptt-arm task performance vs online
         across the same 8 seeds (no new runs).

Run:  python -m credit_memory.b5_1_action_utility
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F
from credit_memory.b4_deploy import b4_layer0_gradient
from credit_memory.b5_train import (set_config, draw_task_batch, loss_of,
                                    L, N, T, DELAY, BATCH, LR)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b5")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory", "b5_1")

SEEDS = list(range(8))
CLIPS = [0.0, 1.0]
CHECKPOINTS = [100, 300, 600]           # early / middle / late
B1_, B2_, EPS = 0.9, 0.999, 1e-8


def load_b4_causal_selection(seed, clip):
    for tag in (str(clip), str(int(clip))):
        path = os.path.join(RESULTS_DIR, f"b5_b4_causal_clip{tag}_s{seed}.json")
        if os.path.exists(path):
            run = json.load(open(path))
            if run["clip"] == clip:
                return {int(k): v for k, v in
                       run["selector_info"]["top_j_by_mode"].items()}
    raise FileNotFoundError(f"no b4_causal run for seed={seed} clip={clip}")


def block_slices(params):
    """Index ranges into the flat gradient/update vector, matching
    toyrig.ssm_rig.flat_grads's exact construction order."""
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


def adam_step(flat, m_, v_, g, step, lr):
    m_ = B1_ * m_ + (1 - B1_) * g
    v_ = B2_ * v_ + (1 - B2_) * g ** 2
    flat_new = flat - lr * (m_ / (1 - B1_ ** step)) / (
        np.sqrt(v_ / (1 - B2_ ** step)) + EPS)
    return flat_new, m_, v_


def replay_online_to_checkpoints(seed, clip, checkpoints):
    """Bitwise-identical replay of credit_memory.b5_train's "online" arm
    loop (same RNG draws, same order) -- not a new experiment, just
    re-deriving intermediate state the original B5 run did not save."""
    set_config()
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    snapshots = {}
    for step in range(1, max(checkpoints) + 1):
        x, y = draw_task_batch(rng)
        loss, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        g = tcg.flat_grads(G, params)
        nrm = np.linalg.norm(g)
        if clip > 0 and nrm > clip:
            g = g * (clip / nrm)
        flat, m_, v_ = adam_step(flat, m_, v_, g, step, LR)
        params = tcg.pack(params, flat)
        if step in checkpoints:
            # capture the RNG state AS IT STANDS right after this
            # checkpoint's update, so the "next batch" drawn for the
            # action-utility test at THIS checkpoint matches what the
            # original online run would have drawn next -- NOT the RNG
            # state after later checkpoints (a real bug caught before
            # running: a single shared `rng` object returned after the
            # full replay would silently give every checkpoint the
            # post-step-600 draw instead of its own next-batch draw).
            snapshots[step] = dict(params=params, flat=flat.copy(),
                                   m=m_.copy(), v=v_.copy(), step=step,
                                   rng_state=rng.get_state())
    return snapshots


def gradients_at(params, top_j_by_mode, x, y):
    """g_on, g_CCM, g_BPTT (raw, unclipped) at the given params/batch."""
    loss, h, r = loss_of(params, x, y)
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)

    G_on = tcg.assemble(params, h, x, r, q, Sa, Sb)
    g_on = tcg.flat_grads(G_on, params)

    lam = tcg.exact_lambda(params, q)
    G_bptt = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
    g_bptt = tcg.flat_grads(G_bptt, params)

    a1 = params["a"][1]
    f_diag = build_F(a1)
    Ga0, Gb0 = b4_layer0_gradient(f_diag, top_j_by_mode, params["b"][1], N,
                                  q[1], Sa[0], Sb[0])
    G_ccm = dict(a=[Ga0] + G_on["a"][1:], b=[Gb0] + G_on["b"][1:],
                c=G_on["c"])
    g_ccm = tcg.flat_grads(G_ccm, params)

    return g_on, g_ccm, g_bptt, loss


def cos_np(u, v):
    return float(np.abs(np.vdot(v, u))
                / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def relerr_np(u, v):
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-300))


def audit_one(seed, clip):
    top_j_by_mode = load_b4_causal_selection(seed, clip)
    snapshots = replay_online_to_checkpoints(seed, clip, CHECKPOINTS)

    rows = []
    for step in CHECKPOINTS:
        snap = snapshots[step]
        params0 = snap["params"]

        # action batch: the SAME next batch the original online run
        # would have drawn AT THIS CHECKPOINT -- restore the RNG to its
        # exact post-checkpoint state (not a shared/continuing stream
        # across checkpoints; see the fix note in
        # replay_online_to_checkpoints).
        rng_ckpt = np.random.RandomState()
        rng_ckpt.set_state(snap["rng_state"])
        x_act, y_act = draw_task_batch(rng_ckpt)
        g_on, g_ccm, g_bptt, loss0_act = gradients_at(params0,
                                                       top_j_by_mode, x_act,
                                                       y_act)

        clipped = {}
        for name, g in (("on", g_on), ("ccm", g_ccm), ("bptt", g_bptt)):
            nrm = np.linalg.norm(g)
            clipped[name] = g * (clip / nrm) if (clip > 0 and nrm > clip) \
                else g

        # B5.1A: independent one-step Adam action from the SAME cloned
        # (m, v); no moment reset (explicit secondary control, not run
        # here, per instruction).
        flat0, m0, v0 = snap["flat"], snap["m"], snap["v"]
        results_arm = {}
        deltas = {}
        for name in ("on", "ccm", "bptt"):
            flat_new, _, _ = adam_step(flat0.copy(), m0.copy(), v0.copy(),
                                       clipped[name], step + 1, LR)
            results_arm[name] = flat_new
            deltas[name] = flat_new - flat0

        # fixed held-out post-update evaluation batch, deterministic,
        # SAME for all three arms and for the pre-update baseline
        eval_rng = np.random.RandomState(99999 + seed * 1000 + step)
        x_eval, y_eval = draw_task_batch(eval_rng)
        params_pre = tcg.pack(params0, flat0)
        loss_pre, _, _ = loss_of(params_pre, x_eval, y_eval)

        delta_L = {}
        for name in ("on", "ccm", "bptt"):
            params_new = tcg.pack(params0, results_arm[name])
            loss_new, _, _ = loss_of(params_new, x_eval, y_eval)
            delta_L[name] = float(loss_new - loss_pre)

        # B5.1B: optimizer-space comparison
        cos_on_bptt = cos_np(deltas["on"], deltas["bptt"])
        cos_ccm_bptt = cos_np(deltas["ccm"], deltas["bptt"])
        rel_on_bptt = relerr_np(deltas["on"], deltas["bptt"])
        rel_ccm_bptt = relerr_np(deltas["ccm"], deltas["bptt"])
        norm_ratio_on = float(np.linalg.norm(deltas["on"])
                              / (np.linalg.norm(deltas["bptt"]) + 1e-300))
        norm_ratio_ccm = float(np.linalg.norm(deltas["ccm"])
                               / (np.linalg.norm(deltas["bptt"]) + 1e-300))

        # B5.1C: parameter-block defect accounting (raw, unclipped
        # gradients for D_b/R_b; post-Adam BPTT update for U_b)
        slices, total = block_slices(params0)
        def sl(vec, key):
            a, b = slices[key]
            return vec[a:b]
        defect_total = g_bptt - g_on
        denom_total = np.sum(np.abs(defect_total) ** 2)
        blocks = {}
        for key in slices:
            d_num = np.sum(np.abs(sl(g_bptt, key) - sl(g_on, key)) ** 2)
            D_b = float(d_num / (denom_total + 1e-300))
            resid_on = np.linalg.norm(sl(g_bptt, key) - sl(g_on, key))
            resid_ccm = np.linalg.norm(sl(g_bptt, key) - sl(g_ccm, key))
            R_b = float(1.0 - resid_ccm / (resid_on + 1e-300))
            U_num = np.sum(np.abs(sl(deltas["bptt"], key)) ** 2)
            U_denom = np.sum(np.abs(deltas["bptt"]) ** 2)
            U_b = float(U_num / (U_denom + 1e-300))
            # per-block COSINE (direction only) alongside R_b (absolute
            # residual): R_b can stay near zero even when cosine improves
            # a lot, if the correction is right in direction but wrong in
            # scale -- the same norm-mismatch pattern documented
            # throughout B1-B4. Reported explicitly so R_b is not
            # misread as "CCM does nothing" for a block where the
            # directional mechanism is, separately, well established.
            cos_on_b = cos_np(sl(g_on, key), sl(g_bptt, key))
            cos_ccm_b = cos_np(sl(g_ccm, key), sl(g_bptt, key))
            norm_ratio_on_b = float(np.linalg.norm(sl(g_on, key))
                                    / (np.linalg.norm(sl(g_bptt, key))
                                       + 1e-300))
            norm_ratio_ccm_b = float(np.linalg.norm(sl(g_ccm, key))
                                     / (np.linalg.norm(sl(g_bptt, key))
                                        + 1e-300))
            blocks[key] = dict(D_b=D_b, R_b=R_b, U_b=U_b,
                               cos_on=cos_on_b, cos_ccm=cos_ccm_b,
                               norm_ratio_on=norm_ratio_on_b,
                               norm_ratio_ccm=norm_ratio_ccm_b)
        # aggregate "upper" = a1+b1
        up_D = blocks["a1"]["D_b"] + blocks["b1"]["D_b"]
        up_U = blocks["a1"]["U_b"] + blocks["b1"]["U_b"]
        up_resid_on = np.linalg.norm(np.concatenate([
            sl(g_bptt, "a1") - sl(g_on, "a1"),
            sl(g_bptt, "b1") - sl(g_on, "b1")]))
        up_resid_ccm = np.linalg.norm(np.concatenate([
            sl(g_bptt, "a1") - sl(g_ccm, "a1"),
            sl(g_bptt, "b1") - sl(g_ccm, "b1")]))
        up_g_on = np.concatenate([sl(g_on, "a1"), sl(g_on, "b1")])
        up_g_ccm = np.concatenate([sl(g_ccm, "a1"), sl(g_ccm, "b1")])
        up_g_bptt = np.concatenate([sl(g_bptt, "a1"), sl(g_bptt, "b1")])
        blocks["upper(a1+b1)"] = dict(
            D_b=float(up_D), R_b=float(1.0 - up_resid_ccm
                                       / (up_resid_on + 1e-300)),
            U_b=float(up_U), cos_on=cos_np(up_g_on, up_g_bptt),
            cos_ccm=cos_np(up_g_ccm, up_g_bptt),
            norm_ratio_on=float(np.linalg.norm(up_g_on)
                                / (np.linalg.norm(up_g_bptt) + 1e-300)),
            norm_ratio_ccm=float(np.linalg.norm(up_g_ccm)
                                 / (np.linalg.norm(up_g_bptt) + 1e-300)))

        rows.append(dict(
            seed=seed, clip=clip, step=step,
            delta_L=delta_L,
            cos_delta_theta=dict(on_vs_bptt=cos_on_bptt,
                                 ccm_vs_bptt=cos_ccm_bptt),
            rel_err_delta_theta=dict(on_vs_bptt=rel_on_bptt,
                                     ccm_vs_bptt=rel_ccm_bptt),
            norm_ratio_delta_theta=dict(on=norm_ratio_on, ccm=norm_ratio_ccm),
            cos_gradient=dict(on_vs_bptt=cos_np(g_on, g_bptt),
                              ccm_vs_bptt=cos_np(g_ccm, g_bptt)),
            blocks=blocks))
    return rows


def b51d_summary(clip):
    """B5.1D: existing B5 bptt-vs-online task performance, no new runs."""
    rows = []
    for seed in SEEDS:
        on_path = None
        bp_path = None
        for tag in (str(clip), str(int(clip))):
            p1 = os.path.join(RESULTS_DIR, f"b5_online_clip{tag}_s{seed}.json")
            p2 = os.path.join(RESULTS_DIR, f"b5_bptt_clip{tag}_s{seed}.json")
            if os.path.exists(p1) and on_path is None:
                r = json.load(open(p1))
                if r["clip"] == clip:
                    on_path = r
            if os.path.exists(p2) and bp_path is None:
                r = json.load(open(p2))
                if r["clip"] == clip:
                    bp_path = r
        if on_path and bp_path:
            rows.append(dict(seed=seed, online_final=on_path["final_loss"],
                             bptt_final=bp_path["final_loss"],
                             online_late=on_path["median_late_loss"],
                             bptt_late=bp_path["median_late_loss"],
                             ratio_final=bp_path["final_loss"]
                             / on_path["final_loss"]))
    return rows


def main() -> None:
    print("=" * 90)
    print("Phase B5.1: action-utility audit (diagnostic only)")
    print("=" * 90)

    all_rows = []
    for clip in CLIPS:
        for seed in SEEDS:
            rows = audit_one(seed, clip)
            all_rows += rows
            for row in rows:
                print(f"seed={seed} clip={clip} step={row['step']}: "
                      f"dL_on={row['delta_L']['on']:+.5f}  "
                      f"dL_ccm={row['delta_L']['ccm']:+.5f}  "
                      f"dL_bptt={row['delta_L']['bptt']:+.5f}   "
                      f"cos(dtheta_on,bptt)={row['cos_delta_theta']['on_vs_bptt']:.3f}"
                      f"  cos(dtheta_ccm,bptt)={row['cos_delta_theta']['ccm_vs_bptt']:.3f}")

    print("-" * 90)
    for clip in CLIPS:
        for step in CHECKPOINTS:
            sub = [r for r in all_rows if r["clip"] == clip
                  and r["step"] == step]
            dl_on = np.array([r["delta_L"]["on"] for r in sub])
            dl_ccm = np.array([r["delta_L"]["ccm"] for r in sub])
            dl_bptt = np.array([r["delta_L"]["bptt"] for r in sub])
            print(f"clip={clip} step={step}: median dL "
                  f"on={np.median(dl_on):+.5f}  ccm={np.median(dl_ccm):+.5f}"
                  f"  bptt={np.median(dl_bptt):+.5f}   "
                  f"median cos(dtheta) on_vs_bptt="
                  f"{np.median([r['cos_delta_theta']['on_vs_bptt'] for r in sub]):.3f}"
                  f"  ccm_vs_bptt="
                  f"{np.median([r['cos_delta_theta']['ccm_vs_bptt'] for r in sub]):.3f}")

    print("-" * 90)
    print("Block accounting (median over seeds, clip=0, step=300):")
    sub = [r for r in all_rows if r["clip"] == 0.0 and r["step"] == 300]
    for key in ["a0", "b0", "upper(a1+b1)", "c"]:
        Ds = [r["blocks"][key]["D_b"] for r in sub]
        Rs = [r["blocks"][key]["R_b"] for r in sub]
        Us = [r["blocks"][key]["U_b"] for r in sub]
        Con = [r["blocks"][key]["cos_on"] for r in sub]
        Cccm = [r["blocks"][key]["cos_ccm"] for r in sub]
        NRon = [r["blocks"][key]["norm_ratio_on"] for r in sub]
        NRccm = [r["blocks"][key]["norm_ratio_ccm"] for r in sub]
        print(f"  {key:16s}  D_b={np.median(Ds):.3f}  R_b={np.median(Rs):.3f}"
              f"  U_b={np.median(Us):.3f}  cos_on={np.median(Con):.3f}"
              f"  cos_ccm={np.median(Cccm):.3f}  norm_ratio_on="
              f"{np.median(NRon):.3f}  norm_ratio_ccm={np.median(NRccm):.3f}")

    print("-" * 90)
    b51d = {clip: b51d_summary(clip) for clip in CLIPS}
    for clip, rows in b51d.items():
        if not rows:
            continue
        ratios = [r["ratio_final"] for r in rows]
        wins = sum(1 for r in rows if r["bptt_final"] < r["online_final"])
        print(f"B5.1D clip={clip}: bptt beats online on final loss "
              f"{wins}/{len(rows)}  median ratio(bptt/online)="
              f"{np.median(ratios):.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
              config=dict(L=L, N=N, T=T, DELAY=DELAY, BATCH=BATCH, lr=LR,
                         seeds=SEEDS, clips=CLIPS, checkpoints=CHECKPOINTS),
              rows=all_rows, b51d_summary=b51d)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "b5_1_action_utility_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
