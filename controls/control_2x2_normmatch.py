"""Final mechanism control on the frozen primary setup — credit repair
vs generic gradient preconditioning. PC0/F1/observer frozen; no new
teacher, no kappa change, no new architecture.

PART 1 — the real 2x2 control (paired 5 seeds, identical streams):

                    M = I           M = M_w
    g^on      |   Online (stored)   PC0 (stored)
    g^BPTT    |   BPTT (stored)     BPTT+w (NEW — oracle/control arm)

BPTT+w: same modal per-mode complex M_w and the SAME post-update
learning construction as PC0 (causal teacher meta-residual — the w
learning is byte-identical to PC0); the main parameter update applies
M_w to the exact/BPTT gradient instead of the online gradient. BPTT
calls audited (oracle arm, allowed).

Precondition: Delta_i^credit = L_online,i - L_BPTT,i per seed — the
interaction is a credit-repair test only if this defect exists
consistently.
Interaction: I_i = (L_online,i - L_PC0,i) - (L_BPTT,i - L_BPTT+w,i).
If M_w were generic preconditioning, BPTT+w would improve over BPTT as
much as PC0 improves over online (I_i ~ 0). If the benefit is
credit-specific, BPTT+w ~= BPTT and I_i ~= the whole PC0 gain.

PART 2 — PC0_normmatched: PC0 with

    g~^NM = (||g^on|| / (||M_w g^on|| + eps)) M_w g^on

per step; everything else identical (w learning unchanged). Separates
directional/phase correction from a global effective-learning-rate
effect. Also logs ||M_w g^on|| / ||g^on|| during ORDINARY PC0 training
(PC0 replay — also serves as another bitwise gate vs stored PC0).

PART 3 — retrieve (no rerun): D2 per-seed held-out oracle values
(complex from gradient_cstat; real from oracle_real_vs_complex):
per-seed cos, median, min/max, IQR — does the 0.901 median hide a
catastrophic seed?

PART 4 — explicit optimizer information (from the frozen code).

Run:  python control_2x2_normmatch.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig import routepc as rp
from toyrig.train_cell import STEPS
from toyrig.probes import make_data
from diagnostics.prospective_kappa import chain_c_stored

SEEDS = [0, 1, 2, 3, 4]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "control_2x2_normmatch")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_bptt_w(seed):
    """BPTT + M_w: exact main gradient rotated by M_w; w learned by
    PC0's causal-teacher construction, byte-identical."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_on = tcg.flat_grads(G, params)
        G_ex = cvm.exact_grad(params, x, y)
        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            r = chain_c_stored(Gp, th_all, u_all, sig_all, h_on)
            w = [wl - LR_M * (-LR) * rl for wl, rl in zip(w, r)]
        G_use = cvm.scale_by_w(G_ex, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(tcg.L)])
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    BPTT+w s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))))


def train_pc0_variant(seed, norm_matched, log_ratio=False):
    """PC0 replay with optional norm-matching and ratio logging."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    ratios = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_n = tcg.flat_grads(G, params)
        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            r = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
            w_pred = [wp - LR_M * (-LR) * r_
                      for wp, r_ in zip(w_pred, r)]
        g_on = tcg.flat_grads(G, params)
        g_rot = tcg.flat_grads(cvm.scale_by_w(G, w_pred), params)
        if log_ratio:
            ratios.append(float(np.linalg.norm(g_rot)
                                / (np.linalg.norm(g_on) + 1e-30)))
        if norm_matched:
            g_rot = g_rot * (np.linalg.norm(g_on)
                             / (np.linalg.norm(g_rot) + 1e-12))
        g = cvm.clip(g_rot)
        flat, m, v = cvm.adam(flat, g, m, v, step)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(tcg.L)])
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    PC0{'NM' if norm_matched else ''} s{seed} "
                  f"step {step}: loss {loss:.4f}", flush=True)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                ratio_med=float(np.median(ratios)) if ratios else None)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "route_pc", "summary.json")))
    fO = {s: ref["finals"]["online"][str(s)] for s in SEEDS}
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}
    fB = {}
    for s in SEEDS:
        fB[s] = json.load(open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "co_variational_metric",
            f"bptt_s{s}.json")))["final_loss"]

    audit0 = dict(rp.BPTT_CALLS)
    fBW, fNM, fPC, ratios = {}, {}, {}, {}
    for seed in SEEDS:
        print(f"BPTT+w s{seed}...", flush=True)
        out = train_bptt_w(seed)
        fBW[seed] = out["final_loss"]
        print(f"  final {out['final_loss']:.4f}  finite {out['finite']}",
              flush=True)
        print(f"PC0_normmatched s{seed}...", flush=True)
        out = train_pc0_variant(seed, norm_matched=True)
        fNM[seed] = out["final_loss"]
        print(f"  final {out['final_loss']:.4f}", flush=True)
        print(f"PC0 replay (ratio log) s{seed}...", flush=True)
        out = train_pc0_variant(seed, norm_matched=False, log_ratio=True)
        fPC[seed] = out["final_loss"]
        ratios[seed] = out["ratio_med"]
        print(f"  final {out['final_loss']:.4f}  "
              f"median |M_w g|/|g| {out['ratio_med']:.3f}", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}

    gate = max(abs(fPC[s] - fC[s]) for s in SEEDS)
    print(f"PC0 replay vs stored: max |dfinal| {gate:.2e} (gate)")
    print(f"BPTT calls (BPTT+w arm + PC0 arms): {audit}")

    # ---- PART 1: the 2x2 ----
    med = lambda f: float(np.median([f[s] for s in SEEDS]))
    D_credit = {s: fO[s] - fB[s] for s in SEEDS}
    I = {s: (fO[s] - fC[s]) - (fB[s] - fBW[s]) for s in SEEDS}
    print("-" * 78)
    print("2x2 finals:")
    print(f"  Online : {['%.4f' % fO[s] for s in SEEDS]}")
    print(f"  PC0    : {['%.4f' % fC[s] for s in SEEDS]}")
    print(f"  BPTT   : {['%.5f' % fB[s] for s in SEEDS]}")
    print(f"  BPTT+w : {['%.5f' % fBW[s] for s in SEEDS]}")
    print(f"Delta_credit (online - BPTT): "
          f"{['%+.4f' % D_credit[s] for s in SEEDS]}")
    print(f"I = (online-PC0) - (BPTT-BPTT+w): "
          f"{['%+.4f' % I[s] for s in SEEDS]}")
    iv = np.array([I[s] for s in SEEDS])
    print(f"  median {np.median(iv):+.4f}  mean {iv.mean():+.4f}  "
          f"sd {iv.std(ddof=1):.4f}")
    precond = sum(D_credit[s] > 0 for s in SEEDS)

    # ---- PART 2 ----
    print(f"PC0_normmatched finals: {['%.4f' % fNM[s] for s in SEEDS]}  "
          f"med {med(fNM):.4f}")
    print(f"PC0 |M_w g|/|g| per-seed medians: "
          f"{['%.3f' % ratios[s] for s in SEEDS]}  "
          f"pooled {float(np.median(list(ratios.values()))):.3f}")

    # ---- PART 3: D2 per-seed held-out ----
    gc = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "gradient_cstat", "summary.json")))
    zseed = [(r["seed"], r["held_out"]["zoracle"][0])
             for r in gc["rows"]]
    zseed.sort()
    rc = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "oracle_real_vs_complex", "summary.json")))
    rseed = [(r["seed"], r["real"][0]) for r in rc["rows"]]
    rseed.sort()
    zv = np.array([v for _, v in zseed])
    rv = np.array([v for _, v in rseed])
    print("D2 per-mode-complex held-out cos per seed: "
          f"{[round(v, 3) for v in zv]}")
    print(f"  median {np.median(zv):.3f}  min {zv.min():.3f}  "
          f"max {zv.max():.3f}  IQR "
          f"[{np.percentile(zv, 25):.3f}, {np.percentile(zv, 75):.3f}]")
    print("D2 per-mode-real held-out cos per seed: "
          f"{[round(v, 3) for v in rv]}")
    print(f"  median {np.median(rv):.3f}  min {rv.min():.3f}  "
          f"max {rv.max():.3f}")

    # ---- PART 4 ----
    print("-" * 78)
    print("optimizer record: theta: Adam (b1=0.9, b2=0.999, eps=1e-8), "
          "LR = 1e-3 constant; grad clipped to global norm 1.0 BEFORE "
          "Adam; w (MetaOpt): plain SGD, LR_M = 1e-3, no clip, no Adam; "
          "M_w = conj(w) per mode on the (a, B) gradient blocks applied "
          "to the RAW gradient BEFORE clipping and BEFORE any Adam "
          "normalization; the meta-chain ignores Adam/clip (documented "
          "cvm simplification).")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               finals=dict(online={str(s): fO[s] for s in SEEDS},
                           pc0={str(s): fC[s] for s in SEEDS},
                           bptt={str(s): fB[s] for s in SEEDS},
                           bptt_w={str(s): fBW[s] for s in SEEDS},
                           pc0_normmatched={str(s): fNM[s]
                                            for s in SEEDS}),
               delta_credit=D_credit, interaction=I,
               interaction_stats=dict(median=float(np.median(iv)),
                                      mean=float(iv.mean()),
                                      sd=float(iv.std(ddof=1))),
               precondition_seeds_positive=precond,
               ratios=ratios,
               d2_complex=dict(per_seed={str(s): float(v)
                                         for s, v in zseed},
                               median=float(np.median(zv)),
                               min=float(zv.min()), max=float(zv.max()),
                               iqr=[float(np.percentile(zv, 25)),
                                    float(np.percentile(zv, 75))]),
               d2_real=dict(per_seed={str(s): float(v)
                                      for s, v in rseed},
                            median=float(np.median(rv)),
                            min=float(rv.min()), max=float(rv.max())),
               gate_pc0_replay=gate, bptt_calls=audit)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
