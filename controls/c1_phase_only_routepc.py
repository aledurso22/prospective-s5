"""C1 — phase-only / unit-modulus RoutePC (addendum control 1 of 3).

Claim-sharpening control on the frozen primary toy setup; NOT a new
algorithm program. PC0's modal geometry is w_j = rho_j e^{i phi_j};
factorize_w showed (routeA regime) that the frozen PHASE carries the
mechanism. Here the causal PC0 learner is constrained to unit modulus:

    |w_j| = 1 at all times:  g~_j = e^{i phi_j} g_j^on,

i.e. M_phi is a pure 2x2 rotation. Realization: the IDENTICAL PC0
correction (same causal post-update signal, same chain, same LR_M) is
followed by a per-mode projection w <- w/|w| after every meta step — the
radial component of the signal is discarded, only phi is learned.
Everything else (data streams, Adam/clip, applied update) is PC0
verbatim. PC0 itself is preserved permanently (toyrig/routepc.py; this
file is a control, not a replacement).

Arms: online (stored finals), PC0 (replay — bitwise gate vs stored),
pcPhase (new). 5 paired seeds {0..4}, frozen protocol (STEPS=1500,
LR=LR_M=1e-3, CLIP=1.0, delayed copy D=50/T=128, L=4, N=16, batch=32).

REGISTERED DECISION RULE (fixed before running, from the addendum):
  pcPhase is "clearly competitive" iff BOTH
    median(pcPhase) <= 1.5 x median(PC0), AND
    pcPhase beats online on >= 4/5 paired seeds.
  Competitive  -> extend online/PC0/pcPhase to 15 paired seeds
                  (seeds 0..14; 0..4 reused from here) for a failure-rate
                  estimate and register pcPhase as a selectable arm.
  Clearly fails -> stop, do not spend the 15-seed budget.
Seed 3 is reported as a genuine outcome, not an artifact.

Reporting (addendum statistical standard): per-seed finals AND marginal
medians AND paired ratios L_arm,i/L_online,i (median), plus
L_pcPhase,i/L_PC0,i. No percentage-of-marginal-medians summaries.

Run:  python -m controls.c1_phase_only_routepc
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "c1_phase_only_routepc")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_pc(seed, phase_only):
    """PC0 replay (phase_only=False — must reproduce stored finals
    bitwise) and the unit-modulus arm (phase_only=True). One difference:
    after the causal correction, w is projected onto the unit circle."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
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
            if phase_only:
                w_pred = [wp_ / np.maximum(np.abs(wp_), 1e-12)
                          for wp_ in w_pred]
        G_use = cvm.scale_by_w(G, w_pred)
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
            print(f"    {'pcPhase' if phase_only else 'PC0replay'} s{seed}"
                  f" step {step}: loss {loss:.4f}", flush=True)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                argw_std=float(np.mean([np.std(np.angle(wl))
                                        for wl in w_pred])))


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                      "summary.json")))
    fO = {s: ref["finals"]["online"][str(s)] for s in SEEDS}
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    fR, fP = {}, {}
    for seed in SEEDS:
        print(f"PC0 replay s{seed}...", flush=True)
        fR[seed] = train_pc(seed, phase_only=False)["final_loss"]
        print(f"  final {fR[seed]:.4f}", flush=True)
        print(f"pcPhase s{seed}...", flush=True)
        fP[seed] = train_pc(seed, phase_only=True)["final_loss"]
        print(f"  final {fP[seed]:.4f}", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}

    gate = max(abs(fR[s] - fC[s]) for s in SEEDS)
    print(f"GATE PC0 replay vs stored: max |dfinal| {gate:.2e}  "
          f"{'PASS' if gate == 0.0 else 'FAIL'}")
    print(f"BPTT calls (both arms): {audit}  (must be 0/0)")
    assert gate == 0.0
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    med = lambda f: float(np.median([f[s] for s in SEEDS]))
    print("-" * 78)
    print(f"{'arm':<10s}{'s0':>9s}{'s1':>9s}{'s2':>9s}{'s3':>9s}{'s4':>9s}"
          f"{'median':>10s}")
    for name, f in [("online", fO), ("PC0", fC), ("pcPhase", fP)]:
        print(f"{name:<10s}" + "".join(f"{f[s]:>9.4f}" for s in SEEDS)
              + f"{med(f):>10.4f}")
    print("seed 3 (genuine outcome): online "
          f"{fO[3]:.4f} / PC0 {fC[3]:.4f} / pcPhase {fP[3]:.4f}")
    rat_o = {s: fP[s] / fO[s] for s in SEEDS}
    rat_c = {s: fP[s] / fC[s] for s in SEEDS}
    print(f"paired ratios pcPhase/online: "
          f"{['%.3f' % rat_o[s] for s in SEEDS]}  "
          f"median {np.median(list(rat_o.values())):.3f}")
    print(f"paired ratios pcPhase/PC0   : "
          f"{['%.3f' % rat_c[s] for s in SEEDS]}  "
          f"median {np.median(list(rat_c.values())):.3f}")
    beats_online = sum(fP[s] < fO[s] for s in SEEDS)
    competitive = (med(fP) <= 1.5 * med(fC)) and beats_online >= 4
    print(f"competitiveness: median {med(fP):.4f} vs 1.5x PC0 "
          f"{1.5 * med(fC):.4f}; beats online {beats_online}/5  ->  "
          f"{'COMPETITIVE — extend to 15 seeds + register selectable arm'
            if competitive else 'NOT COMPETITIVE — stop, keep PC0'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, clip=cvm.CLIP,
                           seeds=SEEDS, rule=("competitive iff median "
                                              "<= 1.5x PC0 AND beats "
                                              "online >=4/5 paired")),
               finals=dict(online={str(s): fO[s] for s in SEEDS},
                           pc0={str(s): fC[s] for s in SEEDS},
                           pcphase={str(s): fP[s] for s in SEEDS}),
               medians=dict(online=med(fO), pc0=med(fC), pcphase=med(fP)),
               paired_ratios=dict(
                   pcphase_over_online={str(s): rat_o[s] for s in SEEDS},
                   pcphase_over_pc0={str(s): rat_c[s] for s in SEEDS}),
               beats_online=beats_online, competitive=bool(competitive),
               gate_pc0_replay=gate, bptt_calls=audit)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
