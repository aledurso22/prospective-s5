"""C2 — diagnose the causal per-mode-REAL geometry (addendum control 2).

Diagnostic only — do not design a new real-valued algorithm from it. The
factorial (diagnostics/route_pc_factorial.py) found the causal
per-mode-real arm competitive on 3/5 seeds even though a positive
constant gain is largely normalized away by Adam. Why does the real
geometry work at all? Candidate mechanisms to discriminate:

  (a) sign flips      — w_j crossing zero acts as a pi phase flip
                        (e^{i pi} = -1): a partial substitute rotation;
  (b) relative modal gain — per-mode |w_j| spreads differently per mode
                        (Adam normalizes a GLOBAL gain, not the relative
                        per-mode structure before clipping... measurable);
  (c) time-varying gain — the trajectory of w_j itself carries signal;
  (d) something else.

Protocol: retrain the per-mode-real arm EXACTLY as in the factorial
(real-only chain update, imaginary pinned 0) with full w-trajectory
logging (every step, all layers/modes), 5 paired seeds, frozen rig.
GATE: final losses must reproduce the stored factorial per-mode-real
finals bitwise (same-protocol proof).

Reported per layer/seed and pooled:
  * Pr(w_j < 0): occupancy over the trajectory and at final time;
  * sign flips per mode (count, median/max, fraction of modes ever
    flipping; first-flip step distribution);
  * |w_j| distribution at final (median/p90/max per layer);
  * temporal variation |w_n,j - w_{n-1,j}| (median/p90 per layer, and
    relative |dw|/(|w|+eps)).

Run:  python -m controls.c2_real_w_diagnostics
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
RESULTS_DIR = os.path.join(ROOT, "results", "c2_real_w_diagnostics")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_real(seed):
    """per-mode-real arm, factorial-verbatim + trajectory logging."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_list = [np.ones(tcg.N) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    traj = np.empty((STEPS, tcg.L, tcg.N))
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_n = tcg.flat_grads(G, params)
        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            c = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
            w_list = [np.real(wl - LR_M * (-LR) * cl.real)
                      for wl, cl in zip(w_list, c)]
        traj[step - 1] = np.asarray(w_list)
        G_use = cvm.scale_by_w(G, [wl + 0j for wl in w_list])
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
            print(f"    realw s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses)))), traj


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(ROOT, "results",
                                      "route_pc_factorial",
                                      "summary.json")))
    fR = {s: ref["finals"]["per-mode-real"][str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    finals, trajs = {}, {}
    for seed in SEEDS:
        print(f"per-mode-real s{seed}...", flush=True)
        out, trajs[seed] = train_real(seed)
        finals[seed] = out["final_loss"]
        print(f"  final {out['final_loss']:.4f}", flush=True)
        np.save(os.path.join(RESULTS_DIR, f"wtraj_s{seed}.npy"),
                trajs[seed])
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    gate = max(abs(finals[s] - fR[s]) for s in SEEDS)
    print(f"GATE vs stored factorial finals: max |dfinal| {gate:.2e}  "
          f"{'PASS' if gate == 0.0 else 'FAIL'}")
    assert gate == 0.0
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    # ---------------- diagnostics ----------------
    L, N = tcg.L, tcg.N
    report = {}
    print("-" * 78)
    print("Pr(w_j < 0): trajectory occupancy / final-time, per layer "
          "(pooled over seeds) and per seed (pooled over layers)")
    occ_layer = np.zeros(L)
    fin_layer = np.zeros(L)
    occ_seed = np.zeros(len(SEEDS))
    for li in range(L):
        occ, fin = [], []
        for seed in SEEDS:
            tr = trajs[seed][:, li, :]
            occ.append((tr < 0).mean())
            fin.append((tr[-1] < 0).mean())
        occ_layer[li], fin_layer[li] = np.mean(occ), np.mean(fin)
    for si, seed in enumerate(SEEDS):
        occ_seed[si] = (trajs[seed] < 0).mean()
    print(f"  per layer occupancy: {np.round(occ_layer, 3).tolist()}  "
          f"final: {np.round(fin_layer, 3).tolist()}")
    print(f"  per seed occupancy : {np.round(occ_seed, 3).tolist()}")
    pooled_occ = float(np.mean(occ_seed))
    print(f"  pooled occupancy   : {pooled_occ:.3f}")
    report["pr_negative"] = dict(per_layer_occupancy=occ_layer.tolist(),
                                 per_layer_final=fin_layer.tolist(),
                                 per_seed_occupancy=occ_seed.tolist(),
                                 pooled=pooled_occ)

    print("sign flips per mode (pooled over seeds):")
    flips_med, flips_max, ever = np.zeros(L), np.zeros(L), np.zeros(L)
    for li in range(L):
        counts = []
        for seed in SEEDS:
            tr = trajs[seed][:, li, :]
            sgn = np.signbit(tr)
            fl = sgn[1:] != sgn[:-1]
            counts.append(fl.sum(axis=0))
        counts = np.concatenate(counts)
        flips_med[li] = np.median(counts)
        flips_max[li] = counts.max()
        ever[li] = (counts > 0).mean()
    print(f"  median flips/mode per layer: {flips_med.tolist()}")
    print(f"  max flips/mode per layer   : {flips_max.tolist()}")
    print(f"  frac modes ever flipping   : {np.round(ever, 3).tolist()}")
    report["sign_flips"] = dict(median_per_layer=flips_med.tolist(),
                                max_per_layer=flips_max.tolist(),
                                frac_ever_per_layer=ever.tolist())

    print("|w_j| at final, per layer (median / p90 / max):")
    abs_rows = []
    for li in range(L):
        vals = np.concatenate([trajs[seed][-1, li, :]
                               for seed in SEEDS])
        abs_rows.append([float(np.median(np.abs(vals))),
                         float(np.percentile(np.abs(vals), 90)),
                         float(np.abs(vals).max())])
    for li, r in enumerate(abs_rows):
        print(f"  L{li}: median {r[0]:.3f}  p90 {r[1]:.3f}  max {r[2]:.3f}")
    report["abs_final"] = abs_rows

    print("temporal variation |dw| per layer (median / p90), and relative:")
    dw_rows = []
    for li in range(L):
        dws, rels = [], []
        for seed in SEEDS:
            tr = trajs[seed][:, li, :]
            dw = np.abs(np.diff(tr, axis=0))
            dws.append(dw.ravel())
            rels.append((dw / (np.abs(tr[:-1]) + 1e-12)).ravel())
        dws, rels = np.concatenate(dws), np.concatenate(rels)
        dw_rows.append([float(np.median(dws)),
                        float(np.percentile(dws, 90)),
                        float(np.median(rels))])
    for li, r in enumerate(dw_rows):
        print(f"  L{li}: median {r[0]:.2e}  p90 {r[1]:.2e}  "
              f"relative median {r[2]:.3f}")
    report["temporal"] = dw_rows

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, seeds=SEEDS),
               finals={str(s): finals[s] for s in SEEDS},
               gate_factorial_replay=gate, bptt_calls=audit,
               diagnostics=report)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json (+ wtraj_s*.npy)")


if __name__ == "__main__":
    main()
