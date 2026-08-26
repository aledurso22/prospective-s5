"""RoutePC geometry factorial — is it a learned LR? does rotation
matter? does the rotation need to be MODAL?

Arms (all the deployable causal correction — same PC0 machinery, only
the geometry's parameterization changes), 5 paired seeds, streams and
optimizer identical to route_pc.py:

  online              w = 1 (reference; stored finals)
  global-real         w = alpha in R, shared over all layers/modes
  global-complex      w = single complex scalar, shared
  per-mode-real       w_{l,j} in R per (layer, mode); dv pinned to 0
  per-mode-complex    w_{l,j} in C per (layer, mode) == PC0

The meta-update projects PC0's per-mode chain c_{l,j} = du + i dv onto
each arm's subspace: summed for the global arms (d/dw of a shared
parameter), du-only for the real arms. Update rule, lr, lr_m, clip,
Adam — all identical to PC0.

REGISTERED MECHANISM BAR (fixed before running, from the directive):
per-mode-complex must beat BOTH per-mode-real AND global-complex on
>= 4/5 paired seeds AND have >= 20% lower median final loss than each.

Run:  python route_pc_factorial.py
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

SEEDS = [0, 1, 2, 3, 4]
ARMS = ["global-real", "global-complex", "per-mode-real",
        "per-mode-complex"]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "route_pc_factorial")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def train_factor(seed, arm):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    if arm.startswith("global"):
        w_val = 1.0 + 0.0j
        w = lambda: [np.full(tcg.N, w_val, np.complex128)
                     for _ in range(tcg.L)]
    else:
        w_val = None
        w_list = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
        w = lambda: w_list
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    prev = None
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_n = tcg.flat_grads(G, params)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            # the contraction with the stored pre-update pieces
            c = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
            if arm == "per-mode-complex":
                w_list = [wl - LR_M * (-LR) * cl
                          for wl, cl in zip(w_list, c)]
            elif arm == "per-mode-real":
                w_list = [np.real(wl - LR_M * (-LR) * cl.real)
                          + 0j for wl, cl in zip(w_list, c)]
            elif arm == "global-complex":
                w_val = w_val - LR_M * (-LR) * sum(
                    cl.sum() for cl in c)
            elif arm == "global-real":
                w_val = np.real(w_val - LR_M * (-LR)
                                * sum(cl.real.sum() for cl in c)) + 0j

        w_use = w() if arm.startswith("global") else w_list
        G_use = cvm.scale_by_w(G, w_use)
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
            print(f"    {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)

    losses = np.asarray(losses)
    w_final = (w() if arm.startswith("global") else w_list)
    wmag = float(np.mean([np.abs(np.atleast_1d(wl)).mean()
                          for wl in (w_final if isinstance(w_final, list)
                                     else [w_final])]))
    return dict(arm=arm, seed=seed,
                final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                w_abs=wmag)


def chain_c_stored(Gp, th_all, u_all, sig_all, h_n):
    out = []
    off = 0
    for l in range(tcg.L):
        th = th_all[l]
        u_mode = u_all[l]
        sigp = sig_all[l]
        A = Gp["a"][l] * np.exp(1j * th)
        Gb = Gp["b"][l]
        M_ = Gb.shape[1]
        gN_rho = h_n[off:off + tcg.N]
        gN_theta = h_n[off + tcg.N:off + 2 * tcg.N]
        gN_bre = h_n[off + 2 * tcg.N:
                    off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
        gN_bim = h_n[off + 2 * tcg.N + tcg.N * M_:
                    off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
        off += 2 * tcg.N + 2 * tcg.N * M_
        du = (gN_rho * sigp * A.real
              + gN_theta * (-u_mode) * A.imag
              + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
        dv = (gN_rho * sigp * A.imag
              + gN_theta * (u_mode) * A.real
              + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
        out.append(du + 1j * dv)
    return out


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "route_pc", "summary.json")))
    finals = {"online": {s: ref["finals"]["online"][str(s)]
                         for s in SEEDS}}
    audit0 = dict(rp.BPTT_CALLS)
    for arm in ARMS:
        finals[arm] = {}
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            out = train_factor(seed, arm)
            finals[arm][seed] = out["final_loss"]
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  |w| {out['w_abs']:.2f}",
                  flush=True)
    delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert delta["exact_grad"] == 0 and delta["exact_lambda"] == 0

    med = {a: float(np.median([finals[a][s] for s in SEEDS]))
           for a in ["online"] + ARMS}
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    for a in ["online"] + ARMS:
        print(f"  {a:<17s} "
              f"{['%.4f' % finals[a][s] for s in SEEDS]}")

    pc, pr, gc = "per-mode-complex", "per-mode-real", "global-complex"
    wins_r = sum(finals[pc][s] < finals[pr][s] for s in SEEDS)
    wins_g = sum(finals[pc][s] < finals[gc][s] for s in SEEDS)
    med_ok = (med[pc] <= 0.8 * med[pr]) and (med[pc] <= 0.8 * med[gc])
    bar = wins_r >= 4 and wins_g >= 4 and med_ok
    print(f"per-mode-complex vs per-mode-real: wins {wins_r}/5  "
          f"medians {med[pc]:.4f} vs {med[pr]:.4f} "
          f"({med[pc] / med[pr]:.2f}x)")
    print(f"per-mode-complex vs global-complex: wins {wins_g}/5  "
          f"medians {med[pc]:.4f} vs {med[gc]:.4f} "
          f"({med[pc] / med[gc]:.2f}x)")
    print(f"BAR (>=4/5 paired wins vs both AND >=20% lower median vs "
          f"both): {'PASS — modal rotation is the mechanism' if bar else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, seeds=SEEDS,
                           bar=("pc beats pr and gc >=4/5 paired AND "
                                ">=20% lower median vs both")),
               finals={a: {str(s): finals[a][s] for s in SEEDS}
                       for a in finals},
               medians=med, wins_vs_real=wins_r, wins_vs_global=wins_g,
               bar_pass=bool(bar))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
