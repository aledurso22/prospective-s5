"""Teacher decomposition — what does the Route A -> RoutePC gap consist
of? Three w-learning arms with IDENTICAL correction chain, timing, and
update; only the teacher h varies:

  A  routeA:     h = g_BPTT(theta_n; B_{n-1})   (same-batch exact)
  B  nextExact:  h = g_BPTT(theta_n; B_n)       (next-batch exact)
  C  routePC:    h = g_on   (theta_n; B_n)      (next-batch causal)

  r_exactNext = -eta J_n^dagger g^BPTT_{n+1}
  r_causal    = -eta J_n^dagger g^on_{n+1}       (identical J_n)

A -> B measures the horizon/batch-shift cost of the delayed correction;
B -> C measures causal-teacher blindness. A and C are the stored
routeA/PC0 runs; B is trained here (BPTT calls = STEPS per seed — B is
a diagnostic teacher probe, NOT a deployable causal arm; the audit
records this explicitly).

Also logged at checkpoints {100, 500, 1000, 1500}, inside B's run with
identical J_n and stored context: cos(r_exactNext, r_causal) (real
(du,dv) vector), ||r_exactNext|| / ||r_causal||, and the per-mode phase
error |arg(r_exactNext conj(r_causal))| (median and 90th percentile).

Registered interpretation: if A ~= B, the delay costs nothing and the
whole A->C gap is causal blindness; if B ~= C, batch shift is the whole
cost. Report both costs as fractions of the online->A gap.

Run:  python teacher_decompose.py
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
CHECKPOINTS = [100, 500, 1000, 1500]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "teacher_decompose")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def chain_c(Gp, th_all, u_all, sig_all, h_n):
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


def train_B(seed):
    """Arm B: next-batch EXACT teacher in PC timing."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    probes = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        # next-batch EXACT teacher at the current params
        h_exact = tcg.flat_grads(cvm.exact_grad(params, x, y), params)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            rB = chain_c(Gp, th_all, u_all, sig_all, h_exact)
            if step in CHECKPOINTS:
                h_caus = tcg.flat_grads(G, params)
                rC = chain_c(Gp, th_all, u_all, sig_all, h_caus)
                vb = np.concatenate([np.ravel(r.real)
                                     for r in rB] + [np.ravel(r.imag)
                                                     for r in rB])
                vc = np.concatenate([np.ravel(r.real)
                                     for r in rC] + [np.ravel(r.imag)
                                                     for r in rC])
                cos = float(np.dot(vb, vc)
                            / (np.linalg.norm(vb) * np.linalg.norm(vc)
                               + 1e-30))
                nrat = float(np.linalg.norm(vb)
                             / (np.linalg.norm(vc) + 1e-30))
                cb = np.concatenate(rB)
                cc = np.concatenate(rC)
                dphi = np.abs(np.angle(cb * np.conj(cc)))
                probes.append(dict(step=step, cos=cos, norm_ratio=nrat,
                                   dphi_med=float(np.median(dphi)),
                                   dphi_p90=float(np.percentile(dphi,
                                                                90))))
            w_pred = [wp - LR_M * (-LR) * r_
                      for wp, r_ in zip(w_pred, rB)]

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
            print(f"    B s{seed} step {step}: loss {loss:.4f}",
                  flush=True)

    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                probes=probes)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "route_pc", "summary.json")))
    fA = {s: ref["finals"]["routeA"][str(s)] for s in SEEDS}
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}
    fO = {s: ref["finals"]["online"][str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    fB, all_probes = {}, {}
    for seed in SEEDS:
        print(f"B (nextExact) s{seed}...", flush=True)
        out = train_B(seed)
        fB[seed] = out["final_loss"]
        all_probes[seed] = out["probes"]
        print(f"  final {out['final_loss']:.4f}  finite {out['finite']}",
              flush=True)
    delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    print(f"BPTT calls (arm B only; diagnostic probe): {delta}")

    med = lambda f: float(np.median([f[s] for s in SEEDS]))
    mO, mA, mB, mC = med(fO), med(fA), med(fB), med(fC)
    gap = mO - mA
    cost_shift = mB - mA
    cost_blind = mC - mB
    print("-" * 70)
    print(f"finals  online {['%.4f' % fO[s] for s in SEEDS]}  med {mO:.4f}")
    print(f"finals  A      {['%.4f' % fA[s] for s in SEEDS]}  med {mA:.4f}")
    print(f"finals  B      {['%.4f' % fB[s] for s in SEEDS]}  med {mB:.4f}")
    print(f"finals  C      {['%.4f' % fC[s] for s in SEEDS]}  med {mC:.4f}")
    print(f"online->A gap {gap:.4f}:  batch-shift cost (B-A) "
          f"{cost_shift:.4f} ({cost_shift / gap:.1%}),  causal-blindness "
          f"cost (C-B) {cost_blind:.4f} ({cost_blind / gap:.1%})")
    print("alignment probes (r_exactNext vs r_causal, identical J_n):")
    for seed in SEEDS:
        for p in all_probes[seed]:
            print(f"  s{seed} n={p['step']:>4d}  cos {p['cos']:+.3f}  "
                  f"norm ratio {p['norm_ratio']:.2f}  "
                  f"dphi med {p['dphi_med']:.3f}  "
                  f"p90 {p['dphi_p90']:.3f}")
    cos_all = [p["cos"] for s in SEEDS for p in all_probes[s]]
    nr_all = [p["norm_ratio"] for s in SEEDS for p in all_probes[s]]
    dp_all = [p["dphi_med"] for s in SEEDS for p in all_probes[s]]
    print(f"  medians: cos {np.median(cos_all):+.3f}  "
          f"norm ratio {np.median(nr_all):.2f}  "
          f"dphi {np.median(dp_all):.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, seeds=SEEDS,
                           checkpoints=CHECKPOINTS),
               bptt_calls_arm_B=delta,
               finals=dict(online={str(s): fO[s] for s in SEEDS},
                           A={str(s): fA[s] for s in SEEDS},
                           B={str(s): fB[s] for s in SEEDS},
                           C={str(s): fC[s] for s in SEEDS}),
               medians=dict(online=mO, A=mA, B=mB, C=mC),
               gap=gap, cost_shift=cost_shift, cost_blind=cost_blind,
               probes=all_probes)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
