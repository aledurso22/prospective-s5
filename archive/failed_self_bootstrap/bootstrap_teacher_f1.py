"""F1 — causal bootstrap-teacher test. Frozen: RoutePC core, kappa, all
prior results. No new filter family, no Hessian/Simonetto, no latent
observer.

Idea: the learned geometry contains information about the exact
teacher (g^exact ~= M_w* g^on). Use a STOP-GRADIENT EMA of the learned
geometry as a standing proxy for the unavailable exact teacher:

    wbar_{n+1} = beta wbar_n + (1 - beta) w_{n+1}     (beta = 0.99,
                                                       fixed, not tuned)
    g^teacher_{n+1}(alpha) = [(1-alpha) I + alpha M_{wbar_n}] g^on_{n+1}
    r^boot_{n+1} = -eta J_n^dag sg[g^teacher_{n+1}(alpha)]
    w_{n+1} = MetaOpt(w_n, r^boot_{n+1})              (identical to PC0)

Arms: alpha in {0, 0.5, 1}. Five paired seeds, same init/RNG/
hyperparameters/geometry/next-batch timing/MetaOpt as PC0.

REGISTERED GATES:
  G1: alpha = 0 reproduces stored PC0 finals bitwise.
  G2: BPTT/exact-gradient calls during TRAINING are 0 in every arm
      (offline diagnostic probes are accounted separately).
  G3: g^on, g^teacher, wbar enter the meta-residual as fixed values —
      numpy-only toy, so the stop-gradient semantics are structural;
      the executable check is that step-1 losses are identical across
      arms (the base trajectory can differ only through the teacher
      for w).
  G4: the base parameter update is byte-identical machinery
      (scale_by_w + clip + Adam) — only the w sequence changes.

OFFLINE DIAGNOSTIC (oracle probes at checkpoints {250,500,1000,1500},
separately accounted): cos(g^boot, g^exact), cos(r^boot, r^exact),
norm ratios. At alpha = 0 these reproduce the causal-vs-exact
alignment (~0.854 from teacher_decompose.py) — anchor check.

PRIMARY PREDICTION: L_{alpha>0} < 0.0073 (PC0 median) on most paired
seeds, with improved cos(r^boot, r^exact). If alpha=1 unstable but
alpha=0.5 helps: useful self-teacher information with self-confirmation
risk. If neither helps: (I - M_w)g^on is not a good enough causal
measurement of the persistent deficit — stop, no observer stage.

Run:  python bootstrap_teacher_f1.py
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
ALPHAS = [0.0, 0.5, 1.0]
BETA = 0.99                    # fixed, not tuned
CHECKPOINTS = [250, 500, 1000, 1500]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "bootstrap_teacher_f1")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def scale_factor(G, fac):
    """G blocks scaled per mode by complex fac[l] (N,)."""
    return dict(a=[fac[l] * G["a"][l] for l in range(tcg.L)],
                b=[fac[l][:, None] * G["b"][l] for l in range(tcg.L)],
                c=G["c"])


def vec(c_list):
    return np.concatenate([np.ravel(c.real) for c in c_list]
                          + [np.ravel(c.imag) for c in c_list])


def cosv(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)
                                 + 1e-30))


def train_f1(seed, alpha):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    wbar = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    first_step_loss = None
    diags = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        if step == 1:
            first_step_loss = loss
        fac = [(1.0 - alpha) + alpha * np.conj(wbar[l])
               for l in range(tcg.L)]
        h_boot = tcg.flat_grads(scale_factor(G, fac), params)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            r_boot = chain_c_stored(Gp, th_all, u_all, sig_all, h_boot)
            w = [wl - LR_M * (-LR) * rb for wl, rb in zip(w, r_boot)]
            wbar = [BETA * wb + (1 - BETA) * wl
                    for wb, wl in zip(wbar, w)]

        # ---- offline diagnostic (oracle; accounted separately) ----
        if step in CHECKPOINTS and prev is not None:
            Gp, th_all, u_all, sig_all = prev
            h_ex = tcg.flat_grads(cvm.exact_grad(params, x, y), params)
            r_ex = chain_c_stored(Gp, th_all, u_all, sig_all, h_ex)
            r_bo = chain_c_stored(Gp, th_all, u_all, sig_all, h_boot)
            diags.append(dict(
                step=step,
                cos_g=cosv(h_boot, h_ex),
                nrat_g=float(np.linalg.norm(h_boot)
                             / (np.linalg.norm(h_ex) + 1e-30)),
                cos_r=cosv(vec(r_bo), vec(r_ex)),
                nrat_r=float(np.linalg.norm(vec(r_bo))
                             / (np.linalg.norm(vec(r_ex)) + 1e-30))))

        G_use = cvm.scale_by_w(G, w)
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
            print(f"    a={alpha} s{seed} step {step}: loss "
                  f"{loss:.4f}", flush=True)

    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                first_step_loss=first_step_loss, diags=diags)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "route_pc", "summary.json")))
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}

    audit_train0 = dict(rp.BPTT_CALLS)
    finals = {a: {} for a in ALPHAS}
    diags = {}
    firsts = {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            print(f"alpha={alpha} s{seed}...", flush=True)
            before = dict(rp.BPTT_CALLS)
            out = train_f1(seed, alpha)
            after = dict(rp.BPTT_CALLS)
            # training-phase BPTT calls = delta minus diagnostic probes
            n_diag = len(out["diags"])
            train_calls = {k: after[k] - before[k] for k in after}
            finals[alpha][seed] = out["final_loss"]
            diags.setdefault(alpha, {})[seed] = out["diags"]
            firsts.setdefault(alpha, {})[seed] = out["first_step_loss"]
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  "
                  f"(calls incl. {n_diag} diagnostic probes: "
                  f"{train_calls})", flush=True)

    # ---- gates ----
    gate1 = max(abs(finals[0.0][s] - fC[s]) for s in SEEDS)
    n_probes = sum(len(diags[a][s]) for a in ALPHAS for s in SEEDS)
    train_bptt = {k: rp.BPTT_CALLS[k] - audit_train0[k] - n_probes
                  for k in rp.BPTT_CALLS}
    gate2 = (train_bptt["exact_grad"] == 0
             and train_bptt["exact_lambda"] == 0)
    gate3 = max(abs(firsts[0.5][s] - firsts[0.0][s]) for s in SEEDS) == 0
    gate3 &= max(abs(firsts[1.0][s] - firsts[0.0][s]) for s in SEEDS) == 0
    print(f"G1 alpha=0 == stored PC0: max |dfinal| {gate1:.2e}")
    print(f"G2 training-phase BPTT calls: {train_bptt} (0 required)")
    print(f"G3 step-1 losses identical across arms: {gate3}")
    print(f"(diagnostic probes: {n_probes} exact_grad evals, "
          f"accounted separately)")

    # ---- table ----
    med = {a: float(np.median([finals[a][s] for s in SEEDS]))
           for a in ALPHAS}
    pc0_med = float(np.median([fC[s] for s in SEEDS]))
    print("-" * 78)
    for a in ALPHAS:
        print(f"  a={a:<4} {['%.4f' % finals[a][s] for s in SEEDS]}  "
              f"med {med[a]:.4f}")
    print(f"  PC0 median reference: {pc0_med:.4f}")
    print("teacher alignment (medians over checkpoints/seeds):")
    for a in ALPHAS:
        all_d = [d for s in SEEDS for d in diags[a][s]]
        print(f"  a={a:<4} cos(g_boot,g_exact) "
              f"{np.median([d['cos_g'] for d in all_d]):+.3f}  "
              f"|g| ratio {np.median([d['nrat_g'] for d in all_d]):.2f}  "
              f"cos(r_boot,r_exact) "
              f"{np.median([d['cos_r'] for d in all_d]):+.3f}  "
              f"|r| ratio {np.median([d['nrat_r'] for d in all_d]):.2f}")

    wins = {a: sum(finals[a][s] < fC[s] for s in SEEDS)
            for a in ALPHAS if a > 0}
    print(f"paired wins vs PC0: {wins}")
    pred = any(med[a] < pc0_med and wins[a] >= 3 for a in wins)
    print(f"PRIMARY PREDICTION (L_alpha>0 < PC0 median on most paired "
          f"seeds): {'CONFIRMED' if pred else 'NOT CONFIRMED'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, alphas=ALPHAS,
                           beta=BETA, checkpoints=CHECKPOINTS),
               gates=dict(g1_bitwise=gate1,
                          g2_training_bptt=train_bptt,
                          g3_step1_identical=bool(gate3)),
               diagnostic_probes=n_probes,
               finals={str(a): {str(s): finals[a][s] for s in SEEDS}
                       for a in ALPHAS},
               medians={str(a): med[a] for a in ALPHAS},
               diags=diags, wins_vs_pc0=wins, prediction=bool(pred))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2, default=lambda o: o.item()
                  if isinstance(o, (np.floating, np.integer)) else o)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
