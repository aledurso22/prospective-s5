"""Oracle lagged-deficit test — is the measured teacher-deficit
persistence ACTIONABLE, and is the actionable part tangential-specific?

ORACLE DIAGNOSTIC, NOT A DEPLOYABLE ALGORITHM: the deficit
eps_n = r_causal - r_exact requires the exact residual (BPTT) at every
step for bookkeeping. A deployable predictor would have to estimate eps
without BPTT; that is downstream and NOT decided here.

Design (registered):
  * PC0 timing (prev-step J_{n-1}, teacher evaluated at theta_n on the
    current batch) — identical to eps_perlayer/arm-B structure.
  * Fit window steps 2..FIT_END: all arms run the PLAIN causal arm and
    accumulate eps per (layer, mode, component); trajectories and the
    fitted coefficients are therefore identical across arms.
  * Held out from step FIT_END+1: per-mode least-squares lag-1
    coefficient rho_j (per component) from the fit window, then
        eps^_n = rho_j * eps_{n-1}   (component values in the current
        w frame; reconstructed and subtracted)
  * Four arms:
      A1  r_causal                         (== PC0; bitwise gate)
      A2  r_causal - eps^_r                (radial deficit only)
      A3  r_causal - eps^_phi              (tangential deficit only)
      A4  r_causal - eps^_r - eps^_phi     (full)
  * Five paired seeds, otherwise identical training (clip + Adam,
    LR_M/LR as every prior arm). Reference oracle ceiling: arm B
    (exact-next teacher, stored median 0.0014).

REGISTERED READS:
  gate: A1 bitwise == stored PC0 finals.
  P1 actionable: A4 median < A1 median AND >= 4/5 paired wins.
  P2 specificity: A3 captures >= 50% of the A1->A4 median improvement
     while A2 captures less  => tangential-specific; otherwise general.
  Null: A2..A4 ~= A1 => lag-1 prediction insufficient (persistence
     measured but not actionable at this order).

Run:  python oracle_lagged_deficit.py
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
FIT_END = 250
ARMS = ["A1", "A2", "A3", "A4"]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "oracle_lagged_deficit")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def frame(w_l):
    u, v = w_l.real, w_l.imag
    nrm = np.abs(w_l) + 1e-12
    return u / nrm, v / nrm


def components(r_l, w_l):
    er_u, er_v = frame(w_l)
    du, dv = r_l.real, r_l.imag
    return er_u * du + er_v * dv, -er_v * du + er_u * dv


def reconstruct(er_c, ep_c, w_l):
    er_u, er_v = frame(w_l)
    return (er_c * er_u - ep_c * er_v) + 1j * (er_c * er_v + ep_c * er_u)


def train_arm(seed, arm):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    eps_hist = []            # per step: (er, ep) per (l,)
    rho_r = rho_p = None
    eps_prev = None
    rho_used = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_on = tcg.flat_grads(G, params)
        need_exact = arm != "A1"
        h_ex = (tcg.flat_grads(cvm.exact_grad(params, x, y), params)
                if need_exact else None)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            rC = chain_c_stored(Gp, th_all, u_all, sig_all, h_on)
            r_use = rC
            if need_exact:
                rE = chain_c_stored(Gp, th_all, u_all, sig_all, h_ex)
                eps_now = [components(rc - re, wp)
                           for rc, re, wp in zip(rC, rE, w_pred)]
                eps_hist.append(eps_now)
                if step == FIT_END:
                    rho_r = []
                    rho_p = []
                    for l in range(tcg.L):
                        sr = np.array([eh[l][0] for eh in eps_hist])
                        sp = np.array([eh[l][1] for eh in eps_hist])
                        rho_r.append((sr[1:] * sr[:-1]).sum(axis=0)
                                     / ((sr[:-1] ** 2).sum(axis=0)
                                        + 1e-30))
                        rho_p.append((sp[1:] * sp[:-1]).sum(axis=0)
                                     / ((sp[:-1] ** 2).sum(axis=0)
                                        + 1e-30))
                    rho_used = [float(np.median(np.abs(np.concatenate(
                        [rr for rr in rho_r])))),
                        float(np.median(np.abs(np.concatenate(
                            [pp for pp in rho_p]))))]
                if step > FIT_END and eps_prev is not None:
                    r_use = []
                    for l in range(tcg.L):
                        er_hat = rho_r[l] * eps_prev[l][0]
                        ep_hat = rho_p[l] * eps_prev[l][1]
                        if arm == "A2":
                            ep_hat = np.zeros_like(ep_hat)
                        elif arm == "A3":
                            er_hat = np.zeros_like(er_hat)
                        corr = reconstruct(er_hat, ep_hat, w_pred[l])
                        r_use.append(rC[l] - corr)
                eps_prev = eps_now
            w_pred = [wp - LR_M * (-LR) * ru
                      for wp, ru in zip(w_pred, r_use)]

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
            print(f"    {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)

    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                rho_med=rho_used)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "route_pc", "summary.json")))
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}
    fB = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "teacher_decompose", "summary.json")))["finals"]["B"]
    fB = {s: fB[str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    finals = {a: {} for a in ARMS}
    rhos = {}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"{arm} s{seed}...", flush=True)
            out = train_arm(seed, arm)
            finals[arm][seed] = out["final_loss"]
            if out["rho_med"]:
                rhos[seed] = out["rho_med"]
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}

    gate = max(abs(finals["A1"][s] - fC[s]) for s in SEEDS)
    print(f"A1 vs stored PC0: max |dfinal| {gate:.2e} (bitwise expected)")
    print(f"BPTT calls (oracle arms A2-A4): {audit}")
    if rhos:
        print(f"fitted |rho| medians (radial, tangential): {rhos}")

    med = {a: float(np.median([finals[a][s] for s in SEEDS]))
           for a in ARMS}
    print("-" * 78)
    for a in ARMS:
        print(f"  {a}  {['%.4f' % finals[a][s] for s in SEEDS]}  "
              f"med {med[a]:.4f}")
    print(f"references: PC0(=A1) {med['A1']:.4f}  arm B (oracle ceiling) "
          f"{float(np.median([fB[s] for s in SEEDS])):.4f}")

    wins4 = sum(finals["A4"][s] < finals["A1"][s] for s in SEEDS)
    p1 = med["A4"] < med["A1"] and wins4 >= 4
    imp = med["A1"] - med["A4"]
    c3 = (med["A1"] - med["A3"]) / imp if imp > 0 else float("nan")
    c2 = (med["A1"] - med["A2"]) / imp if imp > 0 else float("nan")
    p2 = (c3 >= 0.5) and (c2 < c3) if imp > 0 else False
    print(f"P1 actionable (A4 < A1 median AND >=4/5 wins): {p1}  "
          f"(wins {wins4}/5)")
    if imp > 0:
        print(f"P2 shares of A1->A4 improvement: radial-only {c2:.2f}  "
              f"tangential-only {c3:.2f}  -> "
              f"{'TANGENTIAL-SPECIFIC' if p2 else 'GENERAL/mixed'}")
    else:
        print("P2: no A4 improvement to decompose")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, fit_end=FIT_END),
               gate=gate, bptt_calls=audit, rho_medians=rhos,
               finals={a: {str(s): finals[a][s] for s in SEEDS}
                       for a in ARMS},
               medians=med, wins_A4=wins4, p1_actionable=bool(p1),
               shares=dict(radial=c2, tangential=c3),
               p2_tangential_specific=bool(p2))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
