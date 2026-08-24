"""PAC probe — P5 whiteness + Experiment 0: is the learned phase the
optimal scalar projection of exact credit onto causal credit?

Implements the decisive front of the PAC handoff brief in the existing
numpy rig. Conventions pinned by trained_credit_gains.py: online credit
signal is q = spatial_q (already Gamma-routed instantaneous credit);
exact credit is lam = exact_lambda(params, q) (already the stacked
adjoint with the instantaneous cross-layer term, fd-gated to 1e-4).

Quantities per (layer l, mode j), at TRAINED params (derive_phase
taught us init-time phases are the wrong ones):

  rho_j(1)  lag-1 autocorrelation of q_j            [P5 whiteness]
  c*_exact  E[lam conj(q)] / E|q|^2                 [Experiment 0 target]
  c*_id     sum_k conj(a)^k rho(k)                  [identity check of c*]
  c*_AR1    1 / (1 - conj(a) rho(1))                [the PAC closure law]
  resid     1 - |c*|^2 E|q|^2 / E|lam|^2            [scalar ceiling]

REGISTERED BARS (fixed before running):
  P5:  median |rho(1)| > 0.05 in >= 1 layer, else the PAC diagnosis of
       the derive_phase zero is wrong and PAC is dead on arrival.
  P1:  circular agreement R = |mean e^{i(arg c* - arg w)}| > 0.5 in
       >= 1 layer (weighted by E|q|^2), else the learned phase is NOT
       adjoint orientation and the PAC framing is rejected.
  P3l: arg c*_AR1 retains >= 80% of c*_exact's R per layer (AR(1)
       closure adequacy, pre-deployment).
  Convention gate: c*_exact == c*_id to 1e-6 (a flipped conj anywhere
       breaks this), plus the single-mode impulse known-answer test.

Run:  python pac_probe.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
from depth_law import train_cell
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pac_probe")
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "factorize_w")


def known_answer_test():
    """Single mode, impulse q at t0 -> lam_t = conj(a)^(t0-t), t <= t0."""
    keep = (tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH)
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = 1, 1, 16, 4, 1
    try:
        a = 0.9 * np.exp(0.3j)
        params = dict(rho=[np.zeros(1)], theta=[np.zeros(1)],
                      b=[np.ones((1, 1), complex)], c=np.ones(1, complex))
        params["a"] = [np.array([a])]
        q = [np.zeros((tcg.T, 1, 1), complex)]
        t0 = 11
        q[0][t0, 0, 0] = 1.0
        lam = tcg.exact_lambda(params, q)[0][:, 0, 0]
        expect = np.array([np.conj(a) ** (t0 - t) if t <= t0 else 0.0
                           for t in range(tcg.T)])
        err = np.max(np.abs(lam - expect))
        assert err < 1e-12, f"known-answer FAIL: {err}"
        print(f"  [gate] impulse known-answer: max err {err:.2e} PASS")
    finally:
        tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = keep


def autocorr(q, max_lag):
    """rho(k) per mode: E[q_{t+k} conj(q_t)] / E|q_t|^2, over t and batch."""
    T_, B, N_ = q.shape
    denom = np.mean(np.abs(q) ** 2, axis=(0, 1)) + 1e-300
    rho = np.zeros((max_lag, N_), complex)
    for k in range(max_lag):
        rho[k] = np.mean(q[k:] * np.conj(q[:T_ - k]), axis=(0, 1)) / denom
    return rho, denom


def circ_R(dphi, wgt):
    """Weighted mean resultant length of phase differences."""
    z = np.exp(1j * dphi)
    return float(np.abs(np.sum(wgt * z) / np.sum(wgt)))


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    known_answer_test()
    out = {}
    for seed in SEEDS:
        print(f"seed {seed}: training routeA...", flush=True)
        params, w = train_cell(4, 50, seed)
        w_saved = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        det = max(float(np.max(np.abs(w[l] - w_saved[l])))
                  for l in range(tcg.L))
        print(f"  [gate] determinism vs saved w_full: max diff {det:.2e}",
              flush=True)
        rng = np.random.RandomState(900 + seed)
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        lam = tcg.exact_lambda(params, q)

        rows = []
        for l in range(tcg.L):
            rho, denom = autocorr(q[l], tcg.T)
            num = np.mean(lam[l] * np.conj(q[l]), axis=(0, 1))
            c_exact = num / denom
            # identity: c* = sum_k conj(a)^k rho(k)
            ak = np.conj(params["a"][l])
            powers = np.stack([ak ** k for k in range(tcg.T)], axis=0)
            c_id = (powers * rho).sum(axis=0)
            ident_err = float(np.max(np.abs(c_exact - c_id)))
            c_ar1 = 1.0 / (1.0 - ak * rho[1])
            resid = 1.0 - np.abs(c_exact) ** 2 * denom / (
                np.mean(np.abs(lam[l]) ** 2, axis=(0, 1)) + 1e-300)
            dphi_exact = np.angle(c_exact) - np.angle(w[l])
            dphi_ar1 = np.angle(c_ar1) - np.angle(w[l])
            R_ex = circ_R(dphi_exact, denom)
            R_ar1 = circ_R(dphi_ar1, denom)
            rows.append(dict(
                layer=l,
                med_rho1=float(np.median(np.abs(rho[1]))),
                max_rho1=float(np.max(np.abs(rho[1]))),
                ident_err=ident_err,
                R_exact=R_ex, R_ar1=R_ar1,
                med_abs_dphi_exact=float(np.median(np.abs(dphi_exact))),
                med_abs_dphi_ar1=float(np.median(np.abs(dphi_ar1))),
                med_resid=float(np.median(np.clip(resid, 0, 1))),
            ))
            print(f"  L{l}: |rho1| med {rows[-1]['med_rho1']:.3f} "
                  f"max {rows[-1]['max_rho1']:.3f}  "
                  f"R(c*,w) {R_ex:.3f}  R(ar1,w) {R_ar1:.3f}  "
                  f"resid {rows[-1]['med_resid']:.3f}  "
                  f"id-err {ident_err:.1e}", flush=True)
        out[seed] = rows

    print("-" * 70)
    p5 = any(r["med_rho1"] > 0.05 for rows in out.values() for r in rows)
    p1 = any(r["R_exact"] > 0.5 for rows in out.values() for r in rows)
    p3l = all(r["R_ar1"] >= 0.8 * r["R_exact"]
              for rows in out.values() for r in rows)
    gate = all(r["ident_err"] < 1e-6 for rows in out.values() for r in rows)
    print(f"P5 whiteness (need med|rho1|>0.05 somewhere): {'PASS' if p5 else 'FAIL -> PAC dead'}")
    print(f"P1 adjoint orientation (need R>0.5 in >=1 layer): {'PASS' if p1 else 'FAIL -> PAC rejected'}")
    print(f"P3l AR(1) closure retains >=80% of R: {'PASS' if p3l else 'FAIL'}")
    print(f"convention gate (c* == autocorr identity, 1e-6): {'PASS' if gate else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS, per_seed={str(s): out[s] for s in SEEDS},
               bars=dict(P5=p5, P1=p1, P3l=p3l, convention_gate=gate))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
