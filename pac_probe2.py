"""PAC probe v2 — Experiment 0 done properly.

Changes from v1 (per review):
  * identity gate scoped correctly: c* = sum conj(a)^k rho_q(k) holds
    EXACTLY only at the top layer (no cross-layer term). Enforced there
    (relative 1e-8, float64). At shallow layers the residual is the
    cross-layer term itself -> recorded as the P2 readout (expected to
    grow toward layer 0).
  * stronger all-layer gate: with b_t^l = q_t^l + routed(lam_t^{l+1}),
    lam_t^l = sum_k conj(a)^k b_{t+k} holds per mode EXACTLY at every
    layer (validates the stacked recursion incl. the einsum routing).
  * seed-reliability of arg w as P1 precondition + attenuation ceiling:
    report R(c*,w) and R(c*,w)/R_w.
  * selection-confound control: probe c* also at ONLINE-baseline params
    (matched steps); if it still predicts routeA's arg w, the phase is
    task+architecture, not a trajectory artifact.
  * FIR ceiling: per-mode causal FIR with TAPS in {1,2,3,4,6,8} fit by
    least squares (past/current q only); if 2-3 taps recover most of
    the scalar residual, the next method is a short FIR, not a null.

REGISTERED BARS (fixed before running):
  P5:  median |rho(1)| > 0.05 in >= 1 layer (else PAC dead).
  REL: cross-seed R_w > 0.5 in >= 1 layer (else no stable object to
       predict; P1 uninterpretable).
  P1:  R(c*_exact, w) > 0.5 x R_w in >= 1 layer, at TRAINED params.
  CTRL: R at online params >= 0.5 x R at trained params (phase is
       task/architecture, not trajectory artifact).
  P3l: AR(1) closure retains >= 80% of c*_exact's R.
  P2:  q-identity residual grows toward shallow layers (readout only).

Run:  python pac_probe2.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
from depth_law import train_cell
from decompose_w_final import make_data
from factorize_w import train_frozen

SEEDS = [0, 1, 2]
TAPS_GRID = [1, 2, 3, 4, 6, 8]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pac_probe2")
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "factorize_w")


def known_answer_test():
    keep = (tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH)
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = 1, 1, 16, 4, 1
    try:
        a = 0.9 * np.exp(0.3j)
        params = dict(rho=[np.zeros(1)], theta=[np.zeros(1)],
                      b=[np.ones((1, 1), complex)], c=np.ones(1, complex))
        params["a"] = [np.array([a])]
        q = [np.zeros((tcg.T, 1, 1), complex)]
        q[0][11, 0, 0] = 1.0
        lam = tcg.exact_lambda(params, q)[0][:, 0, 0]
        expect = np.array([np.conj(a) ** (11 - t) if t <= 11 else 0.0
                           for t in range(tcg.T)])
        err = np.max(np.abs(lam - expect))
        assert err < 1e-12, f"known-answer FAIL: {err}"
        print(f"  [gate] impulse known-answer: max err {err:.2e} PASS")
    finally:
        tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = keep


def autocorr(sig, max_lag, full_norm=False):
    """rho(k) per mode. full_norm=False: stationary estimator (each lag
    averaged over its T-k samples) for P5/AR(1) statistics.
    full_norm=True: every lag normalized by the FULL window, making
    c* = sum conj(a)^k rho(k) exact algebra whenever
    lam_t = sum_k conj(a)^k sig_{t+k} (the identity gate)."""
    T_, B, N_ = sig.shape
    if full_norm:
        denom = np.sum(np.abs(sig) ** 2, axis=(0, 1)) + 1e-300
        rho = np.zeros((max_lag, N_), complex)
        for k in range(max_lag):
            rho[k] = np.sum(sig[k:] * np.conj(sig[:T_ - k]),
                            axis=(0, 1)) / denom
        return rho, denom / (T_ * B)
    denom = np.mean(np.abs(sig) ** 2, axis=(0, 1)) + 1e-300
    rho = np.zeros((max_lag, N_), complex)
    for k in range(max_lag):
        rho[k] = np.mean(sig[k:] * np.conj(sig[:T_ - k]), axis=(0, 1)) / denom
    return rho, denom


def circ_R(dphi, wgt):
    z = np.exp(1j * dphi)
    return float(np.abs(np.sum(wgt * z) / np.sum(wgt)))


def rel_err(x, y):
    return float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-300))


def probe(params, w, seed):
    """All per-layer measurements at given params."""
    rng = np.random.RandomState(900 + seed)
    x, y = make_data(rng)
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)
    lam = tcg.exact_lambda(params, q)
    # exact recursion's instantaneous drive per layer:
    #   lam_t^l = drive_t^l + conj(a^l) lam_{t+1}^l   (exact_lambda)
    # with drive^l = routed(lam^{l+1}) for l < L-1, drive^{L-1} = q^{L-1}.
    # (spatial q^l is the k=0 term of routed(lam^{l+1}) -- do NOT add it.)
    b = [None] * tcg.L
    b[tcg.L - 1] = q[tcg.L - 1]
    for l in range(tcg.L - 2, -1, -1):
        b[l] = np.einsum("jm,tbj->tbm", params["b"][l + 1],
                         np.conj(lam[l + 1])).real
    rows = []
    for l in range(tcg.L):
        rho_q, denom_q = autocorr(q[l], tcg.T)
        rho_q_full, _ = autocorr(q[l], tcg.T, full_norm=True)
        rho_b_full, _ = autocorr(b[l], tcg.T, full_norm=True)
        ak = np.conj(params["a"][l])
        powers = np.stack([ak ** k for k in range(tcg.T)], axis=0)
        c_q = np.mean(lam[l] * np.conj(q[l]), axis=(0, 1)) / denom_q
        c_qid = (powers * rho_q_full).sum(axis=0)
        denom_b = np.mean(np.abs(b[l]) ** 2, axis=(0, 1)) + 1e-300
        c_b = np.mean(lam[l] * np.conj(b[l]), axis=(0, 1)) / denom_b
        c_bid = (powers * rho_b_full).sum(axis=0)
        c_ar1 = 1.0 / (1.0 - ak * rho_q[1])
        resid = 1.0 - np.abs(c_q) ** 2 * denom_q / (
            np.mean(np.abs(lam[l]) ** 2, axis=(0, 1)) + 1e-300)
        # FIR sweep (causal, past/current q only)
        fir_resid = {}
        for taps in TAPS_GRID:
            X = np.stack([np.roll(q[l], k, axis=0) for k in range(taps)],
                         axis=-1)
            X[:taps - 1] = 0.0
            Xf = X[taps - 1:].reshape(-1, taps)
            yv = lam[l][taps - 1:].reshape(-1)
            f, res, *_ = np.linalg.lstsq(Xf, yv, rcond=None)
            pred = Xf @ f
            fr = 1.0 - float(np.sum(np.abs(yv - pred) ** 2)
                             / (np.sum(np.abs(yv) ** 2) + 1e-300))
            fir_resid[taps] = 1.0 - fr
        rows.append(dict(
            layer=l,
            med_rho1=float(np.median(np.abs(rho_q[1]))),
            id_err_top=float(rel_err(c_q, c_qid)),
            id_err_b=float(rel_err(c_b, c_bid)),
            R_exact=circ_R(np.angle(c_q) - np.angle(w[l]), denom_q),
            R_ar1=circ_R(np.angle(c_ar1) - np.angle(w[l]), denom_q),
            med_abs_dphi=float(np.median(np.abs(
                np.angle(c_q) - np.angle(w[l])))),
            med_resid=float(np.median(np.clip(resid, 0, 1))),
            fir_resid={str(k): float(v) for k, v in fir_resid.items()},
            denom=denom_q.tolist(),
            arg_w=np.angle(w[l]).tolist(),
        ))
    return rows, w


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    known_answer_test()
    trained, online_params, ws = {}, {}, []
    for seed in SEEDS:
        print(f"seed {seed}: routeA train + probe...", flush=True)
        params, w = train_cell(4, 50, seed)
        ws.append(w)
        w_saved = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        det = max(float(np.max(np.abs(w[l] - w_saved[l])))
                  for l in range(tcg.L))
        print(f"  [gate] determinism: {det:.2e}", flush=True)
        trained[seed], _ = probe(params, w, seed)
        print(f"seed {seed}: online-baseline train + probe...", flush=True)
        params_on, _ = train_online(seed)
        online_params[seed], _ = probe(params_on, w, seed)

    # cross-seed reliability of arg w (weighted by each seed's denom)
    print("-" * 70)
    rel = {}
    for l in range(tcg.L):
        Rs = []
        for s1 in range(len(SEEDS)):
            for s2 in range(s1 + 1, len(SEEDS)):
                d1 = np.array(trained[SEEDS[s1]][l]["denom"])
                d2 = np.array(trained[SEEDS[s2]][l]["denom"])
                wgt = np.sqrt(d1 * d2)
                a1 = np.array(trained[SEEDS[s1]][l]["arg_w"])
                a2 = np.array(trained[SEEDS[s2]][l]["arg_w"])
                Rs.append(circ_R(a1 - a2, wgt))
        rel[l] = float(np.mean(Rs))
        print(f"  L{l}: cross-seed R_w {rel[l]:.3f}")
    for seed in SEEDS:
        for rows, tag in ((trained[seed], "trained"),
                          (online_params[seed], "online ")):
            for r in rows:
                l = r["layer"]
                print(f"  s{seed} {tag} L{l}: |rho1| {r['med_rho1']:.3f}  "
                      f"R(c*,w) {r['R_exact']:.3f}  R(ar1,w) {r['R_ar1']:.3f}  "
                      f"q-id {r['id_err_top']:.1e}  b-id {r['id_err_b']:.1e}  "
                      f"resid {r['med_resid']:.3f}  "
                      f"fir3 {r['fir_resid']['3']:.3f}", flush=True)

    flat_tr = [r for rows in trained.values() for r in rows]
    flat_on = [r for rows in online_params.values() for r in rows]
    p5 = any(r["med_rho1"] > 0.05 for r in flat_tr)
    relobj = any(v > 0.5 for v in rel.values())
    p1 = any(r["R_exact"] > 0.5 * rel[r["layer"]] for r in flat_tr)
    ctrl = (np.median([r["R_exact"] for r in flat_on])
            >= 0.5 * np.median([r["R_exact"] for r in flat_tr]))
    p3l = all(r["R_ar1"] >= 0.8 * r["R_exact"] for r in flat_tr)
    gate_top = all(r["id_err_top"] < 1e-8 for rows in trained.values()
                   for r in rows if r["layer"] == tcg.L - 1)
    gate_b = all(r["id_err_b"] < 1e-8 for r in flat_tr)
    p2_grows = bool(
        np.median([r["id_err_top"] for r in flat_tr
                   if r["layer"] < tcg.L - 1])
        > 10 * np.median([r["id_err_top"] for r in flat_tr
                          if r["layer"] == tcg.L - 1]))
    print("-" * 70)
    for name, ok in [("P5 whiteness", p5), ("REL seed-reliability", relobj),
                     ("P1 adjoint orientation", p1), ("CTRL online-params", ctrl),
                     ("P3l AR(1) closure", p3l), ("GATE top-layer identity", gate_top),
                     ("GATE all-layer b-identity", gate_b),
                     ("P2 residual grows shallow", p2_grows)]:
        print(f"  {name:<28s} {'PASS' if ok else 'FAIL'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS, reliability=rel,
               trained={str(s): trained[s] for s in SEEDS},
               online={str(s): online_params[s] for s in SEEDS},
               bars={k: bool(v) for k, v in
                     dict(P5=p5, REL=relobj, P1=p1, CTRL=ctrl, P3l=p3l,
                          gate_top=gate_top, gate_b=gate_b,
                          P2=p2_grows).items()})
    doc["trained"] = {str(s): [{k: v for k, v in r.items()
                                if k not in ("denom", "arg_w")}
                               for r in trained[s]] for s in SEEDS}
    doc["online"] = {str(s): [{k: v for k, v in r.items()
                               if k not in ("denom", "arg_w")}
                              for r in online_params[s]] for s in SEEDS}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


def train_online(seed):
    """Online-baseline params at matched steps (w=1 frozen)."""
    w1 = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    # train_frozen returns final loss only; replicate its loop for params
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    import co_variational_metric as cvm
    from depth_law import STEPS
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        g = cvm.clip(tcg.flat_grads(G, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    return params, None


if __name__ == "__main__":
    main()
