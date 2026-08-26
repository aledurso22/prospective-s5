"""Registered prospective-objective sweep + teacher-deficit
autocorrelation. Frozen: all D1-D4 / E1/E2 records unchanged; no new
filters or geometry families.

PART 1 — prospective-objective kappa sweep (exact teacher).
At each step freeze (theta_n, w_n, g^on_n) and the realized prospective
update theta' = theta_n - eta M_{w_n} g^on_n (the registered routeA
update: clip + Adam). Evaluate the post-update meta-residual at the
SAME w_n on the current and next objectives, with the same J_n (the
chain built from theta_n, G_n):

    r^(0) = -eta J_n^dag g_BPTT(theta'; B_n)      (current objective;
                                                   kappa=0 == routeA,
                                                   bitwise gate)
    r^(1) = -eta J_n^dag g_BPTT(theta'; B_{n+1})  (next objective,
                                                   exact — oracle
                                                   diagnostic)
    r^(k) = (1-kappa) r^(0) + kappa r^(1)
          = r^(0) + kappa (r^(1) - r^(0))

kappa in {0, .5, 1, 1.5, 2, 4} — kappa=1 is the realized one-step
prospective correction; kappa>1 extrapolated prediction-correction.
Same 5 paired seeds, otherwise identical training. The batch buffer is
a one-step lookahead that consumes exactly one draw per step, so
kappa=0 reproduces the stored routeA finals BITWISE (asserted). Do not
tune beyond the registered grid.

PART 2 — teacher-deficit directional autocorrelation.
Replicates teacher_decompose.py's arm B (PC timing, exact-next
teacher) per seed, logging EVERY step both meta-residuals on identical
J_{n-1}: r_exact and r_causal. eps_n = r_causal - r_exact per (layer,
mode), projected onto the current geometry's radial (e_r) and
tangential (e_phi) directions. Report lag-1 directional
autocorrelation (Pearson, per mode, median; and pooled), separately
for radial and tangential components, full series and late half.
Decides whether the phase-teacher deficit is temporally PREDICTABLE
(ac1 above noise) or requires a better teacher (ac1 ~ 0).

No new algorithm until these two diagnostics land.

Run:  python prospective_kappa.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
import route_pc as rp
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2, 3, 4]
KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "prospective_kappa")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def chain_c(G, params, hN):
    """J_n chain (same-step): r = du + i dv per (layer, mode),
    identical to routeA's inline update."""
    out = []
    off = 0
    for l in range(tcg.L):
        th = params["theta"][l]
        u_mode = tcg.sig(params["rho"][l])
        sigp = u_mode * (1 - u_mode)
        A = G["a"][l] * np.exp(1j * th)
        Gb = G["b"][l]
        M_ = Gb.shape[1]
        gN_rho = hN[off:off + tcg.N]
        gN_theta = hN[off + tcg.N:off + 2 * tcg.N]
        gN_bre = hN[off + 2 * tcg.N:
                    off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
        gN_bim = hN[off + 2 * tcg.N + tcg.N * M_:
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


def train_kappa(seed, kappa):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    x_n, y_n = make_data(rng)
    for step in range(1, STEPS + 1):
        x_next, y_next = make_data(rng)
        loss, G = cvm.batch_grad(params, x_n, y_n)[:2]
        losses.append(loss)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
        hB = tcg.flat_grads(cvm.exact_grad(params_next, x_n, y_n),
                            params_next)
        rB = chain_c(G, params, hB)
        if kappa != 0.0:
            hN = tcg.flat_grads(cvm.exact_grad(params_next, x_next,
                                               y_next), params_next)
            rN = chain_c(G, params, hN)
            r_k = [(1.0 - kappa) * r0 + kappa * r1
                   for r0, r1 in zip(rB, rN)]
        else:
            r_k = rB
        w = [wl - LR_M * (-LR) * rk for wl, rk in zip(w, r_k)]
        params = params_next
        x_n, y_n = x_next, y_next
        if step % 500 == 0:
            print(f"    k={kappa} s{seed} step {step}: loss "
                  f"{loss:.4f}", flush=True)
    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))))


def eps_series(seed):
    """arm-B replay with per-step residual logging."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    eps = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        h_ex = tcg.flat_grads(cvm.exact_grad(params, x, y), params)
        h_on = tcg.flat_grads(G, params)
        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            thp = {"theta": th_all}
            # chain with STORED previous-step pieces (identical to arm B)
            rB = chain_c_stored(Gp, th_all, u_all, sig_all, h_ex)
            rC = chain_c_stored(Gp, th_all, u_all, sig_all, h_on)
            eps.append((step, [wl.copy() for wl in w_pred],
                        [r.copy() for r in rB], [r.copy() for r in rC]))
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
    return eps


def chain_c_stored(Gp, th_all, u_all, sig_all, hN):
    out = []
    off = 0
    for l in range(tcg.L):
        th = th_all[l]
        u_mode = u_all[l]
        sigp = sig_all[l]
        A = Gp["a"][l] * np.exp(1j * th)
        Gb = Gp["b"][l]
        M_ = Gb.shape[1]
        gN_rho = hN[off:off + tcg.N]
        gN_theta = hN[off + tcg.N:off + 2 * tcg.N]
        gN_bre = hN[off + 2 * tcg.N:
                    off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
        gN_bim = hN[off + 2 * tcg.N + tcg.N * M_:
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


def eps_ac1(eps):
    """lag-1 directional autocorrelation of eps = r_causal - r_exact,
    split radial/tangential by the logged geometry."""
    er = {}
    ep = {}
    for (step, w, rB, rC) in eps:
        for l in range(tcg.L):
            u = w[l].real
            v = w[l].imag
            nrm = np.abs(w[l]) + 1e-12
            d = [rc - rb for rc, rb in zip(rC[l], rB[l])]
            d = np.asarray(d)
            du, dv = d.real, d.imag
            er.setdefault(l, []).append((u * du + v * dv) / nrm)
            ep.setdefault(l, []).append((-v * du + u * dv) / nrm)
    out = {}
    for name, series in [("radial", er), ("tangential", ep)]:
        acs_full, acs_late = [], []
        for l in range(tcg.L):
            S = np.asarray(series[l])          # (T', N)
            for j in range(tcg.N):
                s = S[:, j]
                sd = np.std(s)
                if sd > 1e-12:
                    acs_full.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
                    sl = s[len(s) // 2:]
                    if np.std(sl) > 1e-12:
                        acs_late.append(float(np.corrcoef(sl[:-1],
                                                          sl[1:])[0, 1]))
        out[name] = dict(ac1_full=float(np.median(acs_full)),
                         ac1_late=float(np.median(acs_late)),
                         p90_full=float(np.percentile(
                             np.abs(acs_full), 90)))
    return out


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "route_pc", "summary.json")))
    fA = {s: ref["finals"]["routeA"][str(s)] for s in SEEDS}
    fO = {s: ref["finals"]["online"][str(s)] for s in SEEDS}
    fC = {s: ref["finals"]["pc_b0.0"][str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    finals = {}
    for kappa in KAPPAS:
        finals[kappa] = {}
        for seed in SEEDS:
            print(f"kappa={kappa} s{seed}...", flush=True)
            out = train_kappa(seed, kappa)
            finals[kappa][seed] = out["final_loss"]
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}", flush=True)

    gate = max(abs(finals[0.0][s] - fA[s]) for s in SEEDS)
    print(f"kappa=0 vs stored routeA: max |dfinal| {gate:.2e} "
          f"(bitwise expected)")
    bptt_delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    print(f"BPTT calls (exact-teacher sweep): {bptt_delta}")

    print("-" * 78)
    med = {k: float(np.median([finals[k][s] for s in SEEDS]))
           for k in KAPPAS}
    print("finals per kappa:")
    for k in KAPPAS:
        print(f"  k={k!r:<5} {['%.4f' % finals[k][s] for s in SEEDS]}  "
              f"med {med[k]:.4f}")
    mO = float(np.median([fO[s] for s in SEEDS]))
    print(f"references: online {mO:.4f}  routeA(k=0) {med[0.0]:.4f}  "
          f"PC0 {float(np.median([fC[s] for s in SEEDS])):.4f}")

    # ---- part 2 ----
    ac_rows = {}
    for seed in SEEDS:
        print(f"eps series s{seed}...", flush=True)
        eps = eps_series(seed)
        ac_rows[seed] = eps_ac1(eps)
        print(f"  radial ac1 {ac_rows[seed]['radial']['ac1_full']:+.3f}  "
              f"tangential ac1 "
              f"{ac_rows[seed]['tangential']['ac1_full']:+.3f}", flush=True)
    ac_med = {name: float(np.median([ac_rows[s][name]["ac1_full"]
                                     for s in SEEDS]))
              for name in ["radial", "tangential"]}
    ac_late = {name: float(np.median([ac_rows[s][name]["ac1_late"]
                                      for s in SEEDS]))
               for name in ["radial", "tangential"]}
    print("-" * 78)
    print(f"eps directional ac1 (full / late): radial "
          f"{ac_med['radial']:+.3f} / {ac_late['radial']:+.3f}   "
          f"tangential {ac_med['tangential']:+.3f} / "
          f"{ac_late['tangential']:+.3f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, kappas=KAPPAS),
               k0_gate=gate, bptt_calls=bptt_delta,
               finals={str(k): {str(s): finals[k][s] for s in SEEDS}
                       for k in KAPPAS},
               medians={str(k): med[k] for k in KAPPAS},
               references=dict(online=mO,
                               routeA=med[0.0],
                               pc0=float(np.median([fC[s]
                                                    for s in SEEDS]))),
               eps_ac1=ac_rows, eps_ac1_median=ac_med,
               eps_ac1_late=ac_late)
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(conv(doc), f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
