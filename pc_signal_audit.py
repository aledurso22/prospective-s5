"""PC signal audit — do delayed/online teachers carry the Route-A phase
signal at all? FROZEN measurement; no PC training here.

AMENDMENT GATE (fixed before running, from the directive): before any PC
training, audit the three candidate outer teachers on the real Route-A
trajectory. At checkpoint steps, for the same inner online update actually
taken by Route A (clip + Adam; theta' = realized post-update params):

    g = g_online(theta; B),  theta' = realized update
    h_BPTT = g_BPTT(theta'; B)       <- Route A's teacher (reference)
    h_same = g_online(theta'; B)     <- causal, same batch
    h_next = g_online(theta'; B')    <- causal, next batch (routePC's rule;
                                        B' from an independent stream so
                                        the audited trajectory stays
                                        identical to registered Route A)

Each teacher feeds the IDENTICAL analytic (u, v) chain (the FD-gated cvm
convention, verbatim):

    c_X = du_X + i dv_X     per (layer, mode)

Recorded per checkpoint/seed:
  * full-update cosine: cos(vec(c_X), vec(c_BPTT))  over the (2LN) real
    (du, dv) vector;
  * phase component per mode: phi_X,lj = Im(c_X,lj * conj(w_lj)) / |w_lj|
    (the arg-w increment direction, c_X ∝ Im(h_X^dagger g) per the
    directive); sign agreement vs phi_BPTT (fraction of modes; chance
    0.5), Pearson between the phi vectors, and phase-energy fraction
    sum(phi^2) / sum|c|^2 (is there any phase signal at all?);
  * alignment with the realized Route-A displacement: after training
    completes, dW = w_final - w_ckpt; cos(vec(c_X), vec(dW)).

GATES (registered here before running):
  * h_same is dead as a phase-learning teacher iff (median phase-energy
    fraction < 0.05 AND median |phi_same| / |phi_BPTT| < 0.1) OR median
    phase sign agreement <= 0.60.
  * same bar for h_next.
  * If a teacher passes, the corresponding PC arm may proceed; if both
    fail, stop the PC line and continue the registered Route-A plan.

Run:  python pc_signal_audit.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [1, 100, 250, 500, 750, 1000, 1500]
LR = cvm.LR
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pc_signal_audit")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def chain_c(G, params, hN):
    """The cvm (u, v) meta-gradient chain, verbatim, with teacher hN
    (flat real vector). Returns [c_l (N,) complex] per layer."""
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


def vec(c_list):
    """(du, dv) real vector over all layers/modes."""
    return np.concatenate([np.ravel(c.real) for c in c_list]
                          + [np.ravel(c.imag) for c in c_list])


def cosv(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-30 or nv < 1e-30:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def audit_run(seed):
    """Route-A retrain (registered protocol) with frozen teacher audits at
    the checkpoint steps."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    snaps = []
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)

        if step in CHECKPOINTS:
            # the three teachers at the realized theta'
            hB = tcg.flat_grads(cvm.exact_grad(params_next, x, y),
                                params_next)
            _, Gs, _, _, _ = cvm.batch_grad(params_next, x, y)
            hS = tcg.flat_grads(Gs, params_next)
            rng_next = np.random.RandomState(2000 + seed * 10000 + step)
            x2, y2 = make_data(rng_next)
            _, Gn, _, _, _ = cvm.batch_grad(params_next, x2, y2)
            hN = tcg.flat_grads(Gn, params_next)
            snaps.append(dict(
                step=step,
                w=[wl.copy() for wl in w],
                c_BPTT=chain_c(G, params, hB),
                c_same=chain_c(G, params, hS),
                c_next=chain_c(G, params, hN)))

        # ---- the registered Route-A w update (BPTT teacher) ----
        G_next = cvm.exact_grad(params_next, x, y)
        gN = tcg.flat_grads(G_next, params_next)
        off = 0
        for l in range(tcg.L):
            th = params["theta"][l]
            u_mode = tcg.sig(params["rho"][l])
            sigp = u_mode * (1 - u_mode)
            A = G["a"][l] * np.exp(1j * th)
            Gb = G["b"][l]
            M_ = Gb.shape[1]
            gN_rho = gN[off:off + tcg.N]
            gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
            gN_bre = gN[off + 2 * tcg.N:
                        off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
            gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                        off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
            off += 2 * tcg.N + 2 * tcg.N * M_
            du = (gN_rho * sigp * A.real
                  + gN_theta * (-u_mode) * A.imag
                  + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
            dv = (gN_rho * sigp * A.imag
                  + gN_theta * (u_mode) * A.real
                  + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            w[l] = w[l] - cvm.LR_M * (-LR) * (du + 1j * dv)
        params = params_next
        if step % 250 == 0:
            print(f"    seed {seed} step {step}: loss {loss:.4f}",
                  flush=True)
    return snaps, w, float(np.asarray(losses)[-100:].mean())


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain with frozen teacher audits...",
              flush=True)
        snaps, w_final, fin = audit_run(seed)
        print(f"  seed {seed}: final loss {fin:.4f} "
              f"({len(snaps)} checkpoints)", flush=True)
        for snap in snaps:
            dW = [wf - wc for wf, wc in zip(w_final, snap["w"])]
            row = dict(seed=seed, step=snap["step"], final_loss=fin)
            phi = {}
            for X in ["BPTT", "same", "next"]:
                c = snap[f"c_{X}"]
                # phase-increment component per mode
                ph = [np.imag(c[l] * np.conj(snap["w"][l]))
                      / np.maximum(np.abs(snap["w"][l]), 1e-12)
                      for l in range(tcg.L)]
                ph = np.concatenate(ph)
                phi[X] = ph
                row[f"phasefrac_{X}"] = float(
                    np.sum(ph ** 2) / max(np.sum(np.abs(
                        np.concatenate(c)) ** 2), 1e-30))
                row[f"absphi_{X}"] = float(np.mean(np.abs(ph)))
                row[f"cosdW_{X}"] = cosv(vec(c), vec(dW))
            for X in ["same", "next"]:
                row[f"cos_{X}"] = cosv(vec(snap[f"c_{X}"]),
                                       vec(snap["c_BPTT"]))
                agree = np.mean(np.sign(phi[X]) == np.sign(phi["BPTT"]))
                row[f"sign_{X}"] = float(agree)
                row[f"pearson_phi_{X}"] = float(np.corrcoef(
                    phi[X], phi["BPTT"])[0, 1])
            rows.append(row)

    # ---- aggregate ----
    def med(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    agg = {}
    for X in ["same", "next"]:
        agg[X] = dict(cos_teacher=med(f"cos_{X}"),
                      sign_agreement=med(f"sign_{X}"),
                      pearson_phi=med(f"pearson_phi_{X}"),
                      phase_energy=med(f"phasefrac_{X}"),
                      abs_phi_ratio=med(f"absphi_{X}") /
                      max(med("absphi_BPTT"), 1e-30),
                      cos_dW=med(f"cosdW_{X}"))
    agg["BPTT"] = dict(phase_energy=med("phasefrac_BPTT"),
                       cos_dW=med("cosdW_BPTT"))

    verdict = {}
    for X in ["same", "next"]:
        a = agg[X]
        dead_signal = (a["phase_energy"] < 0.05
                       and a["abs_phi_ratio"] < 0.1)
        no_agree = a["sign_agreement"] <= 0.60
        verdict[X] = ("DEAD" if (dead_signal or no_agree)
                      else "SIGNAL PRESENT")

    print("-" * 78)
    hdr = f"{'teacher':<8s} {'cos vs BPTT':>12s} {'sign agree':>11s} " \
          f"{'pearson phi':>12s} {'phase E':>8s} {'|phi| ratio':>12s} " \
          f"{'cos dW':>8s}"
    print(hdr)
    for X in ["same", "next"]:
        a = agg[X]
        print(f"h_{X:<5s} {a['cos_teacher']:12.3f} {a['sign_agreement']:11.3f} "
              f"{a['pearson_phi']:12.3f} {a['phase_energy']:8.3f} "
              f"{a['abs_phi_ratio']:12.3f} {a['cos_dW']:8.3f}   "
              f"-> {verdict[X]}")
    a = agg["BPTT"]
    print(f"h_BPTT  {'(ref)':>12s} {'':>11s} {'':>12s} "
          f"{a['phase_energy']:8.3f} {'':>12s} {a['cos_dW']:8.3f}")
    print("per-checkpoint detail:")
    for r in rows:
        print(f"  s{r['seed']} n={r['step']:>4d}  "
              f"cos_same {r['cos_same']:+.3f}  sign_same {r['sign_same']:.2f}"
              f"  cos_next {r['cos_next']:+.3f}  "
              f"sign_next {r['sign_next']:.2f}")
    print(f"GATE: h_same {verdict['same']};  h_next {verdict['next']}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, checkpoints=CHECKPOINTS,
                           L=4, N=16, T=128, delay=50, batch=32,
                           gate=("dead iff (phase energy < 0.05 and "
                                 "|phi| ratio < 0.1) or sign agree <= 0.60")),
               aggregate=agg, verdict=verdict, rows=rows)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
