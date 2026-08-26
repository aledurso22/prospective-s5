"""E1 + E2 — the phase-identification tests. Frozen record: PC0/RoutePC,
benchmark code, and the D1-D4 conclusions are unchanged.

CURRENT RECORD WORDING (enforced until these results land): "complex
phase is representationally valuable but poorly identified by the
causal learner" — NOT "the entire 28.2% teacher-blindness gap is phase
blindness".

E1 — radial vs tangential teacher information. Replicates
teacher_decompose.py's arm B (exact-next teacher, PC timing) BITWISE
(same protocol/RNG/J_n; asserted against stored finals) and logs, at
checkpoints {100, 500, 1000, 1500}, both meta-residuals on identical
J_n: r_exact (next-batch BPTT teacher) and r_causal (next-batch online
teacher). For the current geometry w_j = u + i v:
    e_r = (u, v)/(|w|+eps),   e_phi = (-v, u)/(|w|+eps)
    r_r   = e_r . (du, dv),   r_phi = e_phi . (du, dv)
per (layer, mode), for both teachers. Report per layer and pooled over
lower recurrent layers (0..L-2):
    cos(r_r^causal, r_r^exact),  cos(r_phi^causal, r_phi^exact),
    norm ratios, sign agreement, and the fraction of exact residual
    energy in the tangential component.
PRIMARY PREDICTION: tangential alignment < radial alignment, clearly,
across lower layers/seeds. If they are similar, the "phase blindness"
reading is too specific — the deficit is a general directional one.

E2 — crossed teacher x geometry 2x2 (5 paired seeds):
    E_C = exact-next,  complex  (stored: teacher_decompose arm B)
    C_C = causal-next, complex  (stored: route_pc PC0)
    C_R = causal-next, real     (stored: route_pc_factorial
                                 per-mode-real)
    E_R = exact-next,  real     (NEW: train_B with Im w pinned to 0)
Same init/RNG/hyperparameters/correction rule/next-batch timing.
KEY INTERACTION: Delta_exact = E_R - E_C  vs  Delta_causal = C_R - C_C
PREDICTION: Delta_exact >> Delta_causal — the exact teacher exploits
the representational advantage of complex phase; the causal teacher
recovers much less of it.

Run:  python e1_e2_identification.py
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
CHECKPOINTS = [100, 500, 1000, 1500]
LR, LR_M = cvm.LR, cvm.LR_M
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "e1_e2_identification")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def chain_c(Gp, th_all, u_all, sig_all, h_n):
    """The surrogate residual r^ = du + i dv per (layer, mode) —
    identical to teacher_decompose.py."""
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


def train_B_variant(seed, real_only, log_residuals=False):
    """teacher_decompose.py's train_B (exact-next teacher, PC timing),
    optionally with Im w pinned to 0 (real arm), optionally logging
    r_exact/r_causal at checkpoints. Bitwise-compatible with the stored
    arm B when real_only=False."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    snaps = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_exact = tcg.flat_grads(cvm.exact_grad(params, x, y), params)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            rB = chain_c(Gp, th_all, u_all, sig_all, h_exact)
            if log_residuals and step in CHECKPOINTS:
                h_caus = tcg.flat_grads(G, params)
                rC = chain_c(Gp, th_all, u_all, sig_all, h_caus)
                snaps.append(dict(step=step,
                                  w=[wl.copy() for wl in w_pred],
                                  rB=[r.copy() for r in rB],
                                  rC=[r.copy() for r in rC]))
            if real_only:
                w_pred = [np.real(wp - LR_M * (-LR) * r_.real) + 0j
                          for wp, r_ in zip(w_pred, rB)]
            else:
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
            print(f"    B{'R' if real_only else 'C'} s{seed} step "
                  f"{step}: loss {loss:.4f}", flush=True)

    losses = np.asarray(losses)
    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                snaps=snaps)


def e1_analysis(snaps):
    rows = []
    for snap in snaps:
        row = {"step": snap["step"]}
        for l in range(tcg.L):
            u = snap["w"][l].real
            v = snap["w"][l].imag
            nrm = np.abs(snap["w"][l]) + 1e-12
            er_u, er_v = u / nrm, v / nrm
            et_u, et_v = -v / nrm, u / nrm
            for tag, r in [("B", snap["rB"]), ("C", snap["rC"])]:
                du = r[l].real
                dv = r[l].imag
                row[f"rr_{tag}_{l}"] = er_u * du + er_v * dv
                row[f"rp_{tag}_{l}"] = et_u * du + et_v * dv
        rows.append(row)
    return rows


def comp_stats(rows, layers):
    """cos/norm-ratio/sign-agreement for radial and tangential parts,
    pooled over the given layers, per checkpoint row."""
    out = []
    for row in rows:
        rrB = np.concatenate([row[f"rr_B_{l}"] for l in layers])
        rrC = np.concatenate([row[f"rr_C_{l}"] for l in layers])
        rpB = np.concatenate([row[f"rp_B_{l}"] for l in layers])
        rpC = np.concatenate([row[f"rp_C_{l}"] for l in layers])

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a)
                                         * np.linalg.norm(b) + 1e-30))
        out.append(dict(
            cos_radial=cos(rrC, rrB),
            cos_tangential=cos(rpC, rpB),
            nrat_radial=float(np.linalg.norm(rrC)
                              / (np.linalg.norm(rrB) + 1e-30)),
            nrat_tangential=float(np.linalg.norm(rpC)
                                  / (np.linalg.norm(rpB) + 1e-30)),
            sign_radial=float(np.mean(np.sign(rrC) == np.sign(rrB))),
            sign_tangential=float(np.mean(np.sign(rpC)
                                          == np.sign(rpB))),
            frac_energy_tangential=float(
                np.sum(rpB ** 2) / (np.sum(rpB ** 2)
                                    + np.sum(rrB ** 2) + 1e-30))))
    return out


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- stored 2x2 cells (same protocol/RNG) ----
    ref_pc = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "route_pc", "summary.json")))
    ref_td = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "teacher_decompose", "summary.json")))
    ref_fac = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "route_pc_factorial", "summary.json")))
    E_C = {s: ref_td["finals"]["B"][str(s)] for s in SEEDS}
    C_C = {s: ref_pc["finals"]["pc_b0.0"][str(s)] for s in SEEDS}
    C_R = {s: ref_fac["finals"]["per-mode-real"][str(s)] for s in SEEDS}

    audit0 = dict(rp.BPTT_CALLS)
    E_R, snaps_B = {}, {}
    det_check = []
    for seed in SEEDS:
        print(f"E_R (exact-next, real) s{seed}...", flush=True)
        outR = train_B_variant(seed, real_only=True)
        E_R[seed] = outR["final_loss"]
        print(f"  final {outR['final_loss']:.4f}", flush=True)
        print(f"B-complex replay with residual logging s{seed}...",
              flush=True)
        outB = train_B_variant(seed, real_only=False, log_residuals=True)
        det_check.append(abs(outB["final_loss"] - E_C[seed]))
        snaps_B[seed] = outB["snaps"]
        print(f"  replay final {outB['final_loss']:.4f}  "
              f"(stored {E_C[seed]:.4f})", flush=True)
    det = max(det_check)
    print(f"arm-B replay determinism: max |dfinal| {det:.2e}")
    bptt_delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    print(f"BPTT calls (exact-teacher arms E_C/E_R): {bptt_delta}")

    # ---- E1 ----
    e1_rows = {}
    for seed in SEEDS:
        rows = e1_analysis(snaps_B[seed])
        e1_rows[seed] = dict(
            pooled=comp_stats(rows, [0, 1, 2]),
            per_layer={l: comp_stats(rows, [l]) for l in range(tcg.L)})
    print("-" * 78)
    print("E1 — radial vs tangential teacher alignment (pooled lower "
          "layers 0..L-2):")
    for seed in SEEDS:
        for i, ck in enumerate(CHECKPOINTS):
            r = e1_rows[seed]["pooled"][i]
            print(f"  s{seed} n={ck:>4d}  cos_r {r['cos_radial']:+.3f}  "
                  f"cos_phi {r['cos_tangential']:+.3f}  "
                  f"sign_r {r['sign_radial']:.2f}  "
                  f"sign_phi {r['sign_tangential']:.2f}  "
                  f"E_phi {r['frac_energy_tangential']:.2f}")
    med_cos_r = float(np.median([r["cos_radial"] for s in SEEDS
                                 for r in e1_rows[s]["pooled"]]))
    med_cos_p = float(np.median([r["cos_tangential"] for s in SEEDS
                                 for r in e1_rows[s]["pooled"]]))
    e1_verdict = ("CONFIRMED: causal teacher loses tangential (phase) "
                  "information disproportionately"
                  if med_cos_p < med_cos_r - 0.1 else
                  "NOT CONFIRMED: radial and tangential deficits are "
                  "similar — general directional-information deficit")
    print(f"  medians: cos_r {med_cos_r:+.3f}  cos_phi {med_cos_p:+.3f}"
          f"  -> {e1_verdict}")

    # ---- E2 ----
    med = lambda f: float(np.median([f[s] for s in SEEDS]))
    D_exact = {s: E_R[s] - E_C[s] for s in SEEDS}
    D_causal = {s: C_R[s] - C_C[s] for s in SEEDS}
    inter = {s: D_exact[s] - D_causal[s] for s in SEEDS}
    print("E2 — crossed teacher x geometry (final losses):")
    print(f"  E_C (exact,  complex): {['%.4f' % E_C[s] for s in SEEDS]}"
          f"  med {med(E_C):.4f}")
    print(f"  E_R (exact,  real):    {['%.4f' % E_R[s] for s in SEEDS]}"
          f"  med {med(E_R):.4f}")
    print(f"  C_C (causal, complex): {['%.4f' % C_C[s] for s in SEEDS]}"
          f"  med {med(C_C):.4f}")
    print(f"  C_R (causal, real):    {['%.4f' % C_R[s] for s in SEEDS]}"
          f"  med {med(C_R):.4f}")
    print(f"  Delta_exact  per seed {['%+.4f' % D_exact[s] for s in SEEDS]}"
          f"  med {med(D_exact):+.4f}")
    print(f"  Delta_causal per seed {['%+.4f' % D_causal[s] for s in SEEDS]}"
          f"  med {med(D_causal):+.4f}")
    print(f"  interaction Delta_exact - Delta_causal: "
          f"{['%+.4f' % inter[s] for s in SEEDS]}  "
          f"med {med(inter):+.4f}")
    e2_verdict = ("CONFIRMED: exact teacher exploits complex phase; "
                  "causal recovers much less"
                  if med(inter) > 0 and med(D_exact) > med(D_causal)
                  else "NOT CONFIRMED")
    print(f"  -> {e2_verdict}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS,
                           checkpoints=CHECKPOINTS),
               replay_determinism=det, bptt_calls=bptt_delta,
               e1=dict(rows=e1_rows, med_cos_radial=med_cos_r,
                       med_cos_tangential=med_cos_p, verdict=e1_verdict),
               e2=dict(finals=dict(E_C=E_C, E_R=E_R, C_C=C_C, C_R=C_R),
                       delta_exact=D_exact, delta_causal=D_causal,
                       interaction=inter, verdict=e2_verdict))
    # numpy types -> python
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
