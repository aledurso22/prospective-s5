"""STAGE C — spectrum: is the per-mode-real vs per-mode-complex tie a
low effective-frequency phenomenon?

The factorial (route_pc_factorial.py) found per-mode-real beats
per-mode-complex on 3/5 paired seeds. Hypothesis to test: the error
signal at each mode is effectively BASEBAND after demodulation
(low effective frequency), so the optimal phase correction arg c* is
near zero and constraining arg w = 0 (the real arm) is nearly optimal.

Per (layer, mode) at routeA-trained params (5 seeds, 16 probe batches):
  c_j^stat = sum_k conj(a_j)^k rho_j(k)        (the general prediction —
             kept as primary; phase_probes.py machinery)
  phi_j    = arg c_j^stat                       (predicted optimal phase)
  nu_j     = arg rho'_j(1) in the DEMODULATED frame (effective baseband
             frequency of the mode's error signal)
  bw_j     = 1 - |rho'_j(1)|                    (effective bandwidth)
  weight   = E|q_j|^2

Registered readings:
  * if the energy-weighted distribution of |phi_j| is concentrated near
    0, the real/complex tie is consistent with a low effective-frequency
    regime (the optimal correction is mostly real);
  * for narrowband modes (bw_j small), test the baseband prospective
    relation explicitly: c_baseband = c0*(r) applied to the real part —
    compare against c_j^stat per mode (do not blindly apply a DC
    formula to a complex pole);
  * report correlation between |phi_j| and |nu_j|.

Run:  python spectrum_check.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.probes import make_data
from diagnostics.phase_probes import probe_cstar, setup as pp_setup

SEEDS = [0, 1, 2, 3, 4]
M_SPEC = 16
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "spectrum_check")


def train_routeA(seed):
    out = cvm.train_route("routeA", seed)
    return out["w_final"], out["final_loss"]


def probe_spectrum(params, rng):
    """c_j^stat plus q'_j lag-1 autocorrelation (demodulated frame)."""
    cstar, rho = probe_cstar(params, rng)
    # demodulated lag-1 autocorrelation and E|q|^2 per mode
    nu = [[0.0] * tcg.N for _ in range(tcg.L)]
    bw = [[0.0] * tcg.N for _ in range(tcg.L)]
    eq = [np.zeros(tcg.N) for _ in range(tcg.L)]
    for _ in range(M_SPEC):
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        for l in range(tcg.L):
            a = np.asarray(params["a"][l])
            T = q[l].shape[0]
            theta = np.angle(a)
            ql = np.asarray(q[l], np.complex128)
            qp = ql * np.exp(-1j * np.outer(np.arange(T), theta))[:,
                                                              None, :]
            num1 = np.mean(qp[1:] * np.conj(qp[:-1]), axis=(0, 1))
            num0 = np.mean(np.abs(qp) ** 2, axis=(0, 1)) + 1e-30
            r1 = num1 / num0
            for j in range(tcg.N):
                nu[l][j] += float(np.angle(r1[j])) / M_SPEC
                bw[l][j] += float(1 - abs(r1[j])) / M_SPEC
            eq[l] += np.mean(np.abs(ql) ** 2, axis=(0, 1)) / M_SPEC
    return cstar, nu, bw, eq


def main() -> None:
    pp_setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain...", flush=True)
        # retrain routeA (registered protocol) and probe at final params
        import route_pc as _rp  # audit wrappers (unused here)
        params = tcg.init_params(seed)
        rng = np.random.RandomState(1000 + seed)
        w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
        flat = tcg.flatten(params)
        m = np.zeros_like(flat)
        v = np.zeros_like(flat)
        from toyrig.train_cell import STEPS
        for step in range(1, STEPS + 1):
            x, y = make_data(rng)
            loss, G = cvm.batch_grad(params, x, y)[:2]
            G_use = cvm.scale_by_w(G, w)
            g = cvm.clip(tcg.flat_grads(G_use, params))
            flat, m, v = cvm.adam(flat, g, m, v, step)
            params_next = tcg.pack(params, flat)
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
                            off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
            params = params_next
        prng = np.random.RandomState(333000 + seed)
        cstar, nu, bw, eq = probe_spectrum(params, prng)
        for l in range(tcg.L):
            for j in range(tcg.N):
                rows.append(dict(seed=seed, layer=l, mode=j,
                                 phi=float(np.angle(cstar[l][j])),
                                 abs_cstar=float(np.abs(cstar[l][j])),
                                 nu=nu[l][j], bw=bw[l][j],
                                 weight=float(eq[l][j])))
        print(f"  done; median |arg c*| "
              f"{np.median([abs(r['phi']) for r in rows if r['seed'] == seed]):.3f}",
              flush=True)

    wt = np.array([r["weight"] for r in rows])
    wt = wt / wt.sum()
    phi = np.abs(np.array([r["phi"] for r in rows]))
    nu = np.abs(np.array([r["nu"] for r in rows])
                )
    bw = np.array([r["bw"] for r in rows])
    med_phi = float(np.sum(wt * phi))
    p90_phi = float(np.sort(phi)[int(0.9 * (len(phi) - 1))])
    small = float(np.sum(wt[phi < np.pi / 8]))
    corr = float(np.corrcoef(phi, nu)[0, 1])
    # per-layer medians
    per_layer = {}
    for l in range(tcg.L):
        idx = [i for i, r in enumerate(rows) if r["layer"] == l]
        wl = wt[idx] / wt[idx].sum()
        per_layer[l] = dict(
            med_abs_phi=float(np.sum(wl * phi[idx])),
            med_nu=float(np.median(nu[idx])),
            med_bw=float(np.median(bw[idx])),
            frac_small=float(np.sum(wl[phi[idx] < np.pi / 8])))
    print("-" * 78)
    print(f"weighted mean |arg c*| {med_phi:.3f} rad  "
          f"(unweighted p90 {p90_phi:.3f})")
    print(f"weight fraction with |arg c*| < pi/8: {small:.2%}")
    print(f"corr(|arg c*|, |nu|): {corr:+.3f}")
    for l in per_layer:
        p = per_layer[l]
        print(f"  L{l}: mean|phi| {p['med_abs_phi']:.3f}  "
              f"median|nu| {p['med_nu']:.4f}  median bw {p['med_bw']:.4f} "
              f" frac small {p['frac_small']:.2%}")
    reading = ("LOW EFFECTIVE FREQUENCY — the real/complex tie is "
               "consistent with a near-baseband error signal"
               if med_phi < 0.2 else
               "phase content is material — the tie is NOT explained by "
               "low effective frequency")
    print(f"READING: {reading}  (bar: weighted mean |arg c*| < 0.2 rad)")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS,
               weighted_mean_abs_phi=med_phi, p90_abs_phi=p90_phi,
               frac_weight_small=small, corr_phi_nu=corr,
               per_layer=per_layer, reading=reading, rows=rows[:200])
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
