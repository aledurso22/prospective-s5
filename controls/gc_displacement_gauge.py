"""GC — rho vs actual displacement + common-scale invariance probe
(refinement C).

Global clipping fires on 100% of steps, so the actual parameter
displacement is NOT generally proportional to rho ||g||. Two
measurements on the frozen rig (5 PC0 seeds, extra logging, bitwise
gate vs stored finals):

  1. correlations of the per-step mean modal radius rho(t) with the
     ACTUAL ||theta_{n+1} - theta_n|| and with the (a,B)-block share of
     the pre-clip gradient (Pearson + Spearman per seed).
  2. common-scale invariance probe: multiply ALL w (every layer/mode)
     by a common positive kappa in {1, 3, 10, 100}. M_w acts only on
     the (a,B) gradient blocks — the c blocks are untouched — so the
     assembled clipped gradient is NOT exactly invariant. Measure, on a
     random (params, batch, adam-state): direction cosine of the
     post-clip gradient vs kappa = 1, and relative change of the Adam
     update (direction cosine + norm ratio). Control: scaling the FULL
     gradient (all blocks) — exactly invariant under clip by
     construction.

This quantifies "approximately redundant / weakly identified" vs
"exactly gauge" for the common radial direction.

Run:  python -m controls.gc_displacement_gauge
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy import stats

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig import routepc as rp
from toyrig.probes import make_data
from controls.geometry_traj import setup, train_arm

SEEDS = [0, 1, 2, 3, 4]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def invariance_probe():
    rng = np.random.RandomState(11)
    params = tcg.init_params(0)
    x, y = make_data(rng)
    loss, G = cvm.batch_grad(params, x, y)[:2]
    flat0 = tcg.flatten(params)
    m0 = np.abs(rng.normal(0, 0.01, flat0.shape))
    v0 = np.abs(rng.normal(0, 0.01, flat0.shape))
    w0 = [1.3 * np.exp(0.4j * np.linspace(-1, 1, tcg.N))
          for _ in range(tcg.L)]

    def upd_for(wlist):
        g = tcg.flat_grads(cvm.scale_by_w(G, wlist), params)
        gc = cvm.clip(g)
        _, mv, vv = cvm.adam(flat0, gc, m0.copy(), v0.copy(), 1)
        # recompute the update vector itself
        m_ = 0.9 * m0 + 0.1 * gc
        v_ = 0.999 * v0 + 0.001 * gc ** 2
        return 1e-3 * (m_ / 0.1) / (np.sqrt(v_ / (1 - 0.999)) + 1e-8), gc

    rows = []
    u1, g1 = upd_for(w0)
    for kappa in [1, 3, 10, 100]:
        uk, gk = upd_for([kappa * wl for wl in w0])
        cos_dir = float(np.dot(g1, gk)
                        / (np.linalg.norm(g1) * np.linalg.norm(gk)
                           + 1e-30))
        cos_upd = float(np.dot(u1, uk)
                        / (np.linalg.norm(u1) * np.linalg.norm(uk)
                           + 1e-30))
        rows.append(dict(kappa=kappa, clip_dir_cos=cos_dir,
                         adam_dir_cos=cos_upd,
                         adam_norm_ratio=float(
                             np.linalg.norm(uk)
                             / (np.linalg.norm(u1) + 1e-30))))
    # control: scale the FULL flat gradient (all blocks incl. c)
    g_full = tcg.flat_grads(G, params)
    g1f = cvm.clip(g_full)
    g100f = cvm.clip(100.0 * g_full)
    ctrl = float(np.dot(g1f, g100f)
                 / (np.linalg.norm(g1f) * np.linalg.norm(g100f)
                    + 1e-30))
    return rows, ctrl


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                      "summary.json")))
    audit0 = dict(rp.BPTT_CALLS)
    corr_rows = []
    for seed in SEEDS:
        out, traj = train_arm("pc0", seed, extra=True)
        assert out["final_loss"] == ref["finals"]["pc_b0.0"][str(seed)]
        rho = np.abs(traj["w"]).mean(axis=(1, 2))
        dt = traj["dtheta"]
        ab = traj["ab_share"]
        sl = slice(2, None)
        corr_rows.append(dict(
            seed=seed,
            pearson_rho_dtheta=float(
                stats.pearsonr(rho[sl], dt[sl]).statistic),
            spearman_rho_dtheta=float(
                stats.spearmanr(rho[sl], dt[sl]).statistic),
            pearson_rho_abshare=float(
                stats.pearsonr(rho[sl], ab[sl]).statistic),
            dtheta_med=float(np.median(dt[sl])),
            ab_share_med=float(np.median(ab[sl]))))
        r = corr_rows[-1]
        print(f"pc0 s{seed}: corr(rho, |dtheta|) pearson "
              f"{r['pearson_rho_dtheta']:+.3f} spearman "
              f"{r['spearman_rho_dtheta']:+.3f}  corr(rho, ab_share) "
              f"{r['pearson_rho_abshare']:+.3f}  |dtheta| med "
              f"{r['dtheta_med']:.4f}  ab_share med "
              f"{r['ab_share_med']:.4f}", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}

    probe, ctrl = invariance_probe()
    print("-" * 78)
    print("common-scale invariance probe (all w x kappa, c untouched):")
    for r in probe:
        print(f"  kappa {r['kappa']:>5.0f}: post-clip direction cos "
              f"{r['clip_dir_cos']:.6f}  adam update direction cos "
              f"{r['adam_dir_cos']:.6f}  adam norm ratio "
              f"{r['adam_norm_ratio']:.4f}")
    print(f"  control (FULL gradient x100, exact clip invariance): "
          f"direction cos {ctrl:.6f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, correlations=corr_rows, probe=probe,
               control_full_scale_cos=ctrl, probe_calls=audit)
    with open(os.path.join(OUT, "gc_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote gc_summary.json")


if __name__ == "__main__":
    main()
