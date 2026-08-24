"""Orientation-only Wiener — the decisive control (north-star Directive 2).

wiener_oracle showed: static credit cosine rises to 0.73 with filter
horizon, but deploying the full K=64 Wiener filter destroys training
(-6.97, one seed 26x). The whole program's arc says orientation is the
useful part and gain is the poison. Missing control: deploy ONLY the
Wiener credit's orientation, keeping the online gradient's magnitude:

  lambda_hat = K * q   (frozen causal filter, estimated at trained params)
  r_t = exp(i (arg lambda_hat_t - arg q_t))
  err_t = r_t * q_t    (|err| = |q| — no Wiener gain anywhere)

If long-history orientation survives deployment, the Wiener gain was
the sole reason for the catastrophe, and a new algorithm exists. If
not, routeA's advantage is trajectory co-adaptation, not longer-history
orientation.

Arms (paired seeds {0,1,2}, frozen K from trained params):
  orientK, K in {1, 4, 16, 32, 64, 96}
  clipK64 (full Wiener K=64, per-mode RMS gain capped to q's RMS)
References: online/routeA from pac_deploy/summary.json; full K=64 from
wiener_oracle/summary.json; bptt from tbptt_baseline/summary.json.
Static credit cosine per arm recorded for the D3 accuracy-vs-stability
plot.

REGISTERED BAR (fixed before running): any orientK closes >= 50% of
the online -> routeA gap with all seeds finite  =>
"long-history orientation is useful; Wiener gain was the sole poison".
All orientK < 20% => routeA's advantage is trajectory co-adaptation.

Run:  python orient_wiener.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import train_cell, STEPS
from decompose_w_final import make_data
from wiener_oracle import wiener_fit

SEEDS = [0, 1, 2]
K_GRID = [1, 4, 16, 32, 64, 96]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "orient_wiener")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")
WIENER_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                              "wiener_oracle", "summary.json")


def apply_filter(q, f):
    """Causal FIR: (K*q)_t per mode, complex out."""
    K = f.shape[1]
    out = np.zeros(q.shape, dtype=np.complex128)
    for k in range(K):
        out[k:] += f[:, k][None, None, :] * q[:tcg.T - k]
    return out


def deploy(arm, seed, filters):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses, cosines = [], []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        err = []
        for l in range(tcg.L):
            lh = apply_filter(q[l], filters[l])
            if arm.startswith("orient"):
                r_ = np.exp(1j * (np.angle(lh) - np.angle(q[l])))
                r_ = np.where(np.abs(q[l]) > 1e-12, r_, 1.0)
                err.append(r_ * q[l])
            elif arm == "clipK64":
                rms_q = np.sqrt(np.mean(np.abs(q[l]) ** 2, axis=(0, 1)))
                rms_l = np.sqrt(np.mean(np.abs(lh) ** 2, axis=(0, 1)))
                err.append(lh * (rms_q / (rms_l + 1e-300))[None, None, :])
            else:
                err.append(lh)
        if step % 400 == 0:
            lam = tcg.exact_lambda(params, q)
            num = sum(float(np.abs(np.vdot(lam[l].ravel(),
                                           err[l].ravel())))
                      for l in range(tcg.L))
            den = sum(float(np.linalg.norm(lam[l].ravel())
                            * np.linalg.norm(err[l].ravel()))
                      for l in range(tcg.L))
            cosines.append(num / (den + 1e-300))
        Sa, Sb = tcg.sensitivities(params, h, x)
        G_use = tcg.assemble(params, h, x, r, err, Sa, Sb)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return dict(final=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                cosine=float(np.mean(cosines)) if cosines else None)


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    arms = [f"orient{K}" for K in K_GRID] + ["clipK64"]
    table, cos_table = {}, {}
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + filter estimation...",
              flush=True)
        params, w = train_cell(4, 50, seed)
        rng = np.random.RandomState(900 + seed)
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        lam = tcg.exact_lambda(params, q)
        filters = {}
        for K in K_GRID:
            filters[K] = [wiener_fit(q[l], lam[l], K)[0]
                          for l in range(tcg.L)]
        for arm in arms:
            K = 64 if arm == "clipK64" else int(arm[6:])
            out = deploy(arm, seed, filters[K])
            table.setdefault(arm, []).append(out["final"])
            cos_table.setdefault(arm, []).append(out["cosine"])
            print(f"  {arm:<10s} final {out['final']:.4f} "
                  f"finite {out['finite']} cos {out['cosine']:.3f}",
                  flush=True)

    med = {a: float(np.median(v)) for a, v in table.items()}
    gap = ref["online"] - ref["routeA"]
    fracs = {a: (ref["online"] - med[a]) / gap for a in med}
    best = max(fracs, key=lambda a: fracs[a])
    finite_ok = all(np.isfinite([table[a][i] for a in arms
                                 for i in range(len(SEEDS))]))
    win = fracs[best] >= 0.5 and finite_ok
    print("-" * 70)
    print(f"refs: online {ref['online']:.4f}  routeA {ref['routeA']:.4f}")
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"fracs: { {k: round(v, 2) for k, v in fracs.items()} }")
    print(f"cosines: { {k: round(float(np.median(v)), 3) for k, v in cos_table.items()} }")
    print(f"BAR: any orientK >= 50% and all finite  ->  "
          f"{'LONG-HISTORY ORIENTATION WORKS' if win else 'NO WIN — routeA advantage is trajectory co-adaptation'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, refs=ref, per_arm=table, medians=med,
               fracs=fracs, cosines=cos_table, win=bool(win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
