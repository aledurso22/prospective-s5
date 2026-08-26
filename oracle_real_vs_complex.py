"""Per-mode REAL vs COMPLEX held-out oracle ceiling — is the causal
factorial tie representational or an identification failure?

Same protocol as gradient_cstat.py (identical retrains and probe
batches via identical prng): fit on batches 0..3, evaluate held-out on
batches 4..7, one scalar per (l, j) shared across the {Ga, Gb} block
group (the deployed w_j constraint):

  z_complex = sum conj(g_on) g_ex / sum |g_on|^2
  z_real    = Re[sum conj(g_on) g_ex] / sum |g_on|^2   (optimal real)

Held-out cos/rel for identity / z_real / z_complex. If
z_real ~= z_complex held-out, the real-vs-complex tie is
REPRESENTATIONAL (the family itself doesn't benefit from phase); if
z_complex >> z_real while learned-w training doesn't reach the complex
ceiling, it is identification.

Also: cg^stat (fit window) vs the held-out complex oracle, layer by
layer — arg MRL and |.| correlation (the key hypothesis
cg,j^stat ~= w_j^oracle, rather than c_q,j^stat ~= w_j), plus the same
against learned w.

Run:  python oracle_real_vs_complex.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from decompose_w_final import make_data
from prospective_offline2 import setup, train_routeA, blocks_vec
from gradient_cstat import gather, fit_scalars, grad_with_z, exact_vec, \
    align, mrl

SEEDS = [0, 1, 2, 3, 4]
BATCHES = 8
KFIT = 4
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "oracle_real_vs_complex")


def fit_oracles(packs, params):
    L, N = tcg.L, tcg.N
    zc = [np.zeros(N, np.complex128) for _ in range(L)]
    zr = [np.zeros(N) for _ in range(L)]
    for l in range(L):
        for j in range(N):
            num = 0.0 + 0.0j
            den = 0.0
            for p in packs:
                go = np.concatenate([[p["G_on"]["a"][l][j]],
                                     p["G_on"]["b"][l][j].ravel()])
                ge = np.concatenate([[p["G_ex"]["a"][l][j]],
                                     p["G_ex"]["b"][l][j].ravel()])
                num += np.vdot(go, ge)
                den += np.vdot(go, go).real
            zc[l][j] = num / (den + 1e-30)
            zr[l][j] = float(np.real(num) / (den + 1e-30))
    return zc, zr


def eval_arm(packs, z_map):
    cs, rs = [], []
    for p in packs:
        ge = exact_vec(p)
        gs = grad_with_z(p, z_map)
        c_, r_ = align(gs, ge)
        cs.append(c_)
        rs.append(r_)
    return float(np.median(cs)), float(np.median(rs))


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + probes...", flush=True)
        params, w_learned = train_routeA(seed)
        prng = np.random.RandomState(888000 + seed)
        packs = gather(params, prng)
        fitp, holdp = packs[:KFIT], packs[KFIT:]
        L = tcg.L
        zc_fit, zr_fit = fit_oracles(fitp, params)
        cg_fit, zo_fit = fit_scalars(fitp, params)
        ident = eval_arm(holdp, [np.ones(tcg.N, np.complex128)
                                 for _ in range(L)])
        real_h = eval_arm(holdp, [z.astype(np.complex128)
                                  for z in zr_fit])
        cplx_h = eval_arm(holdp, zc_fit)
        # cg^stat vs held-out-fitted complex oracle + learned w
        cmp = {}
        for l in range(L):
            wt = np.array([sum(np.abs(p["G_ex"]["a"][l][j]) ** 2
                               + np.sum(np.abs(p["G_ex"]["b"][l][j]) ** 2)
                               for p in packs) for j in range(tcg.N)])
            cmp[l] = dict(
                mrl_cg_vs_zoracle=mrl(cg_fit[l], zc_fit[l], wt),
                mrl_cg_vs_learned=mrl(cg_fit[l], w_learned[l], wt),
                magcorr_cg_vs_zoracle=float(np.corrcoef(
                    np.abs(cg_fit[l]), np.abs(zc_fit[l]))[0, 1]),
                magcorr_cg_vs_learned=float(np.corrcoef(
                    np.abs(cg_fit[l]), np.abs(w_learned[l]))[0, 1]))
        rows.append(dict(seed=seed, identity=ident, real=real_h,
                         complex=cplx_h, cmp=cmp))
        print(f"  held-out cos: identity {ident[0]:.3f}  "
              f"real {real_h[0]:.3f}  complex {cplx_h[0]:.3f}", flush=True)
        for l in range(L):
            c_ = cmp[l]
            print(f"    L{l}: MRL cg~zoracle {c_['mrl_cg_vs_zoracle']:.3f}"
                  f"  cg~learned {c_['mrl_cg_vs_learned']:.3f}"
                  f"  |.|corr cg~zoracle {c_['magcorr_cg_vs_zoracle']:+.3f}"
                  f"  cg~learned {c_['magcorr_cg_vs_learned']:+.3f}",
                  flush=True)

    med = lambda k, i: float(np.median([r[k][i] for r in rows]))
    print("-" * 78)
    print(f"held-out cos medians: identity {med('identity', 0):.3f}  "
          f"real {med('real', 0):.3f}  complex {med('complex', 0):.3f}")
    print(f"held-out rel medians: identity {med('identity', 1):.3f}  "
          f"real {med('real', 1):.3f}  complex {med('complex', 1):.3f}")
    repr_vs_ident = ("REPRESENTATIONAL tie (real ~= complex held-out)"
                     if med('real', 0) >= med('complex', 0) - 0.03
                     else "complex ceiling higher — identification gap "
                          "if learned w stays below")
    print(f"reading: {repr_vs_ident}")
    pair = {}
    for l in range(tcg.L):
        for k in rows[0]["cmp"][l]:
            pair[f"L{l}/{k}"] = float(np.median(
                [r["cmp"][l][k] for r in rows]))

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS, batches=BATCHES, kfit=KFIT,
               held_out=dict(identity=[med('identity', 0),
                                       med('identity', 1)],
                             real=[med('real', 0), med('real', 1)],
                             complex=[med('complex', 0),
                                      med('complex', 1)]),
               reading=repr_vs_ident, pair_medians=pair, rows=rows)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
