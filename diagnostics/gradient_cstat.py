"""Gradient-level analytic scalar + held-out per-mode oracle ceiling.

Follow-up to prospective_offline2.py (which is untouched). Two
measurements, same routeA-trained params per seed, 8 probe batches
split fit [0..3] / held-out [4..7]:

  1. cg^stat_j — the eligibility/routing-weighted analytic scalar
     implied by g_exact = sum_t conj(lam_t) s_t with s the exact
     eligibility factor (Sa/Sb in this rig; s_t = c e_t in the
     two-layer scalar derivation):

         R_j(k)  = sum_t conj(s_{t,j}) q_{t+k,j}     (pooled over the
                 {Ga, Gb} block group — the deployed sharing constraint)
         cg^stat_j = sum_{k>=0} conj(a_j)^k R_j(k) / R_j(0)

     In the rig's w-convention (g~ = conj(w) g_on), cg^stat plays the
     role of w directly:  conj(w) = z = conj(cg^stat).

  2. z_oracle_j — the complex least-squares scalar per (l, j) block
     group, FIT ON THE FIT WINDOW ONLY (batches 0..3), evaluated
     HELD-OUT (batches 4..7); in-window (all 8) reported for
     reference. This is the honest representation ceiling of the
     per-mode-complex family — no per-sample refit.

Comparisons (weighted by block energy): arg MRL and |.| correlations
between cg^stat, signal-level cstat, learned RoutePC/routeA w, and
z_oracle. Held-out gradient alignment (cos/rel) for identity, cg^stat,
and z_oracle arms.

Run:  python gradient_cstat.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.probes import make_data
from diagnostics.d1_exact_credit_factorization import (setup, train_routeA, blocks_vec,
                                  autocorr_cstat)

SEEDS = [0, 1, 2, 3, 4]
BATCHES = 8
KFIT = 4
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "gradient_cstat")


def gather(params, prng):
    packs = []
    for _ in range(BATCHES):
        x, y = make_data(prng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        lam = tcg.exact_lambda(params, q)
        G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        G_on = tcg.assemble(params, h, x, r, q, Sa, Sb)
        packs.append(dict(x=x, y=y, h=h, r=r, q=q, Sa=Sa, Sb=Sb,
                          G_ex=G_ex, G_on=G_on))
    return packs


def R_series(packs, l, j, K=60):
    """R_j(k) = sum_t conj(s_{t,j}) q_{t+k,j}, pooled over the Sa and Sb
    block entries of mode j, summed over the fit packs."""
    R = np.zeros(K + 1, np.complex128)
    for p in packs:
        q = np.asarray(p["q"][l], np.complex128)[:, :, j]     # (T, B)
        for s in (np.asarray(p["Sa"][l], np.complex128)[:, :, j],
                  np.asarray(p["Sb"][l], np.complex128)[:, :, j, 0]):
            sq = np.conj(s)                                     # (T, B)
            for k in range(K + 1):
                if k == 0:
                    R[k] += np.sum(sq * q)
                else:
                    R[k] += np.sum(sq[:-k] * q[k:])
    return R


def fit_scalars(packs, params, K=60):
    """cg^stat and z_oracle per (l, j) from the given packs."""
    L, N = tcg.L, tcg.N
    cg = [np.zeros(N, np.complex128) for _ in range(L)]
    zo = [np.zeros(N, np.complex128) for _ in range(L)]
    for l in range(L):
        a = np.asarray(params["a"][l])
        ks = np.arange(K + 1)[:, None]
        for j in range(N):
            R = R_series(packs, l, j, K)
            cg[l][j] = np.sum(np.conj(a[j]) ** ks.ravel() * R) / R[0]
            num = 0.0 + 0.0j
            den = 0.0
            for p in packs:
                go = np.concatenate([[p["G_on"]["a"][l][j]],
                                     p["G_on"]["b"][l][j].ravel()])
                ge = np.concatenate([[p["G_ex"]["a"][l][j]],
                                     p["G_ex"]["b"][l][j].ravel()])
                num += np.vdot(go, ge)
                den += np.vdot(go, go).real
            zo[l][j] = num / (den + 1e-30)
    return cg, zo


def grad_with_z(pack, z_map):
    """g~ = z * g_on per block group (z = conj(w) convention)."""
    L = tcg.L
    parts = []
    for l in range(L):
        za = z_map[l]
        parts.append(np.concatenate(
            [(za * pack["G_on"]["a"][l]).ravel(),
             (za[:, None] * pack["G_on"]["b"][l]).ravel()]))
    parts.append(np.ravel(pack["G_on"]["c"]))
    return np.concatenate(parts)


def exact_vec(pack):
    L = tcg.L
    return np.concatenate([blocks_vec(pack["G_ex"], l) for l in range(L)]
                          + [np.ravel(pack["G_ex"]["c"])])


def align(gs, ge):
    return (float(np.abs(np.vdot(ge, gs))
                  / (np.linalg.norm(ge) * np.linalg.norm(gs) + 1e-30)),
            float(np.linalg.norm(gs - ge) / (np.linalg.norm(ge) + 1e-30)))


def mrl(za, zb, wt):
    za = za / np.maximum(np.abs(za), 1e-30)
    zb = zb / np.maximum(np.abs(zb), 1e-30)
    wt = wt / (wt.sum() + 1e-30)
    return float(np.abs(np.sum(wt * za * np.conj(zb))))


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

        cg_fit, zo_fit = fit_scalars(fitp, params)
        cg_all, zo_all = fit_scalars(packs, params)
        cstat_sig = [autocorr_cstat(
            [np.asarray(p["q"][l], np.complex128) for p in packs],
            np.asarray(params["a"][l])) for l in range(L)]

        # ---- held-out alignment arms
        def eval_arm(z_map, held):
            cs, rs = [], []
            for p in held:
                ge = exact_vec(p)
                gs = grad_with_z(p, z_map)
                c_, r_ = align(gs, ge)
                cs.append(c_)
                rs.append(r_)
            return float(np.median(cs)), float(np.median(rs))

        cos_hold = dict(
            identity=np.median([align(exact_vec(p), exact_vec(p))[0]
                                for p in []]) if False else None,
            cgstat=eval_arm([np.conj(c) for c in cg_fit], holdp),
            zoracle=eval_arm(zo_fit, holdp),
            zoracle_inwindow=eval_arm(zo_all, packs))
        ident = [align(exact_vec(p),
                       grad_with_z(p, [np.ones(tcg.N, np.complex128)
                                       for _ in range(L)]))
                 for p in holdp]
        cos_hold["identity"] = (float(np.median([c for c, _ in ident])),
                                float(np.median([r for _, r in ident])))
        # ---- comparisons (weights = block energy of exact)
        cmp_rows = {}
        for l in range(L):
            wt = np.array([sum(np.abs(p["G_ex"]["a"][l][j]) ** 2
                               + np.sum(np.abs(p["G_ex"]["b"][l][j]) ** 2)
                               for p in packs) for j in range(tcg.N)])
            wl = w_learned[l]
            pairs = dict(
                cgstat_vs_learned=mrl(cg_all[l], wl, wt),
                cgstat_vs_zoracle=mrl(cg_all[l], zo_all[l], wt),
                cstat_vs_learned=mrl(cstat_sig[l], wl, wt),
                cstat_vs_zoracle=mrl(cstat_sig[l], zo_all[l], wt),
                learned_vs_zoracle=mrl(wl, zo_all[l], wt))
            mags = dict(
                cgstat=float(np.median(np.abs(cg_all[l]))),
                zoracle=float(np.median(np.abs(zo_all[l]))),
                cstat=float(np.median(np.abs(cstat_sig[l]))),
                learned=float(np.median(np.abs(wl))))
            cmp_rows[l] = dict(pairs=pairs, mags=mags)
        row = dict(seed=seed, held_out=cos_hold, cmp=cmp_rows)
        rows.append(row)
        print(f"  held-out cos: identity {cos_hold['identity'][0]:.3f}  "
              f"cg^stat {cos_hold['cgstat'][0]:.3f}  "
              f"z_oracle {cos_hold['zoracle'][0]:.3f}  "
              f"(in-window z {cos_hold['zoracle_inwindow'][0]:.3f})",
              flush=True)
        for l in range(L):
            p_ = cmp_rows[l]["pairs"]
            print(f"    L{l} MRL: cg~learned {p_['cgstat_vs_learned']:.3f}"
                  f"  cg~zoracle {p_['cgstat_vs_zoracle']:.3f}"
                  f"  cstat~learned {p_['cstat_vs_learned']:.3f}"
                  f"  learned~zoracle {p_['learned_vs_zoracle']:.3f}",
                  flush=True)

    # ---- aggregate
    def med_arm(key):
        return (float(np.median([r["held_out"][key][0] for r in rows])),
                float(np.median([r["held_out"][key][1] for r in rows])))
    print("-" * 78)
    for key in ["identity", "cgstat", "zoracle", "zoracle_inwindow"]:
        c_, r_ = med_arm(key)
        print(f"  {key:<18s} held-out cos {c_:.3f}  rel {r_:.3f}")
    pair_med = {}
    for l in range(L):
        for pname in rows[0]["cmp"][l]["pairs"]:
            pair_med[(l, pname)] = float(np.median(
                [r["cmp"][l]["pairs"][pname] for r in rows]))
    print("MRL medians per layer:")
    for l in range(L):
        print(f"  L{l}: " + "  ".join(
            f"{k} {pair_med[(l, k)]:.3f}"
            for k in ["cgstat_vs_learned", "cgstat_vs_zoracle",
                      "cstat_vs_learned", "learned_vs_zoracle"]))

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS, batches=BATCHES, kfit=KFIT,
               rows=rows,
               aggregate=dict(
                   identity=med_arm("identity"),
                   cgstat=med_arm("cgstat"),
                   zoracle=med_arm("zoracle"),
                   zoracle_inwindow=med_arm("zoracle_inwindow"),
                   pair_med={f"L{l}/{k}": pair_med[(l, k)]
                             for l in range(L)
                             for k in rows[0]["cmp"][l]["pairs"]}))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
