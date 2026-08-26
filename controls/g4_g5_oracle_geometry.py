"""G4 + G5 — oracle representation ceilings (oracle-only, no causal arms).

Same snapshots and fit/held-out protocol as D2
(diagnostics/d2_modal_oracle.py): routeA retrain per seed, 8 probe
batches (prng 888000+seed), fit on 0..3, held-out on 4..7. GATE: the
per-mode-complex arm here must reproduce the stored D2 held-out values
per seed (identical snapshots), then:

  G4 — full real 2x2 per mode. The deployed family constrains each
  modal block to rotation+scale M_j = [[u,-v],[v,u]]. Fit instead an
  unconstrained M_j in R^{2x2} acting on (Re, Im) of the mode's complex
  gradient block, least squares on the fit window:
      M = T Z^T (Z Z^T)^{-1}     (per l, j; 2x2 system)
  Question: does full 2x2 materially exceed the complex ceiling 0.901?
  (Small gain => complex structure already sufficient.)

  G5 — minimal cross-mode coupling. Per layer, stack the N modal blocks
  (each of complex dim d = 1 + M_in) into one complex vector of length
  N*d; start from the per-mode-complex oracle D and fit a low-rank
  correction M = D + U V^dagger, rank(UV) in {1, 2}, by least squares
  on the fit window:
      R = T - D Z  (N*d x nfit);  M_r = [R Z^T (Z Z^T)^+]_rank-r
  nfit = 4 samples only, strict held-out evaluation — deliberately NOT
  enough samples to interpolate. Question: is the remaining
  0.901 -> 1 oracle gap mainly cross-modal?

Arms: identity / per-mode real / per-mode complex / full 2x2 /
complex+rank1 / complex+rank2. Report per-seed held-out cos AND rel,
ranges, and in-window vs held-out gap (overfit visibility). No causal
implementation unless a held-out gain over complex is material AND
seed-robust.

Run:  python -m controls.g4_g5_oracle_geometry
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from diagnostics.d1_exact_credit_factorization import setup, train_routeA
from diagnostics.gradient_cstat import gather, exact_vec, align
from diagnostics.d2_modal_oracle import fit_oracles, eval_arm

SEEDS = [0, 1, 2, 3, 4]
KFIT = 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def mode_block(p, key, l, j):
    return np.concatenate([[p[key]["a"][l][j]],
                           p[key]["b"][l][j].ravel()])


def fit_2x2(fitp):
    """Unconstrained real 2x2 per (l, j): maps (Re,Im) of g_on block."""
    L, N = tcg.L, tcg.N
    Ms = []
    for l in range(L):
        row = []
        for j in range(N):
            Z, T = [], []
            for p in fitp:
                go = mode_block(p, "G_on", l, j)
                ge = mode_block(p, "G_ex", l, j)
                Z.append(np.stack([go.real, go.imag]))
                T.append(np.stack([ge.real, ge.imag]))
            Z = np.concatenate(Z, axis=1)          # 2 x (d*nfit)
            T = np.concatenate(T, axis=1)
            M = T @ Z.T @ np.linalg.pinv(Z @ Z.T)
            row.append(M)
        Ms.append(row)
    return Ms


def apply_2x2(p, Ms):
    # rebuild in the exact_vec layout: per layer [all a', then all b'
    # raveled]; the 2x2 map acts on the mode's whole [a; b] block.
    L = tcg.L
    parts = []
    for l in range(L):
        a_out, b_out = [], []
        for j in range(tcg.N):
            go = mode_block(p, "G_on", l, j)
            z = np.stack([go.real, go.imag])
            out = Ms[l][j] @ z
            cplx = out[0] + 1j * out[1]
            a_out.append(cplx[0])
            b_out.append(cplx[1:])
        parts.append(np.concatenate(
            [np.asarray(a_out), np.concatenate(b_out)]))
    parts.append(np.ravel(p["G_on"]["c"]))
    return np.concatenate(parts)


def stack_layer(p, key, l):
    return np.concatenate([mode_block(p, key, l, j)
                           for j in range(tcg.N)])


def fit_lowrank(fitp, zc, rank):
    """Per layer: D = diag(zc[l]) (scalar per mode on its block);
    residual LS rank-r correction."""
    L, N = tcg.L, tcg.N
    d = 1 + fitp[0]["G_on"]["b"][0].shape[1]
    Ms = []
    for l in range(L):
        Z = np.concatenate([stack_layer(p, "G_on", l)[:, None]
                            for p in fitp], axis=1)     # (N*d) x nfit
        T = np.concatenate([stack_layer(p, "G_ex", l)[:, None]
                            for p in fitp], axis=1)
        DZ = Z.copy()
        for j in range(N):
            DZ[j * d:(j + 1) * d] *= zc[l][j]
        R = T - DZ
        M = R @ Z.conj().T @ np.linalg.pinv(Z @ Z.conj().T)
        U, S, Vh = np.linalg.svd(M)
        Mr = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        Ms.append((zc[l], Mr, d))
    return Ms


def apply_lowrank(p, lr):
    # stack is per-mode-interleaved [a_0; b_0; a_1; b_1; ...]; after
    # applying M = D + UV, rebuild in the exact_vec layout per layer.
    parts = []
    for l, (zcl, Mr, d) in enumerate(lr):
        z = stack_layer(p, "G_on", l)
        out = Mr @ z
        for j in range(tcg.N):
            out[j * d:(j + 1) * d] += zcl[j] * z[j * d:(j + 1) * d]
        a_out = np.asarray([out[j * d] for j in range(tcg.N)])
        b_out = np.concatenate([out[j * d + 1:(j + 1) * d]
                                for j in range(tcg.N)])
        parts.append(np.concatenate([a_out, b_out]))
    parts.append(np.ravel(p["G_on"]["c"]))
    return np.concatenate(parts)


def eval_generic(packs, apply_fn):
    cs, rs = [], []
    for p in packs:
        ge = exact_vec(p)
        gs = apply_fn(p)
        c_, r_ = align(gs, ge)
        cs.append(c_)
        rs.append(r_)
    return float(np.median(cs)), float(np.median(rs))


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(os.path.join(ROOT, "results",
                                      "oracle_real_vs_complex",
                                      "summary.json")))
    stored_cplx = {r["seed"]: r["complex"][0] for r in ref["rows"]}
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + probes...", flush=True)
        params, w_learned = train_routeA(seed)
        prng = np.random.RandomState(888000 + seed)
        packs = gather(params, prng)
        fitp, holdp = packs[:KFIT], packs[KFIT:]
        zc, zr = fit_oracles(fitp, params)
        ident = eval_arm(holdp, [np.ones(tcg.N, np.complex128)
                                 for _ in range(tcg.L)])
        real_h = eval_arm(holdp, [z.astype(np.complex128)
                                  for z in zr])
        cplx_h = eval_arm(holdp, zc)
        cplx_in = eval_arm(fitp, zc)
        assert abs(cplx_h[0] - stored_cplx[seed]) < 1e-9, \
            f"snapshot mismatch s{seed}: {cplx_h[0]} vs {stored_cplx[seed]}"
        M2 = fit_2x2(fitp)
        m2_h = eval_generic(holdp, lambda p: apply_2x2(p, M2))
        m2_in = eval_generic(fitp, lambda p: apply_2x2(p, M2))
        lr1 = fit_lowrank(fitp, zc, 1)
        lr1_h = eval_generic(holdp, lambda p: apply_lowrank(p, lr1))
        lr1_in = eval_generic(fitp, lambda p: apply_lowrank(p, lr1))
        lr2 = fit_lowrank(fitp, zc, 2)
        lr2_h = eval_generic(holdp, lambda p: apply_lowrank(p, lr2))
        lr2_in = eval_generic(fitp, lambda p: apply_lowrank(p, lr2))
        rows.append(dict(seed=seed, identity=ident, real=real_h,
                         complex=cplx_h, complex_in=cplx_in,
                         m2=m2_h, m2_in=m2_in,
                         rank1=lr1_h, rank1_in=lr1_in,
                         rank2=lr2_h, rank2_in=lr2_in))
        print(f"  held-out cos: id {ident[0]:.3f}  real {real_h[0]:.3f}"
              f"  complex {cplx_h[0]:.3f}  2x2 {m2_h[0]:.3f}"
              f"  +r1 {lr1_h[0]:.3f}  +r2 {lr2_h[0]:.3f}", flush=True)
        print(f"  in-window   : complex {cplx_in[0]:.3f}  2x2 "
              f"{m2_in[0]:.3f}  +r1 {lr1_in[0]:.3f}  +r2 {lr2_in[0]:.3f}",
              flush=True)

    med = lambda k: float(np.median([r[k][0] for r in rows]))
    print("-" * 78)
    for k in ["identity", "real", "complex", "m2", "rank1", "rank2"]:
        vals = [r[k][0] for r in rows]
        print(f"held-out cos {k:<9s}: median {np.median(vals):.3f}  "
              f"per-seed {['%.3f' % v for v in vals]}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, seeds=SEEDS, kfit=KFIT, rows=rows,
               medians={k: med(k) for k in
                        ["identity", "real", "complex", "m2", "rank1",
                         "rank2"]})
    with open(os.path.join(OUT, "g4_g5_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote g4_g5_summary.json")


if __name__ == "__main__":
    main()
