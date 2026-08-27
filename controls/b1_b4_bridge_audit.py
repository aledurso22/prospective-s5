"""B1–B4 — bridge audit: does routePCadam's learned geometry approximate
the exact missing-credit correction? (analysis-only; no new arms)

Motivation: G3X showed the C3 crossover washes out without clipping, so
the direct evidence geometry≈credit-repair must come from measuring the
learned geometry against exact credit itself. Post-hoc analysis:
routePCadam (pc0_adam) is REPLAYED with checkpoint capture (bitwise gate
vs gb_summary finals; exact_grad/exact_lambda calls during training = 0,
asserted; exact credit appears only in the post-hoc probes, which never
enter the trajectory).

Protocol per seed: checkpoints K in {500, 1000, 1500}; at each, 8 probe
batches from the held-out stream (prng 888000+seed, disjoint from
training), fit on 0..3 / evaluate on 4..7 (the D2 convention).

  B1  per layer: C_id = cos(g_on, g_ex), C_learned = cos(M_w g_on, g_ex)
      with the checkpoint's learned w, C_oracle with the fit per-mode
      complex oracle. Report ΔC = C_learned − C_id. Focus: lower
      recurrent layers (the defect site); top recurrent layer L3 is the
      negative control (online gradient exact there).
  B2  analytic correspondence: c_g^stat (fit window) vs learned w per
      layer — phase MRL (exact-energy weighted) and relative log-gain
      correlation after removing the common layerwise scale (the common
      radial direction is a near-gauge under clipping — no absolute-gain
      comparison).
  B3  donor×recipient transplant at K=1500: C_{i->j} = cos(M_{w_i}
      g_on,j, g_ex,j) on recipient j's held-out batches; diagonal vs
      off-diagonal vs identity. (Off-diagonal improvement = shared
      defect structure; lack of it is NOT falsification — w* can depend
      on the current eligibility state.)
  B4  mode shuffle: permute w within each layer (seeded) and repeat B1 —
      is correct mode assignment required?

Run:  python -m controls.b1_b4_bridge_audit
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import routepc as rp
from toyrig.probes import make_data
from diagnostics.d1_exact_credit_factorization import setup, blocks_vec
from diagnostics.gradient_cstat import gather, fit_scalars, mrl
from diagnostics.d2_modal_oracle import fit_oracles
from controls.geometry_traj import train_arm

SEEDS = list(range(15))
CKPTS = [500, 1000, 1500]
KFIT = 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def layer_cos(p, l, z_l):
    """cos(exact, z * g_on) on layer l's (a, B) blocks; z_l per-mode
    complex (conj-w convention), or None for identity."""
    go = blocks_vec(p["G_on"], l)
    ge = blocks_vec(p["G_ex"], l)
    if z_l is not None:
        za = z_l * p["G_on"]["a"][l]
        zb = z_l[:, None] * p["G_on"]["b"][l]
        go = np.concatenate([za.ravel(), zb.ravel()])
    return float(np.abs(np.vdot(ge, go))
                 / (np.linalg.norm(ge) * np.linalg.norm(go) + 1e-30))


def eval_seed(params, w_learned, packs, seed):
    """B1/B2/B4 at one checkpoint. w_learned: list of complex (N,) per
    layer (the geometry); z = conj(w) per the deployed convention."""
    fitp, holdp = packs[:KFIT], packs[KFIT:]
    zc, zr = fit_oracles(fitp, params)
    cg, zo = fit_scalars(fitp, params)
    L = tcg.L
    zl = [np.conj(wl) for wl in w_learned]
    shuf_rng = np.random.RandomState(555 + seed)
    zshuf = [zl[l][shuf_rng.permutation(tcg.N)] for l in range(L)]
    rows = {}
    for l in range(L):
        cid = float(np.median([layer_cos(p, l, None) for p in holdp]))
        cle = float(np.median([layer_cos(p, l, zl[l]) for p in holdp]))
        cor = float(np.median([layer_cos(p, l, zc[l]) for p in holdp]))
        csh = float(np.median([layer_cos(p, l, zshuf[l]) for p in holdp]))
        wt = np.array([sum(np.abs(p["G_ex"]["a"][l][j]) ** 2
                           + np.sum(np.abs(p["G_ex"]["b"][l][j]) ** 2)
                           for p in packs) for j in range(tcg.N)])
        # B2: phase MRL + relative log-gain (common scale removed)
        aw = np.log(np.maximum(np.abs(w_learned[l]), 1e-30))
        ac = np.log(np.maximum(np.abs(cg[l]), 1e-30))
        rel_gain_corr = float(np.corrcoef(aw - aw.mean(),
                                          ac - ac.mean())[0, 1])
        rows[l] = dict(C_id=cid, C_learned=cle, C_oracle=cor,
                       C_shuffle=csh, delta=cle - cid,
                       mrl_cg_w=mrl(cg[l], w_learned[l], wt),
                       rel_gain_corr=rel_gain_corr)
    return rows


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    gb = json.load(open(os.path.join(OUT, "gb_summary.json")))
    stored_finals = {int(s): v for s, v in gb["finals"].items()}
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    onl = {int(s): v for s, v in c15["finals"]["online"].items()}

    audit0 = dict(rp.BPTT_CALLS)
    doc = {"B1_B2_B4": {}, "B3": None, "B5": None}
    final_w, final_params = {}, {}
    for seed in SEEDS:
        pre = dict(rp.BPTT_CALLS)
        out, traj = train_arm("pc0_adam", seed, exact_probes=False,
                              ckpts=CKPTS)
        assert out["final_loss"] == stored_finals[seed], \
            f"replay gate failed s{seed}"
        train_exact = {k: rp.BPTT_CALLS[k] - pre[k]
                       for k in rp.BPTT_CALLS}
        assert train_exact["exact_grad"] == 0 \
            and train_exact["exact_lambda"] == 0, train_exact
        print(f"pc0_adam s{seed} replay == stored ✓ (training exact "
              f"calls 0/0)", flush=True)
        final_w[seed] = out["ckpts"][1500][1]
        final_params[seed] = out["ckpts"][1500][0]
        for K in CKPTS:
            params_K, w_K = out["ckpts"][K]
            prng = np.random.RandomState(888000 + seed)
            packs = gather(params_K, prng)
            rows = eval_seed(params_K, w_K, packs, seed)
            doc["B1_B2_B4"].setdefault(str(K), {})[str(seed)] = {
                str(l): rows[l] for l in rows}
            if K == 1500:
                r = rows
                print(f"  K1500 per layer ΔC (learned-id): "
                      + "  ".join(f"L{l} {r[l]['delta']:+.3f}"
                                 for l in range(tcg.L)), flush=True)

    # ---- B3 donor x recipient at K=1500 ----
    print("B3 transplant matrix (K=1500, aggregate layers L0-L2 + L3)...",
          flush=True)
    rec_packs = {}
    for j in SEEDS:
        prng = np.random.RandomState(888000 + j)
        rec_packs[j] = gather(final_params[j], prng)[KFIT:]

    def agg_cos(j, zmap, layers):
        vals = []
        for p in rec_packs[j]:
            go, ge = [], []
            for l in layers:
                gon = blocks_vec(p["G_on"], l)
                if zmap is not None:
                    za = zmap[l] * p["G_on"]["a"][l]
                    zb = zmap[l][:, None] * p["G_on"]["b"][l]
                    gon = np.concatenate([za.ravel(), zb.ravel()])
                go.append(gon)
                ge.append(blocks_vec(p["G_ex"], l))
            go, ge = np.concatenate(go), np.concatenate(ge)
            vals.append(float(np.abs(np.vdot(ge, go))
                              / (np.linalg.norm(ge) * np.linalg.norm(go)
                                 + 1e-30)))
        return float(np.median(vals))

    for layers, tag in (([0, 1, 2], "lower"), ([3], "top")):
        mat = np.zeros((len(SEEDS), len(SEEDS)))
        ident = {}
        for j in SEEDS:
            ident[j] = agg_cos(j, None, layers)
            for i in SEEDS:
                zmap = [np.conj(wl) for wl in final_w[i]]
                mat[i, j] = agg_cos(j, zmap, layers)
        diag = np.array([mat[j, j] for j in SEEDS])
        offd = np.array([mat[i, j] for j in SEEDS for i in SEEDS
                         if i != j])
        idv = np.array([ident[j] for j in SEEDS])
        doc["B3_" + tag] = dict(
            matrix=mat.tolist(), seeds=SEEDS,
            diag_median=float(np.median(diag)),
            offdiag_median=float(np.median(offd)),
            identity_median=float(np.median(idv)),
            diag_per_seed={str(s): float(diag[k])
                           for k, s in enumerate(SEEDS)})
        print(f"  B3[{tag}]: identity {np.median(idv):.3f}  "
              f"diagonal {np.median(diag):.3f}  off-diagonal "
              f"{np.median(offd):.3f}", flush=True)

    # ---- B5 exact failure ratios ----
    fails = [3, 9, 10]
    ratios = {str(s): stored_finals[s] / onl[s] for s in fails}
    doc["B5"] = dict(failure_seeds=fails,
                     ratios=ratios,
                     note=("exact paired ratios L_pc0adam/L_online; "
                           "previously described as 'marginal'"))
    print(f"B5 exact failure ratios: "
          + "  ".join(f"s{s} {ratios[str(s)]:.3f}" for s in fails))

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc["git"] = git
    doc["probe_exact_calls"] = {k: rp.BPTT_CALLS[k] - audit0[k]
                                for k in rp.BPTT_CALLS}
    with open(os.path.join(OUT, "b1_b4_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote b1_b4_summary.json")


if __name__ == "__main__":
    main()
