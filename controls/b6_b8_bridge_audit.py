"""B6--B8 bridge audit completion (analysis-only; no new training arms).

This script replays the frozen 15-seed RoutePCAdam trajectories only to
recover checkpoints.  Every final is bitwise-gated against ``gb_summary``
and the training replay is asserted to make zero exact-gradient / exact-
lambda calls.  Exact credit is used only by held-out post-hoc probes.

B6: replace learned magnitude by the analytic eligibility-credit magnitude
while retaining learned phase,

    w_hybrid = |c_g^stat| exp(i arg(w_learned)),

and compare identity / learned / hybrid / per-mode-complex oracle cosine.

B7: report |arg(c_g^stat)| and compare the correct learned/analytic phase
MRL with a fixed 256-permutation within-layer null.  The permutation seed is
fixed from (checkpoint, seed, layer), before looking at results.

B8: report recipient-level dispersion and paired transplant differences.
It also reconciles the old D2 identity headline with the bridge identity by
evaluating the D2 global aggregation (all recurrent layers + readout) at the
same RoutePCAdam checkpoints, then comparing with the stored RouteA D2 rows.

Run:  python -m controls.b6_b8_bridge_audit
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from controls.b1_b4_bridge_audit import CKPTS, KFIT, SEEDS, layer_cos
from controls.geometry_traj import train_arm
from diagnostics.d1_exact_credit_factorization import blocks_vec, setup
from diagnostics.d2_modal_oracle import fit_oracles
from diagnostics.gradient_cstat import (align, exact_vec, fit_scalars,
                                         gather, grad_with_z, mrl)
from toyrig import routepc as rp
from toyrig import ssm_rig as tcg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")
NSHUFFLE = 256


def _dist(values):
    """JSON-ready dispersion summary retaining the independent values."""
    x = np.asarray(values, dtype=float)
    return dict(
        n=int(x.size),
        values=x.tolist(),
        median=float(np.median(x)),
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        q25=float(np.percentile(x, 25)),
        q75=float(np.percentile(x, 75)),
        min=float(np.min(x)),
        max=float(np.max(x)),
    )


def _paired(candidate, baseline, bootstrap_seed):
    """Recipient/seed-paired differences with a median bootstrap CI."""
    cand = np.asarray(candidate, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if cand.shape != base.shape:
        raise ValueError(f"paired shapes differ: {cand.shape} vs {base.shape}")
    diff = cand - base
    rng = np.random.default_rng(bootstrap_seed)
    idx = rng.integers(0, diff.size, size=(10000, diff.size))
    boot = np.median(diff[idx], axis=1)
    out = _dist(diff)
    out.update(
        median_bootstrap_95=[float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
        positive=int(np.sum(diff > 0)),
        negative=int(np.sum(diff < 0)),
        zero=int(np.sum(diff == 0)),
    )
    return out


def _phase_summary(cg, learned_w, weights, checkpoint, seed, layer):
    """B7 effect sizes and a deterministic within-layer permutation null."""
    phase_abs = np.abs(np.angle(cg))
    correct = mrl(cg, learned_w, weights)
    rng = np.random.default_rng(
        7_000_000 + 10_000 * checkpoint + 100 * seed + layer)
    shuffled = np.asarray([
        mrl(cg[rng.permutation(tcg.N)], learned_w, weights)
        for _ in range(NSHUFFLE)
    ])
    wt = weights / (np.sum(weights) + 1e-30)
    return dict(
        abs_arg_rad=dict(
            values=phase_abs.tolist(),
            mean=float(np.mean(phase_abs)),
            energy_weighted_mean=float(np.sum(wt * phase_abs)),
            q25=float(np.percentile(phase_abs, 25)),
            median=float(np.median(phase_abs)),
            q75=float(np.percentile(phase_abs, 75)),
            p90=float(np.percentile(phase_abs, 90)),
            max=float(np.max(phase_abs)),
        ),
        mrl_correct=float(correct),
        mrl_shuffle=dict(
            n=NSHUFFLE,
            median=float(np.median(shuffled)),
            mean=float(np.mean(shuffled)),
            p05=float(np.percentile(shuffled, 5)),
            p95=float(np.percentile(shuffled, 95)),
            values=shuffled.tolist(),
        ),
        correct_minus_shuffle_median=float(correct - np.median(shuffled)),
        shuffle_ge_correct_fraction=float(np.mean(shuffled >= correct)),
    )


def _checkpoint_rows(params, learned_w, packs, checkpoint, seed):
    """B6/B7 rows for all layers at one checkpoint."""
    fitp, holdp = packs[:KFIT], packs[KFIT:]
    cg, _ = fit_scalars(fitp, params)
    z_oracle, _ = fit_oracles(fitp, params)
    z_learned = [np.conj(w) for w in learned_w]
    rows = {}
    for layer in range(tcg.L):
        hybrid_w = np.abs(cg[layer]) * np.exp(1j * np.angle(learned_w[layer]))
        c_id = float(np.median([layer_cos(p, layer, None) for p in holdp]))
        c_learned = float(np.median(
            [layer_cos(p, layer, z_learned[layer]) for p in holdp]))
        c_hybrid = float(np.median(
            [layer_cos(p, layer, np.conj(hybrid_w)) for p in holdp]))
        c_oracle = float(np.median(
            [layer_cos(p, layer, z_oracle[layer]) for p in holdp]))
        weights = np.asarray([
            sum(np.abs(p["G_ex"]["a"][layer][mode]) ** 2
                + np.sum(np.abs(p["G_ex"]["b"][layer][mode]) ** 2)
                for p in packs)
            for mode in range(tcg.N)
        ])
        rows[str(layer)] = dict(
            B6=dict(
                C_id=c_id,
                C_learned=c_learned,
                C_hybrid=c_hybrid,
                C_oracle=c_oracle,
                hybrid_minus_id=c_hybrid - c_id,
                hybrid_minus_learned=c_hybrid - c_learned,
            ),
            B7=_phase_summary(cg[layer], learned_w[layer], weights,
                              checkpoint, seed, layer),
        )
    return rows


def _agg_cos(packs, z_map, layers, include_readout=False):
    vals = []
    for pack in packs:
        online, exact = [], []
        for layer in layers:
            gon = blocks_vec(pack["G_on"], layer)
            if z_map is not None:
                za = z_map[layer] * pack["G_on"]["a"][layer]
                zb = z_map[layer][:, None] * pack["G_on"]["b"][layer]
                gon = np.concatenate([za.ravel(), zb.ravel()])
            online.append(gon)
            exact.append(blocks_vec(pack["G_ex"], layer))
        if include_readout:
            online.append(np.ravel(pack["G_on"]["c"]))
            exact.append(np.ravel(pack["G_ex"]["c"]))
        go, ge = np.concatenate(online), np.concatenate(exact)
        vals.append(align(go, ge)[0])
    return float(np.median(vals))


def _aggregate_b6_b7(rows):
    out = {"B6": {}, "B7": {}}
    for checkpoint in CKPTS:
        ks = str(checkpoint)
        out["B6"][ks] = {}
        out["B7"][ks] = {}
        for layer in range(tcg.L):
            ls = str(layer)
            b6 = [rows[ks][str(seed)][ls]["B6"] for seed in SEEDS]
            out["B6"][ks][ls] = {
                key: _dist([r[key] for r in b6])
                for key in ("C_id", "C_learned", "C_hybrid", "C_oracle",
                            "hybrid_minus_id", "hybrid_minus_learned")
            }
            b7 = [rows[ks][str(seed)][ls]["B7"] for seed in SEEDS]
            phase_values = np.concatenate([
                np.asarray(r["abs_arg_rad"]["values"]) for r in b7
            ])
            correct = [r["mrl_correct"] for r in b7]
            shuffle = [r["mrl_shuffle"]["median"] for r in b7]
            out["B7"][ks][ls] = dict(
                abs_arg_rad=_dist(phase_values),
                energy_weighted_mean=_dist([
                    r["abs_arg_rad"]["energy_weighted_mean"] for r in b7]),
                mrl_correct=_dist(correct),
                mrl_shuffle_per_seed_median=_dist(shuffle),
                correct_minus_shuffle=_paired(correct, shuffle,
                                               8100 + checkpoint + layer),
                shuffle_ge_correct_fraction=_dist([
                    r["shuffle_ge_correct_fraction"] for r in b7]),
            )
    return out


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    gb = json.load(open(os.path.join(OUT, "gb_summary.json")))
    stored_finals = {int(s): v for s, v in gb["finals"].items()}
    d2 = json.load(open(os.path.join(
        ROOT, "results", "oracle_real_vs_complex", "summary.json")))
    d2_identity = {int(row["seed"]): float(row["identity"][0])
                   for row in d2["rows"]}

    audit0 = dict(rp.BPTT_CALLS)
    rows = {str(k): {} for k in CKPTS}
    final_w, final_packs = {}, {}
    pc_global_identity = {}
    pc_lower_identity = {}
    for seed in SEEDS:
        pre = dict(rp.BPTT_CALLS)
        out, _ = train_arm("pc0_adam", seed, exact_probes=False,
                           ckpts=CKPTS)
        assert out["final_loss"] == stored_finals[seed], \
            f"RoutePCAdam replay gate failed for seed {seed}"
        train_exact = {k: rp.BPTT_CALLS[k] - pre[k] for k in rp.BPTT_CALLS}
        assert train_exact == {"exact_grad": 0, "exact_lambda": 0}, train_exact
        print(f"routePCadam s{seed} replay == stored; training exact 0/0",
              flush=True)
        for checkpoint in CKPTS:
            params, learned_w = out["ckpts"][checkpoint]
            packs = gather(params, np.random.RandomState(888000 + seed))
            rows[str(checkpoint)][str(seed)] = _checkpoint_rows(
                params, learned_w, packs, checkpoint, seed)
            if checkpoint == 1500:
                final_w[seed] = learned_w
                final_packs[seed] = packs[KFIT:]
                pc_lower_identity[seed] = _agg_cos(
                    packs[KFIT:], None, [0, 1, 2])
                pc_global_identity[seed] = _agg_cos(
                    packs[KFIT:], None, list(range(tcg.L)),
                    include_readout=True)
        krow = rows["1500"][str(seed)]
        print("  B6 K1500 hybrid-learned: " + "  ".join(
            f"L{layer} {krow[str(layer)]['B6']['hybrid_minus_learned']:+.3f}"
            for layer in range(tcg.L)), flush=True)

    # B8 transplant: the independent unit is the recipient seed.  Collapse
    # the 14 off-diagonal donors within recipient before paired uncertainty.
    matrix = np.zeros((len(SEEDS), len(SEEDS)))
    identity = np.asarray([pc_lower_identity[s] for s in SEEDS])
    for donor in SEEDS:
        zmap = [np.conj(w) for w in final_w[donor]]
        for recipient in SEEDS:
            matrix[donor, recipient] = _agg_cos(
                final_packs[recipient], zmap, [0, 1, 2])
    self_cos = np.asarray([matrix[s, s] for s in SEEDS])
    offdiag_by_recipient = np.asarray([
        np.median([matrix[donor, recipient] for donor in SEEDS
                   if donor != recipient])
        for recipient in SEEDS
    ])
    pooled_offdiag = np.asarray([
        matrix[donor, recipient]
        for recipient in SEEDS for donor in SEEDS if donor != recipient
    ])

    first5 = list(range(5))
    d2_vec = np.asarray([d2_identity[s] for s in first5])
    pc_global_5 = np.asarray([pc_global_identity[s] for s in first5])
    pc_lower_5 = np.asarray([pc_lower_identity[s] for s in first5])
    b8 = dict(
        transplant_lower=dict(
            matrix=matrix.tolist(),
            identity=_dist(identity),
            self=_dist(self_cos),
            offdiag_recipient_median=_dist(offdiag_by_recipient),
            offdiag_pooled_descriptive_only=_dist(pooled_offdiag),
            self_minus_identity=_paired(self_cos, identity, 8801),
            offdiag_minus_identity=_paired(offdiag_by_recipient, identity,
                                           8802),
            self_minus_offdiag=_paired(self_cos, offdiag_by_recipient, 8803),
        ),
        d2_reconciliation=dict(
            evaluation_fact=(
                "D2 identity is the held-out global vector (all four recurrent "
                "layers plus readout) at RouteA final parameters, seeds 0--4. "
                "Bridge identity is lower recurrent layers L0--L2 only at "
                "RoutePCAdam final parameters, seeds 0--14. Both are K=1500 "
                "and use held-out batches 4--7 from PRNG 888000+seed."),
            d2_routeA_global_first5=_dist(d2_vec),
            routePCadam_global_first5=_dist(pc_global_5),
            routePCadam_lower_first5=_dist(pc_lower_5),
            routePCadam_global_all15=_dist(
                [pc_global_identity[s] for s in SEEDS]),
            routePCadam_lower_all15=_dist(identity),
            aggregation_global_minus_lower_first5=_paired(
                pc_global_5, pc_lower_5, 8811),
            parameter_regime_routePCadam_minus_routeA_global_first5=_paired(
                pc_global_5, d2_vec, 8812),
            seed_set_median_shift=dict(
                global_first5_minus_all15=float(
                    np.median(pc_global_5)
                    - np.median([pc_global_identity[s] for s in SEEDS])),
                lower_first5_minus_all15=float(
                    np.median(pc_lower_5) - np.median(identity)),
            ),
        ),
    )

    doc = dict(
        protocol=dict(seeds=SEEDS, checkpoints=CKPTS, probe_batches=8,
                      fit_batches=[0, 1, 2, 3], held_out_batches=[4, 5, 6, 7],
                      phase_null_shuffles=NSHUFFLE,
                      training_arm="routePCadam replay only; no new arm"),
        rows=rows,
        aggregate=_aggregate_b6_b7(rows),
        B8=b8,
        training_exact_calls={"exact_grad": 0, "exact_lambda": 0},
        probe_exact_calls={k: rp.BPTT_CALLS[k] - audit0[k]
                           for k in rp.BPTT_CALLS},
        git=subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip(),
    )
    path = os.path.join(OUT, "b6_b8_summary.json")
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
