"""M1/M4 null-control amendment for the frozen mechanism audit.

This is analysis-only and does not change or refit any M2 action oracle.
It replays the same frozen RoutePCAdam trajectories solely to recover the
optimizer-complete checkpoints needed by two pre-registered specificity
controls:

M1: compare learned w with within-layer mode-shuffled learned w and a random
    complex null that exactly preserves each layer's empirical |w| and phase
    marginals (independent permutations destroy mode and magnitude/phase
    assignment).

M4: compare the optimizer-time lag-one modal correlation for online and exact
    gradients with the sequence-time c_g^stat phase, against a fixed
    within-layer c_g mode-shuffle null.  The exact RoutePC correlation identity
    is claimed only for complex B blocks.  A raw (a,B) complex-group statistic
    is retained as a diagnostic and is not identified with the constrained
    (rho,theta) Jacobian contribution.

Optimizer update time is n; sequence time within a batch is t.  If
K_forward = g_n^dagger g_{n+1}, the repo's positive B-block RoutePC drive is
K_drive = conj(K_forward), because scale_by_w deploys conj(w).

Run:  python -m controls.m1_m4_specificity
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from controls import m1_m6_action_mechanism as mech
from diagnostics.gradient_cstat import fit_scalars, mrl
from toyrig import route_a as cvm
from toyrig import routepc as rp
from toyrig import ssm_rig as tcg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")
SEEDS = tuple(range(15))
CKPTS = mech.PRIMARY_CKPTS
N_M1_NULL = 128
N_M4_SHUFFLE = 256


def _dist(values):
    return mech._dist(values)


def _fast_objective(snapshot, w):
    """Same actual action/objective as M1, without unused gradient assembly."""
    flat_next = snapshot["flat"] + mech.optimizer_action(snapshot, w)
    params_next = tcg.pack(snapshot["params"], flat_next)
    _, yhat = tcg.forward(params_next, snapshot["x_next"])
    residual = yhat - snapshot["y_next"]
    residual[:tcg.DELAY] = 0.0
    return 0.5 * float(np.mean(residual ** 2))


def _m1_controls(snapshot):
    learned = snapshot["w"]
    identity = [np.ones(tcg.N, complex) for _ in range(tcg.L)]
    f_identity = _fast_objective(snapshot, identity)
    f_learned = _fast_objective(snapshot, learned)
    rng = np.random.default_rng(
        1_000_000 + 10_000 * snapshot["step"] + snapshot["seed"])
    shuffled, random_matched = [], []
    for _ in range(N_M1_NULL):
        w_shuffle = []
        w_random = []
        for layer in range(tcg.L):
            wl = learned[layer]
            w_shuffle.append(wl[rng.permutation(tcg.N)])
            # Exact empirical marginal match within the layer, with mode
            # identity and magnitude/phase pairing independently destroyed.
            mag = np.abs(wl)[rng.permutation(tcg.N)]
            phase = np.angle(wl)[rng.permutation(tcg.N)]
            w_random.append(mag * np.exp(1j * phase))
        shuffled.append(_fast_objective(snapshot, w_shuffle))
        random_matched.append(_fast_objective(snapshot, w_random))
    shuffled = np.asarray(shuffled)
    random_matched = np.asarray(random_matched)
    return dict(
        identity=float(f_identity), learned=float(f_learned),
        shuffled=_dist(shuffled), random_marginal_matched=_dist(random_matched),
        learned_minus_identity=float(f_learned - f_identity),
        learned_minus_shuffle_median=float(f_learned - np.median(shuffled)),
        learned_minus_random_median=float(
            f_learned - np.median(random_matched)),
        shuffle_le_learned_fraction=float(np.mean(shuffled <= f_learned)),
        random_le_learned_fraction=float(
            np.mean(random_matched <= f_learned)),
    )


def _modal_k(g_prev, g_now, layer, block):
    """Repo-drive K=conj(K_forward), per mode, in complex block storage."""
    out = np.zeros(tcg.N, complex)
    for mode in range(tcg.N):
        if block == "B":
            prev = np.asarray(g_prev["b"][layer][mode]).ravel()
            now = np.asarray(g_now["b"][layer][mode]).ravel()
        elif block == "raw_aB":
            prev = np.concatenate((
                np.asarray([g_prev["a"][layer][mode]]),
                np.asarray(g_prev["b"][layer][mode]).ravel()))
            now = np.concatenate((
                np.asarray([g_now["a"][layer][mode]]),
                np.asarray(g_now["b"][layer][mode]).ravel()))
        else:
            raise ValueError(block)
        out[mode] = np.vdot(now, prev)
    return out


def _phase_specificity(k_drive, cg, weights, rng):
    correct = mrl(k_drive, cg, weights)
    wt = weights / (np.sum(weights) + 1e-30)
    zk = k_drive / np.maximum(np.abs(k_drive), 1e-30)
    zc = cg / np.maximum(np.abs(cg), 1e-30)
    resultant = np.sum(wt * zk * np.conj(zc))
    shuffled = np.asarray([
        mrl(k_drive, cg[rng.permutation(tcg.N)], weights)
        for _ in range(N_M4_SHUFFLE)
    ])
    return dict(
        mrl_correct=float(correct),
        mean_offset_rad=float(np.angle(resultant)),
        mrl_shuffle=_dist(shuffled),
        correct_minus_shuffle_median=float(correct - np.median(shuffled)),
        shuffle_ge_correct_fraction=float(np.mean(shuffled >= correct)),
    )


def _m4_controls(snapshot, packs):
    params_prev, x_prev, y_prev = snapshot["prev_probe"]
    g_on_prev = snapshot["prev"][0]
    g_on_now = snapshot["G"]
    g_ex_prev = cvm.exact_grad(params_prev, x_prev, y_prev)
    g_ex_now = cvm.exact_grad(snapshot["params"], snapshot["x_current"],
                              snapshot["y_current"])
    cg, _ = fit_scalars(packs, snapshot["params"])
    weights = mech._mode_weights(packs)
    layers = {}
    for layer in range(tcg.L):
        layers[str(layer)] = {}
        for block in ("B", "raw_aB"):
            source_rows = {}
            for source, gp, gn in (
                    ("online", g_on_prev, g_on_now),
                    ("exact", g_ex_prev, g_ex_now)):
                k_drive = _modal_k(gp, gn, layer, block)
                rng = np.random.default_rng(
                    4_000_000 + 100_000 * snapshot["step"]
                    + 1_000 * snapshot["seed"] + 100 * layer
                    + 10 * (block == "raw_aB") + (source == "exact"))
                source_rows[source] = _phase_specificity(
                    k_drive, cg[layer], weights[layer], rng)
            layers[str(layer)][block] = source_rows

        go = np.concatenate((
            np.asarray(g_on_now["a"][layer]).ravel(),
            np.asarray(g_on_now["b"][layer]).ravel()))
        ge = np.concatenate((
            np.asarray(g_ex_now["a"][layer]).ravel(),
            np.asarray(g_ex_now["b"][layer]).ravel()))
        layers[str(layer)]["online_exact_current_relative_error"] = float(
            np.linalg.norm(go - ge) / (np.linalg.norm(ge) + 1e-30))
    return layers


def _aggregate(rows):
    out = {}
    for step in CKPTS:
        selected = [row for row in rows if row["step"] == step]
        out[str(step)] = dict(M1={})
        for key in ("learned_minus_identity", "learned_minus_shuffle_median",
                    "learned_minus_random_median",
                    "shuffle_le_learned_fraction",
                    "random_le_learned_fraction"):
            out[str(step)]["M1"][key] = _dist(
                [row["M1"][key] for row in selected])
        out[str(step)]["M1"]["learned_beats_shuffle_median"] = int(sum(
            row["M1"]["learned_minus_shuffle_median"] < 0
            for row in selected))
        out[str(step)]["M1"]["learned_beats_random_median"] = int(sum(
            row["M1"]["learned_minus_random_median"] < 0
            for row in selected))
        out[str(step)]["M4"] = {}
        for layer in range(tcg.L):
            layer_out = {}
            for block in ("B", "raw_aB"):
                block_out = {}
                for source in ("online", "exact"):
                    block_out[source] = {}
                    for key in ("mrl_correct", "mean_offset_rad",
                                "correct_minus_shuffle_median",
                                "shuffle_ge_correct_fraction"):
                        block_out[source][key] = _dist([
                            row["M4"][str(layer)][block][source][key]
                            for row in selected])
                block_out["online_minus_exact_specificity"] = _dist([
                    row["M4"][str(layer)][block]["online"]
                    ["correct_minus_shuffle_median"]
                    - row["M4"][str(layer)][block]["exact"]
                    ["correct_minus_shuffle_median"]
                    for row in selected])
                layer_out[block] = block_out
            layer_out["online_exact_current_relative_error"] = _dist([
                row["M4"][str(layer)]
                ["online_exact_current_relative_error"]
                for row in selected])
            out[str(step)]["M4"][str(layer)] = layer_out
    return out


def _stopping_decision(aggregate):
    final = aggregate[str(CKPTS[-1])]
    m1 = final["M1"]
    m1_specific = bool(
        m1["learned_minus_shuffle_median"]["median"] < 0
        and m1["learned_minus_random_median"]["median"] < 0
        and m1["learned_beats_shuffle_median"] >= 11
        and m1["learned_beats_random_median"] >= 11)

    b = final["M4"]
    lower_online = np.asarray([
        b[str(layer)]["B"]["online"]["correct_minus_shuffle_median"]
        ["median"] for layer in range(3)])
    lower_exact = np.asarray([
        b[str(layer)]["B"]["exact"]["correct_minus_shuffle_median"]
        ["median"] for layer in range(3)])
    top_online = b["3"]["B"]["online"][
        "correct_minus_shuffle_median"]["median"]
    m4_specific = bool(
        np.sum(lower_online > 0) >= 2
        and float(np.median(lower_online)) > float(np.median(lower_exact))
        and float(np.median(lower_online)) > top_online)
    stop = not (m1_specific and m4_specific)
    return dict(
        preregistered_final_checkpoint=CKPTS[-1],
        M1_rule=("both learned-minus-null medians <0 and learned beats each "
                 "within-snapshot null median on >=11/15 seeds"),
        M1_specific=m1_specific,
        M4_rule=(">=2/3 lower B-block online correct-minus-shuffle medians "
                 "positive, median lower-online specificity exceeds both "
                 "lower-exact and top-online specificity"),
        M4_specific=m4_specific,
        stop_mechanism_search=stop,
        next_action=("STOP: report null controls; do not escalate to a "
                     "multi-step diagnostic or algorithm variant"
                     if stop else
                     "specificity gates pass; interpretation still limited "
                     "to post-update controller evidence"),
    )


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    stored = json.load(open(os.path.join(OUT, "gb_summary.json")))
    stored_finals = {int(seed): value
                     for seed, value in stored["finals"].items()}
    rows = []
    exact_batches = 0
    fast_gate_done = False
    for seed in SEEDS:
        before = dict(rp.BPTT_CALLS)
        replayed = mech.replay(seed)
        train_calls = {key: rp.BPTT_CALLS[key] - before[key]
                       for key in rp.BPTT_CALLS}
        assert train_calls == {"exact_grad": 0, "exact_lambda": 0}
        assert replayed["final_loss"] == stored_finals[seed]
        for step in CKPTS:
            snapshot = replayed["snapshots"][step]
            if snapshot["prev_probe"] is None:
                raise AssertionError((seed, step, "missing previous probe"))
            if not fast_gate_done:
                for w in ([np.ones(tcg.N, complex) for _ in range(tcg.L)],
                          snapshot["w"]):
                    assert _fast_objective(snapshot, w) == mech.action(
                        snapshot, w, "actual_adam")
                fast_gate_done = True
            packs = mech._gather_fit(snapshot["params"], seed, step)
            rows.append(dict(seed=seed, step=step,
                             M1=_m1_controls(snapshot),
                             M4=_m4_controls(snapshot, packs)))
            exact_batches += 2
        print(f"  specificity s{seed}: bitwise replay; training exact 0/0",
              flush=True)

    aggregate = _aggregate(rows)
    decision = _stopping_decision(aggregate)
    doc = dict(
        protocol=dict(
            seeds=SEEDS, checkpoints=CKPTS, m1_null_draws=N_M1_NULL,
            m4_shuffle_draws=N_M4_SHUFFLE,
            random_null=("within each layer, exact empirical |w| and phase "
                         "marginals independently permuted"),
            K_convention=("K_forward=g_n^dagger g_n+1; reported K_drive="
                          "conj(K_forward), matching the repo's positive "
                          "B-block RoutePC drive under conj(w) deployment"),
            exact_scope=("lag-one cross-correlation identity only for "
                         "complex B blocks; raw_aB is diagnostic only"),
        ),
        training_exact_calls={"exact_grad": 0, "exact_lambda": 0},
        offline_exact_gradient_batches=exact_batches,
        rows=rows, aggregate=aggregate,
        stopping_decision=decision,
        parent_artifact="results/geometry_audit/m1_m6_action_summary.json",
        git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip(),
    )
    path = os.path.join(OUT, "m1_m4_specificity_summary.json")
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2, default=float)
    print(decision["next_action"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
