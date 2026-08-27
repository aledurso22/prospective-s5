"""M1--M6 mechanism-first offline audit for frozen RoutePCAdam.

No training algorithm is introduced. The frozen 15-seed RoutePCAdam loop is
replayed bitwise only to recover optimizer-complete checkpoints. Exact credit
is forbidden during each replay and used solely by offline probes/objective
derivatives after a checkpoint has been cloned.

Optimizer time is indexed by ``n`` (minibatch/update number). Sequence time
inside a minibatch is ``t``. In particular, RoutePC's schematic residual at
optimizer step n is J_{n-1}^T g_n^on; it is not a sequence-time D^{-1}.

M1 evaluates one actual action at n and the same next-batch objective at n+1
for identity, learned, learned-phase-only, and static credit-oracle geometry.
M2 fits a separate action/meta oracle under the exact clip+Adam action at a
small representative set of adjacent checkpoints. M3 resets optimizer state
or replaces Adam by clipped SGD only as an offline counterfactual. M4 checks
the lag-one complex correlation convention and compares raw/EMA/Adam-filtered
optimizer-time correlations with learned w and sequence-time c_g^stat. M5 is
the separately frozen B6/B7 audit. M6 runs only where the multi-start M2
oracle passes a predeclared stability check.

Run:  python -m controls.m1_m6_action_mechanism
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

from diagnostics.d2_modal_oracle import fit_oracles
from diagnostics.gradient_cstat import fit_scalars, mrl
from diagnostics.prospective_kappa import chain_c_stored
from toyrig import route_a as cvm
from toyrig import routepc as rp
from toyrig import ssm_rig as tcg
from toyrig.probes import make_data
from toyrig.train_cell import STEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")
PRIMARY_CKPTS = (500, 1000, 1500)
M2_CKPTS = (500, 501, 1499, 1500)
CAPTURE_CKPTS = tuple(sorted(set(PRIMARY_CKPTS + M2_CKPTS)))
M2_SEEDS = (0, 1, 3, 9, 10)  # two successes + all three failures
FAIL_SEEDS = (3, 9, 10)
N_FIT = 4
N_MODES = 4 * 16
M2_MAXITER = 60
OFFLINE_EXACT = {"lambda_probe_batches": 0,
                 "next_objective_gradients": 0}


def _dist(values):
    x = np.asarray(values, dtype=float)
    return dict(n=int(x.size), values=x.tolist(),
                median=float(np.median(x)), mean=float(np.mean(x)),
                sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                q25=float(np.percentile(x, 25)),
                q75=float(np.percentile(x, 75)),
                min=float(np.min(x)), max=float(np.max(x)),
                positive=int(np.sum(x > 0)), negative=int(np.sum(x < 0)),
                zero=int(np.sum(x == 0)))


def _params_context(params):
    return ([th.copy() for th in params["theta"]],
            [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
            [tcg.sig(params["rho"][l])
             * (1 - tcg.sig(params["rho"][l])) for l in range(tcg.L)])


def _copy_blocks(G):
    return dict(a=[g.copy() for g in G["a"]],
                b=[g.copy() for g in G["b"]], c=G["c"].copy())


def _gather_fit(params, seed, step):
    """Independent offline fit batches; never enter the trajectory."""
    rng = np.random.RandomState(9_000_000 + 10_000 * seed + step)
    packs = []
    for _ in range(N_FIT):
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        lam = tcg.exact_lambda(params, q)
        OFFLINE_EXACT["lambda_probe_batches"] += 1
        G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        G_on = tcg.assemble(params, h, x, r, q, Sa, Sb)
        packs.append(dict(x=x, y=y, h=h, r=r, q=q, Sa=Sa, Sb=Sb,
                          G_ex=G_ex, G_on=G_on))
    return packs


def _mode_weights(packs):
    weights = []
    for layer in range(tcg.L):
        weights.append(np.asarray([
            sum(np.abs(p["G_ex"]["a"][layer][mode]) ** 2
                + np.sum(np.abs(p["G_ex"]["b"][layer][mode]) ** 2)
                for p in packs)
            for mode in range(tcg.N)
        ]))
    return weights


def _peek_next(rng):
    state = rng.get_state()
    out = make_data(rng)
    rng.set_state(state)
    return out


def replay(seed):
    """Bitwise-frozen RoutePCAdam replay with read-only checkpoint capture."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    mw = [np.zeros(tcg.N, np.complex128) for _ in range(tcg.L)]
    vwre = [np.zeros(tcg.N) for _ in range(tcg.L)]
    vwim = [np.zeros(tcg.N) for _ in range(tcg.L)]
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = np.empty(STEPS)
    snapshots = {}
    prev_probe = None
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses[step - 1] = loss
        h_n = tcg.flat_grads(G, params)
        corr = None
        corr_ema = None
        corr_adam = None
        prev_for_snapshot = None
        if prev is not None:
            G_prev, th_prev, u_prev, sig_prev = prev
            corr = chain_c_stored(G_prev, th_prev, u_prev, sig_prev, h_n)
            drive_ema, drive_adam = [], []
            t_meta = step - 1
            for layer in range(tcg.L):
                gl = (-cvm.LR) * corr[layer]
                mw[layer] = 0.9 * mw[layer] + 0.1 * gl
                vwre[layer] = (0.999 * vwre[layer]
                               + 0.001 * gl.real ** 2)
                vwim[layer] = (0.999 * vwim[layer]
                               + 0.001 * gl.imag ** 2)
                upd_re = (mw[layer].real / (1 - 0.9 ** t_meta)) / (
                    np.sqrt(vwre[layer] / (1 - 0.999 ** t_meta)) + 1e-8)
                upd_im = (mw[layer].imag / (1 - 0.9 ** t_meta)) / (
                    np.sqrt(vwim[layer] / (1 - 0.999 ** t_meta)) + 1e-8)
                upd = upd_re + 1j * upd_im
                w[layer] = w[layer] - cvm.LR_M * upd
                # Positive drive convention: w moves along these objects.
                drive_ema.append(-mw[layer] / cvm.LR)
                drive_adam.append(-upd)
            corr_ema, corr_adam = drive_ema, drive_adam
            prev_for_snapshot = (G_prev, th_prev, u_prev, sig_prev)

        if step in CAPTURE_CKPTS:
            x_next, y_next = _peek_next(rng)
            snapshots[step] = dict(
                seed=seed, step=step,
                params=copy.deepcopy(params), flat=flat.copy(),
                base_m=m.copy(), base_v=v.copy(),
                G=_copy_blocks(G), h_online=h_n.copy(),
                w=[wl.copy() for wl in w],
                corr=([c.copy() for c in corr] if corr is not None else None),
                corr_ema=([c.copy() for c in corr_ema]
                          if corr_ema is not None else None),
                corr_adam=([c.copy() for c in corr_adam]
                           if corr_adam is not None else None),
                prev=prev_for_snapshot,
                prev_probe=prev_probe,
                x_current=x.copy(), y_current=y.copy(),
                x_next=x_next.copy(), y_next=y_next.copy(),
            )

        G_use = cvm.scale_by_w(G, w)
        g_raw = tcg.flat_grads(G_use, params)
        g = cvm.clip(g_raw)
        flat, m, v = cvm.adam(flat, g, m, v, step)
        prev = (_copy_blocks(G),) + _params_context(params)
        # Logging-only context for the M1/M4 specificity amendment. It is
        # retained only when the following step is a requested checkpoint.
        prev_probe = ((copy.deepcopy(params), x.copy(), y.copy())
                      if step + 1 in CAPTURE_CKPTS else None)
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    replay s{seed} step {step}: loss {loss:.4f}",
                  flush=True)
    return dict(final_loss=float(losses[-100:].mean()), snapshots=snapshots)


def _clip_with_vjp(raw, upstream):
    norm = float(np.linalg.norm(raw))
    if norm <= cvm.CLIP:
        return raw, upstream
    unit = raw / norm
    clipped = (cvm.CLIP / norm) * raw
    pulled = (cvm.CLIP / norm) * (
        upstream - unit * float(np.dot(unit, upstream)))
    return clipped, pulled


def action(snapshot, w, optimizer="actual_adam", gradient=False):
    """One corrected-gradient -> clip -> optimizer action, then L_{n+1}."""
    params = snapshot["params"]
    raw = tcg.flat_grads(cvm.scale_by_w(snapshot["G"], w), params)
    norm = float(np.linalg.norm(raw))
    clipped = raw * (cvm.CLIP / norm) if norm > cvm.CLIP else raw
    if optimizer == "actual_adam":
        m0, v0, t = snapshot["base_m"], snapshot["base_v"], snapshot["step"]
    elif optimizer == "reset_adam":
        # M3 specifies m=v=0, not a rewind of optimizer time.
        m0, v0, t = (np.zeros_like(raw), np.zeros_like(raw),
                     snapshot["step"])
    elif optimizer == "clipped_sgd":
        m0 = v0 = None
        t = snapshot["step"]
    else:
        raise ValueError(optimizer)

    if optimizer == "clipped_sgd":
        flat_next = snapshot["flat"] - cvm.LR * clipped
        d_direction = None
    else:
        mn = 0.9 * m0 + 0.1 * clipped
        vn = 0.999 * v0 + 0.001 * clipped ** 2
        mh = mn / (1 - 0.9 ** t)
        vh = vn / (1 - 0.999 ** t)
        sq = np.sqrt(vh) + 1e-8
        direction = mh / sq
        flat_next = snapshot["flat"] - cvm.LR * direction
        sqrt_vh = np.sqrt(np.maximum(vh, 1e-300))
        d_direction = ((0.1 / (1 - 0.9 ** t)) / sq
                       - mh * (0.001 / (1 - 0.999 ** t)) * clipped
                       / (sq ** 2 * sqrt_vh))
    params_next = tcg.pack(params, flat_next)
    value = cvm.batch_grad(params_next, snapshot["x_next"],
                           snapshot["y_next"])[0]
    if not gradient:
        return float(value)

    # exact next-objective gradient in flat coordinates. assemble() returns
    # sums while batch_grad's objective is a mean.
    h = (tcg.flat_grads(cvm.exact_grad(
        params_next, snapshot["x_next"], snapshot["y_next"]), params_next)
         / (tcg.T * tcg.BATCH))
    OFFLINE_EXACT["next_objective_gradients"] += 1
    if optimizer == "clipped_sgd":
        upstream_clip = -cvm.LR * h
    else:
        upstream_clip = -cvm.LR * d_direction * h
    _, upstream_raw = _clip_with_vjp(raw, upstream_clip)
    th, u, sig = _params_context(params)
    grad_w = chain_c_stored(snapshot["G"], th, u, sig, upstream_raw)
    return float(value), grad_w


def optimizer_action(snapshot, w):
    """Actual accumulated-Adam parameter displacement for an M2 geometry."""
    raw = tcg.flat_grads(cvm.scale_by_w(snapshot["G"], w),
                         snapshot["params"])
    norm = float(np.linalg.norm(raw))
    clipped = raw * (cvm.CLIP / norm) if norm > cvm.CLIP else raw
    t = snapshot["step"]
    mn = 0.9 * snapshot["base_m"] + 0.1 * clipped
    vn = 0.999 * snapshot["base_v"] + 0.001 * clipped ** 2
    direction = (mn / (1 - 0.9 ** t)) / (
        np.sqrt(vn / (1 - 0.999 ** t)) + 1e-8)
    return -cvm.LR * direction


def _action_relation(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(
        cosine=float(np.dot(a, b) / (na * nb + 1e-30)),
        distance=float(np.linalg.norm(a - b)),
        symmetric_relative_distance=float(
            np.linalg.norm(a - b) / (0.5 * (na + nb) + 1e-30)),
        norm_a=na, norm_b=nb,
    )


def action_space_summary(snapshot, fit, w_credit):
    """M2 primary readout: action equivalence and objective regret."""
    identities = [np.ones(tcg.N, complex) for _ in range(tcg.L)]
    candidates = dict(identity=identities, learned=snapshot["w"],
                      credit=w_credit, action_oracle=fit["best"]["w"])
    deltas = {name: optimizer_action(snapshot, w)
              for name, w in candidates.items()}
    values = {name: action(snapshot, w, "actual_adam")
              for name, w in candidates.items()}
    fstar = values["action_oracle"]
    candidate_rows = {
        name: dict(objective=float(values[name]),
                   objective_regret=float(values[name] - fstar),
                   action_vs_oracle=_action_relation(
                       deltas[name], deltas["action_oracle"]))
        for name in ("identity", "learned", "credit")
    }
    run_actions = [optimizer_action(snapshot, run["w"])
                   for run in fit["runs"]]
    pairwise = []
    for i in range(len(run_actions)):
        for j in range(i):
            pairwise.append(dict(
                starts=[fit["runs"][j]["start"], fit["runs"][i]["start"]],
                **_action_relation(run_actions[j], run_actions[i])))
    spread_ok = (fit["objective_spread"]
                 <= max(0.1 * max(fit["improvement"], 0.0), 1e-6))
    action_stable = bool(
        spread_ok
        and min(row["cosine"] for row in pairwise) >= 0.999
        and max(row["symmetric_relative_distance"] for row in pairwise)
        <= 0.1)
    return dict(
        definition=("full flat-parameter displacement produced by the actual "
                    "global-clip plus accumulated-Adam action"),
        candidates=candidate_rows,
        fit_start_pairwise_actions=pairwise,
        action_stability_rule=("objective-spread rule plus minimum pairwise "
                               "action cosine >=0.999 and maximum symmetric "
                               "relative action distance <=0.1"),
        action_stable=action_stable,
    )


def _flatten_w(w):
    return np.concatenate([np.asarray(x) for x in w])


def _unflatten_w(w):
    z = np.asarray(w).reshape(tcg.L, tcg.N)
    return [z[layer].copy() for layer in range(tcg.L)]


def _normalize_relative(w):
    z = _flatten_w(w)
    scale = np.exp(np.mean(np.log(np.maximum(np.abs(z), 1e-30))))
    return _unflatten_w(z / scale)


def _pack_polar(w):
    z = _flatten_w(w)
    return np.concatenate([np.log(np.maximum(np.abs(z), 1e-30)),
                           np.angle(z)])


def _unpack_polar(x):
    alpha = np.asarray(x[:N_MODES])
    phase = np.asarray(x[N_MODES:])
    return _unflatten_w(np.exp(alpha + 1j * phase))


def _polar_objective(snapshot, x):
    w = _unpack_polar(x)
    value, gw = action(snapshot, w, "actual_adam", gradient=True)
    z = _flatten_w(w)
    c = _flatten_w(gw)
    grad_alpha = c.real * z.real + c.imag * z.imag
    grad_phase = z.real * c.imag - z.imag * c.real
    return value, np.concatenate([grad_alpha, grad_phase])


def gradient_gate(snapshot):
    """Central-FD gate for the exact clip+Adam action-oracle gradient."""
    x = _pack_polar(snapshot["w"])
    value, grad = _polar_objective(snapshot, x)
    rng = np.random.default_rng(123)
    dims = rng.choice(x.size, 10, replace=False)
    eps = 2e-5
    errors = []
    for dim in dims:
        xp, xm = x.copy(), x.copy()
        xp[dim] += eps
        xm[dim] -= eps
        fp = action(snapshot, _unpack_polar(xp), "actual_adam")
        fm = action(snapshot, _unpack_polar(xm), "actual_adam")
        fd = (fp - fm) / (2 * eps)
        errors.append(abs(fd - grad[dim]) / (abs(fd) + abs(grad[dim]) + 1e-10))
    worst = float(max(errors))
    print(f"M2 action-gradient FD gate: value {value:.6f}, worst symmetric "
          f"relative error {worst:.2e}")
    assert worst < 2e-3, worst
    return worst


def fit_action_oracle(snapshot, w_credit):
    """Multi-start bounded estimate of optimizer-aware w_F*."""
    starts = dict(identity=[np.ones(tcg.N, complex) for _ in range(tcg.L)],
                  learned=snapshot["w"], credit=w_credit)
    runs = []
    bounds = [(-2.5, 2.5)] * N_MODES + [(None, None)] * N_MODES
    for name, start in starts.items():
        res = minimize(lambda x: _polar_objective(snapshot, x),
                       _pack_polar(start), jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options=dict(maxiter=M2_MAXITER, ftol=1e-12,
                                    gtol=1e-7, maxls=30))
        w = _unpack_polar(res.x)
        runs.append(dict(start=name, value=float(res.fun), success=bool(res.success),
                         status=int(res.status), message=str(res.message),
                         nit=int(res.nit), grad_norm=float(np.linalg.norm(res.jac)),
                         alpha_boundary_fraction=float(np.mean(
                             np.abs(res.x[:N_MODES]) >= 2.49)), w=w))
    best = min(runs, key=lambda row: row["value"])
    pair_mrl, pair_phase_rmse = [], []
    pair_loggain_rmse, pair_relative_loggain_rmse = [], []
    for i in range(len(runs)):
        for j in range(i):
            pair_mrl.append(float(np.median([
                mrl(runs[i]["w"][layer], runs[j]["w"][layer],
                    np.ones(tcg.N)) for layer in range(tcg.L)])))
            zi = _flatten_w(runs[i]["w"])
            zj = _flatten_w(runs[j]["w"])
            pair_phase_rmse.append(float(np.sqrt(np.mean(
                np.angle(zi * np.conj(zj)) ** 2))))
            ai = np.log(np.maximum(np.abs(zi), 1e-30))
            aj = np.log(np.maximum(np.abs(zj), 1e-30))
            pair_loggain_rmse.append(float(np.sqrt(np.mean((ai - aj) ** 2))))
            pair_relative_loggain_rmse.append(float(np.sqrt(np.mean(
                ((ai - ai.mean()) - (aj - aj.mean())) ** 2))))
    identity_value = action(snapshot, starts["identity"], "actual_adam")
    improvement = identity_value - best["value"]
    spread = max(row["value"] for row in runs) - best["value"]
    stable = bool(
        all(np.isfinite(row["value"]) for row in runs)
        and all(row["success"] for row in runs)
        and spread <= max(0.1 * max(improvement, 0.0), 1e-6)
        and min(pair_mrl) >= 0.8
        and max(pair_phase_rmse) <= 0.5
        and max(pair_loggain_rmse) <= 0.5
        and max(row["alpha_boundary_fraction"] for row in runs) < 0.1
    )
    return dict(runs=runs, best=best, identity_value=float(identity_value),
                improvement=float(improvement), objective_spread=float(spread),
                pairwise_phase_mrl=pair_mrl,
                pairwise_phase_rmse_rad=pair_phase_rmse,
                pairwise_loggain_rmse=pair_loggain_rmse,
                pairwise_relative_loggain_rmse=pair_relative_loggain_rmse,
                stable=stable)


def _geometry_compare(a, b, weights):
    phase_mrl = [mrl(a[l], b[l], weights[l]) for l in range(tcg.L)]
    phase_offset = []
    for layer in range(tcg.L):
        wt = weights[layer] / (np.sum(weights[layer]) + 1e-30)
        za = a[layer] / np.maximum(np.abs(a[layer]), 1e-30)
        zb = b[layer] / np.maximum(np.abs(b[layer]), 1e-30)
        phase_offset.append(float(np.angle(np.sum(wt * za * np.conj(zb)))))
    za, zb = _flatten_w(a), _flatten_w(b)
    la = np.log(np.maximum(np.abs(za), 1e-30)); la -= la.mean()
    lb = np.log(np.maximum(np.abs(zb), 1e-30)); lb -= lb.mean()
    corr = float(np.corrcoef(la, lb)[0, 1])
    gain_rmse = float(np.sqrt(np.mean((la - lb) ** 2)))
    phase_rmse = float(np.sqrt(np.mean(
        np.angle(za * np.conj(zb)) ** 2)))
    return dict(phase_mrl=phase_mrl,
                phase_mrl_median=float(np.median(phase_mrl)),
                phase_mean_offset_rad=phase_offset,
                phase_rmse_rad=phase_rmse,
                relative_log_gain_corr=corr,
                relative_log_gain_rmse=gain_rmse)


def _m4(snapshot, cg, weights):
    corr = snapshot["corr"]
    G_prev, th_prev, u_prev, sig_prev = snapshot["prev"]
    G_now = snapshot["G"]
    # Exact B-block reduction under the repo's flat convention:
    # c_j = sum_k conj(g_{n,jk}^on) g_{n-1,jk}^on.
    zero_a = [np.zeros_like(x) for x in G_prev["a"]]
    b_only = dict(a=zero_a, b=G_prev["b"],
                  c=np.zeros_like(G_prev["c"]))
    c_b_chain = chain_c_stored(b_only, th_prev, u_prev, sig_prev,
                               snapshot["h_online"])
    c_b_direct = [np.sum(np.conj(G_now["b"][l]) * G_prev["b"][l], axis=1)
                  for l in range(tcg.L)]
    b_rel = max(float(np.max(np.abs(a - b))
                      / (np.max(np.abs(a)) + 1e-30))
                for a, b in zip(c_b_chain, c_b_direct))
    naive = [np.conj(G_now["a"][l]) * G_prev["a"][l] + c_b_direct[l]
             for l in range(tcg.L)]
    rows = {}
    def relation(a, b, wt):
        wt = wt / (np.sum(wt) + 1e-30)
        za = a / np.maximum(np.abs(a), 1e-30)
        zb = b / np.maximum(np.abs(b), 1e-30)
        resultant = np.sum(wt * za * np.conj(zb))
        return dict(mrl=float(np.abs(resultant)),
                    mean_offset_rad=float(np.angle(resultant)))

    for layer in range(tcg.L):
        rows[str(layer)] = dict(
            raw_vs_learned=relation(corr[layer], snapshot["w"][layer],
                                    weights[layer]),
            ema_vs_learned=relation(snapshot["corr_ema"][layer],
                                    snapshot["w"][layer], weights[layer]),
            adam_drive_vs_learned=relation(snapshot["corr_adam"][layer],
                                           snapshot["w"][layer],
                                           weights[layer]),
            raw_vs_cg=relation(corr[layer], cg[layer], weights[layer]),
            ema_vs_cg=relation(snapshot["corr_ema"][layer], cg[layer],
                               weights[layer]),
            adam_drive_vs_cg=relation(snapshot["corr_adam"][layer],
                                      cg[layer], weights[layer]),
            naive_raw_crosscorr_vs_actual=relation(naive[layer], corr[layer],
                                                    weights[layer]),
        )
    return dict(
        convention=("c_n = J_{n-1}^dagger g_n^on; the descended RoutePC "
                    "meta-gradient is -LR*c_n, and subtracting it moves w "
                    "in the +c_n direction"),
        B_block_crosscorr_max_relative_error=b_rel, layers=rows)


def _oracle_distribution(w):
    """Per-layer static-credit oracle phase and magnitude effect sizes."""
    out = {}
    for layer, wl in enumerate(w):
        phase = np.abs(np.angle(wl))
        mag = np.abs(wl)
        logmag = np.log(np.maximum(mag, 1e-30))
        out[str(layer)] = dict(abs_arg_rad=_dist(phase),
                               magnitude=_dist(mag),
                               log_magnitude=_dist(logmag))
    return out


def _polar_json(w):
    z = _flatten_w(w)
    return dict(log_gain=np.log(np.maximum(np.abs(z), 1e-30)).tolist(),
                phase_rad=np.angle(z).tolist())


def _aggregate_credit_oracle(rows):
    out = {}
    for step in PRIMARY_CKPTS:
        out[str(step)] = {}
        selected = [row for row in rows if row["step"] == step]
        for layer in range(tcg.L):
            key = str(layer)
            out[str(step)][key] = {}
            for quantity in ("abs_arg_rad", "magnitude", "log_magnitude"):
                values = []
                for row in selected:
                    values.extend(row["credit_oracle_distribution"][key]
                                      [quantity]["values"])
                out[str(step)][key][quantity] = _dist(values)
    return out


def _curvature(snapshot, w):
    """Cheap diagonal curvature sample in gauge-fixed polar coordinates."""
    x = _pack_polar(w)
    rng = np.random.default_rng(6_000 + snapshot["seed"] + snapshot["step"])
    dims = rng.choice(x.size, 16, replace=False)
    eps = 2e-3
    diag = []
    for dim in dims:
        xp, xm = x.copy(), x.copy()
        xp[dim] += eps; xm[dim] -= eps
        _, gp = _polar_objective(snapshot, xp)
        _, gm = _polar_objective(snapshot, xm)
        diag.append(float((gp[dim] - gm[dim]) / (2 * eps)))
    pos = np.asarray([v for v in diag if v > 1e-10])
    return dict(sampled_dims=dims.tolist(), diagonal=diag,
                median=float(np.median(diag)), max=float(np.max(diag)),
                min=float(np.min(diag)), negative=int(np.sum(np.asarray(diag) < 0)),
                positive_condition=(float(np.max(pos) / np.min(pos))
                                    if pos.size >= 2 else None))


def _summarize_primary(rows):
    out = {}
    for step in PRIMARY_CKPTS:
        sr = [row for row in rows if row["step"] == step]
        out[str(step)] = {}
        for name in ("learned", "phase", "credit"):
            out[str(step)][name + "_minus_identity"] = _dist(
                [row["M1"][name] - row["M1"]["identity"] for row in sr])
        for optimizer in ("actual_adam", "reset_adam", "clipped_sgd"):
            out[str(step)]["M3_" + optimizer] = _dist(
                [row["M3"][optimizer] for row in sr])
    return out


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(OUT, exist_ok=True)
    stored = json.load(open(os.path.join(OUT, "gb_summary.json")))
    stored_finals = {int(k): v for k, v in stored["finals"].items()}
    audit0 = dict(rp.BPTT_CALLS)
    primary_rows = []
    m2_snapshots = {}
    fd_done = False

    for seed in range(15):
        before = dict(rp.BPTT_CALLS)
        replayed = replay(seed)
        train_calls = {k: rp.BPTT_CALLS[k] - before[k] for k in rp.BPTT_CALLS}
        assert train_calls == {"exact_grad": 0, "exact_lambda": 0}, train_calls
        assert replayed["final_loss"] == stored_finals[seed]
        print(f"  s{seed} bitwise final gate PASS; training exact 0/0",
              flush=True)
        for step, snapshot in replayed["snapshots"].items():
            if seed in M2_SEEDS and step in M2_CKPTS:
                m2_snapshots[(seed, step)] = snapshot
            if step not in PRIMARY_CKPTS:
                continue
            packs = _gather_fit(snapshot["params"], seed, step)
            cg, _ = fit_scalars(packs, snapshot["params"])
            z_credit, _ = fit_oracles(packs, snapshot["params"])
            w_credit = [np.conj(z) for z in z_credit]
            weights = _mode_weights(packs)
            candidates = dict(
                identity=[np.ones(tcg.N, complex) for _ in range(tcg.L)],
                learned=snapshot["w"],
                phase=[np.exp(1j * np.angle(w)) for w in snapshot["w"]],
                credit=w_credit,
            )
            m1 = {name: action(snapshot, w, "actual_adam")
                  for name, w in candidates.items()}
            m3 = {}
            for optimizer in ("actual_adam", "reset_adam", "clipped_sgd"):
                f_i = action(snapshot, candidates["identity"], optimizer)
                f_l = action(snapshot, candidates["learned"], optimizer)
                m3[optimizer] = float(f_l - f_i)
            row = dict(seed=seed, step=step, failure=seed in FAIL_SEEDS,
                       M1=m1, M3=m3,
                       M4=_m4(snapshot, cg, weights),
                       credit_oracle_distribution=_oracle_distribution(
                           w_credit),
                       learned_vs_credit=_geometry_compare(
                           snapshot["w"], w_credit, weights))
            primary_rows.append(row)
            if not fd_done:
                gradient_gate(snapshot)
                fd_done = True

    print("M2 multi-start optimizer-aware oracle fits...", flush=True)
    m2_rows = []
    best_by_key = {}
    for seed in M2_SEEDS:
        for step in M2_CKPTS:
            snapshot = m2_snapshots[(seed, step)]
            packs = _gather_fit(snapshot["params"], seed, step)
            z_credit, _ = fit_oracles(packs, snapshot["params"])
            w_credit = [np.conj(z) for z in z_credit]
            weights = _mode_weights(packs)
            fit = fit_action_oracle(snapshot, w_credit)
            w_f = fit["best"]["w"]
            action_space = action_space_summary(snapshot, fit, w_credit)
            best_by_key[(seed, step)] = (w_f, fit["stable"])
            row = dict(
                seed=seed, step=step, failure=seed in FAIL_SEEDS,
                stable=fit["stable"], identity_value=fit["identity_value"],
                action_oracle_value=fit["best"]["value"],
                action_oracle_improvement=fit["improvement"],
                objective_spread=fit["objective_spread"],
                pairwise_phase_mrl=fit["pairwise_phase_mrl"],
                pairwise_phase_rmse_rad=fit["pairwise_phase_rmse_rad"],
                pairwise_loggain_rmse=fit["pairwise_loggain_rmse"],
                pairwise_relative_loggain_rmse=(
                    fit["pairwise_relative_loggain_rmse"]),
                action_oracle_polar=_polar_json(w_f),
                credit_oracle_polar=_polar_json(w_credit),
                action_space=action_space,
                runs=[{k: v for k, v in run.items() if k != "w"}
                      for run in fit["runs"]],
                learned_vs_credit=_geometry_compare(snapshot["w"], w_credit,
                                                    weights),
                learned_vs_action=_geometry_compare(snapshot["w"], w_f,
                                                    weights),
                credit_vs_action=_geometry_compare(w_credit, w_f, weights),
            )
            m2_rows.append(row)
            print(f"  s{seed} K{step}: F_I {fit['identity_value']:.6f}  "
                  f"F_F* {fit['best']['value']:.6f}  spread "
                  f"{fit['objective_spread']:.2e}  phase-null-min "
                  f"{min(fit['pairwise_phase_mrl']):.3f}  "
                  f"stable={fit['stable']}", flush=True)

    # M6 is conditional on stable adjacent action-oracle estimates.
    m6_rows = []
    for seed in M2_SEEDS:
        for n0, n1 in ((500, 501), (1499, 1500)):
            w0, stable0 = best_by_key[(seed, n0)]
            w1, stable1 = best_by_key[(seed, n1)]
            if not (stable0 and stable1):
                continue
            s0 = m2_snapshots[(seed, n0)]
            learned0 = _flatten_w(_normalize_relative(s0["w"]))
            wf0 = _flatten_w(_normalize_relative(w0))
            wf1 = _flatten_w(_normalize_relative(w1))
            m6_rows.append(dict(
                seed=seed, failure=seed in FAIL_SEEDS, n0=n0, n1=n1,
                learned_to_target_raw_rms=float(np.linalg.norm(
                    _flatten_w(s0["w"]) - _flatten_w(w0))
                    / np.sqrt(N_MODES)),
                target_motion_raw_rms=float(np.linalg.norm(
                    _flatten_w(w1) - _flatten_w(w0)) / np.sqrt(N_MODES)),
                learned_to_target_rms=float(np.linalg.norm(learned0 - wf0)
                                            / np.sqrt(N_MODES)),
                target_motion_rms=float(np.linalg.norm(wf1 - wf0)
                                        / np.sqrt(N_MODES)),
                curvature=_curvature(s0, w0),
            ))
    m6_feasible = bool(
        sum(not row["failure"] for row in m6_rows) >= 2
        and sum(row["failure"] for row in m6_rows) >= 2)

    # Aggregate M4 separately by checkpoint/layer.
    m4_agg = {}
    for step in PRIMARY_CKPTS:
        m4_agg[str(step)] = {}
        sr = [row for row in primary_rows if row["step"] == step]
        for layer in range(tcg.L):
            keys = sr[0]["M4"]["layers"][str(layer)].keys()
            m4_agg[str(step)][str(layer)] = {}
            for key in keys:
                m4_agg[str(step)][str(layer)][key] = {
                    statistic: _dist([
                        row["M4"]["layers"][str(layer)][key][statistic]
                        for row in sr])
                    for statistic in ("mrl", "mean_offset_rad")}
        m4_agg[str(step)]["B_block_crosscorr_max_relative_error"] = _dist(
            [row["M4"]["B_block_crosscorr_max_relative_error"] for row in sr])

    doc = dict(
        protocol=dict(primary_checkpoints=PRIMARY_CKPTS,
                      m2_checkpoints=M2_CKPTS, m2_seeds=M2_SEEDS,
                      failure_seeds=FAIL_SEEDS, fit_batches=N_FIT,
                      m2_parameterization=("all per-mode log-gains + phase; "
                                           "log-gain bounds [-2.5,2.5]"),
                      m2_stability_rule=("multi-start objective spread <= 10% "
                                         "of improvement (floor 1e-6), minimum "
                                         "pairwise phase MRL >=0.8, maximum "
                                         "pairwise phase RMS <=0.5 rad and "
                                         "log-gain RMS <=0.5, all starts "
                                         "converged, and <10% gain coordinates "
                                         "on bounds"),
                      reset_adam=("m=v=0 at the checkpoint while retaining "
                                  "the actual optimizer-time index n"),
                      static_credit_oracle=("pooled least-squares modal "
                                            "projection of exact onto online "
                                            "recurrent (a,B) blocks; deployed "
                                            "as w_C=conj(z_C)")),
        training_exact_calls={"exact_grad": 0, "exact_lambda": 0},
        offline_exact_calls=OFFLINE_EXACT,
        gradient_gate_completed=fd_done,
        M1_M3_rows=primary_rows,
        M1_M3_aggregate=_summarize_primary(primary_rows),
        M5_credit_oracle_distribution=_aggregate_credit_oracle(primary_rows),
        M2_rows=m2_rows,
        M2_stable_count=int(sum(row["stable"] for row in m2_rows)),
        M2_action_stable_count=int(sum(
            row["action_space"]["action_stable"] for row in m2_rows)),
        M4_aggregate=m4_agg,
        M5_artifact="results/geometry_audit/b6_b8_summary.json",
        M6=dict(feasible=m6_feasible, rows=m6_rows,
                note=("success/failure comparison requires at least two "
                      "stable adjacent pairs in each group")),
        git=subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip(),
    )
    path = os.path.join(OUT, "m1_m6_action_summary.json")
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2, default=float)
    print(f"wrote {path}")


def backfill_action_space() -> None:
    """Recompute the unchanged M2 fits and add the requested action readout."""
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    path = os.path.join(OUT, "m1_m6_action_summary.json")
    doc = json.load(open(path))
    by_key = {(row["seed"], row["step"]): row for row in doc["M2_rows"]}
    for seed in M2_SEEDS:
        replayed = replay(seed)
        for step in M2_CKPTS:
            snapshot = replayed["snapshots"][step]
            packs = _gather_fit(snapshot["params"], seed, step)
            z_credit, _ = fit_oracles(packs, snapshot["params"])
            w_credit = [np.conj(z) for z in z_credit]
            fit = fit_action_oracle(snapshot, w_credit)
            row = by_key[(seed, step)]
            if abs(fit["best"]["value"] - row["action_oracle_value"]) > 1e-12:
                raise AssertionError((seed, step, fit["best"]["value"],
                                      row["action_oracle_value"]))
            row["action_space"] = action_space_summary(
                snapshot, fit, w_credit)
            regret = row["action_space"]["candidates"]["learned"][
                "objective_regret"]
            print(f"  action-space s{seed} K{step}: learned regret "
                  f"{regret:+.3e}; stable="
                  f"{row['action_space']['action_stable']}", flush=True)
    doc["M2_action_stable_count"] = int(sum(
        row["action_space"]["action_stable"] for row in doc["M2_rows"]))
    doc["protocol"]["action_stability_rule"] = (
        "objective-spread rule plus minimum pairwise action cosine >=0.999 "
        "and maximum symmetric relative action distance <=0.1")
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2, default=float)
    print(f"updated {path}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--backfill-action-space":
        backfill_action_space()
    else:
        main()
