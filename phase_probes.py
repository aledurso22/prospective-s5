"""Phase probes on the trained routeA trajectory — three registered
mechanism measurements, one retrain per seed.

ITEM 3 — per-layer falsification profile. At checkpoints, per layer:
    cos(G^l_on, G^l_BPTT),   ||G^l_on - G^l_BPTT|| / ||G^l_BPTT||
    plus w_l (arg, |w|).
    FRAMING NOTE (registered before running): within Zucchet's rule the
    exact object is the within-layer RTRL sensitivity (exact dh_t/dtheta
    — our rig computes Sa/Sb exactly); the approximation is the
    INSTANTANEOUS-ERROR contraction, which drops A^dagger lam_{t+1} at
    EVERY layer including the top one (tcg.exact_lambda shows it in
    code). The top layer therefore isolates the PURE TEMPORAL DEFECT
    (its q is the true loss cotangent; no cross-layer approximation) —
    it is not expected to be exact; the profile's expected shape is
    smallest error at top, growing shallower. A result that would
    falsify the temporal-repair reading: error NOT concentrated where
    the learned phases act, or the top-layer defect being negligible
    while phases there are large.

ITEM 5 — modal mechanism. Frozen params at each checkpoint; per
    layer/mode from M independent probe batches:
        rho_{l,j}(k) = E[q_{t+k,j} conj(q_{t,j})] / E|q_{t,j}|^2,
        c*_{l,j} = sum_{k=0..K} conj(a_{l,j})^k rho_{l,j}(k)
    (the signal-statistics prediction — NOT w = D^{-1}). Compare
    arg w_{l,j} vs arg c*_{l,j}: weighted mean resultant length of the
    angle differences (weights E|q_j|^2), per layer. Gain (|c*| vs |w|)
    reported but flagged less interpretable (Adam absorbs gain).

ITEM 6 — frozen-model minibatch noise floor. At the same frozen
    checkpoints, with the step's stored (G, theta) context: draw 32
    independent next batches, compute the surrogate residual
    r^^{(m)} per batch, per mode:
        SNR_j = |E_m r^_j|^2 / E_m |r^_j - E_m r^_|^2 .
    Also compare the frozen-batch noise variance against the
    along-training ||Delta r^||^2 from route_pc_pro's stored series:
    fraction ~= 1 => the training-time residual motion is pure
    minibatch noise (nothing systematic to predict).

Run:  python phase_probes.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [250, 500, 1000, 1500]
K_MAX = 60
M_CSTAR = 16
M_NOISE = 32
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "phase_probes")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def chain_c(Gp, th_all, u_all, sig_all, h_n):
    """The surrogate residual r^ = du + i dv per (layer, mode)."""
    out = []
    off = 0
    for l in range(tcg.L):
        th = th_all[l]
        u_mode = u_all[l]
        sigp = sig_all[l]
        A = Gp["a"][l] * np.exp(1j * th)
        Gb = Gp["b"][l]
        M_ = Gb.shape[1]
        gN_rho = h_n[off:off + tcg.N]
        gN_theta = h_n[off + tcg.N:off + 2 * tcg.N]
        gN_bre = h_n[off + 2 * tcg.N:
                    off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
        gN_bim = h_n[off + 2 * tcg.N + tcg.N * M_:
                    off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
        off += 2 * tcg.N + 2 * tcg.N * M_
        du = (gN_rho * sigp * A.real
              + gN_theta * (-u_mode) * A.imag
              + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
        dv = (gN_rho * sigp * A.imag
              + gN_theta * (u_mode) * A.real
              + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
        out.append(du + 1j * dv)
    return out


def probe_cstar(params, rng):
    """rho_{l,j}(k) and c*_{l,j} from M_CSTAR frozen-batch q series."""
    num = [[np.zeros(K_MAX + 1, np.complex128) for _ in range(tcg.N)]
           for _ in range(tcg.L)]
    den = [np.zeros(tcg.N) for _ in range(tcg.L)]
    for _ in range(M_CSTAR):
        x, y = make_data(rng)
        h, yhat = tcg.forward(params, x)
        r = yhat - y
        r[:tcg.DELAY] = 0.0
        q = tcg.spatial_q(params, h, r)
        for l in range(tcg.L):
            ql = np.asarray(q[l], np.complex128)       # (T, B, N)
            den[l] += np.mean(np.abs(ql) ** 2, axis=(0, 1))
            for k in range(K_MAX + 1):
                if k == 0:
                    prod = ql * np.conj(ql)
                else:
                    prod = ql[k:] * np.conj(ql[:-k])
                for j in range(tcg.N):
                    num[l][j][k] += prod[:, :, j].mean()
    cstar = []
    rho = []
    for l in range(tcg.L):
        a = params["a"][l]
        cs = np.zeros(tcg.N, np.complex128)
        rl = []
        for j in range(tcg.N):
            r_k = num[l][j] / (den[l][j] * M_CSTAR + 1e-30)
            rl.append(r_k)
            cs[j] = np.sum(np.conj(a[j]) ** np.arange(K_MAX + 1) * r_k)
        cstar.append(cs)
        rho.append(rl)
    return cstar, rho


def probe_layers(params, x, y):
    """Item 3: per-layer cos and relative error, online vs exact."""
    _, G_on = cvm.batch_grad(params, x, y)[:2]
    G_ex = cvm.exact_grad(params, x, y)
    rows = []
    for l in range(tcg.L):
        on = np.concatenate([G_on["a"][l].ravel(), G_on["b"][l].ravel()])
        ex = np.concatenate([G_ex["a"][l].ravel(), G_ex["b"][l].ravel()])
        cosl = float(np.abs(np.vdot(ex, on))
                     / (np.linalg.norm(ex) * np.linalg.norm(on) + 1e-30))
        rel = float(np.linalg.norm(on - ex)
                    / (np.linalg.norm(ex) + 1e-30))
        rows.append(dict(cos=cosl, rel=rel))
    return rows


def probe_noise_floor(Gp, th_all, u_all, sig_all, params, rng):
    """Item 6: surrogate residual across 32 independent next batches at a
    frozen model — SNR per mode and total noise variance."""
    r_all = []
    for _ in range(M_NOISE):
        x, y = make_data(rng)
        _, G = cvm.batch_grad(params, x, y)[:2]
        h_n = tcg.flat_grads(G, params)
        r_all.append(np.concatenate(chain_c(Gp, th_all, u_all, sig_all,
                                            h_n)))
    r_all = np.asarray(r_all)                     # (M, L*N) complex
    mu = r_all.mean(axis=0)
    var = np.mean(np.abs(r_all - mu) ** 2, axis=0)
    snr = np.abs(mu) ** 2 / (var + 1e-30)
    return snr, float(np.sum(var))


def audit_train(seed):
    """routeA retrain (registered protocol) with checkpoint probes."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    snaps = []
    prev = None
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)

        if step in CHECKPOINTS:
            prng = np.random.RandomState(555000 + seed * 100 + step)
            layer_rows = probe_layers(params_next, x, y)
            cstar, _rho = probe_cstar(params_next, prng)
            if prev is not None:
                Gp, th_all, u_all, sig_all = prev
                snr, nvar = probe_noise_floor(Gp, th_all, u_all,
                                              sig_all, params_next, prng)
            else:
                snr, nvar = None, None
            snaps.append(dict(step=step, w=[wl.copy() for wl in w],
                              params=params_next, layers=layer_rows,
                              cstar=cstar, rho=_rho, snr=snr,
                              noise_var=nvar))

        # registered routeA w update (exact same-batch teacher)
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
                        off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
            off += 2 * tcg.N + 2 * tcg.N * M_
            du = (gN_rho * sigp * A.real
                  + gN_theta * (-u_mode) * A.imag
                  + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
            dv = (gN_rho * sigp * A.imag
                  + gN_theta * (u_mode) * A.real
                  + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(tcg.L)])
        params = params_next
        if step % 500 == 0:
            print(f"    seed {seed} step {step}: loss {loss:.4f}",
                  flush=True)
    return snaps, w, float(np.asarray(losses)[-100:].mean())


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"seeds": {}}
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain + checkpoint probes...",
              flush=True)
        snaps, w_final, fin = audit_train(seed)
        print(f"  final loss {fin:.4f}", flush=True)
        rows = []
        for snap in snaps:
            row = {"step": snap["step"], "layers": snap["layers"]}
            # item 3 attachments: w per layer
            row["w"] = [dict(arg=np.angle(snap["w"][l]).tolist(),
                             abs=np.abs(snap["w"][l]).tolist())
                        for l in range(tcg.L)]
            # item 5: arg w vs arg c* per layer (weighted MRL)
            mrl = []
            mederr = []
            for l in range(tcg.L):
                zw = snap["w"][l]
                zc = snap["cstar"][l]
                # weights: E|q_j|^2 proxy = |rho_j(0)|
                wt = np.array([abs(snap["rho"][l][j][0])
                               for j in range(tcg.N)])
                wt = wt / (wt.sum() + 1e-30)
                dphi = np.angle(zw * np.conj(zc))
                mrl.append(float(np.abs(np.sum(wt * np.exp(1j * dphi)))))
                mederr.append(float(np.sum(wt * np.abs(dphi))))
            row["mrl"] = mrl
            row["med_angle_err"] = mederr
            row["cstar_abs"] = [np.abs(snap["cstar"][l]).tolist()
                                for l in range(tcg.L)]
            row["w_abs"] = [np.abs(snap["w"][l]).tolist()
                            for l in range(tcg.L)]
            # item 6
            if snap["snr"] is not None:
                snr = snap["snr"].reshape(tcg.L, tcg.N)
                row["snr_median_per_layer"] = [
                    float(np.median(snr[l])) for l in range(tcg.L)]
                row["noise_var"] = snap["noise_var"]
            rows.append(row)
        out["seeds"][seed] = dict(final_loss=fin, rows=rows)
        for row in rows:
            print(f"  s{seed} n={row['step']:>4d}  "
                  f"cos {[round(r['cos'], 3) for r in row['layers']]}  "
                  f"rel {[round(r['rel'], 3) for r in row['layers']]}  "
                  f"MRL {[round(m, 3) for m in row['mrl']]}", flush=True)

    # ---- item 6 global: frozen noise vs along-training ||dr||^2 ----
    d0 = os.path.dirname(os.path.abspath(__file__))
    train_dr2 = []
    for seed in SEEDS:
        z = np.load(os.path.join(d0, "results", "route_pc_pro",
                                 f"series_pro0.0_s{seed}.npz"))
        dr = z["dr"]
        train_dr2.append(float(np.mean(dr[len(dr) // 2:] ** 2)))
    frozen = [r["noise_var"] for s in out["seeds"].values()
              for r in s["rows"] if "noise_var" in r]
    frac = float(np.median(frozen) / np.median(train_dr2))
    out["noise_floor"] = dict(frozen_batch_var=float(np.median(frozen)),
                              training_dr2=float(np.median(train_dr2)),
                              fraction=frac)
    print(f"item 6: frozen-batch noise var {np.median(frozen):.3e}  "
          f"training ||dr||^2 {np.median(train_dr2):.3e}  "
          f"fraction {frac:.2f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    out["git"] = git
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else o)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
