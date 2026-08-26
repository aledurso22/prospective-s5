"""Phase-trajectory diagnostic — is the learned orientation a
predictable moving object along training? (Method-gate diagnostic.)

Retrain routeA (3 seeds, same protocol), logging w every 25 steps.
Measure on arg w per (layer, mode):
  * RMS of phase increments ||Delta phi_n||
  * increment autocorrelation (persistence of motion)
  * tracking error of two cheap predictors vs the true next phase:
      hold:      phi_hat = phi_n
      momentum:  phi_hat = phi_n + beta (phi_n - phi_{n-1}), beta=1
    and the best per-mode beta.

REGISTERED BAR (method gate): the momentum predictor reduces median
tracking error >= 20% vs hold => phase motion is predictable and a
Simonetto-style prediction-correction method is worth building.
< 20% => kill the idea early.

Run:  python phase_track.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.train_cell import STEPS
from toyrig.probes import make_data

SEEDS = [0, 1, 2]
LOG_EVERY = 25
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "phase_track")


def train_with_log(seed):
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    track = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
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
        params = params_next
        if step % LOG_EVERY == 0:
            track.append([np.angle(wl) for wl in w])
    return np.array(track)          # (Ck, L, N)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    for seed in SEEDS:
        print(f"seed {seed}: routeA with phase logging...", flush=True)
        track = train_with_log(seed)                    # (Ck, L, N)
        Ck = track.shape[0]
        dphi = np.diff(track, axis=0)                   # (Ck-1, L, N)
        dphi = np.angle(np.exp(1j * dphi))              # wrap
        rms = float(np.sqrt(np.mean(dphi ** 2)))
        # increment autocorrelation (lag 1), per mode then median
        acs = []
        for l in range(track.shape[1]):
            for j in range(track.shape[2]):
                s = dphi[:, l, j]
                if np.std(s) > 1e-12:
                    acs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
        ac1 = float(np.median(acs)) if acs else 0.0
        # predictor tracking error on the wrapped phase path
        hold_err, mom_err, best_err = [], [], []
        for l in range(track.shape[1]):
            for j in range(track.shape[2]):
                p = track[:, l, j]
                for n in range(2, Ck - 1):
                    d1 = np.angle(np.exp(1j * (p[n + 1] - p[n])))
                    hold_err.append(abs(d1))
                    dm = np.angle(np.exp(1j * (p[n] - p[n - 1])))
                    mom_err.append(abs(np.angle(np.exp(1j *
                        (p[n + 1] - (p[n] + dm))))))
                    # best beta in [0, 1.5]
                    betas = np.linspace(0, 1.5, 16)
                    errs = [abs(np.angle(np.exp(1j *
                                (p[n + 1] - (p[n] + b * dm)))))
                            for b in betas]
                    best_err.append(min(errs))
        hold_med = float(np.median(hold_err))
        mom_med = float(np.median(mom_err))
        best_med = float(np.median(best_err))
        out[seed] = dict(rms_dphi=rms, ac1=ac1, hold=hold_med,
                         mom=mom_med, best_beta=best_med)
        print(f"  seed {seed}: RMS|dphi| {rms:.4f}  ac1 {ac1:+.3f}  "
              f"hold {hold_med:.4f}  momentum {mom_med:.4f}  "
              f"best-beta {best_med:.4f}", flush=True)
    hold = np.median([out[s]["hold"] for s in SEEDS])
    mom = np.median([out[s]["mom"] for s in SEEDS])
    red = (hold - mom) / hold
    win = red >= 0.2
    print("-" * 70)
    print(f"median tracking error: hold {hold:.4f}  momentum {mom:.4f}  "
          f"reduction {red:.2f}")
    print(f"BAR (>= 20% reduction): {'PREDICTABLE — Simonetto method viable' if win else 'KILL prediction-correction idea'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(dict(git=git, per_seed=out, hold=hold, mom=mom,
                       reduction=red, win=bool(win)), f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
