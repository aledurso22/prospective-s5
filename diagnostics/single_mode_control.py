"""STAGE D4 — routing-free single-mode bandwidth control.

One isolated recurrent mode, no C, no routing, no eligibility:

    lam_t = q_t + conj(a) lam_{t+1}      (exact future sum, computable
    directly), a = r e^{i theta}.

Regimes (T = 256):
  narrowband   q'_t (demodulated) = smooth lowpass noise (baseband at
               the pole — the Taylor/narrowband condition holds);
  broadband    q_t = white complex noise (the measured-network regime:
               the network's error signal lives ~1.6 rad/sample off the
               pole, spectrum_check.py);
  network-q    the actual top-layer q from the trained rig (seed 0),
               demodulated per mode — the real thing.

Arms: q (identity), c0* q, c0* q + c1* (q_t - q_{t-1}),
ema lead (rho = 0.99 and rho = r), cstat q (exact least-squares scalar
from the regime's own autocorrelation).

Endpoint: signal-level cos(estimate, lam) and |phase error|.

Registered interpretation tree:
  * Taylor arm succeeds ONLY on narrowband -> Stage-A(v1) failure is
    bandwidth/regime;
  * Taylor arm succeeds on realistic q but failed in-network ->
    routing/projection matters;
  * cstat succeeds broadly -> the signal-statistics projection is the
    right reduced object;
  * even cstat poor -> one complex scalar cannot represent future
    credit from current information.

Run:  python single_mode_control.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

T = 256
R = 0.995
THETA = 0.3
SEED = 7
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "single_mode_control")


def exact_lambda(q, a):
    lam = np.zeros_like(q)
    acc = 0.0 + 0.0j
    for t in range(len(q) - 1, -1, -1):
        acc = q[t] + np.conj(a) * acc
        lam[t] = acc
    return lam


def arms_eval(q, a):
    lam = exact_lambda(q, a)
    r = abs(a)
    c0 = 1.0 / (1.0 - r)
    c1 = r / (1.0 - r) ** 2
    # demodulated frame
    t = np.arange(len(q))
    qp = q * np.exp(-1j * np.angle(a) * t)

    def scs(est):
        est = est * np.exp(1j * np.angle(a) * t)
        c = np.abs(np.vdot(lam, est)) / (np.linalg.norm(lam)
                                         * np.linalg.norm(est) + 1e-30)
        ph = np.abs(np.angle(np.vdot(lam, est)))
        rel = np.linalg.norm(est - lam) / (np.linalg.norm(lam) + 1e-30)
        return float(c), float(ph), float(rel)

    # lag-1 autocorrelation for cstat
    rho1 = np.mean(qp[1:] * np.conj(qp[:-1])) / (np.mean(np.abs(qp) ** 2)
                                                 + 1e-30)
    # cstat from the autocorrelation sequence (K < T)
    K = max(2, T // 2)
    rho_k = np.array([np.mean(qp[k:] * np.conj(qp[:-k]))
                      / (np.mean(np.abs(qp) ** 2) + 1e-30)
                      for k in range(1, K)])
    cstat = 1.0 + np.sum(np.conj(a) ** np.arange(1, K) * rho_k)

    d = np.diff(qp, prepend=qp[:1])
    out = {"q": qp,
           "c0": c0 * qp,
           "c0+c1dq": c0 * qp + c1 * d,
           "ema0.99": None, "matched": None, "cstat": cstat * qp}
    for name, rho in [("ema0.99", 0.99), ("matched", r)]:
        m = np.zeros_like(qp)
        for tt in range(1, len(qp)):
            m[tt] = rho * m[tt - 1] + (1 - rho) * qp[tt - 1]
        out[name] = c0 * qp + c1 * (1 - rho) * (qp - m)
    res = {}
    for k, v in out.items():
        if v is None:
            continue
        res[k] = scs(v)
    return res, float(np.abs(rho1)), float(np.angle(rho1))


def make_narrowband(rng):
    env = np.convolve(rng.standard_normal(T + 64),
                      np.ones(32) / 32, mode="same")[32:32 + T]
    return env * np.exp(1j * THETA * np.arange(T))


def make_broadband(rng):
    return (rng.standard_normal(T) + 1j * rng.standard_normal(T)) \
        / np.sqrt(2)


def make_networkq(rng):
    from toyrig import ssm_rig as tcg
    from toyrig import route_a as cvm
    from toyrig.probes import make_data
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    params = tcg.init_params(0)
    x, y = make_data(np.random.RandomState(1000))
    # short online training to make q meaningful
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    rr = np.random.RandomState(1000)
    for step in range(1, 501):
        xx, yy = make_data(rr)
        loss, G = cvm.batch_grad(params, xx, yy)[:2]
        g = cvm.clip(tcg.flat_grads(G, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:tcg.DELAY] = 0.0
    q = tcg.spatial_q(params, h, r)[tcg.L - 1][:, 0, :]     # (T, N)
    # pick the mode closest to the target pole magnitude
    a_all = params["a"][tcg.L - 1]
    j = int(np.argmin(np.abs(np.abs(a_all) - R)))
    return q[:, j], a_all[j]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    a = R * np.exp(1j * THETA)
    out = {}
    qn, an = None, None
    for regime in ["narrowband", "broadband", "network-q"]:
        if regime == "narrowband":
            q = make_narrowband(rng)
            a_use = a
        elif regime == "broadband":
            q = make_broadband(rng)
            a_use = a
        else:
            qn, an = make_networkq(rng)
            q, a_use = qn, an
        res, r1abs, r1ang = arms_eval(q, a_use)
        out[regime] = dict(pole=abs(a_use), theta=float(np.angle(a_use)),
                           rho1_abs=r1abs, rho1_phase=r1ang,
                           arms={k: dict(cos=v[0], phase_err=v[1],
                                         rel=v[2])
                                 for k, v in res.items()})
        print(f"{regime}: pole |a|={abs(a_use):.4f}  "
              f"rho(1): |.|={r1abs:.3f} phase {r1ang:+.3f}")
        for k, v in res.items():
            print(f"    {k:<9s} cos {v[0]:.3f}  phase err {v[1]:.3f}  rel {v[2]:.3f}")
        print()

    nb = out["narrowband"]["arms"]
    bb = out["broadband"]["arms"]
    nq = out["network-q"]["arms"]
    taylor_nb_only = (nb["c0+c1dq"]["cos"] > nb["q"]["cos"]
                      and bb["c0+c1dq"]["cos"] <= bb["q"]["cos"] + 0.02)
    cstat_broad = all(out[reg]["arms"]["cstat"]["cos"]
                      > out[reg]["arms"]["q"]["cos"]
                      for reg in out)
    print("-" * 70)
    print(f"Taylor helps only narrowband: {taylor_nb_only}")
    print(f"cstat beats identity in every regime: {cstat_broad}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    out["git"] = git
    out["interpretation"] = dict(taylor_nb_only=taylor_nb_only,
                                 cstat_broad=cstat_broad)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
