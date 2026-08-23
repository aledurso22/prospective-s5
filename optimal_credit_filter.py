"""The optimal causal k-tap credit filter — the decisive credit-lane test.

Context (research/prospective-credit-s5): the prospective error filter
e_t = q_t - a q_{t-1} is phase-matched to the BPTT adjoint but loses to
online RTRL on gradient cosine everywhere measured (gain inversion).
This script asks the question that closes the family:

    among ALL causal FIR error filters  e_t = sum_{k<K} w_k q_{t-k}
    (complex taps, per layer per mode), what is the best achievable
    gradient cosine to exact BPTT — and where does the prospective
    choice w = (1, -a) sit relative to that optimum?

The gradient is linear in the taps, so the best cosine is the angle
between the true gradient and the tap-span — EXACT least squares per
mode, no training loop:

    cos_opt(K) = || A (A^+ g*) || / || g* ||,   A = tap->gradient map.

Two pairings are measured (see gradient_alignment.py): the S-slot (error
paired with the accumulated RTRL sensitivity; online_full is the identity
tap, EXACT at L=1) and the J-slot (adjoint pairing).

And the decisive follow-up: TRANSFER. The optimal taps are oracle-fitted
against the exact gradient on one realization. If they keep their
advantage on a FRESH realization of the same model, a fixed per-mode
rule exists (learnable in principle); if not, the headroom is
inaccessible to any fixed causal rule and the lane stays closed.

Run:  python optimal_credit_filter.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import gradient_alignment as ga

KS = [1, 2, 3, 4, 8]
L_SWEEP = [1, 2, 4, 8]
MAG_SWEEP = [0.5, 0.9, 0.99]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")


def shift_back(q, k):
    if k == 0:
        return q
    out = np.zeros_like(q)
    out[k:] = q[:-k]
    return out


def cell_matrices(params, x, ystar, L):
    """Per (layer, mode, slot): the tap->gradient map A (real 2n x 2Kmax),
    the exact gradient g_r, and its norm."""
    h, yhat = ga.forward_np(params, x, L)
    r = yhat - ystar
    q = ga.spatial_errors(params, h, r, L)
    lam = ga.exact_adjoint(params, q, L)
    xs = ga.layer_inputs(params, h, x, L)
    Sa, Sb = ga.sensitivities(params, h, x, L)
    Kmax = max(KS)
    mats = {}
    for l in range(L):
        M = xs[l].shape[1]
        h_prev = np.concatenate([np.zeros((1, ga.N)), h[l][:-1]], axis=0)
        for j in range(ga.N):
            n = 1 + M
            g = np.zeros(n, np.complex128)
            cl = np.conj(lam[l][:, j])
            g[0] = np.sum(cl * h_prev[:, j])
            g[1:] = np.einsum("t,tm->m", cl, xs[l])
            g_r = np.concatenate([g.real, -g.imag])
            gn = np.linalg.norm(g_r)
            if gn < 1e-12:
                continue
            for slot in ("S", "J"):
                m = np.zeros((n, Kmax), np.complex128)
                for k in range(Kmax):
                    cq = np.conj(shift_back(q[l][:, j], k))
                    if slot == "S":
                        m[0, k] = np.sum(cq * Sa[l][:, j])
                        m[1:, k] = np.einsum("t,tm->m", cq, Sb[l][:, j, :])
                    else:
                        m[0, k] = np.sum(cq * h_prev[:, j])
                        m[1:, k] = np.einsum("t,tm->m", cq, xs[l])
                A = np.zeros((2 * n, 2 * Kmax))
                A[:n, 0::2] = m.real
                A[:n, 1::2] = m.imag
                A[n:, 0::2] = -m.imag
                A[n:, 1::2] = m.real
                mats[(l, j, slot)] = (A, g_r, gn)
    return mats


def fit_cell(params, x, ystar, L):
    """Fit optimal taps on one realization; report fit cosines."""
    mats = cell_matrices(params, x, ystar, L)
    a = params["a"]
    rows = []
    taps = {}
    for (l, j, slot), (A, g_r, gn) in mats.items():
        row = dict(layer=l, mode=j, slot=slot)
        g_on = A[:, 0]
        row["cos_online"] = float(
            np.dot(g_on, g_r) / (np.linalg.norm(g_on) * gn))
        xp = np.zeros(A.shape[1])
        xp[0], xp[1] = 1.0, 0.0
        xp[2], xp[3] = -a[l][j].real, -a[l][j].imag
        row["cos_prospective"] = float(
            np.dot(A @ xp, g_r) / (np.linalg.norm(A @ xp) * gn))
        for K in KS:
            Ak = A[:, :2 * K]
            coef = np.linalg.lstsq(Ak, g_r, rcond=None)[0]
            row[f"cos_opt_{K}"] = float(np.linalg.norm(Ak @ coef) / gn)
            if K <= 4:
                taps[(l, j, K, slot)] = coef
        rows.append(row)
    return rows, taps


def transfer_cell(params, x2, ystar2, L, taps):
    """Evaluate taps fitted on realization 1 on a fresh realization 2."""
    mats = cell_matrices(params, x2, ystar2, L)
    rows = []
    for (l, j, slot), (A, g_r, gn) in mats.items():
        row = dict(layer=l, mode=j, slot=slot)
        g_on = A[:, 0]
        row["cos_online"] = float(
            np.dot(g_on, g_r) / (np.linalg.norm(g_on) * gn))
        for K in (1, 2, 4):
            coef = taps[(l, j, K, slot)]
            est = A[:, :2 * K] @ coef
            row[f"cos_transfer_{K}"] = float(
                np.dot(est, g_r) / (np.linalg.norm(est) * gn + 1e-300))
        rows.append(row)
    return rows


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def main() -> None:
    print("=" * 78)
    print("The optimal causal k-tap credit filter + transfer check")
    print("=" * 78)
    rng = np.random.RandomState(ga.SEED)
    x_wb = rng.randn(ga.T, ga.D_IN)
    ystar_wb = rng.randn(ga.T, ga.D_OUT)
    rng2 = np.random.RandomState(777)               # fresh realization
    x2_wb = rng2.randn(ga.T, ga.D_IN)
    ystar2_wb = rng2.randn(ga.T, ga.D_OUT)
    B_all, C, phases = ga.make_base_params(ga.SEED)

    ks = sorted(set(
        int(round(th * ga.T / (2 * np.pi))) % ga.T
        for th in np.angle(phases)[[0, 2, 5]]))
    tt = np.arange(ga.T)

    def narrowband(rng_):
        def make(D):
            sig = np.zeros((ga.T, D))
            for d in range(D):
                for k in ks:
                    sig[:, d] += np.sin(2 * np.pi * k * tt / ga.T
                                        + rng_.uniform(0, 2 * np.pi))
            return sig / np.sqrt(len(ks))
        return make(ga.D_IN), make(ga.D_OUT)

    x_nb, ystar_nb = narrowband(np.random.RandomState(ga.SEED))
    x2_nb, ystar2_nb = narrowband(np.random.RandomState(778))

    results = {}
    for regime, x, ystar, x2, ystar2 in [
            ("broadband", x_wb, ystar_wb, x2_wb, ystar2_wb),
            ("narrowband", x_nb, ystar_nb, x2_nb, ystar2_nb)]:
        for L in L_SWEEP:
            for mag in MAG_SWEEP:
                params = ga.cell_params(B_all, C, phases, L, mag)
                rows, taps = fit_cell(params, x, ystar, L)
                rows2 = transfer_cell(params, x2, ystar2, L, taps)
                for slot in ("S", "J"):
                    rs = [r for r in rows if r["slot"] == slot]
                    rt = [r for r in rows2 if r["slot"] == slot]
                    print(f"    {regime} L={L} |a|={mag} [{slot}] "
                          f"online {med(rs, 'cos_online'):.3f}  "
                          f"prosp {med(rs, 'cos_prospective'):.3f}  "
                          f"fit1 {med(rs, 'cos_opt_1'):.3f} "
                          f"fit2 {med(rs, 'cos_opt_2'):.3f} "
                          f"fit4 {med(rs, 'cos_opt_4'):.3f} | "
                          f"transfer1 {med(rt, 'cos_transfer_1'):.3f} "
                          f"transfer2 {med(rt, 'cos_transfer_2'):.3f} "
                          f"transfer4 {med(rt, 'cos_transfer_4'):.3f}")
                results[f"{regime}/L{L}/a{mag}"] = dict(
                    fit={slot: {k: med([r for r in rows if r["slot"] == slot],
                                       k)
                                for k in ["cos_online", "cos_prospective",
                                          "cos_opt_1", "cos_opt_2",
                                          "cos_opt_3", "cos_opt_4",
                                          "cos_opt_8"]}
                         for slot in ("S", "J")},
                    transfer={slot: {k: med([r for r in rows2
                                             if r["slot"] == slot], k)
                                     for k in ["cos_online", "cos_transfer_1",
                                               "cos_transfer_2",
                                               "cos_transfer_4"]}
                              for slot in ("S", "J")})

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, branch=branch,
               config=dict(Ks=KS, L_sweep=L_SWEEP, mag_sweep=MAG_SWEEP,
                           regimes=["broadband", "narrowband"]),
               cells=results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "optimal_credit_filter.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
