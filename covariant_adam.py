"""Gauge-covariant Adam — the one experiment the field-theory reading
predicts (directive: physics -> what to change).

Claim under test: the learned per-mode phase rotation is a GAUGE REPAIR
for a non-covariant optimizer. Adam normalizes Re and Im of each
complex gradient separately (v per real coordinate), which is not
covariant under per-mode U(1) rotations: rotate g_j -> e^{i phi} g_j
and the step changes. A covariant variant normalizes each complex entry
by its modulus (shared v per complex pair): updates then rotate with
the gradient by construction. If the phase defect is an optimizer-
covariance artifact, online credit + covariant Adam should recover the
phase arm's benefit with NO learned w and NO estimation barrier.

Arms (paired seeds {0,1,2}, same protocol):
  online_cov   online credit + covariant Adam (the test)
  routeA_cov   route A meta-gradient + covariant Adam (secondary: if
               the gauge reading is right, there is less phase left to
               repair -> learned |arg w| should shrink vs routeA's)

References (deterministic, same protocol): online/routeA from
pac_deploy/summary.json; phase-only from factorize_w/summary.json.

REGISTERED BARS (fixed before running):
  MAIN: online_cov closes >= 50% of the online -> routeA gap -> the
  defect is optimizer covariance; new algorithm, no learned parts.
  < 20% -> defect is architectural (the .real routing), also valuable.
  SECONDARY: median |arg w| learned under covAdam < routeA's.

Run:  python covariant_adam.py
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

SEEDS = [0, 1, 2]
ARMS = ["online_cov", "routeA_cov"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "covariant_adam")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")
FACT_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "factorize_w", "summary.json")
W_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "factorize_w")
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8


def pair_indices():
    """(re, im) index pairs of complex entries in the flat vector:
    b blocks per layer and the c block at the end."""
    pairs, off = [], 0
    for l in range(tcg.L):
        m = tcg.N if l == 0 else tcg.N  # b shapes: (N, M_IN=1) then (N,N)
        m_in = 1 if l == 0 else tcg.N
        base = off + 2 * tcg.N
        pairs += [(base + i, base + tcg.N * m_in + i)
                  for i in range(tcg.N * m_in)]
        off += 2 * tcg.N + 2 * tcg.N * m_in
    pairs += [(off + i, off + tcg.N + i) for i in range(tcg.N)]
    return pairs


def cov_adam(flat, g, m, v, step, pairs):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g ** 2
    for i, j in pairs:                      # shared v per complex pair
        s = 0.5 * (v[i] + v[j])
        v[i] = v[j] = s
    upd = LR * (m / (1 - B1 ** step)) / (np.sqrt(v / (1 - B2 ** step))
                                         + EPS)
    return flat - upd, m, v


def train_arm(arm, seed, pairs):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    losses = []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        G_use = G if arm == "online_cov" else cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cov_adam(flat, g, m, v, step, pairs)
        params_next = tcg.pack(params, flat)
        if arm == "routeA_cov":
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
                            off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        params = params_next
    losses = np.asarray(losses)
    return (float(losses[-100:].mean()), bool(np.all(np.isfinite(losses))),
            w)


def main() -> None:
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    with open(FACT_SUMMARY) as f:
        fact = json.load(f)
    pairs = pair_indices()
    table, w_cov = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            fl, fin, w = train_arm(arm, seed, pairs)
            table.setdefault(arm, []).append(fl)
            if arm == "routeA_cov":
                w_cov[seed] = w
            print(f"  seed {seed} {arm:<11s} final {fl:.4f} finite {fin}",
                  flush=True)
    med = {a: float(np.median(v)) for a, v in table.items()}
    gap = ref["online"] - ref["routeA"]
    frac = (ref["online"] - med["online_cov"]) / gap
    # secondary: learned phase magnitude under covAdam vs routeA's
    arg_cov = [float(np.median(np.abs(np.concatenate(
        [np.angle(wl) for wl in w_cov[s]])))) for s in SEEDS]
    arg_ref = [float(np.median(np.abs(np.concatenate(
        [np.angle(wl) for wl in np.load(
            os.path.join(W_DIR, f"w_full_s{s}.npy"))])))) for s in SEEDS]
    main_win = frac >= 0.5
    print("-" * 70)
    print(f"medians { {k: round(v, 4) for k, v in med.items()} }  "
          f"refs online {ref['online']:.4f} routeA {ref['routeA']:.4f}")
    print(f"online_cov closes {frac:.2f} of the gap")
    print(f"median |arg w|: covAdam {np.median(arg_cov):.3f} vs "
          f"routeA {np.median(arg_ref):.3f} rad")
    print(f"BAR MAIN (>= 50%): {'OPTIMIZER-COVARIANCE CONFIRMED' if main_win else 'NO WIN'}"
          f"{'' if frac >= 0.2 else ' -> defect is architectural (.real routing)'}")
    print(f"BAR SECONDARY: learned phase {'shrank' if np.median(arg_cov) < np.median(arg_ref) else 'did NOT shrink'}")
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, refs=ref, per_arm=table, medians=med, frac=frac,
               arg_cov=arg_cov, arg_ref=arg_ref, main_win=bool(main_win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
