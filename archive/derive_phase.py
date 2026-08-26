"""Derive, don't learn, the phase — the Level-3 experiment.

factorize_w.py showed a frozen LEARNED phase closes 113% of the
online->full gap. But arg w was meta-learned (with exact credit in the
outer loop). The decisive question for any "the principle derives the
rule" claim: can the phase be computed ANALYTICALLY from the credit
operator and the task, with no meta-gradient and no BPTT?

The theory: causal truncation makes the per-mode credit operator
non-self-adjoint, D_j(w) = 1 - conj(a_j) e^{iw}; exact credit needs
D^{-1}, the matched filter conj(D) is phase-exact (proven:
gradient_alignment.py). w is ONE complex number per mode, so the
derivation must integrate D over frequency — the weighting IS the
theory. Candidate: the mode's own power response W_j(w) = |1 -
a_j e^{-iw}|^{-2} (the frequencies the mode actually carries; residual
spectrum ~flat for white-input copy at init).

Arms (frozen metrics, D=50/T=128 protocol, seeds {0,1,2}):

  online            w = 1
  phase_learned     w = exp(i arg w_full)         (factorize_w's metric)
  phase_alpha_init  w = exp(i arg alpha_j)        alpha fitted ONCE at
                    init via exact credit (no meta-gradient after)
  phase_theory      w = exp(i psi_j), psi_j = arg ∫ W_j conj(D_j) dw
                    — no credit computation anywhere

Diagnostics printed per layer: median arg(alpha), arg(psi_theory) and
their gap (the zero-parameter derivation check), plus the inverse-weight
variant arg ∫ W_j / D_j dw for comparison.

REGISTERED BAR (fixed before running): phase_theory closes >= 50% of
the online -> phase_learned gap on median final loss.

Run:  python derive_phase.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig.probes import probe_blocks
from diagnostics.factorize_w import train_frozen

SEEDS = [0, 1, 2]
ARMS = ["online", "phase_learned", "phase_alpha_init", "phase_theory"]
W_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "factorize_w")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "derive_phase")
NOMEGA = 4096


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def psi_theory(a):
    """Zero-parameter phase candidates for one mode a (complex)."""
    om = np.linspace(0, 2 * np.pi, NOMEGA, endpoint=False)
    D = 1.0 - np.conj(a) * np.exp(1j * om)
    W = 1.0 / np.abs(1.0 - a * np.exp(-1j * om)) ** 2
    matched = np.angle(np.mean(W * np.conj(D)))
    inverse = np.angle(np.mean(W / D))
    return matched, inverse


def alpha_init(params, seed):
    """Per-(layer,mode) fitted exact-credit correction at init params."""
    rng = np.random.RandomState(900 + seed)
    alpha = [[None] * tcg.N for _ in range(tcg.L)]
    for (l, j, u, v, a) in probe_blocks(params, rng):
        alpha[l][j] = a
    return alpha


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table = {}
    for seed in SEEDS:
        params0 = tcg.init_params(seed)
        alpha = alpha_init(params0, seed)
        w_full = list(np.load(os.path.join(W_DIR, f"w_full_s{seed}.npy")))
        w_alpha = [np.exp(1j * np.angle([alpha[l][j]
                                         for j in range(tcg.N)]))
                   for l in range(tcg.L)]
        w_theory, w_inv = [], []
        for l in range(tcg.L):
            pm, pi = zip(*[psi_theory(params0["a"][l][j])
                           for j in range(tcg.N)])
            w_theory.append(np.exp(1j * np.array(pm)))
            w_inv.append(np.exp(1j * np.array(pi)))
        # derivation check: theory phase vs fitted alpha phase
        for l in range(tcg.L):
            da = np.angle([alpha[l][j] for j in range(tcg.N)])
            dt = np.angle(w_theory[l])
            di = np.angle(w_inv[l])
            dl = np.angle(w_full[l])
            gap_t = np.median(np.abs(np.angle(np.exp(1j * (dt - da)))))
            gap_i = np.median(np.abs(np.angle(np.exp(1j * (di - da)))))
            gap_l = np.median(np.abs(np.angle(np.exp(1j * (dl - da)))))
            print(f"  seed {seed} L{l}: med arg(a) {np.median(da):+.3f}  "
                  f"|theory-alpha| {gap_t:.3f}  |inv-alpha| {gap_i:.3f}  "
                  f"|learned-alpha| {gap_l:.3f} rad", flush=True)
        variants = {
            "online": [np.ones(tcg.N, np.complex128)
                       for _ in range(tcg.L)],
            "phase_learned": [np.exp(1j * np.angle(wl)) for wl in w_full],
            "phase_alpha_init": w_alpha,
            "phase_theory": w_theory,
        }
        for arm in ARMS:
            fl, fin = train_frozen(seed, variants[arm])
            table.setdefault(arm, []).append(fl)
            print(f"  seed {seed} {arm:<16s} final {fl:.4f} "
                  f"finite {fin}", flush=True)

    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    gap = med["online"] - med["phase_learned"]
    frac = ((med["online"] - med["phase_theory"]) / gap
            if gap > 0 else float("nan"))
    frac_a = ((med["online"] - med["phase_alpha_init"]) / gap
              if gap > 0 else float("nan"))
    print("-" * 70)
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"gap online->learned {gap:.4f}; closed by theory {frac:.2f}, "
          f"by alpha_init {frac_a:.2f}")
    win = frac >= 0.5
    print(f"BAR: theory phase closes >= 50%  ->  "
          f"{'DERIVED, NOT LEARNED' if win else 'NOT DERIVED (yet)'}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, config=dict(seeds=SEEDS, bar="theory >= 50% of gap"),
               per_arm=table, medians=med, frac_theory=frac,
               frac_alpha_init=frac_a, win=win)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
