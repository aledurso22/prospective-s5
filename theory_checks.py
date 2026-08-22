"""PROSPECTIVE-CREDIT — Phase 1 algebra harness (numpy only, no model).

Verifies the discrete complex-SSM analogue of the GLE phase identity
(PROSPECTIVE_SSM_RESEARCH_HANDOFF.md §9 and §15), mode by mode:

    forward mode : h_{t+1} = a h_t + b x_t,        |a| < 1
    exact adjoint: lambda_t = q_t + conj(a) lambda_{t+1}   (BPTT — NONCAUSAL)
    prospective  : e_t      = q_t - a q_{t-1}              (CAUSAL)

    H_BPTT(w) = 1 / (1 - conj(a) e^{iw}) ,   H_pro(w) = 1 - a e^{-iw}

    H_pro / H_BPTT = |1 - conj(a) e^{iw}|^2   (real, >= 0)
        =>  arg H_pro(w) == arg H_BPTT(w)   for every w, every stable a
        =>  |H_pro| / |H_BPTT| = |1 - conj(a) e^{iw}|^2  (gain mismatch)

NOTE the conjugation asymmetry — it is load-bearing (handoff §25): the
adjoint recursion carries conj(a), the causal filter carries a. The ratio
is real because (1 - a e^{-iw}) = conj(1 - conj(a) e^{iw}).

Checks (handoff §15, Tests A/B/C):
  [A] reverse-recursion adjoint == closed-form H_BPTT on Fourier tones
  [B] causal prospective filter == closed-form H_pro on the same tones
  [C] phase identity (max error < 1e-6) and gain identity, swept over
      real and complex a, |a| in {0, 0.5, 0.9, 0.99(, 0.999)}, arg a != 0

This must pass before any training experiment. Run:  python theory_checks.py
"""
from __future__ import annotations

import numpy as np

PHASE_TOL = 1e-6     # handoff §15 acceptance threshold
IMPL_TOL = 1e-9      # implementation vs closed form (relative)
GAIN_TOL = 1e-12     # gain identity (relative)

FAILURES = []


def adjoint_reverse(q: np.ndarray, a: complex) -> np.ndarray:
    """Exact BPTT adjoint: lambda_t = q_t + conj(a) lambda_{t+1}, lambda_T = 0."""
    T = len(q)
    lam = np.zeros(T + 1, np.complex128)
    for t in range(T - 1, -1, -1):
        lam[t] = q[t] + np.conj(a) * lam[t + 1]
    return lam[:T]


def prospective_causal(q: np.ndarray, a: complex) -> np.ndarray:
    """Causal prospective error: e_t = q_t - a q_{t-1}, q_{-1} = 0."""
    e = q.astype(np.complex128).copy()
    e[1:] -= a * q[:-1]
    return e


def H_bptt(a: complex, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 - np.conj(a) * np.exp(1j * w))


def H_pro(a: complex, w: np.ndarray) -> np.ndarray:
    return 1.0 - a * np.exp(-1j * w)


def check(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    print(f"  {name:<58s} err = {err:.3e}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append(name)


def measure_transfer(a: complex, w: float, T: int = 8192, W: int = 2048):
    """Measured transfer of the two recursions on the tone q_t = e^{iwt}.

    Projects lambda_t and e_t onto the tone over a centered window of W
    samples (integer number of cycles: w = 2 pi k / W), so the mean is the
    exact steady-state transfer value. Window margins of (T - W)/2 >> 1/(1-|a|)
    kill both the causal and the adjoint transients.
    """
    t = np.arange(T)
    q = np.exp(1j * w * t)
    lam = adjoint_reverse(q, a)
    e = prospective_causal(q, a)
    s0 = (T - W) // 2
    sl = slice(s0, s0 + W)
    tone = np.exp(-1j * w * t[sl])
    return np.mean(lam[sl] * tone), np.mean(e[sl] * tone)


def mode_sweep():
    mags = [0.0, 0.5, 0.9, 0.99]
    phases = [0.0, np.pi / 5, -2 * np.pi / 3, np.pi]   # incl. real a (0, pi)
    modes = [m * np.exp(1j * p) for m in mags for p in phases]
    rng = np.random.RandomState(0)
    modes += [m * np.exp(1j * p) for m, p in
              zip(rng.uniform(0.0, 0.999, 8), rng.uniform(-np.pi, np.pi, 8))]
    return modes


def test_AB_implementations() -> None:
    print("[A/B] reverse-recursion adjoint and causal filter vs closed forms")
    W = 2048
    ks = [0, 1, 2, 3, 5, 8, 13, 21, 55, 144, 377, 987, 1023]  # DC .. ~pi
    worst_lam = worst_e = 0.0
    for a in mode_sweep():
        for k in ks:
            w = 2 * np.pi * k / W
            H_lam, H_e = measure_transfer(a, w, W=W)
            Hb, Hp = H_bptt(a, w), H_pro(a, w)
            worst_lam = max(worst_lam,
                            abs(H_lam - Hb) / max(1.0, abs(Hb)))
            worst_e = max(worst_e, abs(H_e - Hp) / max(1.0, abs(Hp)))
    check("adjoint_reverse == H_BPTT (24 modes x 13 tones)",
          worst_lam, IMPL_TOL)
    check("prospective_causal == H_pro (24 modes x 13 tones)",
          worst_e, IMPL_TOL)


def test_C_phase_and_gain_identity() -> None:
    print("[C] phase identity arg H_pro == arg H_BPTT, and gain identity")
    # dense formula sweep
    mags = [0.0, 0.5, 0.9, 0.99, 0.999]
    phases = [0.0, np.pi / 6, -np.pi / 6, np.pi / 2, -2 * np.pi / 3, np.pi]
    modes = [m * np.exp(1j * p) for m in mags for p in phases]
    rng = np.random.RandomState(1)
    modes += [m * np.exp(1j * p) for m, p in
              zip(rng.uniform(0.0, 0.999, 32), rng.uniform(-np.pi, np.pi, 32))]
    w = np.linspace(-np.pi, np.pi, 513)
    worst_phase = worst_gain = 0.0
    for a in modes:
        Hb, Hp = H_bptt(a, w), H_pro(a, w)
        worst_phase = max(worst_phase,
                          np.max(np.abs(np.angle(Hp * np.conj(Hb)))))
        gain_meas = np.abs(Hp) / np.abs(Hb)
        gain_pred = np.abs(1.0 - np.conj(a) * np.exp(1j * w)) ** 2
        worst_gain = max(worst_gain,
                         np.max(np.abs(gain_meas - gain_pred) / gain_pred))
    check("phase identity (62 modes x 513 freqs)", worst_phase, PHASE_TOL)
    check("gain identity |H_pro|/|H_BPTT| = |1-conj(a)e^iw|^2",
          worst_gain, GAIN_TOL)

    # same identity through the IMPLEMENTATIONS (tones, not formulas):
    # phase(lambda) must equal phase(e) per (mode, frequency)
    W = 2048
    worst_impl = 0.0
    for a in mode_sweep():
        for k in [1, 2, 5, 13, 34, 89, 233, 610, 987]:
            w = 2 * np.pi * k / W
            H_lam, H_e = measure_transfer(a, w, W=W)
            worst_impl = max(worst_impl,
                             abs(np.angle(H_e * np.conj(H_lam))))
    check("phase identity via recursions (24 modes x 9 tones)",
          worst_impl, PHASE_TOL)


def main() -> None:
    print("=" * 72)
    print("Prospective-credit phase theorem — algebra checks (handoff §15)")
    print("=" * 72)
    test_AB_implementations()
    test_C_phase_and_gain_identity()
    print("=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) above tolerance")
        raise SystemExit(1)
    print("ALL THEORY CHECKS PASSED (phase err < 1e-6)")
    print("=" * 72)


if __name__ == "__main__":
    main()
