"""The derivation as written, run verbatim — and where it breaks.

No stability fallback, no clamps: f_theta(s,t) = A s + B x with A = Lambda
(the HiPPO/S5 matrix itself), exactly as in the derivation. This script
reports, at the real S5 initialization:

  1. the companion spectrum  mu^2 - a1 mu - a2 = 0,  mu1*mu2 = -a2 = A;
  2. the step at which the state overflows float32;
  3. the same spectrum under partial prospection (--gamma), showing the
     failure survives even at gamma = 0 — explicit Euler cannot integrate
     the HiPPO spectrum, prospective term or not.

Run:  python exact_failure.py
"""
from __future__ import annotations

import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("_hippo", "ssm/shared/hippo.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


def companion_roots(A: np.ndarray, rho: float, gamma: float = 1.0) -> np.ndarray:
    """Roots of mu^2 - a1 mu - a2 for partial prospection strength gamma.

    a1 = (1-rho) + (rho+gamma)A,  a2 = -gamma*A,  so mu1*mu2 = gamma*A:
    gamma scales the parasitic root linearly and kills it at gamma=0.
    """
    a1 = (1.0 - rho) + (rho + gamma) * A
    a2 = -gamma * A
    return np.array([np.roots([1.0, -x, -y]) for x, y in zip(a1, a2)])


def rollout(A: np.ndarray, rho: float, T: int = 784, seed: int = 0):
    """Run s_t = a1 s_{t-1} + a2 s_{t-2} + x~_t on white-noise input.

    Returns the first step at which |s| leaves float32 range (or None).
    """
    rng = np.random.RandomState(seed)
    N = A.shape[0]
    a1 = (1.0 - rho) + (1.0 + rho) * A
    a2 = -A
    x = rng.randn(T).astype(np.float32)
    s1 = np.zeros(N, np.complex128)
    s2 = np.zeros(N, np.complex128)
    peak = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(T):
            xt = (1.0 + rho) * (x[t - 1] if t >= 1 else 0.0) - (x[t - 2] if t >= 2 else 0.0)
            s = a1 * s1 + a2 * s2 + xt
            s2, s1 = s1, s
            m = np.max(np.abs(s))
            peak = max(peak, m) if np.isfinite(m) else peak
            if not np.isfinite(m) or m > np.finfo(np.float32).max:
                return t, m, peak
    return None, np.max(np.abs(s1)), peak


def report(name: str, A: np.ndarray, rho: float) -> None:
    mu = companion_roots(A, rho)
    # The PHYSICAL root is the one approximating the continuous mode, i.e. the
    # one nearest (1 - rho); the other is the parasitic/companion root created
    # by the two-step discretisation. Selecting by magnitude would mislabel
    # them whenever the parasitic root is the larger one — which is exactly
    # what happens in the exact case.
    i = np.argmin(np.abs(mu - (1.0 - rho)), axis=1)
    phys = mu[np.arange(len(A)), i]
    ghost = mu[np.arange(len(A)), 1 - i]
    step, val, peak = rollout(A, rho)
    print(f"  {name}")
    print(f"    |A| max            : {np.abs(A).max():.4e}")
    print(f"    max |mu|           : {np.abs(mu).max():.6f}"
          f"   {'UNSTABLE' if np.abs(mu).max() > 1 else 'stable'}")
    print(f"    physical root range: [{np.abs(phys).min():.6f}, {np.abs(phys).max():.6f}]"
          f"   (1-rho = {1 - rho:.6f})")
    print(f"    parasitic |mu| max : {np.abs(ghost).max():.4e}"
          f"   {'<-- the ghost' if np.abs(ghost).max() > 1 else ''}")
    if step is None:
        print(f"    rollout T=784      : finite, final |s| = {val:.3e}")
    else:
        print(f"    rollout T=784      : OVERFLOWED float32 at step {step}"
              f"  (|s| = {val:.3e})")


def main() -> None:
    N = 64
    Lambda, _, _ = _h.hippo_init(N)
    print("=" * 72)
    print("The prospective SSM derivation at the real S5/HiPPO init (N=64)")
    print("=" * 72)
    print(f"HiPPO spectrum: Re(lambda) = {Lambda.real[0]:.3f} for every mode, "
          f"|lambda| in [{np.abs(Lambda).min():.4f}, {np.abs(Lambda).max():.2f}]")
    print()
    print("EXACT — the derivation verbatim (A = Lambda, no clamps):")
    for rho in (0.5, 0.1, 1e-3):
        report(f"rho = Delta t/tau = {rho}", Lambda, rho)
    print()
    print("mu1*mu2 = A: for the stiff modes the roots split into the physical")
    print("  ~1/(1+rho) and the parasitic ~(1+rho)*A (measured 1433.60 at")
    print("  rho=0.1, |A|max ~ 1303) — the instability is set by the spectrum,")
    print("  not by the step size, so no rho can fix it.")
    print()
    print("TURNING PROSPECTION OFF (--gamma), exact A = Lambda, rho = 0.1:")
    print(f"    {'gamma':>7} {'max|mu|':>14}   {'mu1*mu2 = gamma*A':>18}")
    for g in (1.0, 0.5, 0.1, 0.01, 0.0):
        mu = companion_roots(Lambda, 0.1, g)
        flag = "" if np.abs(mu).max() <= 1 else "  UNSTABLE"
        print(f"    {g:7.2f} {np.abs(mu).max():14.6f}   {g * np.abs(Lambda).max():18.2f}{flag}")
    print("  gamma=0 removes the prospective term entirely: a2 = 0, the second")
    print("  root is 0, and the recurrence is first-order explicit-Euler S5.")
    print()
    print("  BUT IT IS STILL UNSTABLE: max|mu| = 130.33 at gamma=0. That root")
    print("  is the PHYSICAL one, a1 = (1-rho) + rho*A, and |rho*A| = 130 at")
    print("  rho=0.1, |Lambda|max=1303. So explicit Euler cannot integrate the")
    print("  HiPPO spectrum even with no prospective term at all — a THIRD")
    print("  failure, independent of both the cancellation and the parasite.")
    print("  There is no trainable variant of this scheme: the only control")
    print("  that trains is the bilinear baseline (--model baseline).")
    print("=" * 72)


if __name__ == "__main__":
    main()
