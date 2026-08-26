"""The ghost of the explicit prospective SSM — standalone demo (numpy only).

Reproduces the diagnosis in the paper (Sec. 2.2), no training required:

  A. characteristic roots of the prospective two-step recurrence:
        mu^2 - (1 - (1+rho)h) mu - h = 0   =>   mu_phys * mu_ghost = -h
  B. the recurrence in simulation: stable at small stiffness h, divergent at
     large h (the memory directions); clamping |h| kills the memory instead
  C. RK4 integrates the ghost MORE faithfully — order is not stability
  D. the fix: the first-order prospective-metric (IMEX) scheme and its exact
     stability boundary dt <= 2*alpha*tau

Run:  python ghost_demo.py
"""
from __future__ import annotations

import numpy as np


def roots(h, rho=1.0):
    """Characteristic roots of mu^2 - (1-(1+rho)h) mu - h = 0."""
    b = -(1 - (1 + rho) * h)
    disc = np.sqrt(b * b + 4 * h + 0j)
    return (-b + disc) / 2, (-b - disc) / 2


def part_A():
    print("=" * 72)
    print("A. Characteristic roots (rho = dt/tau = 1)")
    print(f"{'h':>8} {'mu_phys':>12} {'mu_ghost':>12}  note")
    for h in [0.1, 1.0, 10.0, 100.0]:
        m1, m2 = roots(h)
        phys, ghost = (m1, m2) if abs(m1) < abs(m2) else (m2, m1)
        note = "ghost suppressed" if abs(ghost) <= 1 else "GHOST UNSTABLE"
        print(f"{h:>8} {phys:>12.3f} {ghost:>12.3f}  {note}")
    print("product check mu_phys*mu_ghost = -h:",
          [f"{np.prod(roots(h)):.1f}" for h in [0.1, 1.0, 10.0]])
    print("large-h asymptote: |mu_ghost| ~ (1+rho)*h")


def simulate_recurrence(h, T=300, rho=1.0):
    """The explicit prospective recurrence on a single stiff mode, tracking a
    constant target s* = 1. Scalar form of the characteristic dynamics:
        s_t = (1-(1+rho)h) s_{t-1} + h s_{t-2} + rho*h
    (fixed point s*=1; the two roots of part A govern the deviation).
    """
    s = np.zeros(T + 2)
    for t in range(2, T + 2):
        s[t] = (1 - (1 + rho) * h) * s[t - 1] + h * s[t - 2] + rho * h
    return s[2:]


def part_B():
    print("=" * 72)
    print("B. The recurrence in simulation (target s*=1, T=300)")
    for h in [0.3, 0.9, 2.0, 10.0]:
        s = simulate_recurrence(h)
        print(f"  h={h:5.1f}:  final |s| = {abs(s[-1]):.3e}   "
              f"{'converged' if abs(s[-1] - 1) < 1e-3 else 'DIVERGED / oscillating'}")
    print("  clamp h<=0.9 to survive  =>  |lambda| clamped well below 1")
    print("  =>  the long-memory directions are exactly what must be sacrificed")


def part_C():
    print("=" * 72)
    print("C. RK4 on the prospective ODE — order is not stability")
    # Prospective coordinate dynamics: tau^2 h s'' = +h(1 - s)  (bare theory,
    # Sec 2.9: negative mass => branches exp(+-t/tau); the + branch is the ghost)
    # Integrate with RK4 at decreasing dt: error on the ghost branch SHRINKS.
    tau = 1.0
    # perturbation delta around the target obeys tau^2 delta'' = +delta
    # (Sec 2.9: negative mass => branches exp(+-t/tau); the + branch = ghost)
    def rhs(y):
        return np.array([y[1], y[0] / tau**2])
    T = 3.0
    for dt in [0.1, 0.05, 0.01]:
        n = int(T / dt)
        y = np.array([1.0, 1.0 / tau])   # pure ghost branch: s = exp(+t/tau)
        for _ in range(n):
            k1 = rhs(y); k2 = rhs(y + dt / 2 * k1)
            k3 = rhs(y + dt / 2 * k2); k4 = rhs(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        exact = np.exp(T / tau)
        print(f"  dt={dt:5.2f}: RK4 s(T)={y[0]:10.4f}  exact ghost e^3={exact:8.4f}"
              f"  rel err {abs(y[0]-exact)/exact:.2e}  <- smaller dt tracks the"
              f" ghost BETTER")


def part_D():
    print("=" * 72)
    print("D. The fix: first-order prospective-metric (IMEX) scheme")
    print("   multiplier mu(h) = 1 - rho*h/((1-alpha) + alpha*h);  |mu|<=1 for all")
    print("   h>=0  iff  dt <= 2*alpha*tau   (CFL law, paper Eq. 2.7)")
    for alpha in [0.0, 0.5, 1.0]:
        # empirical boundary: bisect the largest stable rho on h in [1e-3, 1e6]
        hs = np.logspace(-3, 6, 400)
        def stable(rho):
            mu = 1 - rho * hs / ((1 - alpha) + alpha * hs)
            return np.all(np.abs(mu) <= 1 + 1e-12)
        lo, hi = 0.0, 50.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if stable(mid): lo = mid
            else: hi = mid
        note = ("stiffness-free boundary 2*alpha" if alpha > 0 else
                "finite-spectrum boundary 2/h_max = 2e-6 here")
        print(f"  alpha={alpha:4.1f}:  measured dt_max/tau = {lo:10.6f}   ({note})")
    print("  alpha=1 (full prospective/Newton metric) is the MOST stable point;" )
    print("  alpha=0 (explicit gradient) collapses as 2/h_max.")


if __name__ == "__main__":
    part_A(); part_B(); part_C(); part_D()
