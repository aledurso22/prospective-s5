"""B37b failure characterization: coefficient->root conditioning of the monomial
quotient chart. No training; pure numerical-conditioning measurement.

For each constructive (q, u): perturb the MONOMIAL coefficient vectors a, theta
by a relative amount eps and measure (i) how far the poles of the realization
u(C_q) move, (ii) how often the perturbed realization leaves the unit disk.
"""
import numpy as np
import jax.numpy as jnp
from credit_memory.b37a_universal_quotient import mult_matrix
from credit_memory.b37b_quotient_trainability import (
    FAMILIES, R_VALUES, EVAL_SEEDS, EPS_LADDER, make_teacher,
    constructive_realization, perturb)

print(f"{'family':21s} {'r':>2s} {'eps':>7s} {'rho(M0)':>8s} {'med d(pole)':>12s} "
      f"{'max d(pole)':>12s} {'med rho(M)':>10s} {'P(unstable)':>11s} {'kappa_root':>11s}")
out = []
for f in FAMILIES:
    for r in R_VALUES:
        for eps in EPS_LADDER:
            d_med, d_max, rhos, unst, kap = [], [], [], [], []
            for seed in EVAL_SEEDS:
                t = make_teacher(f, r, seed)
                cons, _ = constructive_realization(t, r)
                M0 = np.asarray(mult_matrix(jnp.array(cons[1]), jnp.array(cons[0]), r))
                e0 = np.sort_complex(np.linalg.eigvals(M0))
                rho0 = np.max(np.abs(e0))
                for rep in range(8):
                    pp = perturb(cons, eps, seed * 100 + rep)
                    M = np.asarray(mult_matrix(jnp.array(pp[1]), jnp.array(pp[0]), r))
                    e = np.sort_complex(np.linalg.eigvals(M))
                    d = np.abs(e - e0)
                    d_med.append(np.median(d)); d_max.append(np.max(d))
                    rr = np.max(np.abs(e)); rhos.append(rr); unst.append(rr > 1.0)
                    kap.append(np.max(d) / eps)          # root displacement per unit rel. perturbation
            print(f"{f:21s} {r:2d} {eps:7.0e} {rho0:8.4f} {np.median(d_med):12.2e} "
                  f"{np.median(d_max):12.2e} {np.median(rhos):10.3f} {np.mean(unst):11.2f} "
                  f"{np.median(kap):11.2e}")
            out.append((f, r, eps, float(np.mean(unst)), float(np.median(kap))))

print()
print("Summary: median root-displacement amplification kappa = max|dpole| / eps, by family (r=8)")
for f in FAMILIES:
    ks = [k for (ff, r, e, u, k) in out if ff == f and r == 8]
    us = [u for (ff, r, e, u, k) in out if ff == f and r == 8]
    print(f"  {f:21s} kappa = {np.median(ks):9.2e}   P(unstable) over eps ladder = {us}")
