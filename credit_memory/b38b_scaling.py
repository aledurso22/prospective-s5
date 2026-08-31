"""B38b section 5 -- persistent exact eligibility memory vs recurrent size N.

The exact sensitivity-module size is  sum_phi dim M_phi = sum_phi |supp phi|
(channels influenced), because J_t = diag(a_t) is exactly diagonal in BOTH arms.
It is measured EMPIRICALLY at small N as the number of structurally nonzero
entries of the dense RTRL sensitivity S_t, and the analytic formula is validated
against that measurement before being used at larger N.

Input dimension m = J (one input channel per tile) so the shared selector
projection's parameter count grows with model width, as it does in Mamba where
the Delta projection is a model-width-sized linear map.
"""
import json
import numpy as np
import jax.numpy as jnp

from credit_memory.b38b_selective import (
    Q_BOTT, M_LOC, init_L, init_S, fixed_R, nparams)
from credit_memory.b38b_exactness import full_dense_rtrl

D_TILE = 4          # bounded tile size, O(1), held FIXED across the sweep


def analytic(J, d, m, arm):
    q = Q_BOTT
    if arm == "L":
        # Within a tile only W, p act through the bottleneck g and so touch all d
        # channels; Atil, uD, cD, uB, cB are per-channel and touch exactly one.
        # (An earlier count assumed every tile parameter touched all d channels;
        # the empirical support measurement below corrected it.)
        p_bott = q * M_LOC + q                       # W, p   -> d channels each
        p_chan = d + d * q + d + d * q + d           # Atil, uD, cD, uB, cB -> 1 each
        Ptau = p_bott + p_chan
        P_rec = J * Ptau
        M = J * (d * p_bott + p_chan)
        return P_rec, M, Ptau
    P_shared = q * m + q + q + 1              # W, p, uD, cD  -> ALL N channels
    P_local = J * d * (1 + q + 1)             # Atil, uB, cB  -> 1 channel each
    N = J * d
    return P_shared + P_local, N * P_shared + P_local, P_shared


print("=" * 118)
print("VALIDATION: analytic module size vs EMPIRICALLY measured support of dense RTRL S_t")
print("=" * 118)
print(f"{'arm':>4s} {'J':>3s} {'d':>2s} {'m':>3s} {'N':>4s} {'P_rec':>7s} "
      f"{'measured sum dim M':>19s} {'analytic':>10s} {'match':>6s}")
ok = True
for arm in ("L", "S"):
    for (J, d, m) in [(2, 2, 2), (3, 2, 3), (4, 3, 4), (3, 4, 3), (5, 3, 5)]:
        R = fixed_R(J, m) if arm == "L" else None
        p = init_L(J, d, m, 0) if arm == "L" else init_S(J, d, m, 0)
        rng = np.random.RandomState(3)
        xs = jnp.asarray(rng.randn(12, m) * 0.7); ys = jnp.asarray(rng.randn(12) * 0.5)
        _, Sup, keys, sizes = full_dense_rtrl(p, xs, ys, J, d, arm, R)
        meas = int(Sup.sum())
        P_rec, M, _ = analytic(J, d, m, arm)
        good = (meas == M)
        ok &= good
        print(f"{arm:>4s} {J:3d} {d:2d} {m:3d} {J*d:4d} {P_rec:7d} {meas:19d} {M:10d} "
              f"{'yes' if good else 'NO':>6s}")
print(f"\n  analytic formula reproduces the measured module size exactly: {ok}")

print()
print("=" * 118)
print(f"SCALING SWEEP: d_tile = {D_TILE} held fixed (bounded), m = J, N = J * d_tile")
print("=" * 118)
print(f"{'N':>6s} {'J':>5s} {'m':>5s} | {'P_rec L':>9s} {'M_elig L':>11s} {'M/P L':>7s} "
      f"| {'P_rec S':>9s} {'M_elig S':>12s} {'M/P S':>9s} | {'S/L ratio':>10s}")
rows = []
for J in (2, 4, 8, 16, 32, 64, 128, 256, 512):
    N, m = J * D_TILE, J
    PL, ML, _ = analytic(J, D_TILE, m, "L")
    PS, MS, Psh = analytic(J, D_TILE, m, "S")
    rows.append(dict(N=N, J=J, m=m, P_L=PL, M_L=ML, rL=ML / PL,
                     P_S=PS, M_S=MS, rS=MS / PS, P_shared=Psh))
    print(f"{N:6d} {J:5d} {m:5d} | {PL:9d} {ML:11d} {ML/PL:7.2f} "
          f"| {PS:9d} {MS:12d} {MS/PS:9.2f} | {(MS/PS)/(ML/PL):10.1f}")
json.dump(rows, open("results/b38b/scaling.json", "w"), indent=1)
print()
print(f"  Arm L  M_elig/P is CONSTANT in N and bounded by d_tile = {D_TILE}  -> sum_phi dim M_phi = O(P)")
r = [x["rS"] for x in rows]
print(f"  Arm S  M_elig/P grows {r[0]:.2f} -> {r[-1]:.2f} over N = {rows[0]['N']}..{rows[-1]['N']}"
      f"  (ratio of ratios {r[-1]/r[0]:.0f}x); M_elig itself is O(N^2), i.e. O(N * P_selector)")
print(f"  Both arms have an EXACTLY DIAGONAL J_t: a small/diagonal Jacobian alone does NOT")
print(f"  imply O(P) exact RTRL -- trainable selector FAN-OUT is what decides it.")
