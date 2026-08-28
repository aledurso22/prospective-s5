"""Phase B18 Parts C/F -- exact credit-state accounting for I_n (x) R
architectures, generalizing B16.2 Part F's scalar z-chain (r=1) to a
general r x r "K-chain".

Setup (matching B16.2 Part F's own convention exactly): layer 0 is a
source h_t^0 = v_m u_t^m (fixed direction v_m, scalar drive u_t^m,
source parameter theta_m). Layer 1 has A_1 = I_n (x) R (shared r x r
block across n copies, N_1 = n*r), routing B_1 (N_1 x dim(v_m)).

Claim: d h_t^1/d theta_m = reshape_i[ K_t @ w_i ], where w_i is copy
i's FIXED r-dim slice of (B_1 v_m), and K_t is a SHARED r x r matrix
recursion K_t = R K_{t-1} + u_t^m * I_r -- independent of n (feature
multiplicity). Persistent credit state for this source-to-layer-1
sensitivity is therefore r^2, not r*n or r*N.

Run:  python -m credit_memory.b18_partCF_credit
"""
from __future__ import annotations

import numpy as np

from credit_memory.b18_temporal_core import R_FAMILIES


def verify_K_chain(n_list, r=4, T_=30, seed=0, family="dense"):
    rng = np.random.RandomState(seed)
    results = []
    for n in n_list:
        N1 = n * r
        R = R_FAMILIES[family](r, rng)
        B1 = rng.randn(N1, 1) / np.sqrt(1)   # single source direction v_m folded in
        u = rng.randn(T_)                     # scalar drive u_t^m

        # (i) ground truth: literal full-vector simulation with theta_m=1,
        # all else zero -- exact by linearity (same style as B16.2 Part F)
        s = np.zeros((n, r))
        s_true = np.zeros((T_, N1))
        for t in range(T_):
            s = s @ R.T + B1[:, 0].reshape(n, r) * u[t]
            s_true[t] = s.reshape(-1)

        # (ii) K-chain closed form: K_t = R K_{t-1} + u_t*I_r (r x r, SHARED),
        # then s_t[i,:] = K_t @ w_i, w_i = B1's i-th r-slice.
        K = np.zeros((r, r))
        s_pred = np.zeros((T_, N1))
        W = B1[:, 0].reshape(n, r)  # n x r, each row is w_i
        for t in range(T_):
            K = R @ K + u[t] * np.eye(r)
            s_pred[t] = (W @ K.T).reshape(-1)  # s_t[i,:] = K_t @ w_i = w_i @ K_t.T

        err = float(np.max(np.abs(s_true - s_pred)))
        scale = float(np.max(np.abs(s_true)) + 1e-30)
        results.append(dict(n=n, r=r, N1=N1, max_abs_err=err, max_rel_err=err / scale,
                            credit_state_Kchain=r * r, credit_state_full_would_be=N1))
    return results


def main() -> None:
    print("=" * 90)
    print("Phase B18 Parts C/F: K-chain credit-state verification for I_n (x) R")
    print("=" * 90)
    for family in ("diagonal", "oscillator", "jordan", "dense"):
        print(f"\n-- family={family}, r=4, n in {{2,4,8,16,32}} --")
        results = verify_K_chain([2, 4, 8, 16, 32], r=4, family=family)
        for row in results:
            print(f"  n={row['n']:3d} N1={row['N1']:4d}  max_rel_err={row['max_rel_err']:.2e}  "
                 f"credit_state(K-chain)={row['credit_state_Kchain']}  "
                 f"(vs full-vector {row['credit_state_full_would_be']})")

    print("\n-- r sweep at n=16, family=dense --")
    for r in (1, 2, 4, 8):
        results = verify_K_chain([16], r=r, family="dense" if r > 1 else "diagonal")
        row = results[0]
        print(f"  r={r:2d}  max_rel_err={row['max_rel_err']:.2e}  "
             f"credit_state(K-chain)=r^2={row['credit_state_Kchain']}")


if __name__ == "__main__":
    main()
