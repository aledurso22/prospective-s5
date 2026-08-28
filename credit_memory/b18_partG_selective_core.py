"""Phase B18 Part G -- selectivity confined to the small temporal core.

B17 Part D showed faithful endogenous selectivity of an unrestricted
wide-state scalar gate destroys the small-credit-module property (true
persistent state O(N0*N1), not O(N0)). This tests the theory-agent's
proposal: A_t = I_n (x) R(q_t), with q_t computed from a low-dimensional
signal (here: mean of the lower layer, matching B17 Part D's own D3
exactly, so the two are directly comparable) -- does confining the
gate's EFFECT to the shared r x r core (rather than a full per-unit
gate) keep the resulting credit-state blowup bounded by r, instead of
by N1 = n*r?

Run:  python -m credit_memory.b18_partG_selective_core
"""
from __future__ import annotations

import numpy as np

from credit_memory.b18_temporal_core import R_FAMILIES


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def part_g_selective_core(n_list, r=4, N0=8, T_=30, seed=0, c_gate=1.0):
    rng = np.random.RandomState(seed)
    results = []
    R0 = R_FAMILIES["dense"](r, rng) * 0.85  # base core, held fixed in magnitude
    dR = rng.randn(r, r) / np.sqrt(r)        # fixed small-core modulation direction
    for n in n_list:
        N1 = n * r
        u = rng.randn(T_, N0)
        B = rng.randn(N1, N0) / np.sqrt(N0)
        theta0 = np.ones(N0)
        a0 = 0.85

        def R_of(q):
            return R0 + q * dR

        def simulate(theta, gate_kind):
            h0 = np.zeros(N0)
            h1 = np.zeros((n, r))
            for t in range(T_):
                h0 = a0 * h0 + theta * u[t]
                if gate_kind == "G1":
                    R_t = R0
                elif gate_kind == "G2":
                    R_t = R_of(c_gate * np.sin(0.3 * t))  # exogenous time-varying
                else:  # G3: endogenous, q_t = mean(h0)
                    R_t = R_of(c_gate * np.mean(h0))
                Bh0 = (B @ h0).reshape(n, r)
                h1 = h1 @ R_t.T + Bh0
            return h1.reshape(-1)

        eps = 1e-5
        for gate_kind in ("G1", "G2", "G3"):
            J_fd = np.zeros((N1, N0))
            for m in range(N0):
                tp = theta0.copy(); tp[m] += eps
                tm = theta0.copy(); tm[m] -= eps
                J_fd[:, m] = (simulate(tp, gate_kind) - simulate(tm, gate_kind)) / (2 * eps)

            # forward pass at theta0, recording shared quantities
            h0 = np.zeros(N0)
            R_t_seq, h0_seq = [], []
            for t in range(T_):
                h0 = a0 * h0 + theta0 * u[t]
                if gate_kind == "G1":
                    R_t = R0
                elif gate_kind == "G2":
                    R_t = R_of(c_gate * np.sin(0.3 * t))
                else:
                    R_t = R_of(c_gate * np.mean(h0))
                R_t_seq.append(R_t); h0_seq.append(h0.copy())
            h1_true = np.zeros((n, r)); h1_traj = []
            for t in range(T_):
                Bh0 = (B @ h0_seq[t]).reshape(n, r)
                h1_true = h1_true @ R_t_seq[t].T + Bh0
                h1_traj.append(h1_true.copy())
            h1_prev_traj = [np.zeros((n, r))] + h1_traj[:-1]

            # naive closed form: K-chain per source (Part C/F's formula),
            # using the SAME time-varying R_t (valid for G1/G2, wrong for G3).
            # Forcing is the source's OWN z-chain (z_t = a0 z_{t-1} + u_t),
            # not the raw drive u_t -- layer 0 has its own memory here,
            # unlike the memoryless single-shot source in Part C/F's check.
            J_naive = np.zeros((N1, N0))
            for m in range(N0):
                K = np.zeros((r, r))
                z = 0.0
                W = B[:, m].reshape(n, r)
                s_t = None
                for t in range(T_):
                    z = a0 * z + u[t, m]
                    K = R_t_seq[t] @ K + z * np.eye(r)
                    s_t = W @ K.T
                J_naive[:, m] = s_t.reshape(-1)

            # rank of the extra selectivity forcing term's own trajectory:
            # for G3, the extra term is (dR @ h1_prev_traj[t][i,:]) * dq/dtheta
            # per block i -- measure whether the STACKED (over t, over the n
            # blocks) extra-forcing directions span more than ~r dimensions.
            extra_dirs = []
            for t in range(T_):
                dq_dtheta_shared = 1.0 / N0  # times z_t^m, applied per-source at readout;
                # the SPATIAL direction injected at each (t,i) is dR @ h1_prev_traj[t][i,:]
                for i in range(n):
                    extra_dirs.append(dR @ h1_prev_traj[t][i, :])
            extra_mat = np.stack(extra_dirs, axis=0)  # (T*n, r)
            sv = np.linalg.svd(extra_mat, compute_uv=False)
            extra_rank = int(np.sum(sv > 1e-9 * sv[0])) if len(sv) and sv[0] > 0 else 0

            # corrected closed form for G3: adds the exact extra term per
            # source (h1_prev_traj @ dR.T) * dq_t/dtheta_m, using the
            # SHARED (not per-source) true trajectory h1_prev_traj and
            # dq_t/dtheta_m = z_t^m / N0 (q_t = mean(h0)).
            J_corrected = np.zeros((N1, N0))
            for m in range(N0):
                z = 0.0
                W = B[:, m].reshape(n, r)
                s = np.zeros((n, r))
                for t in range(T_):
                    z = a0 * z + u[t, m]
                    dq_dtheta = (z / N0) if gate_kind == "G3" else 0.0
                    extra = (h1_prev_traj[t] @ dR.T) * dq_dtheta if gate_kind == "G3" else 0.0
                    s = s @ R_t_seq[t].T + extra + W * z
                J_corrected[:, m] = s.reshape(-1)

            scale = float(np.max(np.abs(J_fd)) + 1e-30)
            gap_naive = float(np.max(np.abs(J_fd - J_naive)))
            gap_corr = float(np.max(np.abs(J_fd - J_corrected)))
            results.append(dict(n=n, N1=N1, r=r, gate=gate_kind,
                                gap_naive_rel=gap_naive / scale,
                                gap_corrected_rel=gap_corr / scale,
                                extra_term_rank=extra_rank))
    return results


def main() -> None:
    print("=" * 90)
    print("Phase B18 Part G: selectivity confined to the small temporal core")
    print("=" * 90)
    results = part_g_selective_core([2, 4, 8, 16, 32], r=4, N0=8)
    for row in results:
        print(f"  n={row['n']:3d} N1={row['N1']:4d} r={row['r']} gate={row['gate']}: "
             f"gap_naive_rel={row['gap_naive_rel']:.3e}  "
             f"gap_corrected_rel={row['gap_corrected_rel']:.3e}  "
             f"extra_term_rank={row['extra_term_rank']}")


if __name__ == "__main__":
    main()
