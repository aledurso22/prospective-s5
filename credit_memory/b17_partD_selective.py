"""Phase B17 Part D -- faithful selective cross-layer test. Standalone
(no tcg dependence): builds the actual two-layer routed model with a
tied SCALAR upper-layer gate, differentiates w.r.t. a LOWER-layer
parameter (not the pole parameter itself, unlike B16.2 Part H), and
checks whether the standard non-selective closed form still gives the
exact gradient, or whether a corrected (but still small, O(N0)) closed
form is needed, or whether closure is genuinely lost.

h_t^0[m] = a0 h_{t-1}^0[m] + theta_m u_t^m      (N0 independent lower
                                                  channels/sources)
h_t^1    = a_t(...) h_{t-1}^1 + B h_t^0          (tied scalar gate)

D1 constant, D2 exogenous time-varying, D3 endogenous a_t = a(mean(h_t^0)).

Run:  python -m credit_memory.b17_partD_selective
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def part_d_faithful_selective(N0_list, N1=8, T_=40, seed=0, c_gate=1.5):
    rng = np.random.RandomState(seed)
    results = []
    for N0 in N0_list:
        u = rng.randn(T_, N0)
        exo_signal = rng.randn(T_)
        a0 = 0.85
        B = rng.randn(N1, N0) / np.sqrt(N0)
        rho_pole = 0.3
        theta0 = np.ones(N0)

        def gate_arg(h0_t, gate_kind, t):
            if gate_kind == "D1":
                return rho_pole
            if gate_kind == "D2":
                return rho_pole + c_gate * exo_signal[t]
            return rho_pole + c_gate * np.mean(h0_t)  # D3

        def simulate(theta, gate_kind):
            h0 = np.zeros(N0)
            h1 = np.zeros(N1)
            for t in range(T_):
                h0 = a0 * h0 + theta * u[t]
                a_t = sigmoid(gate_arg(h0, gate_kind, t))
                h1 = a_t * h1 + B @ h0
            return h1.copy()

        eps = 1e-5
        for gate_kind in ("D1", "D2", "D3"):
            # (i) ground truth: central finite difference on the true simulator
            J_fd = np.zeros((N1, N0))
            for m in range(N0):
                tp = theta0.copy(); tp[m] += eps
                tm = theta0.copy(); tm[m] -= eps
                J_fd[:, m] = (simulate(tp, gate_kind) - simulate(tm, gate_kind)) / (2 * eps)

            # forward pass at theta0, recording shared quantities once
            h0 = np.zeros(N0)
            a_t_seq, darg_seq, h0_seq = [], [], []
            for t in range(T_):
                h0 = a0 * h0 + theta0 * u[t]
                arg = gate_arg(h0, gate_kind, t)
                a_t_seq.append(sigmoid(arg)); darg_seq.append(dsigmoid(arg))
                h0_seq.append(h0.copy())
            h1_true = np.zeros(N1)
            h1_traj = []
            for t in range(T_):
                h1_true = a_t_seq[t] * h1_true + B @ h0_seq[t]
                h1_traj.append(h1_true.copy())
            h1_prev_traj = [np.zeros(N1)] + h1_traj[:-1]

            # (ii) naive closed form: standard non-selective per-source scalar
            # z-chain (B16.2 Part F's exact formula) -- correct for D1/D2 by
            # construction, missing the selectivity cross-term for D3.
            J_naive = np.zeros((N1, N0))
            for m in range(N0):
                z, w = 0.0, np.zeros(N1)
                for t in range(T_):
                    z = a0 * z + u[t, m]
                    w = a_t_seq[t] * w + B[:, m] * z
                J_naive[:, m] = w

            # (iii) corrected closed form: adds the exact extra term
            # h_{t-1}^1 * (d a_t/d theta_m), with d a_t/d theta_m =
            # dsigmoid(arg) * c_gate/N0 * z_t^m for this scalar-mean gate --
            # still only needs m's OWN z-chain plus the two SHARED (already
            # source-independent) time series a_t[t], dsigmoid(arg)[t].
            J_corrected = np.zeros((N1, N0))
            for m in range(N0):
                z, w = 0.0, np.zeros(N1)
                for t in range(T_):
                    z = a0 * z + u[t, m]
                    extra = (darg_seq[t] * c_gate / N0 * z) * h1_prev_traj[t] \
                        if gate_kind == "D3" else 0.0
                    w = a_t_seq[t] * w + extra + B[:, m] * z
                J_corrected[:, m] = w

            scale = float(np.max(np.abs(J_fd)) + 1e-30)
            gap_naive = float(np.max(np.abs(J_fd - J_naive)))
            gap_corr = float(np.max(np.abs(J_fd - J_corrected)))

            # Does the corrected recursion's extra forcing term stay along a
            # FIXED direction (as the non-selective case's B[:,m] does), or
            # does it genuinely explore an N1-dimensional subspace over time?
            # Rank of the shared h1_prev_traj trajectory answers this: rank 1
            # would mean the correction is still compressible to a scalar
            # chain; rank > 1 means it is not, and the true persistent state
            # needed is O(N0*N1), not O(N0).
            traj_mat = np.stack(h1_prev_traj, axis=0)  # (T, N1)
            sv = np.linalg.svd(traj_mat, compute_uv=False)
            eff_rank = int(np.sum(sv > 1e-9 * sv[0])) if len(sv) and sv[0] > 0 else 0

            results.append(dict(N0=N0, gate=gate_kind,
                                gap_naive_rel=gap_naive / scale,
                                gap_corrected_rel=gap_corr / scale,
                                extra_term_rank=eff_rank,
                                credit_state_naive=2 * N0,          # N0 z-chains (scalar each)
                                credit_state_corrected_true=2 * N0 * N1))  # N0 running N1-vectors
    return results


def main() -> None:
    print("=" * 90)
    print("Phase B17 Part D: faithful selective cross-layer test")
    print("=" * 90)
    results = part_d_faithful_selective([1, 2, 4, 8, 16])
    for r in results:
        print(f"  N0={r['N0']:3d} gate={r['gate']}: "
             f"gap_naive_rel={r['gap_naive_rel']:.3e}  gap_corrected_rel={r['gap_corrected_rel']:.3e}  "
             f"extra_term_rank={r['extra_term_rank']}  "
             f"state(naive/corrected)={r['credit_state_naive']}/{r['credit_state_corrected_true']}")


if __name__ == "__main__":
    main()
