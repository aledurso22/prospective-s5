"""Phase A: finite-horizon causal-dual (E1/E2) derivation, verified.

Branch: credit-memory-repair. Analysis/verification only -- no training,
no Stage 0, no new causal-learning mechanism. See
docs/CREDIT_MEMORY_PHASE_A.md for the full derivation this script checks.

Two checks, both against the repo's own trusted exact machinery
(toyrig.ssm_rig.exact_lambda + assemble(direct=True), the same BPTT
reference used throughout controls/ and diagnostics/):

  A2a (idealized, complex-linear two-layer scalar chain; no inter-layer
      Re(.)): hand-rolled, self-contained, no repo dependency. This is the
      textbook single-channel E1/E2 recursion s~_u = a s~_{u-1} + s_u
      exactly as schematically written in the handoff.

  A2b (repo's *actual* convention; toyrig.ssm_rig with L=2, arbitrary N,
      the real inter-layer coupling x^{l+1}=Re(h^l)): shows the naive
      single-channel recursion (A2a's form) does NOT close the gap
      exactly once a Re(.) sits at the layer boundary, and derives +
      verifies the two-channel forward recursion (P, Q) that does close
      it exactly. P and Q use the SAME pole a_1 (resp. conj(a_1)) already
      used by the upper layer's own recurrence, filtering the SAME
      already-existing lower-layer eligibility trace Sa^0 -- no new
      within-layer machinery, only new cross-layer routing state.

Run:  python -m credit_memory.phase_a_causal_dual
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import ssm_rig as tcg

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")


# ---------------------------------------------------------------------------
# A2a: idealized complex-linear two-layer scalar chain (no Re(.))
#
#   h^1_t = a1 h^1_{t-1} + b1 x_t                  (lower layer, scalar)
#   h^2_t = a2 h^2_{t-1} +  c h^1_t                 (upper layer, scalar;
#                                                     c couples layers
#                                                     WITHOUT a Re(.))
#   yhat_t = Re(w * h^2_t)          (only the final readout is real-valued;
#                                     the inter-layer coupling itself is
#                                     complex-linear)
#
# Exact upper adjoint (top layer, Null-1 case, no cross-layer dependence):
#   lambda_t = q_t + conj(a2) lambda_{t+1},   q_t = conj(w) r_t
# Exact lower-layer "a1" gradient (BPTT):
#   G_exact = sum_t conj(lambda_t^1) h^1_{t-1},
#   lambda_t^1 = c q_t^{(routed)} ... but since coupling is complex-linear
#   (no Re), the lower layer's own adjoint recursion is driven by the
#   INSTANTANEOUS routed signal  s_t := conj(c) lambda_t^2  directly (no
#   extra Wirtinger real-part split needed):
#     lambda_t^1 = s_t + conj(a1) lambda_{t+1}^1,   s_t = conj(c) lambda_t^2
#   By the single-recursion forward/backward duality lemma (LEMMA 1 in the
#   writeup), this is EXACTLY:
#     G_exact = sum_u conj(s_u) * s~_u,   s~_u = a1 s~_{u-1} + h^1_{u-1}
#   which reindexed onto q^2 (not s) gives the handoff's schematic form:
#     G_exact = sum_u conj(q_u^2) * s~~_u,
#     s~~_u = a2 * s~~_{u-1} + c * s~_u        (E1, single complex channel)
# ---------------------------------------------------------------------------

def idealized_two_layer(seed=0, T=40, rtol=1e-11):
    rng = np.random.RandomState(seed)
    a1 = 0.6 * np.exp(1j * 0.9)     # nontrivial complex pole, layer 1
    a2 = 0.8 * np.exp(-1j * 1.7)    # nontrivial complex pole, layer 2
    b1 = 0.7 - 0.3j
    c = 0.5 + 1.1j                  # complex-linear inter-layer coupling
    w = 1.0 - 0.4j                  # complex readout

    x = rng.randn(T) + 0.0j
    h1 = np.zeros(T, complex)
    h2 = np.zeros(T, complex)
    sp1 = 0.0 + 0.0j
    sp2 = 0.0 + 0.0j
    for t in range(T):
        sp1 = a1 * sp1 + b1 * x[t]
        h1[t] = sp1
        sp2 = a2 * sp2 + c * h1[t]
        h2[t] = sp2
    yhat = (w * h2).real
    y = rng.randn(T)                # arbitrary target -> arbitrary r_t
    r = yhat - y                    # r_t = dL/dyhat_t for L = 0.5 mean(r^2)

    # exact adjoint, both layers (hand BPTT, complex-linear coupling)
    q2 = np.conj(w) * r
    lam2 = np.zeros(T, complex)
    lam2_next = 0.0 + 0.0j
    for t in range(T - 1, -1, -1):
        lam2_next = q2[t] + np.conj(a2) * lam2_next
        lam2[t] = lam2_next
    s_t = np.conj(c) * lam2          # routed instantaneous signal into L1
    lam1 = np.zeros(T, complex)
    lam1_next = 0.0 + 0.0j
    for t in range(T - 1, -1, -1):
        lam1_next = s_t[t] + np.conj(a1) * lam1_next
        lam1[t] = lam1_next

    h1_prev = np.concatenate([[0.0 + 0.0j], h1[:-1]])
    G_exact = np.sum(np.conj(lam1) * h1_prev)

    # E1: single forward causal recursion, THEN re-expand through the
    # upper layer's own pole (E2), giving a single complex channel that
    # can be contracted with q2 alone -- no reverse-time pass at all.
    Sa1 = np.zeros(T, complex)       # existing within-layer eligibility
    sa = 0.0 + 0.0j
    for t in range(T):
        Sa1[t] = h1_prev[t] + a1 * sa if t > 0 else h1_prev[t]
        sa = Sa1[t]
    stilde = np.zeros(T, complex)    # E1: routed + upper-pole filtered
    st = 0.0 + 0.0j
    for t in range(T):
        st = a2 * st + c * Sa1[t]
        stilde[t] = st
    G_E1 = np.sum(np.conj(q2) * stilde)   # E2: contract with q2, no lam

    err_abs = abs(G_E1 - G_exact)
    err_rel = err_abs / max(abs(G_exact), 1e-300)
    ok = err_rel < rtol
    print(f"[A2a idealized complex-linear] G_exact={G_exact:.10e}  "
          f"G_E1={G_E1:.10e}  rel_err={err_rel:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return dict(case="idealized_complex_linear", seed=seed, T=T,
                G_exact=[G_exact.real, G_exact.imag],
                G_E1=[G_E1.real, G_E1.imag],
                rel_err=float(err_rel), passed=bool(ok))


# ---------------------------------------------------------------------------
# A2b: repo's actual convention (toyrig.ssm_rig, L=2, x^{l+1}=Re(h^l))
# ---------------------------------------------------------------------------

def _set_small_config(L, N, T, BATCH, DELAY):
    keep = (tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY)
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY = L, N, T, BATCH, DELAY
    return keep


def _restore_config(keep):
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY = keep


def repo_two_layer(seed=0, N=3, T=14, BATCH=3, rtol=1e-11):
    keep = _set_small_config(2, N, T, BATCH, 0)
    try:
        params = tcg.init_params(seed)
        rng = np.random.RandomState(1000 + seed)
        x = rng.randn(T, BATCH)
        h, yhat = tcg.forward(params, x)
        r = rng.randn(T, BATCH)          # arbitrary residual (Phase 1
                                          # protocol: "generate arbitrary
                                          # error sequences q_t")
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        lam = tcg.exact_lambda(params, q)
        G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)
        G_exact_a0 = G_ex["a"][0]        # (N,) complex, layer-0 "a" grad

        a1 = params["a"][1]              # (N,) upper-layer poles
        B1 = params["b"][1]              # (N, N): B1[j, m], j=upper mode
        Sa0 = Sa[0]                      # (T, BATCH, N) existing eligibility
        q1 = q[1]                        # (T, BATCH, N) naive spatial error

        # -- naive single-channel attempt (A2a's form, ignoring the Re(.)
        #    boundary): stilde_u[j,m] = a1[j] stilde_{u-1}[j,m] + B1[j,m]*Sa0_u[m]
        st_naive = np.zeros((T, BATCH, N, N), np.complex128)  # [t,b,j,m]
        run = np.zeros((BATCH, N, N), np.complex128)
        for t in range(T):
            run = a1[None, :, None] * run \
                + B1[None, :, :] * Sa0[t][:, None, :]
            st_naive[t] = run
        G_naive = np.einsum("tbj,tbjm->m", np.conj(q1), st_naive)

        # -- two-channel forward recursion (P uses a1, Q uses conj(a1)),
        #    both driven by the SAME existing Sa0, no B1 inside the filter
        P = np.zeros((T, BATCH, N, N), np.complex128)   # [t,b,j,m]
        Q = np.zeros((T, BATCH, N, N), np.complex128)
        runP = np.zeros((BATCH, N, N), np.complex128)
        runQ = np.zeros((BATCH, N, N), np.complex128)
        for t in range(T):
            runP = a1[None, :, None] * runP + Sa0[t][:, None, :]
            runQ = np.conj(a1)[None, :, None] * runQ + Sa0[t][:, None, :]
            P[t] = runP
            Q[t] = runQ
        term1 = 0.5 * np.einsum("jm,tbj,tbjm->m", B1, np.conj(q1), P)
        term2 = 0.5 * np.einsum("jm,tbj,tbjm->m", np.conj(B1), q1, Q)
        G_causal_dual = term1 + term2

        err_naive = np.linalg.norm(G_naive - G_exact_a0) \
            / max(np.linalg.norm(G_exact_a0), 1e-300)
        err_dual = np.linalg.norm(G_causal_dual - G_exact_a0) \
            / max(np.linalg.norm(G_exact_a0), 1e-300)
        ok = err_dual < rtol
        print(f"[A2b repo Re(.)-coupled, N={N}] "
              f"naive single-channel rel_err={err_naive:.3e} (expected "
              f"large -- demonstrates the Re(.) boundary breaks a single "
              f"complex-scalar recursion)")
        print(f"[A2b repo Re(.)-coupled, N={N}] "
              f"two-channel (P,Q) causal-dual rel_err={err_dual:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

        # independent FD sanity check of G_exact itself, at a REAL loss
        # (matches ssm_rig.fd_gate's own bar, rel < 1e-4; looser than the
        # 1e-11 algebraic identity above by design -- FD noise floor)
        y = rng.randn(T, BATCH)
        loss0, h0, r0, x0 = tcg.loss_batch(params, rng, xy=(x, y))
        flat = tcg.flatten(params)
        eps = 1e-6
        fd_rel = []
        for idx in [1, N + 2, 2 * N + 3]:
            fp = flat.copy(); fp[idx] += eps
            fm = flat.copy(); fm[idx] -= eps
            lp = tcg.loss_batch(tcg.pack(params, fp), rng, xy=(x, y))[0]
            lm = tcg.loss_batch(tcg.pack(params, fm), rng, xy=(x, y))[0]
            fd = (lp - lm) / (2 * eps)
            q0 = tcg.spatial_q(params, h0, r0)
            Sa0_, Sb0_ = tcg.sensitivities(params, h0, x0)
            lam0 = tcg.exact_lambda(params, q0)
            G0 = tcg.assemble(params, h0, x0, r0, lam0, Sa0_, Sb0_,
                              direct=True)
            g0 = tcg.flat_grads(G0, params) / (T * BATCH)
            rel = abs(fd - g0[idx]) / max(abs(g0[idx]), 1e-12)
            fd_rel.append(float(rel))
        fd_ok = all(v < 1e-4 for v in fd_rel)
        print(f"[A2b FD sanity of BPTT reference] rel errs {fd_rel}  "
              f"{'PASS' if fd_ok else 'FAIL'}")

        return dict(case="repo_re_coupled", seed=seed, N=N, T=T,
                    BATCH=BATCH,
                    naive_single_channel_rel_err=float(err_naive),
                    causal_dual_two_channel_rel_err=float(err_dual),
                    passed=bool(ok),
                    fd_sanity_rel_errs=fd_rel, fd_sanity_passed=bool(fd_ok))
    finally:
        _restore_config(keep)


def main() -> None:
    print("=" * 78)
    print("Phase A2: causal-dual (E1/E2) machine-precision verification")
    print("=" * 78)
    rows = []
    for seed in range(5):
        rows.append(idealized_two_layer(seed=seed))
    for seed in range(5):
        rows.append(repo_two_layer(seed=seed))
    # one extra nontrivial-mode stress case: near-unit-magnitude slow pole
    rows.append(repo_two_layer(seed=99, N=5, T=30, BATCH=2))

    all_pass = all(r["passed"] for r in rows)
    fd_rows = [r for r in rows if "fd_sanity_passed" in r]
    fd_all_pass = all(r["fd_sanity_passed"] for r in fd_rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, rows=rows, all_identity_checks_passed=bool(all_pass),
              all_fd_sanity_checks_passed=bool(fd_all_pass))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "phase_a_causal_dual_summary.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print("-" * 78)
    print(f"ALL IDENTITY CHECKS PASSED: {all_pass}")
    print(f"ALL FD SANITY CHECKS PASSED: {fd_all_pass}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
