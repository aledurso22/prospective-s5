# Phase B35h — ProductLocal Hessian-transport viability diagnostic

Branch `S5-CCM-scale-validation`. Diagnostic only: no training, no
change to the architecture, optimizer, projection, credit budget, or
B35d's results. Code: `credit_memory/b35h_hessian_transport_diagnostic.py`
(`/tmp/b35h_diagnostic.log`). C=64, seed=11, checkpoints t=(50,150,300)
from a fresh B35e-style recorded trajectory (lr=0.02, interval=1),
fixed random unit direction v, signed etas in {±0.01,±0.03,±0.1}.

## Derivation (verified numerically before use, not just asserted)

For the linear regular-block recurrence, the established first-order
sensitivity recursion is `s_{k+1}=alg_mult(theta,s_k)+h_k`. Writing
`w_k := d(s_k)/dtheta[Delta]`, differentiating that recursion once more
via the SAME associativity/commutativity of the algebra used
throughout this whole project gives:

```
w_{k+1} = alg_mult(theta, w_k, Q, d) + 2*alg_mult(s_k, Delta, Q, d),   w_0 = 0
H_t[Delta] = M_{w_t}   (exact)
```

an O(P) companion recursion to s_t's own -- same per-factor `alg_mult`
cost, not O(P^2).

## Verification: O(P) recursion vs brute-force JVP-of-jacobian

Relative error 2.5e-16 to 6.0e-16 at every checkpoint (t=50,150,300) --
the derivation is exact, confirmed independently via `jax.jvp` of
`jax.jacobian` (forward-over-reverse autodiff), not merely asserted.

## E0/E1 scaling (Frobenius-norm comparison of full (r,P) sensitivity matrices)

| t | E0 slope (predict 1) | E1_full slope (predict 2) | E1_diag slope | frac(E0) removed, full | frac(E0) removed, diag |
|---|---|---|---|---|---|
| 50 | 1.001 | 2.001 | 1.002 | 95.71% | 22.54% |
| 150 | 1.001 | 2.001 | 1.003 | 95.43% | 12.79% |
| 300 | 1.002 | 2.015 | 1.004 | 95.57% | 34.17% |

**Both predicted scalings confirmed exactly** via log-log slope fit:
uncorrected sensitivity error is first-order (`Θ(|eta|)`, slopes
1.001-1.002); the full block-local Hessian correction gives a clean
second-order residual regime (`Θ(eta^2)`, slopes 2.001-2.015). **The
diagonal-only Hessian approximation does NOT achieve second-order
accuracy at all** -- its slope stays at ~1.0 (same order as no
correction), and it removes only 13-34% of the mismatch vs the full
correction's consistent 95%+. Cross-parameter (non-diagonal) Hessian
terms are therefore essential, not a refinement.

## State transport

`h_t^+ ≈ h_t + S_t Delta + 0.5 H_t[Delta,Delta]`, contracting `H_t[v]`
(the verified O(P) recursion) with v once more for the quadratic term.
Uncorrected state error scales as `Θ(eta^2)` (slope 2.001, the expected
residual order for a first-order/linear state approximation) and the
Hessian-corrected transported state residual scales as `Θ(eta^3)`
(slope 3.000-3.011, one order higher, exactly as expected for adding
the quadratic correction term) -- both confirmed cleanly at every
checkpoint.

## O(P) scaling of the complete Hessian representation

| C | P | persistent scalars (s_t + w_t) | combined update step time |
|---|---|---|---|
| 64 | 64 | 128 | 6.07us |
| 128 | 128 | 256 | 5.68us |
| 256 | 256 | 512 | 5.74us |

Persistent storage is exactly 2P (s_t and w_t together) at every size
tested -- confirmed O(P), not O(P^2). Per-step update time is flat
(5.7-6.1us) across a 4x range in P, consistent with these operations
being dispatch-overhead-bound at this scale (matching the earlier
B35a-3 observation) -- certainly not the O(P^2)-or-worse blowup a dense
per-parameter Hessian representation would show.

## Decision rule -- all four conditions checked explicitly

1. Uncorrected sensitivity error first-order in update size? **YES**
   (E0 slope 1.001-1.002).
2. Full Hessian correction gives a clear second-order residual regime?
   **YES** (E1_full slope 2.001-2.015).
3. Removes a substantial fraction of the frozen-current mismatch?
   **YES** (95.4-95.7%, consistent across all three checkpoints).
4. Full block-local correction materially outperforms diagonal
   correction? **YES** (diagonal stays first-order and removes only
   13-34%; full is second-order and removes 95%+).

**All four conditions pass.** Per the predeclared decision rule, this
diagnostic supports pursuing a Hessian-corrected continual learner as a
promising direction. No corrected model is trained here, per
instruction -- this is a viability check only, not a benchmark result.

## Commit hash

See the commit introducing this file.
