# Phase B26 — exact feature bottleneck

Branch `S5-CCM-scale-validation`. Architecture: a q-dim PERSISTENT
recurrent state driven through a wide (n-unit) nonlinear transition
that is never itself persisted — `a_t=Us_t+Eu_t+b`, `x_{t+1}=σ(a_t)`
(ephemeral), `s_{t+1}=V^Tx_{t+1}` (the only thing carried forward).
Code: `credit_memory/b26_feature_bottleneck.py` (new; `main()`
reproduces every number below). Uses JAX (float64) for exact autodiff
Jacobians and BPTT throughout. No S5.

This report incorporates four corrections made during review, before
any verdict was finalized — each is called out explicitly in its own
section below rather than silently folded in.

**Headline: A — FULL CONFIRMATION. The exact integrated learner
(feature-state factorization + temporal prefix factorization + depth)
is single-pass and confirmed linear in n at fixed q,r,k,L through
L=1,2,3, width genuinely improves nonlinear transition capacity, and
the q-bottleneck is now established via a rigorous differentiable-
decoder argument (not a training-dependent illustration).**

## 1. Part 1 — standalone exactness

Three methods compared: naive per-step autodiff Jacobians (A),
closed-form `A_lat,t=V^TD_tU` + autodiff direct-source term (B),
BPTT (C):

| n | q | family | \|naive−BPTT\| | \|fact−BPTT\| |
|---|---|---|---|---|
| 4 | 2 | U,V,E,b | 1.4–5.6e-17 | 1.4–5.6e-17 |
| 8 | 4 | U,V,E,b | 6.9–27.8e-18 | 6.9–41.6e-18 |
| 16 | 8 | U,V,E,b | 1.4–5.6e-17 | 1.0–5.6e-17 |

**Machine precision at every (n,q,family)** — no bugs found in this
part.

## 2. Part 1b — genuine n×P realization reduction (Correction 1)

**Original gap**: the phase compared the reduced (q×m) method against
a *hypothetical* n×m baseline (what a naive implementation would cost
if it wrongly persisted `x_t`'s sensitivity), not a genuine third
method operating on the same real object. **Fixed** by implementing
the exactly-equivalent persistent-WIDE realization (the `D=0` case of
Part 5's extended architecture, verified algebraically:
`s_{t+1}=V^Tx_{t+1}=V^Tσ(UV^Tx_t+Eu_t+b)=V^Tσ(Us_t+Eu_t+b)` when
`s_t=V^Tx_t`, exactly recovering the original recurrence):
`x_{t+1}=σ(U(V^Tx_t)+Eu_t+b)`, `s_t=V^Tx_t`. Three genuine methods:
**A** (full wide RTRL, `dx_t/dθ∈R^{n×m}`), **B** (reduced, reusing
`q_factorized_rtrl` on the mathematically-equivalent `s_t`), **C**
(BPTT through the x-space rollout — an independently-formulated
reference, not the same code path as Part 1's BPTT).

Careful handling of `s_0=V^Tx_0` for family `V` (a direct dependency,
since `x_0` itself is a fixed external initial condition independent
of any parameter, but the *readout* `s_0=V^Tx_0` depends on `V`
directly) — seeded via the same `Et0`-mechanism validated in Part 5.

| n | q | family | wide−BPTT | reduced−BPTT | wide−reduced |
|---|---|---|---|---|---|
| 4 | 2 | U,V,E,b | 4.3e-19–6.9e-18 | 4.3e-19–6.9e-18 | 3.5e-19–3.5e-18 |
| 8 | 4 | U,V,E,b | 4.3e-18–5.6e-17 | 3.5e-18–5.6e-17 | 2.2e-18–2.1e-17 |
| 16 | 6 | U,V,E,b | 1.0e-17–1.1e-16 | 1.4e-17–1.1e-16 | 2.8e-17–5.6e-17 |

**Machine precision across all three methods, every family, every
config.** This is now a genuine `n×P→q×P` realization reduction,
verified exactly — not a comparison against a hypothetical baseline.

## 3. Part 2 — scaling (analytic, not wall-clock)

| n | q | params | reduced floats | naive-hypothetical floats | fwd ops | eligibility-update ops |
|---|---|---|---|---|---|---|
| 4 | 2 | 28 | 56 | 112 | 20 | 112 |
| 64 | 2 | 448 | 896 | 28672 | 320 | 1792 |
| 4 | 8 | 76 | 608 | 304 | 44 | 4864 |
| 64 | 8 | 1216 | 9728 | 77824 | 704 | 77824 |

(full grid n=4..64 × q=2,4,8 in `main()`'s own output.) At every fixed
`q`, all four quantities scale **exactly linearly in n**, confirmed
directly from the formulas. "naive-hypothetical" remains explicitly
labeled as a hypothetical mis-implementation cost (kept for scale
context alongside Part 1b's now-genuine reduction).

## 4. Part 3 — width genuinely adds transition capacity

Teacher `n=12,q=3`; approximators at fixed `q=3`, growing
`n∈{2,4,8,16}`, trained via the factorized RTRL gradients themselves:

| n | final loss |
|---|---|
| 2 | 0.0638 |
| 4 | 0.0381 |
| 8 | 0.0148 |
| 16 | 0.0099 |

**Monotonically decreasing with width** — `n` is a genuine
nonlinear-transition-capacity resource.

## 5. Part 4 — the q bottleneck: the differentiable-decoder argument (Correction 3)

**Original gap**: the first version tried to establish the q-lower-bound
via gradient-descent training failure, which does not distinguish a
genuine capacity ceiling from ordinary curve-fitting difficulty on a
short, fixed sequence (a wide nonlinear model can fit a small dataset
well regardless of true persistent-state capacity). **Fixed** by
replacing it with an exact, training-free structural argument, refined
once more for mathematical precision:

**Setup**: inject `v∈R^{q_teacher}` once at `t=0` (`u_0=v`), remove it
(`u_t=0` for `t≥1`), form `F: v↦s_T` (smooth, `R^{q_teacher}→R^q`).

**The differentiable-decoder argument** (the precise statement, not
merely "F must be a local diffeomorphism"): if a differentiable decoder
`G` exists with `G(F(v))=v` on an open set, the chain rule gives
`DG(F(v))·DF(v)=I_{q_teacher}` at every `v` in that set, forcing
`rank(DF(v))=q_teacher` there. But `DF(v)` has shape `(q,q_teacher)`,
so `rank(DF(v))≤q` always. **Hence `q<q_teacher` makes
`rank(DF(v))=q_teacher` impossible, so no such `G` can exist —
regardless of `n`.** This is a shape fact, requiring no numerics.

For `q≥q_teacher`, a numerically full-column-rank `DF(v)` shows only
that **the local differential obstruction is removed / full local rank
is achievable** — explicitly *not* a proof of global exact
recoverability (existence of `G` is a separate, stronger claim, not
tested here).

| q | n | \|J\| shape | verdict |
|---|---|---|---|
| 1,2,3 | 8, 32 | (q,4) | decoder existence ruled out by shape alone, for every n |
| 4, 6 | 8, 32 | (4,4),(6,4) | full column rank achieved — local obstruction removed |

(Full singular-value data for all 12 configs in `main()`'s own output;
`q≥q_teacher` cases checked for a nonzero smallest singular value at
every config, confirming rank is genuinely achieved, not a borderline
call.)

**Training kept only as an illustration** (least-squares readout
reconstruction MSE over held-out random `v`'s, no gradient descent
needed for this sub-step): `q=2,3` (below `q_teacher=4`) show *no*
improving trend with `n` (0.088→0.124→0.137 and 0.082→0.059→0.084 —
flat/noisy, consistent with a genuine ceiling), while `q=6` (above)
improves clearly toward near-zero (0.046→0.033→0.0057). Illustrative,
not the falsification itself — the rank argument above is.

## 6. Part 5 — causal-bottleneck falsification, and the structural V finding (Correction 2)

Extended architecture `x_{t+1}=Dx_t+σ(U(V^Tx_t)+Eu_t+b)`, loss on
`s_t=V^Tx_t` only. True gradient: BPTT through the full n-dim `x_t`.
Naive: the q-only recurrence, ignoring `D`.

| D mode | U err | V err | E err | b err |
|---|---|---|---|---|
| D=0 | 6.9e-18 | 2.8e-17 | 5.2e-18 | 5.6e-17 |
| generic D | 2.1e-02 | 3.8e-02 | 6.9e-03 | 7.9e-02 |
| D unobservable (`im(D)⊆ker(V^T)`) | 1.4e-17 | **4.9e-02** | 5.2e-18 | 5.6e-17 |

**Original framing (revised per review)**: V's nonzero error under
"unobservable" D was initially flagged as a loose end. **Corrected
interpretation, now stated as the finding itself**: current-state
unobservability (`V^TD=0` at one parameter point) is *insufficient* —
**causal sufficiency must hold on a parameter-invariant family, not
merely at one parameter point.** Perturbing `V` generally destroys
`V^TD=0` (the condition is evaluated at the *current* `V`; a tangent
direction in `V` moves off that constraint surface), so `V`'s own
tangent is exactly the direction the unobservability condition does
not protect. This is retained as a structural finding, not smoothed
into an anomaly.

**Positive control** (the exact control the finding calls for):
freezing `V` (excluding it from the compared families) gives an EXACT
positive control for the remaining families:

```
{'U': 1.39e-17, 'E': 5.20e-18, 'b': 5.55e-17}
```

**A genuine, load-bearing negative control**: fails badly for generic
`D` (0.007–0.08, real errors), restored to exactness for every family
whose tangent direction cannot perturb the unobservability condition.

## 7. Part 6 — L=1, L=2, L=3 integration (Correction 4)

**Original gap**: L=1's 9/9-family match was reported without testing
depth, despite the phase asking explicitly for L=2,3 before any final
verdict. **Fixed** by extending to a genuine multi-layer stack.
Architecture insight that simplified the extension: `u_{l,t}=z_{l-1,t}`
enters `h_l`'s update **linearly** (via `Bu_l`) in this design — unlike
B25/B25.1's nonlinear routing — so cross-layer composition needs only
a fixed linear term `Bu_l·C_{l-1}·Eh_{l-1}`, no F_ab/G_ab decomposition.
Three methods: naive full `(Σr_l+Σq_l)`-dim autodiff RTRL, reduced
(per-layer local `(Eh_l,Es_l)` recursion + the linear cross term),
BPTT.

**One real bug found and fixed before trusting any depth result**: `C`
serves double duty — the local temporal readout `z_l=C_lh_l` *and* the
cross-layer coupling variable (`u_{l+1}=z_l`) — so a `family='C'`
source has a *direct* dependency on `C_source` at the first hop
(`∂z_{source,t}/∂C_source`, holding `h_{source,t}` fixed) distinct from
its local `∂h_next/∂C` and `∂s_next/∂C` terms — the exact same lesson
from B25.1's `z_direct_term`, recurring in a new architecture. First
attempt at L=2 gave a real, large discrepancy for `layer0.C`
(`0.329`, not noise); fixed by adding the missing direct term at the
first cross-layer hop, exactly mirroring B25.1's fix.

| L | tested combinations | \|naive−BPTT\| | \|reduced−BPTT\| |
|---|---|---|---|
| 1 | 9/9 families | 1.4–1.1e-16 | 1.0–1.1e-16 |
| 2 | 18 (9 families × 2 layers) | 1.7e-18–8.9e-16 | 1.7e-18–1.8e-15 |
| 3 | 13 (earliest/middle/final layer, 2-hop propagation) | 1.1e-19–4.2e-17 | 1.1e-19–2.8e-17 |

**Machine precision at every depth, every family, including the
2-hop earliest-to-final-layer propagation at L=3.** The integration of
B26's feature-eligibility recurrence with a B25/B25.1-style temporal
chain now genuinely composes through depth — not just at L=1.

## 8. Part 7 — integrated end-to-end scaling (light)

`r=3,k=2,q=4`, swept `n=8,16,32,64`:

| n | forward_state | temporal_credit | feature_credit | total_credit |
|---|---|---|---|---|
| 8 | 7 | 96 | 352 | 448 |
| 16 | 7 | 96 | 704 | 800 |
| 32 | 7 | 96 | 1408 | 1504 |
| 64 | 7 | 96 | 2816 | 2912 |

`forward_state=r+q=7` exactly n-independent at every n; `temporal_
credit` exactly flat (96); `feature_credit`/`total_credit` scale
exactly linearly in n.

## 9. Verdict

Checking against the four offered options, now that all four
corrections have resolved cleanly rather than left as open caveats:

- **A — FULL CONFIRMATION.** The exact integrated learner (feature-state
  factorization `q×P` + temporal prefix factorization + depth) is
  single-pass — a genuine forward-only accumulating recurrence, no
  history cache, no backward sweep at any depth tested (L=1,2,3) — and
  confirmed linear in `n` at fixed `q,r,k,L` (Parts 2, 7, both from
  direct formula counts). Width improves nonlinear transition
  approximation (Part 3). Part 1b turns the storage claim into a
  genuine exact realization reduction (not a hypothetical comparison).
  Part 4's q-bottleneck is now a rigorous, training-free mathematical
  fact (the differentiable-decoder/chain-rule argument), with training
  correctly demoted to illustration only. Part 5's negative control
  works as intended, and its one apparent loose end (V's nonzero
  error) is resolved as a genuine structural finding with an exact
  positive control, not left ambiguous.
- **B (bottleneck confirmed, integration adds a new cost term)**: not
  selected — Part 6 shows the reduced method matches BPTT with no
  additional term beyond the local + linear-cross structure already
  accounted for in Part 7's scaling; no new asymptotic cost appears
  with depth.
- **C (q-bottleneck too severe)**: ruled out — nothing shows q damaging
  useful capacity beyond the structurally-necessary floor the
  decoder argument itself predicts.
- **D (exactness failure)**: ruled out — every exactness check (Parts
  1, 1b, 5's positive control, 6 at L=1/2/3) passed to machine
  precision; the two real bugs found during development (the `C`
  double-duty direct term at depth, mirroring B25.1's lesson; and an
  initial, later-corrected design flaw in Part 4's training-based
  approach before it was replaced by the rank argument) were caught
  and fixed, not left as unresolved mismatches.

No new production online-credit training rule deployed. No S5 run. No
large benchmark, no timing-based performance claims, per the phase's
own instruction.

## 10. Commit hash

See the commit introducing this file.
