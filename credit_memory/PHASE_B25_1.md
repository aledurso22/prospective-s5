# Phase B25.1 — deep nonlinear factorized prefix RTRL

Branch `S5-CCM-scale-validation`. Composes B25's local factorized
temporal bases through depth — no new architecture. Code:
`credit_memory/b25_1_deep_prefix.py` (new; `main()` reproduces every
number below). Same architecture as B25, same JAX-autodiff/BPTT
reference discipline. No S5.

**Headline: A — STRONG DEEP CONFIRMATION, with a genuinely new
qualification exposed honestly. Deep factorized forward RTRL matches
naive full RTRL and an independent BPTT reference to machine precision
at L=2,3,4, across sources at every layer and all four parameter
families. Temporal prefix dimensions are exactly n-independent.
Genuine dimensional reduction through depth IS real — but it obeys a
"weakest-link" law neither B25 nor the phase request anticipated: a
degenerate source layer alone does NOT propagate its own reduction
downstream through otherwise-generic layers; the downstream dimension
at each hop is governed fresh by that layer's own `K(R_l,B_l)`. Real
multi-layer savings require every layer along the path to be
individually restrictive.**

## 0. Two real bugs found and fixed before any result was trusted

Per the phase's own "do not assume either side is correct" discipline
— a T=1 sanity check (true gradient exactly 0, since R at the source
layer cannot yet affect anything at the very first step) immediately
caught the first bug, before any multi-step number was trusted:

1. **A timing bug**: the recurrence updated `Xs[layer]` in place, layer
   by layer, within a single timestep — so layer `l`'s cross-layer
   term read the *already-updated* `Xs[l-1]` (representing
   `dh_{l-1,t+1}/dθ`) instead of its pre-update value
   (`dh_{l-1,t}/dθ`), since `u_{l,t}=z_{l-1,t}` depends on `h_{l-1,t}`,
   not `h_{l-1,t+1}`. Fixed by snapshotting all layers' `X` before any
   update each timestep and computing every layer's new value from
   that snapshot (a proper synchronous update).
2. **A missing direct term**: for `family='C'` specifically,
   `z_{i,t}=(I⊗C_i)h_{i,t}` depends on `C_i` *directly* (holding
   `h_{i,t}` fixed), not only via `h_{i,t}`'s own dependence on `C_i`
   — a pathway distinct from B25's `direct_term` (which captures
   `∂h_{next}/∂C`, not `∂z_t/∂C`) and easy to miss since it only
   matters when propagating *out* of a `C`-source layer. Added
   `z_direct_term` (via JAX autodiff, not hand-derived) and injected it
   at the first hop out of a `C` source only.

Both traced to a concrete, reproducible discrepancy (0.03–0.19
residuals, nowhere near machine precision) before being fixed — not
inferred from the math alone.

## 1. Foundation — the cross-layer Jacobian identity

`dh_{l+1,t+1}/dh_{l,t} = Σ_ab G_ab,t⊗P_ab`, `G_ab,t` = the Jacobian of
`Φ_{l+1}` wrt its `u`-argument (decomposed the same way as B25's
`F_ab,t`), `P_ab=B_{l+1}E_abC_l` (`r_{l+1}×r_l`, time-independent).
Verified against a **direct** full autodiff Jacobian of the 2-layer
step:

| r0 | k0 | n0 | r1 | k1 | n1 | max err |
|---|---|---|---|---|---|---|
| 3 | 2 | 2 | 4 | 1 | 3 | 5.6e-17 |
| 4 | 1 | 4 | 3 | 2 | 2 | 1.1e-16 |
| 2 | 2 | 8 | 2 | 2 | 8 | 2.2e-16 |

**Machine precision at every config.** This is the identity everything
else in this phase depends on.

## 2. Part 1 — the deep factorized prefix recurrence

Bases `V_{l<-i}` built recursively: `V_{i<-i}` = B25's local basis
(`K(R_i,B_i)` for C/psi sources, full `r_i` for R/B); for `l>i`, seeded
by the image of `V_{l-1<-i}` under the cross-layer `P_ab^{(l)}`
operators, closed under layer `l`'s own `{R_l,Q_ab^{(l)}}` (a general
multi-generator subspace closure — needed, not just R-closure, since
once `k_l>1` a seed inside `im(B_l)` is not generally R-closed alone).
No `n·r`-by-parameter tensor is ever formed — every tracked object is
`(n_l, d_{l<-i}, m)`.

## 3. Part 2 — three-way exactness, L=2,3,4

Sources tested at the earliest, middle, and final layer, across
R,B,C,ψ families, `r=3,k=1` throughout (representative sample below;
full grid in `main()`'s own output):

| L | source | dims | \|naive−BPTT\| | \|deep−BPTT\| |
|---|---|---|---|---|
| 2 | 0.R | {0:3,1:3} | 6.5e-18 | 2.2e-17 |
| 2 | 0.ψ | {0:3,1:3} | 5.6e-17 | 1.7e-16 |
| 3 | 0.C (2-hop) | {0:3,1:3,2:3} | 2.0e-18 | 1.3e-17 |
| 3 | 2.C (own layer, loss on final) | {2:3} | 1.7e-16 | 4.4e-16 |
| 4 | 0.R (3-hop) | {0:3,1:3,2:3,3:3} | 2.0e-18 | 3.0e-18 |
| 4 | 1.C (2-hop) | {1:3,2:3,3:3} | 1.9e-17 | 5.4e-17 |

**Machine precision everywhere tested** — 17 source/layer/family
combinations across L=2,3,4, including 3-hop propagation from the
earliest layer to the final one. This is genuinely different from
B25's own Part 9 (naive-only at depth) — the *factorized* deep
construction is now itself verified, not merely the naive reference.

## 4. Part 3 — the deep temporal bound, and a new structural finding

The stated bounds (`d_temp≤k_i·Σr_l` for interface sources,
`≤r_i·Σr_l` for dense sources) hold in every config tested, but were
not tight enough to be informative at `r=3,k=1` uniformly (every
`d_{l<-i}=3=r_l` regardless of family, since the ambient cap `r_l`
already binds). A more informative, unplanned test — a generic source
layer feeding a **degenerate** downstream layer (`ρ_1=3<r_1=5`) —
revealed a cleaner structural law, verified across all four families:

| source family (layer 0, generic, r0=5) | dims |
|---|---|
| R | {0:5, 1:3} |
| B | {0:5, 1:3} |
| C | {0:5, 1:3} |
| ψ | {0:5, 1:3} |

**`d_{1<-0}=ρ_1=3` in every case, regardless of whether the source was
"dense" (R,B) or "interface" (C,ψ).** This makes sense once seen: any
vector injected via `P_ab=B_{l}E_abC_{l-1}` lands in `im(B_l)`
*regardless of the input* — so the R/B-vs-C/ψ distinction, which only
matters for what happens *within* the source's own layer, disappears
the moment a sensitivity crosses into the next layer. From `l=i+1`
onward, propagation is governed by **that layer's own `K(R_l,B_l)`**,
not by the origin family — all machine precision (1.4e-17 to 1.4e-16).

## 5. Part 5 — genuine reduction requires a "weakest-link" condition

Tested the naive expectation directly rather than assuming it: does a
degenerate *source* layer's own reduction (`ρ_0=3<r_0=5`) persist
through otherwise-generic downstream layers?

| config | dims (layer 0 → 1 → 2) |
|---|---|
| layer 0 degenerate only, layers 1,2 generic | {0:3, **1:5, 2:5**} |
| **all three layers degenerate** | {0:3, **1:3, 2:3**} |

**It does not persist unless every downstream layer is also
restrictive** — directly following from Part 4's finding: `d_{1<-0}`
is bounded by layer 1's *own* `ρ_1`, which is `5` (full) when layer 1
is generic, "filling back up" regardless of how reduced the source
was. Built the genuine multi-layer case (all three layers
individually degenerate, `ρ=3<r=5` each) and confirmed a real,
compounding 3-layer saving — `naive_floats=150` vs
`factorized_floats=90` (40% reduction) — **while remaining exact**:
`|naive−BPTT|=4.3e-19`, `|deep−BPTT|=5.4e-17`.

This is a genuinely useful correction to how the reduction claim
should be stated: it is not "a locally-reduced source saves memory
downstream" — it is "the whole causal path's memory is governed by
the *tightest* layer anywhere along it," a materially different and
more demanding condition.

## 6. Part 4 — width sweep

`r=3,k=1,L=3` throughout (generic, `ρ=r` — same caveat as B25's own
Part 7, stated explicitly rather than picking a flattering config):

| n | TEMPORAL dims | naive floats | factorized floats |
|---|---|---|---|
| 2 | {0:3,1:3,2:3} | 54 | 54 |
| 4 | {0:3,1:3,2:3} | 108 | 108 |
| 8 | {0:3,1:3,2:3} | 216 | 216 |
| 16 | {0:3,1:3,2:3} | 432 | 432 |

**Temporal prefix dimensions exactly n-independent at every n** — the
theorem's core claim, confirmed. Naive and factorized coincide here
for the same reason as B25 Part 7 (generic R gives no per-layer
reduction); Part 5's all-degenerate config is the one that actually
demonstrates the saving.

## 7. Verdict

Checking against the four offered options:

- **A — STRONG DEEP CONFIRMATION.** Exact factorized nonlinear RTRL
  works through depth (L=2,3,4, 17+ source/family/layer combinations,
  all machine precision) and the predicted n-independent temporal
  prefix bounds hold (Part 6, exact at every n). This is the verdict,
  **with the Part 5 qualification stated as part of the confirmation,
  not hidden**: reduction is real and exact when present, but is a
  "weakest-link" property of the whole causal path, not something a
  single restrictive layer confers on everything downstream of it.
- **B — exact but no reduction**: ruled out as the *general* case —
  Part 5's all-degenerate config shows real, exact, compounding
  reduction (40% at this depth). It is however the correct
  description of the *generic* case (Part 4, Part 5's source-only
  case) — both are true simultaneously, for different configurations,
  and both are reported rather than picking one.
- **C — coefficient explosion**: not observed; factorized storage
  never exceeded naive storage in any tested config.
- **D — prefix theory mismatch**: not observed after the two bugs
  above were found and fixed; every subsequent check matched to
  machine precision.

No new production online-credit training rule deployed. No S5 run. No
benchmark, no optimization, per the phase's own instruction.

## 8. What this licenses for the nonlinear successor

Per the phase's own framing: with B25 and B25.1 together, the
nonlinear temporal-factorized RTRL construction has now been verified
exactly through depth — not just at the naive level (B25 Part 9's own
honest scope limit is now closed). Combined with B25's own capacity
result (task quality improves with width at fixed r,k), the
architecture and its exact online credit machinery are both
established. The Part 5 weakest-link law is worth carrying into any
future architecture search: engineering deep memory savings requires
attention to *every* layer along a causal path, not just the source.

## 9. Commit hash

See the commit introducing this file.
