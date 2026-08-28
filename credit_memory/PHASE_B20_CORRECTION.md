# B20 correction — multiplicity/temporal decomposition of the credit module

Short, focused theoretical correction, prompted by a real distinction
B20's implementation exposed but did not fully account for. Three
concrete claims below were verified numerically (cheap, small-scale,
matching this project's own established discipline) before being
reported; the fourth (exact multi-hop growth law) is reported as
open, not guessed at.

## The refinement

B19's theorem (`d_credit = dim(A P A)`) is correct and remains
verified to machine precision — but it was implicitly scoped to a
**single temporal core in isolation**: one `R`, one tangent set `P`,
one bimodule. It says nothing about what happens when a sensitivity
must cross a layer boundary into a *different* core, repeatedly. B20's
implementation forced that question, and the answer requires a second
axis B19 didn't model:

```
H = M ⊗ T   (multiplicity ⊗ temporal, as in B18/B19)
```

A forcing term generically has the form `Σ_a c_{a,t} ⊗ E_a`, with
`c_{a,t} ∈ M` (a multiplicity-side coefficient trajectory) and `E_a`
in the temporal tangent module. The minimal exact realization is then
governed by **both**:

- `d_temporal = dim(A P A)` (or the tighter `dim(A)` when `P`
  commutes — B19's own correction to B18), and
- `d_mult`, the reachable rank of the coefficient trajectory
  `c_{a,t}` across multiplicity copies —

and the right object is closer to a **minimal realization of the
coefficient-times-temporal module**, not `A P A` alone. `d_total` is
bounded by `d_mult · d_temporal` before any query quotient — matching
the refinement proposed. Below, each specific case is derived and
checked.

## 1. Routing parameter source — B20's own K-chain was itself over-realized

Verified directly: for `K_t^{(k)} = R·K_{t-1}^{(k)} + v_t[k]·I_r`
(B20's own recursion), 30 independently-driven trajectories of
`K_t^{(k)}` (r=6) span a measured rank of exactly 6 — matching
`dim(A(R))` computed via `bimodule_basis(R,[I])`, not `r²=36`. **B20's
implementation stored `r_lower` separate `r_upper × r_upper` matrices
(`r_lower · r_upper²` numbers) when the true minimal state is
`r_lower · dim(A(R_upper))` (`r_lower · r_upper` generically) — B20
was itself over-realized by a further factor of `r_upper`.** Reason:
`K_t^{(k)}` is *always* a scalar-coefficient polynomial in `R_upper`
(by construction of the recursion — the tangent direction entering it
is `I`, which trivially commutes), so it never leaves `A(R_upper)`,
exactly B19's own `E=I` finding, now shown to apply recursively at
every hop, not just the original B18 case.

**Multiplicity here stays genuinely small** (`r_lower`, i.e. the
reachable rank of the ORIGINATING layer's own local block) because a
single routing-matrix entry injects through exactly one fixed
spatial direction at the source layer — `d_mult = r_lower` by
construction, independent of upper `n`.

## 2. Shared-core parameter — the concern is confirmed, not resolved by A P A alone

For `R → R+εE_jk`, forcing `(I⊗E_jk)h_{t-1}`: **yes, exact credit
generically requires a coefficient space scaling with the source
layer's own multiplicity `n_l`**, even though `E_jk ∈ End(T)` has
`dim(A P A)=r²` independent of `n`. Reason, stated precisely: `d_mult`
here is the multiplicity-rank of `h_{t-1}` itself (the network's own
forward state across copies), and `E_jk` is applied *identically to
every copy simultaneously* — it cannot reduce that rank, only
transform within it. B17's own structural-rank findings already
established this rank is generically full (`=min(n_l,M_l)`) once a
layer's own input width is non-trivial — this phase's correction just
names why that matters for *credit propagation specifically*, not
only for forward expressivity.

**Exact sufficient condition to avoid the n-scaling**: `h_{t-1}`'s own
multiplicity-rank must stay bounded by some `ρ<<n_l` as it evolves.
Since `E_jk` cannot reduce rank, this is a property of what FEEDS
layer `l` — recursively, of every layer below it and the routing
connecting them. **A query quotient does not help** (B19 Part C:
restricting the query alone, without a matching source restriction,
does not shrink the module). **Structured routing plausibly does**:
Kronecker-form `B=M⊗I_r` (or `ΣM_k⊗Q_k`, small `q`) maps the
temporal factor through `I_r` (or a small `q`-dimensional family),
which cannot inflate multiplicity-rank the way a generic dense `B`
can — this is the load-bearing, currently-unverified hypothesis for B21.

## 3. Top-layer routing eligibility — confirmed compressible, B20 just didn't do it

Verified directly: for a top-layer's own `B_L[i,m]`, the local
sensitivity trajectory measured **exactly zero** norm outside copy
`q0=i//r_L`'s block (0.00e+00, all 15 timesteps) and the entire signal
(norm 163) inside it. **Reason, exactly**: `A_L=I_n⊗R_L` never mixes
copies, and `B_L[i,m]`'s injection point (`i`, hence `q0`) never
changes — so the sensitivity trivially never leaves that one block.
**B20's naive local storage (`O(N_L)` per parameter) should be
`O(r_L)` per parameter — a free reduction B20 simply left on the
table, not a hard problem.** This applies to `B`'s own parameters at
any layer that is itself the endpoint of the propagation (no further
hops needed); it does NOT extend to `R`'s own parameters at the same
layer (Case 2 — `R`'s forcing hits every copy at once, not one).

## 4. Multi-hop depth — genuinely super-additive, exact law not yet pinned down

Direct 3-layer test (`r0=3,r1=4,r2=5`, one `B0` source propagated
through two layer boundaries): measured minimal rank of the resulting
top-layer sensitivity trajectory = **10**. Compare: additive guess
(`~r0`) = 3 (far too small — ruled out); full multiplicative product
(`r0·r1·r2`) = 60 (far too large — also ruled out). **The true law is
somewhere between "stays flat" and "multiplies fully across every
layer" — composition is confirmed genuinely super-additive (NOT the
hoped-for flat `O(r)` bound), but this single measurement does not
pin down whether it is `r0·d1` capped by something, a different
combination, or requires per-case derivation.** Reported as open, not
guessed at, per the instruction to be adversarial rather than
optimistic here. **This is the single most consequential unresolved
question** — it directly determines whether deep (`L≥3`) exact online
credit can ever be `O(poly(L,r))` independent of width, or whether it
is doomed to something closer to `r^L` under generic routing.

## 5. Revised theorem

Distinguishing the four objects explicitly, as requested:

- **(a) Temporal operator-module dimension**: `dim(A P A)` (B19,
  unchanged, verified) for a tangent set `P` acting on a SINGLE core
  `R` — tight when `P` is the full local parameter tangent (Case 2);
  reduces to `dim(A) ≤ r` when `P` is a commuting/simple injection
  (`E=I`, Cases 1/3, B19's own correction, now shown to recurse).
- **(b) Multiplicity/source coefficient dimension**: the reachable
  rank, across copies, of whatever is actually driving the forcing —
  `r_lower` for a fixed-direction routing source (small, structural);
  the full forward-state rank (generically `O(n)`) for a layer's own
  core parameters (Case 2); and, across multiple hops, **composes with
  each intermediate layer's own `dim(A(R_l))` in a way confirmed
  super-additive but not yet exactly characterized** (Case 4).
- **(c) Parameter-count / gradient-output dimension**: `r_l²` per
  layer for a dense core's own entries — a LOCAL, unavoidable cost
  (B19's own finding), orthogonal to (a)/(b): it is what the optimizer
  must touch per step, not a propagation cost.
- **(d) Query-observable quotient**: shrinks (a)+(b) only when the
  READOUT is aligned with an actual invariant subspace of the
  propagating dynamics (B19 Part C) — a generic/rich readout does not
  help, and cannot substitute for source-side or routing-side
  structure.

**The correct invariant object, per this correction, is not `A P A`
alone but a minimal realization of the composed coefficient-tensor-
temporal module across the actual multi-layer path a sensitivity
takes** — `A P A` is the right answer for one core in isolation
(unchanged, still exact), and the multiplicity/composition axis above
is the necessary extension for the deep, multi-layer setting B20 is
actually built in.

## Answers to the six explicit questions

1. **Falsify or refine?** Refine. `A P A` remains exactly correct and
   machine-precision-verified for a single core; it was never scoped
   to multi-layer composition, and this correction supplies that
   missing scope rather than contradicting the original result.
2. **Minimal exact state for a shared-core R parameter?** `r²` for the
   LOCAL (within-layer) gradient — confirmed still correct, unavoidable.
   Its FORWARD PROPAGATION to a readout above the layer is NOT
   `r²`-bounded in general — it inherits the forward state's own
   multiplicity-rank, generically `O(N_upper)`, confirmed not
   `O(r_upper)` by direct argument (Case 2).
3. **Can top-layer/local B eligibility go from N to r?** Yes, exactly
   — verified directly, zero sensitivity outside the injection copy's
   block at every timestep tested. A straightforward implementation
   fix, not an open research question.
4. **Exact L≥3 composition law?** Not yet known exactly. Confirmed
   super-additive (neither flat nor fully multiplicative across all
   layers) by direct measurement; the precise formula is the top
   priority open question, not resolved here per the "stay focused"
   instruction.
5. **What scaling claim survives for the complete deep
   parameter-to-query realization?** Only a qualified one: local
   core-parameter gradients (`O(r²)` per layer) and single-hop
   routing-source propagation (`O(r_lower · r_upper)`, corrected down
   from B20's own `O(r_lower · r_upper²)`) are solid and
   multiplicity-independent. Multi-hop routing-source propagation and
   any cross-layer propagation of a layer's own core-parameter
   gradient are NOT shown to be multiplicity-independent — the former
   is confirmed super-additive with an unknown exact law, the latter
   is confirmed to inherit full forward-state rank in general.
6. **B21 priority?** **Algebraic derivation of the exact multi-hop
   composition law FIRST, specifically testing whether structured
   (Kronecker-form) routing prevents the multiplicative blowup —
   before vectorization or broader L≥3 engineering.** Vectorizing an
   algorithm whose core multi-layer viability is still open would
   optimize the wrong thing; L≥3 engineering without the composition
   law risks re-deriving (or re-missing, as B20 twice did before
   verification caught it) the same over-realization pattern found
   three times now (B18→B19 for the single-core case, and twice more
   in this correction, for K-chain storage and top-layer B) at a
   fourth site. Also apply the two confirmed-cheap fixes from Cases 1
   and 3 (K-chain storage `r_lower·r_upper` not `r_lower·r_upper²`;
   top-layer B state `O(r)` not `O(N)`) regardless of what the
   composition-law derivation finds, since both are free.

No new training implementation in this correction. No S5.
