# Phase A — finite-horizon causal-dual (E1/E2) derivation

Branch `credit-memory-repair`. Analysis/theory + machine-precision
verification only. No training, no Stage 0, no new causal-learning
mechanism. Verification code: `credit_memory/phase_a_causal_dual.py`;
artifact: `results/credit_memory/phase_a_causal_dual_summary.json`.

Sequence time is `t`/`u`. There is no optimizer/meta time anywhere in
this document. `D^{-1}` is not used as a name for anything here; the
object below is called the **causal-dual cross-layer state**.

---

## A1. Derivation

### A1.0 The one tool everything reduces to (LEMMA 1)

For a complex scalar pole `a`, `|a|<1`, horizon `T`, and two arbitrary
complex sequences `{s_t}`, `{h_t}` (`h_{-1} := 0`), define:

```
backward (adjoint):   lambda_t = s_t + conj(a) lambda_{t+1},  t=T-1..0,  lambda_T := 0
forward (eligibility): e_t     = h_{t-1} + a e_{t-1},          t=0..T-1,  e_{-1}   := 0
```

Then, **exactly, for any finite horizon**:

```
sum_{t=0}^{T-1} conj(lambda_t) h_{t-1}   =   sum_{t=0}^{T-1} conj(s_t) e_t          (LEMMA 1)
```

Proof: expand the finite geometric sums `lambda_t = sum_{k=0}^{T-1-t}
conj(a)^k s_{t+k}` and `e_t = sum_{k=0}^{t-1} a^k h_{t-1-k}`, substitute
the first into the LHS, and reindex `u=t+k`; both sides reduce to `sum_u
conj(s_u) sum_k a^k h_{u-1-k}`, i.e. the RHS with `e_u`. This is exactly
the finite-horizon forward/backward duality of a first-order causal LTI
filter — it is **not** an approximation, and it holds for every `T`, not
just `T -> infinity`. `e_t` in this lemma is precisely the object the
repo already implements as the within-layer RTRL eligibility trace
(`Sa`/`Sb` in `toyrig/ssm_rig.py:115-131`, `Sa`/`Sb` in
`ssm/online_s5/scan.py:70-77`).

LEMMA 1 alone reproduces the handoff's schematic single-layer E1/E2 and
**Null 1** (`docs/PROSPECTIVE_SSM_RESEARCH_HANDOFF.md` Sec. 10): for one
independent recurrent layer, `s_t = q_t` (the layer's own instantaneous
error, no cross-layer content), so `G_exact = G_on` identically — online
RTRL is already exact for a single layer, with no correction available or
needed.

### A1.1 Two-layer stack, repo convention

The repo's stacked model (`toyrig/ssm_rig.py:88-102`, and the same
structure in `ssm/online_s5`) is, per mode:

```
h^0_t = a_0 h^0_{t-1} + b_0 x_t                     (layer 0, lower)
h^1_t = a_1 h^1_{t-1} + B_1 Re(h^0_t)                (layer 1, upper;
                                                        note the Re(.) —
                                                        NOT complex-linear)
yhat_t = Re(c . h^1_t)
```

with real loss `L`. The exact BPTT adjoint (`exact_lambda`,
`ssm_rig.py:134-147`) is, per mode, exactly two nested copies of LEMMA
1's backward recursion:

```
lambda^1_t = q^1_t + conj(a_1) lambda^1_{t+1},   q^1_t = conj(c) r_t        (top layer; exact, Null-1 case)
lambda^0_t = up^0_t + conj(a_0) lambda^0_{t+1},  up^0_t = Re(B_1^T conj(lambda^1_t))
```

`up^0_t` is the *instantaneous, same-timestep* cross-layer coupling term:
it is real-valued (it is `dL/d Re(h^0_t)`), computed at each backward step
from the layer above's **already time-resolved** `lambda^1_t` — this is
the code's actual convention (`ssm_rig.py:140-144`); the sign/conjugation
here (`conj(a)` in the recursion, `B^T` not `B^dagger`, `.real` after the
`B^T conj(lambda)` contraction) is exactly what `exact_lambda` implements
and what the finite-difference gate in `fd_gate()` certifies.

By LEMMA 1 applied to layer 0's own recursion (`s_t := up^0_t`, a
same-layer application — the layer's *own* eligibility trace is already
exact, `up^0_t` is just the driving signal):

```
G_exact^0 = sum_t conj(lambda^0_t) h^0_{t-1} = sum_t conj(up^0_t) Sa^0_t      (*)
```

**This is the first key finding**: layer 0's own within-layer eligibility
`Sa^0` (already implemented, unchanged) is *already exact* — nothing
about `Sa^0` itself needs to change. The entire cross-layer defect lives
in what gets fed into `(*)` as the driving signal: the online rule uses
the naive, purely-spatial `q^0_t` (chained through `spatial_q`, which
never touches `lambda^1`, only `q^1 = conj(c) r_t` with **no temporal
recursion at any layer above 0**), whereas the exact rule needs
`up^0_t`, which requires `lambda^1_t` — the *upper layer's own future
credit* — at the same timestep.

### A1.2 Making it forward-causal: the two-channel state (E1/E2)

`up^0_t = Re(B_1^T conj(lambda^1_t))` needs `lambda^1_t`, whose own
finite expansion is `lambda^1_t = sum_k conj(a_1)^k q^1_{t+k}` —
manifestly **non-causal** (needs the future of `q^1`). Substituting into
`(*)` and using `Re(z) = (z + conj(z))/2`:

```
G_exact^0[m] = sum_t Sa^0_t[m] * (1/2) sum_j [ B_1[j,m] conj(lambda^1_t[j]) + conj(B_1[j,m]) lambda^1_t[j] ]
```

Expanding `lambda^1_t[j] = sum_k conj(a_1[j])^k q^1_{t+k}[j]`, reindexing
`u=t+k` in both halves (exactly the LEMMA-1 reindexing, applied twice —
once per conjugate branch of the `Re(.)` split) gives, **exactly, finite
horizon, no approximation**:

```
G_exact^0[m] = (1/2) sum_j B_1[j,m]      sum_u conj(q^1_u[j]) P_u[j,m]
             + (1/2) sum_j conj(B_1[j,m]) sum_u q^1_u[j]      Q_u[j,m]        (E2)

P_u[j,m] = a_1[j]      P_{u-1}[j,m] + Sa^0_u[m],   P_{-1} := 0               (E1, channel P)
Q_u[j,m] = conj(a_1[j]) Q_{u-1}[j,m] + Sa^0_u[m],   Q_{-1} := 0               (E1, channel Q)
```

`P` and `Q` are both **purely forward, causal, finite-horizon-exact**
recursions, using the pole the *upper* layer already owns (`a_1` and
`conj(a_1)` respectively) to re-filter the *already-existing* lower-layer
eligibility `Sa^0` — no new within-layer machinery, only new cross-layer
routing state. The final contraction against `q^1` (itself purely
instantaneous/spatial, already computed at every timestep) can be
accumulated online, in the same forward pass, exactly like the existing
`Ga`/`Gb` accumulators in `assemble()` — **no reverse-time pass is needed
anywhere in (E2)**.

**Verified**: `credit_memory/phase_a_causal_dual.py::repo_two_layer`
computes `(E2)` independently of `exact_lambda` and compares it to
`toyrig.ssm_rig.assemble(..., direct=True)` (the repo's trusted BPTT
reference, itself cross-checked against finite differences at the
existing `fd_gate()` bar, rel `< 1e-4`). Result: relative error
`2e-16`–`3e-15` across 6 seeds/configs, including nontrivial complex
poles (random phase, `|a|` up to the sigmoid-bounded regime the repo
uses). A naive **single-channel** attempt (only `P`, i.e. the literal
handoff-schematic form applied naively across the `Re(.)` boundary) is
also computed for comparison and **fails by 65%–200%** — confirming the
two-channel structure is not optional once a `Re(.)` sits at the layer
boundary.

The idealized fully complex-linear two-layer chain (no `Re(.)` anywhere
except the final readout) is verified separately
(`idealized_two_layer`): there, a **single** channel suffices exactly
(`stilde_u = a_1 stilde_{u-1} + c * Sa^0_u`, contracted against `q^1`
directly), matching the handoff's schematic `E1` literally. This is the
`Re(.)`-free special case of `(E2)` — `up^0_t` degenerates to the plain
complex `s_t = conj(c) lambda^1_t` (no real-part split, `B_1 = c` scalar
enters only through the driving signal, never through a conjugate-vs-not
branch), so the two channels collapse to one. Verified to relative error
`2e-16`–`5e-15` across 5 seeds with nontrivial complex poles.

### A1.3 What's already implemented vs. what's additional

- **Already implemented, unchanged**: the within-layer eligibility trace
  `Sa`/`Sb` (`toyrig/ssm_rig.py:115-131`; `ssm/online_s5/scan.py:70-77`).
  It is exact for its own layer's own parameters given any driving signal
  (LEMMA 1) — this was never the defective part.
- **Additional state needed for the exact cross-layer term**: for every
  ordered pair of (upper-layer mode `j`, lower-layer mode `m`) with
  nonzero coupling `B_1[j,m]`, two extra forward complex scalar
  recursions `P[j,m]`, `Q[j,m]`, using the *upper* layer's own pole
  (`a_1[j]`, `conj(a_1[j])`), driven by the *lower* layer's *existing*
  `Sa^0[m]`. Per-pair cost is the same O(1)-per-step cost as `Sa` itself;
  the new cost is the `O(N_1 x N_0)` pair count, not a per-recursion
  overhead.
- **Additional routing needed**: the final accumulation `(E2)` still
  needs `q^1` (already computed every step, purely spatial) contracted
  against `P`/`Q` — a running dot-product accumulator, exactly analogous
  to the existing `Ga`/`Gb` accumulation.

---

## A2. Machine-precision verification — summary

See `results/credit_memory/phase_a_causal_dual_summary.json` for raw
numbers. All 11 rows (5 idealized-complex-linear seeds, 6 repo-convention
configs including one 5-mode stress case) pass at relative error
`< 1e-11` (actual: `2e-16`–`5e-15`); FD sanity checks of the BPTT
reference itself pass at `< 1e-4` (actual: `1e-10`–`6e-6`, matching the
existing FD noise floor used throughout the repo, e.g.
`ssm_rig.fd_gate()`).

---

## A3. Exact state-shape / scaling table

Notation: `L` layers, `N` modes/layer (repo toy: `N=16`; S5: `state_size`
`N=64`, `d_model` `H=96` channels vmapped per layer, so read `N -> H*N`
for S5 rows below), diagonal (per-mode/per-channel independent)
recurrence throughout — `a` is always a per-mode scalar, never a dense
matrix, in both the toy and S5.

| quantity | shape | memory/layer | compute/step | exact/approx | causal? |
|---|---|---|---|---|---|
| **(1) within-layer online eligibility** `Sa,Sb` (toy: `ssm_rig.py:115-131`; S5: `scan.py:70-77`) | `(N,)` toy / `(H,N)` S5, per mode | `O(N)` / `O(H*N)` | `O(N)` / `O(H*N)` | exact **for that layer's own gradient given its driving signal** (LEMMA 1) | yes — forward only |
| **(2) exact BPTT/adjoint cross-layer credit** (`exact_lambda`, `ssm_rig.py:134-147`) | `(N,)` per layer, `L` layers coupled via `B` at each backward step | `O(L*N)` total (all layers held simultaneously) | `O(L*N)` per backward step, `O(T*L*N)` total | exact | **no** — needs a full reverse-time pass over the completed sequence |
| **(3) exact causal-dual cross-layer state** (this document's `P`,`Q`, and their L>2 generalization) | `O(N_l * N_{l'} * 2^{d})` per ordered layer pair `(l',l)`, `d` = number of `Re(.)` boundaries crossed (`d=1` for adjacent layers, `d=L-1` for bottom-to-top) | `O(N^2 * 2^{L-1})` worst case, bottom-to-top | `O(N^2 * 2^{L-1})` per step, worst case | exact (verified `d=1`; `d=2` derived algebraically by the identical method, not yet numerically checked — see below) | **yes** — forward only, no reverse-time pass |
| **(4) hypothetical compressed per-mode state** (Phase B target) | `(N,)` or `(H,N)` per layer — same order as (1) | `O(N)` / `O(H*N)` | `O(N)` / `O(H*N)` | **approximate** — not yet shown sufficient | yes — forward only |

**Where mode-separability survives, and where it breaks.** Rows (1) and
(4) are mode-separable by construction (diagonal recurrence, no
cross-mode mixing in the recursion itself — `B` only mixes modes at the
*routing* step, not inside the temporal filter). Row (2) stays additive
in `L` (not multiplicative) *because it pays for a reverse-time pass* —
the backward recursion resolves each layer's adjoint using the layer
above's **already-time-resolved** value at the same timestep, so no
combinatorial blowup occurs; this is the standard reason BPTT/RTRL-style
backward passes are cheap. Row (3) is where separability is lost: the
requirement to avoid *any* reverse-time pass forces every `Re(.)` layer
boundary crossed to be represented by *both* the `a`- and `conj(a)`-
filtered branch simultaneously (Section A1.2's derivation), and these
branches compose multiplicatively across boundaries (worked by hand for
`L=3`, four channels `P2P, P2Q, Q2P, Q2Q`, matching the `2^{L-1}`
prediction; not implemented/verified numerically here — flagged as
algebraic-only pending A2-style stress testing on a real 3-layer
configuration).

**The central Phase A conclusion**: the "expensive" cross-layer
sensitivity the handoff refers to is **not intrinsically expensive** —
exact multilayer credit is cheap (`O(L*N)`, row 2) via the ordinary
backward pass. The expense is specifically the *price of causality*: an
exact *forward-only* reformulation of the same quantity costs
exponentially more state in depth (row 3), because each `Re(.)` boundary
independently needs both a same-pole and a conjugate-pole causal filter
and these compose across boundaries.

**On the parallel to the modal-geometry ceiling structure (hedged).**
The real/conjugate-pole doubling found here (`P`,`Q`) is a *temporal*
doubling — needing both `a`- and `conj(a)`-filtered copies of the same
eligibility trace. The modal-geometry program's real `2x2` representation
(`M_w`, `FINAL_MODAL_GEOMETRY_AUDIT.md` Sec. and `docs/THEORY.md`
Sec. 5) is a *spatial* generalization — a per-mode map that need not be
complex-linear. Both arise from the same root cause (a real-valued,
non-complex-linear operation — `Re(.)` — sitting between complex-linear
stages), but they are **not shown here to be the same object**, and this
document does not claim they are. It is flagged as a structural parallel
worth testing directly in Phase B, not asserted.

---

## Phase-A stopping-point report

- **Verified identity**: `(E2)` above (two-channel causal-dual
  contraction), exact to floating-point precision (`2e-16`–`5e-15`) for
  the two-layer repo-convention toy across 6 configurations, and its
  `Re(.)`-free single-channel special case (`5e-15`–`2e-16`) for the
  idealized complex-linear toy.
- **Numerical error vs. BPTT**: see A2 table above; the BPTT reference
  itself was independently FD-checked (`< 1e-4`, matching the repo's
  existing bar).
- **Exact state dimensions/scaling**: A3 table; the key result is the
  `O(N^2 * 2^{L-1})` worst-case cost of an exact *causal* (forward-only)
  cross-layer state, versus `O(L*N)` for the ordinary (non-causal, needs
  a completed-sequence reverse pass) exact adjoint.
- **Natural modal/rank factorization before approximation**: yes — the
  causal-dual state factors exactly into per-(layer-boundary) two-channel
  filters (`a`-branch, `conj(a)`-branch) composed across boundaries; this
  is an *exact* factorization (not a truncation), and it is the natural
  starting point for Phase B's compression (does a low-order truncation
  of the `2^{L-1}`-channel cascade — e.g. depth-1 only, or a rank-limited
  combination across boundaries — recover most of the known `0.596 ->
  0.901` gradient-cosine gap?).
- **Finite-horizon/conjugation subtleties that change the handoff's
  schematic equation**: (i) the handoff's schematic `s~_u = a s~_{u-1} +
  s_u` is exact only when the inter-layer coupling is complex-linear; the
  repo's actual `Re(.)` coupling requires the two-channel `(P,Q)` form
  instead, not a single recursion; (ii) all recursions here are exact
  finite-horizon sums (`lambda_T=0`, `Sa_{-1}=0`, no `T -> infinity`
  approximation); (iii) the adjoint recursion's pole is `conj(a)`, not
  `a`, matching `exact_lambda`'s `np.conj(a[...])`, and this conjugation
  flips consistently into the channel that must use `conj(a_1)` (channel
  `Q`) versus `a_1` (channel `P`) — getting this backwards is exactly
  what makes the naive single-channel attempt fail by 65%-200% in the
  verification.

**Not done in Phase A** (by design, per scope): no `L=3+` numerical
verification (algebraic-only, pattern-consistent extrapolation); no
compressor design or fitting; no training of any kind.
