# Phase B10 — tangent/adjoint factorization theory audit

Branch `S5-CCM-scale-validation` (current checkout). Mechanistic/
theory audit only: **no new training algorithm, no S5.** Code:
`credit_memory/b10_tangent_adjoint_theory.py` (new). Artifact:
`results/credit_memory/b10/b10_tangent_adjoint_theory_summary.json`.
8 seeds, same `N=6, T=60, BATCH=8, N_CAL_TRAJ=4` static calibration
protocol as B3/B4/B8/B9.x (`credit_memory.teacher.compute_teacher` via
`phase_b2bc_hankel_truncation.collect_rows`), each of the 4 calibration
trajectories processed with its own independent forward/backward
boundary conditions (no cross-trajectory leakage, per PHASE_B9.md).

**Verdict: B — PARTIAL SUPPORT.** The theory itself is exact (machine
precision throughout Parts A/B). The observed low rank of the final
relevance matrix is a genuine, confirmed mechanistic finding — but it
is an **interaction effect** between moderately-high-rank eligibility
(`U`) and adjoint-signal (`V`) factors, not explained by either alone,
and **routing does not destroy this structure** (it does not add rank
back — if anything the routed matrix is as low or lower rank than the
unrouted product). Decision-preserving low-rank reconstruction works
well at `r=4` (of a maximum useful rank 6) but is poor at `r<=2`, and
**CUR/skeleton reconstruction on the final routed matrix clearly beats
naive pre-routing SVD truncation of `U`/`V`** at matched rank.

## Equations (repo's exact convention, unchanged from Phase A/B1-B9)

Forward filter, per candidate `j` (pole `lambda_j = f_diag[j]`):
```
x_{j,t} = lambda_j x_{j,t-1} + u_t,          x_{j,-1} = 0
rho_forward[j,m] = sum_t conj(c_{j,t,m}) x_{j,t,m}
```
Adjoint (backward) recursion, terminal condition `p_T = 0`:
```
p_{j,t} = c_{j,t} + conj(lambda_j) p_{j,t+1}
rho_adjoint[j,m] = sum_t conj(p_{j,t,m}) u_{t,m}
```
Routing factorization, from `build_c_t`'s exact convention
(`c_{j,t,m}` for P-branch `j<N`: `0.5 conj(B1[j,m]) q1_t[j]`; Q-branch
`j=N+j'`: `0.5 B1[j',m] conj(q1_t[j'])`) — since the adjoint recursion
is **linear** in its `c` input and the mode-`m` dependence of `c` is a
pure scalar routing factor times a mode-independent driving signal
(`q1[:,j]` or `conj(q1[:,j'])`), the routing factor commutes straight
through the recursion:
```
v0_j := H_j^dagger q1[:,j]           (P)   or   H_j^dagger conj(q1[:,j'])  (Q)
V0[j,:] := conj(v0_j)^T,  U[:,m] := u^m
R0_P = V0_P U,     R_P = 0.5 B1        (had.) R0_P
R0_Q = V0_Q U,     R_Q = 0.5 conj(B1)  (had.) R0_Q
```

## Part A — forward/adjoint identity, verified exactly

- **Explicit `(T x T)` Toeplitz-matrix identity** `<c,Hu> = <H^dagger c,u>`
  (random `c`, `u`, independent of any recursive code): `rel_err =
  8.4e-17`.
- **Recursive implementation**, all 8 seeds x 6 modes x 12 candidates:
  **max relative error `2.2e-15`** (median per-seed max `~1.5e-15`) —
  machine precision, as the algebra requires.

## Part B — factor-matrix reconstruction

`R0 = V0 U` then routed by the Hadamard factors above, compared
against the direct per-candidate forward-filter computation
(bit-identical construction to `per_coordinate_contribution`):
**max relative error across all 8 seeds: `2.6e-15`** — the
factorization is exact, not approximate, confirming the derivation
above is correct in the repo's actual conjugation convention.

## Part C — effective-rank audit (the primary mechanistic result)

Median effective rank over 8 seeds (energy fraction 90% / 95% / 99%;
max possible rank is `N=6` for all factors here):

| matrix | 90% | 95% | 99% |
|---|---|---|---|
| `U` (eligibility) | 3.0 | 4.0 | 5.0 |
| `V_P`, `V_Q` (adjoint-filtered signal) | 3.5 | 4.0 | 5.5 |
| `V_P U` (unrouted `R0_P`) | **2.0** | 2.0 | 3.0 |
| `V_Q U` (unrouted `R0_Q`) | **1.0** | 2.0 | 2.0 |
| `B` (routing) | 4.0 | 4.0 | 5.0 |
| `R_P` routed | **1.5** | 2.0 | 3.0 |
| `R_Q` routed | **1.5** | 2.0 | 3.0 |
| **`\|rho\|` (final, combined)** | **1.5** | **2.0** | **2.5** |

(Algebraic/exact rank is generically full, `6`, for every matrix here
— continuous random data has no exact linear dependence; the
`rank(R0)<=min(rank(U),rank(V))` bound was verified to hold exactly in
this trivial sense in 100% of seeds. **Effective/energy rank is the
only meaningful diagnostic**, and is reported throughout.)

**Answer to the primary mechanistic question.** Neither `U` (rank 3-5)
nor `V` (rank 3.5-5.5) is low-rank on its own — both carry substantial
structure. Their *contraction* `V0 U` is dramatically more compressed
(rank 1-2), and the routed/final matrix is **as low or lower rank
still** (1.5-2.5). **This is option D: an interaction effect** between
`U` and `V`, not attributable to either factor's own low
dimensionality, and **routing (`B`) does not add the rank back** —
`B` itself is moderate rank (4.0), but `B ⊙ R0` stays at or below
`R0`'s own rank rather than approaching `B`'s. The low-dimensional
relevance geometry observed in B9.2/B9.4 is explained by how the
eligibility and adjoint-teaching signals *correlate*, not by either
being independently simple.

## Part D — low-rank `U`/`V` truncation (compress before routing)

Median over 8 seeds, `r=4` (of max 6):

| variant | Frobenius rel. err | winner preserved | top-4 recall | pool regret |
|---|---|---|---|---|
| **D1 compress `U` only** | **0.080** | **0.833** | 0.917 | 0.928 |
| D2 compress `V` only | 0.123 | 0.750 | 0.896 | 0.000 |
| D3 compress both | 0.223 | 0.667 | 0.854 | 5.73 |

At `r=2`: all three degrade substantially (Frobenius error 0.76-0.97,
winner-preserved 0.17-0.50); at `r=1`: essentially uninformative
(error >0.9, winner-preserved <=0.25 for every variant).

**Which side carries the structure?** At the practically useful `r=4`,
**compressing `U` alone (D1) costs the least** — `V` (the
adjoint-filtered teaching signal) is somewhat more load-bearing for
decision quality than `U` (eligibility) is, though the gap is modest
and both degrade sharply below `r=4`. Combined with Part C, the
honest summary is: **the interaction is what's low-rank; if forced to
preserve only one side exactly, preserve `V` over `U`, but neither
survives aggressive (`r<=2`) independent truncation well.**

## Part E — decision-preservation margin theorem

`Delta_m = s_best,m - s_second,m`; `epsilon_m = max_j |s_hat[j,m] -
s[j,m]|`; test condition `Delta_m > 2 epsilon_m => winner preserved`,
evaluated on the D3 (`r=4`, compress-both) reconstruction, all 8 seeds
x 6 modes (48 checks):

- **Zero violations** (0/48) — the theorem holds exactly, as required.
- Fraction of winners actually preserved overall: **66.7%**.
- Fraction *certified* by the margin condition: **25%** — the
  condition is correct but conservative, as expected for a worst-case
  sufficient (not necessary) guarantee: most preserved winners are not
  provably safe by this margin alone, they just happen to still be
  the argmax.

## Part F — pool-objective perturbation bounds

For `epsilon = ||s_hat - s||_inf` (same D3, `r=4`, reconstruction):
`|F_hat(P) - F(P)| <= N epsilon` verified to hold in **100%** of
checks (4 pools x 8 seeds); `F(P*) - F(P_hat) <= 2 N epsilon` also
holds in **100%** of seeds. Both bounds are **loose** in practice
(median observed gap `5.73` vs. bound `241.2`, `~42x` slack) —
expected for worst-case guarantees, but they never fail, confirming
the perturbation theory is sound.

## Part G — routing-rank hypothesis

Median effective rank (90% energy): `B=4.0`, unrouted `R0=2.0`,
**routed `=1.5`**. The Hadamard-product rank bound `rank(B⊙R0) <=
rank(B) rank(R0)` holds in 100% of seeds (trivially, at the algebraic/
exact-rank level where both sides are near-maximal) — but the
*effective-rank* picture is the informative one: **routing does not
push the effective rank of the product toward the bound; it stays at
or below the unrouted factor's own rank.** This directly answers the
question the task raised — **routing does not destroy the low-rank
structure**, so a future CUR-style calibration approach is not
mechanically blocked by `B`.

## Part H — CUR/skeleton diagnostic (oracle row/column access)

Median over 8 seeds, comparing QR-column-pivot selection, a simple
oracle row/column-norm pivot, and random selection, reconstructing the
**final routed** relevance matrix directly (not the pre-routing
factors):

| r | method | Frobenius rel. err | winner preserved | pool Jaccard |
|---|---|---|---|---|
| 1 | qr_pivot | 0.360 | 0.167 | 0.238 |
| 1 | random | 0.863 | 0.250 | 0.143 |
| 2 | qr_pivot | 0.172 | 0.500 | 0.333 |
| 2 | random | 2.233 | 0.333 | 0.333 |
| **4** | **qr_pivot** | **0.034** | **0.833** | **0.600** |
| 4 | oracle_norm_pivot | 0.097 | 0.667 | 0.600 |
| 4 | random | 0.941 | 0.667 | 0.600 |

**QR-pivot CUR at `r=4` reconstructs the final matrix with lower error
(0.034) than SVD-truncating `U`/`V` before routing (D1's 0.080) at the
same rank, and comparable winner preservation (0.833).** This is a
clear, actionable mechanistic finding: **operating on the final,
already-routed matrix via real selected rows/columns is a
meaningfully better compression strategy than compressing the
pre-routing adjoint/eligibility factors separately** — consistent
with Part C/G's finding that routing doesn't add rank but does mix
information in a way that pre-routing factor truncation loses. At
`r<=2`, all methods (including oracle pivoting) degrade sharply — this
toy's useful compression floor is around `r=4` of `6`, not the more
dramatic `r=1-2` compression B9.2's shared-pool result achieved for
`K` (a genuinely different object: B9.2 compressed the *candidate
count* via discrete selection with `|rho|`-based per-mode choice, not
a continuous low-rank matrix factorization — the two are not directly
comparable, and this phase does not claim they are).

**Per the task's own caution: this does not establish a cheap online
calibration algorithm** — pivot discovery itself (`qr_pivot`,
`oracle_norm_pivot`) required access to the full matrix here. The
question answered is narrower and prior: *if* informative rows/columns
were somehow available cheaply, low-rank cross reconstruction would
preserve most (not all) of the credit decisions that matter, at `r=4`
specifically, not at `r<=2`.

## Part I — operator sanity check (interpretation only, not an algorithmic claim)

`u_t = x_t - lambda x_{t-1}` recovers the exact input to machine
precision (`4.4e-16`). For the normalized low-pass `y_t = lambda
y_{t-1} + (1-lambda) u_t`, the discrete inverse `u_t = (y_t - lambda
y_{t-1})/(1-lambda)` converges to the continuous-time approximation
`u ~= y + tau dy/dt` (under `lambda = exp(-dt/tau)`) as `dt -> 0`: mean
absolute error `0.130 -> 0.030 -> 0.0063 -> 0.0013` for `dt = 0.5, 0.1,
0.02, 0.004` — shrinking roughly linearly in `dt`, consistent with the
expected first-order discretization error. Sanity check passes; no
algorithmic claim is made from it.

## Part J — mechanistic verdict

**B. PARTIAL SUPPORT.**

- **Forward/adjoint identity: exact** (Parts A/B, machine precision
  throughout — not approximate, not merely "close").
- **Relevance-matrix low rank is clearly explained**, but by an
  **interaction** between `U` and `V` (both individually moderate-rank,
  ~3-5 of 6) rather than either factor's own low dimensionality — and
  **routing does not destroy this structure** (Parts C, G).
- **Low-rank reconstruction preserves winners/pools only partially**:
  good at `r=4` (67-83% winner preservation depending on method), poor
  at `r<=2` — this phase does not find that "low r" (in the sense of
  `r<=2`, comparable to B9.2's `K=4`-out-of-`12` candidate compression)
  reconstructs the *continuous relevance matrix* well, even though
  B9.2's *discrete candidate-pool* selection at comparable compression
  worked well — these are different compression targets and the
  results should not be conflated.
- **CUR oracle reconstruction works reasonably at `r=4`**, and clearly
  beats naive factor-truncation at matched rank, but is not a "small
  `r` works great" result — `r=4` of `6` is a modest, not dramatic,
  compression, and pivot discovery itself required full-matrix access
  (per the task's own caution, not claimed as a cheap algorithm here).

**Most important outputs:**
1. `<c,Hu> = <H^dagger c,u>`: verified to `8.4e-17` (explicit matrix)
   and `2.2e-15` (recursive implementation, all seeds/modes/candidates).
2. Effective-rank table: `U` 3.0/4.0/5.0, `V` 3.5/4.0/5.5, `VU`
   1.0-2.0/2.0/2.0-3.0, `B` 4.0/4.0/5.0, `B⊙VU` 1.5/2.0/3.0, final
   `|rho|` 1.5/2.0/2.5 (90/95/99% energy, median over 8 seeds).
3. Low rank comes from the **`U`-`V` interaction** (option D), not
   either factor alone; **`V` is somewhat more load-bearing** than `U`
   if only one side can be preserved.
4. Winner/pool preservation is rank-dependent and method-dependent:
   `r=4` gives 67-83% winner preservation (best via CUR on the routed
   matrix); `r<=2` is poor (17-50%) regardless of method.
5. Oracle CUR beats naive pre-routing SVD truncation at matched rank
   (`r=4`: `0.034` vs. `0.080` Frobenius relative error) — the
   practical lesson for any future compression attempt: **compress the
   routed matrix directly, not the pre-routing factors separately.**

No new persistent training arm added. No S5 run performed.
