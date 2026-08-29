# Phase B27 — noncommutative temporal advantage falsification

Branch `S5-CCM-scale-validation`. Tests whether B25/B25.1's surviving
theorem buys any REPRESENTATIONAL advantage over diagonal/RTU-style
exact-RTRL recurrence — not another "linear in n" credit-cost check
(retracted from novelty claims in the B26 audit). Code:
`credit_memory/b27_noncommutative_advantage.py` (new; `main()`
reproduces every number below, including the final corrected
protocol). No S5. No wall-clock claims (JIT is used purely for
practical runtime, never cited as a result).

This report went through two rounds of review corrections before any
verdict was drawn — both are documented in full below because they
materially changed the conclusion, and hiding that history would
misrepresent how the result was actually obtained.

**Headline: D — STILL CONFOUNDED. Parts 1–3 (the noncommutative
teacher's construction, its genuine use of multiple noncommuting
generators, and our exact online credit on it) are solid and verified
to machine precision. The representational comparison itself is not
yet resolved: after fixing every identified confound (a random-and-
unobserved teacher initial state, an optimization-budget mismatch
between architectures, a readout bottleneck, cost-normalization
errors), a real gap between our architecture and the diagonal
baseline persists on the noncommutative teacher — but the SAME gap
also persists, undiminished, on a teacher whose temporal generators
are verified EXACTLY commuting. Per the phase's own stated criterion
("if [the gap] does not [close on the commuting control], the
experiment is confounded"), this rules out attributing the observed
gap specifically to noncommutativity, and the phase stops here rather
than forcing a positive verdict.**

## 1. Parts 1–3 — solid, unaffected by the corrections below

**Part 1** (teacher construction): `r∈{3,4,5}, k∈{1,2}`, all achieve
`d_T=r²` exactly with nonzero `[R,Q_ab]` and `[Q_ab,Q_cd]` commutators
— genuinely noncommutative at every config tested.

**Part 2** (teacher usage verification, `r=4,k=2,n=6`): all 4/4
`(a,b)` generator pairs active (`active_frac=1.00`), real temporal
variation in the dominant generator's coefficient (`cv=0.053`), and
ablating the most active generator changes the trajectory by `0.98`
— roughly twice the state's own std (`0.48`), a materially large
effect, not noise.

**Part 3** (our exact credit on this teacher): naive RTRL, factorized
RTRL, and BPTT agree to machine precision (`4.4e-16` to `1.8e-15`) for
every parameter family, re-confirmed post-hoc on the final
BPTT-trained model (`8.9e-16` to `2.0e-14`) as the corrected protocol
below required.

**Part 6** (depth exactness, cheap reuse of B25.1): re-verified at
L=2 on this specific noncommutative teacher, machine precision
(`3.3e-16` to `1.3e-15`). This is exactness evidence only — per
explicit instruction, it does **not** resolve the representational
question below, and depth was correctly paused pending that
resolution.

## 2. First round of corrections (methodology)

The first representational comparison (single fixed trajectory,
diagonal baseline with a `k=2`-bottlenecked readout, no cost
normalization) suggested a clean, large separation. Review caught
three real issues before that result was trusted:

1. **Rank-1 ≠ commuting**: the first "commuting" control only forced
   `Φ`'s Jacobian to rank 1, which does not imply `[R,Q*]=0` — a
   single active generator can still fail to commute with `R`. Fixed
   by constructing a **true** commuting teacher: `R` diagonal
   (distinct real eigenvalues), every interface channel's `B`-column
   and `C`-row aligned to the *same* state coordinate, so every
   `Q_ab` is itself diagonal. Verified: `max‖[R,Q_ab]‖=0.0`,
   `max‖[Q_ab,Q_cd]‖=0.0` exactly (not merely small), `d_T=r=4` (not
   `r²`) — genuinely abelian by construction, not inferred from rank.
2. **Single-sequence overfitting risk**: fixed by training/evaluating
   on batches of independently-sampled input sequences, not one fixed
   trajectory.
3. **Readout bottleneck**: the diagonal baseline's original `C_diag`
   projection to `k=2` before its nonlinear head was replaced with a
   full-state stateless head `y_t=MLP([h_{t+1},u_t])` seeing the
   entire recurrent state — tested under both a parameter-matched and
   a deliberately over-provisioned regime.

After these fixes, "ours" appeared to do *worse* than the diagonal
baseline on both teachers — which turned out to be a **second,
more serious confound**, described next.

## 3. Second round of corrections (two critical bugs)

1. **Random, unobserved teacher initial state** (critical): each
   training/test sequence had drawn a fresh random `h0_teacher`,
   unseen by either student. The target was therefore
   `y=F(U_seq,h0_teacher)` while students were asked to predict `y`
   from `U_seq` alone — not a deterministic function of the student's
   own input, creating an irreducible test-error floor capable of
   fully explaining an apparent plateau on its own. **Fixed**: `h0=0`
   for every sequence (both teacher and students), making the task a
   genuine deterministic function of `U_seq`.
2. **Optimization-budget mismatch**: the diagonal baseline (fast,
   JIT-compiled) trained for 800 steps × 20 sequences; "ours" (via
   B25's non-JIT-compatible per-step factorized-RTRL machinery) could
   only afford 150 steps × 4 sequences in comparable wall-clock time
   — a genuine confound working *against* "ours," not for it.
   **Fixed**: since factorized RTRL has been repeatedly verified
   exact against BPTT, "ours" was retrained via ordinary JIT-compiled
   BPTT+Adam (`train_ours_bptt_adam`) — legitimate because this
   subtest asks what each *model class* can represent, not how fast
   the online algorithm trains — with a **post-hoc re-verification**
   that factorized RTRL still equals BPTT on the final trained model
   (machine precision, confirmed above). Both architectures now train
   under an identical optimizer, step count, and sequence set.

## 4. Main comparison (noncommutative teacher), after both correction rounds

`r=4,k=2,n=4` (total recurrent scalar state `n·r=16`, `params=344`,
`credit floats=5504`), 800 steps BPTT+Adam, `NMSE=MSE/Var(y_train)`:

| model | total state | params | credit floats | test NMSE |
|---|---|---|---|---|
| **ours** | 16 | 344 | 5504 | **0.0048** |
| diag matched, r_diag=4..128 | 4–128 | 357–781 | 24–768 | 0.29–0.51 |
| diag strong (hidden=64), r_diag=4..128 | 4–128 | 525–8833 | 24–768 | 0.30–0.48 |

**A real, persistent gap**: diagonal test NMSE plateaus tightly
around 0.29–0.51 across the *entire* range of state size (4 to 128)
and parameter count (357 to 8833 — up to 25× "ours" own count), with
clear overfitting signatures (train MSE keeps dropping toward 0 while
test MSE does not). "Ours" achieves NMSE 60–100× lower with far less
state and far fewer parameters.

## 5. Control A (true commuting teacher) — the gap does NOT close

Per §2.1, verified exactly abelian (`d_T=4=r`, zero commutators).
Same protocol:

| model | best test NMSE |
|---|---|
| **ours** | **0.0081** |
| diag (complex/2×2 blocks), matched, r_diag=4..64 | 0.109–0.199 |
| diag, real-diagonal (**structurally matched** to the teacher's real eigenvalues), matched, r_diag=4..32 | 0.101–0.111 |
| diag, real-diagonal, strong (hidden=64), r_diag=4..32 | 0.108–0.178 |

Anticipating that the standard complex/2×2-block RTU parameterization
might be mismatched to a *real*-eigenvalue teacher (it can only reach
a real eigenvalue at the hard-to-optimize `θ=0/π` boundary of its own
angle parameterization), a **second, structurally-matched real-diagonal
baseline** was built specifically for this control (independent real
eigenvalues, no rotation) — the most direct, most favorable possible
comparison for a genuinely commuting, real-generator teacher. It does
not close the gap either: best NMSE `≈0.10`, still roughly 12× worse
than "ours."

**Sanity check** (rules out a basic bug in the diagonal training
pipeline itself, not just the teacher): the identical real-diagonal
training code fits a trivial scalar AR(1) target to `NMSE=0.0006` —
essentially perfect. The pipeline works; the commuting-teacher result
is not an artifact of broken training.

**Conclusion, stated as the phase's own criterion requires**: the
commuting control does not remove or reduce the gap. This means the
gap observed on the noncommutative teacher cannot yet be attributed
specifically to noncommutativity — some other factor (most plausibly:
where the nonlinearity sits — inside vs. outside the recurrence —
and its effect on optimization landscape or expressivity, independent
of commutativity) may be driving both results. This is exactly the
falsification the phase asked for, and it did its job: it caught a
confound the corrected single-condition comparison alone would not
have revealed.

## 6. An identified limitation of the commuting-teacher construction

The single-coordinate-alignment used to guarantee exact commutativity
has a side effect worth flagging honestly: since every interface
channel's `B`-column and `C`-row touch the *same* single state
coordinate, the `k=2`-dimensional readout `z_t` becomes rank-1 across
its own two channels at every timestep (both channels are scalar
multiples of the same single coordinate's value) — collapsing the
nonlinear MLP `Φ`'s effective input to `n` independent values instead
of `n·k`, and the teacher's causally-relevant temporal order to 1
rather than the nominal `r=4` (the other `r-1` coordinates are
dynamically present but never read or written). This makes the
commuting teacher's *true* underlying complexity considerably simpler
than the noncommutative one — which, if anything, should have made it
*easier* to fit with a small diagonal model, not harder. That it
remains hard is itself informative (ruling out "the commuting teacher
was accidentally too easy to be a fair comparison" as an explanation
for the gap persisting) but also means this specific commuting
construction is an imperfect control — a cleaner one (e.g., a
shared-eigenbasis construction with a full-rank `C` reading multiple
coordinates, preserving the teacher's full nonlinear input richness
while remaining exactly abelian) is a clear, concrete next step,
not yet built here.

## 7. Verdict

**D — STILL CONFOUNDED**, per the phase's own explicit stopping rule:
*"do not proceed with further depth-capacity experiments until this
single-layer test is clean"* and *"if [the gap] does not [close on
the commuting control], the experiment is confounded."*

- **A (clean separation)**: not selected — requires the commuting
  control to remove or strongly reduce the gap; it does not.
- **B (weak separation)**: not selected for the same reason — a
  surviving gap under a working commuting control is the definition
  of weak/clean separation; here the control itself is inconclusive
  about what's driving the gap.
- **C (no separation)**: not selected — a substantial, robust gap
  does exist on the noncommutative teacher and survives every
  correction; it simply hasn't been isolated to noncommutativity
  specifically.
- **D (confounded)** — selected. Two genuine, sequentially-discovered
  confounds (rank-1-vs-commuting, then the unobserved-h0/optimization-
  budget pair) were found and fixed, and the resulting clean-protocol
  comparison still cannot attribute the gap to the intended cause,
  because the control condition built to isolate that cause does not
  behave as the falsification logic requires.

Per the phase's own explicit instruction, **depth (L=2/L=3
representational tests) is not attempted** — the L=2 result in
§1/Part 6 is exactness evidence only, kept because it was cheap and
already available, not as progress on the open question.

## 8. What remains open, explicitly

1. A cleaner commuting control that does not collapse the teacher's
   effective input dimensionality (shared-eigenbasis construction
   with a full-rank interface) — the most direct next step.
2. Whether the observed gap is actually about *where the nonlinearity
   sits* (inside vs. outside the recurrence) rather than temporal
   noncommutativity per se — worth testing directly with a matched
   pair of architectures that differ ONLY in nonlinearity placement,
   holding commutativity fixed.
3. Multi-seed robustness (teacher seeds, student init seeds, dataset
   draws) with median/spread reporting — not done here given the
   scope already required to resolve the two confound rounds above;
   an explicit scope limit, not an oversight.
4. Diagonal persistent-credit float accounting was corrected in code
   (a 2×2 rotation block gives a genuinely 2-dimensional sensitivity
   trace per parameter, not one scalar — `2·r_diag·(1+u_dim)`, not the
   original `2·n_blocks+r_diag·u_dim`) — the asymptotic `O(r_diag)`
   law is unchanged, only the constant factor.

No new production online-credit training rule deployed. No S5 run. No
wall-clock performance claims — JIT compilation was used purely to
make the corrected multi-seed, common-optimizer protocol tractable,
never cited as a result.

## 9. Commit hash

See the commit introducing this file.
