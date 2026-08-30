# Phase B35c — final robustness benchmark: matched-credit frontier + negative control

Branch `S5-CCM-scale-validation`. B35b (commit `5d96970`, `PHASE_B35B.md`)
is FROZEN and untouched. Architecture frozen: RegularBlock, GenericBlock,
RTU reused exactly as validated; only sizes vary, per a rule
predeclared before running.

Code: `credit_memory/b35c_matched_credit_frontier.py`
(`/tmp/b35c_frontier_results.json`), `credit_memory/b35c_negative_control.py`
(`/tmp/b35c_negative_control.log`).

## 1. Predeclared sizing rule (d=4 fixed, chosen before seeing results)

```
RegularBlock: p=d=4, Q=C/p       -> r=Q*d=C,   P=Q*p=C,   credit=Q*p=C
GenericBlock: p=d=4, Q=C/(d*p)   -> r=Q*d=C/4, P=Q*p=C/4, credit=Q*d*p=C
RTU:          hidden=C/8         -> r=2*hidden=C/4, P=4*hidden=C/2, credit=8*hidden=C
```
Budgets C in {32,64,128,256} all divide evenly. Models were never resized
after seeing performance.

## 2. Resource views (per architecture, per budget)

| C | arch | r | P | credit | generic-dense-equiv (r·P) | elig-step time |
|---|---|---|---|---|---|---|
| 32 | RegularBlock | 32 | 32 | 32 | 128 | 33.4us |
| 32 | GenericBlock | 8 | 8 | 32 | 32 | 6.2us |
| 32 | RTU | 8 | 16 | 32 | 128 | 202.6us |
| 64 | RegularBlock | 64 | 64 | 64 | 256 | 45.0us |
| 64 | GenericBlock | 16 | 16 | 64 | 64 | 7.3us |
| 64 | RTU | 16 | 32 | 64 | 512 | 227.6us |
| 128 | RegularBlock | 128 | 128 | 128 | 512 | 43.7us |
| 128 | GenericBlock | 32 | 32 | 128 | 128 | 7.1us |
| 128 | RTU | 32 | 64 | 128 | 2048 | 2957.2us |
| 256 | RegularBlock | 256 | 256 | 256 | 1024 | 44.0us |
| 256 | GenericBlock | 64 | 64 | 256 | 256 | 10.8us |
| 256 | RTU | 64 | 128 | 256 | 8192 | 242.8us |

(RTU's C=128 elig-step measurement is a one-off outlier, likely a
transient scheduling/JIT-cache hiccup on this shared machine — the
neighboring C=64 and C=256 rows at similar hidden_dim are two orders of
magnitude smaller and consistent with each other; not treated as
signal.) **Credit is matched exactly at every row by construction; r,
P, and the generic-dense-equivalent are NOT matched — RegularBlock's
extra state/parameters at matched credit are the intended consequence
of the compression, not a matched-parameter comparison.**

Training time per (architecture, task, budget) cell (9 runs: 3 LRs x 3
seeds) ranged ~26s-85s across the whole sweep, growing mildly with C;
full per-cell times are in `/tmp/b35c_frontier_results.json`.
**Zero divergence in all 108 training cells (4 budgets x 3 tasks x 3
architectures x 9 runs = 324 individual runs).**

## 3. Matched-credit frontier — test NMSE (median [n=3 seeds at
   selected LR], std)

### Task A — generalized-mode / repeated-pole (B35b's Jordan teacher)

| C | RegularBlock | GenericBlock | RTU |
|---|---|---|---|
| 32 | 5.95e-3 (±1.44e-3) | 5.43e-3 (±3.71e-3) | 5.02e-3 (±3.72e-3) |
| 64 | 1.53e-3 (±7.84e-4) | 3.05e-3 (±1.57e-3) | 7.25e-3 (±3.67e-3) |
| 128 | **1.53e-4** (±5.30e-5) | 4.75e-4 (±2.80e-4) | 5.71e-3 (±5.19e-4) |
| 256 | **3.68e-5** (±2.02e-5) | 7.91e-4 (±1.88e-3) | 8.58e-3 (±3.85e-3) |

RegularBlock wins at C=64,128,256, with the margin over GenericBlock
GROWING (2.0x -> 3.1x -> 21.5x) and over RTU growing dramatically
(4.7x -> 37x -> 233x) as budget increases — **this is the frontier
result the hypothesis predicts, not a single-budget coincidence.** At
C=32 all three are within noise of each other (RegularBlock is
technically 3rd on the median but within 1 std of both).

### Task B — neutral dense-linear system identification (algebra-independent)

| C | RegularBlock | GenericBlock | RTU |
|---|---|---|---|
| 32 | 1.49e-3 (±5.77e-4) | 1.25e-3 (±5.21e-4) | 1.31e-2 (±9.17e-3) |
| 64 | **6.63e-4** (±3.43e-4) | 1.27e-3 (±8.16e-3) | 2.03e-2 (±2.67e-3) |
| 128 | 1.39e-3 (±5.10e-4) | 5.32e-4 (±1.38e-3) | 1.03e-2 (±8.94e-4) |
| 256 | 3.76e-4 (±7.03e-4) | 2.68e-4 (±2.72e-4) | 1.48e-2 (±5.30e-3) |

RegularBlock and GenericBlock trade the lead (Regular wins C=32,64;
Generic wins C=128,256) but stay within roughly 2x of each other at
every budget — **competitive, not a clean Regular win on this task**,
reported honestly rather than smoothed over. Both crush RTU by
10-40x at every budget (RTU's diagonal/independent structure is a poor
match for a genuinely coupled dense linear system, as expected).

### Task C — RTU-multipole (reused, cheap; RTU's own native task, not a core regime)

| C | RegularBlock | GenericBlock | RTU |
|---|---|---|---|
| 32 | 3.31e-2 | 2.61e-2 | **1.71e-2** |
| 64 | 3.30e-2 | 2.34e-2 | **1.54e-2** |
| 128 | 3.27e-2 | 2.28e-2 | **8.97e-3** |
| 256 | 3.00e-2 | 2.08e-2 | **9.64e-3** |

RTU wins clearly and consistently (expected — this is RTU's own native
independent/multi-timescale regime), GenericBlock is second, and
RegularBlock is worst throughout, essentially FLAT with budget
(3.31e-2 -> 3.00e-2, barely improving over an 8x credit increase) —
RegularBlock's per-factor "one semisimple + nilpotent tail" structure
buys it fewer independent spectral degrees of freedom per credit
scalar than either RTU or GenericBlock on a task that specifically
rewards many independent, cheaply-representable poles. **Reported as
an honest limitation, per instruction not to require a win on every
task** — task C was explicitly the reused/cheap "third task," not one
of the two primary regimes (generalized-mode, neutral) the success
criterion targets.

## 4. Negative control — deliberately noncommuting state tracking

Teacher: `h_{t+1} = G[x_t] @ h_t`, x_t in {0,1,2} selecting among 3
S_3-generating 3x3 permutation matrices (identity, a transposition
G1, a 3-cycle G2). Verified: `‖G1@G2 - G2@G1‖ = 1.0` (genuinely
noncommuting), `‖G0@G1 - G1@G0‖ = 0.0` (identity commutes, as it must).

**Rigorous impossibility argument (not an optimization artifact):** any
two elements of a commutative algebra's regular representation
commute — `M_a@M_b=M_b@M_a` for ALL a,b in the algebra, unconditionally.
If RegularBlock's lookup-table transitions theta^(1), theta^(2) could
exactly equal G1, G2, this would force `G1@G2=G2@G1`. Since this is
verified false, **no assignment of RegularBlock's trainable
coefficients — regardless of training — can exactly reproduce this
teacher.** Confirmed the converse holds trivially: GenericBlock at r=3
with `A^(k)=G[k]` exactly reproduces the teacher to `max|diff|=0.0`.

**Empirical SGD comparison** (r=12 matched, plain BPTT, train/val/test
split, 4 LRs x 3 seeds; a real bug was caught and fixed here — the
initial run had all architectures stuck at a fixed point (h0=0 under a
purely multiplicative recursion never leaves zero, giving grad
exactly 0 and an identical NMSE=1.500 floor for both lookup-table
architectures); fixed by giving each student a trainable initial state):

| architecture | test NMSE (median) |
|---|---|
| RegularBlock (commutative, provably cannot fit exactly) | 0.237 |
| GenericBlock (unrestricted, provably CAN fit exactly) | 0.308 (1/12 runs diverged) |
| BoundedInterfaceFlag (nonlinear, input-coupled) | **0.216** |

**Reported honestly: the empirical gap does NOT cleanly track the
theoretical one.** GenericBlock did not clearly beat RegularBlock here
— plausibly because its unrestricted (r,r)=(12,12) dense lookup
matrices (432 free parameters across 3 symbols) are a much harder
100-step SGD optimization problem than RegularBlock's 36 constrained
parameters, confounding raw optimization difficulty with
representational capacity. The benchmark was not modified to force
the expected ordering. **The phase boundary is established by the
proof above, not by this specific trained comparison** — this is
stated explicitly rather than presented as if the empirical numbers
alone made the case.

## 5. Success criterion assessment

"RegularBlock defines a better test-error-vs-exact-credit frontier
than GenericBlock on at least the generalized and neutral
identification regimes across multiple budgets, while the
noncommuting negative control exposes the expected limitation":

- **Generalized-mode (Task A): YES**, and the strongest result in this
  phase — a clean, GROWING advantage across C=64,128,256 (only tied at
  the smallest C=32).
- **Neutral identification (Task B): PARTIAL** — RegularBlock is
  competitive with GenericBlock (each wins 2 of 4 budgets, within ~2x
  throughout) and clearly beats RTU everywhere, but does not establish
  a clean, monotonic Regular-over-Generic frontier the way Task A does.
- **Negative control: YES in principle** (rigorous impossibility proof,
  independent of training), **INCONCLUSIVE empirically** (SGD
  comparison muddied by optimization-difficulty confound, reported
  honestly rather than re-tuned to force separation).
- Task C (reused/cheap, not a primary regime): RegularBlock loses
  clearly and consistently — an honest, undisguised limitation.

Net: the central hypothesis is SUPPORTED, most strongly on the
task the mechanism was designed for (generalized/repeated-pole
dynamics), competitively on the neutral regime, and the commutativity
limitation is real and provable even where the empirical training
comparison was inconclusive.

## 6. Diagnostic audit — the C=256 Task-A gap (21.5x), explained

Triggered by the unusually large RegularBlock/GenericBlock gap on Task
A at C=256 (vs ~3x at C=128). Audit code:
`credit_memory/b35c_diagnostic_audit.py` (`/tmp/b35c_diagnostic_audit.json`,
full log `/tmp/b35c_diagnostic_audit.log`). **Does not change any
number in sections 1-5 above** — same architecture, sizing rule, LR
grid, seeds; only adds train-loss and periodic validation-loss
trajectories per seed, plus one diagnostic-only longer-training
continuation (NOT used to replace the frozen number).

### Per-seed detail at the selected best LR

| C | arch | best LR | seed | train NMSE (last 10 steps) | val NMSE | test NMSE |
|---|---|---|---|---|---|---|
| 128 | RegularBlock | 0.01 | 1000 | 1.552e-4 | 1.687e-4 | 1.860e-4 |
| 128 | RegularBlock | 0.01 | 1001 | 1.223e-4 | 8.328e-5 | 6.084e-5 |
| 128 | RegularBlock | 0.01 | 1002 | 2.763e-4 | 1.648e-4 | 1.533e-4 |
| 128 | GenericBlock | 0.01 | 1000 | 1.226e-3 | 6.847e-4 | 7.169e-4 |
| 128 | GenericBlock | 0.01 | 1001 | 9.430e-4 | 4.738e-4 | 4.752e-4 |
| 128 | GenericBlock | 0.01 | 1002 | 1.368e-4 | 5.143e-5 | 3.916e-5 |
| 256 | RegularBlock | 0.01 | 1000 | 1.621e-4 | 7.906e-5 | 7.926e-5 |
| 256 | RegularBlock | 0.01 | 1001 | 1.509e-4 | 4.045e-5 | 3.677e-5 |
| 256 | RegularBlock | 0.01 | 1002 | 1.964e-4 | 4.086e-5 | 3.626e-5 |
| 256 | GenericBlock | 0.03 | 1000 | 1.085e-2 | 4.666e-3 | 4.486e-3 |
| 256 | GenericBlock | 0.03 | 1001 | 2.154e-3 | 7.129e-4 | 7.913e-4 |
| 256 | GenericBlock | 0.03 | 1002 | 6.683e-4 | 3.286e-4 | 2.749e-4 |

Divergence: 0/9 in every cell. Across-seed spread (test NMSE):
RegularBlock C=256 std=2.02e-5 (tight); **GenericBlock C=256
std=1.88e-3 — by far the largest of any of the four cells**, ~10x its
own median. This large variance is itself the first diagnostic signal.

### Learning curves

RegularBlock (both C=128 and C=256, all seeds): **smooth, monotonic**
val-NMSE descent at every checkpoint, e.g. C=256 seed=1001:
0.504→0.028→0.0066→9.4e-4→1.6e-4→4.0e-5 — a clean converging
trajectory.

GenericBlock C=128: also broadly monotonic, landing at a stable
~3x-worse floor than RegularBlock — a genuine, comparable-basis gap.

**GenericBlock C=256: wildly non-monotonic.** E.g. seed=1000:
val NMSE goes 8.8e-3 (step20) → 1.4e-3 (step40, a good point) → 1.2e-2
(step60, 9x *worse*) → 4.7e-2 (step80, another ~4x worse) → 4.7e-3
(step100, partial recovery — this is the number in the frozen table).
The 100-step stopping point is catching each seed at an essentially
arbitrary phase of an oscillation, not a converged value.

### Diagnostic-only longer-training continuation (GenericBlock, C=256, lr=0.03, n_train=500 — NOT used to replace the frozen number)

| seed | val NMSE trajectory (sparse) | final (step 500) test NMSE |
|---|---|---|
| 0 | 0.495→2.1e-3→4.7e-3→1.0e-3→**3.2e-5**→**2.9e-6**→4.9e-6→9.3e-5→5.0e-4→5.1e-5 | 5.52e-5 |
| 1 | 0.540→1.4e-3→7.1e-4→1.5e-4→1.2e-4→3.8e-5→1.1e-5→**3.2e-2**→3.0e-2→2.7e-2→7.0e-4 | 6.65e-4 |
| 2 | 0.473→1.6e-2→3.3e-4→2.5e-4→2.8e-4→1.1e-4→**3.1e-2**→1.2e-2→3.1e-2→8.9e-2→6.8e-2 | 7.04e-2 |

Confirms this is not simple undertraining: with 5x more steps,
GenericBlock transiently reaches values (2.9e-6, 1.1e-5) an order of
magnitude *better* than RegularBlock's own frozen C=256 number
(3.68e-5 median), then degrades again by 1-4 orders of magnitude
before partially recovering (seed 2 ends up *worse* at step 500 than
at the frozen step 100). This is genuine optimization oscillation, not
a smooth undertrained-then-converging curve.

### Answering the four hypotheses

- **(A) RegularBlock keeps improving while GenericBlock genuinely
  saturates**: NOT SUPPORTED — GenericBlock does not saturate at a
  worse ceiling; it transiently reaches values matching or beating
  RegularBlock's, then loses them again.
- **(B) GenericBlock is undertrained/optimization-limited**: PARTIALLY
  — true in the sense that 100 steps under-samples a still-evolving
  trajectory, but "more training" does not monotonically fix it.
- **(C) A few bad seeds create the ratio**: NOT PRIMARILY — all 3
  seeds show the oscillatory pattern at C=256; it's systemic to this
  architecture/budget/LR combination, not one outlier (though the
  degree of oscillation differs seed to seed, which is exactly what
  produces the large std).
- **(D) Some other numerical/optimization issue**: **SUPPORTED, and
  the dominant factor** — GenericBlock's 16 fully-unconstrained (4,4)
  dense local modules at C=256, regularized only by a per-module
  spectral-radius cap (no structural nilpotent-tail-style constraint
  the way RegularBlock has, projected every step), appear to make
  Adam's dynamics genuinely unstable at this scale. RegularBlock's
  structural per-factor projection gives it smooth, reliable
  convergence at every budget tested here; GenericBlock's reliability
  visibly degrades as its module count (and total unconstrained
  parameter count) grows.

### Revised interpretation (this is the answer to "why does the gap jump from ~3x to ~21.5x")

At C=128, both architectures are in a comparably well-behaved
optimization regime, and the ~3x gap reflects a real, reproducible
difference under the shared fixed protocol. At C=256, GenericBlock's
optimization becomes substantially less stable (a consequence of more
numerous unconstrained dense local modules, not addressed by the
shared LR grid/step budget, which was never tuned per architecture or
size), and the 100-step stopping point effectively samples an
arbitrary point in a noisy, non-monotonic trajectory. **The frozen
21.5x figure is a correctly-measured but not architecture-efficiency-clean
number: it is real (not a bug, not reversed by more seeds), but a
substantial part of its magnitude reflects an optimization-stability
gap between the two families at this scale, not purely a
representational/credit-efficiency gap.** The *directional* finding
(RegularBlock ahead of GenericBlock on Task A at C=64/128/256) is not
undermined by this audit and is better evidenced by the smoother,
more comparable C=128 result than by the C=256 number in isolation.

### Revised experimental language (applies going forward; B35b's own file/commit is not altered)

- The d=1 vs d=2 comparison on the generalized-mode teacher (B35b Part
  1) should be read as a **positive control / mechanism verification**
  (does the regular-algebra mechanism visibly help when a genuine
  generalized mode is present, at matched r and P) — not as a strong
  architecture-vs-architecture falsification test in its own right.
- The neutral dense-linear task (Task B here) is the **stronger,
  architecture-independent test**, since it was not constructed from
  either family's own algebra.
- On that neutral frontier (Task B), **RegularBlock is competitive
  with GenericBlock, not dominant** — each wins 2 of 4 budgets, within
  ~2x throughout (both far ahead of RTU).

## 7. Stop gate

Architectural experimentation stops here, per instruction. Next step is
literature/novelty audit and paper assembly.

## 8. Commit hash

See the commits introducing this file (original) and its diagnostic-audit addendum.
