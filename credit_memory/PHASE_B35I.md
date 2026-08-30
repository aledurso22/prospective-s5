# Phase B35i — training the genuinely online Hessian-transported ProductLocal learner (Stage A)

Branch `S5-CCM-scale-validation`. Architecture, optimizer, projection,
teacher, data stream, and the B35d protocol (LR, EVAL_SEEDS, T_TOTAL,
T_CHANGE, windows) are all unchanged. Code: `credit_memory/
b35i_hessian_transport_training.py` (`/tmp/b35i_stage_a.log`). C=128,
frozen lr=0.01 (from PHASE_B35D.md), EVAL_SEEDS=(11,12,13).

## Unit test (required before long training)

An initial version compared the transport formula against the wrong
reference (the carried/moving-parameter trace, itself already stale)
and failed badly (slopes -0.09 / 0.89 instead of 2 / 3) -- a real bug,
caught and fixed by switching to the correct fixed-theta_t replay
reference (matching B35h's own methodology exactly), using a REAL
Delta_t extracted from one actual continual-training step:

| eta | \|\|S_replay-S+\|\| | \|\|h_replay-h+\|\| |
|---|---|---|
| 0.25 | 1.24e-3 | 1.61e-6 |
| 0.50 | 4.92e-3 | 1.29e-5 |
| 1.00 | 1.94e-2 | 1.02e-4 |
| 2.00 | 7.60e-2 | 8.01e-4 |
| 4.00 | 2.91e-1 | 6.20e-3 |

Log-log slopes: 1.970 (predict 2), 2.978 (predict 3). **Confirmed** --
the A2 transport formula is correctly implemented.

## Stage A: A0 vs A1 vs A2, frozen B35d protocol, C=128

| | A0 (baseline) | A1 (s-only, deliberate ablation) | A2 (full transport) |
|---|---|---|---|
| pre_nmse median | 0.265 | 0.242 | 0.355 |
| post_nmse median | 0.078 | 0.194 | 0.094 |
| loss volatility (std) | 0.383 | 0.391 | 0.361 |
| \|\|Delta_t\|\| median | 4.39e-3 | 4.96e-3 | 4.53e-3 |
| \|\|correction_s\|\|/\|\|s\|\| median | 0 (n/a) | 3.58e-3 | 3.03e-3 |
| \|\|correction_h\|\|/\|\|h\|\| median | 0 (n/a) | 0 (untransported) | 9.65e-4 |
| divergence | 0/3 | 0/3 | 0/3 |

Per-seed: A0 (0.387/0.078, 0.123/0.139, 0.265/0.076); A1
(0.315/0.224, 0.187/0.194, 0.242/0.084); A2 (0.368/0.094, 0.076/0.239,
0.355/0.002). **No consistent ordering across seeds for either A1 or
A2 relative to A0** -- each variant wins on some seeds and loses on
others.

The Hessian correction's own magnitude is tiny relative to the
uncorrected quantities (median ~0.1-0.4% of \|\|s\|\| or \|\|h\|\|),
because \|\|Delta_t\|\| itself is small (Adam steps ~4-5e-3) and the
state correction scales with \|\|Delta_t\|\|^2.

## Direct mechanism test: does A2 reduce carried-vs-frozen mismatch?

At representative checkpoints (T=50,150,300), same B35e-style
diagnostic (carried gradient vs the exact fixed-theta_t frozen
counterfactual):

| T | A0 eps_frozen | A2 eps_frozen | reduction factor |
|---|---|---|---|
| 50 | 0.0111 | 0.0001 | ~111x |
| 150 | 0.0121 | 0.0001 | ~121x |
| 300 | 0.3432 | 0.0049 | ~70x |

**The mechanism itself works exactly as designed** -- A2 reduces the
carried-vs-frozen gradient mismatch by 70-121x at every checkpoint
tested, a dramatic and unambiguous confirmation that the Hessian
correction does what B35h/B35h-online proved it should do.

## Decision (Stage A)

Checking all five required conditions:

1. Actual continual carried-vs-frozen mismatch decreases? **YES**,
   dramatically (70-121x).
2. State+trace transport beats the uncorrected baseline consistently?
   **NO** -- A2's pre/post NMSE are mixed relative to A0 across the 3
   evaluation seeds (better on some, worse on others), not a
   consistent win.
3. Effect survives unseen evaluation seeds? The mismatch reduction
   (point 1) does; the NMSE improvement (point 2) does not.
4. Not explained solely by a smaller effective optimizer step?
   Moot given point 2 already fails -- \|\|Delta_t\|\| is comparable
   across all three variants (4.4-5.0e-3 median) regardless.
5. Matched-credit performance remains scientifically useful? Not
   reached, since point 2 already fails.

**Stage A fails the consistency requirement (condition 2), despite
condition 1 succeeding decisively.** Per instruction: stop; do not add
third-order traces or tune further; do not proceed to Stage B.

## Interpretation

This is a genuinely informative negative result, not a wash. The
correction is mathematically exact (B35h, B35h-online) and
mechanistically effective at its stated goal (reducing staleness by
1-2 orders of magnitude, confirmed directly here) -- but fixing
staleness does not consistently improve RegularBlock's actual
continual-learning behavior on this task. Combined with B35f
(transient/Jordan amplification rejected) and B35g (Adam/projection
interaction rejected), this narrows the live hypothesis space further:
none of the three most concrete, mechanistically well-motivated and
now *directly and successfully corrected* candidate mechanisms
(transient amplification, optimizer/projection mismatch, parameter
staleness) explains the bulk of RegularBlock's B35d instability. The
primary mechanism remains unidentified after B35e-i.

## Commit hash

See the commit introducing this file.
