# Phase B35j — direct ComplexLocal vs DualLocal comparison

Branch `S5-CCM-scale-validation`. Does not redesign ProductLocal, does
not introduce Hessian transport. Code: `credit_memory/b35j_complex_vs_dual.py`
(`/tmp/b35j_full.log`).

## Factorization check (reported, not silently bypassed)

B35d's actual RegularBlock uses `D_LOCAL=4` (`b35c_matched_credit_frontier.py`),
not d=2. "DualLocal" here is a fresh, smaller d=2 instance of the same
jet-algebra machinery (`alg_mult_blockwise`, unmodified, at d=2) --
exactly reproducing `M_D(a,b)=[[a,0],[b,a]]`. "ComplexLocal" is a new,
parallel implementation with the identical per-factor structure (Q
factors, 2 real coordinates each, block-diagonal) but the complex-
multiplication rule. Capacity match: Q=64 factors, d=2 => r=P=128 real
coordinates for both, matching B35d's primary C=128 scale.

## Validity (before training)

Reduced RTRL vs full real-coordinate RTRL vs BPTT, fixed theta:
DualLocal rel_err=2.388e-16 (reduced), 2.391e-16 (full); ComplexLocal
rel_err=3.217e-16, 3.285e-16. **Both exact.**

## LR selection (K=1, mixed teacher, tuning seeds 100/101)

Both select lr=0.01 from the shared grid {0.01,0.02,0.05} (DualLocal
scores 0.220/0.531/9.12; ComplexLocal 0.202/0.628/4.19).

## K-sweep, mixed teacher (EVAL_SEEDS=11,12,13)

| K | Dual pre / post / loss_std | Complex pre / post / loss_std | grad norm (both, ~) |
|---|---|---|---|
| 1 | 0.241 / 0.0801 / 0.299 | 0.233 / 0.0247 / 0.284 | 0.03 |
| 5 | 0.0707 / 0.00025 / 0.0838 | 0.1276 / 0.00071 / 0.0887 | 0.08 |
| 20 | 0.0144 / 0.00924 / 0.157 | 0.0148 / 0.00806 / 0.154 | 0.35 |
| 100 | 0.1499 / 0.0470 / 0.268 | 0.1350 / 0.0411 / 0.268 | ~5.0 (at CLIP_NORM) |

Zero divergence at every K, both architectures. Both track each other
closely and share the SAME non-monotonic shape: K=1 worse than K=5;
K=20 comparable to K=5; K=100 worse again, coinciding with the
accumulated block gradient reaching `CLIP_NORM=5.0` for both --
a shared optimizer-interaction artifact, not an algebra-specific one.

## Diagnostic teachers (K=1 and K=100)

| teacher | K | Dual pre/post | Complex pre/post |
|---|---|---|---|
| oscillatory | 1 | 0.1193 / 0.2322 | 0.2151 / 0.2330 |
| oscillatory | 100 | 0.0345 / 0.0284 | 0.0338 / 0.0298 |
| generalized_mode | 1 | **0.0440 / 0.0170** | 0.0701 / 0.0547 |
| generalized_mode | 100 | 0.0806 / 0.0540 | **0.0363 / 0.0248** |

DualLocal's predicted advantage on generalized-mode IS confirmed at
K=1 (clearly lower pre AND post NMSE). ComplexLocal's predicted
advantage on oscillatory is NOT confirmed at K=1 -- DualLocal actually
has the better pre-NMSE there (0.119 vs 0.215), post roughly tied.
Notably the generalized-mode advantage REVERSES at K=100 (Complex
becomes better). The predicted symmetric pattern holds for one half
(Dual/generalized-mode) but not the other (Complex/oscillatory), and
even the confirmed half is not robust across K.

## Carried-vs-frozen mismatch (K=1, mixed teacher, B35e-style diagnostic)

| t | DualLocal eps_frozen / cos | ComplexLocal eps_frozen / cos |
|---|---|---|
| 50 | 0.0143 / 0.9999 | 0.0182 / 0.9999 |
| 150 | 0.0120 / 0.9998 | 0.0216 / 0.9998 |
| 300 | 0.1412 / 0.9993 | 0.1895 / 0.9819 |

Nearly identical magnitude and growth pattern for both algebras --
ComplexLocal's mismatch is if anything slightly LARGER, not smaller,
at every checkpoint. The moving-weight staleness effect is not a
Jordan/nonsemisimple-specific phenomenon.

## Answers

**Q1** (does ComplexLocal train materially more reliably at K=1?):
**No** -- comparable pre-NMSE and loss volatility; a modest post-NMSE
edge for Complex (0.025 vs 0.080), not a decisive difference. Neither
diverges.

**Q2** (does increasing K improve both, or mainly DualLocal?): **Both**
-- near-identical, non-monotonic K-dependence for both architectures.

**Q3** (are both well-behaved under fixed/blockwise parameters?):
**Yes** in the sense of zero divergence at every K, though neither
improves monotonically with K (both degrade at K=100, coinciding with
the shared clip-norm interaction).

**Q4** (does ComplexLocal win oscillatory while DualLocal wins
generalized-mode?): **Half-confirmed** -- Dual wins its native regime
at K=1; Complex does not win its native regime at K=1; the pattern is
asymmetric, and even Dual's confirmed half reverses at K=100.

**Q5** (is moving-weight mismatch qualitatively similar for both
algebras?): **Yes** -- same order of magnitude, same growth pattern,
Complex if anything slightly worse.

## Interpretation

Matches the predeclared rule "if both fail or become unstable at K=1,
but both work for larger K, evidence points to aggressive continual
updating, not Jordan structure" -- in a softer form (neither
architecture literally fails at K=1, but both are consistently worse
there than at the K=5 sweet spot, with near-identical staleness
magnitudes and near-identical K-dependence throughout). There IS a
real, secondary algebra-dependent inductive-bias signal (Dual's edge on
its own native generalized-mode teacher), but it is asymmetric (not
mirrored by Complex on its own native oscillatory teacher) and not
robust across K -- a minor, second-order effect superimposed on a
dominant, shared "aggressive per-sample continual updating is hard for
both algebras" pattern. Neither architecture dominates everywhere;
neither claim is overstated in either direction.

## Commit hash

See the commit introducing this file.
