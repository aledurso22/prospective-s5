# Phase B32a — minimal r=2, d=1 MOVING invariant-bundle construction

Branch `S5-CCM-scale-validation`. Generalizes B29/B30/B31's FIXED-flag
theorem (`E_t=V` for all t) to the theory's more general claim: exact
`d<r` sensitivity requires only a causally moving, source-compatible
bundle `E_t` with `J_t E_t + im(G_t) ⊆ E_{t+1}`, which can coexist with
an ambient Jacobian that is fully dense, alternates between two
matrices with no common real invariant line, and generates the full
matrix algebra `M_r`. Code: `credit_memory/b32a_moving_bundle_r2d1.py`
(`main()` reproduces every number below). Standalone, no training.

**Headline: the moving-bundle case passes at machine precision on all
four independent gradient paths (BPTT, full RTRL, prescribed
moving-bundle reduced RTRL, and a general no-prior-knowledge
dynamic-rank RTRL), and the falsification shows the expected
qualitative split — a deliberately blind prescribed path degrades with
leakage while the dynamic-rank path correctly discovers and adapts to
the leak, remaining exact.**

## 1. Construction

`A=[[0.8,0.3],[0,0.5]]` (e1 is an eigenvector, `A e1=0.8 e1` — the SAME
underlying fact as B29). `P_{2k}=I`, `P_{2k+1}` = a 45° rotation.
z-space recurrence `z_{t+1}=A z_t + e1(theta.phi_t)` (theta enters
linearly); ambient state `h_t=P_t z_t`. Consequently
`J_t^h=P_{t+1} A P_t^{-1}` is fully dense and alternates between two
different matrices, while the exact sensitivity stays confined to the
MOVING bundle `E_t=span(P_t e1)` — because in z-coordinates it is
still confined to the fixed line `span(e1)` (B29's mechanism), just
viewed through a rotating frame.

Four independent paths: (1) ambient BPTT; (2) full ambient RTRL
(`r×P_c`, autodiff `J_t^h`/`G_t^h`); (3) prescribed moving-bundle
reduced RTRL (`1×P_c`, projecting the SAME autodiff Jacobians onto the
KNOWN basis vectors `b_t=P_t e1` — a general restricted-operator
algorithm, not a hand-derived formula, generalized to a time-varying
basis); (4) general dynamic-QR/rank-factorization RTRL (maintains the
full ambient `S_t` but re-factorizes via SVD and truncates to the
*discovered* numerical rank at every step, with **no** knowledge of
`P_t` or `e1` at all).

## 2. Correctness (5 seeds × T∈{1,5,20,100}, eps=0)

| quantity | value |
|---|---|
| worst full-vs-BPTT relative error | 3.224e-15 |
| worst prescribed-bundle-vs-BPTT relative error | 9.219e-16 |
| worst dynamic-QR-vs-BPTT relative error | 1.715e-15 |
| worst reconstructed-sensitivity error (S_recon vs full S_t) | 1.110e-15 |
| `ALL < 1e-8` | **True** |

Long-sequence check (T=100, dynamic QR, no prior knowledge of the
bundle): unique ranks observed = **{1}** throughout; max 2nd singular
value over all 100 steps = **5.309e-16** — the rank-1 structure is
*discovered*, not assumed, and holds machine-precision-tightly for the
whole sequence. Persistent storage at this toy scale: full=2×6=12
floats, reduced=1×6=6 floats, ratio=2.0x (the r/d=2/1 ratio at this
minimal scale — not a claim about the eventual r=64,d=4 scale-up).

(One bug caught and fixed during this phase: the verification script's
`S_recon` initially compared against the WRONG time index of the
moving basis vector — `b_t` instead of `b_{t+1}` — giving spurious
large discrepancies even though the actual gradients already agreed to
~1e-16. Root-caused and fixed before reporting; not a defect in the
RTRL implementations themselves.)

## 3. Falsification (persistent source leak `eps*e2` outside the bundle, deliberately kept in the prescribed path's blind assumption)

| eps | full_rel (all T) | prescribed-bundle_rel, T=100 | dynamic-QR_rel, T=100 | QR ranks observed |
|---|---|---|---|---|
| 0 | ~1e-16–3e-16 | 2.913e-16 | 3.952e-16 | {1} |
| 1e-8 | ~2e-16 | 4.399e-09 | 9.382e-12 | {1,2} |
| 1e-6 | ~6e-16 | 4.399e-07 | 5.658e-16 | {1,2} |
| 1e-4 | ~5e-16 | 4.399e-05 | 6.873e-16 | {1,2} |
| 1e-2 | ~2e-16 | 4.429e-03 | 7.962e-16 | {1,2} |
| 1e-1 | ~7e-16 | 4.636e-02 | 6.460e-16 | {1,2} |

**Full RTRL stays exact for every eps** (it assumes nothing). **The
prescribed moving-bundle path degrades systematically with eps**,
exactly as B29's falsification predicted (deliberately kept blind to
the leak). **The general dynamic-QR path correctly discovers the
leak** — its observed rank grows to {1,2} the moment eps>0 (for T≥5;
a single step at T=1 doesn't yet reveal enough structure to cross the
SVD threshold) — and stays at machine precision throughout, with only
a tiny (~9e-12) truncation-threshold artifact at the very smallest
eps=1e-8, T=100. This is the qualitative split required: a naive
prescribed reduction breaks under leakage while a general, no-prior-
knowledge algorithm adapts and remains exact.

## 4. Structural diagnostics (eps=0)

| diagnostic | value |
|---|---|
| both ambient Jacobians dense | **True** (no zero entries in either alternating `J_t^h`) |
| static generated algebra dim(span{J(even→odd), J(odd→even)}) | **4/4 = M_2** |
| common real invariant line | **none** (ROT's eigenvalues are complex, `0.707±0.707i` — a genuine rotation has no real eigenvector at all) |
| commutant dimension | **1/4** (scalar multiples of I) |
| moving-bundle residual `‖J_t b_t − proj_{b_{t+1}}(J_t b_t)‖` | **1.570e-16** (exact) |
| source-alignment residual `‖G_t − proj_{b_{t+1}}(G_t)‖` | **1.815e-16** (exact) |

## 5. Commit hash

See the commit introducing this file.
