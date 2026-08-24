# Preregistration — PAC analyses A & B (directive 03)

Committed before running. Data: existing protocol (routeA retrains,
deterministic — gated against saved `w_full_s{0,1,2}.npy`; probe batch
RandomState(900+seed)). No new data is collected after this commit that
could tune the predictions.

## Analysis A — pairing test (theory vs decoration)

Per (seed, layer), weighted by `E|q_j|²`:

- COMB: `−arg(1 − conj(a_j)·ρ_j(1))` (the resolvent combination = arg K)
- SINGLE-ρ: `arg ρ_j(1)` (error statistic alone)
- SINGLE-a: `arg a_j` (mode alone)

**Prediction:** R(COMB, arg w) > R(SINGLE-ρ, arg w) AND
R(COMB, arg w) > R(SINGLE-a, arg w) in every layer whose cross-seed
reliability R_w > 0.5 (ceilings from pac_probe2: L0 0.649, L1 0.380,
L2 0.630, L3 0.995 → the test binds at L0, L2, L3).

**Kill:** COMB fails to beat either single in all layers with
R_w > 0.5 → drop the resolvent framing (it is decoration).

## Analysis B — horizon test (is τ the meta-objective's horizon?)

`c(H) = Σ_{k=0}^{H} conj(a_j)^k ρ_j(k)`, H ∈ {1, 2, 4, 8, 16, 32, T−1},
weighted by `E|q_j|²`.

**Prediction (one-step-horizon reading):** R(c(H), arg w) is monotone
non-increasing in H (within noise ±0.02) in ≥ 3 of 4 layers. Signature
of a true horizon effect: H=1 at or near the top from the start.

**Boring alternative (bias–variance):** rise-then-fall — high-k terms
estimated from fewer samples. That pattern reads as estimation noise,
not a horizon effect.

**Kill:** R increases materially (> +0.05) beyond H = 1 in ≥ 2 layers →
the "τ = one-step lookahead" reading is wrong; P3l's twist needs
another explanation.

## Notes

- e-prop baseline for TBPTT: the `online` arm IS the e-prop-family
  local rule (eligibility/S-slot); it is already in the table.
- TBPTT windows are {1, 4, 16, 64} + full BPTT (T = 128 ⇒ the
  W ≥ 128 cell is the full adjoint; window 256 ≡ bptt).
