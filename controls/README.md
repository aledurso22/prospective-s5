# controls/ — exact/oracle controls (BPTT allowed, audited)

Experiments that use exact credit to pin down what PC0's improvement is
(and is not). **None of these are deployable arms** — BPTT calls are
explicitly allowed and audited.

| script | question | frozen answer |
|---|---|---|
| `control_2x2_normmatch.py` | is PC0 generic preconditioning? is it just gain? | BPTT+w ≈ BPTT (±2e-05) → credit-repair-specific; norm-matched PC0 == PC0 at norm ratios up to 564 → direction, not gain. Interaction +0.0193 median. |
| `tbptt_baseline.py` | does buffering W steps of exact credit beat streaming? | tbptt64 0.0003 (beats routeA ~5×); truncation below the delay always loses to online; streaming edge = O(1) memory, no backward pass. |
| `lr_control.py` | is any win just a learning rate? | best standard-Adam LR median 0.0136 — nowhere near PC0/routeA; the rescue is structural, not rate. |

## Addendum controls C1–C3 (claim-sharpening; results frozen in `../RESULTS_LEDGER.md` §10)

| script | question | frozen answer |
|---|---|---|
| `c1_phase_only_routepc.py` (+ `c1b_phase_only_15seeds.py`) | does unit-modulus RoutePC retain the benefit and improve stability? | COMPETITIVE (5 seeds, registered rule); over 15 seeds: median 0.0137 vs PC0 0.0167, failures 4/15 vs 6/15, beats PC0 9/15 paired. Unit modulus = the more stable variant; registered as `--arm routePCphase` in `train_bench.py`. |
| `c2_real_w_diagnostics.py` | why is the causal real geometry competitive? | Pr(w<0)=0, zero sign flips, quasi-static; effect = relative modal gain structure (directional reweighting), not sign flips, not time-varying gain. |
| `c3_matched_budget_bptt_w.py` | is BPTT+w≈BPTT a floor artifact? | No — BPTT+w is WORSE than BPTT at every budget with Δ_credit>0 on 5/5 seeds; generic-preconditioning rejected with headroom present. |

The routeA exact-teacher control itself lives in `toyrig/route_a.py`
(shared machinery); run `python -m toyrig.route_a`.
