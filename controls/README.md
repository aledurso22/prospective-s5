# controls/ — exact/oracle controls (BPTT allowed, audited)

Experiments that use exact credit to pin down what PC0's improvement is
(and is not). **None of these are deployable arms** — BPTT calls are
explicitly allowed and audited.

| script | question | frozen answer |
|---|---|---|
| `control_2x2_normmatch.py` | is PC0 generic preconditioning? is it just gain? | BPTT+w ≈ BPTT (±2e-05) → credit-repair-specific; norm-matched PC0 == PC0 at norm ratios up to 564 → direction, not gain. Interaction +0.0193 median. |
| `tbptt_baseline.py` | does buffering W steps of exact credit beat streaming? | tbptt64 0.0003 (beats routeA ~5×); truncation below the delay always loses to online; streaming edge = O(1) memory, no backward pass. |
| `lr_control.py` | is any win just a learning rate? | best standard-Adam LR median 0.0136 — nowhere near PC0/routeA; the rescue is structural, not rate. |

The routeA exact-teacher control itself lives in `toyrig/route_a.py`
(shared machinery); run `python -m toyrig.route_a`.
