# Core RoutePC/PC0 — standalone reproduction artifact

Fully causal modal prediction–correction rule vs the Zucchet-style
online learner; everything else identical. No oracle epsilon, no
bootstrap teacher, no exact teacher, no kappa modification, no
mechanism extras. Later mechanism results are separate and do not
enter this artifact.

## Algorithm (the frozen core)

```
g^on_n      = OnlineGrad(theta_n; B_n)                # RTRL/S-slot
theta_{n+1} = theta_n - eta * M_{w_n} g^on_n          # Adam on clip
r^_n+1     = -eta [d_w (M_w sg(g^on_n))]^dag g^on_{n+1}
w_{n+1}     = MetaOpt(w_n, r^_n+1)                   # SGD, LR_M
```

Implementation: `route_pc.py`, function `train_pc(seed, beta=0)`.

## Protocol (from the source run, verbatim)

```json
{
  "steps": 1500,
  "lr": 0.001,
  "lr_m": 0.001,
  "clip": 1.0,
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "betas": [
    0.0,
    0.25,
    0.5
  ],
  "delta_max": 0.2,
  "L": 4,
  "N": 16,
  "T": 128,
  "delay": 50,
  "batch": 32,
  "bar": "best PC beats online >=4/5 paired seeds and median R_gap >= 0.30"
}
```

## Results (stored values, not forced)

| method | s0 | s1 | s2 | s3 | s4 | median | BPTT calls |
|---|---|---|---|---|---|---|---|
| online | 0.0727 | 0.0224 | 0.0284 | 0.0109 | 0.0118 | 0.0224 | 0 |
| PC0 | 0.0167 | 0.0031 | 0.0025 | 0.0889 | 0.0073 | 0.0073 | 0 |

Paired deltas (online − PC0): ['+0.0560', '+0.0193', '+0.0259', '-0.0781', '+0.0045'] — PC0 wins 4/5 paired seeds.

**Relative improvement (median): (L_online − L_PC0)/L_online = 0.674** (per seed: ['0.77', '0.86', '0.91', '-7.18', '0.38']).

## Audits

- BPTT calls (counting wrappers on `cvm.exact_grad`, `tcg.exact_lambda` in the source run): `{'exact_grad': 0, 'exact_lambda': 0}` for PC0, `{'exact_grad': 0, 'exact_lambda': 0}` for online — zero in both deployed arms.
- Paired RNG streams (60-step md5 of batch bytes, per seed, arm-loop replicas): all equal = **True**
  - s0: `2b0132d481507ce6d4b3e8dcf2aae1df`
  - s1: `eb78a00a735dd5ee55b76f79c8712a80`
  - s2: `aea5b7a1236c1cf0cd77272cb5e513a9`
  - s3: `e39f327435f064c539f9a56dd5a2d6dd`
  - s4: `94282a01e31280d4b3b5f4ad386a0ae1`
- Stop-gradient: h_n bitwise invariant to arbitrary w perturbations = **True**
- Regression gates (`check_route_pc.py`, output in `results/mech_gates_run.log`): stored-PC0 bitwise reproduction (max |diff| 0.0), paired streams, stop-gradient, zero BPTT — ALL PASS.

## Provenance

- Source run: `results/route_pc/summary.json` (git 107a9d282f2863f2397fb0beb9fd6da572b41a25) + `results/route_pc_run.log`
- This packaging: git 5ef1c15eaa6fad3d92ff0a3e3cef39f1915dd3d7, `package_core_routepc.py` (read-only; retrains nothing)
- Reproduce gates: `python check_route_pc.py`; source experiment: `python route_pc.py`

## What this establishes (and only this)

The fully causal modal prediction–correction rule improves the
scalable online learner, with zero BPTT calls in deployment.
