# README_ROUTEPC — the causal prediction–correction learning rule

**RoutePC / PC0** is a fully causal online-learning rule for diagonal
state-space models: a per-(layer, mode) complex geometry `w` applied to the
streaming online gradient, learned **without any BPTT call** by a delayed
correction off the realized online gradient on the next batch.

Core toy result (frozen, tag `routepc-pc0-frozen`):
**online 0.0224 → PC0 0.0073 median final loss, 4/5 paired seeds,
zero BPTT calls.**

---

## 1. The algorithm (six lines)

Per minibatch `B_n`, with modal geometry `M_w` (per-(layer,mode) complex
`w_j = u_j + i v_j` acting as `conj(w_j)` on the `(a, B)` gradient blocks):

```
g_n       = OnlineGrad(θ_n, B_n)                  # streaming online gradient (RTRL-style)
θ_{n+1}   = BaseOpt(θ_n, M_{w_n} g_n)             # model update through the geometry
# --- when batch B_{n+1} has actually arrived: ---
J_n       = ∂_w [ M_w sg(g_n) ]_{w_n}             # analytic Jacobian of the geometry map
r̂_{n+1}   = −η J_nᵀ g_{n+1}                       # causal surrogate meta-residual
w_{n+1}   = MetaOpt(w_n, r̂_{n+1})                 # delayed correction of the geometry
```

`sg` = stop-gradient. The teacher is the **realized online gradient at the
post-update parameters on the newly arrived batch** — no exact gradient,
no reverse-time cotangent, anywhere.

## 2. Optimizer semantics (as implemented — read this before reusing)

- **θ:** Adam, lr `η = 1e-3`. Global-norm clipping (`CLIP = 1.0`) is applied
  to the flat gradient **before** Adam.
- **`M_w` acts on the raw online gradient**, before clipping and Adam.
  Complex convention: `conj(w)` on the `(a, B)` blocks of each mode; the
  `c` (readout) blocks are untouched. Flat layout is `(Re G, −Im G)`,
  finite-difference gated.
- **w (MetaOpt): plain SGD**, lr `η_m = 1e-3`. The meta chain is a
  **simplified surrogate**: it does not differentiate through Adam or
  clipping.
- **Deployment invariant:** `exact_grad = 0`, `exact_lambda = 0` — counted
  wrappers on both non-causal entry points, asserted in every PC run and in
  `tests/test_pc0_regression.py`.

## 3. Where things live

| what | path |
|---|---|
| **Canonical PC0 implementation + 5-seed protocol** | `toyrig/routepc.py` |
| Canonical entry points | `python -m core.train_routepc` · `python -m core.train_online` |
| Shared numpy rig (forward, online/exact grads, Adam) | `toyrig/ssm_rig.py` |
| routeA (exact-teacher geometry, oracle control) | `toyrig/route_a.py` |
| Probe/data helpers | `toyrig/probes.py`, `toyrig/train_cell.py` |
| **JAX/S5 RoutePC (cluster)** | `train_bench.py --arm routePC` (+ `ssm/online_s5/`) |
| Regression gates | `tests/` (bitwise PC0, JAX meta FD, scan suite) |
| Core result artifact | `results/core_routepc_reproduction/` |
| Frozen headline numbers | `RESULTS_LEDGER.md` |

## 4. Reproduce the core result

```bash
python -m core.train_routepc     # full paired protocol (~tens of minutes, CPU)
python -m tests.test_pc0_regression   # bitwise regression vs stored finals
```

Expected: online `[0.0727, 0.0224, 0.0284, 0.0109, 0.0118]` (median
0.0224), PC0 `[0.0167, 0.0031, 0.0025, 0.0889, 0.0073]` (median 0.0073),
4/5 paired wins, relative improvement 0.674, BPTT audit 0/0.

## 5. Deployable vs oracle — keep them separate

- **Deployable (zero BPTT):** online baseline; PC0 (`toyrig/routepc.py`);
  JAX `routePC` / `routePCreal` arms of `train_bench.py`.
- **Oracle / diagnostic only (BPTT allowed, audited):** routeA
  (exact-teacher geometry); BPTT+w
  (`controls/control_2x2_normmatch.py`); D1/D2
  (`diagnostics/d1_exact_credit_factorization.py`,
  `diagnostics/d2_modal_oracle.py`); teacher-decompose, κ-sweep, E1/E2
  exact-teacher arms, lagged-deficit oracle (`diagnostics/`).

## 6. The controls that pin the claim

| control | result | meaning |
|---|---|---|
| 2×2 (online/PC0/BPTT/BPTT+w) | BPTT+w ≈ BPTT ±2e-05 | credit-regime-specific interaction, **not** generic preconditioning; later audits show it is clipped-Adam/path-dependent, not static BPTT reconstruction |
| norm-matched PC0 | == PC0 (median 0.0073) at ‖M_w g‖/‖g‖ up to 564 | **direction, not gain** |
| LR control | best tuned-Adam LR median 0.0136 | not a learning-rate effect |
| D1 exact-D⁻¹ factorization | cos 1.0, rel 2.4e-15 | placement/factorization theory exact |
| D2 modal oracle | complex 0.901 vs real 0.765 held-out | phase is representationally valuable |
| tbptt64 | 0.0003 (beats routeA) | buffered exact credit is the loss ceiling; streaming edge is O(1) memory |

## 7. Known traps (read before extending)

- **Seed-3 bistable basin.** The delayed-copy task is bistable; several
  arms land in a bad basin at seed 3 (PC0: 0.0889 vs 0.0025–0.0167
  healthy). Report per-seed always; within-script pairing is the robust
  unit; float-order noise can flip basins across processes.
- **‖w‖ drifts to 30–1600.** Adam absorbs the gain; norm-matched PC0 is
  identical. Judge by direction, not magnitude.
- **Failed self-bootstrap (F1).** Replacing the realized-online teacher by
  a stop-grad EMA of w self-amplifies: NaN on most seeds
  (`archive/failed_self_bootstrap/`). Per the preregistered rule, the
  latent-observer stage was **not** built. Deployable estimation of the
  teacher deficit ε remains the open question.
- **w-momentum / TSS residual prospection:** measured, **correction-only**
  verdict (κ-sweep plateau at the matched horizon; no exploitable residual
  drift). Do not re-add prediction terms without new evidence.

## 8. What RoutePC is not (yet)

- Not validated on recognized benchmarks — the cluster gate
  (`train_bench.py` grid) is built and CPU-smoke-clean but **unlaunched**.
- Not the same object as **state prospection** (`ssm/prospective/`, the
  "ghost": forward-dynamics prospective term, diverges at HiPPO init by
  design). The two are independent flags; see
  `S5_ROUTEPC_INTEGRATION_PLAN.md`.
