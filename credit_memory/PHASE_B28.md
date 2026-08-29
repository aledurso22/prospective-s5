# Phase B28 Stage 1 — POPGym correctness harness

Branch `S5-CCM-scale-validation`. First stage of the real-benchmark
plan following B27's frozen result (STRONGLY SUPPORTED). Verifies,
**inside actual actor-critic network plumbing** (embedding → recurrent
core → policy head + value head — not standalone cells), that both
architectures' exact online recurrent-credit machinery still matches
BPTT on real POPGym trajectories, before any RL training is trusted.
Code: `credit_memory/b28_popgym_stage1.py` (new; `main()` reproduces
every number below). No S5. No wall-clock claims.

**Headline: Stage 1 passes, machine precision, on both tasks (Autoencode,
RepeatFirst) and both architectures (ours, Nonlinear RTU) — after
finding and fixing one genuinely new bug that this harness was
specifically designed to catch: a shared-parameter role conflict at
the loss level (not previously exercised by any prior phase's
verification, though present in earlier *training* code paths).**

## 1. Scope, stated precisely

Per instruction, this verifies two of the three distinct gradient
notions named in the request:
1. **recurrent-state sensitivity** (`ds_t/dθ` for the recurrent core),
2. the resulting **policy/value loss gradient** (the same sensitivity
   chained through policy and value heads and a genuine actor-critic
   loss, using a fixed, precomputed discounted-return/advantage signal
   from one real random-policy rollout per task).

**Not yet built, explicitly**: (3) any eligibility trace belonging to
a *streaming RL update rule itself* — no such rule exists yet in this
codebase. Per instruction §7, this is kept conceptually and
practically separate from recurrent-core RTRL; Stage 2 will introduce
it, and this phase does not claim anything about it. The embedding is
a **fixed** one-hot encoding (not trained) for this stage, isolating
the recurrent-core/head gradient question before adding a trainable
embedding family.

## 2. Environments used

Read directly from the installed `popgym` package source
(`popgym/envs/repeat_first.py`, `popgym/envs/autoencode.py`), not
assumed:

- **RepeatFirstEasy**: `observation_space=Discrete(4)`,
  `action_space=Discrete(4)`. Reward at every step is `±1/(T-1)`
  depending on whether the action matches the *first* observation —
  genuinely dense, per-step feedback tied to a single persistent cue.
- **AutoencodeEasy**: `observation_space=Tuple(Discrete(2),Discrete(4))`
  (mode flag + card suit), `action_space=Discrete(4)`. Two phases:
  `WATCH` (cards dealt one at a time) then `PLAY` (agent must recite
  cards in **reverse** order, reward `±1/num_cards` per correct/incorrect
  card). Full episodes are long (103 steps for 52 cards); Stage 1 uses
  a short prefix (10–12 steps) of a real trajectory — sufficient for a
  gradient-exactness check, since BPTT and factorized RTRL must agree
  on *any* prefix of *any* trajectory, not just complete episodes.

Trajectories collected via a genuine random policy stepping the real
`gymnasium`-style env (`reset`/`step`), not synthetic data.

## 3. A real bug found and fixed, caught specifically by this harness

`ours_target_fn` reads `z_seq = H@C.T` **directly** in the loss — the
exact same pattern used throughout B25/B26/B27's own training loops
(`train_ours_student`, `train_approximator`, `combined_factorized_grads`
callers). This gives `C` **two roles**: inside the recurrence (via
`Φ(C@h_t,...)`, already correctly handled by `factorized_rtrl_run`'s
`direct_term`) and as the loss's own *direct* readout
(`z_t=H_t@C.T`) — a dependence `factorized_rtrl_run` has no way to
see, since it only tracks how `C` affects `h_t`'s own recurrence, not
how a downstream loss re-uses `C` explicitly.

**First run showed a real, non-noise discrepancy**: `C` error `0.038`
on RepeatFirst (all other families and both architectures already at
machine precision) — investigated rather than assumed to be tuning
noise. Diagnosed precisely and fixed by adding the missing direct
term: `d(loss)/dC|_{H fixed}` (the loss's own dependence on `C`,
holding the already-computed state sequence constant), verified in
isolation first (`1.2e-17` after the fix) before folding into the full
harness. This is the **same lesson recurring in a new context** — the
"shared parameter, two roles" pattern already seen in B25.1
(`z_direct_term`, cross-layer routing) and B26 (the integrated-model
`C` cross-term) — now appearing at the **loss level** rather than the
architecture level.

**Scope of this finding, stated precisely**: B27's frozen, accepted
result is **unaffected** — its training used `train_ours_bptt_adam`
(pure BPTT via a single `jax.grad` call over the flattened parameter
vector), never `factorized_rtrl_run`, so this gap could not have
appeared there. It **does** affect B25's own `train_approximator`
(Part 8, PHASE_B25.md's nonlinear-capacity check), which calls
`combined_factorized_grads` with a loss reading `C` directly — that
training used a `C`-gradient missing this term. The likely practical
consequence is a *less efficient* gradient step for `C` specifically
(a real but incomplete gradient direction, not a zero or reversed
one) — unlikely to have manufactured the qualitative "loss decreases
with n" trend (which does not depend on optimizing `C` to
completion), but the exact reported loss values there should not be
read as fully-optimized numbers. **Not re-run here** — an explicit,
stated scope limit given this phase's focus, flagged as a follow-up
cleanup item rather than silently left unmentioned.

## 4. Verified results

| task | family | ours \|err\| | RTU family | RTU \|err\| |
|---|---|---|---|---|
| RepeatFirst | R,B,C,ψ | 5.2e-18–6.9e-17 | θ,log_radius,Wx | 2.6e-18–2.1e-17 |
| Autoencode | R,B,C,ψ | 5.6e-17–3.3e-16 | θ,log_radius,Wx | 1.4e-17–2.8e-17 |
| both | head (W_π,b_π,W_v,b_v) | exactly `0.0` | head (same) | exactly `0.0` |

(Head-parameter errors are exactly `0.0`, not merely small, because
both methods compute them via the identical plain-autodiff path for
non-recurrent parameters — a consistency check, not an independent
verification of anything new; the recurrent-core families are the
substantive result.)

**Both architectures' exact online recurrent credit matches BPTT to
machine precision, inside genuine actor-critic network plumbing, on
real POPGym trajectories from both target tasks**, after the fix
above.

## 5. What Stage 1 does and does not establish

Establishes: the exact-gradient machinery for both "ours" and
Nonlinear RTU composes correctly with a real policy/value network and
a genuine actor-critic-style loss on real, independently-defined
environments — the necessary precondition before trusting any
training run built on top of it.

Does **not** yet establish: anything about task performance, learning
curves, or whether B27's structural advantage transfers to these
tasks. That is Stage 2 (matched streaming-RL training, ours-exact-RTRL
vs. RTU-exact-RTRL, BPTT versions as offline references only) and
Stage 3 (state/credit Pareto sweep) — not attempted here, per the
plan's own staging.

## 6. Commit hash

See the commit introducing this file.
