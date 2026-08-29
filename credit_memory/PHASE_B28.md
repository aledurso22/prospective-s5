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

## 7. Stage 2 — OURS H=8 Autoencode seed-0, 1M frames: target-independent
policy collapse, not a memory-transfer result (FROZEN, no further compute)

Ports "ours" (structured/factored exact-RTRL, full unreduced temporal
basis `V_theta=I_r`) into the identical corrected outer scaffold
verified for RTU (streaming AC(λ), ObGD, literal Elsayed/Farr reward
scaling and observation normalization, sparse-init, separate
actor/critic networks) — code: `credit_memory/b28_ours_faithful_jit.py`,
`credit_memory/b28_ours_calibration_run.py`. Capacity-matched config
`r=16, k=2, n=24` (`nr=384`, matching RTU's `2*hidden_dim=384`
head-input features), encoder output width=64 (matching RTU's actual
Dense-layer output), Phi hidden=8. Correctness gate (forward dynamics +
structured exact RTRL vs BPTT, all 8 families) passes at machine
precision before this run; a real R-spectral-radius instability bug
(RTU's parameterization keeps `rho<1` by construction, ours' dense `R`
does not) was found and fixed via a stability projection
(`project_stable_jnp`, `rho_max=0.95`) before launch.

**Headline: this run does NOT demonstrate natural-task memory
transfer. A target-independent policy bias was already present at the
earliest checkpoint audited (250k) and later hardened into a
near-deterministic collapse onto two fixed actions (~600k onward); the
transient recent500-return peak at ~600k is not sufficient evidence of
memory acquisition.** Archived artifacts:
`credit_memory/b28_stage2_ours_h8_autoencode_seed0/` (production
log/JSON, reconstructed return trajectory, two read-only shadow
diagnostics). No further Autoencode compute on this run unless
explicitly requested.

**Production checkpoint numbers** (recent500 is the primary window;
all_finite=True throughout):

| frame | recent500 mean/median | recent50 mean/median | entropy | step_actor |
|---|---|---|---|---|
| 100k | −0.503/−0.500 | −0.499/−0.519 | ~1.1 | ~5e-4 |
| 250k | −0.499/−0.500 | −0.493/−0.500 | ~1.1–1.3 | ~1e-3 |
| 500k | −0.4845/−0.500 | −0.4846/−0.500 | ~1.1–1.2 | ~1e-3 |
| 1M | −0.4872/−0.500 | −0.4908/−0.500 | 0.0000 | **1.0000** |

RTU seed-0 reference (recent500 only, same scaffold): 500k≈−0.487,
1M≈−0.481 — RTU improves modestly over this window; OURS does not.
Reconstructed rolling recent500 between the official checkpoints (from
the per-episode return list, not separate checkpoints — see the
archived `return_trajectory_500k_to_1M.json`): 500k −0.4845 → **600k
−0.4735 (peak)** → 700k −0.4888 → 800k −0.4918 → 900k −0.4900 → 1M
−0.4872. Actor entropy collapses sharply in the 594k–624k window,
coincident with the return peak, and `step_actor` (the ObGD step-size
multiplier, capped at `actor_alpha=1.0`) rises from its steady
~8e-4–1e-3 to the ceiling. **This is not a runaway-update/ObGD
instability**: the actual actor parameter update RMS (`upd_actor(enc)
/(ours)_rms`) *fell* by roughly 10x across the same transition (e.g.
~7e-4 pre-collapse to ~6e-5 by 700k) — the eligibility trace shrank
faster than the step-size multiplier grew, so `Δθ=step·δ·z` net
decreased even as `step_actor→1`.

**Read-only diagnostic (frozen parameters, no training), identical
eval procedure at the 250k and 1M checkpoints** (5 fixed eval seeds ×
3000 steps each; true-PLAY step and target symbol identified directly
from `env.mode`/`env.deck`, not the off-by-one `obs_next[0]`
convention used in the production logger):

| | 250k (nearest available pre-collapse; **no 500k checkpoint was
preserved** — the live run's single state file is overwritten in
place at each checkpoint, and a copy was only made at 250k and 1M) | 1M |
|---|---|---|
| entropy WATCH / PLAY (mean) | 0.832 / 1.230 | 3.9e-5 / 4.5e-36 |
| true-PLAY accuracy, argmax / sampled | 0.255 / 0.243 | 0.259 / 0.259 |
| argmax during PLAY | 95% action 2, target-independent | 50/50 split actions {0,2}, target-independent |
| target-conditioned action-prob rows | nearly identical across all 4 targets | nearly identical across all 4 targets |
| linear probe, h→target (ridge, 50/50 train/test, n=3770 each, chance=0.25) | train 0.334 / **test 0.279** | train 0.292 / **test 0.276** |

Both test accuracies sit only ~1.8–2.1 SE above chance (SE≈0.0141 at
n=3770) — a weak, borderline signal at best, not evidence of a robust
learned target representation, and materially unchanged between 250k
and 1M.

**Formal interpretation** (retracting the earlier in-session reading
of the 600k peak as evidence of natural-task transfer):
1. At 250k, the true-PLAY policy's argmax was already ~95% one action
   regardless of target, and linear decoding of the target from the
   recurrent state was only marginally above chance. A target-dependent
   memory policy was not established at the earliest audited point.
2. At 1M, the policy has hardened into a near-deterministic two-action
   pattern, still target-independent; hidden-state target decoding
   remains at the same marginal level.
3. Therefore the ~600k entropy transition is best read as **hardening
   of an already non-target-tracking policy**, not destruction of a
   previously-correct memory representation. **Autoencode H=8 seed0 did
   not establish genuine memory learning; a target-independent policy
   bias was already present by 250k and later hardened into
   near-deterministic collapse. The transient return peak around 600k
   is therefore not sufficient evidence of memory acquisition.**

**Explicitly not claimed**: this is **not** a clean Stage-2/C
comparison against RTU (resources are not matched — OURS H=8 has ~6.4k
total params vs RTU's ~53k, and the two runs' checkpoints are not
exactly paired at every frame), and it is **not** a pure Stage-2/E
optimization-collapse verdict either — an optimization/stability
collapse is present (the entropy/step_actor transition), but there is
no evidence a correct memory solution existed before it to be
collapsed *from*.

**Known gaps, recorded rather than left implicit**: no 500k parameter
checkpoint exists (see above); no frozen RTU parameter checkpoint
exists at any frame (`b28_rtu_calibration_run_v2.py` never pickled
state — it predates the resumability feature built for this run), so
item-for-item RTU comparison at the representation level was not
possible without retraining RTU, which was not done.
