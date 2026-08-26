# Prospective SSM: Research Handoff and Next-Branch Plan

**Purpose:** research memo for discussion with the professor and implementation brief for the next codebase agent  
**Repository context:** this repository contains the *original prospective-S5/SSM attempt* and the associated memory-cancellation / discretization-ghost failure. Most of the later algebraic, architectural, and literature investigations described below were performed **outside this repository**. They should be treated as research findings and hypotheses to reproduce, not as results already established by this codebase.

---

## Executive summary

The project began from a physically motivated idea: import **prospective dynamics** from latent-equilibrium / prospective-neuron models into an S5-like state-space model, with the hope that looking ahead in the local dynamics would improve long-range memory and perhaps learning.

The first construction was

\[
\tau \dot s
=
-s + f_\theta(s,t)
+\tau \frac{d}{dt} f_\theta(s,t),
\qquad
f_\theta(s,t)=A_f s+B_f x.
\]

It failed for a structural reason, not merely because of tuning. Substituting the linear SSM target into the prospective equation gives

\[
\tau(I-A_f)\dot s
=
-(I-A_f)s+B_fx+\tau B_f\dot x,
\]

and, when \(I-A_f\) is invertible,

\[
\tau\dot s
=
-s+(I-A_f)^{-1}B_f(x+\tau\dot x).
\]

For the homogeneous dynamics \(x=0\),

\[
\boxed{\tau\dot s=-s.}
\]

The learnable SSM memory spectrum disappears. In other words, the prospective mechanism cancels the lag that the SSM was using *as memory*. The explicit discretization used in the initial code then introduced a second, non-physical root (“ghost mode”), creating an additional numerical failure.

That negative result turned out to be conceptually important. The deeper NLA/GLE/VLE literature suggests that **prospection is naturally a mechanism for cancelling temporal lag and synchronizing teaching signals**, whereas the SSM forward state should remain the retrospective memory substrate. This leads to a new direction:

> **Do not make the S5 memory dynamics prospective. Keep the ordinary S5/LRU forward recurrence, and apply the prospective operator to temporal credit/error signals during online or truncated learning.**

For a single discrete complex SSM mode

\[
h_{t+1}=a h_t+\cdots,
\]

exact BPTT satisfies

\[
\lambda_t=q_t+\bar a\,\lambda_{t+1},
\]

where \(q_t=\partial \ell_t/\partial h_t\) is the instantaneous state error. In the frequency domain,

\[
H_{\mathrm{BPTT}}(\omega)
=
\frac{1}{1-\bar a e^{i\omega}}.
\]

A causal one-step prospective error

\[
\boxed{e_t=q_t-aq_{t-1}}
\]

has transfer function

\[
H_{\mathrm{pro}}(\omega)=1-ae^{-i\omega}.
\]

The ratio is

\[
\frac{H_{\mathrm{pro}}(\omega)}
     {H_{\mathrm{BPTT}}(\omega)}
=
\left|1-\bar a e^{i\omega}\right|^2>0,
\]

hence

\[
\boxed{
\arg H_{\mathrm{pro}}(\omega)
=
\arg H_{\mathrm{BPTT}}(\omega)
}
\]

for every frequency and every stable complex diagonal mode. The causal prospective signal has the **same temporal phase as the future-facing BPTT adjoint**, with a frequency-dependent gain error. This is the discrete SSM analogue of the central GLE phase-matching idea.

The immediate experiment is therefore not another modified forward SSM. It is a controlled test of **Prospective-RTRL / VLE-corrected Prospective-RTRL** in a stacked diagonal SSM, using exact BPTT gradients as the ground truth.

---

# 1. Where the project started

## 1.1 Original physical intuition

The initial idea came from prospective / latent-equilibrium dynamics: rather than responding only to the current state, a dynamical system evaluates a quantity at a locally predicted future state. In its simplest form this introduces terms proportional to temporal derivatives.

The original SSM proposal was

\[
\boxed{
\tau\dot s
=
-s+f_\theta(s,t)
+\tau\frac{d}{dt}f_\theta(s,t)
}
\]

with

\[
f_\theta(s,t)=A_fs+B_fx.
\]

The hope was that the prospective term would improve temporal propagation while retaining the S5/S4 advantages of a linear state-space recurrence.

This was a reasonable hypothesis to test. It was also the wrong place to apply prospectivity.

---

## 1.2 Continuous-time failure: the memory spectrum cancels

Substitute

\[
f_\theta=A_fs+B_fx
\]

and

\[
\frac{d}{dt}f_\theta
=
A_f\dot s+B_f\dot x.
\]

Then

\[
\tau\dot s
=
-s+A_fs+B_fx
+\tau A_f\dot s
+\tau B_f\dot x.
\]

Collecting the state derivative terms gives

\[
\tau(I-A_f)\dot s
=
-(I-A_f)s+B_fx+\tau B_f\dot x.
\]

Assuming \(I-A_f\) is invertible,

\[
\tau\dot s
=
-s+(I-A_f)^{-1}B_f(x+\tau\dot x).
\]

With no input,

\[
\boxed{
\tau\dot s=-s.
}
\]

The key failure is not simply “the system forgets.” The stronger statement is:

> **The learnable \(A_f\)-dependent memory spectrum is cancelled from the homogeneous dynamics.**

Every mode receives the same physical time constant \(\tau\). That destroys the multi-timescale spectrum that gives an SSM its long-memory capability.

This failure is continuous-time. A better numerical integrator cannot repair it.

### Interpretation after studying NLA/GLE

This cancellation now makes physical sense.

The prospective mechanism was designed to **cancel dynamical lag**. But in an SSM, the lag of the recurrent state is not an unwanted hardware delay: it *is the memory*. Applying the NLA-like cancellation mechanism directly to the memory state therefore attacks the very property we wanted to preserve.

This is the first major lesson of the project:

\[
\boxed{
\text{Do not use a lag-cancellation mechanism as the memory mechanism.}
}
\]

---

# 2. The second failure in the original code: the discretization ghost

The initial explicit implementation also produced a separate numerical pathology.

Because the prospective equation contains \(\dot f\), and \(f\) itself depends on the state, the explicit discretization used mismatched finite differences for the state derivative and the prospective derivative. The result became a **two-step recurrence**, despite the underlying continuous-time equation being first order.

A generic scalar form has characteristic polynomial

\[
\mu^2-c\mu+a=0,
\]

with

\[
\mu_{\mathrm{physical}}\mu_{\mathrm{ghost}}=a.
\]

The second root is not a second physical state mode. It is created by the discretization.

In the earlier analysis the collision threshold was found to be

\[
\boxed{
\delta^\star
=
\frac{1-\sqrt a}{1+\sqrt a}.
}
\]

Near the slow-memory regime \(a\rightarrow1\), the admissible step shrinks catastrophically. This is exactly the regime an SSM cares most about.

Two important distinctions must be preserved:

1. **Continuous cancellation:** the prospective target equation removes the SSM spectrum even before discretization.
2. **Discrete ghost:** the chosen explicit scheme adds an extra numerical mode.

They are independent failures.

The current repository is useful precisely because it records these failures. The new branch should **not delete them or silently replace them**. They are controls and part of the research story.

---

# 3. Other issues found in the initial implementation

During the audit, several confounds were identified in addition to the two fundamental failures.

## 3.1 Step-size clamps were compensating for the integrator

Some instability initially attributed to prospectivity persisted even when the prospective horizon was removed. The high-frequency HiPPO/S5 spectrum was itself incompatible with the explicit Euler-like integration at the tested steps.

Therefore the clamp was not a clean test of “how much prospectivity is stable.” It was partly compensating for the base integrator.

## 3.2 Input scaling was inconsistent with the baseline

The original baseline and prospective arm did not use identical discretized input scaling. This produced a large input-gain discrepancy and made the original task accuracy comparison uninterpretable as an architecture comparison.

## 3.3 General lesson

Before interpreting benchmark accuracy, the recurrence, discretization, initialization, and input scaling must be matched.

The next branch should therefore begin from an **unchanged standard S5/LRU forward model**.

---

# 4. First salvage attempt: reinterpret prospection as a mobility

A useful theoretical reframing was

\[
\dot q
=
F(q+\Gamma\dot q).
\]

Linearizing gives

\[
\dot q
=
(I-\Gamma J_F)^{-1}F(q).
\]

The factor

\[
\boxed{
M=(I-\Gamma J_F)^{-1}
}
\]

acts as a mobility / preconditioner.

A Rayleighian/proximal interpretation can be written schematically as

\[
\mathcal R_\tau[\dot q]
=
\frac12\|\dot q\|^2
+
\frac{\Phi(q+\tau\dot q)-\Phi(q)}{\tau},
\]

whose stationarity produces an implicit prospective flow.

This explained why the original explicit discretization was conceptually mismatched: an implicitly defined prospective principle was being forced into an explicit multistep update.

However, another negative result emerged:

> **For a fully linear SSM with free linear parameters, a linear mobility can often be absorbed into a reparameterization of the state-space realization.**

So a generic linear prospective mobility is not, by itself, a source of new useful SSM expressivity.

This pushed the project toward either:
- special/degenerate modes,
- nonlinear/selective mechanisms,
- or using prospection in the **learning dynamics rather than the forward dynamics**.

---

# 5. Constructive forward result obtained elsewhere: the gated increment register

This section describes research performed **outside this repository**. It is included because it explains what was learned, but the codebase agent should not assume these results have been reproduced here.

Applying prospection to the **generator**

\[
F(s,x)=\Lambda s+\tilde Bx
\]

rather than to the equilibrium target, and discretizing with a one-step trapezoidal rule, yields per mode

\[
s_t
=
A_t s_{t-1}
+
C_t x_t
+
E_t x_{t-1}.
\]

For a special \(\lambda=0\) mode with a zero flow step, this reduces to

\[
\boxed{
s_t
=
s_{t-1}
+
\bar g_t(x_t-x_{t-1}).
}
\]

This is an exact gated increment register:
- exact retention,
- selective acquisition,
- no state decay during hold,
- one affine recurrence.

Synthetic experiments outside this repository found that the derived tied controller could be learned and could outperform a more freely parameterized untied controller on the constructed register task.

This was a real constructive result, but later literature and algebra narrowed its novelty:
- input-increment driven dynamics are closely related to controlled differential equation formulations;
- modern gated recurrent/linear-attention architectures already separate important aspects of erase/write;
- the special register task is not enough to establish broad sequence-model advantage.

Therefore this result should remain part of the history, but it should not currently be the main branch objective.

---

# 6. Two-horizon prospection: what the algebra actually says

A second external investigation separated prospectivity in the state and in the input:

\[
\dot s
=
\lambda(s+\Gamma_s\dot s)
+
b(x+\Gamma_x\dot x).
\]

Trapezoidal integration gives

\[
A
=
\frac{1+(\Delta/2-\Gamma_s)\lambda}
     {1-(\Delta/2+\Gamma_s)\lambda},
\]

\[
C
=
\frac{b(\Delta/2+\Gamma_x)}
     {1-(\Delta/2+\Gamma_s)\lambda},
\]

\[
E
=
\frac{b(\Delta/2-\Gamma_x)}
     {1-(\Delta/2+\Gamma_s)\lambda}.
\]

Thus

\[
s_t=As_{t-1}+Cx_t+Ex_{t-1}.
\]

The important decomposition is

\[
C+E
=
\frac{b\Delta}
     {1-(\Delta/2+\Gamma_s)\lambda},
\]

\[
C-E
=
\frac{2b\Gamma_x}
     {1-(\Delta/2+\Gamma_s)\lambda}.
\]

So:
- \(\Delta\) controls the **level / DC** input channel,
- \(\Gamma_x\) controls the **difference / zero-DC** input channel.

For the simple stable mode \(\lambda=-1/\tau,\ b=1/\tau,\ \Gamma_s=0\), define

\[
d=\frac{\Delta}{2\tau},
\qquad
r=\frac{\Gamma_x}{\tau}.
\]

Then

\[
\boxed{
(A,C,E)=
\frac{(1-d,\ d+r,\ d-r)}{1+d}.
}
\]

Special corners include

| operation | \((d,r)\) | \((A,C,E)\) |
|---|---:|---:|
| hold | \((0,0)\) | \((1,0,0)\) |
| add input increment | \((0,1)\) | \((1,1,-1)\) |
| set to current input | \((1,1)\) | \((0,1,0)\) |
| set to previous input | \((1,-1)\) | \((0,0,1)\) |

This clarified the role of the input prospective horizon.

It also exposed a subtlety: exact hold/add for a stable leaky mode uses \(\Delta=0\). This is sensible if \(\Delta\) is interpreted as a **latent flow clock** that may pause while an observation/control path jumps, but it is not ordinary dissipative evolution over positive physical time.

A cleaner mathematical language is therefore a controlled system with two drivers,

\[
dh
=
F(h,v)\,d\tau
+
G(h,v)\,dv,
\]

where \(d\tau\) describes internal relaxation and \(dv\) an observation/control increment.

Again, this is useful understanding, but it points toward controlled-system prior art and does not yet establish the strongest SSM learning contribution.

---

# 7. Non-abelian / Cayley / low-rank transition branch

Another external branch asked whether prospective mobility could generate token-dependent non-commuting transitions.

Low-rank skew/Cayley transitions were able to solve small non-abelian group-tracking tasks in synthetic tests, while purely diagonal transition families showed the expected structural limitations.

However, this direction was deprioritized for three reasons.

### 7.1 Prior art

Orthogonal/unitary recurrent models have long used Cayley or exponential maps. More recent state-space work also uses non-diagonal/block transition structures for state tracking.

So “Cayley transition,” “diagonal plus low rank,” or “non-abelian state tracking” are not defensible novelty claims on their own.

### 7.2 Parallel-scan closure

Although a token-specific diagonal-plus-rank-\(k\) transition can be cheap for one sequential update, products of arbitrary such matrices do not generally remain rank \(k\). Therefore the efficient constant-structure associative scan can be lost.

### 7.3 It is not the clean consequence of the prospective physics

The dissipative/proximal part of the physics is symmetric/contractive, whereas rotations and conservative group actions belong to reversible, antisymmetric dynamics. A full physical model would require a reversible–dissipative split rather than claiming all non-commuting rotations arise from prospective dissipation.

This branch may remain interesting, but it should not be the next experiment in this repository.

---

# 8. The conceptual turn: what NLA/GLE/VLE say prospection is *for*

The most important research development came from returning to the physics rather than trying to rescue a forward transition.

## 8.1 NLA: prospectivity cancels lag

The prospective neuron / latent-equilibrium construction is designed so that a dynamical neuron can react as if it were closer to its future equilibrium. Conceptually, it counteracts the retrospective low-pass filtering of a membrane.

That is exactly why inserting the same mechanism into an SSM memory state is dangerous: an SSM intentionally uses slow modes to encode history.

The original cancellation result is therefore not an isolated algebraic accident. It is consistent with the purpose of the prospective mechanism.

## 8.2 GLE: separate retrospective state dynamics from prospective error dynamics

Generalized Latent Equilibrium (GLE) makes the more useful separation.

Schematically, retrospective dynamics use an integration operator of the form

\[
\mathcal I^-_\tau
=
\frac{1}{1+\tau\partial_t},
\]

whereas a prospective operation has the form

\[
\mathcal D^+_\tau
=
1+\tau\partial_t.
\]

In the GLE credit-assignment interpretation, prospectivity is used to compensate temporal delays in teaching/error signals.

The central observation is not that the causal prospective signal exactly equals the future adjoint in magnitude. Rather, its **phase/timing** can match the future-facing adjoint while remaining causal.

## 8.3 VLE: derive the adjoint first, then understand the approximation

Variational Latent Equilibrium (VLE) provides a particularly useful viewpoint for this project:

1. start from the variational problem;
2. recover the continuous-time adjoint/BPTT object;
3. identify the causal/local prospective approximation;
4. correct the gain mismatch through learnable backward couplings.

That is much closer to a principled learning rule than adding a prospective term to the forward SSM recurrence.

## 8.4 Newer prospective-neuron work supports a hybrid

Recent work on teaching-signal synchronization makes an especially relevant separation: slowly integrating recurrent neurons can provide memory, while prospective mechanisms synchronize the teaching signals needed to train them.

That is almost exactly the architecture suggested here:

\[
\boxed{
\text{retrospective SSM memory}
+
\text{prospective credit dynamics}.
}
\]

---

# 9. Discrete SSM analogue of the GLE phase result

This is the central mathematical hypothesis for the next branch.

Consider one discrete SSM eigenmode

\[
h_{t+1}
=
a h_t + b x_t,
\qquad |a|<1.
\]

Let

\[
q_t
=
\frac{\partial\ell_t}{\partial h_t}
\]

denote the instantaneous state error.

The exact discrete adjoint obeys

\[
\boxed{
\lambda_t
=
q_t+\bar a\,\lambda_{t+1}.
}
\]

Expanding,

\[
\lambda_t
=
q_t+\bar a q_{t+1}
+\bar a^2q_{t+2}+\cdots.
\]

The exact BPTT state credit is therefore a **future-weighted temporal filter**.

For a Fourier component \(q_t=e^{i\omega t}\),

\[
H_{\mathrm{BPTT}}(\omega)
=
\frac{1}
{1-\bar a e^{i\omega}}.
\]

Now define the causal prospective error

\[
\boxed{
e_t
=
q_t-aq_{t-1}.
}
\]

Its frequency response is

\[
H_{\mathrm{pro}}(\omega)
=
1-ae^{-i\omega}.
\]

Since

\[
1-ae^{-i\omega}
=
\left(1-\bar a e^{i\omega}\right)^*,
\]

we have

\[
\frac{
H_{\mathrm{pro}}(\omega)
}{
H_{\mathrm{BPTT}}(\omega)
}
=
\left|
1-\bar a e^{i\omega}
\right|^2.
\]

The right-hand side is real and nonnegative. Therefore

\[
\boxed{
\arg H_{\mathrm{pro}}(\omega)
=
\arg H_{\mathrm{BPTT}}(\omega).
}
\]

### Interpretation

The causal filter

\[
q_t-aq_{t-1}
\]

has exactly the same temporal phase as the noncausal future-facing BPTT adjoint for the same SSM eigenmode.

The mismatch is only the gain

\[
R_a(\omega)
=
\left|
1-\bar a e^{i\omega}
\right|^2.
\]

This gives a very clean SSM translation of the GLE idea:

- the **forward SSM eigenvalue** \(a\) defines the memory timescale and oscillation;
- the same \(a\) defines the matched **prospective credit zero**;
- VLE-like gain correction can then address the remaining amplitude distortion.

---

# 10. Critical caveat: do not claim this improves exact one-layer RTRL

This is essential.

For a digital diagonal recurrent model, exact RTRL sensitivities already give the exact online gradient of the cumulative loss for a single recurrent layer when combined with the correct instantaneous error.

Therefore simply replacing a correct one-layer error with a prospective approximation would not make the gradient “more exact.”

The prospective mechanism becomes interesting in the regime where the online algorithm already makes a temporal approximation in propagating errors through a **deep recurrent hierarchy**.

The 2023 online-LRU work is directly relevant here. Its efficient rule maintains RTRL-like sensitivities for independent recurrent modules but uses instantaneous spatial error propagation across stacked layers. This is the setting where future temporal components of upper-layer credit can be lost.

So the target hypothesis is:

\[
\boxed{
\text{Prospective credit may improve deep online recurrent learning,}
}
\]

not

\[
\text{prospective credit beats exact RTRL in one layer.}
\]

The one-layer case is a required negative control.

---

# 11. Proposed new research direction

Keep the forward recurrence unchanged:

\[
h_{t+1}^{(\ell)}
=
A_\ell h_t^{(\ell)}
+
B_\ell z_t^{(\ell-1)}.
\]

Maintain the ordinary online sensitivity / eligibility object

\[
S_t^{(\ell)}
=
\frac{\partial h_t^{(\ell)}}{\partial\theta_\ell}.
\]

Let \(q_t^{(\ell)}\) denote the instantaneous error obtained by spatial propagation from upper layers.

Define the prospective error

\[
\boxed{
e_t^{(\ell)}
=
q_t^{(\ell)}
-
A_\ell q_{t-1}^{(\ell)}
}
\]

in the diagonal complex eigenbasis.

A first VLE-inspired gain correction is

\[
\boxed{
\tilde e_t^{(\ell)}
=
G_\ell
\left(
q_t^{(\ell)}
-
A_\ell q_{t-1}^{(\ell)}
\right),
}
\]

where \(G_\ell\) is initially a positive real scalar per recurrent mode or per channel.

The parameter update is then formed from the prospective credit and the existing online sensitivities, schematically

\[
\Delta\theta_\ell
\propto
-\operatorname{Re}
\left[
\tilde e_t^{(\ell)*}
S_t^{(\ell)}
\right].
\]

This should be treated as a hypothesis to test, not as an established algorithm.

---

# 12. Why S5/LRU is a good benchmark

S5 is especially useful because its recurrent dynamics are diagonalized into complex modes and its forward computation already supports efficient sequence processing.

A complex mode

\[
a_i=r_i e^{i\theta_i}
\]

contains:
- a memory timescale through \(r_i\),
- an oscillation/frequency through \(\theta_i\).

The prospective credit filter

\[
q_t-a_iq_{t-1}
\]

therefore automatically inherits the same temporal geometry.

This gives a strong conceptual pairing:

\[
\boxed{
\text{forward pole}
\longleftrightarrow
\text{prospective credit zero}.
}
\]

For the first implementation, however, the **Online-learning-LR-dependencies** codebase is the most useful external reference because it already contains:
- complex diagonal recurrent modules,
- BPTT,
- the efficient online/RTRL-style rule,
- spatial backpropagation,
- one-step truncated BP,
- SnAp-1,
- tests for custom learning rules.

This repository does not need to be replaced by that codebase. The new branch can use the current S5 implementation and transplant/reimplement only the minimal learning-rule machinery needed for the controlled experiment.

---

# 13. Instructions for the codebase agent: create a new branch

## 13.1 Branch policy

Create a new branch, for example

```text
research/prospective-credit-s5
```

Do **not** rewrite the historical prospective-forward branch.

Preserve:
- the original prospective equation implementation;
- the failure reproduction;
- ghost-mode diagnostics;
- any baseline logs/configurations associated with the original experiment.

The historical branch is evidence.

The new branch should start from a **standard, non-prospective S5 forward recurrence**.

---

# 14. Phase 0 — establish a trustworthy baseline

Before implementing prospective credit:

1. Identify the current standard S5 forward path in the repository.
2. Disable/bypass the original prospective forward-state modification.
3. Reproduce at least one known baseline configuration.
4. Verify that discretization and input scaling match the standard S5 implementation.
5. Record package versions and random seeds.
6. Add a small deterministic unit test for a diagonal scalar/complex SSM.

Do not begin with a large benchmark.

---

# 15. Phase 1 — algebra/unit-test harness

Implement a minimal test independent of model accuracy.

For a scalar or complex mode

\[
h_{t+1}=ah_t+bx_t
\]

generate arbitrary error sequences \(q_t\).

## Test A: exact adjoint

Compute the reference

\[
\lambda_t
=
q_t+\bar a\lambda_{t+1}
\]

by reverse recursion.

## Test B: prospective signal

Compute causally

\[
e_t=q_t-aq_{t-1}.
\]

## Test C: Fourier phase identity

For a bank of frequencies, compare

\[
H_{\mathrm{BPTT}}
=
\frac{1}{1-\bar a e^{i\omega}}
\]

and

\[
H_{\mathrm{pro}}
=
1-ae^{-i\omega}.
\]

Verify numerically that

```text
max_phase_error < 1e-6
```

up to numerical precision.

Test:
- real \(a\);
- complex \(a\);
- \(|a|=0\);
- \(|a|=0.5\);
- \(|a|=0.9\);
- \(|a|=0.99\);
- oscillatory phases \(\arg a\neq0\).

Also verify the gain identity

\[
\frac{|H_{\mathrm{pro}}|}
{|H_{\mathrm{BPTT}}|}
=
|1-\bar a e^{i\omega}|^2.
\]

This must pass before any training experiment.

---

# 16. Phase 2 — gradient-ground-truth experiment

Construct a very small stacked diagonal recurrent network where exact BPTT is cheap.

For the same minibatch/model parameters, compute gradients using:

1. **BPTT** — ground truth.
2. Existing/implemented **online_full / RTRL-style** rule.
3. **Spatial-only** online error.
4. **1-step TBPTT**.
5. **Prospective-RTRL**.
6. **VLE-Prospective-RTRL**.

Primary metric:

\[
\boxed{
\cos(
\nabla_\theta L_{\mathrm{method}},
\nabla_\theta L_{\mathrm{BPTT}}
).
}
\]

Also record:
- relative gradient error;
- gradient norm ratio;
- training loss;
- wall-clock time;
- peak memory.

Accuracy is secondary at this stage.

---

# 17. Mandatory null tests

These are more important than obtaining a positive result.

## Null 1: one recurrent layer

For a single independent diagonal recurrent layer, exact RTRL is already correct.

Therefore:

\[
\boxed{
\text{Prospective credit should not systematically beat exact RTRL.}
}
\]

If it does, first suspect an implementation bug or an unfair comparison.

## Null 2: no recurrence

Set

\[
A=0.
\]

Then

\[
e_t=q_t.
\]

Prospective credit must collapse exactly to the instantaneous error rule.

## Null 3: zero prospective correction

Provide a configuration in which the prospective correction is disabled. It must exactly reproduce the ordinary online baseline.

## Null 4: shuffled/error-phase control

Destroy temporal alignment in the error sequence. Any advantage specifically attributed to phase synchronization should disappear or substantially weaken.

---

# 18. Phase 3 — depth × memory-timescale sweep

This is the decisive mechanistic experiment.

Sweep depth

\[
L\in\{1,2,4,8\}
\]

and recurrence magnitude

\[
|a|
\in
\{0,\ 0.5,\ 0.9,\ 0.99,\ 0.999\}.
\]

For complex modes also sweep representative phases.

The hypothesis is not a specific numerical law, but a directional prediction:

\[
\boxed{
\text{prospective advantage should increase
when both depth and memory timescale increase.}
}
\]

Reason:
- deeper recurrent hierarchies create larger temporal misalignment in approximate online spatial error propagation;
- \(|a|\rightarrow1\) means future credit extends farther in time.

Plot heatmaps for:
- gradient cosine to BPTT;
- relative gradient error;
- training convergence.

This is a stronger test than a single benchmark accuracy number.

---

# 19. Phase 4 — isolate the VLE gain claim

The raw prospective signal has correct phase but wrong frequency-dependent amplitude.

Use controlled temporal targets:

\[
q_t
=
\sum_k c_k\sin(\omega_k t+\phi_k).
\]

Compare:

1. exact BPTT adjoint;
2. raw prospective signal;
3. prospective signal with analytic oracle gain for a single frequency;
4. learned scalar gain per mode;
5. optionally, a slightly richer positive gain parameterization.

For one dominant frequency \(\omega_0\), the ideal gain is related to

\[
\frac{1}
{|1-\bar a e^{i\omega_0}|^2}.
\]

Questions:
- Does learned \(G_i\) correlate with the analytic gain needed by that mode?
- Does gain correction improve gradient cosine without destroying the phase advantage?
- Is the correction more important for broadband/high-frequency targets?

Do not introduce a large neural network for \(G\) until the scalar-per-mode experiment is understood.

---

# 20. Phase 5 — sequence-learning benchmarks

Proceed only if the gradient experiments support the mechanism.

Recommended order:

1. synthetic copy / delayed-memory task;
2. sMNIST / psMNIST;
3. sequential CIFAR;
4. ListOps or another structured long-range task.

For each benchmark compare:
- BPTT;
- existing online rule;
- prospective online rule;
- VLE-corrected prospective online rule.

Report both task performance and gradient-alignment diagnostics.

Do not start with Path-X or large language modeling.

---

# 21. Stop/go criteria

The project should be willing to kill the new hypothesis.

### Strong evidence to continue

Continue if, across seeds:

1. the phase identity tests pass exactly;
2. one-layer/null controls behave as predicted;
3. prospective credit improves BPTT-gradient alignment specifically for deeper and slower recurrent hierarchies;
4. VLE gain correction improves over raw prospective credit in regimes where amplitude distortion matters;
5. the improvement translates into better online/truncated training at comparable compute/memory.

### Reasons to stop or rethink

Stop this branch if:
- improvement appears equally in \(L=1\) and deep networks;
- improvement persists at \(A=0\);
- phase tests do not match theory;
- gains improve benchmark accuracy but not gradient alignment and have no interpretable relation to the predicted amplitude error;
- the method only wins after substantially more computation than BPTT/online baselines;
- results disappear under fair seed/configuration sweeps.

A clean negative is valuable.

---

# 22. What the agent should **not** do

Do not:

1. try to rescue the original forward prospective recurrence by more clamps;
2. tune \(\tau\), \(\gamma\), or the step size until the memory-cancelled model appears competitive;
3. treat the discretization ghost as the only problem;
4. modify the S5 forward memory spectrum in the first prospective-credit experiment;
5. claim prospectivity creates new SSM expressivity;
6. claim the causal signal equals BPTT in magnitude;
7. claim it should beat exact one-layer RTRL;
8. evaluate only final accuracy;
9. jump directly to LRA/LM benchmarks;
10. delete or overwrite the old negative-result code.

---

# 23. What is potentially novel, and what is not

This section is intentionally conservative.

## Not safe novelty claims

Do **not** claim as new:
- prospective neurons / latent equilibrium;
- GLE-style causal temporal credit;
- variational/local approximations to BPTT;
- online RTRL-style learning for LRUs/SSMs;
- input-increment driven recurrent dynamics;
- Cayley/orthogonal recurrent transitions;
- low-rank or block non-diagonal SSM transitions;
- simply separating erase and write.

## More defensible research gap to test

A narrower possible contribution is:

> **Derive and test the exact discrete complex-SSM analogue of GLE phase matching, then use it as a mode-matched prospective credit filter in deep online SSM learning, with a VLE-inspired gain correction.**

The attractive feature is that the same SSM eigenvalue determines:
- the forward memory pole;
- the phase-matched causal credit zero.

This yields a very specific hypothesis with exact null tests.

Whether that is publishable depends entirely on the experiments.

---

# 24. Relation to the original professor idea

The project has not abandoned the original physics. It has become more faithful to it.

The first attempt effectively said:

\[
\text{“make the SSM memory state prospective.”}
\]

The algebra said no: prospectivity cancelled the memory spectrum.

The deeper physics suggests why:

\[
\text{prospection cancels temporal lag.}
\]

The new proposal instead says:

\[
\boxed{
\text{use retrospective SSM dynamics to store memory,}
}
\]

\[
\boxed{
\text{use prospective dynamics to synchronize credit.}
}
\]

This is a more coherent mapping from the biological/variational theory to the computational role of an SSM.

The failed recurrence is therefore not wasted work. It is the result that forced the correct separation between **memory dynamics** and **learning dynamics**.

---

# 25. Minimal implementation sketch

The precise JAX/Flax implementation will depend on the current repository, but conceptually the new component should look like this.

```python
class ProspectiveCreditState:
    prev_q: Array  # same shape as diagonal recurrent state error


def prospective_error(q_t, prev_q, Abar):
    # Abar is the discretized complex diagonal recurrence.
    e_t = q_t - Abar * prev_q
    return e_t, ProspectiveCreditState(prev_q=q_t)
```

For complex states, verify the code's conjugation convention carefully. The theoretical adjoint recursion uses the conjugate/adjoint of the forward transition:

\[
\lambda_t=q_t+A^\dagger\lambda_{t+1}.
\]

The prospective causal filter must be implemented consistently with the representation and real-valued loss.

VLE gain:

```python
e_t = prospective_error(...)
e_t = gain * e_t
```

Start with a positive scalar or vector gain:

```python
gain = softplus(raw_gain)
```

The gain is allowed to be positive because it is correcting amplitude, not changing temporal phase.

---

# 26. Suggested files/modules in the new branch

Adapt names to the repository.

```text
research/
    prospective_credit/
        theory_checks.py
        gradient_alignment.py
        spectral_sweep.py
        multifrequency_gain.py

s5/
    prospective_credit.py

tests/
    test_prospective_phase.py
    test_prospective_nulls.py
    test_gradient_reference.py

configs/
    prospective_credit/
        tiny_linear.yaml
        depth_sweep.yaml
        spectral_sweep.yaml
        smnist.yaml
```

Keep the implementation modular enough that the same standard S5 forward model can be trained with either:
- BPTT;
- existing online learning;
- prospective online learning.

---

# 27. Logging requirements

Every experiment should log:

```text
git commit
branch
seed
dataset/task
depth
state dimension
Abar statistics: |a| and phase
learning rule
prospective enabled/disabled
gain type
loss
accuracy if applicable
gradient cosine vs BPTT
relative gradient error
gradient norm ratio
wall time
peak memory
```

For spectral experiments also save per-mode:
- eigenvalue magnitude;
- eigenvalue phase;
- learned VLE gain;
- predicted/oracle gain where defined.

---

# 28. External references for the agent

These are the main references that motivated the new direction.

### S5

Jimmy T. H. Smith, Andrew Warrington, Scott W. Linderman,  
**Simplified State Space Layers for Sequence Modeling**, ICLR 2023.

- Paper: https://arxiv.org/abs/2208.04933
- Code: https://github.com/lindermanlab/S5

S5 provides the relevant complex diagonalized state-space substrate and efficient parallel scan.

### Online learning of long-range dependencies

Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, João Sacramento,  
**Online learning of long-range dependencies**, NeurIPS 2023.

- Paper: https://arxiv.org/abs/2305.15947
- Code: https://github.com/NicolasZucchet/Online-learning-LR-dependencies

This is the most useful external implementation reference for efficient online learning with independent complex recurrent modules. The repository already contains `online_full`, spatial BP, one-step truncated BP, SnAp-1, and BPTT.

### Generalized Latent Equilibrium

**Backpropagation through space, time and the brain**, Nature Communications, 2025.

- https://www.nature.com/articles/s41467-025-66666-z

GLE is the main conceptual source for using prospective dynamics to make causal/local teaching signals temporally align with future-facing credit.

### Variational Latent Equilibrium

Simon Brandt, Paul Haider, Walter Senn, Federico Benitez, Mihai A. Petrovici,  
**A Variational Latent Equilibrium for Learning in Cortex**, 2026.

- https://arxiv.org/abs/2603.09600

VLE is important because it starts from the variational/adjoint problem and clarifies the relation between exact temporal credit and the local prospective approximation, including gain correction.

### Teaching-signal synchronization with prospective neurons

Nicolas Zucchet, Qianqian Feng, Axel Laborieux, Friedemann Zenke, Walter Senn, João Sacramento,  
**Teaching signal synchronization in deep neural networks with prospective neurons**, 2025/2026.

- https://arxiv.org/abs/2511.14917

This work is especially relevant to the separation proposed here: slow recurrent dynamics can provide memory while prospective mechanisms compensate timing delays in teaching signals.

---

# 29. Final working hypothesis

The project should now test the following statement:

> **In a deep online-trained diagonal SSM, the recurrent modes that create long temporal memory also create temporal misalignment in approximate instantaneous hierarchical credit. A causal GLE-inspired filter \(e_t=q_t-Aq_{t-1}\) is exactly phase-matched to the future-facing BPTT adjoint mode by mode. A VLE-inspired gain can correct the remaining amplitude distortion. This may improve deep online/truncated SSM training without changing the forward memory dynamics.**

This statement is:
- derived from the physics;
- specific to the SSM spectrum;
- falsifiable;
- compatible with the original negative result;
- testable against exact BPTT;
- and narrow enough not to rely on broad novelty claims already occupied by the literature.

---

# 30. Immediate instruction to the next agent

**Do not continue tuning the old prospective-forward SSM.**

Create a new branch and implement the smallest possible **prospective-credit** experiment on top of the standard S5 forward recurrence.

The first deliverable should not be a benchmark score. It should be a report containing:

1. exact reproduction of the scalar/complex phase theorem;
2. exact BPTT gradient reference;
3. one-layer null result;
4. \(A=0\) null result;
5. depth × memory-timescale gradient-alignment sweep;
6. raw prospective vs VLE-gain-corrected prospective comparison.

Only after those six items behave as predicted should the branch move to sMNIST, sequential CIFAR, ListOps, or other sequence benchmarks.

---

## One-sentence research story

> We began by making the S5 state itself prospective and discovered that the prospective mechanism cancels the learnable memory spectrum; after tracing that failure back through NLA, GLE, and VLE, the more faithful hypothesis is to leave S5's retrospective memory untouched and use prospectivity where the physics uses it most naturally: to phase-align temporally delayed credit signals during online learning.
