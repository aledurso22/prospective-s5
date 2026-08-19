# Prospective SSM vs baseline S5 (JAX)

A research codebase comparing the standard **S5** diagonal state-space layer
(first-order recurrence, bilinear discretization) against a **prospective
SSM** layer that implements the Euler-discretized prospective update as a
per-channel 2x2 matrix recurrence — both evaluated with
`jax.lax.associative_scan`, both trained with an identical budget on
sequential MNIST.

**> To reproduce the headline result (the prospective recurrence's
> instability, "the ghost"), start at
> [REPRODUCE_GHOST.md](REPRODUCE_GHOST.md).**

---

## 1. Code map — what is S5, what we add

The package is split three ways so the contribution is isolated:

```
prospective-s5/
├── ssm/
│   ├── shared/              ── SHARED: identical in both arms ──────────────
│   │   ├── hippo.py           HiPPO-LegS matrix + S5/DPLR diagonalization
│   │   │                      (Lambda, unitary V, B~ = V^{-1} B)
│   │   ├── params.py          ComplexParam container + the S5 initializers
│   │   └── block.py           S5Block: LayerNorm -> SSM -> GLU -> residual
│   │
│   ├── baseline_s5/         ── BASELINE ARM: standard S5, the control ──────
│   │   ├── scan.py            first-order elementwise associative scan
│   │   └── layer.py           S5SSM + bilinear (Tustin) discretization
│   │
│   ├── prospective/         ── PROSPECTIVE ARM: THE NEW WORK ───────────────
│   │   ├── scan.py            2x2 companion affine associative scan
│   │   └── layer.py           ProspectiveSSM (Euler-discretized prospective
│   │                          update) + the causal 2-tap input term
│   │
│   └── model.py             ── SHARED: SequenceClassifier, the arm switch ──
│
├── train.py                 sMNIST training loop, --model {baseline,prospective}
├── test_scan.py             9 correctness checks: every scan == sequential ref
├── ghost_demo.py            standalone numpy diagnosis of the instability
├── exact_failure.py         the derivation verbatim + where it overflows
├── data/                    MNIST IDX .gz files (auto-downloaded)
└── results/                 metrics_<model>.json written here
```

**Everything outside `ssm/prospective/` is standard S5.** The two arms share
the HiPPO initialization, the parameters (`Lambda, B, C, D`), the
block, the pooling, the head, the optimizer and the training loop.

### Suggested reading order

1. `ssm/shared/hippo.py` — where `Lambda`, `V`, `B~` come from, and why the
   normal/DPLR route is used instead of a direct eigendecomposition.
2. `ssm/baseline_s5/scan.py` — the first-order scalar affine map and why it
   is associative (the whole reason SSMs parallelize).
3. `ssm/baseline_s5/layer.py` — the control layer, end to end.
4. `ssm/prospective/scan.py` — the same idea lifted to 2x2 matrix affine
   maps, because the prospective update is second order.
5. `ssm/prospective/layer.py` — written to mirror (3) line for line; the
   diff between those two files *is* the contribution.
6. `ssm/model.py` — the single `if model_type == ...` switch.
7. `test_scan.py`, then `ghost_demo.py`.

### The delta, precisely

| | baseline `S5SSM` | prospective `ProspectiveSSM` |
|---|---|---|
| recurrence order | first: `s_t = f(s_{t-1})` | second: `s_t = f(s_{t-1}, s_{t-2})` |
| scan element | scalar affine `(a, b)` | 2x2 matrix affine `(M, b)` |
| discretization | bilinear / Tustin | Euler (of the prospective ODE) |
| input term | `B_bar x_t` | `x~_t = (1+rho) B x_{t-1} - B x_{t-2}` (2-tap causal conv) |
| extra params | — | `log_ratio = log(rho)`, per channel |
| `A` used as | `Lambda_bar` (bilinear) | **`Lambda`** (default, verbatim) / `Delta * Lambda` clamped (`--stabilized`) |
| init | `log_step ~ U[log 1e-3, log 1e-1]` | `log_step ~ U[log 1e-4, log 5e-4]` |
| shared | HiPPO init, `Lambda/B/C/D`, block, classifier, training loop | same |

---

## 2. Math summary (the prospective derivation)

Continuous prospective dynamics (Zucchet et al. 2025 form), with
`f_theta(s, t) = A s + B x_t`:

```
tau * ds/dt = -s + f_theta(s, t) + tau * (d/dt) f_theta(s, t)
```

Euler discretization with step `Delta t` (write `rho = Delta t / tau`):

```
s_t - s_{t-1} = -rho * s_{t-1} + (1 + rho) f_{t-1} - f_{t-2}
```

i.e.

```
s_t = [ (1 - rho) I + (1 + rho) A ] s_{t-1}  -  A s_{t-2}
      + (1 + rho) B x_{t-1}  -  B x_{t-2}
```

Define

```
A1~ = (1 - rho) I + (1 + rho) A,   A2~ = -A
x~_t = (1 + rho) B x_{t-1} - B x_{t-2}
```

and stack `z_t = [s_t ; s_{t-1}]`:

```
z_t = [[A1~, A2~], [I, 0]] z_{t-1} + [x~_t ; 0]
```

With `A` diagonal (complex, S5-style — HiPPO-LegS eigenvalues), `A1~`, `A2~`
are diagonal, so the recurrence is a **per-channel 2x2 matrix recurrence**

```
[s_t ; s_{t-1}]_i = M_i [s_{t-1} ; s_{t-2}]_i + [u_{t,i} ; 0],
M_i = [[a1_i, a2_i], [1, 0]]
```

which is associative-scan compatible: each time step is an affine map
`(M, b)`, and maps compose as

```
(M2, b2) o (M1, b1) = (M2 @ M1, M2 @ b1 + b2)
```

evaluated in parallel over time with `jax.lax.associative_scan`
(`ssm/prospective/scan.py`). The input term `x~_t` is a causal conv1d with
kernel `[(1+rho), -1]` acting on the B-transformed inputs (taps multiply
`[Bx_{t-1}, Bx_{t-2}]`), see `ssm/prospective/layer.py: causal_conv1d_time`.

The baseline S5 layer uses the standard first-order diagonal recurrence

```
s_t = Lambda_d ⊙ s_{t-1} + B_d x_t       (bilinear/Tustin discretization)
```

with the S5 elementwise associative scan on `(a, b)` pairs
(`ssm/baseline_s5/scan.py`).

### HiPPO initialization (important numerics note)

The raw HiPPO-LegS matrix has **real integer eigenvalues -1, ..., -N** and an
exponentially ill-conditioned eigenvector matrix (measured: cond(V) ~ 8e10 at
N=16, ~8e20 at N=64), so `V^{-1} B` cannot be computed directly at N=64. Like
the actual S5 implementation, we therefore diagonalize the **normal part** of
the NPLR decomposition instead: `A + P P^T = S = -0.5 I + K` (K
skew-symmetric) is normal, so `S = V diag(Lambda) V*` with **unitary** V
(cond = 1) and complex `Lambda = -0.5 + i*eig(K)` (computed stably via
`eigh(S * -1j)`). Then `B~ = V* B_hippo = V^{-1} B_hippo` and outputs map back
with `Re(V .)` — i.e. the diagonal-basis contract of the SPEC holds with a
perfectly conditioned V, exactly as in S5
(`ssm/shared/hippo.py: make_dplr_hippo`).

### Stability note (documented fallback)

The prospective Euler recurrence is a *second-order* difference equation; the
companion matrix `M_i = [[a1_i, a2_i], [1, 0]]` has a parasitic mode with
`|eig| ~ |A_i|` (the product of its two eigenvalues equals `-a2_i = A_i`).
With the HiPPO eigenvalues (`|Lambda|` up to ~1.3e3 at N=64) the recurrence is
**unstable for every rho** if A is used unscaled. Measured at the real
DPLR/HiPPO init, N=64 (`python exact_failure.py`):

| rho = Delta t/tau | physical root | parasitic root | float32 overflow |
|---|---|---|---|
| 0.5   | ~0.60-0.67 | **1954.91** | step 13 |
| 0.1   | ~0.90-0.91 | **1433.60** | step 14 |
| 1e-3  | ~0.999     | **1304.58** | step 14 |

The physical root behaves exactly as intended (`~ 1 - rho`); it is the
parasitic root, `|mu| ~ |A|`, that destroys the run — and shrinking `rho`
does not touch it, because `mu1*mu2 = A` is set by the spectrum, not the
step size.

**As of the `exact` flag, this fallback is OPT-IN and off by default.** The
prospective layer now runs the derivation verbatim (`A = Lambda`, `B`
unscaled, no clamps) and diverges — `python exact_failure.py` measures the
companion spectrum and the overflow step. Pass `--stabilized` to `train.py`
to enable the fallback below, which is what the numbers in `results/` were
produced with.

Fallback (opt-in, `--stabilized`):

* `A := Delta * Lambda` with a **trainable log-Delta** exactly like S5 — the
  consistent discrete-time reading of the derivation's `f_theta` — while
  `B` is used **unscaled**, exactly as written in the derivation;
* the trainable ranges are clamped to `Delta in [1e-5, 5e-4]` and
  `rho in [1e-3, 0.25]`, for which `max |eig(M_i)| <= 0.90` at
  initialization (measured; the clamp is a `jnp.clip` on the log-parameters,
  transparent to the optimizer).

Everything else in the update is implemented EXACTLY as derived above.

---

### Turning prospection off (the matched control)

`--gamma` scales the prospective term (`ssm/prospective/layer.py`), following
the partial-prospection form

```
s_t = [(1-rho)I + (rho+gamma)A] s_{t-1} - gamma*A s_{t-2}
      + (rho+gamma) B x_{t-1} - gamma*B x_{t-2}
```

* `--gamma 1` (default) — the derivation as written;
* `--gamma 0` — the prospective term is gone, `a2 = 0`, and the recurrence is
  first-order **explicit-Euler S5**: same parameters, same code path, same
  discretization.

This is a cleaner control than `--model baseline`, which is also plain S5 but
uses **bilinear** discretization — so it changes the integrator and the
prospective term at once.

Since `mu1*mu2 = gamma*A`, gamma scales the parasitic root linearly. Measured
at `rho=0.1`, exact `A = Lambda` (from `exact_failure.py`):

| gamma | 1.0 | 0.5 | 0.1 | 0.01 | 0.0 |
|---|---|---|---|---|---|
| max\|mu\| | 1433.60 | 781.96 | 260.65 | 143.36 | **130.33** |

Note the floor: at `gamma = 0` there is no prospective term and no parasitic
root, and it is **still unstable** — the surviving physical root is
`a1 = (1-rho) + rho*A`, and `|rho*A| = 130` at the HiPPO spectrum. Explicit
Euler cannot integrate this spectrum at all. That is a *third* failure,
independent of both the continuous cancellation and the parasitic root, and
it is why a trainable control needs `--gamma 0 --stabilized`.

---

## 3. Usage

```bash
pip install -r requirements.txt      # or: bash setup.sh   (GPU auto-detect)

# 1. correctness of both associative scans (must pass, 9/9)
python test_scan.py

# 1b. the derivation run verbatim, and exactly where it breaks
python exact_failure.py

# 2. the instability diagnosis, no training, numpy only
python ghost_demo.py

# 3. train (identical budget for both arms)
python train.py --model baseline    --epochs 3 --subset 20000
python train.py --model prospective --epochs 3 --subset 20000
```

`train.py` downloads the four MNIST IDX `.gz` files directly from
`https://storage.googleapis.com/cvdf-datasets/mnist/` into `data/` and parses
them with `numpy + gzip + struct` (no torchvision/tensorflow). Sequential
MNIST: each 28x28 image is a 784-length scalar sequence of intensities in
`[0, 1]` (`--standardize` optional).

Default config (per SPEC): `H = 96, N = 64, L = 3, batch = 64, Adam,
lr = 1e-3 with cosine decay to 0, epochs = 3`, trained on a 20k-sample
subset, evaluated on the full 10k test set. Metrics (per-epoch train loss /
test acc, wall time, steps/s, param count, device) are written to
`results/metrics_<model>.json`.

Useful flags: `--scan {assoc,lax}` (parallel associative scan vs sequential
`lax.scan`; `assoc` on GPU, `lax` is faster on few CPU cores), `--rho-init`
(prospective friction at init — sets the memory horizon ~ `1/rho`),
`--downsample 2` (14x14, T=196), `--tag` (suffix for the metrics filename so
parallel runs coexist).

### Running on GPU

Nothing to change: JAX places computation on the default device. On a machine
with a CUDA GPU, install the CUDA build of JAX
(`pip install "jax[cuda12]"`, or just run `bash setup.sh`) and the same
commands run on GPU (`jax.devices()` is printed at startup). Use
`--scan assoc` (the default) on GPU; `--scan lax` is only faster on few CPU
cores.

### Running on the SLURM cluster

```bash
git clone https://github.com/aledurso22/prospective-s5.git && cd prospective-s5
bash setup.sh                    # builds .venv, auto-detects the GPU
python test_scan.py              # 9/9 must pass before trusting any run

sbatch -p <partition> -A <account> scripts/train.sbatch baseline
sbatch -p <partition> -A <account> scripts/train.sbatch prospective "--rho-init 1e-3 --tag _rho1e-3"
```

`scripts/train.sbatch` runs the full default budget (T=784, 60k train, H=96,
N=64, L=3, 3 epochs) and writes `results/metrics_<model><tag>.json`. MNIST
downloads itself on first run. **`train.py` has no checkpoint/resume** — if a
job hits its `--time` limit the run is lost, so size the limit generously.

---

## 4. Models

Both models share the same budget: `H (d_model) = 96`, `N (state) = 64`,
`L (layers) = 3`, dropout, LR schedule, epochs, batch size.

* `baseline` — `S5SSM`: bilinear-discretized diagonal SSM, trainable
  `log-Delta`, `Lambda` (real/imag, HiPPO init), `B`, `C`, `D`.
* `prospective` — `ProspectiveSSM`: the prospective update above; trainable
  `log-Delta` and `log(Delta t / tau)` (per channel), `Lambda`, `B`, `C`,
  `D`.

Block: `LayerNorm -> SSM -> dropout -> GLU -> dropout -> residual`.
Classifier: linear encoder -> `L` blocks -> mean pool over time -> LayerNorm
-> linear head.

---

## 5. Results

### Final sandbox results (2026-08-04, 2-core CPU sandbox)

Sequential MNIST, images average-pooled 2x2 -> 14x14 (T=196); identical budget
for both models: H=64, N=32, L=2, batch=32, Adam lr=1e-3 cosine, 1 epoch on a
10k train subset (312 steps), eval on 2000 test samples, seed 0.
Training used scan_impl="lax" (XLA-CPU associative_scan overhead); the
associative scans are the default in the layers and are validated against
sequential references in test_scan.py (all tests PASS, rtol < 1e-4).

| model        | params | test acc | test loss | s/step | wall    |
|--------------|--------|----------|-----------|--------|---------|
| baseline S5  | 34,570 | 0.3225   | 1.855     | 0.98   | 340 s   |
| prospective  | 34,698 | 0.2545   | 1.941     | 4.89   | 599 s   |

Source of record: `results/metrics_baseline.json` and
`results/metrics_prospective.json` (copied in from the original run). Params,
accuracy, loss and wall time above match those files exactly. **The `s/step`
column does not**: the JSONs report `steps_per_sec` 0.918 and 0.521, i.e.
1.09 and 1.92 s/step, so the recorded 4.9x per-step slowdown is really ~1.8x
by that measure. Left as-is pending a decision on which measurement the
column was meant to report.

Interpretation: both models are severely undertrained at this budget (312
steps; loss ~2.0 vs ln(10)=2.30 at chance), so the accuracy gap is NOT a
meaningful scientific comparison — it is a smoke test showing both variants
train stably end-to-end. The prospective layer is ~5x slower per step on
XLA-CPU (second-order recurrence + doubled state). A real comparison needs
the lab GPU (full 784-length sMNIST, multi-epoch, tuned rho/Delta ranges).

### Key scientific finding

(see "Stability note" above and `ghost_demo.py`) The Euler-discretized
prospective update `s_t = [(1-rho)I + (1+rho)A] s_{t-1} - A s_{t-2} + ...`
has a parasitic mode with `|mu| ~ |A_i|`; with raw HiPPO eigenvalues it is
unconditionally unstable for any `rho = Delta t/tau`. The documented fallback
(`A := Delta * Lambda` with trainable clamped log-Delta, `rho` clamped to
`[1e-3, 0.25]`) keeps `max |eig(M_i)| <= 0.90` at init. In other words: the
prospective arm only trains with its long-memory eigenvalues clamped away —
the clamps are the empirical footprint of the ghost.

---

## Appendix — later stages of the project (code NOT in this folder)

### What "PESM" means, and why it is not "prospective S5"

**PESM = Prospective Equilibrium Sequence Model.** It is the name of the
*successor* model, not another name for the work in this repo — the two are
different objects and the naming marks the pivot:

| | this repo (`prospective/`) | PESM (`../pesm/ssm/pesm.py`) |
|---|---|---|
| state defined by | a **recurrence** (S5-style, second order) | an **energy** `E(s; x)`, state = equilibrium `grad E = 0` |
| prospectiveness lives in | the update rule itself | the **solver metric** (Hessian mass), `gamma=1` = Newton |
| family | S5 / SSM | DEQ (deep equilibrium) |
| how it runs | one associative scan | K damped-Newton steps, 3 scans each |

So "prospective S5" is accurate for *this* folder. PESM got a new name
because Stage 8 concluded that the prospective idea does not belong in the
recurrence at all — the recurrence version is exactly what fails here (the
ghost). At `K=1, beta=0` PESM collapses back to a plain linear SSM step,
which is how the two families are tied together.

The sections below record findings from that wider project. Their code
(`ssm/pesm.py`, `train_lm.py`, `train_copy.py`, `test_pesm.py`,
`results/EXPERIMENTS.md`) lives in `../pesm/`, not in this
minimal repo — kept here so the experimental record stays in one place.

### Stage 8 — Prospective DEQ LM (`ssm/pesm.py` + `train_lm.py`)

The prospective-equilibrium layer as a causal character-level LM.
Theory: `../The_Damped_Prospective_Action.docx` — prospectiveness lives in
the **solver metric** (Hessian mass), not the recurrence.

**Model.** Energy over the state trajectory:
`E = sum_t 1/2||s_t - (lam*s_{t-1} + B x_t)||^2 + beta/2||tanh(s_t) - tanh(W_u x_t)||^2`.
Each token settles to the local equilibrium by K damped-Newton steps
(elementwise per channel; warm start = the linear SSM step itself).
`gamma=1` = prospective (Newton in the Hessian metric); `gamma=0` = control
(Euclidean gradient steps, size `--eta`). At `K=1, beta=0` the layer is
EXACTLY the linear SSM recurrence (consistency test 3).

`test_pesm.py` — 6 tests, ALL PASS: tridiag solve vs dense (2e-6); sequence
K=1 quadratic == rollout (6e-4); causal K=1,beta=0 == SSM step (exact);
stiff-problem ablation gamma=1 vs gamma=0 residual at K=4:
**0.88 vs 20.9 (~24x faster convergence with the mass on)**; LM smoke.

**Sandbox smoke (CPU, toy corpus, 62,880 params, 120 steps)**

| arm | final val loss | bpc |
|---|---|---|
| gamma=1 (prospective), K=2 | 0.149 | 0.215 |
| gamma=0 (control), K=2 | 0.153 | 0.221 |

(toy corpus too easy to discriminate arms — the discriminating experiment is
stiff/long-range data on GPU.)

**Planned GPU experiments (the paper figures)**
1. Loss-vs-K curves, gamma=1 vs gamma=0 (expect: prospective decays faster
   and lower — isochronous correction vs stiffness-limited).
2. Long-range recall probe (needle in long context): mass should help most
   there (stiff directions = memory directions).
3. K-at-inference ("thinking steps"): train K=2, eval K=1..8 per token.
4. beta ablation: beta=0 recovers linear SSM quality; beta>0 adds the
   nonlinear-equilibrium expressivity.

### Stage 9 — first experimental campaign (CPU)

Controlled comparisons, identical init per arm pair (same seed):
1. **Char-LM Tiny Shakespeare** (seq 128 & 512): γ=1 vs γ=0 statistically
   indistinguishable (±0.03 bpc, gap oscillates around 0) — natural text is
   not stiff along task-relevant directions; consistent with theory.
2. **Delayed-copy probe** (`train_copy.py`, controllable lag G, L=8):
   G=16/64: arms tied (0.76/0.79, 0.54/0.56). G=256 (seed 0): γ=0 arm was
   ahead until step 1350 (0.327 vs 0.273), then **collapsed in the last 150
   steps** (acc → 0.152); γ=1 stable to 0.272. Seed replicates in progress.
3. Raw logs/JSONs + figures in `results/`; full write-up in
   `results/EXPERIMENTS.md`.

Interpretation: no accuracy free lunch at small scale; the candidate
prospective benefit is *stability in the stiff/long-memory regime* —
exactly where the CFL law says the Euclidean solver is fragile.

### Stage 10 — Phase 0 of the PTB plan: NULL

char-PTB K-sweep: more settle iterations hurt both arms; no gamma
advantage. Read: per-token causal settle is not the regime where the
solver matters. Phase 1 spec in `EXPERIMENT_SPEC_PTB.md` (true DEQ,
word-level, tight tolerance — the regime where the few-NFE gap is
documented to exist).
