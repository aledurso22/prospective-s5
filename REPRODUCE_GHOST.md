# Reproducing the ghost: explicit prospective SSM vs baseline S5

This package reproduces the first experiment of the project: the professor's
prospective SSM recurrence (notes Eq. (1); Zucchet et al. Eq. (1.10)
Euler-discretised), implemented EXACTLY as derived, with a parallel
associative scan — and its instability ("the ghost").

Paper reference: `Prospective_Equilibrium_Paper.docx`, Secs. 2.1–2.2.

Requires Python 3.10+ (developed on 3.12).

## 0. Install (GPU)

Easiest — one command (auto-detects GPU, creates .venv, runs a sanity check):
```bash
bash setup.sh          # or: bash setup.sh cpu
```

Manual equivalent:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -U "jax[cuda12]==0.11.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.12.8 optax==0.2.8 numpy scipy
# CPU-only fallback: pip install -r requirements.txt
```

Check: `python -c "import jax; print(jax.devices())"` should list a GPU.

## 1. The ghost in 30 seconds (no training, no GPU, numpy only)

```bash
python ghost_demo.py
```

Expected output:
- **A.** roots table: mu_phys*mu_ghost = -h; ghost |mu| > 1 for h >= 1,
  |mu_ghost| ~ (1+rho) h. At h=1 the roots are 0.618 and -1.618.
- **B.** the recurrence simulated: converges at h=0.3, diverges at h>=0.9.
  Survival requires clamping stiffness — i.e. killing the |lambda| -> 1
  modes, which ARE the long memory.
- **C.** RK4 tracks the ghost with rel. error 2.3e-6 -> 2.5e-10 as dt
  shrinks: higher order integrates the ghost MORE faithfully.
  Order is not stability.
- **D.** the fix: first-order prospective-metric (IMEX) scheme; measured
  stability boundary dt_max/tau = 2*alpha exactly (the CFL law).

## 2. Scan correctness (the implementation is exact)

```bash
python test_scan.py
```

9/9 tests: the prospective 2x2 affine associative scan agrees with the
sequential reference to float32 tolerance (~1e-6), including the HiPPO-form
transition and the full layer-vs-sequential check. The scan is NOT the
problem — the recurrence is.

## 3. The training comparison (sequential MNIST, the failing experiment)

```bash
# baseline S5 (control arm)
python train.py --model baseline    --epochs 3 --subset 0      # full 60k

# prospective SSM (the notes' recurrence)
python train.py --model prospective --epochs 3 --subset 0
```

On a single GPU (e.g. RTX 3090) each epoch at full 60k / T=784 takes minutes.
Data (MNIST IDX files) downloads automatically on first run.

**What you will see:** the prospective arm only trains because of the
built-in clamps (`ssm/prospective/layer.py`: Delta in [1e-5, 5e-4], rho in
[1e-3, 0.25], giving max |eig(M_i)| <= 0.90 at init — measured in test [3] of
test_scan.py). The clamps are the empirical footprint of the ghost: the
unclamped recurrence has a parasitic root whose magnitude grows with
stiffness, so the model may not use stiff (long-memory) eigenvalues.

**Sandbox reference numbers** (2-core CPU, T=196 downsampled, 10k subset):
baseline test acc ~0.32, prospective ~0.25. WARNING: the baseline is far
below real S5 (~0.98) — that comparison is NOT evidence of anything; only
the instability is. On GPU with full data, the baseline should reach S5-level
accuracy; the prospective arm is expected to underperform and to require the
clamps regardless.

## 4. What to tell the professor (the three-sentence version)

1. Your Eq. (1) discretised explicitly is a two-step recurrence; two-step
   recurrences carry a parasitic root per mode (Dahlquist), and here
   |mu_ghost| ~ (1+rho)h grows with stiffness — the memory directions.
2. So the scheme can only train with the eigenvalues clamped away from 1,
   i.e. with its long memory removed; no step size or integrator order
   fixes this (RK4 demo C).
3. The fix is not in the recurrence but in the solver metric — the
   prospective-metric (damped-Newton) scheme is first-order per mode, has no
   ghost, and its exact stability boundary is dt <= 2*alpha*tau (demo D).

## Files

- `ghost_demo.py` — the 4-part ghost demo (numpy only)
- `ssm/baseline_s5/` — the S5 control arm: `scan.py` (first-order elementwise
  associative scan), `layer.py` (`S5SSM`, bilinear discretization)
- `ssm/prospective/` — **the new work**: `scan.py` (2x2 companion affine
  associative scan), `layer.py` (`ProspectiveSSM`, clamps documented)
- `ssm/shared/` — identical in both arms: `hippo.py` (HiPPO init),
  `params.py` (complex params + S5 initializers), `block.py` (`S5Block`)
- `ssm/model.py` — the classifier and the single arm switch
- `test_scan.py` — 9/9 correctness suite (both layer modes)
- `exact_failure.py` — the derivation verbatim: companion spectrum,
  overflow step, and the fallback's cost, at the real HiPPO init
- `train.py` — sequential-MNIST training, both arms
- `results/` — `metrics_{baseline,prospective}.json` from the original
  sandbox run; new runs are written here too

See [README.md](README.md) §1 for the full code map and a suggested reading
order.
